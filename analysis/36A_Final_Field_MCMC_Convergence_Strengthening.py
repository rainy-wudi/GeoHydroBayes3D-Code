# -*- coding: utf-8 -*-
"""
STEP 12A: Final convergence-strengthened FIELD hierarchical Bayesian inversion.

Likelihood:
y_obs | theta,sigma_rep ~ N(
    f_DNN(theta) + b_surr,
    Sigma_RF + Sigma_surr_LW + 0.5^2 I + sigma_rep^2 I
)

sigma_rep ~ HalfNormal(scale=10 m)

Corrected priors:
Lower log10K  [-9,-7]
Middle log10K [-8,-6]
Upper log10K  [-7,-5]
COV           [0.4,1.0]
dh            [50,300] m
dv            [1,5] m
"""

import os, json, random, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error

try:
    import emcee
except ImportError as exc:
    raise ImportError(
        "Install emcee with:\n"
        "python -m pip install emcee"
    ) from exc


try:
    from scipy.stats import rankdata, norm
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


# ---------------------------------------------------------------------------
# 0. Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------------------------
WORK_DIR = os.environ.get("GHB_WORK_DIR", os.path.dirname(os.path.abspath(__file__)))

SIGMA_RF_PATH = os.path.join(
    WORK_DIR, "Sigma_RF_Pooled_23x23_CorrectedRF_FinalGW.csv"
)

STEP8_DIR = os.path.join(
    WORK_DIR, "Step8_5Fold_OOF_SurrogateError_CorrectedRF_FinalGW"
)

SIGMA_SURR_PATH = os.path.join(
    STEP8_DIR, "Sigma_surr_LedoitWolf_23x23_CorrectedRF.csv"
)

BIAS_PATH = os.path.join(
    STEP8_DIR, "Surrogate_Bias_Vector_CorrectedRF.csv"
)

PRODUCTION_DIR = os.path.join(STEP8_DIR, "Production_DNN_All50")

MODEL_PATH = os.path.join(
    PRODUCTION_DIR, "GroupMean_DNN_All50_CorrectedRF_253epochs.pth"
)

SCALER_X_PATH = os.path.join(
    PRODUCTION_DIR, "scaler_X_All50_CorrectedRF.pkl"
)

SCALER_Y_PATH = os.path.join(
    PRODUCTION_DIR, "scaler_Y_All50_CorrectedRF.pkl"
)

OUTPUT_DIR = os.path.join(
    WORK_DIR, "Step12A_Final_Field_MCMC_Strengthened"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 2. Parameter / borehole order
# ---------------------------------------------------------------------------
X_COLS = [
    "log10_K_Lower", "log10_K_Middle", "log10_K_Upper",
    "COV", "dh", "dv"
]
PARAMS = X_COLS + ["sigma_rep"]

Y_COLS = [
    "zk101","zk102","zk103","zk104","zk105","zk106",
    "zk107","zk108","zk109","zk110","zk120","zk131",
    "zk132","zk133","zk134","zk135","zk136","zk137",
    "zk138","zk139","zk140","zk141","zk410"
]
NOBS = len(Y_COLS)
NDIM = 7


# ---------------------------------------------------------------------------
# 3. Historical-minimum field heads (non-synchronous)
# ---------------------------------------------------------------------------
OBSERVED_HEADS = {
    "zk101":496.20, "zk102":489.70, "zk103":492.31,
    "zk104":495.09, "zk105":554.80, "zk106":543.74,
    "zk107":514.78, "zk108":502.34, "zk109":491.18,
    "zk110":476.07, "zk120":479.12, "zk131":490.96,
    "zk132":489.47, "zk133":484.04, "zk134":491.21,
    "zk135":482.89, "zk136":508.40, "zk137":521.43,
    "zk138":494.36, "zk139":509.17, "zk140":496.48,
    "zk141":493.23, "zk410":550.28
}
Y_OBS = np.array([OBSERVED_HEADS[k] for k in Y_COLS], dtype=float)


# ---------------------------------------------------------------------------
# 4. Priors
# ---------------------------------------------------------------------------
PRIOR_BOUNDS = {
    "log10_K_Lower":(-9.0,-7.0),
    "log10_K_Middle":(-8.0,-6.0),
    "log10_K_Upper":(-7.0,-5.0),
    "COV":(0.4,1.0),
    "dh":(50.0,300.0),
    "dv":(1.0,5.0)
}
LOWER = np.array([PRIOR_BOUNDS[k][0] for k in X_COLS], float)
UPPER = np.array([PRIOR_BOUNDS[k][1] for k in X_COLS], float)
WIDTH = UPPER - LOWER
LOG_PRIOR_PHYS = -np.sum(np.log(WIDTH))

SIGMA_REP_SCALE = 10.0
SIGMA_OBS = 0.5
LOG_HN_CONST = 0.5*np.log(2.0/np.pi) - np.log(SIGMA_REP_SCALE)


# ---------------------------------------------------------------------------
# 5. MCMC settings
# ---------------------------------------------------------------------------
N_WALKERS = 64
N_STEPS = int(os.environ.get("GHB_N_STEPS", "50000"))
BURN_IN = int(os.environ.get("GHB_BURN_IN", "10000"))
N_PRESCREEN = int(os.environ.get("GHB_N_PRESCREEN", "30000"))
PRESCREEN_BATCH = 2048
TOP_POOL = int(os.environ.get("GHB_TOP_POOL", "640"))
N_PP = int(os.environ.get("GHB_N_PP", "10000"))


# ---------------------------------------------------------------------------
# 6. Network
# ---------------------------------------------------------------------------
class GroupMeanDNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6,32), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(32,64), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(64,32), nn.ReLU(),
            nn.Linear(32,23)
        )
    def forward(self,x):
        return self.net(x)


def torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def load_cov(path):
    d = pd.read_csv(path, index_col=0)
    missing = [x for x in Y_COLS if x not in d.index or x not in d.columns]
    if missing:
        raise RuntimeError("Covariance labels missing: %s" % missing)
    m = d.loc[Y_COLS,Y_COLS].to_numpy(float)
    return 0.5*(m+m.T)


def load_bias(path):
    d = pd.read_csv(path)
    if not {"Borehole","Residual_Mean_Bias_m"}.issubset(d.columns):
        raise RuntimeError("Bias file columns are wrong.")
    s = d.set_index("Borehole")["Residual_Mean_Bias_m"]
    return s.loc[Y_COLS].to_numpy(float)


def hdi(x, mass=0.95):
    x = np.sort(np.asarray(x,float))
    n = len(x)
    m = int(np.floor(mass*n))
    widths = x[m:] - x[:n-m]
    i = int(np.argmin(widths))
    return float(x[i]), float(x[i+m])


def split_rhat(chain):
    # chain shape = (step, walker, parameter)
    ns,nw,nd = chain.shape
    n = ns//2
    a = np.transpose(chain[:n], (1,0,2))
    b = np.transpose(chain[-n:], (1,0,2))
    c = np.concatenate([a,b],axis=0)
    means = c.mean(axis=1)
    vars_ = c.var(axis=1,ddof=1)
    W = vars_.mean(axis=0)
    B = n*means.var(axis=0,ddof=1)
    V = ((n-1.0)/n)*W + B/n
    return np.sqrt(V/W)


def rank_normalized_split_rhat(chain):
    """
    Modern robustness check: maximum of rank-normalized and folded
    rank-normalized split-Rhat for each parameter.
    """
    if not HAS_SCIPY:
        return np.full(chain.shape[2], np.nan)

    def rank_normalize(x):
        flat = np.asarray(x, float).reshape(-1)
        ranks = rankdata(flat, method="average")
        n = len(flat)
        p = (ranks - 3.0/8.0) / (n + 1.0/4.0)
        return norm.ppf(p).reshape(np.asarray(x).shape)

    out = np.full(chain.shape[2], np.nan)
    for j in range(chain.shape[2]):
        x = chain[:,:,j]
        z = rank_normalize(x)
        r_rank = split_rhat(z[:,:,None])[0]

        folded = np.abs(x - np.median(x))
        z_fold = rank_normalize(folded)
        r_fold = split_rhat(z_fold[:,:,None])[0]

        out[j] = max(r_rank, r_fold)
    return out


# ---------------------------------------------------------------------------
# 7. Load corrected products
# ---------------------------------------------------------------------------
for path in [
    SIGMA_RF_PATH, SIGMA_SURR_PATH, BIAS_PATH,
    MODEL_PATH, SCALER_X_PATH, SCALER_Y_PATH
]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

print("="*96)
print("STEP 12A: FINAL FIELD MCMC - CONVERGENCE STRENGTHENING")
print("="*96)
print("Production model :", MODEL_PATH)
print("Sigma_RF         :", SIGMA_RF_PATH)
print("Sigma_surr_LW    :", SIGMA_SURR_PATH)
print("Bias             :", BIAS_PATH)
print("Output           :", OUTPUT_DIR)

Sigma_RF = load_cov(SIGMA_RF_PATH)
Sigma_surr = load_cov(SIGMA_SURR_PATH)
bias = load_bias(BIAS_PATH)

Sigma_base = (
    Sigma_RF + Sigma_surr + SIGMA_OBS**2*np.eye(NOBS)
)
Sigma_base = 0.5*(Sigma_base+Sigma_base.T)

lam0,Q = np.linalg.eigh(Sigma_base)
if lam0.min() <= 0:
    raise RuntimeError("Sigma_base not positive definite.")

print("\nUncertainty:")
print("Mean RF STD      = %.4f m" % np.sqrt(np.diag(Sigma_RF)).mean())
print("Mean surr STD LW = %.4f m" % np.sqrt(np.diag(Sigma_surr)).mean())
print("sigma_obs        = %.4f m" % SIGMA_OBS)
print("Mean bias        = %+.4f m" % bias.mean())
print("Sigma_base min eig = %.6e" % lam0.min())


# ---------------------------------------------------------------------------
# 8. Load production surrogate
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device           :", device)

sx = joblib.load(SCALER_X_PATH)
sy = joblib.load(SCALER_Y_PATH)

model = GroupMeanDNN().to(device)
model.load_state_dict(torch_load(MODEL_PATH,device))
model.eval()

sx_mean = np.asarray(sx.mean_,float)
sx_scale = np.asarray(sx.scale_,float)
sy_mean = np.asarray(sy.mean_,float)
sy_scale = np.asarray(sy.scale_,float)


def surrogate(theta):
    theta = np.asarray(theta,float)
    if theta.ndim == 1:
        theta = theta[None,:]
    xs = (theta-sx_mean)/sx_scale
    xt = torch.tensor(xs,dtype=torch.float32,device=device)
    with torch.no_grad():
        ys = model(xt).cpu().numpy()
    y = ys*sy_scale + sy_mean
    # OOF residual definition was Abaqus - DNN
    return y + bias[None,:]


# ---------------------------------------------------------------------------
# 9. Fast vectorized log posterior
# ---------------------------------------------------------------------------
LOG2PI = np.log(2.0*np.pi)

def log_prob_batch(par):
    par = np.asarray(par,float)
    if par.ndim == 1:
        par = par[None,:]

    out = np.full(par.shape[0], -np.inf, float)
    theta = par[:,:6]
    sig = par[:,6]

    ok = (
        np.all((theta>=LOWER[None,:]) & (theta<=UPPER[None,:]),axis=1)
        & np.isfinite(sig) & (sig>0.0)
    )
    if not np.any(ok):
        return out

    ii = np.where(ok)[0]
    mu = surrogate(theta[ii])
    r = Y_OBS[None,:] - mu

    # Q columns are eigenvectors; row-vector coordinates are r @ Q
    z = r @ Q
    den = lam0[None,:] + sig[ii,None]**2

    quad = np.sum(z*z/den,axis=1)
    logdet = np.sum(np.log(den),axis=1)
    ll = -0.5*(NOBS*LOG2PI + logdet + quad)

    lp_rep = LOG_HN_CONST - 0.5*(sig[ii]/SIGMA_REP_SCALE)**2
    out[ii] = LOG_PRIOR_PHYS + lp_rep + ll
    return out


# ---------------------------------------------------------------------------
# 10. Prescreen and walker initialization
# ---------------------------------------------------------------------------
print("\nPRESCREEN...")
rng = np.random.RandomState(SEED+100)
phys = LOWER[None,:] + rng.rand(N_PRESCREEN,6)*WIDTH[None,:]
sig = np.maximum(
    np.abs(rng.normal(0.0,SIGMA_REP_SCALE,N_PRESCREEN)),
    1e-6
)
cand = np.column_stack([phys,sig])
lp = np.empty(N_PRESCREEN,float)

for i in range(0,N_PRESCREEN,PRESCREEN_BATCH):
    j = min(i+PRESCREEN_BATCH,N_PRESCREEN)
    lp[i:j] = log_prob_batch(cand[i:j])

order = np.argsort(lp)[::-1]
pool = order[:TOP_POOL]
chosen = rng.choice(pool,size=N_WALKERS,replace=False)
p0 = cand[chosen].copy()

print("Best prescreen logP = %.6f" % lp[order[0]])
print("Initial sigma_rep   = [%.3f, %.3f] m" % (p0[:,6].min(),p0[:,6].max()))


# ---------------------------------------------------------------------------
# 11. MCMC
# ---------------------------------------------------------------------------
print("\nRUNNING MCMC: %d walkers x %d steps ..." % (N_WALKERS,N_STEPS))

sampler = emcee.EnsembleSampler(
    N_WALKERS, NDIM, log_prob_batch, vectorize=True
)
sampler.run_mcmc(p0,N_STEPS,progress=True)

chain = sampler.get_chain()
logp = sampler.get_log_prob()
acceptance = float(np.mean(sampler.acceptance_fraction))

post_chain = chain[BURN_IN:]
post_logp = logp[BURN_IN:]

flat = post_chain.reshape(-1,NDIM)
flat_logp = post_logp.reshape(-1)

rhat = split_rhat(post_chain)
rank_rhat = rank_normalized_split_rhat(post_chain)

try:
    tau = np.asarray(
        emcee.autocorr.integrated_time(post_chain,quiet=True),
        float
    )
except Exception as exc:
    warnings.warn("Tau unavailable: %s" % exc)
    tau = np.full(NDIM,np.nan)

total_post = post_chain.shape[0]*post_chain.shape[1]
ess = total_post/tau


# ---------------------------------------------------------------------------
# 12. Summaries
# ---------------------------------------------------------------------------
map_i = int(np.nanargmax(flat_logp))
map_par = flat[map_i]
med_par = np.median(flat,axis=0)

rows = []
for j,name in enumerate(PARAMS):
    lo,hi = hdi(flat[:,j],0.95)
    rows.append({
        "Parameter":name,
        "MAP":float(map_par[j]),
        "Mean":float(np.mean(flat[:,j])),
        "Median":float(np.median(flat[:,j])),
        "SD":float(np.std(flat[:,j],ddof=1)),
        "HDI95_Low":lo,
        "HDI95_High":hi,
        "Split_Rhat":float(rhat[j]),
        "RankNormalized_Split_Rhat":float(rank_rhat[j]) if np.isfinite(rank_rhat[j]) else np.nan,
        "Tau":float(tau[j]) if np.isfinite(tau[j]) else np.nan,
        "ESS":float(ess[j]) if np.isfinite(ess[j]) else np.nan
    })
summary = pd.DataFrame(rows)

SUMMARY_PATH = os.path.join(
    OUTPUT_DIR,"Final_Field_Posterior_Parameter_Summary_Strengthened.csv"
)
summary.to_csv(SUMMARY_PATH,index=False,encoding="utf-8-sig")

samples = pd.DataFrame(flat,columns=PARAMS)
samples["log_posterior"] = flat_logp
SAMPLES_PATH = os.path.join(
    OUTPUT_DIR,"Final_Field_Posterior_Samples_Strengthened.csv"
)
samples.to_csv(SAMPLES_PATH,index=False,encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# 13. MAP / median-parameter mean predictions
# ---------------------------------------------------------------------------
map_mu = surrogate(map_par[:6])[0]
med_mu = surrogate(med_par[:6])[0]

map_rmse = float(np.sqrt(mean_squared_error(Y_OBS,map_mu)))
map_mae = float(mean_absolute_error(Y_OBS,map_mu))
med_rmse = float(np.sqrt(mean_squared_error(Y_OBS,med_mu)))
med_mae = float(mean_absolute_error(Y_OBS,med_mu))


# ---------------------------------------------------------------------------
# 14. Posterior predictive
# ---------------------------------------------------------------------------
print("POSTERIOR PREDICTIVE...")

rngpp = np.random.RandomState(SEED+900)
npp = min(N_PP,len(flat))
ii = rngpp.choice(len(flat),size=npp,replace=False)
pp_par = flat[ii]
pp_mu = surrogate(pp_par[:,:6])
pp = np.empty_like(pp_mu)

for i in range(npp):
    lam = lam0 + pp_par[i,6]**2
    eps = rngpp.normal(size=NOBS)
    pp[i] = pp_mu[i] + Q @ (np.sqrt(lam)*eps)

pred_lo = np.quantile(pp,0.025,axis=0)
pred_md = np.quantile(pp,0.50,axis=0)
pred_hi = np.quantile(pp,0.975,axis=0)
covered = (Y_OBS>=pred_lo) & (Y_OBS<=pred_hi)
coverage = float(covered.mean())

pp_df = pd.DataFrame({
    "Borehole":Y_COLS,
    "Observed_Head_m":Y_OBS,
    "MAP_Mean_Head_m":map_mu,
    "MedianParam_Mean_Head_m":med_mu,
    "PP_P2.5_m":pred_lo,
    "PP_Median_m":pred_md,
    "PP_P97.5_m":pred_hi,
    "Observed_Inside_PP95":covered,
    "MAP_Residual_ObsMinusPred_m":Y_OBS-map_mu
})
PP_PATH = os.path.join(
    OUTPUT_DIR,"Final_Field_Posterior_Predictive_Strengthened.csv"
)
pp_df.to_csv(PP_PATH,index=False,encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# 15. Diagnostics
# ---------------------------------------------------------------------------
diag = {
    "n_walkers":N_WALKERS,
    "n_steps":N_STEPS,
    "burn_in":BURN_IN,
    "postburn_samples":int(len(flat)),
    "acceptance_fraction":acceptance,
    "max_split_rhat":float(np.nanmax(rhat)),
    "max_rank_normalized_split_rhat":float(np.nanmax(rank_rhat)) if np.isfinite(rank_rhat).any() else None,
    "min_ess":float(np.nanmin(ess)) if np.isfinite(ess).any() else None,
    "convergence_target":"max split-Rhat < 1.01 and min ESS > 3000",
    "convergence_pass":bool((np.nanmax(rhat) < 1.01) and ((np.nanmax(rank_rhat) < 1.01) if np.isfinite(rank_rhat).any() else True) and (np.nanmin(ess) > 3000)) if np.isfinite(ess).any() else False,
    "sigma_obs_m":SIGMA_OBS,
    "sigma_rep_prior":"HalfNormal(scale=10 m)",
    "map_head_rmse_m":map_rmse,
    "map_head_mae_m":map_mae,
    "median_parameter_head_rmse_m":med_rmse,
    "median_parameter_head_mae_m":med_mae,
    "posterior_predictive_95_coverage_percent":100.0*coverage,
    "mean_rf_std_m":float(np.sqrt(np.diag(Sigma_RF)).mean()),
    "mean_surrogate_std_m":float(np.sqrt(np.diag(Sigma_surr)).mean()),
    "mean_surrogate_bias_m":float(bias.mean()),
    "production_model":MODEL_PATH,
    "prior_bounds":{k:list(v) for k,v in PRIOR_BOUNDS.items()}
}
DIAG_PATH = os.path.join(
    OUTPUT_DIR,"Step12A_Final_Field_MCMC_Diagnostics.json"
)
with open(DIAG_PATH,"w",encoding="utf-8") as f:
    json.dump(diag,f,indent=4,ensure_ascii=False)


# ---------------------------------------------------------------------------
# 19. Console summary
# ---------------------------------------------------------------------------
sr = summary.loc[summary.Parameter=="sigma_rep"].iloc[0]

print("\n"+"="*96)
print("STEP 12A COMPLETE - FINAL FIELD MCMC CONVERGENCE STRENGTHENING")
print("="*96)
print("Acceptance fraction             : %.4f" % acceptance)
print("Max classic split-Rhat          : %.5f" % np.nanmax(rhat))
if np.isfinite(rank_rhat).any():
    print("Max rank-normalized split-Rhat  : %.5f" % np.nanmax(rank_rhat))
if np.isfinite(ess).any():
    print("Min approximate ESS             : %.1f" % np.nanmin(ess))
else:
    print("Min approximate ESS             : unavailable")
print("sigma_rep MAP                   : %.4f m" % sr["MAP"])
print("sigma_rep median                : %.4f m" % sr["Median"])
print("sigma_rep 95%% HDI              : [%.4f, %.4f] m" %
      (sr["HDI95_Low"],sr["HDI95_High"]))
print("MAP mean-head RMSE              : %.4f m" % map_rmse)
print("MAP mean-head MAE               : %.4f m" % map_mae)
print("Posterior predictive 95%% cover : %.2f%%" % (100.0*coverage))
print("\nPosterior parameter summary:")
print(summary.to_string(index=False))
print("\nSaved:")
print(" ",SUMMARY_PATH)
print(" ",SAMPLES_PATH)
print(" ",PP_PATH)
print(" ",DIAG_PATH)
conv_pass = ((np.nanmax(rhat) < 1.01) and ((np.nanmax(rank_rhat) < 1.01) if np.isfinite(rank_rhat).any() else True) and (np.nanmin(ess) > 3000)) if np.isfinite(ess).any() else False
print("Convergence target              : max Rhat < 1.01, min ESS > 3000")
print("FINAL CONVERGENCE PASS          : %s" % str(bool(conv_pass)))
print("\nNEXT: if PASS, strengthen the 3 independent synthetic MCMC chains without changing their likelihood.")
