# -*- coding: utf-8 -*-
"""
36B_Final_Independent_Synthetic_MCMC_Strengthening.py

RUN WITH ORDINARY PCBNN PYTHON (NOT Abaqus Python).

STEP 12B
--------
Final convergence strengthening for the 3 independent corrected-RF V2 truth models.

Only sampling length and convergence diagnostics are strengthened; the likelihood, priors, truths, DNN, covariance terms, and synthetic observations are unchanged.

Synthetic likelihood:
    y_syn | theta ~ N(
        f_DNN(theta) + b_surr,
        Sigma_RF + Sigma_surr_LW + 0.5^2 I
    )

NO sigma_rep is used.

Validation outputs:
1) Forward consistency at the exact true theta:
   - DNN-vs-Abaqus truth RMSE
   - Mahalanobis^2 under model covariance
2) 6D posterior for each truth
3) Truth-in-95%-HDI for all 18 parameter cases
4) Posterior contraction relative to prior width
5) Posterior predictive coverage for all 23 boreholes
6) Convergence diagnostics: acceptance, classic split-Rhat, rank-normalized split-Rhat, ESS
"""

import os
import json
import random
import warnings

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
    HAS_SCIPY_RHAT = True
except Exception:
    HAS_SCIPY_RHAT = False


# ============================================================================
# 0. Reproducibility
# ============================================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ============================================================================
# 1. Paths
# ============================================================================

WORK_DIR = os.environ.get("GHB_WORK_DIR", os.path.dirname(os.path.abspath(__file__)))

TRUTH_DIR = os.path.join(WORK_DIR, "synthetic")

STEP11_DIR = os.path.join(
    TRUTH_DIR,
    "Step11_Independent_Validation"
)

TRUTH_PARAM_PATH = os.path.join(
    TRUTH_DIR,
    "Truth_Parameters_3_CorrectedRF_V2.csv"
)

RAW_HEADS_PATH = os.path.join(
    STEP11_DIR,
    "Truth_Heads_Raw_Abaqus_3x23.csv"
)

SYN_OBS_PATH = os.path.join(
    STEP11_DIR,
    "Truth_Synthetic_Observations_3x23.csv"
)

SIGMA_RF_PATH = os.path.join(
    WORK_DIR,
    "Sigma_RF_Pooled_23x23_CorrectedRF_FinalGW.csv"
)

STEP8_DIR = os.path.join(
    WORK_DIR,
    "Step8_5Fold_OOF_SurrogateError_CorrectedRF_FinalGW"
)

SIGMA_SURR_PATH = os.path.join(
    STEP8_DIR,
    "Sigma_surr_LedoitWolf_23x23_CorrectedRF.csv"
)

BIAS_PATH = os.path.join(
    STEP8_DIR,
    "Surrogate_Bias_Vector_CorrectedRF.csv"
)

PRODUCTION_DIR = os.path.join(
    STEP8_DIR,
    "Production_DNN_All50"
)

MODEL_PATH = os.path.join(
    PRODUCTION_DIR,
    "GroupMean_DNN_All50_CorrectedRF_253epochs.pth"
)

SCALER_X_PATH = os.path.join(
    PRODUCTION_DIR,
    "scaler_X_All50_CorrectedRF.pkl"
)

SCALER_Y_PATH = os.path.join(
    PRODUCTION_DIR,
    "scaler_Y_All50_CorrectedRF.pkl"
)

OUTPUT_DIR = os.path.join(
    STEP11_DIR,
    "Step12B_Final_Independent_MCMC_Strengthened"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================================
# 2. Parameter / borehole order
# ============================================================================

X_COLS = [
    "log10_K_Lower",
    "log10_K_Middle",
    "log10_K_Upper",
    "COV",
    "dh",
    "dv"
]

Y_COLS = [
    "zk101", "zk102", "zk103", "zk104", "zk105", "zk106",
    "zk107", "zk108", "zk109", "zk110", "zk120", "zk131",
    "zk132", "zk133", "zk134", "zk135", "zk136", "zk137",
    "zk138", "zk139", "zk140", "zk141", "zk410"
]

NDIM = 6
NOBS = 23


# ============================================================================
# 3. Priors
# ============================================================================

PRIOR_BOUNDS = {
    "log10_K_Lower":  (-9.0, -7.0),
    "log10_K_Middle": (-8.0, -6.0),
    "log10_K_Upper":  (-7.0, -5.0),
    "COV":             (0.4, 1.0),
    "dh":              (50.0, 300.0),
    "dv":              (1.0, 5.0)
}

LOWER = np.array(
    [
        PRIOR_BOUNDS[k][0]
        for k in X_COLS
    ],
    dtype=float
)

UPPER = np.array(
    [
        PRIOR_BOUNDS[k][1]
        for k in X_COLS
    ],
    dtype=float
)

WIDTH = UPPER - LOWER

LOG_PRIOR_CONST = -np.sum(
    np.log(WIDTH)
)

PRIOR_SD = WIDTH / np.sqrt(12.0)


# ============================================================================
# 4. Synthetic likelihood
# ============================================================================

SIGMA_OBS = 0.5

# Chi-square 95% threshold for 23 dimensions.
CHI2_23_95 = 35.17246162690806


# ============================================================================
# 5. MCMC settings
# ============================================================================

N_WALKERS = 64
N_STEPS = int(os.environ.get("GHB_N_STEPS", "50000"))
BURN_IN = int(os.environ.get("GHB_BURN_IN", "10000"))

N_PRESCREEN = int(os.environ.get("GHB_N_PRESCREEN", "30000"))
PRESCREEN_BATCH = 2048
TOP_POOL = int(os.environ.get("GHB_TOP_POOL", "640"))

N_PP = int(os.environ.get("GHB_N_PP", "8000"))

CSV_THIN = 20


# ============================================================================
# 6. DNN
# ============================================================================

class GroupMeanDNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Dropout(0.10),

            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.10),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 23)
        )

    def forward(self, x):
        return self.net(x)


# ============================================================================
# 7. Helpers
# ============================================================================

def torch_load(path, device):

    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=True
        )

    except TypeError:
        return torch.load(
            path,
            map_location=device
        )


def load_cov(path):

    d = pd.read_csv(
        path,
        index_col=0
    )

    missing = [
        x for x in Y_COLS
        if x not in d.index
        or x not in d.columns
    ]

    if missing:
        raise RuntimeError(
            "Covariance labels missing: %s"
            % missing
        )

    M = d.loc[
        Y_COLS,
        Y_COLS
    ].to_numpy(float)

    return 0.5 * (
        M + M.T
    )


def load_bias(path):

    d = pd.read_csv(path)

    need = {
        "Borehole",
        "Residual_Mean_Bias_m"
    }

    if not need.issubset(
        d.columns
    ):
        raise RuntimeError(
            "Bias CSV columns are wrong."
        )

    s = d.set_index(
        "Borehole"
    )[
        "Residual_Mean_Bias_m"
    ]

    return s.loc[
        Y_COLS
    ].to_numpy(float)


def hdi_1d(samples, mass=0.95):

    x = np.sort(
        np.asarray(
            samples,
            dtype=float
        )
    )

    n = len(x)

    m = int(
        np.floor(
            mass * n
        )
    )

    if m <= 0 or m >= n:
        return (
            float(
                x[0]
            ),
            float(
                x[-1]
            )
        )

    widths = (
        x[m:]
        - x[:n-m]
    )

    i = int(
        np.argmin(
            widths
        )
    )

    return (
        float(
            x[i]
        ),
        float(
            x[i+m]
        )
    )


def split_rhat(chain):

    # chain = (steps, walkers, params)
    ns, nw, nd = chain.shape

    n = ns // 2

    a = np.transpose(
        chain[:n],
        (1, 0, 2)
    )

    b = np.transpose(
        chain[-n:],
        (1, 0, 2)
    )

    c = np.concatenate(
        [
            a,
            b
        ],
        axis=0
    )

    means = c.mean(
        axis=1
    )

    vars_ = c.var(
        axis=1,
        ddof=1
    )

    W = vars_.mean(
        axis=0
    )

    B = n * means.var(
        axis=0,
        ddof=1
    )

    V = (
        ((n - 1.0) / n) * W
        + B / n
    )

    with np.errstate(
        invalid="ignore",
        divide="ignore"
    ):

        return np.sqrt(
            V / W
        )


def _rank_normalize(values):

    if not HAS_SCIPY_RHAT:
        return None

    a = np.asarray(values, dtype=float)
    flat = a.reshape(-1)
    ranks = rankdata(flat, method="average")
    n = len(flat)
    p = (ranks - 3.0 / 8.0) / (n + 1.0 / 4.0)
    z = norm.ppf(p)
    return z.reshape(a.shape)


def rank_normalized_split_rhat(chain):

    if not HAS_SCIPY_RHAT:
        return np.full(chain.shape[2], np.nan)

    nd = chain.shape[2]
    out = np.full(nd, np.nan)

    for j in range(nd):
        x = chain[:, :, j]

        z = _rank_normalize(x)
        r_rank = split_rhat(z[:, :, None])[0]

        folded = np.abs(x - np.median(x))
        z_fold = _rank_normalize(folded)
        r_fold = split_rhat(z_fold[:, :, None])[0]

        out[j] = max(r_rank, r_fold)

    return out


def mahalanobis2(
    residual,
    covariance
):

    residual = np.asarray(
        residual,
        dtype=float
    )

    L = np.linalg.cholesky(
        covariance
    )

    z = np.linalg.solve(
        L,
        residual
    )

    return float(
        z @ z
    )


# ============================================================================
# 8. Required files
# ============================================================================

required_paths = [
    TRUTH_PARAM_PATH,
    RAW_HEADS_PATH,
    SYN_OBS_PATH,
    SIGMA_RF_PATH,
    SIGMA_SURR_PATH,
    BIAS_PATH,
    MODEL_PATH,
    SCALER_X_PATH,
    SCALER_Y_PATH
]

for path in required_paths:

    if not os.path.exists(path):
        raise FileNotFoundError(path)


# ============================================================================
# 9. Load all inputs
# ============================================================================

print("=" * 100)
print("STEP 11B: INDEPENDENT SYNTHETIC VALIDATION - 3 CORRECTED-RF V2 TRUTHS")
print("=" * 100)

truth_df = pd.read_csv(
    TRUTH_PARAM_PATH
)

raw_df = pd.read_csv(
    RAW_HEADS_PATH
)

obs_df = pd.read_csv(
    SYN_OBS_PATH
)

for d, name in [
    (truth_df, "truth parameters"),
    (raw_df, "raw truth heads"),
    (obs_df, "synthetic observations")
]:

    if len(d) != 3:
        raise RuntimeError(
            "%s must contain exactly 3 rows."
            % name
        )


truth_df[
    "Truth_ID"
] = truth_df[
    "Truth_ID"
].astype(int)

raw_df[
    "Truth_ID"
] = raw_df[
    "Truth_ID"
].astype(int)

obs_df[
    "Truth_ID"
] = obs_df[
    "Truth_ID"
].astype(int)

truth_df = truth_df.sort_values(
    "Truth_ID"
).reset_index(drop=True)

raw_df = raw_df.sort_values(
    "Truth_ID"
).reset_index(drop=True)

obs_df = obs_df.sort_values(
    "Truth_ID"
).reset_index(drop=True)

if list(
    truth_df[
        "Truth_ID"
    ]
) != [1, 2, 3]:
    raise RuntimeError(
        "Truth parameter IDs must be 1,2,3."
    )


Sigma_RF = load_cov(
    SIGMA_RF_PATH
)

Sigma_surr = load_cov(
    SIGMA_SURR_PATH
)

bias = load_bias(
    BIAS_PATH
)

Sigma_model = (
    Sigma_RF
    + Sigma_surr
)

Sigma_model = 0.5 * (
    Sigma_model
    + Sigma_model.T
)

Sigma_syn = (
    Sigma_model
    + SIGMA_OBS ** 2
    * np.eye(NOBS)
)

Sigma_syn = 0.5 * (
    Sigma_syn
    + Sigma_syn.T
)

eig_syn, Q_syn = np.linalg.eigh(
    Sigma_syn
)

if eig_syn.min() <= 0:
    raise RuntimeError(
        "Sigma_syn is not positive definite."
    )

inv_syn = np.linalg.inv(
    Sigma_syn
)

logdet_syn = np.linalg.slogdet(
    Sigma_syn
)[1]

LOG2PI = np.log(
    2.0 * np.pi
)

print("\nSynthetic covariance:")
print(
    "  Mean RF STD      = %.4f m"
    % np.sqrt(
        np.diag(
            Sigma_RF
        )
    ).mean()
)

print(
    "  Mean surr STD LW = %.4f m"
    % np.sqrt(
        np.diag(
            Sigma_surr
        )
    ).mean()
)

print(
    "  sigma_obs        = %.4f m"
    % SIGMA_OBS
)

print(
    "  Mean bias        = %+.4f m"
    % bias.mean()
)

print(
    "  Sigma_syn min eig = %.6e"
    % eig_syn.min()
)


# ============================================================================
# 10. Load production DNN
# ============================================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else
    "cpu"
)

print(
    "Device             :",
    device
)

sx = joblib.load(
    SCALER_X_PATH
)

sy = joblib.load(
    SCALER_Y_PATH
)

model = GroupMeanDNN().to(
    device
)

model.load_state_dict(
    torch_load(
        MODEL_PATH,
        device
    )
)

model.eval()

sx_mean = np.asarray(
    sx.mean_,
    dtype=float
)

sx_scale = np.asarray(
    sx.scale_,
    dtype=float
)

sy_mean = np.asarray(
    sy.mean_,
    dtype=float
)

sy_scale = np.asarray(
    sy.scale_,
    dtype=float
)


def surrogate(theta):

    theta = np.asarray(
        theta,
        dtype=float
    )

    if theta.ndim == 1:
        theta = theta.reshape(
            1,
            -1
        )

    xs = (
        theta
        - sx_mean
    ) / sx_scale

    xt = torch.tensor(
        xs,
        dtype=torch.float32,
        device=device
    )

    with torch.no_grad():

        ys = model(
            xt
        ).cpu().numpy()

    y = (
        ys * sy_scale
        + sy_mean
    )

    # Step-8 residual sign:
    # Abaqus group mean - DNN OOF
    return (
        y
        + bias.reshape(
            1,
            -1
        )
    )


# ============================================================================
# 11. Result collectors
# ============================================================================

all_parameter_rows = []
truth_metric_rows = []
predictive_rows = []


# ============================================================================
# 12. Loop through the 3 independent truths
# ============================================================================

for truth_id in [
    1,
    2,
    3
]:

    print("\n" + "#" * 100)
    print(
        "TRUTH %d"
        % truth_id
    )
    print("#" * 100)

    out_truth = os.path.join(
        OUTPUT_DIR,
        "Truth_%d"
        % truth_id
    )

    os.makedirs(
        out_truth,
        exist_ok=True
    )

    row_theta = truth_df.loc[
        truth_df[
            "Truth_ID"
        ] == truth_id
    ].iloc[0]

    row_raw = raw_df.loc[
        raw_df[
            "Truth_ID"
        ] == truth_id
    ].iloc[0]

    row_obs = obs_df.loc[
        obs_df[
            "Truth_ID"
        ] == truth_id
    ].iloc[0]

    theta_true = row_theta[
        X_COLS
    ].to_numpy(float)

    y_raw = row_raw[
        Y_COLS
    ].to_numpy(float)

    y_obs = row_obs[
        Y_COLS
    ].to_numpy(float)

    # ------------------------------------------------------------------------
    # 12A. Forward consistency at exact true theta
    # ------------------------------------------------------------------------

    mu_true = surrogate(
        theta_true
    )[0]

    forward_rmse_raw = float(
        np.sqrt(
            mean_squared_error(
                y_raw,
                mu_true
            )
        )
    )

    forward_mae_raw = float(
        mean_absolute_error(
            y_raw,
            mu_true
        )
    )

    mah2_raw = mahalanobis2(
        y_raw - mu_true,
        Sigma_model
    )

    mah2_obs = mahalanobis2(
        y_obs - mu_true,
        Sigma_syn
    )

    print(
        "Forward DNN-vs-raw-Abaqus RMSE at true theta = %.4f m"
        % forward_rmse_raw
    )

    print(
        "Forward DNN-vs-raw-Abaqus MAE  at true theta = %.4f m"
        % forward_mae_raw
    )

    print(
        "Mahalanobis^2 raw/model covariance             = %.4f"
        % mah2_raw
    )

    print(
        "Mahalanobis^2 noisy/synthetic covariance       = %.4f"
        % mah2_obs
    )

    print(
        "Chi-square(23) 95%% reference                  = %.4f"
        % CHI2_23_95
    )

    # ------------------------------------------------------------------------
    # 12B. Log posterior for this truth
    # ------------------------------------------------------------------------

    def log_prob_batch(theta):

        theta = np.asarray(
            theta,
            dtype=float
        )

        if theta.ndim == 1:
            theta = theta.reshape(
                1,
                -1
            )

        out = np.full(
            theta.shape[0],
            -np.inf,
            dtype=float
        )

        valid = np.all(
            (
                theta >= LOWER.reshape(
                    1,
                    -1
                )
            )
            &
            (
                theta <= UPPER.reshape(
                    1,
                    -1
                )
            ),
            axis=1
        )

        if not np.any(
            valid
        ):
            return out

        idx = np.where(
            valid
        )[0]

        mu = surrogate(
            theta[
                idx
            ]
        )

        r = (
            y_obs.reshape(
                1,
                -1
            )
            - mu
        )

        quad = np.einsum(
            "bi,ij,bj->b",
            r,
            inv_syn,
            r
        )

        ll = -0.5 * (
            NOBS * LOG2PI
            + logdet_syn
            + quad
        )

        out[
            idx
        ] = (
            LOG_PRIOR_CONST
            + ll
        )

        return out

    # ------------------------------------------------------------------------
    # 12C. Prescreen / initialize
    # ------------------------------------------------------------------------

    rng = np.random.RandomState(
        SEED
        + 1000 * truth_id
    )

    cand = (
        LOWER.reshape(
            1,
            -1
        )
        + rng.rand(
            N_PRESCREEN,
            NDIM
        )
        * WIDTH.reshape(
            1,
            -1
        )
    )

    lp = np.empty(
        N_PRESCREEN,
        dtype=float
    )

    for start in range(
        0,
        N_PRESCREEN,
        PRESCREEN_BATCH
    ):

        stop = min(
            start + PRESCREEN_BATCH,
            N_PRESCREEN
        )

        lp[
            start:stop
        ] = log_prob_batch(
            cand[
                start:stop
            ]
        )

    order = np.argsort(
        lp
    )[::-1]

    pool = order[
        :TOP_POOL
    ]

    chosen = rng.choice(
        pool,
        size=N_WALKERS,
        replace=False
    )

    p0 = cand[
        chosen
    ].copy()

    print(
        "Best prescreen log posterior = %.6f"
        % lp[
            order[0]
        ]
    )

    # ------------------------------------------------------------------------
    # 12D. MCMC
    # ------------------------------------------------------------------------

    sampler = emcee.EnsembleSampler(
        N_WALKERS,
        NDIM,
        log_prob_batch,
        vectorize=True
    )

    sampler.run_mcmc(
        p0,
        N_STEPS,
        progress=True
    )

    chain = sampler.get_chain()

    logp_chain = sampler.get_log_prob()

    acceptance = float(
        np.mean(
            sampler.acceptance_fraction
        )
    )

    post_chain = chain[
        BURN_IN:
    ]

    post_logp = logp_chain[
        BURN_IN:
    ]

    flat = post_chain.reshape(
        -1,
        NDIM
    )

    flat_lp = post_logp.reshape(
        -1
    )

    rhat = split_rhat(
        post_chain
    )

    rank_rhat = rank_normalized_split_rhat(
        post_chain
    )

    try:

        tau = np.asarray(
            emcee.autocorr.integrated_time(
                post_chain,
                quiet=True
            ),
            dtype=float
        )

    except Exception as exc:

        warnings.warn(
            "Truth %d tau unavailable: %s"
            % (
                truth_id,
                str(exc)
            )
        )

        tau = np.full(
            NDIM,
            np.nan
        )

    total_post = (
        post_chain.shape[0]
        * post_chain.shape[1]
    )

    ess = (
        total_post
        / tau
    )

    # ------------------------------------------------------------------------
    # 12E. Posterior parameter summary
    # ------------------------------------------------------------------------

    map_idx = int(
        np.nanargmax(
            flat_lp
        )
    )

    map_theta = flat[
        map_idx
    ]

    median_theta = np.median(
        flat,
        axis=0
    )

    param_rows_this_truth = []

    n_hit = 0

    for j, name in enumerate(
        X_COLS
    ):

        samples_j = flat[
            :,
            j
        ]

        lo, hi = hdi_1d(
            samples_j,
            0.95
        )

        true_value = float(
            theta_true[j]
        )

        hit = bool(
            (
                true_value >= lo
            )
            and
            (
                true_value <= hi
            )
        )

        if hit:
            n_hit += 1

        hdi_width = (
            hi
            - lo
        )

        prior_width = WIDTH[
            j
        ]

        width_ratio = (
            hdi_width
            / prior_width
        )

        width_contraction = (
            1.0
            - width_ratio
        )

        posterior_sd = float(
            np.std(
                samples_j,
                ddof=1
            )
        )

        sd_ratio = (
            posterior_sd
            / PRIOR_SD[
                j
            ]
        )

        rec = {
            "Truth_ID":
                truth_id,

            "Parameter":
                name,

            "True":
                true_value,

            "MAP":
                float(
                    map_theta[j]
                ),

            "Mean":
                float(
                    np.mean(
                        samples_j
                    )
                ),

            "Median":
                float(
                    np.median(
                        samples_j
                    )
                ),

            "Posterior_SD":
                posterior_sd,

            "HDI95_Low":
                lo,

            "HDI95_High":
                hi,

            "Truth_in_HDI95":
                hit,

            "Prior_Width":
                float(
                    prior_width
                ),

            "HDI95_Width":
                float(
                    hdi_width
                ),

            "HDI95_Width_Ratio_to_Prior":
                float(
                    width_ratio
                ),

            "HDI95_Width_Contraction":
                float(
                    width_contraction
                ),

            "Prior_SD":
                float(
                    PRIOR_SD[
                        j
                    ]
                ),

            "Posterior_SD_Ratio_to_Prior_SD":
                float(
                    sd_ratio
                ),

            "Split_Rhat":
                float(
                    rhat[j]
                ),

            "RankNormalized_Split_Rhat":
                float(
                    rank_rhat[j]
                )
                if np.isfinite(
                    rank_rhat[j]
                )
                else np.nan,

            "Tau":
                float(
                    tau[j]
                )
                if np.isfinite(
                    tau[j]
                )
                else np.nan,

            "ESS":
                float(
                    ess[j]
                )
                if np.isfinite(
                    ess[j]
                )
                else np.nan
        }

        all_parameter_rows.append(
            rec
        )

        param_rows_this_truth.append(
            rec
        )

    pd.DataFrame(
        param_rows_this_truth
    ).to_csv(
        os.path.join(
            out_truth,
            "Posterior_Parameter_Summary_Truth%d.csv"
            % truth_id
        ),
        index=False,
        encoding="utf-8-sig"
    )

    # Save full chain compactly + thinned human-readable CSV.
    np.savez_compressed(
        os.path.join(
            out_truth,
            "MCMC_Full_PostBurn_Truth%d.npz"
            % truth_id
        ),
        chain=post_chain,
        log_posterior=post_logp
    )

    thin_idx = np.arange(
        0,
        len(
            flat
        ),
        CSV_THIN
    )

    thin_df = pd.DataFrame(
        flat[
            thin_idx
        ],
        columns=X_COLS
    )

    thin_df[
        "log_posterior"
    ] = flat_lp[
        thin_idx
    ]

    thin_df.to_csv(
        os.path.join(
            out_truth,
            "Posterior_Samples_Thinned_Truth%d.csv"
            % truth_id
        ),
        index=False,
        encoding="utf-8-sig"
    )

    # ------------------------------------------------------------------------
    # 12F. Posterior predictive
    # ------------------------------------------------------------------------

    rng_pp = np.random.RandomState(
        SEED
        + 9000
        + truth_id
    )

    npp = min(
        N_PP,
        len(
            flat
        )
    )

    pp_idx = rng_pp.choice(
        len(
            flat
        ),
        size=npp,
        replace=False
    )

    pp_theta = flat[
        pp_idx
    ]

    pp_mu = surrogate(
        pp_theta
    )

    pp_draws = np.empty_like(
        pp_mu
    )

    for i in range(
        npp
    ):

        eps = rng_pp.normal(
            size=NOBS
        )

        pp_draws[
            i
        ] = (
            pp_mu[
                i
            ]
            + Q_syn @ (
                np.sqrt(
                    eig_syn
                )
                * eps
            )
        )

    pp_lo = np.quantile(
        pp_draws,
        0.025,
        axis=0
    )

    pp_md = np.quantile(
        pp_draws,
        0.50,
        axis=0
    )

    pp_hi = np.quantile(
        pp_draws,
        0.975,
        axis=0
    )

    obs_covered = (
        (
            y_obs >= pp_lo
        )
        &
        (
            y_obs <= pp_hi
        )
    )

    raw_covered = (
        (
            y_raw >= pp_lo
        )
        &
        (
            y_raw <= pp_hi
        )
    )

    obs_coverage = float(
        np.mean(
            obs_covered
        )
    )

    raw_coverage = float(
        np.mean(
            raw_covered
        )
    )

    map_mu = surrogate(
        map_theta
    )[0]

    median_mu = surrogate(
        median_theta
    )[0]

    map_rmse_obs = float(
        np.sqrt(
            mean_squared_error(
                y_obs,
                map_mu
            )
        )
    )

    map_mae_obs = float(
        mean_absolute_error(
            y_obs,
            map_mu
        )
    )

    map_rmse_raw = float(
        np.sqrt(
            mean_squared_error(
                y_raw,
                map_mu
            )
        )
    )

    # Per-borehole table.
    pred_table = pd.DataFrame({
        "Borehole":
            Y_COLS,

        "Raw_Abaqus_Truth_Head_m":
            y_raw,

        "Synthetic_Observation_m":
            y_obs,

        "Surrogate_Mean_at_TrueTheta_m":
            mu_true,

        "MAP_Mean_Head_m":
            map_mu,

        "PosteriorPredictive_P2.5_m":
            pp_lo,

        "PosteriorPredictive_Median_m":
            pp_md,

        "PosteriorPredictive_P97.5_m":
            pp_hi,

        "SyntheticObs_Inside_PP95":
            obs_covered,

        "RawTruthHead_Inside_PP95":
            raw_covered
    })

    pred_table.to_csv(
        os.path.join(
            out_truth,
            "Posterior_Predictive_Truth%d.csv"
            % truth_id
        ),
        index=False,
        encoding="utf-8-sig"
    )

    for j, bh in enumerate(
        Y_COLS
    ):

        predictive_rows.append({
            "Truth_ID":
                truth_id,

            "Borehole":
                bh,

            "Raw_Abaqus_Truth_Head_m":
                float(
                    y_raw[j]
                ),

            "Synthetic_Observation_m":
                float(
                    y_obs[j]
                ),

            "Surrogate_Mean_at_TrueTheta_m":
                float(
                    mu_true[j]
                ),

            "PP95_Low_m":
                float(
                    pp_lo[j]
                ),

            "PP_Median_m":
                float(
                    pp_md[j]
                ),

            "PP95_High_m":
                float(
                    pp_hi[j]
                ),

            "SyntheticObs_Inside_PP95":
                bool(
                    obs_covered[j]
                ),

            "RawTruthHead_Inside_PP95":
                bool(
                    raw_covered[j]
                )
        })

    # ------------------------------------------------------------------------
    # 12G. Truth-level metrics
    # ------------------------------------------------------------------------

    mean_contraction = float(
        np.mean(
            [
                r[
                    "HDI95_Width_Contraction"
                ]
                for r in param_rows_this_truth
            ]
        )
    )

    metric_row = {
        "Truth_ID":
            truth_id,

        "Nearest_Training_P_index":
            int(
                row_theta[
                    "Nearest_Training_P_index"
                ]
            )
            if "Nearest_Training_P_index"
            in row_theta.index
            else np.nan,

        "Nearest_Training_Distance_Normalized":
            float(
                row_theta[
                    "Nearest_Training_Distance_Normalized"
                ]
            )
            if "Nearest_Training_Distance_Normalized"
            in row_theta.index
            else np.nan,

        "Forward_RMSE_TrueTheta_vs_RawAbaqus_m":
            forward_rmse_raw,

        "Forward_MAE_TrueTheta_vs_RawAbaqus_m":
            forward_mae_raw,

        "Mahalanobis2_Raw_ModelCov":
            mah2_raw,

        "Mahalanobis2_Noisy_SyntheticCov":
            mah2_obs,

        "Chi2_23_95":
            CHI2_23_95,

        "Raw_Mahalanobis_PASS_95":
            bool(
                mah2_raw
                <= CHI2_23_95
            ),

        "Noisy_Mahalanobis_PASS_95":
            bool(
                mah2_obs
                <= CHI2_23_95
            ),

        "Acceptance_Fraction":
            acceptance,

        "Max_Split_Rhat":
            float(
                np.nanmax(
                    rhat
                )
            ),

        "Max_RankNormalized_Split_Rhat":
            float(
                np.nanmax(
                    rank_rhat
                )
            )
            if np.isfinite(
                rank_rhat
            ).any()
            else np.nan,

        "Min_ESS":
            float(
                np.nanmin(
                    ess
                )
            )
            if np.isfinite(
                ess
            ).any()
            else np.nan,

        "Truth_Parameters_in_HDI95_Count":
            n_hit,

        "Truth_Parameters_in_HDI95_Total":
            6,

        "Truth_Parameter_HDI95_Coverage_Percent":
            100.0
            * n_hit
            / 6.0,

        "Mean_HDI95_Width_Contraction":
            mean_contraction,

        "MAP_RMSE_to_SyntheticObs_m":
            map_rmse_obs,

        "MAP_MAE_to_SyntheticObs_m":
            map_mae_obs,

        "MAP_RMSE_to_RawTruthHead_m":
            map_rmse_raw,

        "PosteriorPredictive95_Coverage_SyntheticObs_Percent":
            100.0
            * obs_coverage,

        "PosteriorPredictive95_Coverage_RawTruthHead_Percent":
            100.0
            * raw_coverage
    }

    truth_metric_rows.append(
        metric_row
    )

    # ------------------------------------------------------------------------
    print("\nTruth %d summary:" % truth_id)
    print("  acceptance                   = %.4f" % acceptance)
    print("  max classic split-Rhat       = %.5f" % np.nanmax(rhat))

    if np.isfinite(rank_rhat).any():
        print("  max rank-normalized Rhat     = %.5f" % np.nanmax(rank_rhat))
    else:
        print("  max rank-normalized Rhat     = NA")

    if np.isfinite(ess).any():
        print("  min ESS                      = %.1f" % np.nanmin(ess))

    print("  true parameters in 95%% HDI  = %d / 6" % n_hit)
    print("  mean HDI-width contraction   = %.2f%%" % (100.0*mean_contraction))
    print("  MAP RMSE to synthetic obs    = %.4f m" % map_rmse_obs)
    print("  PP95 coverage synthetic obs  = %.2f%%" % (100.0*obs_coverage))
    print("  PP95 coverage raw truth head = %.2f%%" % (100.0*raw_coverage))


# ============================================================================
# 13. Aggregate outputs
# ============================================================================

param_all_df = pd.DataFrame(
    all_parameter_rows
)

PARAM_ALL_PATH = os.path.join(
    OUTPUT_DIR,
    "Independent_Validation_All18_Parameter_Results.csv"
)

param_all_df.to_csv(
    PARAM_ALL_PATH,
    index=False,
    encoding="utf-8-sig"
)


truth_metrics_df = pd.DataFrame(
    truth_metric_rows
)

TRUTH_METRICS_PATH = os.path.join(
    OUTPUT_DIR,
    "Independent_Validation_Truth_Level_Metrics.csv"
)

truth_metrics_df.to_csv(
    TRUTH_METRICS_PATH,
    index=False,
    encoding="utf-8-sig"
)


predictive_df = pd.DataFrame(
    predictive_rows
)

PRED_ALL_PATH = os.path.join(
    OUTPUT_DIR,
    "Independent_Validation_All69_Predictive_Results.csv"
)

predictive_df.to_csv(
    PRED_ALL_PATH,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================================
# 14. Aggregate parameter-wise results
# ============================================================================

parameter_summary_rows = []

for name in X_COLS:

    d = param_all_df.loc[
        param_all_df[
            "Parameter"
        ] == name
    ]

    parameter_summary_rows.append({
        "Parameter":
            name,

        "Truth_Count":
            len(d),

        "Truth_in_HDI95_Count":
            int(
                d[
                    "Truth_in_HDI95"
                ].sum()
            ),

        "Truth_in_HDI95_Percent":
            100.0
            * float(
                d[
                    "Truth_in_HDI95"
                ].mean()
            ),

        "Mean_HDI95_Width_Contraction":
            float(
                d[
                    "HDI95_Width_Contraction"
                ].mean()
            ),

        "Mean_Posterior_SD_Ratio_to_Prior_SD":
            float(
                d[
                    "Posterior_SD_Ratio_to_Prior_SD"
                ].mean()
            )
    })


PARAM_SUMMARY_PATH = os.path.join(
    OUTPUT_DIR,
    "Independent_Validation_ParameterWise_Summary.csv"
)

pd.DataFrame(
    parameter_summary_rows
).to_csv(
    PARAM_SUMMARY_PATH,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================================
# 15. Overall summary
# ============================================================================

total_hits = int(
    param_all_df[
        "Truth_in_HDI95"
    ].sum()
)

total_params = len(
    param_all_df
)

overall_param_coverage = (
    100.0
    * total_hits
    / total_params
)

all_obs_cov = float(
    predictive_df[
        "SyntheticObs_Inside_PP95"
    ].mean()
)

all_raw_cov = float(
    predictive_df[
        "RawTruthHead_Inside_PP95"
    ].mean()
)

overall = {
    "workflow":
        "Independent synthetic validation with 3 corrected-RF V2 Abaqus truths",

    "truth_count":
        3,

    "parameters_per_truth":
        6,

    "total_truth_parameter_cases":
        total_params,

    "truth_parameter_cases_in_95_HDI":
        total_hits,

    "truth_parameter_95_HDI_coverage_percent":
        overall_param_coverage,

    "mean_HDI95_width_contraction":
        float(
            param_all_df[
                "HDI95_Width_Contraction"
            ].mean()
        ),

    "mean_posterior_SD_ratio_to_prior_SD":
        float(
            param_all_df[
                "Posterior_SD_Ratio_to_Prior_SD"
            ].mean()
        ),

    "synthetic_observation_PP95_coverage_percent_all69":
        100.0
        * all_obs_cov,

    "raw_truth_head_PP95_coverage_percent_all69":
        100.0
        * all_raw_cov,

    "mean_forward_RMSE_trueTheta_vs_rawAbaqus_m":
        float(
            truth_metrics_df[
                "Forward_RMSE_TrueTheta_vs_RawAbaqus_m"
            ].mean()
        ),

    "all_truths_raw_Mahalanobis_PASS_95":
        bool(
            truth_metrics_df[
                "Raw_Mahalanobis_PASS_95"
            ].all()
        ),

    "all_truths_noisy_Mahalanobis_PASS_95":
        bool(
            truth_metrics_df[
                "Noisy_Mahalanobis_PASS_95"
            ].all()
        ),

    "mean_acceptance_fraction":
        float(
            truth_metrics_df[
                "Acceptance_Fraction"
            ].mean()
        ),

    "max_split_Rhat_across_truths":
        float(
            truth_metrics_df[
                "Max_Split_Rhat"
            ].max()
        ),

    "max_rank_normalized_split_Rhat_across_truths":
        float(
            truth_metrics_df[
                "Max_RankNormalized_Split_Rhat"
            ].max()
        )
        if truth_metrics_df[
            "Max_RankNormalized_Split_Rhat"
        ].notna().any()
        else np.nan,

    "min_ESS_across_truths":
        float(
            truth_metrics_df[
                "Min_ESS"
            ].min()
        ),

    "likelihood_covariance":
        "Sigma_RF + Sigma_surr_LW + 0.5^2 I",

    "sigma_rep_used":
        False
}


OVERALL_PATH = os.path.join(
    OUTPUT_DIR,
    "Independent_Validation_Overall_Summary.json"
)

with open(
    OVERALL_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        overall,
        f,
        indent=4,
        ensure_ascii=False
    )


# ============================================================================
# 16. Final console report
# ============================================================================

print("\n" + "=" * 100)
print("STEP 12B COMPLETE - FINAL INDEPENDENT SYNTHETIC MCMC STRENGTHENING")
print("=" * 100)

print(
    "Truth parameters inside 95%% HDI : %d / %d = %.2f%%"
    % (
        total_hits,
        total_params,
        overall_param_coverage
    )
)

print(
    "Mean HDI95 width contraction    : %.2f%%"
    % (
        100.0
        * param_all_df[
            "HDI95_Width_Contraction"
        ].mean()
    )
)

print(
    "Synthetic obs PP95 coverage     : %.2f%% (%d / %d)"
    % (
        100.0 * all_obs_cov,
        int(
            predictive_df[
                "SyntheticObs_Inside_PP95"
            ].sum()
        ),
        len(
            predictive_df
        )
    )
)

print(
    "Raw truth head PP95 coverage    : %.2f%% (%d / %d)"
    % (
        100.0 * all_raw_cov,
        int(
            predictive_df[
                "RawTruthHead_Inside_PP95"
            ].sum()
        ),
        len(
            predictive_df
        )
    )
)

print(
    "Mean true-theta forward RMSE    : %.4f m"
    % truth_metrics_df[
        "Forward_RMSE_TrueTheta_vs_RawAbaqus_m"
    ].mean()
)

print(
    "All raw Mahalanobis tests PASS  : %s"
    % str(
        bool(
            truth_metrics_df[
                "Raw_Mahalanobis_PASS_95"
            ].all()
        )
    )
)

print(
    "All noisy Mahalanobis tests PASS: %s"
    % str(
        bool(
            truth_metrics_df[
                "Noisy_Mahalanobis_PASS_95"
            ].all()
        )
    )
)

print(
    "Mean acceptance fraction        : %.4f"
    % truth_metrics_df[
        "Acceptance_Fraction"
    ].mean()
)

print(
    "Max classic split-Rhat          : %.5f"
    % truth_metrics_df[
        "Max_Split_Rhat"
    ].max()
)

if truth_metrics_df[
    "Max_RankNormalized_Split_Rhat"
].notna().any():
    print(
        "Max rank-normalized Rhat      : %.5f"
        % truth_metrics_df[
            "Max_RankNormalized_Split_Rhat"
        ].max()
    )

print(
    "Min ESS across truths           : %.1f"
    % truth_metrics_df[
        "Min_ESS"
    ].min()
)

print("\nTruth-level metrics:")
print(
    truth_metrics_df.to_string(
        index=False
    )
)

print("\nSaved:")
print("  " + PARAM_ALL_PATH)
print("  " + TRUTH_METRICS_PATH)
print("  " + PARAM_SUMMARY_PATH)
print("  " + PRED_ALL_PATH)
print("  " + OVERALL_PATH)


# Final strengthened convergence verdict
max_classic_final = float(truth_metrics_df["Max_Split_Rhat"].max())
max_rank_final = (
    float(truth_metrics_df["Max_RankNormalized_Split_Rhat"].max())
    if truth_metrics_df["Max_RankNormalized_Split_Rhat"].notna().any()
    else np.nan
)
min_ess_final = float(truth_metrics_df["Min_ESS"].min())

final_pass = (
    max_classic_final < 1.01
    and (np.isnan(max_rank_final) or max_rank_final < 1.01)
    and min_ess_final > 3000.0
)

print("\nConvergence target              : max Rhat < 1.01, min ESS > 3000")
print("ALL SYNTHETIC MCMC PASS         : %s" % str(bool(final_pass)))
print("NOTE                            : NO sigma_rep was used.")
