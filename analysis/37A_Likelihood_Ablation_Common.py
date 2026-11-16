"""Shared utilities for Step-13 likelihood-component ablation experiments.

The scientific control is strict: all cases use the same production DNN,
parameter priors, observations, random seeds, MCMC settings and diagnostics.
Only the likelihood error components declared in ablation_cases.json change.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
import random
from pathlib import Path
import emcee
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import differential_evolution
from scipy.stats import norm, rankdata
from sklearn.metrics import mean_absolute_error, mean_squared_error
WORK_DIR = Path(os.environ.get('GHB_WORK_DIR', str(Path(__file__).resolve().parent)))
RESULT_ROOT = WORK_DIR / 'Step13_Likelihood_Ablation_CorrectedRF_FinalGW'
CONFIG_PATH = RESULT_ROOT / '00_config' / 'ablation_cases.json'
STEP8_DIR = WORK_DIR / 'Step8_5Fold_OOF_SurrogateError_CorrectedRF_FinalGW'
PRODUCTION_DIR = STEP8_DIR / 'Production_DNN_All50'
SIGMA_RF_PATH = WORK_DIR / 'Sigma_RF_Pooled_23x23_CorrectedRF_FinalGW.csv'
SIGMA_SURR_PATH = STEP8_DIR / 'Sigma_surr_LedoitWolf_23x23_CorrectedRF.csv'
BIAS_PATH = STEP8_DIR / 'Surrogate_Bias_Vector_CorrectedRF.csv'
MODEL_PATH = PRODUCTION_DIR / 'GroupMean_DNN_All50_CorrectedRF_253epochs.pth'
SCALER_X_PATH = PRODUCTION_DIR / 'scaler_X_All50_CorrectedRF.pkl'
SCALER_Y_PATH = PRODUCTION_DIR / 'scaler_Y_All50_CorrectedRF.pkl'
TRUTH_DIR = WORK_DIR / 'synthetic'
STEP11_DIR = TRUTH_DIR / 'Step11_Independent_Validation'
TRUTH_PARAM_PATH = TRUTH_DIR / 'Truth_Parameters_3_CorrectedRF_V2.csv'
RAW_HEADS_PATH = STEP11_DIR / 'Truth_Heads_Raw_Abaqus_3x23.csv'
SYN_OBS_PATH = STEP11_DIR / 'Truth_Synthetic_Observations_3x23.csv'
X_COLS = ['log10_K_Lower', 'log10_K_Middle', 'log10_K_Upper', 'COV', 'dh', 'dv']
Y_COLS = ['zk101', 'zk102', 'zk103', 'zk104', 'zk105', 'zk106', 'zk107', 'zk108', 'zk109', 'zk110', 'zk120', 'zk131', 'zk132', 'zk133', 'zk134', 'zk135', 'zk136', 'zk137', 'zk138', 'zk139', 'zk140', 'zk141', 'zk410']
NOBS = len(Y_COLS)
PRIOR_BOUNDS = {'log10_K_Lower': (-9.0, -7.0), 'log10_K_Middle': (-8.0, -6.0), 'log10_K_Upper': (-7.0, -5.0), 'COV': (0.4, 1.0), 'dh': (50.0, 300.0), 'dv': (1.0, 5.0)}
LOWER = np.array([PRIOR_BOUNDS[k][0] for k in X_COLS], dtype=float)
UPPER = np.array([PRIOR_BOUNDS[k][1] for k in X_COLS], dtype=float)
WIDTH = UPPER - LOWER
PRIOR_SD = WIDTH / np.sqrt(12.0)
LOG_PRIOR_PHYS = -float(np.sum(np.log(WIDTH)))
SIGMA_OBS = 0.5
CHI2_23_95 = 35.17246162690806
LOG2PI = float(np.log(2.0 * np.pi))
OBSERVED_HEADS = {'zk101': 496.2, 'zk102': 489.7, 'zk103': 492.31, 'zk104': 495.09, 'zk105': 554.8, 'zk106': 543.74, 'zk107': 514.78, 'zk108': 502.34, 'zk109': 491.18, 'zk110': 476.07, 'zk120': 479.12, 'zk131': 490.96, 'zk132': 489.47, 'zk133': 484.04, 'zk134': 491.21, 'zk135': 482.89, 'zk136': 508.4, 'zk137': 521.43, 'zk138': 494.36, 'zk139': 509.17, 'zk140': 496.48, 'zk141': 493.23, 'zk410': 550.28}
Y_FIELD = np.array([OBSERVED_HEADS[k] for k in Y_COLS], dtype=float)
MODE_SETTINGS = {'smoke': {'n_walkers': 64, 'n_steps': 300, 'burn_in': 100, 'n_prescreen': 2000, 'prescreen_batch': 1024, 'top_pool': 256, 'n_pp': 200, 'csv_thin': 5}, 'full': {'n_walkers': 64, 'n_steps': 50000, 'burn_in': 10000, 'n_prescreen': 30000, 'prescreen_batch': 2048, 'top_pool': 640, 'n_pp': 8000, 'csv_thin': 20}, 'extended': {'n_walkers': 64, 'n_steps': 200000, 'burn_in': 40000, 'n_prescreen': 50000, 'prescreen_batch': 2048, 'top_pool': 1024, 'n_pp': 10000, 'csv_thin': 80, 'use_de_moves': True}}

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_cases() -> dict:
    with CONFIG_PATH.open('r', encoding='utf-8') as f:
        return json.load(f)

def get_case(case_name: str) -> dict:
    cases = load_cases()
    if case_name not in cases:
        raise KeyError('Unknown case %r; available: %s' % (case_name, sorted(cases)))
    case = dict(cases[case_name])
    case['case_name'] = case_name
    return case

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def write_input_manifest(out_dir: Path, extra_paths: list[Path] | None=None) -> Path:
    paths = [CONFIG_PATH, SIGMA_RF_PATH, SIGMA_SURR_PATH, BIAS_PATH, MODEL_PATH, SCALER_X_PATH, SCALER_Y_PATH]
    if extra_paths:
        paths.extend(extra_paths)
    manifest = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        manifest.append({'path': str(path), 'size_bytes': int(path.stat().st_size), 'sha256': sha256(path)})
    out = out_dir / 'Input_Files_and_Hashes.json'
    with out.open('w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return out

class GroupMeanDNN(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 23))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)

def load_cov(path: Path) -> np.ndarray:
    d = pd.read_csv(path, index_col=0)
    missing = [x for x in Y_COLS if x not in d.index or x not in d.columns]
    if missing:
        raise RuntimeError('Covariance labels missing in %s: %s' % (path, missing))
    m = d.loc[Y_COLS, Y_COLS].to_numpy(float)
    return 0.5 * (m + m.T)

def load_bias(path: Path) -> np.ndarray:
    d = pd.read_csv(path)
    required = {'Borehole', 'Residual_Mean_Bias_m'}
    if not required.issubset(d.columns):
        raise RuntimeError('Bias file columns are wrong: %s' % path)
    s = d.set_index('Borehole')['Residual_Mean_Bias_m']
    return s.loc[Y_COLS].to_numpy(float)

class ModelContext:

    def __init__(self) -> None:
        for path in [SIGMA_RF_PATH, SIGMA_SURR_PATH, BIAS_PATH, MODEL_PATH, SCALER_X_PATH, SCALER_Y_PATH]:
            if not path.exists():
                raise FileNotFoundError(path)
        self.sigma_rf = load_cov(SIGMA_RF_PATH)
        self.sigma_surr = load_cov(SIGMA_SURR_PATH)
        self.bias = load_bias(BIAS_PATH)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler_x = joblib.load(SCALER_X_PATH)
        self.scaler_y = joblib.load(SCALER_Y_PATH)
        self.sx_mean = np.asarray(self.scaler_x.mean_, dtype=float)
        self.sx_scale = np.asarray(self.scaler_x.scale_, dtype=float)
        self.sy_mean = np.asarray(self.scaler_y.mean_, dtype=float)
        self.sy_scale = np.asarray(self.scaler_y.scale_, dtype=float)
        self.model = GroupMeanDNN().to(self.device)
        self.model.load_state_dict(torch_load(MODEL_PATH, self.device))
        self.model.eval()

    def predict(self, theta: np.ndarray, use_bias: bool) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        if theta.ndim == 1:
            theta = theta.reshape(1, -1)
        xs = (theta - self.sx_mean) / self.sx_scale
        xt = torch.tensor(xs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            ys = self.model(xt).cpu().numpy()
        y = ys * self.sy_scale + self.sy_mean
        if use_bias:
            y = y + self.bias.reshape(1, -1)
        return y

    def build_covariance(self, case: dict) -> np.ndarray:
        sigma = SIGMA_OBS ** 2 * np.eye(NOBS)
        if case['use_sigma_rf']:
            sigma = sigma + self.sigma_rf
        if case['use_sigma_surr']:
            sigma = sigma + self.sigma_surr
        sigma = 0.5 * (sigma + sigma.T)
        eig = np.linalg.eigvalsh(sigma)
        if eig.min() <= 0:
            raise RuntimeError('Case %s covariance is not positive definite; min eigenvalue %.6e' % (case['case_name'], eig.min()))
        return sigma

def hdi_1d(samples: np.ndarray, mass: float=0.95) -> tuple[float, float]:
    x = np.sort(np.asarray(samples, dtype=float))
    n = len(x)
    m = int(np.floor(mass * n))
    if m <= 0 or m >= n:
        return (float(x[0]), float(x[-1]))
    widths = x[m:] - x[:n - m]
    i = int(np.argmin(widths))
    return (float(x[i]), float(x[i + m]))

def split_rhat(chain: np.ndarray) -> np.ndarray:
    ns, _, _ = chain.shape
    n = ns // 2
    a = np.transpose(chain[:n], (1, 0, 2))
    b = np.transpose(chain[-n:], (1, 0, 2))
    c = np.concatenate([a, b], axis=0)
    means = c.mean(axis=1)
    vars_ = c.var(axis=1, ddof=1)
    w = vars_.mean(axis=0)
    bvar = n * means.var(axis=0, ddof=1)
    v = (n - 1.0) / n * w + bvar / n
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.sqrt(v / w)

def _rank_normalize(values: np.ndarray) -> np.ndarray:
    a = np.asarray(values, dtype=float)
    flat = a.reshape(-1)
    ranks = rankdata(flat, method='average')
    n = len(flat)
    p = (ranks - 3.0 / 8.0) / (n + 1.0 / 4.0)
    return norm.ppf(p).reshape(a.shape)

def rank_normalized_split_rhat(chain: np.ndarray) -> np.ndarray:
    out = np.full(chain.shape[2], np.nan)
    for j in range(chain.shape[2]):
        x = chain[:, :, j]
        r_rank = split_rhat(_rank_normalize(x)[:, :, None])[0]
        folded = np.abs(x - np.median(x))
        r_fold = split_rhat(_rank_normalize(folded)[:, :, None])[0]
        out[j] = max(r_rank, r_fold)
    return out

def mahalanobis2(residual: np.ndarray, covariance: np.ndarray) -> float:
    lmat = np.linalg.cholesky(covariance)
    z = np.linalg.solve(lmat, np.asarray(residual, dtype=float))
    return float(z @ z)

def make_log_prob(context: ModelContext, y_obs: np.ndarray, case: dict) -> tuple[callable, dict]:
    base_cov = context.build_covariance(case)
    eig, qmat = np.linalg.eigh(base_cov)
    use_bias = bool(case['use_surrogate_bias'])
    use_rep = bool(case['use_sigma_rep'])
    rep_scale = float(case.get('sigma_rep_scale_m', 10.0))
    log_hn_const = 0.5 * np.log(2.0 / np.pi) - np.log(rep_scale)

    def log_prob_batch(par: np.ndarray) -> np.ndarray:
        par = np.asarray(par, dtype=float)
        if par.ndim == 1:
            par = par.reshape(1, -1)
        out = np.full(par.shape[0], -np.inf, dtype=float)
        theta = par[:, :6]
        valid = np.all((theta >= LOWER.reshape(1, -1)) & (theta <= UPPER.reshape(1, -1)), axis=1)
        if use_rep:
            sig = par[:, 6]
            valid = valid & np.isfinite(sig) & (sig > 0.0)
        if not np.any(valid):
            return out
        idx = np.where(valid)[0]
        mu = context.predict(theta[idx], use_bias=use_bias)
        r = y_obs.reshape(1, -1) - mu
        z = r @ qmat
        if use_rep:
            den = eig.reshape(1, -1) + sig[idx].reshape(-1, 1) ** 2
            quad = np.sum(z * z / den, axis=1)
            logdet = np.sum(np.log(den), axis=1)
            lp_rep = log_hn_const - 0.5 * (sig[idx] / rep_scale) ** 2
            out[idx] = LOG_PRIOR_PHYS + lp_rep - 0.5 * (NOBS * LOG2PI + logdet + quad)
        else:
            quad = np.sum(z * z / eig.reshape(1, -1), axis=1)
            logdet = float(np.sum(np.log(eig)))
            out[idx] = LOG_PRIOR_PHYS - 0.5 * (NOBS * LOG2PI + logdet + quad)
        return out
    info = {'case_name': case['case_name'], 'optimized_init': bool(case.get('optimized_init', False)), 'base_cov': base_cov, 'eig': eig, 'qmat': qmat, 'use_bias': use_bias, 'use_rep': use_rep, 'rep_scale': rep_scale, 'ndim': 7 if use_rep else 6}
    return (log_prob_batch, info)

def prescreen_initial_state(log_prob_batch, info: dict, settings: dict, seed: int) -> tuple[np.ndarray, dict]:
    rng = np.random.RandomState(seed + 100)
    n = int(settings['n_prescreen'])
    theta = LOWER.reshape(1, -1) + rng.rand(n, 6) * WIDTH.reshape(1, -1)
    optimization_result = None
    if (info.get('case_name') == 'A0_obs_only' or info.get('optimized_init')) and (not info['use_rep']):

        def objective(x):
            value = log_prob_batch(np.asarray(x, dtype=float))[0]
            return -value if np.isfinite(value) else 1e+300
        optimization_result = differential_evolution(objective, bounds=list(zip(LOWER, UPPER)), seed=seed + 77, maxiter=160, popsize=15, tol=1e-08, polish=True, updating='immediate', workers=1)
        xopt = np.asarray(optimization_result.x, dtype=float)
        local_scale = 10.0 ** rng.uniform(-3.3, -1.7, size=(n, 1))
        theta = xopt.reshape(1, -1) + rng.normal(size=(n, 6)) * WIDTH.reshape(1, -1) * local_scale
        eps = 1e-09 * WIDTH
        theta = np.clip(theta, LOWER + eps, UPPER - eps)
    if info['use_rep']:
        sig = np.abs(rng.normal(0.0, info['rep_scale'], size=n))
        sig = np.maximum(sig, 1e-06)
        candidates = np.column_stack([theta, sig])
    else:
        candidates = theta
    scores = np.empty(n, dtype=float)
    batch = int(settings['prescreen_batch'])
    for start in range(0, n, batch):
        stop = min(start + batch, n)
        scores[start:stop] = log_prob_batch(candidates[start:stop])
    finite = np.isfinite(scores)
    if finite.sum() < settings['n_walkers']:
        raise RuntimeError('Too few finite prescreen candidates: %d' % finite.sum())
    finite_idx = np.where(finite)[0]
    order = finite_idx[np.argsort(scores[finite_idx])]
    pool_size = min(int(settings['top_pool']), len(order))
    pool = order[-pool_size:]
    chosen = rng.choice(pool, size=int(settings['n_walkers']), replace=False)
    p0 = candidates[chosen].copy()
    result = {'prescreen_finite': int(finite.sum()), 'prescreen_best_log_prob': float(np.max(scores[finite])), 'prescreen_selected_min_log_prob': float(np.min(scores[chosen])), 'prescreen_selected_max_log_prob': float(np.max(scores[chosen]))}
    if optimization_result is not None:
        result.update({'initialization_strategy': 'differential_evolution_plus_local_multiscale_prescreen', 'optimizer_success': bool(optimization_result.success), 'optimizer_fun': float(optimization_result.fun), 'optimizer_nfev': int(optimization_result.nfev)})
    else:
        result['initialization_strategy'] = 'uniform_global_prescreen'
    return (p0, result)

def _safe_tau(post_chain: np.ndarray) -> np.ndarray:
    try:
        return np.asarray(emcee.autocorr.integrated_time(post_chain, quiet=True), dtype=float)
    except Exception:
        return np.full(post_chain.shape[2], np.nan)

def run_single_dataset(context: ModelContext, y_obs: np.ndarray, case: dict, out_dir: Path, mode: str, seed: int, raw_heads: np.ndarray | None=None, truth_theta: np.ndarray | None=None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = dict(MODE_SETTINGS[mode])
    if raw_heads is None and mode == 'full':
        settings['n_pp'] = 10000
    set_seed(seed)
    run_config = {'case': case, 'mode': mode, 'seed': int(seed), 'settings': settings, 'x_cols': X_COLS, 'y_cols': Y_COLS, 'sigma_obs_m': SIGMA_OBS, 'device': str(context.device)}
    with (out_dir / 'Run_Config.json').open('w', encoding='utf-8') as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)
    log_prob_batch, info = make_log_prob(context, y_obs, case)
    p0, prescreen = prescreen_initial_state(log_prob_batch, info, settings, seed)
    sampler_kwargs = {}
    if settings.get('use_de_moves', False):
        sampler_kwargs['moves'] = [(emcee.moves.DEMove(), 0.55), (emcee.moves.DESnookerMove(), 0.2), (emcee.moves.StretchMove(a=1.6), 0.25)]
    sampler = emcee.EnsembleSampler(int(settings['n_walkers']), int(info['ndim']), log_prob_batch, vectorize=True, **sampler_kwargs)
    sampler.run_mcmc(p0, int(settings['n_steps']), progress=True)
    chain = sampler.get_chain()
    logp = sampler.get_log_prob()
    burn = int(settings['burn_in'])
    post_chain = chain[burn:]
    post_logp = logp[burn:]
    flat = post_chain.reshape(-1, info['ndim'])
    flat_logp = post_logp.reshape(-1)
    map_index = int(np.argmax(flat_logp))
    map_par = flat[map_index]
    med_par = np.median(flat, axis=0)
    params = X_COLS + (['sigma_rep'] if info['use_rep'] else [])
    rhat = split_rhat(post_chain)
    rank_rhat = rank_normalized_split_rhat(post_chain)
    tau = _safe_tau(post_chain)
    n_total = post_chain.shape[0] * post_chain.shape[1]
    ess = np.where(np.isfinite(tau) & (tau > 0), n_total / tau, np.nan)
    rows = []
    for j, name in enumerate(params):
        lo, hi = hdi_1d(flat[:, j])
        rows.append({'Parameter': name, 'MAP': float(map_par[j]), 'Mean': float(np.mean(flat[:, j])), 'Median': float(med_par[j]), 'SD': float(np.std(flat[:, j], ddof=1)), 'HDI95_Low': lo, 'HDI95_High': hi, 'Split_Rhat': float(rhat[j]), 'RankNormalized_Split_Rhat': float(rank_rhat[j]), 'Tau': float(tau[j]) if np.isfinite(tau[j]) else np.nan, 'ESS': float(ess[j]) if np.isfinite(ess[j]) else np.nan})
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / 'Posterior_Parameter_Summary.csv', index=False)
    thin = int(settings['csv_thin'])
    np.savez_compressed(out_dir / 'Posterior_Samples_Thin.npz', samples=flat[::thin], log_prob=flat_logp[::thin], parameters=np.asarray(params), thinning=thin)
    rng = np.random.RandomState(seed + 900)
    npp = min(int(settings['n_pp']), len(flat))
    pp_idx = rng.choice(len(flat), size=npp, replace=False)
    pp_par = flat[pp_idx]
    pp_mu = context.predict(pp_par[:, :6], use_bias=info['use_bias'])
    pp_draws = np.empty_like(pp_mu)
    for i in range(npp):
        eps = rng.normal(size=NOBS)
        total_eig = info['eig'].copy()
        if info['use_rep']:
            total_eig = total_eig + pp_par[i, 6] ** 2
        pp_draws[i] = pp_mu[i] + info['qmat'] @ (np.sqrt(total_eig) * eps)
    pred_lo = np.quantile(pp_draws, 0.025, axis=0)
    pred_md = np.quantile(pp_draws, 0.5, axis=0)
    pred_hi = np.quantile(pp_draws, 0.975, axis=0)
    covered = (y_obs >= pred_lo) & (y_obs <= pred_hi)
    map_mu = context.predict(map_par[:6], use_bias=info['use_bias'])[0]
    med_mu = context.predict(med_par[:6], use_bias=info['use_bias'])[0]
    pred_data = {'Borehole': Y_COLS, 'Observed_Head_m': y_obs, 'MAP_Mean_Head_m': map_mu, 'MedianParam_Mean_Head_m': med_mu, 'PP_P2.5_m': pred_lo, 'PP_Median_m': pred_md, 'PP_P97.5_m': pred_hi, 'PP95_Width_m': pred_hi - pred_lo, 'Observed_Inside_PP95': covered, 'MAP_Residual_ObsMinusPred_m': y_obs - map_mu}
    if raw_heads is not None:
        raw_covered = (raw_heads >= pred_lo) & (raw_heads <= pred_hi)
        pred_data['Raw_Abaqus_Head_m'] = raw_heads
        pred_data['Raw_Abaqus_Inside_PP95'] = raw_covered
    else:
        raw_covered = None
    pd.DataFrame(pred_data).to_csv(out_dir / 'Posterior_Predictive.csv', index=False)
    base_cov = info['base_cov']
    map_total_cov = base_cov.copy()
    if info['use_rep']:
        map_total_cov = map_total_cov + map_par[6] ** 2 * np.eye(NOBS)
    diagnostics = {'case_name': case['case_name'], 'mode': mode, 'seed': int(seed), 'n_walkers': int(settings['n_walkers']), 'n_steps': int(settings['n_steps']), 'burn_in': burn, 'postburn_samples': int(len(flat)), 'acceptance_fraction': float(np.mean(sampler.acceptance_fraction)), 'max_split_Rhat': float(np.nanmax(rhat)), 'max_rank_normalized_split_Rhat': float(np.nanmax(rank_rhat)), 'min_ESS': float(np.nanmin(ess)), 'convergence_pass': bool(np.nanmax(rhat) < 1.01 and np.nanmax(rank_rhat) < 1.01 and (np.nanmin(ess) > 1000)) if mode in ('full', 'extended') else None, 'covariance_min_eigenvalue': float(np.min(info['eig'])), 'covariance_condition_number': float(np.max(info['eig']) / np.min(info['eig'])), 'mean_case_covariance_std_m': float(np.mean(np.sqrt(np.diag(base_cov)))), 'MAP_RMSE_to_observation_m': float(np.sqrt(mean_squared_error(y_obs, map_mu))), 'MAP_MAE_to_observation_m': float(mean_absolute_error(y_obs, map_mu)), 'MedianParam_RMSE_to_observation_m': float(np.sqrt(mean_squared_error(y_obs, med_mu))), 'PP95_coverage_observation_percent': float(100.0 * np.mean(covered)), 'PP95_covered_observation_count': int(np.sum(covered)), 'PP95_mean_width_m': float(np.mean(pred_hi - pred_lo)), 'MAP_Mahalanobis2_observation': mahalanobis2(y_obs - map_mu, map_total_cov), 'MAP_log_posterior': float(flat_logp[map_index]), 'use_sigma_rf': bool(case['use_sigma_rf']), 'use_sigma_surr': bool(case['use_sigma_surr']), 'use_surrogate_bias': bool(case['use_surrogate_bias']), 'use_sigma_rep': bool(case['use_sigma_rep']), **prescreen}
    if raw_heads is not None:
        diagnostics.update({'MAP_RMSE_to_raw_Abaqus_m': float(np.sqrt(mean_squared_error(raw_heads, map_mu))), 'MAP_MAE_to_raw_Abaqus_m': float(mean_absolute_error(raw_heads, map_mu)), 'PP95_coverage_raw_Abaqus_percent': float(100.0 * np.mean(raw_covered)), 'PP95_covered_raw_Abaqus_count': int(np.sum(raw_covered))})
    if truth_theta is not None:
        mu_true = context.predict(truth_theta, use_bias=info['use_bias'])[0]
        diagnostics.update({'Forward_RMSE_TrueTheta_vs_RawAbaqus_m': float(np.sqrt(mean_squared_error(raw_heads, mu_true))), 'Forward_MAE_TrueTheta_vs_RawAbaqus_m': float(mean_absolute_error(raw_heads, mu_true)), 'TrueTheta_Mahalanobis2_observation': mahalanobis2(y_obs - mu_true, base_cov), 'TrueTheta_Mahalanobis_PASS_95': bool(mahalanobis2(y_obs - mu_true, base_cov) <= CHI2_23_95)})
    with (out_dir / 'MCMC_Diagnostics.json').open('w', encoding='utf-8') as f:
        json.dump(diagnostics, f, indent=2, ensure_ascii=False)
    del sampler, chain, post_chain, post_logp, flat, flat_logp, pp_draws
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return diagnostics

def ensure_directory_tree() -> None:
    paths = [RESULT_ROOT / '00_config', RESULT_ROOT / '01_reproduction_check' / 'Synthetic_A3', RESULT_ROOT / '01_reproduction_check' / 'Field_A4', RESULT_ROOT / '02_synthetic', RESULT_ROOT / '03_field', RESULT_ROOT / '04_summary' / 'tables',  RESULT_ROOT / '05_logs']
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
