from pathlib import Path
import sys, importlib, json
import numpy as np
import pandas as pd
import torch
root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / "analysis"))
m = importlib.import_module("37A_Likelihood_Ablation_Common")
ctx = m.ModelContext()
d = pd.read_csv(m.PRODUCTION_DIR / "Production_DNN_All50_Fit_Predictions.csv")
expected = d[[c + "_ProductionDNN" for c in m.Y_COLS]].to_numpy(float)
actual = ctx.predict(d[m.X_COLS].to_numpy(float), use_bias=False)
assert np.max(np.abs(expected - actual)) < 0.002
raw = pd.read_csv(root / "analysis/Final_ML_Dataset_CorrectedRF_FinalGW.csv")
means = raw.groupby("P_index")[m.Y_COLS].transform("mean")
res = raw[m.Y_COLS].to_numpy(float) - means.to_numpy(float)
pooled = res.T @ res / (len(raw) - raw.P_index.nunique())
assert np.allclose(pooled, ctx.sigma_rf, rtol=1e-6, atol=1e-7)
for name in m.load_cases():
    case = m.get_case(name)
    logp, info = m.make_log_prob(ctx, m.Y_FIELD, case)
    mid = (m.LOWER + m.UPPER) / 2
    point = np.r_[mid, 10.] if case["use_sigma_rep"] else mid
    assert np.isfinite(logp(point)).all()
    outside = point.copy(); outside[0] = m.LOWER[0] - 1
    assert np.isneginf(logp(outside)).all()
report = {"model_predictions_match_reference": True, "pooled_RF_covariance_reproduced": True, "ablation_likelihood_cases_validated": 5, "max_prediction_difference_m": float(np.max(np.abs(expected-actual)))}
print(json.dumps(report, indent=2))

# Verify the frozen independent-test model against the manuscript metric.
a = root / "analysis/Step7_GroupMean_DNN_CorrectedRF_FinalGW"
import joblib
sx = joblib.load(a / "scaler_X_GroupMean_CorrectedRF.pkl")
sy = joblib.load(a / "scaler_Y_GroupMean_CorrectedRF.pkl")
model = m.GroupMeanDNN().to(ctx.device)
model.load_state_dict(m.torch_load(a / "GroupMean_DNN_CorrectedRF_Best.pth", ctx.device))
model.eval()
group = pd.read_csv(root / "analysis/Group_Mean_Dataset_50_CorrectedRF_FinalGW.csv")
split = pd.read_csv(root / "analysis/Group_Split_35_7_8_CorrectedRF.csv")
test = group[group.P_index.isin(split.loc[split.Split == "Test", "P_index"])]
with torch.no_grad():
    yhat = model(torch.tensor((test[m.X_COLS].to_numpy()-sx.mean_)/sx.scale_, dtype=torch.float32, device=ctx.device)).cpu().numpy()*sy.scale_+sy.mean_
rmse = float(np.sqrt(np.mean((yhat-test[m.Y_COLS].to_numpy())**2)))
assert abs(rmse-1.7921)<0.001
print("Independent-test RMSE reproduced: %.6f m" % rmse)
# Check each likelihood against a separate dense covariance evaluation.
from scipy.stats import multivariate_normal
for name in m.load_cases():
    case = m.get_case(name)
    logp, info = m.make_log_prob(ctx, m.Y_FIELD, case)
    theta = (m.LOWER+m.UPPER)/2
    covariance = ctx.build_covariance(case)
    prior = m.LOG_PRIOR_PHYS
    par = theta
    if case["use_sigma_rep"]:
        par = np.r_[theta,10.]
        covariance = covariance+100.*np.eye(23)
        scale = case["sigma_rep_scale_m"]
        prior += 0.5*np.log(2/np.pi)-np.log(scale)-0.5*(10/scale)**2
    dense = multivariate_normal.logpdf(m.Y_FIELD, mean=ctx.predict(theta,case["use_surrogate_bias"])[0], cov=covariance)+prior
    assert np.allclose(logp(par)[0], dense, rtol=1e-8, atol=1e-8)
print("Five vectorized log posteriors agree with dense Gaussian calculations.")
