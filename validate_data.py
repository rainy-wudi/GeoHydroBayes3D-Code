from pathlib import Path
import numpy as np
import pandas as pd
r = Path(__file__).resolve().parent / "analysis"
d = pd.read_csv(r / "Final_ML_Dataset_CorrectedRF_FinalGW.csv")
assert len(d) == 1000
assert d.P_index.nunique() == 50
assert (d.groupby("P_index").size() == 20).all()
y = [c for c in d.columns if c.startswith("zk")]
assert len(y) == 23
assert np.isfinite(d[y].to_numpy(float)).all()
paths = [r / "Sigma_RF_Pooled_23x23_CorrectedRF_FinalGW.csv", r / "Step8_5Fold_OOF_SurrogateError_CorrectedRF_FinalGW" / "Sigma_surr_LedoitWolf_23x23_CorrectedRF.csv"]
for path in paths:
    c = pd.read_csv(path, index_col=0).loc[y, y].to_numpy(float)
    assert c.shape == (23, 23)
    assert np.isfinite(c).all()
    assert np.allclose(c, c.T)
    np.linalg.cholesky(c)
print("PASS: 1000 rows, 50 groups of 20, 23 heads; RF/surrogate covariances are positive definite.")
