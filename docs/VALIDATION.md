# Release validation

Checks performed on 16 November 2026 on Windows with Python 3.12.7, CPU PyTorch 2.15.0 and emcee 3.1.6:

- All Python modules pass syntax parsing.
- Dataset dimensions, group sizes, finite hydraulic heads and covariance positive definiteness pass.
- Recomputed pooled RF covariance agrees with the archived covariance matrix.
- Production predictions match archived predictions, maximum absolute difference 0.00005846 m.
- Frozen independent-test network reproduces RMSE 1.792099 m.
- All five ablation log posteriors agree with independent dense Gaussian evaluations, including bounds and the field HalfNormal prior.
- Field and all three synthetic inversions execute successfully for 300 steps, 64 walkers and 100 burn-in steps.
- Independent-test training executes for eight epochs; five-fold OOF estimation and production retraining execute using that short-run selected epoch count in a separate scratch copy.
- Field A4 and synthetic A3 (truth 1) ablation smoke runs complete and export numerical outputs.

These execution tests do not establish posterior convergence. Published models and reference results were not overwritten by test outputs. MATLAB/Abaqus simulations and fresh full-length production chains were not rerun in this audit. The pinned scikit-learn 1.6.1 dependency matches the version recorded in the pretrained scaler artifacts.
