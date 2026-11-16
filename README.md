# GeoHydroBayes3D

Code accompanying **Bayesian Inversion of Geostatistical Hyperparameters for Three-Dimensional Seepage with Limited Hydraulic-Head Observations and Multiple Error Sources**.

Authors: Yucheng Wu, Chunhui Ma, Jie Yang, Ibrokhimov Khofiz and Lin Cheng.

Corresponding author: Dr. Chunhui Ma; shanximachunhui@foxmail.com; +86 18700917478.

Repository: https://github.com/rainy-wudi/GeoHydroBayes3D-Code

## Purpose

This package implements the manuscript's conditional group-mean DNN and Bayesian inversion with separate observation, random-field, surrogate and field representation errors. It contains the 50 parameter groups × 20 random-field realizations database, observations at 23 boreholes, the frozen 35/7/8 group split, independent-test and all-50 production networks, scalers, OOF bias/covariance estimates, three independent synthetic validation cases and likelihood-ablation scripts. Code and comments are in English. Numerical outputs are saved as CSV and JSON files.

The repository includes the engineering data, observations and trained models used in the study, released with the authors' permission.

## Installation

Python 3.12 is recommended (release checks used Python 3.12.7 on Windows, an Intel Core i7-14700KF and 32 GB RAM). A CPU is sufficient; CUDA is optional. This tested hardware is not a measured minimum requirement. Leave sufficient disk space for posterior CSV files: the unthinned field chain alone can approach 400 MB. MATLAB random-field generation requires substantially more memory than surrogate prediction.

Create a dedicated environment and install the pinned numerical dependencies:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate
python -m pip install -r requirements.txt
```

The pretrained scikit-learn scalers were saved with scikit-learn 1.6.1. Load only trusted model artifacts. MATLAB with Statistics and Machine Learning Toolbox and a licensed Abaqus installation are needed only to regenerate finite-element results. The archived solver log identifies Abaqus/Standard 2024; MATLAB version information was not retained. Neither proprietary solver is distributed.

## Quick checks

From the repository root:

```bash
python validate_data.py
python validate_model.py
```

Expected: 1000 samples, 50 groups of 20, 23 heads, positive-definite RF/surrogate covariance matrices, production predictions matching stored predictions within 0.002 m, recomputed pooled RF covariance matching the supplied matrix, independent-test RMSE approximately 1.792099 m, and five vectorized log posteriors matching independent dense Gaussian evaluations. These checks do not run long MCMC chains.

A short field MCMC example is:

```bash
python analysis/37C_Run_Field_Likelihood_Ablation.py --case A4_full_sigma_rep --mode smoke
```

A short independent synthetic example is:

```bash
python analysis/37B_Run_Synthetic_Likelihood_Ablation.py --case A3_obs_rf_surr_bias --mode smoke --truth-id 1
```

Smoke mode uses 64 walkers, 300 steps and 100 burn-in steps. Successful execution verifies software operation; these short chains are not expected to meet the publication convergence targets.

## Reproduce numerical analyses

The original production settings are retained unless an environment override is explicitly supplied.

1. `analysis/30_Train_GroupMean_DNN_CorrectedRF_FinalGW.py`: independent-test DNN training with the frozen 35/7/8 group split. Exports weights, scalers, training history, predictions and metrics.
2. `analysis/31_5Fold_OOF_SurrogateError_CorrectedRF_FinalGW.py`: five-fold group-level OOF predictions; residual bias defined as numerical group mean minus DNN; raw and Ledoit-Wolf covariance; final all-50-group production model. Run after step 1 for the selected training epoch count. The published all-50 model uses 253 epochs.
3. `analysis/36A_Final_Field_MCMC_Convergence_Strengthening.py`: field hierarchical inversion of six physical parameters and sigma_rep, using 64 walkers, 50000 steps and 10000 burn-in steps. Sigma_rep has a HalfNormal(10 m) prior; observation standard deviation is 0.5 m.
4. `analysis/36B_Final_Independent_Synthetic_MCMC_Strengthening.py`: six-dimensional inversion for all three independent synthetic truths, without sigma_rep.
5. `analysis/37B_Run_Synthetic_Likelihood_Ablation.py` and `analysis/37C_Run_Field_Likelihood_Ablation.py`: controlled error-component ablations. Use `--help` for case names and `--mode full` or `--mode extended` for long runs.
6. `analysis/37D_Summarize_Likelihood_Ablation.py`: collect full/extended ablation results into CSV summary tables.

Steps 3–6 can use the included models and covariances without retraining. Retraining changes model artifacts and subsequent posterior results; use a separate working copy when comparing against the included reference results. GPU/library differences can affect numerical results. Evaluate the saved diagnostics rather than assuming convergence from the step count.

Scripts locate inputs relative to their own directory. `GHB_WORK_DIR` optionally redirects the analysis root to a copied directory. For short tests of scripts 30/36A/36B, optional integer overrides are `GHB_MAX_EPOCHS`, `GHB_N_STEPS`, `GHB_BURN_IN`, `GHB_N_PRESCREEN`, `GHB_TOP_POOL` and `GHB_N_PP`. Keep burn-in smaller than the number of steps and the initial candidate pool at least as large as the walker count. These overrides change computation length; they are not the settings used to obtain the paper's reported results.

## Inputs and outputs

| Location | Contents |
|---|---|
| `analysis/Final_ML_Dataset_CorrectedRF_FinalGW.csv` | 1000 numerical realizations, group/realization identifiers, six inputs and 23 hydraulic-head outputs |
| `analysis/Group_Mean_Dataset_50_CorrectedRF_FinalGW.csv` | Conditional means used as DNN targets |
| `analysis/Group_Split_35_7_8_CorrectedRF.csv` | Frozen parameter-group assignments; no realization-level train/test leakage |
| `analysis/Sigma_RF_Pooled_23x23_CorrectedRF_FinalGW.csv` | Pooled within-group covariance, 950 residual degrees of freedom |
| `analysis/Step7_GroupMean_DNN_CorrectedRF_FinalGW/` | Independent-test model, scalers and numerical training/test results |
| `analysis/Step8_5Fold_OOF_SurrogateError_CorrectedRF_FinalGW/` | OOF residuals, bias, covariances and production model |
| `analysis/synthetic/` | Three independent truth parameters and extracted/noisy synthetic heads |
| `analysis/36A_Final_Field_MCMC_Convergence_Strengthening.py` | Field observation values and production prior/settings |
| `reference_results/` | Compact numerical reference results from original runs; not outputs of release smoke tests |
| `matlab/inputs/` | Site model template and layer classification points |

Input order: log10_K_Lower, log10_K_Middle, log10_K_Upper, COV, dh, dv. The first three are base-10 logarithms of hydraulic conductivity in m/s; COV is dimensionless; horizontal/vertical correlation lengths dh/dv are in metres. Outputs zk101 through zk410 follow the explicit 23-borehole order in the source. Hydraulic heads, residual biases and standard deviations are in metres; covariance entries are in square metres.

Field outputs are written under `analysis/Step12A_Final_Field_MCMC_Strengthened/`; synthetic validation outputs under `analysis/synthetic/Step11_Independent_Validation/Step12B_Final_Independent_MCMC_Strengthened/`. They include parameter HDIs, MAP/mean/median estimates, samples, Rhat/ESS diagnostics, and 95% posterior predictive bounds and coverage. Ablation outputs are under `analysis/Step13_Likelihood_Ablation_CorrectedRF_FinalGW/`.

## Finite-element data regeneration

In MATLAB, run `matlab/a_27B_Regenerate_Corrected_RF_FinalGW_1000.m`. Its default `MODE = 'SMOKE'` prepares a limited test; change to `ALL` after verifying the test to prepare all 1000 inputs. The script uses the fixed 50 parameter groups, corrected V2 generator, 20×20×25 reference grid, 100 material bins and fixed final groundwater boundary. Required helper functions and template/layer inputs are included. Job submission scripts are generated by MATLAB. Execute them with your separately licensed Abaqus installation.

After ODBs are generated, run `abaqus python analysis/28_Extract_Corrected_RF_FinalGW_1000_Heads.py`. Set `GHB_ODB_DIR` if your ODB directory differs from the default generated INP directory. Extraction uses H = z + POR/9810 and writes head/mapping/status tables under `analysis/abaqus_extraction/`. The included final ML dataset already supplies the downstream numerical responses, so Abaqus is unnecessary for DNN and inversion reproduction.

Large ODB databases and the full field posterior CSV are omitted to keep the repository compact. The supplied model/data permit regeneration of posterior samples. MATLAB/Abaqus generation is provided but was not rerun during this release audit; exact reproduction of all solver outputs therefore remains unverified. The release audit checked the Python model/data calculations and shortened execution paths, not fresh 50000-step production chains.

## License and citation

MIT license; see `LICENSE`. Cite the manuscript title and authors above, and identify release v2.0.0 when describing this code. `CITATION.cff` provides software metadata. See `docs/CODE_AVAILABILITY.md` for the manuscript statement and `docs/VALIDATION.md` for the release checks.
