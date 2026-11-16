"""Run Step-13 likelihood ablation for the three independent synthetic truths."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from importlib import import_module
common = import_module('37A_Likelihood_Ablation_Common')

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', required=True, choices=['A0_obs_only', 'A1_obs_rf', 'A2_obs_rf_surr', 'A3_obs_rf_surr_bias'])
    parser.add_argument('--mode', choices=['smoke', 'full', 'extended'], default='full')
    parser.add_argument('--reproduction-check', action='store_true')
    parser.add_argument('--truth-id', type=int, choices=[1, 2, 3], default=None)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    common.ensure_directory_tree()
    case = common.get_case(args.case)
    if case['use_sigma_rep']:
        raise RuntimeError('Synthetic validation must not use sigma_rep.')
    if args.reproduction_check:
        case_root = common.RESULT_ROOT / '01_reproduction_check' / 'Synthetic_A3' / args.mode
    else:
        case_root = common.RESULT_ROOT / '02_synthetic' / args.case / args.mode
    case_root.mkdir(parents=True, exist_ok=True)
    common.write_input_manifest(case_root, [common.TRUTH_PARAM_PATH, common.RAW_HEADS_PATH, common.SYN_OBS_PATH])
    truth_df = pd.read_csv(common.TRUTH_PARAM_PATH).sort_values('Truth_ID')
    raw_df = pd.read_csv(common.RAW_HEADS_PATH).sort_values('Truth_ID')
    obs_df = pd.read_csv(common.SYN_OBS_PATH).sort_values('Truth_ID')
    truth_ids = [args.truth_id] if args.truth_id is not None else [1, 2, 3]
    context = common.ModelContext()
    parameter_frames = []
    predictive_frames = []
    truth_metrics = []
    print('Case:', args.case)
    print('Mode:', args.mode)
    print('Device:', context.device)
    print('Output:', case_root)
    for truth_id in truth_ids:
        theta_true = truth_df.loc[truth_df['Truth_ID'] == truth_id, common.X_COLS].iloc[0].to_numpy(float)
        y_raw = raw_df.loc[raw_df['Truth_ID'] == truth_id, common.Y_COLS].iloc[0].to_numpy(float)
        y_obs = obs_df.loc[obs_df['Truth_ID'] == truth_id, common.Y_COLS].iloc[0].to_numpy(float)
        out_dir = case_root / ('Truth_%d' % truth_id)
        print('\nRunning Truth', truth_id, '->', out_dir)
        diagnostics = common.run_single_dataset(context=context, y_obs=y_obs, case=case, out_dir=out_dir, mode=args.mode, seed=42 + 1000 * truth_id, raw_heads=y_raw, truth_theta=theta_true)
        param = pd.read_csv(out_dir / 'Posterior_Parameter_Summary.csv')
        true_map = dict(zip(common.X_COLS, theta_true))
        param['Truth_ID'] = truth_id
        param['Truth_Value'] = param['Parameter'].map(true_map)
        param['Truth_Inside_HDI95'] = (param['Truth_Value'] >= param['HDI95_Low']) & (param['Truth_Value'] <= param['HDI95_High'])
        widths = np.array([common.WIDTH[common.X_COLS.index(p)] for p in param['Parameter']])
        prior_sd = np.array([common.PRIOR_SD[common.X_COLS.index(p)] for p in param['Parameter']])
        param['HDI95_Width'] = param['HDI95_High'] - param['HDI95_Low']
        param['HDI95_Width_Contraction'] = 1.0 - param['HDI95_Width'] / widths
        param['Posterior_SD_Ratio_to_Prior_SD'] = param['SD'] / prior_sd
        param['Median_Absolute_Error'] = np.abs(param['Median'] - param['Truth_Value'])
        parameter_frames.append(param)
        pred = pd.read_csv(out_dir / 'Posterior_Predictive.csv')
        pred['Truth_ID'] = truth_id
        predictive_frames.append(pred)
        diagnostics['Truth_ID'] = truth_id
        diagnostics['Parameter_HDI95_Hits'] = int(param['Truth_Inside_HDI95'].sum())
        diagnostics['Parameter_HDI95_Total'] = int(len(param))
        diagnostics['Mean_HDI95_Width_Contraction'] = float(param['HDI95_Width_Contraction'].mean())
        diagnostics['Mean_Posterior_SD_Ratio_to_Prior_SD'] = float(param['Posterior_SD_Ratio_to_Prior_SD'].mean())
        truth_metrics.append(diagnostics)
    parameter_all = pd.concat(parameter_frames, ignore_index=True)
    predictive_all = pd.concat(predictive_frames, ignore_index=True)
    truth_metrics_df = pd.DataFrame(truth_metrics)
    parameter_all.to_csv(case_root / 'Synthetic_Parameter_Results_All.csv', index=False)
    predictive_all.to_csv(case_root / 'Synthetic_Predictive_Results_All.csv', index=False)
    truth_metrics_df.to_csv(case_root / 'Synthetic_Truth_Level_Metrics.csv', index=False)
    overall = {'case_name': args.case, 'mode': args.mode, 'truth_count': int(len(truth_ids)), 'parameter_95_HDI_hits': int(parameter_all['Truth_Inside_HDI95'].sum()), 'parameter_95_HDI_total': int(len(parameter_all)), 'parameter_95_HDI_coverage_percent': float(100.0 * parameter_all['Truth_Inside_HDI95'].mean()), 'synthetic_observation_PP95_coverage_percent': float(100.0 * predictive_all['Observed_Inside_PP95'].mean()), 'raw_Abaqus_PP95_coverage_percent': float(100.0 * predictive_all['Raw_Abaqus_Inside_PP95'].mean()), 'mean_MAP_RMSE_to_observation_m': float(truth_metrics_df['MAP_RMSE_to_observation_m'].mean()), 'mean_MAP_RMSE_to_raw_Abaqus_m': float(truth_metrics_df['MAP_RMSE_to_raw_Abaqus_m'].mean()), 'mean_PP95_width_m': float(truth_metrics_df['PP95_mean_width_m'].mean()), 'mean_HDI95_width_contraction': float(parameter_all['HDI95_Width_Contraction'].mean()), 'mean_posterior_SD_ratio_to_prior_SD': float(parameter_all['Posterior_SD_Ratio_to_Prior_SD'].mean()), 'mean_acceptance_fraction': float(truth_metrics_df['acceptance_fraction'].mean()), 'max_split_Rhat': float(truth_metrics_df['max_split_Rhat'].max()), 'max_rank_normalized_split_Rhat': float(truth_metrics_df['max_rank_normalized_split_Rhat'].max()), 'min_ESS': float(truth_metrics_df['min_ESS'].min()), 'all_convergence_pass': bool(truth_metrics_df['convergence_pass'].fillna(False).all()) if args.mode in ('full', 'extended') else None}
    with (case_root / 'Synthetic_Ablation_Overall.json').open('w', encoding='utf-8') as f:
        json.dump(overall, f, indent=2, ensure_ascii=False)
    print('\nCompleted:', overall)
if __name__ == '__main__':
    main()
