"""Aggregate Step-13 likelihood ablation results into numerical summary tables."""
from __future__ import annotations
import json
from importlib import import_module
from pathlib import Path
import numpy as np
import pandas as pd
common = import_module('37A_Likelihood_Ablation_Common')
CASE_ORDER = ['A0_obs_only', 'A1_obs_rf', 'A2_obs_rf_surr', 'A3_obs_rf_surr_bias', 'A4_full_sigma_rep']

def preferred_result_root(base: Path) -> Path:
    extended = base / 'extended'
    if (extended / 'Field_Ablation_Overall.json').exists() or (extended / 'Synthetic_Ablation_Overall.json').exists():
        return extended
    return base / 'full'

def read_json(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)

def collect_synthetic() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall_rows = []
    parameter_frames = []
    predictive_frames = []
    for case in CASE_ORDER[:4]:
        root = preferred_result_root(common.RESULT_ROOT / '02_synthetic' / case)
        overall_path = root / 'Synthetic_Ablation_Overall.json'
        if not overall_path.exists():
            continue
        overall_rows.append(read_json(overall_path))
        p = pd.read_csv(root / 'Synthetic_Parameter_Results_All.csv')
        p['Case'] = case
        parameter_frames.append(p)
        q = pd.read_csv(root / 'Synthetic_Predictive_Results_All.csv')
        q['Case'] = case
        predictive_frames.append(q)
    overall = pd.DataFrame(overall_rows)
    params = pd.concat(parameter_frames, ignore_index=True) if parameter_frames else pd.DataFrame()
    pred = pd.concat(predictive_frames, ignore_index=True) if predictive_frames else pd.DataFrame()
    return (overall, params, pred)

def collect_field() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall_rows = []
    parameter_frames = []
    predictive_frames = []
    for case in CASE_ORDER:
        root = preferred_result_root(common.RESULT_ROOT / '03_field' / case)
        diag_path = root / 'Field_Ablation_Overall.json'
        if not diag_path.exists():
            continue
        d = read_json(diag_path)
        d['Case'] = case
        overall_rows.append(d)
        p = pd.read_csv(root / 'Posterior_Parameter_Summary.csv')
        p['Case'] = case
        parameter_frames.append(p)
        q = pd.read_csv(root / 'Posterior_Predictive.csv')
        q['Case'] = case
        predictive_frames.append(q)
    overall = pd.DataFrame(overall_rows)
    params = pd.concat(parameter_frames, ignore_index=True) if parameter_frames else pd.DataFrame()
    pred = pd.concat(predictive_frames, ignore_index=True) if predictive_frames else pd.DataFrame()
    return (overall, params, pred)

def case_rank(series: pd.Series) -> pd.Categorical:
    return pd.Categorical(series, categories=CASE_ORDER, ordered=True)

def main() -> None:
    tables = common.RESULT_ROOT / '04_summary' / 'tables'
    tables.mkdir(parents=True, exist_ok=True)
    syn_overall, syn_params, syn_pred = collect_synthetic()
    field_overall, field_params, field_pred = collect_field()
    syn_overall.to_csv(tables / 'Synthetic_Ablation_Overall.csv', index=False)
    syn_params.to_csv(tables / 'Synthetic_Ablation_Parameter_Detail.csv', index=False)
    syn_pred.to_csv(tables / 'Synthetic_Ablation_Predictive_Detail.csv', index=False)
    field_overall.to_csv(tables / 'Field_Ablation_Overall.csv', index=False)
    field_params.to_csv(tables / 'Field_Ablation_Parameter_Detail.csv', index=False)
    field_pred.to_csv(tables / 'Field_Ablation_Predictive_Detail.csv', index=False)
    conv_rows = []
    if not syn_overall.empty:
        for _, r in syn_overall.iterrows():
            conv_rows.append({'Dataset': 'Synthetic', 'Case': r['case_name'], 'Max_Split_Rhat': r['max_split_Rhat'], 'Max_RankNormalized_Split_Rhat': r['max_rank_normalized_split_Rhat'], 'Min_ESS': r['min_ESS'], 'Pass': r['all_convergence_pass']})
    if not field_overall.empty:
        for _, r in field_overall.iterrows():
            conv_rows.append({'Dataset': 'Field', 'Case': r['Case'], 'Max_Split_Rhat': r['max_split_Rhat'], 'Max_RankNormalized_Split_Rhat': r['max_rank_normalized_split_Rhat'], 'Min_ESS': r['min_ESS'], 'Pass': r['convergence_pass']})
    pd.DataFrame(conv_rows).to_csv(tables / 'Ablation_Convergence_Check.csv', index=False)
    status = {'synthetic_cases_found': syn_overall.get('case_name', pd.Series(dtype=str)).tolist(), 'field_cases_found': field_overall.get('Case', pd.Series(dtype=str)).tolist(), 'synthetic_complete': set(CASE_ORDER[:4]).issubset(set(syn_overall.get('case_name', []))), 'field_complete': set(CASE_ORDER).issubset(set(field_overall.get('Case', []))), 'all_convergence_pass': bool(pd.DataFrame(conv_rows)['Pass'].fillna(False).all()) if conv_rows else False}
    with (common.RESULT_ROOT / '04_summary' / 'Step13_Ablation_Status.json').open('w', encoding='utf-8') as f:
        json.dump(status, f, indent=2, ensure_ascii=False)
    print(status)
if __name__ == '__main__':
    main()
