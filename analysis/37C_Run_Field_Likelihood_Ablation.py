"""Run Step-13 likelihood ablation for the 23 field observations."""
from __future__ import annotations
import argparse
import json
from importlib import import_module
common = import_module('37A_Likelihood_Ablation_Common')

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', required=True, choices=['A0_obs_only', 'A1_obs_rf', 'A2_obs_rf_surr', 'A3_obs_rf_surr_bias', 'A4_full_sigma_rep'])
    parser.add_argument('--mode', choices=['smoke', 'full', 'extended'], default='full')
    parser.add_argument('--reproduction-check', action='store_true')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    common.ensure_directory_tree()
    case = common.get_case(args.case)
    if not case['use_sigma_rep']:
        case['optimized_init'] = True
    if args.reproduction_check:
        if args.case != 'A4_full_sigma_rep':
            raise RuntimeError('Field reproduction check is defined for A4 only.')
        out_dir = common.RESULT_ROOT / '01_reproduction_check' / 'Field_A4' / args.mode
    else:
        out_dir = common.RESULT_ROOT / '03_field' / args.case / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)
    common.write_input_manifest(out_dir)
    context = common.ModelContext()
    print('Case:', args.case)
    print('Mode:', args.mode)
    print('Device:', context.device)
    print('Output:', out_dir)
    diagnostics = common.run_single_dataset(context=context, y_obs=common.Y_FIELD, case=case, out_dir=out_dir, mode=args.mode, seed=42, raw_heads=None, truth_theta=None)
    with (out_dir / 'Field_Ablation_Overall.json').open('w', encoding='utf-8') as f:
        json.dump(diagnostics, f, indent=2, ensure_ascii=False)
    print('\nCompleted:', diagnostics)
if __name__ == '__main__':
    main()
