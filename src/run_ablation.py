"""Ablation studies under leave-one-day-out validation.
(1) feature ablation, (2) CO2-threshold sensitivity. Uses the three best models from the main experiment."""
import sys, time
import pandas as pd
from common import load_config, load_data, run_protocol

cfg = load_config()
main = pd.read_csv('results/main_summary_leave_one_day_out.csv')
top = main.sort_values('f1', ascending=False).model.head(3).tolist() if cfg['ablation']['models'] == 'auto_top3' else cfg['ablation']['models']
import json, collections
sel = json.load(open('results/selected_params_leave_one_day_out.json'))
def modal(m):
    c = collections.Counter(json.dumps({k: v for k, v in d.items() if k != 'test_day'}, sort_keys=True) for d in sel[f'leave_one_day_out/{m}'])
    return json.loads(c.most_common(1)[0][0])
FIXED = {m: modal(m) for m in top}
json.dump(FIXED, open('results/ablation_fixed_params.json', 'w'), indent=1)
print('ablation models:', FIXED, flush=True)
part = sys.argv[1] if len(sys.argv) > 1 else 'all'
rows = []
if part in ('all', 'features'):
    df = load_data(cfg)
    for fs_name, feats in cfg['ablation']['feature_sets'].items():
        for m in top:
            t = time.time()
            pooled, pdy, _, _ = run_protocol(cfg, df, m, 'leave_one_day_out', feats, fixed=FIXED[m])
            rows.append(dict(study='features', setting=fs_name, model=m, **pooled))
            print(f'{fs_name:22s} {m:6s} F1={pooled["f1"]:.3f} ({time.time()-t:.0f}s)', flush=True)
    pd.DataFrame([r for r in rows if r['study'] == 'features']).to_csv('results/ablation_features.csv', index=False)
if part in ('all', 'threshold'):
    for thr in cfg['ablation']['thresholds_ppm']:
        df = load_data(cfg, threshold=thr)
        for m in top:
            pooled, pdy, _, _ = run_protocol(cfg, df, m, 'leave_one_day_out', cfg['data']['base_features'], fixed=FIXED[m])
            rows.append(dict(study='threshold', setting=thr, model=m, prevalence=df.poor.mean(), **pooled))
            print(f'threshold {thr} {m:6s} F1={pooled["f1"]:.3f}', flush=True)
    pd.DataFrame([r for r in rows if r['study'] == 'threshold']).to_csv('results/ablation_threshold.csv', index=False)
