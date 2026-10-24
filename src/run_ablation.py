"""Study 1 ablations under leave-one-day-out validation, for the three best models.
Hyperparameters and the decision threshold are re-tuned inside every fold, exactly as in the main experiment.
(1) feature ablation; (2) CO2-threshold sensitivity (800 / 1000 / 1200 ppm)."""
import pandas as pd
from common import load_config, load_data
from runner import flat
import run_main as R

cfg = R.cfg
s = pd.read_csv('results/study1/summary.csv'); s = s[s.protocol == 'leave_one_day_out']
top = s.sort_values('f1', ascending=False).model.head(3).tolist()
print('ablation models:', top, flush=True)
df = load_data(cfg); rows = []
for name, feats in cfg['ablation']['feature_sets'].items():
    for m in top:
        r = R.run(df, m, 'leave_one_day_out', feats, f'ablation/{name}/{m}')
        rows.append(dict(study='features', setting=name, model=m, **{k: flat(v) for k, v in r['pooled'].items()}))
pd.DataFrame(rows).to_csv('results/study1/ablation_features.csv', index=False)
rows = []
for thr in cfg['ablation']['thresholds_ppm']:
    d = load_data(cfg, threshold=thr)
    for m in top:
        r = R.run(d, m, 'leave_one_day_out', cfg['data']['base_features'], f'threshold/{thr}/{m}')
        rows.append(dict(study='threshold', setting=thr, model=m, prevalence=d.poor.mean(), **{k: flat(v) for k, v in r['pooled'].items()}))
pd.DataFrame(rows).to_csv('results/study1/ablation_threshold.csv', index=False)
