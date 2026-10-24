"""Study 2 ablations (leave-sites-out, composite label) for the three best models, re-tuned inside every fold.
(1) feature ablation; (2) PM2.5 limit: 15 (WHO 2021 guideline), 25 (WHO interim target 4), 35 ug/m3 (MAAQS 2020)."""
import pandas as pd
from runner import flat
import run_multisite as R

cfg, M = R.cfg, R.M
s = pd.read_csv('results/study2/summary_composite.csv'); s = s[s.protocol == 'leave_sites_out']
top = s.sort_values('f1', ascending=False).model.head(3).tolist(); print('models:', top, flush=True)
w = R.load('composite'); rows = []
for name, feats in M['ablation']['feature_sets'].items():
    for m in top:
        r = R.run(w, m, 'leave_sites_out', feats, f'ablation/{name}/{m}')
        rows.append(dict(study='features', setting=name, model=m, **{k: flat(v) for k, v in r['pooled'].items()}))
pd.DataFrame(rows).to_csv('results/study2/ablation_features.csv', index=False)
rows = []
for thr in M['ablation']['pm25_thresholds']:
    wt = R.load('composite', pm25=thr)
    for m in top:
        r = R.run(wt, m, 'leave_sites_out', M['inputs'], f'pm25/{thr}/{m}')
        rows.append(dict(study='pm25_threshold', setting=thr, model=m, prevalence=wt.poor.mean(), pm25_share=wt['exceed_PM2.5'].mean(), **{k: flat(v) for k, v in r['pooled'].items()}))
pd.DataFrame(rows).to_csv('results/study2/ablation_pm25_threshold.csv', index=False)
# label sensitivity: 24-h PM2.5 averaging and composite without TVOC
for lab in ['composite_pm24h', 'composite_noTVOC']:
    wl = R.load(lab)
    for m in top:
        r = R.run(wl, m, 'leave_sites_out', M['inputs'], f'{lab}/leave_sites_out/{m}')
rows = []
for lab in ['composite', 'CO2', 'PM2.5', 'composite_pm24h', 'composite_noTVOC']:
    wl = R.load(lab)
    for m in top:
        r = R.load_run('study2', f'{lab}/leave_sites_out/{m}')
        if r is not None:
            rows.append(dict(label=lab, model=m, prevalence=wl.poor.mean(), **{k: flat(v) for k, v in r['pooled'].items()}))
pd.DataFrame(rows).to_csv('results/study2/label_sensitivity.csv', index=False)
