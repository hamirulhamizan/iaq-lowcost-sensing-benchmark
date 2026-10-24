"""Study 2 ablations (home offices, leave-sites-out, composite label).
(1) Feature ablation for the three best models, hyperparameters fixed at their most frequently selected values.
(2) PM2.5 limit sensitivity: 15 (WHO 2021 guideline), 25 (WHO 2021 interim target 4) and 35 ug/m3 (MAAQS 2020)."""
import json, collections, time, warnings
warnings.filterwarnings('ignore')
import pandas as pd
import run_multisite as R

cfg, M = R.cfg, R.M
A = M['ablation']
summ = pd.read_csv('results/multisite/summary_composite_random_leave_sites_out_chronological.csv')
top = summ[summ.protocol == 'leave_sites_out'].sort_values('f1', ascending=False).model.head(3).tolist()
params = json.load(open('results/multisite/params_composite_random_leave_sites_out_chronological.json'))
def modal(m):
    c = collections.Counter(json.dumps({k: v for k, v in d.items() if k != 'fold'}, sort_keys=True) for d in params[f'leave_sites_out/{m}'])
    return json.loads(c.most_common(1)[0][0])
FIXED = {m: modal(m) for m in top}
json.dump(FIXED, open('results/multisite/ablation_fixed_params.json', 'w'), indent=1)
print('models:', FIXED, flush=True)

w = R.load('composite'); rows = []
for name, feats in A['feature_sets'].items():
    for m in top:
        t = time.time(); pooled, _, _, _ = R.run(w, m, 'leave_sites_out', feats, fixed=FIXED[m])
        rows.append(dict(study='features', setting=name, model=m, **pooled))
        print(f"{name:26s} {m:4s} F1={pooled['f1']:.3f} MCC={pooled['mcc']:.3f} ({time.time()-t:.0f}s)", flush=True)
pd.DataFrame(rows).to_csv('results/multisite/ablation_features.csv', index=False)

rows = []
for thr in A['pm25_thresholds']:
    M['thresholds']['PM25_ugm3'] = thr
    wt = R.load('composite')
    for m in top:
        pooled, _, _, _ = R.run(wt, m, 'leave_sites_out', M['inputs'], fixed=FIXED[m])
        rows.append(dict(study='pm25_threshold', setting=thr, model=m, prevalence=wt.poor.mean(), pm25_share=wt['exceed_PM2.5'].mean(), **pooled))
        print(f"PM2.5 > {thr} {m:4s} prevalence={wt.poor.mean():.3f} F1={pooled['f1']:.3f} MCC={pooled['mcc']:.3f}", flush=True)
pd.DataFrame(rows).to_csv('results/multisite/ablation_pm25_threshold.csv', index=False)
