"""Study 1 tables (written to tables/*.csv)."""
import json, collections, os
import numpy as np, pandas as pd
from common import load_config, load_data

cfg = load_config(); df = load_data(cfg)
os.makedirs('tables', exist_ok=True)
summ = pd.read_csv('results/study1/summary.csv'); per = pd.read_csv('results/study1/per_day.csv')
order = summ[summ.protocol == 'leave_one_day_out'].sort_values('f1', ascending=False).model.tolist()

d = df.groupby('day').agg(records=('poor', 'size'), poor=('poor', 'sum'), occupied=('Occupancy', 'sum'),
                          co2_max=('CO2', 'max'), rh_mean=('Humidity', 'mean'), t_mean=('Temperature', 'mean')).reset_index()
d.to_csv('tables/dataset_by_day.csv', index=False)

rows = []
for m in order:
    r = {'model': m}
    for p in ['random', 'chronological', 'leave_one_day_out']:
        s = summ[(summ.protocol == p) & (summ.model == m)].iloc[0]
        for k in ['accuracy', 'precision', 'recall', 'f1', 'balanced_accuracy', 'mcc', 'pr_auc', 'f1_default']:
            r[f'{p}_{k}'] = s[k]
    s = summ[(summ.protocol == 'leave_one_day_out') & (summ.model == m)].iloc[0]
    r['f1_ci_low'], r['f1_ci_high'] = s.f1_ci_low, s.f1_ci_high
    pdm = per[(per.protocol == 'leave_one_day_out') & (per.model == m) & (per.n_poor > 0)]
    r.update(f1_day_mean=pdm.f1.mean(), f1_day_sd=pdm.f1.std(ddof=1), f1_day_min=pdm.f1.min(), f1_day_max=pdm.f1.max(),
             days_f1_below_05=int((pdm.f1 < 0.5).sum()), days_with_poor=len(pdm))
    rows.append(r)
pd.DataFrame(rows).to_csv('tables/main_results.csv', index=False)

sel = json.load(open('results/study1/params.json')); out = []
for m in order:
    L = sel[f'leave_one_day_out/{m}']
    c = collections.Counter(json.dumps({k: v for k, v in x.items() if k not in ('fold', 'threshold')}, sort_keys=True) for x in L)
    best, n = c.most_common(1)[0]
    th = [x['threshold'] for x in L]
    out.append(dict(model=m, modal_params=best, folds=f'{n}/{len(L)}', threshold_median=np.median(th), threshold_min=min(th), threshold_max=max(th)))
pd.DataFrame(out).to_csv('tables/selected_params.csv', index=False); print(pd.DataFrame(out).to_string())

pr = pd.read_csv('results/study1/lodo_predictions.csv.gz')
occ = pr.Occupancy.values; poor = pr.poor.values; rows = []
for m in order:
    p = pr[f'{m}_pred'].values
    for lab, mask in [('occupied', (poor == 1) & (occ == 1)), ('unoccupied', (poor == 1) & (occ == 0))]:
        rows.append(dict(model=m, subset=lab, n_poor=int(mask.sum()), recall=float((p[mask] == 1).mean())))
    rows.append(dict(model=m, subset='false_alarm_occupied', n_poor=0, recall=float((p[(poor == 0) & (occ == 1)] == 1).mean())))
    rows.append(dict(model=m, subset='false_alarm_unoccupied', n_poor=0, recall=float((p[(poor == 0) & (occ == 0)] == 1).mean())))
t = pd.DataFrame(rows).pivot(index='model', columns='subset', values='recall').loc[order]
t.to_csv('tables/recall_by_occupancy.csv'); print(t.round(3))
