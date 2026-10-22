"""Tables used in the paper (written to tables/*.csv)."""
import json, collections, os
import pandas as pd
from common import load_config, load_data

cfg = load_config(); df = load_data(cfg)
os.makedirs('tables', exist_ok=True)
summ = pd.concat([pd.read_csv('results/main_summary_random_chronological.csv'), pd.read_csv('results/main_summary_leave_one_day_out.csv')])
order = summ[summ.protocol == 'leave_one_day_out'].sort_values('f1', ascending=False).model.tolist()

# dataset description per day
d = df.groupby('day').agg(records=('poor', 'size'), poor=('poor', 'sum'), occupied=('Occupancy', 'sum'),
                          co2_max=('CO2', 'max'), rh_mean=('Humidity', 'mean'), t_mean=('Temperature', 'mean')).reset_index()
d.to_csv('tables/dataset_by_day.csv', index=False)

# main comparison
rows = []
for m in order:
    r = {'model': m}
    for p in ['random', 'chronological', 'leave_one_day_out']:
        s = summ[(summ.protocol == p) & (summ.model == m)].iloc[0]
        for k in ['accuracy', 'precision', 'recall', 'f1', 'balanced_accuracy', 'mcc']:
            r[f'{p}_{k}'] = s[k]
        if p == 'leave_one_day_out':
            for k in ['f1_ci_low', 'f1_ci_high', 'f1_day_mean', 'f1_day_sd', 'f1_day_min', 'f1_day_max', 'days_f1_below_0.5']:
                r[k] = s[k]
    rows.append(r)
pd.DataFrame(rows).to_csv('tables/main_results.csv', index=False)

# selected hyperparameters (modal across leave-one-day-out folds, with frequency)
sel = json.load(open('results/selected_params_leave_one_day_out.json'))
out = []
for m in order:
    c = collections.Counter(json.dumps({k: v for k, v in x.items() if k != 'test_day'}, sort_keys=True) for x in sel[f'leave_one_day_out/{m}'])
    best, n = c.most_common(1)[0]
    out.append(dict(model=m, modal_params=best, folds=f'{n}/{len(sel[f"leave_one_day_out/{m}"])}'))
pd.DataFrame(out).to_csv('tables/selected_params.csv', index=False)
print(pd.DataFrame(out).to_string())

# recall of Poor IAQ split by whether the room was occupied (Occupancy column used for analysis only, never as input)
pr = pd.read_csv('results/lodo_predictions.csv')
occ = df['Occupancy'].values; poor = df['poor'].values
rows = []
for m in order:
    p = pr[m].values
    for lab, mask in [('occupied', (poor == 1) & (occ == 1)), ('unoccupied', (poor == 1) & (occ == 0))]:
        rows.append(dict(model=m, subset=lab, n_poor=int(mask.sum()), recall=float((p[mask] == 1).mean())))
pd.DataFrame(rows).pivot(index='model', columns='subset', values='recall').loc[order].to_csv('tables/recall_by_occupancy.csv')
print(pd.DataFrame(rows).pivot(index='model', columns='subset', values='recall').loc[order].round(3))
