"""Simple operational baselines evaluated under the same outer splits as the classifiers (Protocol B).
Rule parameters are chosen on the training part of each fold to maximise F1 (no test information).
Study 1: always Poor, always Good, light threshold, working-hours rule (weekday and hour in [a, b)).
Study 2: always Poor, always Good, RH threshold, sound-level threshold, working-hours rule."""
import os
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from common import load_config, load_data, metrics
from engine import block_bootstrap_f1
import run_main as R1, run_multisite as R2

cfg = load_config(); os.makedirs('results/baselines', exist_ok=True)


def thr_rule(col):
    def fit(d):
        cand = np.unique(np.quantile(d[col], np.linspace(0.01, 0.99, 99)))
        best = max(cand, key=lambda t: f1_score(d.poor, d[col] > t, zero_division=0)); return lambda e: (e[col] > best).astype(int), {col: float(best)}
    return fit


def hours_rule(d):
    wk = d.ts.dt.dayofweek < 5; h = d.ts.dt.hour; best, bs = None, -1
    for a in range(5, 13):
        for b in range(a + 2, 23):
            s = f1_score(d.poor, wk & (h >= a) & (h < b), zero_division=0)
            if s > bs: best, bs = (a, b), s
    a, b = best
    return (lambda e: ((e.ts.dt.dayofweek < 5) & (e.ts.dt.hour >= a) & (e.ts.dt.hour < b)).astype(int)), {'start_hour': a, 'end_hour': b}


const = lambda v: (lambda d: ((lambda e: np.full(len(e), v)), {}))


def run(df, outer, rules, block, tag):
    rows, preds, params = [], {}, []
    for name, fit in rules.items():
        p = np.full(len(df), -1)
        for tr, te, f in outer:
            fn, prm = fit(df.iloc[tr]); p[te] = np.asarray(fn(df.iloc[te])); params.append(dict(study=tag, rule=name, fold=str(f), **prm))
        m = p >= 0; y = df.poor.values[m]; q = p[m]; mt = metrics(y, q)
        sub = df.loc[m, [block]].assign(y=y, p=q)
        per = pd.DataFrame([dict(block=b, **{k: metrics(s.y.values, s.p.values)[k] for k in ['tp', 'fp', 'fn', 'tn']}) for b, s in sub.groupby(block)])
        lo, hi = block_bootstrap_f1(per, cfg)
        rows.append(dict(study=tag, rule=name, **mt, f1_ci_low=lo, f1_ci_high=hi, alarm_time=float(q.mean()), fpr=mt['fp'] / max(mt['fp'] + mt['tn'], 1)))
        preds[f'{name}_pred'] = p
    return pd.DataFrame(rows), preds, pd.DataFrame(params)


df = load_data(cfg).rename(columns={'date': 'ts'})
outer, _, _ = R1.outer_splits(df, 'leave_one_day_out')
s1, p1, a1 = run(df, outer, {'Always Poor': const(1), 'Always Good': const(0), 'Light threshold': thr_rule('Light'), 'Working hours': hours_rule}, 'day', 'study1')
pd.DataFrame(p1).assign(date=df.ts.astype(str), day=df.day, poor=df.poor, Occupancy=df.Occupancy).to_csv('results/baselines/preds_study1.csv.gz', index=False)
w = R2.load('composite'); outer, _ = R2.outer_splits(w, 'leave_sites_out')
s2, p2, a2 = run(w, outer, {'Always Poor': const(1), 'Always Good': const(0), 'RH threshold': thr_rule('Humidity'), 'Sound-level threshold': thr_rule('SPL'), 'Working hours': hours_rule}, 'site', 'study2')
pd.DataFrame(p2).assign(site=w.site.values, ts=w.ts.astype(str).values, poor=w.poor.values).to_csv('results/baselines/preds_study2.csv.gz', index=False)
pd.concat([s1, s2]).to_csv('results/baselines/summary.csv', index=False); pd.concat([a1, a2]).to_csv('results/baselines/params.csv', index=False)
print(pd.concat([s1, s2])[['study', 'rule', 'f1', 'f1_ci_low', 'f1_ci_high', 'precision', 'recall', 'mcc', 'alarm_time', 'fpr']].round(3).to_string())
