"""Episode-level evaluation of the out-of-fold predictions under grouped validation.
An episode is a run of consecutive Poor-IAQ time steps (Study 1: minutes, at least 5 min long; Study 2: hours, per home).
Detected: at least one Poor prediction inside the episode. Delay: time from episode start to the first Poor prediction.
False-alarm run: a run of consecutive Poor predictions that does not overlap any Poor-IAQ time step."""
import os
import numpy as np, pandas as pd
os.makedirs('results/episodes', exist_ok=True)


def runs(flag, ts, step, group=None):
    """start/end indices of runs of flag==1 that are consecutive in time (and within the same group)."""
    f = np.asarray(flag).astype(bool); n = len(f)
    brk = np.ones(n, bool); brk[1:] = (np.diff(ts) != step)
    if group is not None: brk[1:] |= (group[1:] != group[:-1])
    out = []; i = 0
    while i < n:
        if f[i]:
            j = i
            while j + 1 < n and f[j + 1] and not brk[j + 1]: j += 1
            out.append((i, j)); i = j + 1
        else:
            i += 1
    return out


def evaluate(P, models, ts, step, unit_min, min_len, group, n_days):
    y = P.poor.values; ep = [(a, b) for a, b in runs(y, ts, step, group) if b - a + 1 >= min_len]; rows = []
    for m in models:
        p = P[f'{m}_pred'].values; det, delay = 0, []
        for a, b in ep:
            hit = np.flatnonzero(p[a:b + 1] == 1)
            if len(hit): det += 1; delay.append(hit[0] * unit_min)
        fa = [(a, b) for a, b in runs(p, ts, step, group) if y[a:b + 1].sum() == 0]
        rows.append(dict(model=m, episodes=len(ep), detected=det, episode_recall=det / len(ep),
                         median_delay=float(np.median(delay)) if delay else np.nan,
                         p75_delay=float(np.percentile(delay, 75)) if delay else np.nan,
                         false_alarm_runs=len(fa), false_alarm_runs_per_day=len(fa) / n_days,
                         false_alarm_runs_ge_len=sum(1 for a, b in fa if b - a + 1 >= min_len) / n_days))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    models = pd.read_csv('results/study1/summary.csv').model.unique().tolist()
    P = pd.read_csv('results/study1/lodo_predictions.csv.gz', parse_dates=['date']).sort_values('date').reset_index(drop=True)
    ts = (P.date.astype('int64') // 60_000_000_000).values
    e1 = evaluate(P, models, ts, 1, 1, 5, None, P.day.nunique()); e1.to_csv('results/episodes/study1.csv', index=False)
    print('Study 1 (minutes)\n', e1.round(3).to_string())
    P = pd.read_csv('results/study2/lso_predictions_composite.csv.gz', parse_dates=['ts']).sort_values(['site', 'ts']).reset_index(drop=True)
    ts = (P.ts.astype('int64') // 3_600_000_000_000).values
    site_days = P.groupby('site').ts.apply(lambda s: s.dt.normalize().nunique()).sum()
    e2 = evaluate(P, models, ts, 1, 60, 1, P.site.astype(str).values, site_days); e2.to_csv('results/episodes/study2.csv', index=False)
    print('Study 2 (hours; delay in minutes; false alarms per home-day)\n', e2.round(3).to_string())
