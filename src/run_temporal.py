"""Temporal-separation sensitivity analyses (all 12 models, identical tuning).
Study 1  buffered_lodo   : leave-one-day-out, but the calendar days adjacent to the test day are removed from training,
                           so an episode spanning midnight (e.g. 17-18 Feb) cannot be split between training and test.
Study 1  forward_chaining: prospective; each test day is predicted by a model trained only on earlier days, with
                           chronological inner validation (earlier days -> next block of days). Test days: 6-18 Feb.
Study 2  homes_and_future: leave-sites-out AND chronological at once: train on the first 70% of the period in the
                           training homes, test on the last 30% in the held-out homes."""
import sys
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from common import load_config, load_data
from engine import evaluate
from runner import cached, flat, load
import run_multisite as RMS

cfg = load_config()


def s1_outer(df, kind):
    days = np.array(sorted(df.day.unique())); dt = pd.to_datetime(days); out = []
    for i, d in enumerate(days):
        te = np.flatnonzero(df.day.values == d)
        if kind == 'buffered_lodo':
            near = days[np.abs((dt - dt[i]).days) <= 1]
            tr = np.flatnonzero(~np.isin(df.day.values, near))
        else:
            if i < 4: continue
            tr = np.flatnonzero(np.isin(df.day.values, days[:i]))
        out.append((tr, te, d))
    return out


if __name__ == '__main__':
    which = sys.argv[1].split(',') if len(sys.argv) > 1 else ['buffered_lodo', 'forward_chaining', 'homes_and_future']
    rows = []
    df = load_data(cfg); feats = cfg['data']['base_features']
    for kind in [k for k in which if k != 'homes_and_future']:
        outer = s1_outer(df, kind); inner = 'group' if kind == 'buffered_lodo' else 'time'
        for m in cfg['models']:
            r = cached('temporal', f'study1/{kind}/{m}', lambda: dict(zip(['pooled', 'per', 'params', 'preds'],
                       evaluate(cfg, df, m, feats, outer, block='day', inner=inner, inner_group='day', folds=3))))
    if 'homes_and_future' in which:
        w = RMS.load('composite'); M = RMS.M
        cut = w.ts.min() + (w.ts.max() - w.ts.min()) * M['chrono_train_frac']; idx = np.arange(len(w))
        outer = [(tr[w.ts.values[tr] <= cut], te[w.ts.values[te] > cut], k) for k, (tr, te) in enumerate(GroupKFold(n_splits=M['outer_folds']).split(idx, w.poor, w.site))]
        for m in cfg['models']:
            cached('temporal', f'study2/homes_and_future/{m}', lambda: dict(zip(['pooled', 'per', 'params', 'preds'],
                   evaluate(cfg, w, m, M['inputs'], outer, block='site', inner='group', inner_group='site', folds=3))))
    for st, kinds in [('study1', ['buffered_lodo', 'forward_chaining']), ('study2', ['homes_and_future'])]:
        for kind in kinds:
            for m in cfg['models']:
                r = load('temporal', f'{st}/{kind}/{m}')
                if r: rows.append(dict(study=st, protocol=kind, model=m, **{k: flat(v) for k, v in r['pooled'].items()}))
    pd.DataFrame(rows).to_csv('results/temporal_summary.csv', index=False)
