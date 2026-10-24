"""Study 1 (office room, CO2 > 1000 ppm): 12 classifiers x 3 validation protocols.
Writes results/study1/summary.csv, per_day.csv, params.json, lodo_predictions.csv."""
import json, sys
import numpy as np, pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, train_test_split
from common import load_config, load_data
from engine import evaluate
from runner import cached, flat, load

cfg = load_config()


def outer_splits(df, protocol):
    y = df.poor.values; idx = np.arange(len(df))
    if protocol == 'random':
        tr, te = train_test_split(idx, test_size=cfg['protocols']['random']['test_size'], stratify=y, random_state=cfg['seed'])
        return [(tr, te, 'random')], 'stratified', cfg['protocols']['random']['inner_folds']
    if protocol == 'leave_one_day_out':
        g = df.day.values
        return [(tr, te, g[te][0]) for tr, te in LeaveOneGroupOut().split(idx, y, g)], 'group', cfg['protocols']['leave_one_day_out']['inner_folds']
    pc = cfg['protocols']['chronological']
    tr = np.where(df.date <= pd.Timestamp(pc['train_end']))[0]; te = np.where(df.date >= pd.Timestamp(pc['test_start']))[0]
    return [(tr, te, 'chronological')], 'group', pc['inner_folds']


def run(df, name, protocol, feats, key):
    def fn():
        outer, inner, folds = outer_splits(df, protocol)
        pooled, per, params, pr = evaluate(cfg, df, name, feats, outer, block='day', inner=inner, inner_group='day', folds=folds)
        return dict(pooled=pooled, per=per, params=params, preds=pr)
    return cached('study1', key, fn)


if __name__ == '__main__':
    df = load_data(cfg); feats = cfg['data']['base_features']
    protocols = sys.argv[1].split(',') if len(sys.argv) > 1 else ['random', 'leave_one_day_out', 'chronological']
    for p in protocols:
        for m in cfg['models']:
            run(df, m, p, feats, f'main/{p}/{m}')
    # collect everything available
    summ, per, params, preds = [], [], {}, {}
    for p in ['random', 'leave_one_day_out', 'chronological']:
        for m in cfg['models']:
            r = load('study1', f'main/{p}/{m}')
            if r is None: continue
            summ.append({'protocol': p, 'model': m, **{k: flat(v) for k, v in r['pooled'].items()}, 'seconds': r['seconds']})
            per.append(r['per'].assign(protocol=p, model=m)); params[f'{p}/{m}'] = r['params']
            if p == 'leave_one_day_out':
                preds[f'{m}_pred'] = r['preds']['pred']; preds[f'{m}_score'] = r['preds']['score']
    pd.DataFrame(summ).to_csv('results/study1/summary.csv', index=False)
    pd.concat(per).rename(columns={'block': 'day'}).to_csv('results/study1/per_day.csv', index=False)
    json.dump(params, open('results/study1/params.json', 'w'), indent=1, default=str)
    if preds:
        pd.DataFrame(preds).assign(date=df.date.astype(str), day=df.day, poor=df.poor, Occupancy=df.Occupancy).to_csv('results/study1/lodo_predictions.csv.gz', index=False)
