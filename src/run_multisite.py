"""Study 2 (95 home offices, Rysanek 2024): multi-pollutant label, inputs temperature, RH and sound level only.
Label (composite): Poor if CO2 > 1000 ppm (ICOP-IAQ 2010) OR PM2.5 > 15 ug/m3 (WHO 2021) OR TVOC > 3000 ppb (ICOP-IAQ 2010).
Other labels: CO2, PM2.5 (single pollutant); composite_pm24h (PM2.5 as trailing 24-h mean, the averaging time of the
WHO guideline); composite_noTVOC (CO2 or PM2.5 only).
Protocols: random 70/30, leave-sites-out (GroupKFold 5 by site), chronological hold-out (first 70% of the period).
Usage: python src/run_multisite.py <protocol,...> <label> [model,...]"""
import json, sys
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split
from common import load_config
from engine import evaluate
from runner import cached, load as load_run, flat

cfg = load_config(); M = cfg['multisite']
EXC = ['exceed_CO2', 'exceed_PM2.5', 'exceed_TVOC']


def load(label='composite', pm25=None):
    w = pd.read_csv(M['file'], parse_dates=['ts']).dropna(subset=M['inputs'] + ['CO2', 'PM2.5', 'tVOC']).sort_values(['site', 'ts']).reset_index(drop=True)
    th = M['thresholds']; pm25 = th['PM25_ugm3'] if pm25 is None else pm25
    pm24 = w.set_index('ts').groupby('site')['PM2.5'].rolling('24h', min_periods=18).mean().values  # trailing 24-h mean
    w['PM2.5_24h'] = pm24
    ex = {'CO2': w.CO2 > th['CO2_ppm'], 'PM2.5': w['PM2.5'] > pm25, 'TVOC': w.tVOC > th['TVOC_ppb']}
    lab = {'composite': ex['CO2'] | ex['PM2.5'] | ex['TVOC'], 'CO2': ex['CO2'], 'PM2.5': ex['PM2.5'],
           'composite_noTVOC': ex['CO2'] | ex['PM2.5'],
           'composite_pm24h': ex['CO2'] | (w['PM2.5_24h'] > pm25) | ex['TVOC']}[label]
    w['poor'] = lab.astype(int)
    for k, v in ex.items(): w[f'exceed_{k}'] = v.astype(int)
    h = w.ts.dt.hour; w['hour_sin'] = np.sin(2 * np.pi * h / 24); w['hour_cos'] = np.cos(2 * np.pi * h / 24)
    for c in ['Temperature', 'Humidity', 'SPL']:
        w['d' + c] = w.groupby('site')[c].diff(1).fillna(0.0)
    w['site'] = w.site.astype(str)
    return w


def outer_splits(w, protocol):
    y = w.poor.values; idx = np.arange(len(w))
    if protocol == 'random':
        tr, te = train_test_split(idx, test_size=0.3, stratify=y, random_state=cfg['seed'])
        return [(tr, te, 'random')], 'stratified'
    if protocol == 'leave_sites_out':
        return [(tr, te, k) for k, (tr, te) in enumerate(GroupKFold(n_splits=M['outer_folds']).split(idx, y, w.site.values))], 'group'
    cut = w.ts.min() + (w.ts.max() - w.ts.min()) * M['chrono_train_frac']
    return [(np.where(w.ts <= cut)[0], np.where(w.ts > cut)[0], str(cut))], 'group'


def run(w, name, protocol, feats, key):
    def fn():
        outer, inner = outer_splits(w, protocol)
        pooled, per, params, pr = evaluate(cfg, w, name, feats, outer, block='site', inner=inner, inner_group='site', folds=3)
        return dict(pooled=pooled, per=per, params=params, preds=pr)
    return cached('study2', key, fn)


if __name__ == '__main__':
    protocols = sys.argv[1].split(',') if len(sys.argv) > 1 else ['random', 'leave_sites_out', 'chronological']
    label = sys.argv[2] if len(sys.argv) > 2 else 'composite'
    models = sys.argv[3].split(',') if len(sys.argv) > 3 else list(cfg['models'])
    w = load(label)
    print(f'label={label} n={len(w)} sites={w.site.nunique()} prevalence={w.poor.mean():.3f}', flush=True)
    for p in protocols:
        for m in models:
            run(w, m, p, M['inputs'], f'{label}/{p}/{m}')
    # collect all stored runs for this label
    summ, per, params, preds = [], [], {}, {}
    for p in ['random', 'leave_sites_out', 'chronological']:
        for m in cfg['models']:
            r = load_run('study2', f'{label}/{p}/{m}')
            if r is None: continue
            summ.append({'label': label, 'protocol': p, 'model': m, 'prevalence': w.poor.mean(), **{k: flat(v) for k, v in r['pooled'].items()}, 'seconds': r['seconds']})
            per.append(r['per'].assign(label=label, protocol=p, model=m)); params[f'{p}/{m}'] = r['params']
            if p == 'leave_sites_out':
                preds[f'{m}_pred'] = r['preds']['pred']; preds[f'{m}_score'] = r['preds']['score']
    pd.DataFrame(summ).to_csv(f'results/study2/summary_{label}.csv', index=False)
    pd.concat(per).rename(columns={'block': 'site'}).to_csv(f'results/study2/per_site_{label}.csv', index=False)
    json.dump(params, open(f'results/study2/params_{label}.json', 'w'), indent=1, default=str)
    if preds:
        pd.DataFrame(preds).assign(site=w.site.values, ts=w.ts.astype(str).values, poor=w.poor.values,
                                   **{c: w[c].values for c in EXC}).to_csv(f'results/study2/lso_predictions_{label}.csv.gz', index=False)
