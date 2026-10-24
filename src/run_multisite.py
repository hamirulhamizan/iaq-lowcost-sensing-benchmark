"""Multi-pollutant, multi-site experiment (work-from-home dataset, Rysanek 2024).
Label: Poor IAQ if CO2 > 1000 ppm (ICOP-IAQ 2010) OR PM2.5 > 15 ug/m3 (WHO 2021) OR TVOC > 3000 ppb (ICOP-IAQ 2010, 3 ppm).
Inputs: low-cost temperature, RH and sound pressure level only.
Protocols: random 70/30, leave-sites-out (GroupKFold by site), chronological hold-out.
Usage: python src/run_multisite.py [protocol,...] [label]"""
import os, sys, json, time, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, yaml
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedKFold, train_test_split
from common import build, metrics

cfg = yaml.safe_load(open('config.yaml')); M = cfg['multisite']
for _m, _o in (M.get('model_overrides') or {}).items():
    cfg['models'][_m] = {**cfg['models'][_m], **_o}
rng = np.random.default_rng(cfg['seed'])

def load(label=None):
    w = pd.read_csv(M['file'], parse_dates=['ts']).dropna(subset=M['inputs'] + ['CO2', 'PM2.5', 'tVOC']).sort_values(['site', 'ts']).reset_index(drop=True)
    th = M['thresholds']
    comp = {'CO2': w.CO2 > th['CO2_ppm'], 'PM2.5': w['PM2.5'] > th['PM25_ugm3'], 'TVOC': w.tVOC > th['TVOC_ppb']}
    w['poor'] = (comp['CO2'] | comp['PM2.5'] | comp['TVOC']).astype(int) if label in (None, 'composite') else comp[label].astype(int)
    for k, v in comp.items(): w[f'exceed_{k}'] = v.astype(int)
    h = w.ts.dt.hour; w['hour_sin'] = np.sin(2 * np.pi * h / 24); w['hour_cos'] = np.cos(2 * np.pi * h / 24)
    for c in ['Temperature', 'Humidity', 'SPL']:
        w['d' + c] = w.groupby('site')[c].diff(1).fillna(0.0)
    w['day'] = w.ts.dt.strftime('%Y-%m-%d')
    return w

def fit(name, X, y, groups=None, inner='group', folds=3, fixed=None):
    pipe, grid = build(cfg, name)
    cap = M.get('svm_max_train', None)
    if name == 'SVM' and cap and len(y) > cap:  # kernel SVM scales quadratically: stratified random subsample
        idx = train_test_split(np.arange(len(y)), train_size=cap, stratify=y, random_state=cfg['seed'])[0]
        X, y = X[idx], y[idx]; groups = None if groups is None else groups[idx]
    if fixed is not None:
        step = pipe.steps[-1][0]; pipe.set_params(**{f'{step}__{k}': (tuple(v) if isinstance(v, list) else v) for k, v in fixed.items()})
        return pipe.fit(X, y), dict(fixed)
    if not grid: return pipe.fit(X, y), {}
    cv = list(GroupKFold(n_splits=folds).split(X, y, groups)) if inner == 'group' else StratifiedKFold(folds, shuffle=True, random_state=cfg['seed'])
    gs = GridSearchCV(pipe, grid, scoring=cfg['tuning_metric'], cv=cv, n_jobs=2, error_score=0.0).fit(X, y)
    return gs.best_estimator_, {k.split('__', 1)[1]: v for k, v in gs.best_params_.items()}

def site_bootstrap(per, B=2000):
    tp, fp, fn = (per[c].values for c in ['tp', 'fp', 'fn']); n = len(per); vals = []
    for _ in range(B):
        i = rng.integers(0, n, n); a, b, c = tp[i].sum(), fp[i].sum(), fn[i].sum(); vals.append(0 if a == 0 else 2 * a / (2 * a + b + c))
    return np.percentile(vals, [2.5, 97.5])

def run(w, name, protocol, feats, fixed=None):
    X = w[feats].values; y = w.poor.values; s = w.site.values.astype(str)
    pred = np.full(len(w), -1); params = []
    if protocol == 'random':
        tr, te = train_test_split(np.arange(len(w)), test_size=0.3, stratify=y, random_state=cfg['seed'])
        est, bp = fit(name, X[tr], y[tr], inner='strat', fixed=fixed); pred[te] = est.predict(X[te]); params.append(bp)
    elif protocol == 'leave_sites_out':
        for k, (tr, te) in enumerate(GroupKFold(n_splits=M['outer_folds']).split(X, y, s)):
            est, bp = fit(name, X[tr], y[tr], groups=s[tr], fixed=fixed); pred[te] = est.predict(X[te]); params.append({'fold': k, **bp})
    elif protocol == 'chronological':
        cut = w.ts.min() + (w.ts.max() - w.ts.min()) * M['chrono_train_frac']
        tr = np.where(w.ts <= cut)[0]; te = np.where(w.ts > cut)[0]
        est, bp = fit(name, X[tr], y[tr], groups=s[tr], fixed=fixed); pred[te] = est.predict(X[te]); params.append({'cut': str(cut), **bp})
    m = pred >= 0; pooled = metrics(y[m], pred[m])
    sub = w.loc[m, ['site']].assign(y=y[m], p=pred[m])
    per = pd.DataFrame([dict(site=k, n=len(g), n_poor=int(g.y.sum()), **{q: metrics(g.y.values, g.p.values)[q] for q in ['tp', 'fp', 'fn', 'tn']}) for k, g in sub.groupby('site')])
    if protocol != 'random':
        pooled['f1_ci_low'], pooled['f1_ci_high'] = site_bootstrap(per)
    return pooled, per, params, pred

if __name__ == '__main__':
    protocols = sys.argv[1].split(',') if len(sys.argv) > 1 else ['random', 'leave_sites_out', 'chronological']
    label = sys.argv[2] if len(sys.argv) > 2 else 'composite'
    w = load(label); feats = M['inputs']; os.makedirs('results/multisite', exist_ok=True)
    print(f'label={label} n={len(w)} sites={w.site.nunique()} prevalence={w.poor.mean():.3f}', flush=True)
    tag = f"{label}_{'_'.join(protocols)}"; part = f'results/multisite/partial_{tag}.json'
    state = json.load(open(part)) if os.path.exists(part) else {'summ': [], 'params': {}, 'preds': {}}
    summ, params, preds = state['summ'], state['params'], state['preds']
    for p in protocols:
        for name in cfg['models']:
            if any(r['protocol'] == p and r['model'] == name for r in summ): continue
            t = time.time(); pooled, per, prm, pred = run(w, name, p, feats)
            summ.append({k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in dict(label=label, protocol=p, model=name, prevalence=w.poor.mean(), **pooled, seconds=round(time.time() - t, 1)).items()})
            params[f'{p}/{name}'] = prm
            if p == 'leave_sites_out': preds[name] = pred.tolist()
            print(f"{label:9s} {p:16s} {name:9s} F1={pooled['f1']:.3f} BA={pooled['balanced_accuracy']:.3f} ({time.time()-t:.0f}s)", flush=True)
            json.dump({'summ': summ, 'params': params, 'preds': preds}, open(part, 'w'), default=str)
    pd.DataFrame(summ).to_csv(f'results/multisite/summary_{tag}.csv', index=False)
    json.dump(params, open(f'results/multisite/params_{tag}.json', 'w'), indent=1, default=str)
    if preds:
        pd.DataFrame(preds).assign(site=w.site.values, ts=w.ts.astype(str).values, poor=w.poor.values,
                                   **{c: w[c].values for c in ['exceed_CO2', 'exceed_PM2.5', 'exceed_TVOC']}).to_csv(f'results/multisite/lso_predictions_{label}.csv', index=False)
