"""Sensitivity analysis on two high-pollution datasets under Malaysian limits.
Label: Poor if CO2 > 1000 ppm (ICOP-IAQ 2010) OR PM2.5 > 35 ug/m3 OR PM10 > 100 ug/m3 (MAAQS 2020, 24-h).
Inputs: temperature and RH only (the only low-cost inputs common to both datasets).
DALTON (India): hourly, validation grouped by site.  Lovric (Croatia): 1-min, validation grouped by day.
Also reports label prevalence under the WHO 2021 limits used in the main analysis (PM2.5 15, PM10 45)."""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import run_multisite as R
from sklearn.model_selection import GroupKFold, train_test_split
from common import metrics

cfg = R.cfg; S = cfg['sensitivity']
os.makedirs('results/sensitivity', exist_ok=True)

def label(d, pm25, pm10):
    return ((d.CO2 > 1000) | (d['PM2.5'] > pm25) | (d.PM10 > pm10)).astype(int)

def load(name):
    if name == 'DALTON':
        d = pd.read_csv(S['dalton_file'], parse_dates=['ts']).rename(columns={'T': 'Temperature', 'H': 'Humidity', 'PMS2_5': 'PM2.5', 'PMS10': 'PM10'})
        d['group'] = d.site
    else:
        d = pd.read_csv(S['lovric_file'], parse_dates=['ts']); d['group'] = d.ts.dt.strftime('%Y-%m-%d')
    return d.dropna(subset=['Temperature', 'Humidity', 'CO2', 'PM2.5', 'PM10']).reset_index(drop=True)

def run(d, model, protocol):
    X = d[['Temperature', 'Humidity']].values; y = d.poor.values; g = d.group.values.astype(str)
    pred = np.full(len(d), -1)
    if protocol == 'random':
        tr, te = train_test_split(np.arange(len(d)), test_size=0.3, stratify=y, random_state=cfg['seed'])
        est, _ = R.fit(model, X[tr], y[tr], inner='strat'); pred[te] = est.predict(X[te])
    else:
        for tr, te in GroupKFold(n_splits=min(5, len(set(g)))).split(X, y, g):
            est, _ = R.fit(model, X[tr], y[tr], groups=g[tr]); pred[te] = est.predict(X[te])
    m = pred >= 0
    return metrics(y[m], pred[m])

rows, prev = [], []
for name in ['DALTON', 'Lovric']:
    d = load(name)
    prev.append(dict(dataset=name, n=len(d), groups=d.group.nunique(),
                     who_poor=label(d, 15, 45).mean(), my_poor=label(d, 35, 100).mean(),
                     co2=(d.CO2 > 1000).mean(), pm25_who=(d['PM2.5'] > 15).mean(), pm25_my=(d['PM2.5'] > 35).mean(),
                     pm10_who=(d.PM10 > 45).mean(), pm10_my=(d.PM10 > 100).mean()))
    d['poor'] = label(d, 35, 100)
    for model in S['models']:
        for protocol in ['random', 'grouped']:
            r = run(d, model, protocol)
            rows.append(dict(dataset=name, model=model, protocol=protocol, prevalence=d.poor.mean(), **r))
            print(f"{name:7s} {model:5s} {protocol:8s} F1={r['f1']:.3f} BA={r['balanced_accuracy']:.3f} MCC={r['mcc']:.3f}", flush=True)
pd.DataFrame(prev).to_csv('results/sensitivity/prevalence.csv', index=False)
pd.DataFrame(rows).to_csv('results/sensitivity/summary.csv', index=False)
print(pd.DataFrame(prev).round(3).to_string())
