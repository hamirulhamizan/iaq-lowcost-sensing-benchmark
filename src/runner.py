"""Checkpointed execution: each (study, label, protocol, model, setting) run is stored in results/<study>/runs/
so an interrupted job resumes where it stopped. collect() assembles the stored runs into tables."""
import os, pickle, time, glob
import numpy as np, pandas as pd


def path(study, key):
    d = f'results/{study}/runs'; os.makedirs(d, exist_ok=True)
    return f"{d}/{key.replace('/', '_').replace(' ', '_').replace('>', 'gt').replace('+', 'plus')}.pkl"


def load(study, key):
    f = path(study, key)
    return pickle.load(open(f, 'rb')) if os.path.exists(f) else None


def cached(study, key, fn):
    f = path(study, key)
    if os.path.exists(f):
        return pickle.load(open(f, 'rb'))
    t = time.time(); out = fn(); out['seconds'] = round(time.time() - t, 1)
    pickle.dump(out, open(f + '.tmp', 'wb')); os.replace(f + '.tmp', f)
    p = out['pooled']
    print(f"{key:60s} F1={p['f1']:.3f} MCC={p['mcc']:.3f} PR-AUC={p['pr_auc']:.3f} F1@default={p['f1_default']:.3f} ({out['seconds']:.0f}s)", flush=True)
    return out


def flat(v):
    return float(v) if isinstance(v, (np.floating, np.integer)) else v
