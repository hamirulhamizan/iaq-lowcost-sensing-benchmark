"""Operating point with a fixed false-alarm budget for the main grouped-validation runs.
For every outer fold the hyperparameters selected in the main run are reused, the inner out-of-fold scores are
recomputed, and the threshold is set to the highest-recall value whose inner false-alarm rate is <= 10%
(training data only). Recall, precision and false-alarm rate on the held-out days/homes are reported."""
import numpy as np, pandas as pd
from common import load_config, load_data
from engine import evaluate
from runner import cached, flat, load
import run_main as R1, run_multisite as R2

cfg = load_config(); rows = []
df = load_data(cfg)
outer, inner, folds = R1.outer_splits(df, 'leave_one_day_out')
for m in cfg['models']:
    fx = [{k: v for k, v in p.items() if k not in ('fold', 'threshold')} for p in load('study1', f'main/leave_one_day_out/{m}')['params']]
    r = cached('budget', f'study1/{m}', lambda: dict(zip(['pooled', 'per', 'params', 'preds'], evaluate(cfg, df, m, cfg['data']['base_features'], outer, 'day', inner, 'day', folds, fixed_per_fold=fx))))
    rows.append(dict(study='study1', model=m, **{k: flat(v) for k, v in r['pooled'].items()}))
w = R2.load('composite'); outer, inner = R2.outer_splits(w, 'leave_sites_out')
for m in cfg['models']:
    fx = [{k: v for k, v in p.items() if k not in ('fold', 'threshold')} for p in load('study2', f'composite/leave_sites_out/{m}')['params']]
    r = cached('budget', f'study2/{m}', lambda: dict(zip(['pooled', 'per', 'params', 'preds'], evaluate(cfg, w, m, R2.M['inputs'], outer, 'site', inner, 'site', 3, fixed_per_fold=fx))))
    rows.append(dict(study='study2', model=m, **{k: flat(v) for k, v in r['pooled'].items()}))
pd.DataFrame(rows).to_csv('results/budget_summary.csv', index=False)
