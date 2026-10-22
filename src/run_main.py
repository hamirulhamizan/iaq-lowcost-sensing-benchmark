"""Main experiment: 12 classifiers x 3 validation protocols. Writes results/*.csv and results/selected_params.json."""
import json, sys, time
import pandas as pd
from common import load_config, load_data, run_protocol

cfg = load_config()
df = load_data(cfg)
feats = cfg['data']['base_features']
protocols = sys.argv[1].split(',') if len(sys.argv) > 1 else ['random', 'leave_one_day_out', 'chronological']
summary, perday, params, preds = [], [], {}, {}
for proto in protocols:
    for name in cfg['models']:
        t = time.time()
        pooled, pdy, prm, pred = run_protocol(cfg, df, name, proto, feats)
        summary.append(dict(protocol=proto, model=name, **pooled, seconds=round(time.time() - t, 1)))
        perday.append(pdy.assign(protocol=proto, model=name))
        params[f'{proto}/{name}'] = prm
        if proto == 'leave_one_day_out':
            preds[name] = pred.tolist()
        print(f"{proto:18s} {name:9s} F1={pooled['f1']:.3f} BA={pooled['balanced_accuracy']:.3f} ({time.time()-t:.0f}s)", flush=True)
tag = '_'.join(protocols) if len(sys.argv) > 1 else 'all'
pd.DataFrame(summary).to_csv(f'results/main_summary_{tag}.csv', index=False)
pd.concat(perday).to_csv(f'results/main_per_day_{tag}.csv', index=False)
json.dump(params, open(f'results/selected_params_{tag}.json', 'w'), indent=1, default=str)
if preds:
    pd.DataFrame(preds).assign(date=df['date'].astype(str), poor=df['poor']).to_csv('results/lodo_predictions.csv', index=False)
