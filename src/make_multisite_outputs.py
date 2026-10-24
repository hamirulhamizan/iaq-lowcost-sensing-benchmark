"""Tables and figures for the multi-pollutant (work-from-home) and sensitivity analyses."""
import glob, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 8.5})
NAVY, MID, LIGHT = '#1f4e79', '#5b84b1', '#9fb7d3'
os.makedirs('tables', exist_ok=True); os.makedirs('figures', exist_ok=True)

S = pd.concat([pd.read_csv(f) for f in glob.glob('results/multisite/summary_*.csv')])
comp = S[S.label == 'composite']
order = comp[comp.protocol == 'leave_sites_out'].sort_values('f1', ascending=False).model.tolist()
t = comp.pivot(index='model', columns='protocol', values='f1').loc[order]
lso = comp[comp.protocol == 'leave_sites_out'].set_index('model').loc[order]
out = pd.DataFrame({'random_f1': t['random'], 'chronological_f1': t['chronological'], 'lso_f1': lso.f1, 'lso_ci_low': lso.f1_ci_low,
                    'lso_ci_high': lso.f1_ci_high, 'lso_precision': lso.precision, 'lso_recall': lso.recall,
                    'lso_bal_acc': lso.balanced_accuracy, 'lso_mcc': lso.mcc})
for lab in ['CO2', 'PM2.5']:
    s = S[(S.label == lab) & (S.protocol == 'leave_sites_out')].set_index('model')
    if len(s): out[f'lso_f1_{lab}_only'] = s.f1.reindex(order)
out.to_csv('tables/multisite_results.csv')

# recall by which pollutant caused Poor IAQ (composite label, leave-sites-out predictions)
P = pd.read_csv('results/multisite/lso_predictions_composite.csv')
rows = []
for m in order:
    p = P[m].values
    for name, mask in [('CO2 only', (P['exceed_CO2'] == 1) & (P['exceed_PM2.5'] == 0) & (P['exceed_TVOC'] == 0)),
                       ('PM2.5 only', (P['exceed_PM2.5'] == 1) & (P['exceed_CO2'] == 0) & (P['exceed_TVOC'] == 0)),
                       ('TVOC only', (P['exceed_TVOC'] == 1) & (P['exceed_CO2'] == 0) & (P['exceed_PM2.5'] == 0)),
                       ('Several pollutants', (P[['exceed_CO2', 'exceed_PM2.5', 'exceed_TVOC']].sum(axis=1) >= 2))]:
        rows.append(dict(model=m, cause=name, n=int(mask.sum()), recall=float((p[mask] == 1).mean()) if mask.sum() else np.nan))
rc = pd.DataFrame(rows); rc.to_csv('tables/multisite_recall_by_cause.csv', index=False)

# Fig 6: F1 by protocol
fig, ax = plt.subplots(figsize=(6.6, 2.9)); x = np.arange(len(order)); w = 0.27
for k, (col, lab, c) in enumerate([('random_f1', 'Random 70/30', LIGHT), ('chronological_f1', 'Chronological hold-out', MID), ('lso_f1', 'Leave-sites-out', NAVY)]):
    ax.bar(x + (k - 1) * w, out[col], w, color=c, label=lab)
ax.errorbar(x + w, out.lso_f1, yerr=[out.lso_f1 - out.lso_ci_low, out.lso_ci_high - out.lso_f1], fmt='none', ecolor='black', elinewidth=0.6, capsize=1.5)
ax.set_xticks(x); ax.set_xticklabels(order, fontsize=7.5); ax.set_ylabel('F1-score (Poor IAQ, multi-pollutant)'); ax.set_ylim(0, 0.75)
ax.legend(fontsize=7, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.16), frameon=False); ax.grid(axis='y', alpha=0.3, lw=0.4)
fig.tight_layout(); fig.savefig('figures/fig6_multisite_protocols.png', dpi=300); plt.close()

# Fig 7: recall by cause, top 4 models
top = order[:4]; causes = ['CO2 only', 'PM2.5 only', 'TVOC only', 'Several pollutants']
fig, ax = plt.subplots(figsize=(6.6, 2.7)); x = np.arange(len(causes)); w = 0.8 / len(top)
cols = [NAVY, MID, LIGHT, '#c9d6e8']
for k, m in enumerate(top):
    v = rc[rc.model == m].set_index('cause').loc[causes].recall
    ax.bar(x + (k - (len(top) - 1) / 2) * w, v, w, color=cols[k], label=m)
n = rc[rc.model == top[0]].set_index('cause').loc[causes].n
ax.set_xticks(x); ax.set_xticklabels([f'{c}\n(n = {n[c]:,} h)' for c in causes], fontsize=7.5); ax.set_ylabel('Recall of Poor IAQ'); ax.set_ylim(0, 1)
ax.legend(fontsize=7, ncol=4, frameon=False, loc='upper right'); ax.grid(axis='y', alpha=0.3, lw=0.4)
fig.tight_layout(); fig.savefig('figures/fig7_recall_by_pollutant.png', dpi=300); plt.close()
print(out.round(3).to_string()); print(rc[rc.model.isin(top)].round(3).to_string())

# selected hyperparameters in Study 2 (modal over the 5 leave-sites-out folds)
import json, collections
prm = json.load(open('results/multisite/params_composite_random_leave_sites_out_chronological.json'))
sel = []
for m in order:
    c = collections.Counter(json.dumps({k: v for k, v in d.items() if k != 'fold'}, sort_keys=True) for d in prm[f'leave_sites_out/{m}'])
    best, n = c.most_common(1)[0]; sel.append(dict(model=m, modal_params=best, folds=f'{n}/{len(prm[f"leave_sites_out/{m}"])}'))
pd.DataFrame(sel).to_csv('tables/multisite_selected_params.csv', index=False)

# Study 2 ablation table and figure
if os.path.exists('results/multisite/ablation_features.csv'):
    import yaml
    sets = list(yaml.safe_load(open('config.yaml'))['multisite']['ablation']['feature_sets'].keys())
    ab = pd.read_csv('results/multisite/ablation_features.csv')
    f1 = ab.pivot(index='setting', columns='model', values='f1').loc[sets]
    mcc = ab.pivot(index='setting', columns='model', values='mcc').loc[sets]
    models = ab.model.unique().tolist(); f1 = f1[models]; mcc = mcc[models]
    out2 = pd.concat({'f1': f1, 'mcc': mcc}, axis=1); out2.to_csv('tables/multisite_ablation_features.csv')
    fig, ax = plt.subplots(figsize=(6.6, 3.1)); y = np.arange(len(sets)); hgt = 0.8 / len(models)
    for k, m in enumerate(models):
        ax.barh(y + (k - (len(models) - 1) / 2) * hgt, f1[m], hgt, color=[NAVY, MID, LIGHT][k], label=m)
    ax.set_yticks(y); ax.set_yticklabels(sets, fontsize=7.5); ax.invert_yaxis(); ax.set_xlabel('F1-score (leave-sites-out, multi-pollutant label)')
    ax.axvline(f1.loc[sets[0]].max(), color='grey', ls=':', lw=0.8); ax.set_xlim(0, max(0.6, f1.values.max() + 0.05))
    ax.legend(fontsize=7, frameon=False, loc='lower right'); ax.grid(axis='x', alpha=0.3, lw=0.4)
    fig.tight_layout(); fig.savefig('figures/fig8_multisite_ablation.png', dpi=300); plt.close()
    print(out2.round(3).to_string())
if os.path.exists('results/multisite/ablation_pm25_threshold.csv'):
    th = pd.read_csv('results/multisite/ablation_pm25_threshold.csv'); print(th[['setting', 'model', 'prevalence', 'pm25_share', 'f1', 'mcc']].round(3).to_string())
