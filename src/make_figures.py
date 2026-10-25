"""Figures for the paper, generated from results/*.csv."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch
from common import load_config, load_data

plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 8.5})
cfg = load_config(); df = load_data(cfg)
NAVY, LIGHT, RED, GREEN = '#1f4e79', '#9fb7d3', '#c62828', '#2e7d32'
summ = pd.read_csv('results/study1/summary.csv')
perday = pd.read_csv('results/study1/per_day.csv'); perday = perday[perday.protocol == 'leave_one_day_out']
order = summ[summ.protocol == 'leave_one_day_out'].sort_values('f1', ascending=False).model.tolist()

# ---- Fig 1: framework (drawn by make_framework.py)
import runpy; runpy.run_path('src/make_framework.py')

# ---- Fig 2: time series (gaps shown as breaks)
d = df.set_index('date')[['Temperature', 'Humidity', 'Light', 'CO2', 'poor']].resample('1min').mean()
fig, axs = plt.subplots(4, 1, figsize=(6.6, 6.2), sharex=True)
for a, (c, lab, band) in zip(axs, [('Temperature', 'Temperature\n(°C)', (23, 26)), ('Humidity', 'RH\n(%)', (40, 70)), ('Light', 'Light\n(lux)', None), ('CO2', 'CO$_2$\n(ppm)', None)]):
    a.plot(d.index, d[c], lw=0.5, color=NAVY); a.set_ylabel(lab); a.grid(alpha=0.3, lw=0.4)
    if band: a.axhspan(*band, color=GREEN, alpha=0.12, lw=0)
axs[0].set_ylim(19, 26.5); axs[1].set_ylim(15, 45)
axs[3].axhline(1000, color=RED, ls='--', lw=0.8)
axs[3].fill_between(d.index, 400, 2150, where=d['poor'] > 0.5, color=RED, alpha=0.22, lw=0); axs[3].set_ylim(400, 2150)
for wk in pd.date_range('2015-02-07', periods=2, freq='7D'):
    for a in axs: a.axvspan(wk, wk + pd.Timedelta('2D'), color='grey', alpha=0.08, lw=0)
axs[3].xaxis.set_major_locator(mdates.DayLocator(interval=2)); axs[3].xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
fig.tight_layout(); fig.savefig('figures/fig2_timeseries.png', dpi=300); plt.close()

# ---- Fig 3: F1 by model and protocol
P = [('random', 'A: Random 70/30', LIGHT), ('chronological', 'C: Chronological hold-out', '#5b84b1'), ('leave_one_day_out', 'B: Leave-one-day-out', NAVY)]
fig, ax = plt.subplots(figsize=(6.6, 3.0)); x = np.arange(len(order)); w = 0.27
for k, (p, lab, col) in enumerate(P):
    v = summ[summ.protocol == p].set_index('model').loc[order]
    ax.bar(x + (k - 1) * w, v.f1, w, color=col, label=lab)
    if p == 'leave_one_day_out':
        ax.errorbar(x + (k - 1) * w, v.f1, yerr=[v.f1 - v.f1_ci_low, v.f1_ci_high - v.f1], fmt='none', ecolor='black', elinewidth=0.6, capsize=1.5)
ax.set_xticks(x); ax.set_xticklabels(order, fontsize=7.5); ax.set_ylabel('F1-score (exceedance class)'); ax.set_ylim(0, 1.08)
ax.legend(fontsize=7, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.15), frameon=False); ax.grid(axis='y', alpha=0.3, lw=0.4)
fig.tight_layout(); fig.savefig('figures/fig3_protocols.png', dpi=300); plt.close()

# ---- Fig 4: per-day F1 heatmap (leave-one-day-out), days that contain Poor IAQ
pos = perday[perday.n_poor > 0]
hm = pos.pivot(index='model', columns='day', values='f1').loc[order]
fig, ax = plt.subplots(figsize=(6.6, 3.6))
im = ax.imshow(hm.values, cmap='Blues', vmin=0, vmax=1, aspect='auto')
for i in range(hm.shape[0]):
    for j in range(hm.shape[1]):
        v = hm.values[i, j]; ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=6.3, color='white' if v > 0.6 else 'black')
npoor = pos.groupby('day').n_poor.first()
ax.set_xticks(range(hm.shape[1])); ax.set_xticklabels([f"{pd.Timestamp(c):%d %b}\n(n={npoor[c]})" for c in hm.columns], fontsize=6.5)
ax.set_yticks(range(hm.shape[0])); ax.set_yticklabels(order, fontsize=7.5)
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01); cb.set_label('F1-score', fontsize=7.5)
fig.tight_layout(); fig.savefig('figures/fig4_perday_heatmap.png', dpi=300); plt.close()

# ---- Fig 5: feature ablation
try:
    ab = pd.read_csv('results/study1/ablation_features.csv')
    sets = list(cfg['ablation']['feature_sets'].keys()); models = ab.model.unique().tolist()
    fig, ax = plt.subplots(figsize=(6.6, 3.2)); y = np.arange(len(sets)); hgt = 0.8 / len(models)
    cols = [NAVY, '#5b84b1', LIGHT]
    for k, m in enumerate(models):
        v = ab[ab.model == m].set_index('setting').loc[sets]
        ax.barh(y + (k - (len(models) - 1) / 2) * hgt, v.f1, hgt, color=cols[k], label=m)
    ax.set_yticks(y); ax.set_yticklabels(sets, fontsize=7.5); ax.invert_yaxis(); ax.set_xlabel('F1-score (leave-one-day-out)'); ax.set_xlim(0, 1)
    ax.axvline(ab[ab.setting == sets[0]].f1.max(), color='grey', ls=':', lw=0.8)
    ax.legend(fontsize=7, frameon=False, loc='lower right'); ax.grid(axis='x', alpha=0.3, lw=0.4)
    fig.tight_layout(); fig.savefig('figures/fig5_ablation.png', dpi=300); plt.close()
except FileNotFoundError:
    pass
print('figures written')
