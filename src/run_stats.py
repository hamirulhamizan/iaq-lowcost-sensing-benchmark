"""Statistical comparison of the 12 classifiers under grouped validation.
Blocks: test days that contain Poor IAQ (Study 1, leave-one-day-out) and homes with at least 10 Poor hours (Study 2, leave-sites-out).
Friedman test on per-block F1, Nemenyi critical difference (alpha = 0.05), and Wilcoxon signed-rank tests of the
best-ranked model against every other model with Holm correction."""
import os
import numpy as np, pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon, studentized_range
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 8.5})
os.makedirs('results/stats', exist_ok=True); os.makedirs('figures', exist_ok=True)


def analyse(tab, study, metric='f1'):
    W = tab.pivot(index='block', columns='model', values=metric).dropna()
    N, k = W.shape
    ranks = W.rank(axis=1, ascending=False).mean().sort_values()
    chi, p = friedmanchisquare(*[W[c].values for c in W.columns])
    q = studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2); cd = q * np.sqrt(k * (k + 1) / (6 * N))
    best = ranks.index[0]; rows = []
    for m in ranks.index[1:]:
        d = W[best] - W[m]
        pv = wilcoxon(W[best], W[m], zero_method='zsplit').pvalue if (d != 0).any() else 1.0
        rows.append(dict(model=m, mean_rank=ranks[m], median_diff=float(d.median()), p=pv))
    wt = pd.DataFrame(rows).sort_values('p').reset_index(drop=True)
    mh = len(wt); wt['p_holm'] = np.minimum(1, np.maximum.accumulate([(mh - i) * p for i, p in enumerate(wt.p)]))
    wt['different_from_best'] = wt.p_holm < 0.05
    summ = dict(study=study, metric=metric, blocks=N, models=k, friedman_chi2=chi, friedman_p=p, nemenyi_cd=cd, best=best,
                n_within_cd_of_best=int(((ranks - ranks.iloc[0]) <= cd).sum() - 1))
    ranks.rename('mean_rank').to_frame().assign(within_cd_of_best=(ranks - ranks.iloc[0]) <= cd).to_csv(f'results/stats/ranks_{study}.csv')
    wt.to_csv(f'results/stats/wilcoxon_{study}.csv', index=False)
    return summ, ranks, cd


def cd_plot(ax, ranks, cd, title):
    """Mean rank per model (1 = best); shaded band = within the Nemenyi critical difference of the best model."""
    r = ranks.sort_values(ascending=False); y = np.arange(len(r)); b = ranks.min()
    ax.axvspan(b, b + cd, color='#9fb7d3', alpha=0.35, lw=0, label=f'within CD of best (CD = {cd:.2f})')
    ax.scatter(r.values, y, color=['#1f4e79' if v - b <= cd else '#c62828' for v in r.values], s=18, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(r.index, fontsize=7.5); ax.set_xlim(len(r) + 0.5, 0.5)
    ax.set_xlabel('Mean rank (1 = best)'); ax.set_title(title, fontsize=8, loc='left'); ax.grid(axis='x', alpha=0.3, lw=0.4)
    ax.legend(fontsize=6.5, loc='upper center', bbox_to_anchor=(0.5, -0.2), frameon=False)


if __name__ == '__main__':
    s1 = pd.read_csv('results/study1/per_day.csv'); s1 = s1[(s1.protocol == 'leave_one_day_out') & (s1.n_poor > 0)].rename(columns={'day': 'block'})
    s2 = pd.read_csv('results/study2/per_site_composite.csv'); s2 = s2[(s2.protocol == 'leave_sites_out') & (s2.n_poor >= 10)].rename(columns={'site': 'block'})
    out = []; fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.5))
    for (tab, st, ttl), ax in zip([(s1, 'study1', '(a) Office room, 10 days'),
                                   (s2, 'study2', '(b) Home offices, 86 homes')], axs):
        summ, ranks, cd = analyse(tab, st); out.append(summ)
        out.append(analyse(tab, st + '_mcc', 'mcc')[0])
        cd_plot(ax, ranks, cd, ttl + ('\nFriedman p < 0.001' if summ['friedman_p'] < 0.001 else f"\nFriedman p = {summ['friedman_p']:.2f}"))
    fig.tight_layout(); fig.savefig('figures/fig_cd.png', dpi=300); plt.close()
    pd.DataFrame(out).to_csv('results/stats/friedman.csv', index=False); print(pd.DataFrame(out).round(4).to_string())
