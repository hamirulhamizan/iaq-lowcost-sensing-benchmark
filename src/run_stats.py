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


def paired_diff(per, best, cfg_seed=42, B=2000):
    """Pooled-F1 difference (best - model) with a block bootstrap over days/homes, both models evaluated on the same resample."""
    rng = np.random.default_rng(cfg_seed); blocks = sorted(per.block.unique()); out = []
    T = {m: g.set_index('block').reindex(blocks)[['tp', 'fp', 'fn']].values for m, g in per.groupby('model')}
    f1 = lambda a: 0.0 if a[0] == 0 else 2 * a[0] / (2 * a[0] + a[1] + a[2])
    idx = [rng.integers(0, len(blocks), len(blocks)) for _ in range(B)]
    for m in T:
        if m == best: continue
        d0 = f1(T[best].sum(0)) - f1(T[m].sum(0)); v = [f1(T[best][i].sum(0)) - f1(T[m][i].sum(0)) for i in idx]
        lo, hi = np.percentile(v, [2.5, 97.5]); out.append(dict(model=m, diff=d0, ci_low=lo, ci_high=hi, ci_excludes_zero=bool(lo > 0 or hi < 0)))
    return pd.DataFrame(out).sort_values('diff')


if __name__ == '__main__':
    p1 = pd.read_csv('results/study1/per_day.csv'); p1 = p1[p1.protocol == 'leave_one_day_out'].rename(columns={'day': 'block'})
    p2 = pd.read_csv('results/study2/per_site_composite.csv'); p2 = p2[p2.protocol == 'leave_sites_out'].rename(columns={'site': 'block'})
    s1 = pd.read_csv('results/study1/summary.csv'); b1 = s1[s1.protocol == 'leave_one_day_out'].sort_values('f1').model.iloc[-1]
    s2 = pd.read_csv('results/study2/summary_composite.csv'); b2 = s2[s2.protocol == 'leave_sites_out'].sort_values('f1').model.iloc[-1]
    paired_diff(p1, b1).assign(best=b1).to_csv('results/stats/paired_f1_diff_study1.csv', index=False)
    paired_diff(p2, b2).assign(best=b2).to_csv('results/stats/paired_f1_diff_study2.csv', index=False)
    # Friedman over ALL 17 days on the per-day error rate (FP + FN) / n, which also counts false alarms on days without Poor IAQ
    p1['error_rate'] = (p1.fp + p1.fn) / p1.n
    W = p1.pivot(index='block', columns='model', values='error_rate'); chi, pv = friedmanchisquare(*[W[c] for c in W.columns])
    rk = W.rank(axis=1).mean().sort_values()
    pd.DataFrame([dict(study='study1_all_days_error', blocks=len(W), friedman_chi2=chi, friedman_p=pv, best=rk.index[0], best_rank=rk.iloc[0])]).to_csv('results/stats/friedman_all_days.csv', index=False)
    print(pd.read_csv('results/stats/paired_f1_diff_study1.csv').round(3).to_string()); print(pd.read_csv('results/stats/paired_f1_diff_study2.csv').round(3).to_string())
    print(pd.read_csv('results/stats/friedman_all_days.csv').round(4).to_string())
