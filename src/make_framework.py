"""Fig. 1: research framework (two-lane flowchart)."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({'font.family': 'DejaVu Sans', 'mathtext.default': 'regular'})
NAVY, INK, GREY = '#1f3b5c', '#222222', '#5f6b78'
S1F, S1E = '#e7eef7', '#4a6f9a'      # Study 1 lane
S2F, S2E = '#fbefe3', '#b87333'      # Study 2 lane
SHF, SHE = '#f2f4f6', '#7b8794'      # shared steps

W, H, YB = 7.0, 9.4, 1.42
fig = plt.figure(figsize=(W, H - YB)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(YB, H); ax.axis('off')
X0, X1 = 1.2, 6.85                   # content area (left margin holds stage labels)
GAP = 0.14; XM = (X0 + X1) / 2


def box(x, y, w, h, title, lines, fc, ec, tsize=7.6, lsize=6.9, align='left'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.07', fc=fc, ec=ec, lw=0.9))
    tx = x + 0.12 if align == 'left' else x + w / 2; ha = 'left' if align == 'left' else 'center'
    ax.text(tx, y + h - 0.12, title, ha=ha, va='top', fontsize=tsize, fontweight='bold', color=INK)
    for i, l in enumerate(lines):
        ax.text(tx, y + h - 0.37 - i * 0.19, l, ha=ha, va='top', fontsize=lsize, color=INK)


def stage(y, h, n, label):
    yc = y + h / 2
    ax.add_patch(plt.Circle((0.32, yc), 0.17, fc=NAVY, ec='none'))
    ax.text(0.32, yc, str(n), ha='center', va='center', color='white', fontsize=8, fontweight='bold')
    ax.text(0.56, yc, label, ha='left', va='center', fontsize=7.4, fontweight='bold', color=NAVY, rotation=0)


def arrow(y_from, y_to, x=XM):
    ax.add_patch(FancyArrowPatch((x, y_from), (x, y_to), arrowstyle='-|>', mutation_scale=9, color=NAVY, lw=0.9))


def two(y, h, n, label, t1, l1, t2, l2):
    stage(y, h, n, label); w = (X1 - X0 - GAP) / 2
    box(X0, y, w, h, t1, l1, S1F, S1E); box(X0 + w + GAP, y, w, h, t2, l2, S2F, S2E)


def one(y, h, n, label, title, lines):
    stage(y, h, n, label); box(X0, y, X1 - X0, h, title, lines, SHF, SHE)


def chips(y, h, n, label, title, items, fc, ec):
    stage(y, h, n, label)
    ax.add_patch(FancyBboxPatch((X0, y), X1 - X0, h, boxstyle='round,pad=0,rounding_size=0.07', fc=SHF, ec=SHE, lw=0.9))
    ax.text(X0 + 0.12, y + h - 0.12, title, ha='left', va='top', fontsize=7.3, fontweight='bold', color=INK)
    k = len(items); cw = (X1 - X0 - 0.24 - (k - 1) * 0.08) / k; ch = h - 0.5
    for i, (head, sub) in enumerate(items):
        cx = X0 + 0.12 + i * (cw + 0.08)
        ax.add_patch(FancyBboxPatch((cx, y + 0.12), cw, ch, boxstyle='round,pad=0,rounding_size=0.05', fc=fc[i] if isinstance(fc, list) else fc, ec=ec[i] if isinstance(ec, list) else ec, lw=0.7))
        ax.text(cx + cw / 2, y + 0.12 + ch - 0.1, head, ha='center', va='top', fontsize=6.6, fontweight='bold', color=INK)
        for j, s in enumerate(sub):
            ax.text(cx + cw / 2, y + 0.12 + ch - 0.33 - j * 0.17, s, ha='center', va='top', fontsize=6.2, color=INK)


# column headers
w = (X1 - X0 - GAP) / 2
ax.text(X0 + w / 2, 9.22, 'Study 1 · office room', ha='center', va='center', fontsize=8.6, fontweight='bold', color=S1E)
ax.text(X0 + w + GAP + w / 2, 9.22, 'Study 2 · home offices', ha='center', va='center', fontsize=8.6, fontweight='bold', color=S2E)

rows = []
y = 8.05; h = 0.95
two(y, h, 1, 'Data', 'Occupancy Detection dataset', ['Candanedo & Feldheim (2016)', '1 room · 17 days · 1-min records', 'n = 20,560'],
    'Work-from-home IEQ dataset', ['Manu & Rysanek (2024)', '95 homes · 53 days · hourly records', 'n = 120,590'])
arrow(y, y - 0.17); y -= 0.17 + 0.62; h = 0.62
one(y, h, 2, 'Prepare', 'Pre-processing', ['merge and de-duplicate · gap and validity checks · z-score scaling fitted on training folds only'])
arrow(y, y - 0.17); y -= 0.17 + 0.78; h = 0.78
two(y, h, 3, 'Label', 'CO$_2$ exceedance', ['CO$_2$ > 1000 ppm (ICOP-IAQ 2010)', 'exceedances: 15.0% of minutes'],
    'Multi-pollutant exceedance', ['CO$_2$ > 1000 ppm  or  TVOC > 3 ppm (ICOP-IAQ 2010)', 'or  PM2.5 > 15 µg/m$^3$ (WHO 2021) · 14.5% of hours'])
arrow(y, y - 0.17); y -= 0.17 + 0.78; h = 0.78
two(y, h, 4, 'Inputs', 'Ambient signals', ['temperature · relative humidity · light', 'pollutants never used as inputs'],
    'Ambient signals', ['temperature · relative humidity · sound level', 'pollutants never used as inputs'])
arrow(y, y - 0.17); y -= 0.17 + 1.12; h = 1.12
chips(y, h, 5, 'Models', 'Identical treatment for every model',
      [('12 classifiers', ['LR · LDA · NB · KNN · DT · RF', 'ET · AdaBoost · GB · HGB · SVM · MLP']),
       ('Tuning (training data only)', ['grid search on PR-AUC', 'F1-optimal threshold, inner folds']),
       ('Simple-rule baselines', ['always alarm · working hours', 'light / RH / sound threshold'])],
      ['white', 'white', 'white'], [SHE, SHE, SHE])
arrow(y, y - 0.17); y -= 0.17 + 1.12; h = 1.12
chips(y, h, 6, 'Validate', 'Validation protocols and the scenario each represents',
      [('A · Random', ['70/30 split', 'interpolation']), ('B · Grouped', ['unseen day (S1)', 'unseen home (S2)']),
       ('C · Future', ['later period,', 'same room/homes']), ('D · Buffered', ['unseen day,', '±1 day removed']),
       ('E · Forward', ['prospective:', 'past days only']), ('F · New homes', ['unseen homes,', 'later period'])],
      ['white', 'white', 'white', S1F, S1F, S2F], [SHE, SHE, SHE, S1E, S1E, S2E])
ax.text(X1 - 0.12, y + h - 0.13, 'blue: Study 1 only   orange: Study 2 only', ha='right', va='top', fontsize=6.2, color=GREY, style='italic')
arrow(y, y - 0.17); y -= 0.17 + 1.12; h = 1.12
chips(y, h, 7, 'Evaluate', 'Evaluation',
      [('Classification', ['F1 · MCC · PR-AUC', 'block-bootstrap 95% CI']), ('Statistics', ['Friedman · Nemenyi', 'paired differences']),
       ('Alarm burden', ['time in alarm · episodes', 'recall at 10% budget']), ('Explanation', ['occupancy · pollutant', 'ablation · label sensitivity'])],
      'white', SHE)
fig.savefig('figures/fig1_framework.png', dpi=300); print('saved', y)
