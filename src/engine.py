"""Model selection and evaluation engine shared by every study.

The same procedure is applied to every classifier, so no model gets an imbalance remedy the others lack:
  1. hyperparameters are chosen by grid search on average precision (PR-AUC), which does not depend on a cut-off;
  2. the decision threshold is then chosen to maximise F1 on out-of-fold scores from the same inner folds;
  3. the model is refitted on the whole training part and applied to the held-out part with that threshold.
All three steps use training data only. Kernel SVM is fitted on a stratified sample of at most `svm_max_train`
training rows in every study (its cost grows quadratically with the number of rows)."""
import warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import precision_recall_curve, average_precision_score
from common import build, metrics


def scores(est, X):
    """Continuous score for the Poor class: probability when available, otherwise the decision function."""
    if hasattr(est, 'predict_proba'):
        return est.predict_proba(X)[:, 1], 'proba'
    return est.decision_function(X), 'decision'


def best_threshold(y, s):
    p, r, t = precision_recall_curve(y, s)
    f = np.where(p[:-1] + r[:-1] > 0, 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12), 0)
    return float(t[int(np.argmax(f))]) if len(t) else 0.5


def fit_select(cfg, name, X, y, groups=None, inner='group', folds=3, fixed=None):
    pipe, grid = build(cfg, name)
    cap = cfg.get('svm_max_train')
    if name == 'SVM' and cap and len(y) > cap:
        idx = train_test_split(np.arange(len(y)), train_size=cap, stratify=y, random_state=cfg['seed'])[0]
        X, y = X[idx], y[idx]; groups = None if groups is None else groups[idx]
    cv = list(GroupKFold(n_splits=folds).split(X, y, groups)) if inner == 'group' else \
        list(StratifiedKFold(folds, shuffle=True, random_state=cfg['seed']).split(X, y))
    step = pipe.steps[-1][0]
    if fixed is not None:
        pipe.set_params(**{f'{step}__{k}': (tuple(v) if isinstance(v, list) else v) for k, v in fixed.items() if k != 'threshold'})
        params = {k: v for k, v in fixed.items() if k != 'threshold'}
    elif grid:
        gs = GridSearchCV(pipe, grid, scoring=cfg['tuning_metric'], cv=cv, n_jobs=2, error_score=0.0, refit=False).fit(X, y)
        pipe.set_params(**gs.best_params_); params = {k.split('__', 1)[1]: v for k, v in gs.best_params_.items()}
    else:
        params = {}
    method = 'predict_proba' if hasattr(pipe, 'predict_proba') else 'decision_function'
    oof = cross_val_predict(clone(pipe), X, y, cv=cv, method=method, n_jobs=2)
    oof = oof[:, 1] if oof.ndim == 2 else oof
    thr = best_threshold(y, oof) if cfg['threshold_tuning'] else (0.5 if method == 'predict_proba' else 0.0)
    est = pipe.fit(X, y)
    return est, {**params, 'threshold': thr}


def block_bootstrap_f1(per, cfg):
    rng = np.random.default_rng(cfg['seed']); tp, fp, fn = (per[c].values for c in ['tp', 'fp', 'fn']); n = len(per); v = []
    for _ in range(cfg['bootstrap']['n_resamples']):
        i = rng.integers(0, n, n); a, b, c = tp[i].sum(), fp[i].sum(), fn[i].sum(); v.append(0.0 if a == 0 else 2 * a / (2 * a + b + c))
    return tuple(np.percentile(v, [2.5, 97.5]))


def evaluate(cfg, df, name, feats, outer, block, inner='group', inner_group=None, folds=3, fixed=None):
    """outer: list of (train_idx, test_idx, tag). block: column used for per-block metrics and the bootstrap CI.
    Returns pooled metrics, per-block table, selected parameters per outer fold, and out-of-fold predictions/scores."""
    X = df[feats].values; y = df['poor'].values
    g = None if inner_group is None else df[inner_group].values.astype(str)
    pred = np.full(len(df), -1); pred05 = np.full(len(df), -1); sc = np.full(len(df), np.nan); fold_id = np.full(len(df), -1)
    params, ap_fold = [], []
    for k, (tr, te, tag) in enumerate(outer):
        est, bp = fit_select(cfg, name, X[tr], y[tr], None if g is None else g[tr], inner, folds, fixed)
        s, kind = scores(est, X[te])
        pred[te] = (s >= bp['threshold']).astype(int); pred05[te] = est.predict(X[te]); sc[te] = s; fold_id[te] = k
        if y[te].sum() > 0: ap_fold.append(average_precision_score(y[te], s))
        params.append({'fold': tag, **bp})
    m = pred >= 0
    pooled = metrics(y[m], pred[m])
    d = metrics(y[m], pred05[m]); pooled.update(f1_default=d['f1'], mcc_default=d['mcc'], recall_default=d['recall'])
    pooled['pr_auc'] = float(np.mean(ap_fold)); pooled['prevalence_test'] = float(y[m].mean())
    sub = df.loc[m, [block]].assign(y=y[m], p=pred[m], s=sc[m])
    rows = []
    for b, s in sub.groupby(block):
        mm = metrics(s.y.values, s.p.values)
        rows.append(dict(block=b, n=len(s), n_poor=int(s.y.sum()), **{q: mm[q] for q in ['tp', 'fp', 'fn', 'tn']},
                         f1=(mm['f1'] if s.y.sum() > 0 else np.nan), mcc=mm['mcc'],
                         false_alarm_rate=(mm['fp'] / (mm['fp'] + mm['tn']) if (mm['fp'] + mm['tn']) else np.nan)))
    per = pd.DataFrame(rows)
    if len(outer) > 1 or len(per) > 1:
        pooled['f1_ci_low'], pooled['f1_ci_high'] = block_bootstrap_f1(per, cfg)
    return pooled, per, params, dict(pred=pred, pred_default=pred05, score=sc, fold=fold_id)
