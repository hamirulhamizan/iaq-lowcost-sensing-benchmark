"""Shared utilities: data loading, feature construction, model building and nested validation."""
import os, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
import numpy as np
import pandas as pd
import yaml
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold, GroupKFold, LeaveOneGroupOut, train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             balanced_accuracy_score, matthews_corrcoef, confusion_matrix)
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, HistGradientBoostingClassifier)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

ESTIMATORS = {c.__name__: c for c in [LogisticRegression, LinearDiscriminantAnalysis, GaussianNB, KNeighborsClassifier,
                                      DecisionTreeClassifier, RandomForestClassifier, ExtraTreesClassifier,
                                      AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier,
                                      SVC, MLPClassifier]}
SEEDED = {'LogisticRegression', 'DecisionTreeClassifier', 'RandomForestClassifier', 'ExtraTreesClassifier',
          'AdaBoostClassifier', 'GradientBoostingClassifier', 'HistGradientBoostingClassifier', 'SVC', 'MLPClassifier'}


def load_config(path='config.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)


def load_data(cfg, threshold=None):
    frames = [pd.read_csv(f, parse_dates=['date']) for f in cfg['data']['files']]
    df = pd.concat(frames).sort_values('date').drop_duplicates('date').reset_index(drop=True)
    thr = cfg['data']['co2_threshold_ppm'] if threshold is None else threshold
    df['poor'] = (df['CO2'] > thr).astype(int)
    df['day'] = df['date'].dt.strftime('%Y-%m-%d')
    # contiguous segments (gaps > 2 min start a new segment) for rate-of-change features
    seg = (df['date'].diff().dt.total_seconds() > 120).cumsum()
    w = cfg['data']['delta_window_min']
    for c in ['Temperature', 'Humidity', 'Light']:
        df['d' + c] = df.groupby(seg)[c].diff(w).fillna(0.0)
    # causal baseline correction: value minus its minimum over the previous 24 h (past data only)
    r = df.set_index('date')[['Temperature', 'Humidity']].rolling('24h').min()
    df['bTemperature'] = df['Temperature'].values - r['Temperature'].values
    df['bHumidity'] = df['Humidity'].values - r['Humidity'].values
    h = df['date'].dt.hour + df['date'].dt.minute / 60
    df['hour_sin'] = np.sin(2 * np.pi * h / 24)
    df['hour_cos'] = np.cos(2 * np.pi * h / 24)
    return df


def build(cfg, name):
    m = cfg['models'][name]
    cls = ESTIMATORS[m['estimator']]
    fixed = {k: (tuple(v) if isinstance(v, list) else v) for k, v in (m.get('fixed') or {}).items()}
    if m['estimator'] in SEEDED:
        fixed['random_state'] = cfg['seed']
    pipe = make_pipeline(StandardScaler(), cls(**fixed))
    step = pipe.steps[-1][0]
    grid = {f'{step}__{k}': [tuple(x) if isinstance(x, list) else x for x in v] for k, v in (m.get('grid') or {}).items()}
    return pipe, grid


def fit_tuned(cfg, name, X, y, groups=None, inner='group_kfold', folds=4, fixed=None):
    pipe, grid = build(cfg, name)
    if fixed is not None:  # ablation: use given hyperparameters, no re-tuning
        step = pipe.steps[-1][0]
        pipe.set_params(**{f'{step}__{k}': (tuple(v) if isinstance(v, list) else v) for k, v in fixed.items()})
        return pipe.fit(X, y), dict(fixed)
    if not grid:
        return pipe.fit(X, y), {}
    if inner == 'group_kfold':
        cv = list(GroupKFold(n_splits=folds).split(X, y, groups))
    else:
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=cfg['seed'])
    gs = GridSearchCV(pipe, grid, scoring=cfg['tuning_metric'], cv=cv, n_jobs=2, error_score=0.0)
    gs.fit(X, y)
    return gs.best_estimator_, {k.split('__', 1)[1]: v for k, v in gs.best_params_.items()}


def metrics(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return dict(accuracy=accuracy_score(y, p), precision=precision_score(y, p, zero_division=0),
                recall=recall_score(y, p, zero_division=0), f1=f1_score(y, p, zero_division=0),
                balanced_accuracy=balanced_accuracy_score(y, p), mcc=matthews_corrcoef(y, p),
                tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))


def day_bootstrap_f1(per_day, cfg):
    """95% CI of pooled F1 by resampling calendar days with replacement."""
    rng = np.random.default_rng(cfg['seed'])
    tp, fp, fn = (per_day[c].values for c in ['tp', 'fp', 'fn'])
    n = len(per_day); vals = []
    for _ in range(cfg['bootstrap']['n_resamples']):
        i = rng.integers(0, n, n)
        a, b, c = tp[i].sum(), fp[i].sum(), fn[i].sum()
        vals.append(0.0 if a == 0 else 2 * a / (2 * a + b + c))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return lo, hi


def run_protocol(cfg, df, name, protocol, feats, fixed=None):
    X = df[feats].values; y = df['poor'].values; g = df['day'].values
    rows, params = [], []
    if protocol == 'random':
        pc = cfg['protocols']['random']
        idx = np.arange(len(df))
        tr, te = train_test_split(idx, test_size=pc['test_size'], stratify=y, random_state=cfg['seed'])
        est, bp = fit_tuned(cfg, name, X[tr], y[tr], inner='stratified', folds=pc['inner_folds'])
        pred = np.full(len(df), -1); pred[te] = est.predict(X[te]); params.append(bp)
        mask = pred >= 0
    elif protocol == 'leave_one_day_out':
        pc = cfg['protocols']['leave_one_day_out']
        pred = np.full(len(df), -1)
        for tr, te in LeaveOneGroupOut().split(X, y, g):
            est, bp = fit_tuned(cfg, name, X[tr], y[tr], groups=g[tr], folds=pc['inner_folds'], fixed=fixed)
            pred[te] = est.predict(X[te]); params.append({'test_day': g[te][0], **bp})
        mask = pred >= 0
    elif protocol == 'chronological':
        pc = cfg['protocols']['chronological']
        tr = np.where(df['date'] <= pd.Timestamp(pc['train_end']))[0]
        te = np.where(df['date'] >= pd.Timestamp(pc['test_start']))[0]
        est, bp = fit_tuned(cfg, name, X[tr], y[tr], groups=g[tr], folds=pc['inner_folds'])
        pred = np.full(len(df), -1); pred[te] = est.predict(X[te]); params.append(bp)
        mask = pred >= 0
    yt, yp = y[mask], pred[mask]
    pooled = metrics(yt, yp)
    sub = df.loc[mask, ['day']].assign(y=yt, p=yp)
    per_day = []
    for d, s in sub.groupby('day'):
        m = metrics(s.y.values, s.p.values)
        per_day.append(dict(day=d, n=len(s), n_poor=int(s.y.sum()), **{k: m[k] for k in ['tp', 'fp', 'fn', 'tn']},
                            f1=(m['f1'] if s.y.sum() > 0 else np.nan),
                            false_alarm_rate=(m['fp'] / (m['fp'] + m['tn']) if (m['fp'] + m['tn']) else np.nan)))
    per_day = pd.DataFrame(per_day)
    if protocol != 'random':
        pooled['f1_ci_low'], pooled['f1_ci_high'] = day_bootstrap_f1(per_day, cfg)
        pos_days = per_day[per_day.n_poor > 0]
        pooled['f1_day_mean'] = pos_days.f1.mean(); pooled['f1_day_sd'] = pos_days.f1.std(ddof=1)
        pooled['f1_day_min'] = pos_days.f1.min(); pooled['f1_day_max'] = pos_days.f1.max()
        pooled['days_f1_below_0.5'] = int((pos_days.f1 < 0.5).sum())
    return pooled, per_day, params, pred
