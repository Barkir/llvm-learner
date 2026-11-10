#!/usr/bin/env python3
"""
train_retry_type_models.py — Predict, from the INPUT kernel features, whether a kernel
will hit each retry type (overflow / buffercountbump / vfstacklimit) and any retry at all.

Data:
  kernel_features.csv        one row per kernel (~700), input-kernel features
  kernel_features_after.csv  output of extract_end_features.py: one row per end dump.
                             Only 'kernel' + the retry type are used, to build labels:
                               y_<type> = 1 if the kernel has a dump of that type
                               y_any    = 1 if it has any dump
                             The type comes from end_retry_type, or else from the
                             end_dump_file name (see RETRY_TYPE_KEYWORDS).

  End-of-pipeline features are NEVER used as model inputs: at prediction time you only
  have the input kernel. A kernel can hit several types, so each target gets its own
  binary model (multi-label), not one multi-class model.

For each target (any + every type with >= --min-positives kernels):
  * compares logistic regression vs gradient-boosted trees with repeated stratified,
    grouped K-fold CV (same func_name stays in one fold -> no near-duplicate leakage)
  * reports ROC-AUC / PR-AUC vs base rate, best single features, a precision/recall
    table, a learning curve, permutation importances, worst false positives/negatives
  * trains the better model on all kernels and saves it with a threshold

Usage:
  python3 train_retry_type_models.py [kernel_features.csv] [kernel_features_after.csv] [model_dir]
  python3 train_retry_type_models.py --predict new_kernel_features.csv --model-dir ~/retry_type_models
"""

import argparse
import json
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

try:
    import xgboost as xgb
    HAVE_XGB = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAVE_XGB = False

warnings.filterwarnings('ignore', category=UserWarning)

ID_COLS = {'kernel'}
MAX_ONEHOT_LEVELS = 20

# same mapping as extract_end_features.py (used only if end_retry_type is missing)
RETRY_TYPE_KEYWORDS = [
    ('buffercount', 'buffercountbump'),
    ('vfstack', 'vfstacklimit'),
    ('stacklimit', 'vfstacklimit'),
    ('overflow', 'overflow'),
]


def retry_type_from_name(fname):
    low = str(fname).lower().replace('_', '').replace('-', '')
    for kw, t in RETRY_TYPE_KEYWORDS:
        if kw in low:
            return t
    return os.path.splitext(str(fname))[0]


# ── data ───────────────────────────────────────────────────────────────────
def load_labels_csv(feat, labels_csv, label_cols):
    """Any CSV with a kernel column and one or more 0/1 outcome columns."""
    lab = pd.read_csv(labels_csv, dtype={'kernel': str}).drop_duplicates('kernel')
    cols = ([c.strip() for c in label_cols.split(',')] if label_cols
            else [c for c in lab.columns if c != 'kernel'])
    missing = [c for c in cols if c not in lab.columns]
    if missing:
        sys.exit(f'{labels_csv} has no column(s) {missing}; available: '
                 f'{[c for c in lab.columns if c != "kernel"]}')
    lab = lab.set_index('kernel')
    labels = pd.DataFrame(index=feat['kernel'])
    for c in cols:
        v = labels.index.map(lab[c])
        labels[c] = [0 if pd.isna(x) else
                     int(str(x).strip().lower() not in ('0', 'false', '', 'nan'))
                     for x in v]
    return labels.reset_index(drop=True), cols


def load_data(features_csv, after_csv, group_col, labels_csv=None, label_cols=None):
    feat = pd.read_csv(features_csv, dtype={'kernel': str})
    feat['kernel'] = feat['kernel'].astype(str)
    if feat['kernel'].duplicated().any():
        print(f"  ! {feat['kernel'].duplicated().sum()} duplicate kernels in {features_csv}, keeping first rows")
        feat = feat.drop_duplicates('kernel')

    if labels_csv:
        labels, types = load_labels_csv(feat, labels_csv, label_cols)
        if 'any' not in labels.columns:
            labels.insert(0, 'any', (labels[types].sum(axis=1) > 0).astype(int))
        groups = feat[group_col].astype(str).values \
            if group_col and group_col in feat.columns else None
        print(f'Units: {len(feat)}' +
              (f'   groups ({group_col}): {len(set(groups))}' if groups is not None else ''))
        for t in ['any'] + [c for c in types if c != 'any']:
            print(f'  {t:20s} {labels[t].sum():4d} ({labels[t].mean():.1%})')
        return feat, labels, types, groups

    after = pd.read_csv(after_csv, dtype={'kernel': str})
    after['kernel'] = after['kernel'].astype(str)

    if 'end_retry_type' in after.columns:
        after['retry_type'] = after['end_retry_type'].astype(str)
    elif 'end_dump_file' in after.columns:
        after['retry_type'] = after['end_dump_file'].map(retry_type_from_name)
    else:
        print('  ! no end_retry_type / end_dump_file column: treating every dump as one type "retry"')
        after['retry_type'] = 'retry'

    missing = set(after['kernel']) - set(feat['kernel'])
    if missing:
        print(f'  ! {len(missing)} dumped kernels have no input features (ignored), e.g. {sorted(missing)[:3]}')

    types = sorted(after['retry_type'].unique())
    labels = pd.DataFrame(index=feat['kernel'])
    labels['any'] = labels.index.isin(set(after['kernel'])).astype(int)
    for t in types:
        labels[t] = labels.index.isin(set(after.loc[after.retry_type == t, 'kernel'])).astype(int)
    labels = labels.reset_index(drop=True)

    groups = None
    if group_col and group_col in feat.columns:
        groups = feat[group_col].astype(str).values
    elif group_col:
        print(f'  ! group column {group_col!r} not found; using plain stratified K-fold')

    n = len(feat)
    print(f'Kernels: {n}' + (f'   groups ({group_col}): {len(set(groups))}' if groups is not None else ''))
    print(f"  any retry       {labels['any'].sum():4d} ({labels['any'].mean():.1%})")
    for t in types:
        print(f'  {t:15s} {labels[t].sum():4d} ({labels[t].mean():.1%})')
    n_types = labels[types].sum(axis=1)
    if (n_types > 1).any():
        print(f'  {int((n_types > 1).sum())} kernels hit more than one type. Co-occurrence:')
        co = labels[types].T.dot(labels[types])
        print('    ' + co.to_string().replace('\n', '\n    '))
    return feat, labels, types, groups


def build_matrix(df, columns=None, drop_extra=()):
    """Numeric features + one-hot small categoricals. Pass `columns` to align to a trained model."""
    skip = ID_COLS | set(drop_extra)
    parts, dropped = [], []
    for c in df.columns:
        if c in skip or c.startswith('end_'):          # never use end-of-pipeline features
            continue
        s = df[c]
        if pd.api.types.is_bool_dtype(s):
            parts.append(s.astype(int).rename(c))
        elif pd.api.types.is_numeric_dtype(s):
            parts.append(pd.to_numeric(s, errors='coerce').rename(c))
        elif s.nunique() <= MAX_ONEHOT_LEVELS:
            parts.append(pd.get_dummies(s.astype(str), prefix=c, dtype=int))
        else:
            dropped.append(c)
    X = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0)
    else:
        X = X.loc[:, X.nunique() > 1]
    return X, dropped


# ── models ─────────────────────────────────────────────────────────────────
def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


def make_models(y):
    n_pos = max(int(y.sum()), 1)
    spw = (len(y) - n_pos) / n_pos
    logreg = make_pipeline(
        FunctionTransformer(signed_log1p),      # sizes/counts span orders of magnitude
        StandardScaler(),
        LogisticRegression(C=0.5, class_weight='balanced', max_iter=5000),
    )
    if HAVE_XGB:
        gbm = xgb.XGBClassifier(
            n_estimators=400, max_depth=3, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
            reg_lambda=2.0, scale_pos_weight=spw,
            eval_metric='logloss', random_state=42, verbosity=0, n_jobs=-1)
        name = 'xgboost'
    else:
        gbm = HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=10,
            l2_regularization=1.0, class_weight='balanced', random_state=42)
        name = 'hist_gbm'
    return {'logreg': logreg, name: gbm}


def make_splitter(groups, n_splits, seed):
    if groups is not None:
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


# ── evaluation helpers ─────────────────────────────────────────────────────
def cross_validate(models, X, y, groups, n_splits, repeats):
    oof = {m: np.zeros(len(y)) for m in models}
    fold_auc = {m: [] for m in models}
    fold_ap = {m: [] for m in models}
    for r in range(repeats):
        for tr, te in make_splitter(groups, n_splits, seed=r).split(X, y, groups):
            if y[te].sum() == 0 or y[te].sum() == len(te) or y[tr].sum() == 0:
                continue
            for name, model in models.items():
                p = clone(model).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
                oof[name][te] += p / repeats
                fold_auc[name].append(roc_auc_score(y[te], p))
                fold_ap[name].append(average_precision_score(y[te], p))
    return oof, fold_auc, fold_ap


def univariate_auc(X, y, top=8):
    rows = []
    for c in X.columns:
        v = X[c].values
        if np.unique(v).size < 2:
            continue
        auc = roc_auc_score(y, v)
        rows.append((c, max(auc, 1 - auc), 'higher' if auc >= 0.5 else 'lower'))
    return sorted(rows, key=lambda r: -r[1])[:top]


def threshold_table(y, p):
    print(f"  {'thresh':>7s} {'precision':>9s} {'recall':>7s} {'F1':>6s} {'TP':>4s} {'FP':>4s} {'FN':>4s} {'flagged':>7s}")
    best = (0, 0.5)
    for t in np.round(np.arange(0.1, 0.91, 0.1), 2):
        flag = p >= t
        tp = int((flag & (y == 1)).sum()); fp = int((flag & (y == 0)).sum()); fn = int((~flag & (y == 1)).sum())
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        print(f'  {t:>7.2f} {prec:>9.3f} {rec:>7.3f} {f1:>6.3f} {tp:>4d} {fp:>4d} {fn:>4d} {int(flag.sum()):>7d}')
    for t in np.round(np.arange(0.05, 0.96, 0.01), 2):      # finer grid for the best-F1 pick
        flag = p >= t
        tp = (flag & (y == 1)).sum(); fp = (flag & (y == 0)).sum(); fn = (~flag & (y == 1)).sum()
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        best = max(best, (f1, float(t)))
    return best[1]


def threshold_for_recall(y, p, target):
    pos = np.sort(p[y == 1])[::-1]
    k = int(np.ceil(target * len(pos))) - 1
    return float(pos[max(k, 0)]) if len(pos) else 0.5


def learning_curve(model, X, y, groups, n_splits, fracs=(0.25, 0.5, 0.75, 1.0)):
    rng = np.random.RandomState(0)
    out = []
    for frac in fracs:
        aps = []
        for tr, te in make_splitter(groups, n_splits, seed=0).split(X, y, groups):
            if y[te].sum() == 0:
                continue
            keep = tr if frac >= 1.0 else np.concatenate([
                rng.choice(idx, max(2, int(round(frac * len(idx)))), replace=False)
                for idx in (tr[y[tr] == 0], tr[y[tr] == 1])])
            m = clone(model).fit(X.iloc[keep], y[keep])
            aps.append(average_precision_score(y[te], m.predict_proba(X.iloc[te])[:, 1]))
        out.append((frac, int(round(frac * len(y) * (n_splits - 1) / n_splits)), np.mean(aps)))
    return out


def cv_permutation_importance(model, X, y, groups, n_splits, top=12):
    imp, n = pd.Series(0.0, index=X.columns), 0
    for tr, te in make_splitter(groups, n_splits, seed=0).split(X, y, groups):
        if y[te].sum() == 0:
            continue
        m = clone(model).fit(X.iloc[tr], y[tr])
        r = permutation_importance(m, X.iloc[te], y[te], scoring='average_precision',
                                   n_repeats=5, random_state=0, n_jobs=-1)
        imp += r.importances_mean
        n += 1
    return (imp / max(n, 1)).sort_values(ascending=False).head(top)


# ── one target ─────────────────────────────────────────────────────────────
def run_target(target, X, y, groups, labels, types, kernels, args):
    base = y.mean()
    print(f'\n{"=" * 70}\nTARGET: {target}   positives {int(y.sum())}/{len(y)} ({base:.1%})\n{"=" * 70}')
    print(f'Baselines: always-"no" accuracy {1 - base:.3f};  random ranking PR-AUC {base:.3f}, ROC-AUC 0.500')

    print('\n--- Best single features (ROC-AUC of the raw value) ---')
    for c, auc, d in univariate_auc(X, y):
        print(f'  {c:40s} {auc:.3f}   ({d} = more {target})')

    models = make_models(y)
    oof, fold_auc, fold_ap = cross_validate(models, X, y, groups, args.folds, args.repeats)
    print(f'\n--- {args.repeats}x {args.folds}-fold CV ({"grouped" if groups is not None else "stratified"}) ---')
    cv = {}
    for name in models:
        cv[name] = {'roc_auc': float(np.mean(fold_auc[name])), 'roc_auc_std': float(np.std(fold_auc[name])),
                    'pr_auc': float(np.mean(fold_ap[name])), 'pr_auc_std': float(np.std(fold_ap[name]))}
        print(f"  {name:10s} ROC-AUC {cv[name]['roc_auc']:.3f} ± {cv[name]['roc_auc_std']:.3f}   "
              f"PR-AUC {cv[name]['pr_auc']:.3f} ± {cv[name]['pr_auc_std']:.3f}")
    best = max(cv, key=lambda n: cv[n]['pr_auc'])
    p = oof[best]
    print(f'  -> selected {best}')

    print(f'\n--- Precision / recall, out-of-fold ({best}) ---')
    t_f1 = threshold_table(y, p)
    t_rec = threshold_for_recall(y, p, args.target_recall)
    thr = t_rec if args.threshold_mode == 'recall' else t_f1
    flag = p >= thr
    print(f'  best-F1 threshold {t_f1:.2f};  recall>={args.target_recall:.0%} threshold {t_rec:.3f};  '
          f'using {args.threshold_mode}: {thr:.3f}')
    print(f'  -> catches {int((flag & (y == 1)).sum())}/{int(y.sum())}, '
          f'false alarms {int((flag & (y == 0)).sum())}/{int((y == 0).sum())}')

    if target == 'any' and len(types) > 1:
        print('\n--- "any" model: recall per retry type at this threshold ---')
        for t in types:
            m = labels[t].values == 1
            print(f'  {t:15s} n={m.sum():4d}  recall={flag[m].mean():.3f}  median P={np.median(p[m]):.3f}')

    if not args.brief:
        print(f'\n--- Learning curve ({best}, PR-AUC); still rising at 100% => more data helps ---')
        for frac, n_train, ap in learning_curve(models[best], X, y, groups, args.folds):
            print(f'  {frac:>4.0%} (~{n_train:4d} train kernels): {ap:.3f}')
        print(f'\n--- Permutation importance ({best}, PR-AUC drop) ---')
        for c, v in cv_permutation_importance(models[best], X, y, groups, args.folds).items():
            print(f'  {c:40s} {v:+.4f}')

    res = pd.DataFrame({'kernel': kernels, 'y': y, 'p': p})
    print('\n--- Most confident false positives ---')
    print(res[res.y == 0].nlargest(5, 'p')[['kernel', 'p']].to_string(index=False))
    print('--- Most confident false negatives ---')
    print(res[res.y == 1].nsmallest(5, 'p')[['kernel', 'p']].to_string(index=False))

    final = clone(models[best]).fit(X, y)
    meta = {'target': target, 'model': best, 'threshold': thr, 'threshold_mode': args.threshold_mode,
            'threshold_best_f1': t_f1, 'threshold_target_recall': t_rec,
            'n_pos': int(y.sum()), 'n': int(len(y)), 'base_rate': float(base), 'cv': cv}
    return final, meta, p, flag


# ── train all targets ──────────────────────────────────────────────────────
def train(args):
    print('Model backend:', 'xgboost' if HAVE_XGB else 'sklearn HistGradientBoosting (xgboost not installed)')
    feat, labels, types, groups = load_data(args.features_csv, args.after_csv,
                                            args.group_col, args.labels,
                                            args.label_cols)
    X, dropped = build_matrix(feat, drop_extra=[args.group_col] if args.group_col else [])
    if dropped:
        print(f'Dropped high-cardinality text columns: {dropped}')
    print(f'Features: {X.shape[1]}')

    targets = ['any'] + [t for t in types if t != 'any']
    if args.targets != 'all':
        wanted = [t.strip() for t in args.targets.split(',') if t.strip()]
        unknown = set(wanted) - set(targets)
        if unknown:
            sys.exit(f'Unknown --targets {sorted(unknown)}; available: {targets}')
        targets = wanted
    kept = [t for t in targets if labels[t].sum() >= args.min_positives]
    for t in set(targets) - set(kept):
        print(f'  ! skipping {t}: only {labels[t].sum()} positives (< --min-positives {args.min_positives})')

    os.makedirs(args.model_dir, exist_ok=True)
    preds = pd.DataFrame({'kernel': feat['kernel']})
    if args.group_col in feat.columns:
        preds[args.group_col] = feat[args.group_col]
    summary = {'feature_cols': list(X.columns), 'group_col': args.group_col,
               'backend': 'xgboost' if HAVE_XGB else 'sklearn', 'targets': {}}
    for t in kept:
        y = labels[t].values
        model, meta, p, flag = run_target(t, X, y, groups, labels, types, feat['kernel'].values, args)
        tdir = os.path.join(args.model_dir, t)
        os.makedirs(tdir, exist_ok=True)
        joblib.dump(model, os.path.join(tdir, 'model.joblib'))
        with open(os.path.join(tdir, 'meta.json'), 'w') as fh:
            json.dump(meta, fh, indent=2)
        summary['targets'][t] = meta
        preds[f'y_{t}'] = y
        preds[f'p_{t}'] = p
        preds[f'flag_{t}'] = flag.astype(int)

    with open(os.path.join(args.model_dir, 'summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)
    preds.to_csv(os.path.join(args.model_dir, 'cv_predictions.csv'), index=False)

    print(f'\n{"=" * 70}\nSUMMARY (out-of-fold, {args.repeats}x{args.folds}-fold CV)\n{"=" * 70}')
    print(f"  {'target':15s} {'pos':>5s} {'base':>6s} {'model':>9s} {'ROC-AUC':>8s} {'PR-AUC':>7s} {'thresh':>7s}")
    for t, m in summary['targets'].items():
        c = m['cv'][m['model']]
        print(f"  {t:15s} {m['n_pos']:>5d} {m['base_rate']:>6.3f} {m['model']:>9s} "
              f"{c['roc_auc']:>8.3f} {c['pr_auc']:>7.3f} {m['threshold']:>7.3f}")
    print(f'  (PR-AUC should be compared with "base"; ROC-AUC with 0.5)')
    print(f'\nSaved to {args.model_dir}/: <target>/model.joblib + meta.json for each target, '
          f'summary.json, cv_predictions.csv')


# ── predict ────────────────────────────────────────────────────────────────
def predict(args):
    with open(os.path.join(args.model_dir, 'summary.json')) as fh:
        summary = json.load(fh)
    df = pd.read_csv(args.predict, dtype={'kernel': str})
    X, _ = build_matrix(df, columns=summary['feature_cols'], drop_extra=[summary.get('group_col') or ''])
    out = pd.DataFrame({'kernel': df['kernel'] if 'kernel' in df else np.arange(len(df))})
    for t, meta in summary['targets'].items():
        model = joblib.load(os.path.join(args.model_dir, t, 'model.joblib'))
        out[f'p_{t}'] = model.predict_proba(X)[:, 1]
        out[f'flag_{t}'] = (out[f'p_{t}'] >= meta['threshold']).astype(int)
    type_cols = [t for t in summary['targets'] if t != 'any']
    out['predicted_types'] = out.apply(
        lambda r: ','.join(t for t in type_cols if r[f'flag_{t}']) or '-', axis=1)
    sort_col = 'p_any' if 'p_any' in out else out.columns[1]
    out_path = args.out or os.path.splitext(args.predict)[0] + '_retry_type_pred.csv'
    out.sort_values(sort_col, ascending=False).to_csv(out_path, index=False)
    print(f'Scored {len(out)} kernels -> {out_path}')
    for t in summary['targets']:
        print(f"  {t:15s} flagged {int(out[f'flag_{t}'].sum())}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('features_csv', nargs='?', default=os.path.expanduser('~/kernel_features.csv'))
    ap.add_argument('after_csv', nargs='?', default=os.path.expanduser('~/kernel_features_after.csv'),
                    help='one row per end dump; having a row is the label')
    ap.add_argument('--labels', help='CSV with a kernel column and 0/1 outcome '
                                     'columns; use instead of after_csv for any outcome')
    ap.add_argument('--label-cols', help='comma-separated columns of --labels to model '
                                         '(default: all non-kernel columns)')
    ap.add_argument('model_dir', nargs='?', default=os.path.expanduser('~/retry_type_models'))
    ap.add_argument('--group-col', default='func_name',
                    help="kernels with equal values stay in one CV fold (default func_name; '' disables)")
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--min-positives', type=int, default=15,
                    help='skip a retry type with fewer positive kernels than this (default 15)')
    ap.add_argument('--threshold-mode', choices=['recall', 'f1'], default='recall')
    ap.add_argument('--target-recall', type=float, default=0.9)
    ap.add_argument('--targets', default='any',
                    help="which models to train: 'any' = yes/no retry (default), 'all' = any + one per "
                         "retry type, or a comma list like 'any,overflow'")
    ap.add_argument('--brief', action='store_true', help='skip learning curves and permutation importance')
    ap.add_argument('--predict', metavar='CSV', help='score a features CSV with trained models')
    ap.add_argument('--model-dir', dest='model_dir_opt', help='model folder for --predict')
    ap.add_argument('--out', help='output CSV for --predict')
    args = ap.parse_args()
    if args.model_dir_opt:
        args.model_dir = args.model_dir_opt
    predict(args) if args.predict else train(args)


if __name__ == '__main__':
    main()
