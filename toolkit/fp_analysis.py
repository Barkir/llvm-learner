#!/usr/bin/env python3
"""
fp_analysis.py — How to cut false positives of the overflow rule
                 `<feature> >= threshold`.

1. Trade-off table: caught overflows vs false alarms for a range of cutoffs
2. Among the kernels the rule flags (true overflows vs false alarms), which
   other features separate the two groups best
3. Two-feature rules: keep the main rule, add one more condition
   (feature >= x or feature <= x) and report how many false alarms that removes
   while keeping (almost) all overflows

Usage:
  python3 fp_analysis.py [kernel_features.csv] [kernel_features_after.csv]
                         [--feature total_tensor_elements] [--threshold 425044]
                         [--type overflow] [--keep 1.0]
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

DTYPE_BYTES = {'f64': 8, 'f32': 4, 'f16': 2, 'bf16': 2, 'i64': 8, 'i32': 4, 'i16': 2, 'i8': 1}
KEYWORDS = [('buffercount', 'buffercountbump'), ('vfstack', 'vfstacklimit'),
            ('stacklimit', 'vfstacklimit'), ('overflow', 'overflow')]


def retry_type(name):
    low = str(name).lower().replace('_', '').replace('-', '')
    return next((t for kw, t in KEYWORDS if kw in low), str(name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('features_csv', nargs='?', default='kernel_features.csv')
    ap.add_argument('after_csv', nargs='?', default='kernel_features_after.csv')
    ap.add_argument('--feature', default='total_tensor_elements')
    ap.add_argument('--threshold', type=float, default=None,
                    help='main rule cutoff (default: smallest value that catches 95%% of overflows)')
    ap.add_argument('--type', default='overflow', help="retry type to analyse ('any' = all types)")
    ap.add_argument('--keep', type=float, default=1.0,
                    help='fraction of the flagged overflows the 2nd condition must keep (default 1.0 = all)')
    args = ap.parse_args()

    f = pd.read_csv(args.features_csv, dtype={'kernel': str})
    a = pd.read_csv(args.after_csv, dtype={'kernel': str})
    if 'end_retry_type' in a.columns:
        a['t'] = a['end_retry_type'].astype(str)
    elif 'end_dump_file' in a.columns:
        a['t'] = a['end_dump_file'].map(retry_type)
    else:
        a['t'] = 'overflow'
    pos = set(a['kernel']) if args.type == 'any' else set(a.loc[a.t == args.type, 'kernel'])
    f['y'] = f['kernel'].isin(pos).astype(int)

    # extra candidate features: size in bytes rather than elements
    if 'primary_dtype' in f.columns:
        b = f['primary_dtype'].map(DTYPE_BYTES).fillna(4)
        for c in ('total_tensor_elements', 'max_tensor_elements'):
            if c in f.columns:
                f[c.replace('elements', 'bytes_primary')] = f[c] * b
    if {'total_tensor_elements', 'n_tensors'} <= set(f.columns):
        f['mean_tensor_elements'] = f['total_tensor_elements'] / f['n_tensors'].clip(lower=1)
    if {'max_tensor_elements', 'total_tensor_elements'} <= set(f.columns):
        f['max_share_of_total'] = f['max_tensor_elements'] / f['total_tensor_elements'].clip(lower=1)

    y = f['y'].values
    x = f[args.feature].values
    print(f'{len(f)} kernels, {y.sum()} positives ({args.type}), main feature: {args.feature}')

    # ── 1. trade-off for the main feature ──
    print(f'\n--- 1. Cutoff trade-off for {args.feature} ---')
    print(f"  {'cutoff':>14s} {'caught':>8s} {'missed':>7s} {'false alarms':>13s} {'precision':>10s}")
    cuts = np.unique(np.concatenate([x[y == 1], np.quantile(x, np.linspace(0.5, 0.99, 12))]))
    rows = []
    for t in cuts:
        flag = x >= t
        tp, fp = int((flag & (y == 1)).sum()), int((flag & (y == 0)).sum())
        rows.append((t, tp, int(y.sum()) - tp, fp, tp / max(tp + fp, 1)))
    seen = set()
    for t, tp, fn, fp, prec in sorted(rows, key=lambda r: r[0]):
        if (tp, fp) in seen or tp == 0:
            continue
        seen.add((tp, fp))
        print(f'  {t:>14,.0f} {tp:>8d} {fn:>7d} {fp:>13d} {prec:>10.1%}')

    thr = args.threshold
    if thr is None:
        thr = float(np.quantile(x[y == 1], 0.05, method='lower'))
    flag = x >= thr
    sub = f[flag].copy()
    n_tp, n_fp = int(sub.y.sum()), int((sub.y == 0).sum())
    print(f'\nMain rule: {args.feature} >= {thr:,.0f}  ->  {n_tp} caught, {n_fp} false alarms')
    if n_tp < 2 or n_fp < 2:
        print('Too few kernels in one group to compare.')
        return

    # ── 2. what separates true overflows from false alarms among flagged kernels ──
    num = [c for c in sub.columns if c not in ('kernel', 'y', args.feature)
           and pd.api.types.is_numeric_dtype(sub[c]) and sub[c].nunique() > 1]
    res = []
    for c in num:
        auc = roc_auc_score(sub.y, sub[c])
        res.append((c, max(auc, 1 - auc), 'higher' if auc >= 0.5 else 'lower',
                    sub.loc[sub.y == 1, c].median(), sub.loc[sub.y == 0, c].median()))
    res.sort(key=lambda r: -r[1])
    print(f'\n--- 2. Among the {len(sub)} flagged kernels: features that separate '
          f'real overflows ({n_tp}) from false alarms ({n_fp}) ---')
    print(f"  {'feature':32s} {'AUC':>5s}  {'overflow is':>11s} {'median overflow':>16s} {'median false alarm':>19s}")
    for c, auc, d, m1, m0 in res[:12]:
        print(f'  {c:32s} {auc:>5.2f}  {d:>11s} {m1:>16,.2f} {m0:>19,.2f}')
    for c in ('primary_dtype', 'mix_mode', 'parallel_mode'):
        if c in sub.columns:
            ct = pd.crosstab(sub[c], sub.y.map({1: 'overflow', 0: 'false_alarm'}))
            print(f'\n  {c}:\n    ' + ct.to_string().replace('\n', '\n    '))

    # ── 3. best second condition ──
    need = int(np.ceil(args.keep * n_tp))
    print(f'\n--- 3. Second condition that keeps >= {need}/{n_tp} overflows and removes the most false alarms ---')
    cands = []
    for c in num:
        v, yy = sub[c].values, sub.y.values
        for t in np.unique(v):
            for op, keep in (('>=', v >= t), ('<=', v <= t)):
                tp = int((keep & (yy == 1)).sum())
                if tp >= need:
                    fp = int((keep & (yy == 0)).sum())
                    cands.append((fp, -tp, c, op, t))
    best = {}
    for fp, ntp, c, op, t in sorted(cands):
        if c not in best:
            best[c] = (fp, -ntp, op, t)
    ranked = sorted(best.items(), key=lambda kv: (kv[1][0], -kv[1][1]))[:10]
    print(f"  {'rule added to main rule':48s} {'caught':>7s} {'false alarms':>13s} {'removed':>8s}")
    for c, (fp, tp, op, t) in ranked:
        print(f'  AND {c} {op} {t:<,.2f}'.ljust(50) + f'{tp:>7d} {fp:>13d} {n_fp - fp:>8d}')

    print('\nCaution: with this few overflows, a second cutoff tuned on the same kernels looks better '
          'than it will on new ones. Prefer conditions that remove many false alarms with a margin '
          '(not exactly at one kernel), and that make physical sense.')

    out = 'flagged_kernels.csv'
    sub.assign(group=np.where(sub.y == 1, 'overflow', 'false_alarm')) \
       .sort_values(['group', args.feature], ascending=[False, False]).to_csv(out, index=False)
    print(f'\nFlagged kernels with all features written to {out} (open the false alarms next to the overflows).')


if __name__ == '__main__':
    main()
