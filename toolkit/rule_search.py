#!/usr/bin/env python3
"""
rule_search.py — find a readable threshold rule that predicts a retry type,
optionally one cutoff per group (e.g. per mix_mode), and emit it as C++.

This is the last step of the pipeline: a model tells you WHICH feature matters,
this turns that feature into a rule a cost model can evaluate.

  # one cutoff
  python3 rule_search.py features.csv after.csv --feature total_tensor_elements

  # one cutoff per mix_mode -- usually far fewer false alarms
  python3 rule_search.py features.csv after.csv --feature total_tensor_elements \
      --group-col mix_mode --target-recall 0.95 --list-kernels

  # rank every numeric feature by how good its best rule is, then stop
  python3 rule_search.py features.csv after.csv --rank-features

  # save the rule and generate C++
  python3 rule_search.py features.csv after.csv --feature total_tensor_elements \
      --group-col mix_mode --save-rule ub_rule.json --emit-cpp ub_rule.h

Exit code is 0 if a rule was found, 1 if the feature carries no usable signal.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

KEYWORDS = [('buffercount', 'buffercountbump'), ('vfstack', 'vfstacklimit'),
            ('stacklimit', 'vfstacklimit'), ('overflow', 'overflow')]


def retry_type(name: str) -> str:
    low = str(name).lower().replace('_', '').replace('-', '')
    return next((t for kw, t in KEYWORDS if kw in low), str(name))


def load(features_csv, after_csv, type_name, labels_csv=None,
         label_col=None, id_col='kernel'):
    """Features + a binary label.

    Two label sources:
      labels_csv  any CSV with an id column and a 0/1 (or true/false) column
      after_csv   one row per end dump; a kernel having a row IS the label
    """
    f = pd.read_csv(features_csv, dtype={id_col: str}).copy()
    if labels_csv:
        lab = pd.read_csv(labels_csv, dtype={id_col: str}).copy()
        if label_col is None:
            cands = [c for c in lab.columns if c != id_col]
            if len(cands) != 1:
                raise SystemExit(f'--label-col needed; {labels_csv} has {cands}')
            label_col = cands[0]
        m = lab.drop_duplicates(id_col).set_index(id_col)[label_col]
        y = f[id_col].map(m)
        f['y'] = y.map(lambda v: 0 if pd.isna(v) else
                       int(str(v).strip().lower() not in ('0', 'false', '', 'nan'))
                       ).astype(int)
        return f
    a = pd.read_csv(after_csv, dtype={id_col: str}).copy()
    if 'end_retry_type' in a.columns:
        a['t'] = a['end_retry_type'].astype(str)
    elif 'end_dump_file' in a.columns:
        a['t'] = a['end_dump_file'].map(retry_type)
    else:
        a['t'] = 'overflow'
    pos = set(a['kernel']) if type_name == 'any' else \
        set(a.loc[a.t == type_name, 'kernel'])
    f['y'] = f['kernel'].isin(pos).astype(int)
    return f


def best_cutoff(x: np.ndarray, y: np.ndarray, target_recall: float,
                round_down: bool = True):
    """Lowest cutoff that still catches `target_recall` of the positives."""
    pos = np.sort(x[y == 1])
    if len(pos) == 0:
        return None
    k = int(np.floor((1 - target_recall) * len(pos)))
    t = float(pos[min(k, len(pos) - 1)])
    if round_down and t > 0:                 # leave margin: exact values overfit
        mag = 10 ** int(np.floor(np.log10(t)))
        t = np.floor(t / (mag / 10)) * (mag / 10)
    return float(t)


def score(x, y, cutoff):
    flag = x >= cutoff
    tp = int((flag & (y == 1)).sum())
    fp = int((flag & (y == 0)).sum())
    fn = int((~flag & (y == 1)).sum())
    return {'cutoff': cutoff, 'tp': tp, 'fp': fp, 'fn': fn,
            'recall': tp / max(tp + fn, 1),
            'precision': tp / max(tp + fp, 1),
            'flagged': tp + fp}


def rank_features(f, top):
    from sklearn.metrics import roc_auc_score
    y = f['y'].values
    rows = []
    for c in f.columns:
        if c in ('kernel', 'y') or not pd.api.types.is_numeric_dtype(f[c]):  # noqa
            continue
        x = f[c].values
        if len(np.unique(x)) < 2:
            continue
        auc = roc_auc_score(y, x)
        direction = 'higher' if auc >= 0.5 else 'lower'
        auc = max(auc, 1 - auc)
        xs = x if direction == 'higher' else -x
        s = score(xs, y, best_cutoff(xs, y, 0.95, round_down=False) or 0)
        rows.append((c, auc, direction, s['tp'], s['fp']))
    rows.sort(key=lambda r: -r[1])
    if not rows:
        print('No usable feature: every numeric column is constant across these '
              'units, so nothing can distinguish them. Check that the features '
              'CSV really describes different kernels.')
        return rows
    print(f"{'feature':38s} {'AUC':>5s} {'dir':>7s} {'caught':>7s} {'false alarms':>13s}")
    for c, auc, d, tp, fp in rows[:top]:
        print(f'{c:38s} {auc:>5.3f} {d:>7s} {tp:>7d} {fp:>13d}')
    print('\nAUC 0.5 = useless, >0.8 = a rule on this feature is worth trying.')
    return rows


def camel(s: str, upper_first: bool = False) -> str:
    parts = [p for p in s.replace('-', '_').split('_') if p]
    head = parts[0].capitalize() if upper_first else parts[0].lower()
    return head + ''.join(p.capitalize() for p in parts[1:])


def emit_cpp(rule, path):
    feat = rule['feature']
    fn = camel(feat)                     # est_memory_bytes -> estMemoryBytes
    guard = 'MLIRFEAT_' + feat.upper() + '_RULE_H'
    lines = [f'// Generated by rule_search.py from {rule["n"]} kernels '
             f'({rule["n_pos"]} {rule["type"]}).',
             f'// recall {rule["overall"]["recall"]:.3f}, '
             f'precision {rule["overall"]["precision"]:.3f} on that data.',
             '// Features must be computed the same way as in training.',
             f'#ifndef {guard}', f'#define {guard}', '', '#include <cstdint>',
             '#include <string_view>', '', 'namespace kernel_heuristics {', '']
    if rule['group_col']:
        lines += [f'// {feat} cutoff, chosen per {rule["group_col"]}',
                  f'inline int64_t {fn}Cutoff(std::string_view {rule["group_col"]}) {{']
        for g, r in rule['groups'].items():
            lines.append(f'  if ({rule["group_col"]} == "{g}") return '
                         f'{int(r["cutoff"])}; // catches {r["tp"]}/{r["tp"] + r["fn"]}, '
                         f'{r["fp"]} false alarms')
        lines += [f'  return {int(rule["fallback_cutoff"])};  // unseen group',
                  '}', '',
                  f'inline bool willOverflow(int64_t {fn}, '
                  f'std::string_view {rule["group_col"]}) {{',
                  f'  return {fn} >= {fn}Cutoff({rule["group_col"]});', '}']
    else:
        c = int(rule['overall']['cutoff'])
        k = 'k' + camel(feat, upper_first=True) + 'Cutoff'
        lines += [f'inline constexpr int64_t {k} = {c};', '',
                  f'inline bool willOverflow(int64_t {fn}) {{',
                  f'  return {fn} >= {k};', '}']
    lines += ['', '}  // namespace kernel_heuristics', '', f'#endif  // {guard}', '']
    open(path, 'w').write('\n'.join(lines))
    print(f'\nWrote C++ rule to {path}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('features_csv', help='one row per unit, features known in advance')
    ap.add_argument('after_csv', nargs='?',
                    help='one row per end dump; having a row is the label. '
                         'Omit when using --labels.')
    ap.add_argument('--labels', help='CSV with an id column and a 0/1 outcome column; '
                                     'use instead of after_csv for any outcome')
    ap.add_argument('--label-col', help='which column of --labels is the outcome')
    ap.add_argument('--id-col', default='kernel', help='join column (default kernel)')
    ap.add_argument('--feature', default='total_tensor_elements')
    ap.add_argument('--type', default='overflow', help="retry type, or 'any'")
    ap.add_argument('--group-col', help='separate cutoff per value, e.g. mix_mode')
    ap.add_argument('--target-recall', type=float, default=0.95)
    ap.add_argument('--no-round', action='store_true',
                    help='use the exact smallest positive instead of a round number')
    ap.add_argument('--rank-features', action='store_true',
                    help='rank every numeric feature and exit')
    ap.add_argument('--top', type=int, default=15)
    ap.add_argument('--list-kernels', action='store_true',
                    help='print caught / missed / false-alarm kernel names')
    ap.add_argument('--save-rule', help='write the rule as JSON')
    ap.add_argument('--emit-cpp', help='write the rule as a C++ header')
    args = ap.parse_args(argv)

    if not args.after_csv and not args.labels:
        ap.error('give after_csv or --labels')
    f = load(args.features_csv, args.after_csv, args.type,
             args.labels, args.label_col, args.id_col)
    label_name = args.label_col or args.type
    n_pos = int(f['y'].sum())
    print(f'{len(f)} units, {n_pos} positive "{label_name}" '
          f'({n_pos / len(f):.1%})\n')
    if n_pos == 0:
        print(f'no positive examples for {label_name!r}', file=sys.stderr)
        return 1

    if args.rank_features:
        rank_features(f, args.top)
        return 0

    if args.feature not in f.columns:
        print(f'no column {args.feature!r}; try --rank-features', file=sys.stderr)
        return 1

    y = f['y'].values
    x = f[args.feature].values.astype(float)
    rnd = not args.no_round
    rule = {'feature': args.feature, 'type': label_name, 'group_col': args.group_col,
            'target_recall': args.target_recall, 'n': len(f), 'n_pos': n_pos,
            'groups': {}}

    overall_cut = best_cutoff(x, y, args.target_recall, rnd)
    print(f'single cutoff: {args.feature} >= {overall_cut:,.0f}')
    s = score(x, y, overall_cut)
    print(f'  caught {s["tp"]}/{n_pos}  false alarms {s["fp"]}  '
          f'precision {s["precision"]:.1%}\n')
    rule['fallback_cutoff'] = overall_cut

    if args.group_col:
        if args.group_col not in f.columns:
            print(f'no column {args.group_col!r}', file=sys.stderr)
            return 1
        print(f'per {args.group_col}:')
        flag = np.zeros(len(f), bool)
        for g, sub in f.groupby(args.group_col):
            xs, ys = sub[args.feature].values.astype(float), sub['y'].values
            cut = best_cutoff(xs, ys, args.target_recall, rnd)
            if cut is None:                       # no positives in this group
                cut = float('inf')
                print(f'  {g:10s} no {args.type} kernels -> never flag '
                      f'({len(sub)} kernels)')
            else:
                sg = score(xs, ys, cut)
                rule['groups'][str(g)] = sg
                print(f'  {g:10s} >= {cut:>14,.0f}   caught {sg["tp"]}/'
                      f'{sg["tp"] + sg["fn"]}   false alarms {sg["fp"]}')
            flag |= (f[args.group_col] == g).values & (x >= cut)
        tp = int((flag & (y == 1)).sum())
        fp = int((flag & (y == 0)).sum())
        rule['overall'] = {'cutoff': overall_cut, 'tp': tp, 'fp': fp,
                           'fn': n_pos - tp,
                           'recall': tp / max(n_pos, 1),
                           'precision': tp / max(tp + fp, 1), 'flagged': tp + fp}
        print(f'\ncombined: caught {tp}/{n_pos}  false alarms {fp}  '
              f'precision {tp / max(tp + fp, 1):.1%}')
        print(f'(single cutoff gave {s["tp"]}/{n_pos} with {s["fp"]} false alarms)')
    else:
        flag = x >= overall_cut
        rule['overall'] = s

    if args.list_kernels:
        cols = [c for c in (args.id_col, args.group_col, args.feature) if c]
        for label, mask in (('CAUGHT', flag & (y == 1)),
                            ('MISSED', ~flag & (y == 1)),
                            ('FALSE ALARMS', flag & (y == 0))):
            sub = f[mask].sort_values(args.feature, ascending=False)
            print(f'\n{label} ({len(sub)}):')
            if len(sub):
                print(sub[cols].to_string(index=False))

    if args.save_rule:
        json.dump(rule, open(args.save_rule, 'w'), indent=2)
        print(f'\nWrote {args.save_rule}')
    if args.emit_cpp:
        emit_cpp(rule, args.emit_cpp)

    return 0 if rule['overall']['tp'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
