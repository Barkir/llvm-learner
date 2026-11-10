#!/usr/bin/env python3
"""
rule_search.py — find a readable threshold rule that predicts a binary outcome,
optionally one cutoff per group, and emit it as C++.

Nothing here is specific to any one outcome: give it a features CSV and a
labels CSV and it finds `<feature> >= cutoff` (or `<=`, whichever way the
outcome lies) and reports exactly what that rule would do.

  # which feature carries the signal?
  python3 rule_search.py feats.csv --labels labels.csv --label-col regressed \
      --rank-features

  # build the rule on the best feature (picked automatically if --feature is omitted)
  python3 rule_search.py feats.csv --labels labels.csv --label-col regressed

  # one cutoff per group -- often far fewer false alarms
  python3 rule_search.py feats.csv --labels labels.csv --label-col regressed \
      --feature loop_max_depth --group-col struct_mix_mode --list-units

  # ship it
  python3 rule_search.py feats.csv --labels labels.csv --label-col regressed \
      --feature X --save-rule rule.json --emit-cpp rule.h

Labels come from either:
  --labels CSV    an id column plus a 0/1 (or true/false) outcome column
  after_csv       positional: one row per dump/event; having a row IS the label
                  (use --type to select one value of end_retry_type)

Exit code 0 if a rule was found, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import re
import sys

import numpy as np
import pandas as pd

# only used by the positional after_csv path
KEYWORDS = [('buffercount', 'buffercountbump'), ('vfstack', 'vfstacklimit'),
            ('stacklimit', 'vfstacklimit'), ('overflow', 'overflow')]
NON_FEATURE_COLS = {'y', 'file', 'applied_sets', 'backend', 'source', 'outdir'}


def retry_type(name: str) -> str:
    low = str(name).lower().replace('_', '').replace('-', '')
    return next((t for kw, t in KEYWORDS if kw in low), str(name))


def truthy(v) -> int:
    if pd.isna(v):
        return 0
    return int(str(v).strip().lower() not in ('0', 'false', '', 'nan', '0.0'))


def load(features_csv, after_csv=None, type_name='any', labels_csv=None,
         label_col=None, id_col='kernel'):
    f = pd.read_csv(features_csv, dtype={id_col: str}).copy()
    if id_col not in f.columns:
        raise SystemExit(f'{features_csv} has no id column {id_col!r}')

    if labels_csv:
        lab = pd.read_csv(labels_csv, dtype={id_col: str}).copy()
        if id_col not in lab.columns:
            raise SystemExit(f'{labels_csv} has no id column {id_col!r}')
        if label_col is None:
            cands = [c for c in lab.columns if c != id_col]
            if len(cands) != 1:
                raise SystemExit(f'--label-col needed; {labels_csv} has {cands}')
            label_col = cands[0]
        if label_col not in lab.columns:
            raise SystemExit(f'{labels_csv} has no column {label_col!r}; '
                             f'available: {[c for c in lab.columns if c != id_col]}')
        m = lab.drop_duplicates(id_col).set_index(id_col)[label_col]
        f['y'] = [truthy(v) for v in f[id_col].map(m)]
        return f, label_col

    a = pd.read_csv(after_csv, dtype={id_col: str}).copy()
    if 'end_retry_type' in a.columns:
        a['t'] = a['end_retry_type'].astype(str)
    elif 'end_dump_file' in a.columns:
        a['t'] = a['end_dump_file'].map(retry_type)
    else:
        a['t'] = 'any'
    pos = set(a[id_col]) if type_name == 'any' else set(a.loc[a.t == type_name, id_col])
    f['y'] = f[id_col].isin(pos).astype(int)
    return f, type_name


# ─────────────────────────────────────────────────────────────────────────
# rule fitting
# ─────────────────────────────────────────────────────────────────────────
def orient(x: np.ndarray, direction: str) -> np.ndarray:
    """Work internally with 'higher = more positive'; flip when it is not."""
    return -x if direction == 'lower' else x


def pick_direction(x: np.ndarray, y: np.ndarray) -> str:
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2 or len(np.unique(x)) < 2:
        return 'higher'
    return 'higher' if roc_auc_score(y, x) >= 0.5 else 'lower'


def best_cutoff(xo: np.ndarray, y: np.ndarray, target_recall: float,
                round_out: bool = True):
    """Loosest cutoff on oriented values that still catches `target_recall`."""
    pos = np.sort(xo[y == 1])
    if len(pos) == 0:
        return None
    k = int(np.floor((1 - target_recall) * len(pos)))
    t = float(pos[min(k, len(pos) - 1)])
    if round_out and t != 0 and np.isfinite(t):
        mag = 10 ** int(np.floor(np.log10(abs(t))))
        step = mag / 10
        t = np.floor(t / step) * step          # loosen: never lose a positive
    return float(t)


def score(xo, y, cutoff):
    flag = xo >= cutoff
    tp = int((flag & (y == 1)).sum())
    fp = int((flag & (y == 0)).sum())
    fn = int((~flag & (y == 1)).sum())
    return {'cutoff': cutoff, 'tp': tp, 'fp': fp, 'fn': fn,
            'recall': tp / max(tp + fn, 1),
            'precision': tp / max(tp + fp, 1), 'flagged': tp + fp}


def num(v: float, integral: bool) -> str:
    """Readable number: no scientific notation for ordinary magnitudes."""
    if integral or (abs(v) >= 1000 and float(v).is_integer()):
        return f'{v:,.0f}'
    return f'{v:,.0f}' if abs(v) >= 1000 else f'{v:,.4g}'


def display(rule_cut: float, direction: str) -> tuple[float, str]:
    """Internal oriented cutoff -> the value and operator a human/C++ sees."""
    return (-rule_cut, '<=') if direction == 'lower' else (rule_cut, '>=')


def feature_columns(f, id_col, group_col):
    skip = NON_FEATURE_COLS | {id_col} | ({group_col} if group_col else set())
    return [c for c in f.columns
            if c not in skip and pd.api.types.is_numeric_dtype(f[c])
            and f[c].nunique() > 1]


def rank_features(f, top, id_col, group_col, target_recall):
    from sklearn.metrics import roc_auc_score
    y = f['y'].values
    rows = []
    for c in feature_columns(f, id_col, group_col):
        x = f[c].values.astype(float)
        auc = roc_auc_score(y, x)
        d = 'higher' if auc >= 0.5 else 'lower'
        xo = orient(x, d)
        s = score(xo, y, best_cutoff(xo, y, target_recall, round_out=False) or 0)
        rows.append((c, max(auc, 1 - auc), d, s['tp'], s['fp']))
    rows.sort(key=lambda r: -r[1])
    if not rows:
        print('No usable feature: every numeric column is constant across these '
              'units, so nothing can distinguish them. Check that the features '
              'CSV really describes different units.')
        return rows
    print(f"{'feature':38s} {'AUC':>5s} {'positive when':>14s} "
          f"{'caught':>7s} {'false alarms':>13s}")
    for c, auc, d, tp, fp in rows[:top]:
        print(f'{c:38s} {auc:>5.3f} {d:>14s} {tp:>7d} {fp:>13d}')
    print('\nAUC 0.5 = useless, >0.8 = a rule on this feature is worth trying.')
    return rows


# ─────────────────────────────────────────────────────────────────────────
# C++
# ─────────────────────────────────────────────────────────────────────────
def camel(s: str, upper_first: bool = False) -> str:
    parts = [p for p in re.split(r'[^0-9A-Za-z]+', s) if p]
    if not parts:
        return 'value'
    head = parts[0].capitalize() if upper_first else parts[0].lower()
    return head + ''.join(p.capitalize() for p in parts[1:])


def emit_cpp(rule, path, func_name=None):
    feat, label = rule['feature'], rule['label']
    var = camel(feat)
    fn = func_name or ('predict' + camel(label, upper_first=True))
    ctype = 'int64_t' if rule['integral'] else 'double'

    def lit(v):
        return str(int(v)) if rule['integral'] else repr(float(v))

    guard = re.sub(r'[^0-9A-Z]', '_', f'{label}_{feat}_RULE_H'.upper())
    op = rule['operator']
    L = [f'// Generated by rule_search.py from {rule["n"]} units '
         f'({rule["n_pos"]} positive "{label}").',
         f'// On that data: recall {rule["overall"]["recall"]:.3f}, '
         f'precision {rule["overall"]["precision"]:.3f} '
         f'({rule["overall"]["tp"]} true / {rule["overall"]["flagged"]} flagged).',
         '// The feature must be computed exactly as it was for training.',
         f'#ifndef {guard}', f'#define {guard}', '', '#include <cstdint>']
    if rule['group_col']:
        L.append('#include <string_view>')
    L += ['', 'namespace kernel_heuristics {', '']

    if rule['group_col']:
        g = camel(rule['group_col'])
        L += [f'// {feat} cutoff, chosen per {rule["group_col"]}',
              f'inline {ctype} {var}Cutoff(std::string_view {g}) {{']
        for name, r in rule['groups'].items():
            cut, _ = display(r['cutoff'], rule['direction'])
            L.append(f'  if ({g} == "{name}") return {lit(cut)};'
                     f'  // catches {r["tp"]}/{r["tp"] + r["fn"]}, '
                     f'{r["fp"]} false alarms')
        fb, _ = display(rule['fallback_cutoff'], rule['direction'])
        L += [f'  return {lit(fb)};  // group not seen during fitting', '}', '',
              f'inline bool {fn}({ctype} {var}, std::string_view {g}) {{',
              f'  return {var} {op} {var}Cutoff({g});', '}']
    else:
        cut, _ = display(rule['overall']['cutoff'], rule['direction'])
        k = 'k' + camel(feat, upper_first=True) + 'Cutoff'
        L += [f'inline constexpr {ctype} {k} = {lit(cut)};', '',
              f'inline bool {fn}({ctype} {var}) {{',
              f'  return {var} {op} {k};', '}']
    L += ['', '}  // namespace kernel_heuristics', '', f'#endif  // {guard}', '']
    open(path, 'w').write('\n'.join(L))
    print(f'\nWrote C++ rule to {path}  ({fn}, {ctype})')


# ─────────────────────────────────────────────────────────────────────────
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('features_csv', help='one row per unit, features known in advance')
    ap.add_argument('after_csv', nargs='?',
                    help='one row per dump/event; having a row is the label')
    ap.add_argument('--labels', help='CSV with an id column and a 0/1 outcome column')
    ap.add_argument('--label-col', help='which column of --labels is the outcome')
    ap.add_argument('--id-col', default='kernel', help='join column (default kernel)')
    ap.add_argument('--type', default='any',
                    help="with after_csv: which end_retry_type counts as positive "
                         "(default 'any' = any row)")
    ap.add_argument('--feature', help='feature to threshold (default: best ranked)')
    ap.add_argument('--direction', choices=['auto', 'higher', 'lower'], default='auto',
                    help='is the outcome more likely at higher or lower values? '
                         'auto decides from the data')
    ap.add_argument('--group-col', help='fit a separate cutoff per value of this column')
    ap.add_argument('--target-recall', type=float, default=0.95)
    ap.add_argument('--no-round', action='store_true',
                    help='use the exact extreme positive instead of a round number')
    ap.add_argument('--rank-features', action='store_true',
                    help='rank every numeric feature and exit')
    ap.add_argument('--top', type=int, default=15)
    ap.add_argument('--list-units', '--list-kernels', dest='list_units',
                    action='store_true', help='print caught / missed / false-alarm ids')
    ap.add_argument('--save-rule', help='write the rule as JSON')
    ap.add_argument('--emit-cpp', help='write the rule as a C++ header')
    ap.add_argument('--cpp-func', help='name of the generated C++ predicate')
    args = ap.parse_args(argv)

    if not args.after_csv and not args.labels:
        ap.error('give after_csv or --labels')
    f, label = load(args.features_csv, args.after_csv, args.type,
                    args.labels, args.label_col, args.id_col)
    y = f['y'].values
    n_pos = int(y.sum())
    print(f'{len(f)} units, {n_pos} positive "{label}" ({n_pos / len(f):.1%})\n')
    if n_pos == 0:
        print(f'no positive examples for {label!r}', file=sys.stderr)
        return 1
    if n_pos == len(f):
        print('every unit is positive; there is nothing to separate', file=sys.stderr)
        return 1

    if args.rank_features:
        rank_features(f, args.top, args.id_col, args.group_col, args.target_recall)
        return 0

    feature = args.feature
    if feature is None:
        ranked = rank_features(f, 3, args.id_col, args.group_col, args.target_recall)
        if not ranked:
            return 1
        feature = ranked[0][0]
        print(f'\nNo --feature given; using the best ranked: {feature}\n')
    if feature not in f.columns:
        print(f'no column {feature!r}; try --rank-features', file=sys.stderr)
        return 1

    x = f[feature].values.astype(float)
    direction = args.direction if args.direction != 'auto' else pick_direction(x, y)
    xo = orient(x, direction)
    integral = bool(np.all(np.isfinite(x)) and np.all(x == np.floor(x)))
    rnd = not args.no_round

    rule = {'feature': feature, 'label': label, 'direction': direction,
            'operator': '<=' if direction == 'lower' else '>=',
            'integral': integral, 'group_col': args.group_col,
            'target_recall': args.target_recall, 'n': len(f), 'n_pos': n_pos,
            'groups': {}}

    overall_cut = best_cutoff(xo, y, args.target_recall, rnd)
    s = score(xo, y, overall_cut)
    shown, op = display(overall_cut, direction)
    print(f'single cutoff: {feature} {op} {num(shown, integral)}')
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
            xs = orient(sub[feature].values.astype(float), direction)
            ys = sub['y'].values
            cut = best_cutoff(xs, ys, args.target_recall, rnd)
            if cut is None:
                print(f'  {str(g):12s} no positive units -> never flag '
                      f'({len(sub)} units)')
                continue
            sg = score(xs, ys, cut)
            rule['groups'][str(g)] = sg
            sh, _ = display(cut, direction)
            print(f'  {str(g):12s} {op} {num(sh, integral):>14s}   caught '
                  f'{sg["tp"]}/{sg["tp"] + sg["fn"]}   false alarms {sg["fp"]}')
            flag |= (f[args.group_col].astype(str) == str(g)).values & (xo >= cut)
        tp = int((flag & (y == 1)).sum())
        fp = int((flag & (y == 0)).sum())
        rule['overall'] = {'cutoff': overall_cut, 'tp': tp, 'fp': fp,
                           'fn': n_pos - tp, 'recall': tp / max(n_pos, 1),
                           'precision': tp / max(tp + fp, 1), 'flagged': tp + fp}
        print(f'\ncombined: caught {tp}/{n_pos}  false alarms {fp}  '
              f'precision {tp / max(tp + fp, 1):.1%}')
        print(f'(single cutoff gave {s["tp"]}/{n_pos} with {s["fp"]} false alarms; '
              f'keep the grouped rule only if it is better)')
    else:
        flag = xo >= overall_cut
        rule['overall'] = s

    if args.list_units:
        cols = [c for c in (args.id_col, args.group_col, feature) if c]
        for name, mask in (('CAUGHT', flag & (y == 1)),
                           ('MISSED', ~flag & (y == 1)),
                           ('FALSE ALARMS', flag & (y == 0))):
            sub = f[mask].sort_values(feature, ascending=(direction == 'lower'))
            print(f'\n{name} ({len(sub)}):')
            if len(sub):
                print(sub[cols].to_string(index=False))

    if args.save_rule:
        json.dump(rule, open(args.save_rule, 'w'), indent=2)
        print(f'\nWrote {args.save_rule}')
    if args.emit_cpp:
        emit_cpp(rule, args.emit_cpp, args.cpp_func)
    return 0 if rule['overall']['tp'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
