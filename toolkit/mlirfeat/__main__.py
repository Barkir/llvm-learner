"""
CLI: extract features from a tree of kernels into one CSV.

  python3 -m mlirfeat ~/COLLECTED -o kernel_features.csv
  python3 -m mlirfeat ~/COLLECTED --file-name kernel.mlir -o input_features.csv
  python3 -m mlirfeat ~/COLLECTED --pattern '*overflow*.mlir' --column-prefix end_
  python3 -m mlirfeat one_kernel.mlir --print
  python3 -m mlirfeat --list

Layout: <root>/<kernel hash>/<file>.mlir, so the 'kernel' column joins with
retry_data.csv. A single .mlir path also works.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import sys

from .featureset import Extractor, FeatureSet
from .ir import parse_file


def find_inputs(root: str, pattern: str, file_name: str | None):
    if os.path.isfile(root):
        kernel = os.path.basename(os.path.dirname(os.path.abspath(root)))
        return [(kernel, root)]
    jobs = []
    for d in sorted(os.listdir(root)):
        kdir = os.path.join(root, d)
        if not os.path.isdir(kdir):
            continue
        for fn in sorted(os.listdir(kdir)):
            path = os.path.join(kdir, fn)
            if not os.path.isfile(path):
                continue
            if file_name is not None:
                if fn == file_name:
                    jobs.append((d, path))
            elif fnmatch.fnmatch(fn, pattern):
                jobs.append((d, path))
    return jobs


def main(argv=None):
    ap = argparse.ArgumentParser(prog='mlirfeat', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('root', nargs='?', help='directory of <hash>/<file>.mlir, or one file')
    ap.add_argument('-o', '--output', default='features.csv')
    ap.add_argument('--pattern', default='*.mlir', help="default '*.mlir'")
    ap.add_argument('--file-name', help='exact file name inside each kernel folder')
    ap.add_argument('--only', help='comma-separated feature sets to run')
    ap.add_argument('--skip', help='comma-separated feature sets to skip')
    ap.add_argument('--column-prefix', default='', help="e.g. 'end_' for end-of-pipeline dumps")
    ap.add_argument('--no-bindings', action='store_true', help='force the text parser')
    ap.add_argument('--strict', action='store_true', help='stop on the first failing set')
    ap.add_argument('--print', dest='show', action='store_true', help='print one row instead of writing CSV')
    ap.add_argument('--list', action='store_true', help='list registered feature sets and exit')
    args = ap.parse_args(argv)

    ex = Extractor(only=set(args.only.split(',')) if args.only else None,
                   skip=set(args.skip.split(',')) if args.skip else None)
    if args.list:
        print(ex.describe())
        print(f'\n{len(ex.columns) - 2} feature columns')
        return 0
    if not args.root:
        ap.error('root is required (or use --list)')

    jobs = find_inputs(args.root, args.pattern, args.file_name)
    if not jobs:
        print(f'no files matched in {args.root}', file=sys.stderr)
        return 1
    print(f'Extracting from {len(jobs)} file(s) with '
          f'{len(ex.sets)} feature sets...', file=sys.stderr)

    rows = []
    for kernel, path in jobs:
        module = parse_file(path, prefer_bindings=not args.no_bindings)
        row = {'kernel': kernel, 'file': os.path.basename(path)}
        row.update(ex.extract(module, strict=args.strict))
        rows.append(row)

    if args.show:
        for k, v in rows[0].items():
            print(f'  {k:42s} {v}')
        return 0

    if args.column_prefix:
        p = args.column_prefix
        rows = [{(k if k in ('kernel',) else p + k): v for k, v in r.items()}
                for r in rows]
        cols = ['kernel'] + [p + c for c in ['file'] + ex.columns]
    else:
        cols = ['kernel', 'file'] + ex.columns

    import csv
    with open(args.output, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols, restval=0, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print(f'Wrote {len(rows)} rows x {len(cols) - 2} features to {args.output}',
          file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
