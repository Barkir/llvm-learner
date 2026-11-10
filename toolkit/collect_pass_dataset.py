#!/usr/bin/env python3
"""
collect_pass_dataset.py — find which kernels a given MLIR pass touches, and
collect the before/after IR for each of them.

For every input .mlir it runs your compiler with

    --mlir-print-ir-before=<pass> --mlir-print-ir-after=<pass>
    [--mlir-print-ir-after-change]      # only print After if the IR changed
    [--mlir-print-ir-module-scope --mlir-disable-threading]

parses the IR dumps out of stderr, and writes the kernels where the pass
actually ran into <out>/<kernel>/{before_000.mlir, after_000.mlir, meta.json},
plus one manifest.csv for the whole run. Inputs are processed in parallel.

Examples
--------
  # which kernels does the pass run on, and what does it do to them
  python3 collect_pass_dataset.py \
      --cmd 'hivm-opt {input} --my-pipeline' \
      --pass optimize-ave --inputs ~/KERNELS --out ~/DATA/ave -j 16

  # you only know the C++ class name: find the registered CLI name first
  python3 collect_pass_dataset.py --cmd 'hivm-opt {input}' --list-passes ave

  # ...or skip the lookup: dump every pass and filter by name in Python
  python3 collect_pass_dataset.py --cmd 'hivm-opt {input} --my-pipeline' \
      --pass OptimizeAvePass --select all --inputs ~/KERNELS --out ~/DATA/ave

  # just the kernels where the pass CHANGED the IR
  ... --changed-only

Notes
-----
* --mlir-print-ir-before/after take the pass's *registered CLI argument*
  (`optimize-ave`), not its C++ class name (`OptimizeAvePass`). If you only
  have the class name use `--select all`, which asks for every pass and filters
  here; names are matched loosely (case, '-', '_' and a trailing 'Pass' are
  ignored).
* The dumps go to stderr, so your compiler's normal stdout output is untouched.
* A pass nested under a function runs once per function, so a kernel can have
  several before/after pairs; all are kept and counted in the manifest.
* --mlir-print-ir-module-scope requires threading off inside the compiler;
  --module-scope adds --mlir-disable-threading for you. That is the compiler's
  own threading, unrelated to -j here, which runs separate processes.
* Recent MLIR also has --mlir-print-ir-tree-dir=<dir>, where the compiler writes
  the dumps itself instead of printing to stderr. If your build has it, pass it
  with --extra-flag for very large runs; this script still records which kernels
  the pass ran on, from the stderr it does see.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# A dump banner, in either of the two formats MLIR has used:
#   // -----// IR Dump Before CSE (cse) //----- //
#   *** IR Dump After CSE ***  ('func.func' operation: @bar)
RE_BANNER = re.compile(r'IR Dump (Before|After)\s+(.+?)\s*(?:\*\*\*|//-+\s*//|$)')
RE_PASS_ARG = re.compile(r'\(([A-Za-z0-9][\w\-]*)\)')
RE_SCOPE = re.compile(r"'([\w.]+)' operation:\s*(@?[\w$.]+)")
RE_HELP_PASS = re.compile(r'^\s+--?([\w\-]+)\s+-\s+(.*)$')


def norm(name: str) -> str:
    """OptimizeAvePass / optimize-ave / Optimize_Ave  ->  optimizeave"""
    n = name.strip().lower().replace('-', '').replace('_', '').replace(' ', '')
    return n[:-4] if n.endswith('pass') else n


# ─────────────────────────────────────────────────────────────────────────
# parsing the dumps out of stderr
# ─────────────────────────────────────────────────────────────────────────
class Dump:
    __slots__ = ('phase', 'pass_name', 'pass_arg', 'scope_op', 'scope_sym', 'body')

    def __init__(self, phase, pass_name, pass_arg, scope_op, scope_sym):
        self.phase = phase              # 'Before' | 'After'
        self.pass_name = pass_name      # as printed, e.g. 'CSE' or 'OptimizeAve'
        self.pass_arg = pass_arg        # '(cse)' if present
        self.scope_op = scope_op        # 'func.func' if printed
        self.scope_sym = scope_sym      # '@bar'
        self.body: list[str] = []

    @property
    def text(self) -> str:
        return '\n'.join(self.body).strip() + '\n'

    def matches(self, wanted: str) -> bool:
        w = norm(wanted)
        return w in (norm(self.pass_name), norm(self.pass_arg or ''))

    def __repr__(self):
        return f'<Dump {self.phase} {self.pass_name} {self.scope_sym or ""}>'


def parse_dumps(stderr_text: str) -> list[Dump]:
    dumps: list[Dump] = []
    cur: Dump | None = None
    for line in stderr_text.split('\n'):
        if 'IR Dump Before' in line or 'IR Dump After' in line:
            m = RE_BANNER.search(line)
            if m:
                phase, rest = m.group(1), m.group(2)
                ma = RE_PASS_ARG.search(rest)
                ms = RE_SCOPE.search(line)
                name = re.split(r'[(\s]', rest.strip(), maxsplit=1)[0]
                cur = Dump(phase, name, ma.group(1) if ma else None,
                           ms.group(1) if ms else None, ms.group(2) if ms else None)
                dumps.append(cur)
                continue
        if cur is not None:
            cur.body.append(line)
    return dumps


# ─────────────────────────────────────────────────────────────────────────
# running one input
# ─────────────────────────────────────────────────────────────────────────
class Runner:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.done = 0

    def flags(self) -> list[str]:
        a = self.args
        if a.select == 'all':
            f = ['--mlir-print-ir-before-all', '--mlir-print-ir-after-all']
        else:
            arg = a.pass_arg or a.pass_name
            f = [f'--mlir-print-ir-before={arg}', f'--mlir-print-ir-after={arg}']
        if a.changed_only:
            f.append('--mlir-print-ir-after-change')
        if a.module_scope:
            f += ['--mlir-print-ir-module-scope', '--mlir-disable-threading']
        f += a.extra_flag
        return f

    def command(self, path: str) -> list[str] | str:
        tmpl = self.args.cmd
        cmd = tmpl.replace('{input}', shlex.quote(path)) if '{input}' in tmpl \
            else f'{tmpl} {shlex.quote(path)}'
        cmd += ' ' + ' '.join(self.flags())
        return cmd if self.args.shell else shlex.split(cmd)

    def kernel_id(self, path: str) -> str:
        mode = self.args.id_mode
        if mode == 'parent':
            return os.path.basename(os.path.dirname(os.path.abspath(path)))
        if mode == 'hash':
            return hashlib.sha1(os.path.abspath(path).encode()).hexdigest()[:16]
        rel = os.path.relpath(path, self.args.inputs
                              if os.path.isdir(self.args.inputs) else '.')
        return re.sub(r'[^\w.-]', '_', os.path.splitext(rel)[0])

    def run_one(self, path: str) -> dict:
        a = self.args
        kid = self.kernel_id(path)
        outdir = os.path.join(a.out, kid)
        row = {'kernel': kid, 'source': path, 'ran': 0, 'changed': 0,
               'n_before': 0, 'n_after': 0, 'scopes': '', 'exit_code': '',
               'seconds': 0.0, 'status': '', 'outdir': ''}

        meta_path = os.path.join(outdir, 'meta.json')
        if a.resume and os.path.exists(meta_path):
            # reuse what the earlier run recorded, so the manifest stays complete
            try:
                old = json.load(open(meta_path))
                row.update({k: old[k] for k in row if k in old})
            except Exception:
                pass
            row['status'] = 'skipped (resume)'
            row['outdir'] = outdir
            return row

        cmd = self.command(path)
        t0 = time.time()
        try:
            p = subprocess.run(cmd, shell=a.shell, capture_output=True, text=True,
                               timeout=a.timeout, errors='replace')
            out, err, code = p.stdout, p.stderr, p.returncode
        except subprocess.TimeoutExpired as e:
            out = e.stdout or ''
            err = (e.stderr or '') if isinstance(e.stderr, str) else ''
            code, row['status'] = None, f'timeout after {a.timeout}s'
        except FileNotFoundError as e:
            row['status'] = f'command not found: {e.filename}'
            return row
        row['seconds'] = round(time.time() - t0, 2)
        row['exit_code'] = '' if code is None else code

        dumps = [d for d in parse_dumps(err) if d.matches(a.pass_name)]
        before = [d for d in dumps if d.phase == 'Before']
        after = [d for d in dumps if d.phase == 'After']
        row['n_before'] = len(before)
        row['n_after'] = len(after)
        row['ran'] = int(bool(dumps))
        # with --mlir-print-ir-after-change an After dump only appears on change;
        # otherwise compare the text
        if a.changed_only:
            row['changed'] = int(bool(after))
        else:
            row['changed'] = int(any(
                b.text != c.text for b, c in zip(before, after))) or int(
                len(before) != len(after))
        row['scopes'] = ';'.join(sorted({d.scope_sym for d in dumps if d.scope_sym}))
        if not row['status']:
            row['status'] = 'ok' if code == 0 else 'compiler error'

        keep = row['ran'] and (row['changed'] or not a.changed_only)
        if keep or a.keep_all:
            os.makedirs(outdir, exist_ok=True)
            row['outdir'] = outdir
            for i, d in enumerate(before):
                self._write(os.path.join(outdir, f'before_{i:03d}.mlir'), d.text)
            for i, d in enumerate(after):
                self._write(os.path.join(outdir, f'after_{i:03d}.mlir'), d.text)
            if a.copy_input:
                shutil.copyfile(path, os.path.join(outdir, 'input.mlir'))
            if a.save_stderr:
                self._write(os.path.join(outdir, 'stderr.log'), err)
                self._write(os.path.join(outdir, 'stdout.log'), out)
            meta = dict(row)
            meta['command'] = cmd if isinstance(cmd, str) else ' '.join(cmd)
            meta['pass_query'] = a.pass_name
            meta['dumps'] = [{'phase': d.phase, 'pass': d.pass_name,
                              'arg': d.pass_arg, 'scope_op': d.scope_op,
                              'scope_sym': d.scope_sym, 'lines': len(d.body)}
                             for d in dumps]
            self._write(os.path.join(outdir, 'meta.json'),
                        json.dumps(meta, indent=2))

        with self.lock:
            self.done += 1
            if self.done % a.progress_every == 0 or a.verbose:
                mark = '+' if row['ran'] else '.'
                print(f'  [{self.done}] {mark} {kid} {row["status"]} '
                      f'({row["seconds"]}s)', file=sys.stderr)
        return row


    @staticmethod
    def _write(path: str, text: str):
        with open(path, 'w') as fh:
            fh.write(text)


# ─────────────────────────────────────────────────────────────────────────
# driver
# ─────────────────────────────────────────────────────────────────────────
def find_inputs(root: str, pattern: str, limit: int | None) -> list[str]:
    if os.path.isfile(root):
        paths = ([root] if root.endswith('.mlir')
                 else [l.strip() for l in open(root) if l.strip()])
    else:
        import fnmatch
        paths = []
        for dirpath, _, files in os.walk(root):
            for fn in sorted(files):
                if fnmatch.fnmatch(fn, pattern):
                    paths.append(os.path.join(dirpath, fn))
        paths.sort()
    return paths[:limit] if limit else paths


def list_passes(cmd: str, needle: str, shell: bool):
    base = cmd.replace('{input}', '').strip()
    argv = f'{base} --help-list-hidden'
    try:
        p = subprocess.run(argv if shell else shlex.split(argv), shell=shell,
                           capture_output=True, text=True, timeout=120)
    except Exception as e:
        print(f'could not run the tool: {e}', file=sys.stderr)
        return 1
    text = p.stdout + p.stderr
    n = norm(needle)
    hits = [l for l in text.split('\n') if n in norm(l)]
    if not hits:
        print(f'no registered pass matching {needle!r}. Try --select all, which '
              f'does not need the CLI name.', file=sys.stderr)
        return 1
    print(f'passes matching {needle!r} (use the --flag name with --pass-arg):')
    for l in hits:
        print(' ', l.strip())
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cmd', required=True,
                    help="compiler command; '{input}' is replaced by the file "
                         "(appended if absent), e.g. 'hivm-opt {input} --pipeline=x'")
    ap.add_argument('--pass', dest='pass_name',
                    help='pass to look for; class or CLI name (OptimizeAvePass, optimize-ave)')
    ap.add_argument('--pass-arg',
                    help='exact registered CLI name for the --mlir-print-ir-* flags, '
                         'when it differs from --pass')
    ap.add_argument('--inputs', help='directory to walk, a .mlir file, or a file listing paths')
    ap.add_argument('--out', help='output directory for the dataset')
    ap.add_argument('--pattern', default='*.mlir')
    ap.add_argument('--id-mode', choices=['path', 'parent', 'hash'], default='path',
                    help="how to name each output folder: 'parent' uses the "
                         "containing directory (right for <root>/<hash>/kernel.mlir)")
    ap.add_argument('-j', '--jobs', type=int, default=os.cpu_count() or 4)
    ap.add_argument('--timeout', type=float, default=300, help='seconds per kernel')
    ap.add_argument('--limit', type=int, help='only the first N inputs')
    ap.add_argument('--select', choices=['pass', 'all'], default='pass',
                    help="'pass' uses --mlir-print-ir-before/after=<name> (needs the "
                         "CLI name); 'all' dumps every pass and filters here")
    ap.add_argument('--changed-only', action='store_true',
                    help='add --mlir-print-ir-after-change and keep only kernels it changed')
    ap.add_argument('--module-scope', action='store_true',
                    help='print the whole module (adds --mlir-disable-threading)')
    ap.add_argument('--extra-flag', action='append', default=[],
                    help='extra compiler flag, repeatable')
    ap.add_argument('--keep-all', action='store_true',
                    help='write a folder for every kernel, not only matches')
    ap.add_argument('--copy-input', action='store_true', help='copy the input next to the dumps')
    ap.add_argument('--save-stderr', action='store_true', help='keep raw stdout/stderr')
    ap.add_argument('--resume', action='store_true', help='skip kernels already done')
    ap.add_argument('--shell', action='store_true', help='run --cmd through a shell')
    ap.add_argument('--dry-run', action='store_true', help='print the first command and exit')
    ap.add_argument('--list-passes', metavar='SUBSTR',
                    help='list registered passes matching SUBSTR and exit')
    ap.add_argument('--progress-every', type=int, default=25)
    ap.add_argument('-v', '--verbose', action='store_true')
    args = ap.parse_args(argv)

    if args.list_passes:
        return list_passes(args.cmd, args.list_passes, args.shell)
    for required in ('pass_name', 'inputs', 'out'):
        if not getattr(args, required):
            ap.error(f'--{required.replace("_", "-")} is required')

    runner = Runner(args)
    paths = find_inputs(args.inputs, args.pattern, args.limit)
    if not paths:
        print(f'no files matching {args.pattern} under {args.inputs}', file=sys.stderr)
        return 1

    if args.dry_run:
        cmd = runner.command(paths[0])
        print(cmd if isinstance(cmd, str) else ' '.join(shlex.quote(c) for c in cmd))
        return 0

    os.makedirs(args.out, exist_ok=True)
    print(f'{len(paths)} inputs, {args.jobs} workers, looking for '
          f'{args.pass_name!r} (select={args.select})', file=sys.stderr)

    rows = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(runner.run_one, p): p for p in paths}
        try:
            for fut in as_completed(futures):
                rows.append(fut.result())
        except KeyboardInterrupt:
            print('\ninterrupted, writing the manifest for what finished',
                  file=sys.stderr)
            pool.shutdown(wait=False, cancel_futures=True)

    rows.sort(key=lambda r: (-r['ran'], r['kernel']))
    cols = ['kernel', 'source', 'ran', 'changed', 'n_before', 'n_after',
            'scopes', 'exit_code', 'seconds', 'status', 'outdir']
    manifest = os.path.join(args.out, 'manifest.csv')
    with open(manifest, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    ran = sum(r['ran'] for r in rows)
    changed = sum(r['changed'] for r in rows)
    bad = [r for r in rows if r['status'] not in ('ok', 'skipped (resume)')]
    print(f'\n{len(rows)} kernels in {time.time() - t0:.1f}s', file=sys.stderr)
    print(f'  pass ran on   {ran}', file=sys.stderr)
    print(f'  changed IR on {changed}', file=sys.stderr)
    if bad:
        from collections import Counter
        print(f'  problems:     {len(bad)} '
              f'{dict(Counter(r["status"] for r in bad))}', file=sys.stderr)
    if ran == 0:
        print('\nThe pass never appeared. Likely the name is wrong for the\n'
              '--mlir-print-ir-* flags (they want the registered CLI name, not the\n'
              f'C++ class). Try:  --list-passes {args.pass_name}   or  --select all',
              file=sys.stderr)
    print(f'  manifest:     {manifest}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
