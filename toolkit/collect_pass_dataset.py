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
import tempfile
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─────────────────────────────────────────────────────────────────────────
# Command presets.  EDIT THIS to add your own compiler invocations; then run
# with  --preset <name>  instead of spelling out --cmd.
#
# Placeholders available in a cmd string:
#   {input}   absolute path of the .mlir being compiled
#   {indir}   directory containing it        {stem}   its filename without .mlir
#   {kernel}  the kernel id for this row     {workdir} this compile's scratch dir
#   {outdir}  <out>/<kernel>/ for this row   {target} value of --target
#   plus anything returned by the preset's "paths" function
#
# Do NOT put --mlir-print-ir-* / --bishengir-print-ir-* flags in a preset:
# the script appends those itself from --pass / --select / --print-flag-prefix.
# ─────────────────────────────────────────────────────────────────────────
def _ascend_paths():
    """Locate the Triton Ascend backend; env vars win, else ask Triton itself."""
    root = os.environ.get('TRITON_ASCEND_DIR')
    if not root:
        try:
            import triton
            root = os.path.join(os.path.dirname(triton.__file__), 'backends', 'ascend')
        except Exception:
            root = ''
    return {
        'bishengir': os.environ.get(
            'BISHENGIR_COMPILE',
            os.path.join(root, 'bishengir', 'bin', 'bishengir-compile') if root else
            'bishengir-compile'),
        'libdevice': os.environ.get(
            'ASCEND_LIBDEVICE',
            os.path.join(root, 'lib', 'libdevice.10.bc') if root else
            'libdevice.10.bc'),
    }


PRESETS = {
    'bishengir': {
        'help': 'Triton Ascend bishengir-compile, as the Triton backend invokes it',
        'paths': _ascend_paths,
        'defaults': {'id_mode': 'parent', 'target': 'Ascend950PR_9579'},
        'cmd': (
            '{bishengir} {input}'
            ' --target={target}'
            ' --enable-auto-multi-buffer=True'
            ' --enable-auto-bind-sub-block=True'
            ' --enable-optimized-metaop=True'
            ' --disable-ffts'
            ' --enable-debug-info=true'
            ' --limit-auto-multi-buffer-of-local-buffer=no-limit'
            ' --enable-auto-blockify-loop'
            ' --enable-hfusion-compile=true'
            ' --enable-triton-kernel-compile=true'
            " '--append-bisheng-options=-cce-link-aicore-ll-module {libdevice}'"
            ' --mlir-print-stacktrace-on-diagnostic'
            ' --enable-vf-merge-level=1'
            ' -o {workdir}/kernel.o'
        ),
    },
    'mlir-opt': {
        'help': 'plain mlir-opt, for trying the toolkit out',
        'defaults': {},
        'cmd': 'mlir-opt {input}',
    },
}


def resolve_preset(name):
    if name not in PRESETS:
        raise SystemExit(f'unknown preset {name!r}; available: {list(PRESETS)}')
    return PRESETS[name]


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
    def __init__(self, args, extra_subs=None):
        self.args = args
        self.extra_subs = extra_subs or {}
        self.lock = threading.Lock()
        self.done = 0

    def flags(self) -> list[str]:
        a = self.args
        pre = a.print_flag_prefix            # 'mlir', or 'bishengir' for drivers
        f = []                               # that namespace their own options
        if a.select == 'all':
            if not a.after_only:
                f.append(f'--{pre}-print-ir-before-all')
            f.append(f'--{pre}-print-ir-after-all')
        else:
            arg = a.pass_arg or a.pass_name
            if not a.after_only:
                f.append(f'--{pre}-print-ir-before={arg}')
            f.append(f'--{pre}-print-ir-after={arg}')
        if a.changed_only:
            f.append(f'--{pre}-print-ir-after-change')
        if a.module_scope:
            f += [f'--{pre}-print-ir-module-scope', '--mlir-disable-threading']
        f += a.extra_flag
        return f

    def command(self, path: str, kernel: str = '', workdir: str = '',
                outdir: str = '') -> list[str] | str:
        tmpl = self.args.cmd
        subs = dict(self.extra_subs)
        subs.update({'input': os.path.abspath(path), 'kernel': kernel,
                'indir': os.path.dirname(os.path.abspath(path)),
                'stem': os.path.splitext(os.path.basename(path))[0],
                'workdir': workdir or os.getcwd(),
                'target': self.args.target,
                'outdir': os.path.abspath(outdir) if outdir else ''})
        cmd = tmpl
        for key, val in subs.items():
            cmd = cmd.replace('{' + key + '}', shlex.quote(str(val)))
        if '{input}' not in tmpl:
            cmd = f'{cmd} {shlex.quote(subs["input"])}'
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

        # A real compiler driver writes output files into its working
        # directory. Run each input somewhere of its own or parallel jobs
        # overwrite each other's artifacts.
        tmpdir = None
        if a.workdir == 'temp':
            tmpdir = tempfile.mkdtemp(prefix=f'cpd_{kid}_')
            cwd = tmpdir
        elif a.workdir == 'out':
            os.makedirs(outdir, exist_ok=True)
            cwd = outdir
        else:
            cwd = None                                  # inherit; only safe at -j 1

        cmd = self.command(path, kid, cwd or os.getcwd(), outdir)
        t0 = time.time()
        try:
            p = subprocess.run(cmd, shell=a.shell, capture_output=True, text=True,
                               timeout=a.timeout, errors='replace', cwd=cwd)
            out, err, code = p.stdout, p.stderr, p.returncode
        except subprocess.TimeoutExpired as e:
            out = e.stdout or ''
            err = (e.stderr or '') if isinstance(e.stderr, str) else ''
            code, row['status'] = None, f'timeout after {a.timeout}s'
        except FileNotFoundError as e:
            if tmpdir:
                shutil.rmtree(tmpdir, ignore_errors=True)
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
            if a.save_artifacts and tmpdir and os.path.isdir(tmpdir):
                dest = os.path.join(outdir, 'artifacts')
                os.makedirs(dest, exist_ok=True)
                for fn in os.listdir(tmpdir):
                    src = os.path.join(tmpdir, fn)
                    if os.path.isfile(src):
                        shutil.copy2(src, dest)
            meta = dict(row)
            meta['command'] = cmd if isinstance(cmd, str) else ' '.join(cmd)
            meta['pass_query'] = a.pass_name
            meta['dumps'] = [{'phase': d.phase, 'pass': d.pass_name,
                              'arg': d.pass_arg, 'scope_op': d.scope_op,
                              'scope_sym': d.scope_sym, 'lines': len(d.body)}
                             for d in dumps]
            self._write(os.path.join(outdir, 'meta.json'),
                        json.dumps(meta, indent=2))

        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)

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
    base = re.sub(r'\{\w+\}', '', cmd).strip()
    text = ''
    for flag in ('--help-list-hidden', '--help-hidden', '--help-list', '--help'):
        argv = f'{base} {flag}'
        try:
            p = subprocess.run(argv if shell else shlex.split(argv), shell=shell,
                               capture_output=True, text=True, timeout=120)
        except Exception as e:
            print(f'could not run the tool: {e}', file=sys.stderr)
            return 1
        text = p.stdout + p.stderr
        if len(text.strip()) > 200:
            print(f'(used {flag})', file=sys.stderr)
            break
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
    ap.add_argument('--preset', help='use a built-in command template (see --list-presets)')
    ap.add_argument('--list-presets', action='store_true',
                    help='show the built-in command templates and exit')
    ap.add_argument('--target', default='Ascend950PR_9579',
                    help='value for {target} in a preset command')
    ap.add_argument('--cmd',
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
    ap.add_argument('--print-flag-prefix', default='mlir',
                    help="namespace of the IR-printing flags. 'mlir' gives "
                         "--mlir-print-ir-after=<pass>; use 'bishengir' for a driver "
                         "that exposes --bishengir-print-ir-after=<pass>")
    ap.add_argument('--after-only', action='store_true',
                    help='only ask for the After dump (some drivers have no Before)')
    ap.add_argument('--probe', action='store_true',
                    help='run ONE input, show its raw stderr and any dump banners '
                         'found, then stop. Use this first on an unfamiliar driver.')
    ap.add_argument('--select', choices=['pass', 'all'], default='pass',
                    help="'pass' uses --mlir-print-ir-before/after=<name> (needs the "
                         "CLI name); 'all' dumps every pass and filters here")
    ap.add_argument('--changed-only', action='store_true',
                    help='add --mlir-print-ir-after-change and keep only kernels it changed')
    ap.add_argument('--module-scope', action='store_true',
                    help='print the whole module (adds --mlir-disable-threading)')
    ap.add_argument('--extra-flag', action='append', default=[],
                    help='extra compiler flag, repeatable. A value starting with '
                         '"-" needs the equals form: --extra-flag=--my-flag')
    ap.add_argument('--keep-all', action='store_true',
                    help='write a folder for every kernel, not only matches')
    ap.add_argument('--copy-input', action='store_true', help='copy the input next to the dumps')
    ap.add_argument('--save-stderr', action='store_true', help='keep raw stdout/stderr')
    ap.add_argument('--resume', action='store_true', help='skip kernels already done')
    ap.add_argument('--workdir', choices=['temp', 'out', 'cwd'], default='temp',
                    help="where each compile runs. 'temp' (default) gives every "
                         "input a private scratch directory, so parallel jobs "
                         "cannot overwrite each other's output files; 'out' uses "
                         "the kernel's result folder; 'cwd' inherits this shell's "
                         "directory (only safe at -j 1, or if the command writes "
                         "nothing but stdout/stderr)")
    ap.add_argument('--save-artifacts', action='store_true',
                    help='copy files the compiler produced into <out>/<kernel>/artifacts/')
    ap.add_argument('--shell', action='store_true', help='run --cmd through a shell')
    ap.add_argument('--dry-run', action='store_true', help='print the first command and exit')
    ap.add_argument('--list-passes', metavar='SUBSTR',
                    help='list registered passes matching SUBSTR and exit')
    ap.add_argument('--progress-every', type=int, default=25)
    ap.add_argument('-v', '--verbose', action='store_true')
    args = ap.parse_args(argv)

    if args.list_presets:
        for name, pre in PRESETS.items():
            print(f'{name}\n  {pre["help"]}')
            subs = pre.get('paths', dict)()
            print('  ' + pre['cmd'].format(**{**subs, 'input': '{input}',
                                              'indir': '{indir}', 'stem': '{stem}',
                                              'kernel': '{kernel}',
                                              'workdir': '{workdir}',
                                              'outdir': '{outdir}',
                                              'target': '{target}'}) + '\n')
        return 0

    extra_subs = {}
    if args.preset:
        pre = resolve_preset(args.preset)
        extra_subs = pre.get('paths', dict)()
        for key, val in pre.get('defaults', {}).items():
            if getattr(args, key, None) in (None, ap.get_default(key)):
                setattr(args, key, val)
        if not args.cmd:
            args.cmd = pre['cmd']
    if not args.cmd:
        ap.error('give --cmd or --preset (see --list-presets)')

    if args.list_passes:
        return list_passes(args.cmd.format(**{**extra_subs, 'input': '', 'indir': '',
                                              'stem': '', 'kernel': '', 'workdir': '.',
                                              'outdir': '.', 'target': args.target}),
                           args.list_passes, args.shell)
    for required in ('pass_name', 'inputs', 'out'):
        if not getattr(args, required):
            ap.error(f'--{required.replace("_", "-")} is required')

    runner = Runner(args, extra_subs)
    paths = find_inputs(args.inputs, args.pattern, args.limit)
    if not paths:
        print(f'no files matching {args.pattern} under {args.inputs}', file=sys.stderr)
        return 1

    if args.dry_run:
        cmd = runner.command(paths[0], runner.kernel_id(paths[0]),
                             '<per-input workdir>',
                             os.path.join(args.out, runner.kernel_id(paths[0])))
        print(cmd if isinstance(cmd, str) else ' '.join(shlex.quote(c) for c in cmd))
        return 0

    if args.probe:
        path = paths[0]
        kid = runner.kernel_id(path)
        cmd = runner.command(path, kid, os.getcwd(), args.out)
        print('command:\n  ' + (cmd if isinstance(cmd, str)
                                 else ' '.join(shlex.quote(c) for c in cmd)) + '\n')
        try:
            p = subprocess.run(cmd, shell=args.shell, capture_output=True,
                               text=True, timeout=args.timeout, errors='replace')
        except Exception as e:
            print(f'failed to run: {e}', file=sys.stderr)
            return 1
        err = p.stderr
        print(f'exit code {p.returncode}, {len(err.splitlines())} stderr lines, '
              f'{len(p.stdout.splitlines())} stdout lines\n')
        banners = [l for l in err.split('\n')
                   if 'IR Dump' in l or 'Dump Before' in l or 'Dump After' in l]
        if banners:
            print(f'{len(banners)} dump banner(s) found; first few:')
            for l in banners[:5]:
                print('  ' + l.strip()[:150])
            dumps = parse_dumps(err)
            names = sorted({f'{d.pass_name}({d.pass_arg})' for d in dumps})
            print(f'\nparsed {len(dumps)} dumps from passes: {names[:12]}')
            matched = [d for d in dumps if d.matches(args.pass_name)]
            print(f'matching --pass {args.pass_name!r}: {len(matched)}')
            if not matched:
                print('  -> name mismatch: use one of the names above, or --select all')
        else:
            print('NO dump banners in stderr. Either the flag namespace is wrong '
                  '(try --print-flag-prefix), the pass never ran, or the driver '
                  'writes dumps to files instead. First 25 stderr lines:')
            for l in err.split('\n')[:25]:
                print('  | ' + l[:150])
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
