"""Feature sets that apply to any MLIR module."""

from __future__ import annotations

import re
from collections import Counter

from ..analysis import Loops, TargetSpec
from ..featureset import FeatureSet

FUNC_OPS = ('func.func', 'llvm.func')
TRACKED_DIALECTS = ['arith', 'math', 'vector', 'memref', 'tensor', 'scf', 'affine',
                    'linalg', 'bufferization', 'hivm', 'llvm', 'annotation', 'func']


class TargetSpecFeatures(FeatureSet):
    """Hardware limits declared on the module (dlti target spec)."""
    name = 'target'
    features = ['ub_size', 'l1_size', 'l0a_size', 'l0b_size', 'l0c_size',
                'ai_core_count', 'vector_core_count', 'arch', 'chip', 'core_type']
    default = 0

    def applies(self, module) -> bool:
        return bool(module.analysis(TargetSpec).raw)

    def extract(self, module):
        spec = module.analysis(TargetSpec)
        f = {f'{k}_size': spec.sizes.get(k, 0)
             for k in ('ub', 'l1', 'l0a', 'l0b', 'l0c')}
        f['ai_core_count'] = spec.raw.get('AI_CORE_COUNT', 0)
        f['vector_core_count'] = spec.raw.get('VECTOR_CORE_COUNT', 0)
        f['arch'] = spec.raw.get('ARCH', 'unknown')
        f['chip'] = spec.target
        f['core_type'] = spec.core_type
        return f


class StructureFeatures(FeatureSet):
    """Size and shape of the module: ops, functions, dialect mix, entry signature."""
    name = 'struct'
    features = (['n_ops', 'n_lines', 'n_funcs', 'n_entry_funcs', 'n_vector_funcs',
                 'n_backup_funcs', 'n_distinct_ops', 'n_dialects', 'stage',
                 'entry_name', 'entry_n_args', 'entry_n_dynamic_args',
                 'mix_mode', 'parallel_mode', 'max_region_depth']
                + [f'n_{d}_ops' for d in TRACKED_DIALECTS])
    default = 0

    def extract(self, module):
        funcs = [o for o in module.ops if o.name in FUNC_OPS]
        entries = [f for f in funcs if 'hacc.entry' in f.text]
        vfs = [f for f in funcs if 'vector_function' in f.text]
        backups = [f for f in funcs if 'backup_function' in f.text
                   or re.search(r'@[\w$.]*_backup\b', f.text)]
        by_dialect = Counter(o.dialect for o in module.ops)

        f = {
            'n_ops': len(module.ops),
            'n_lines': len(module.lines),
            'n_funcs': len(funcs),
            'n_entry_funcs': len(entries),
            'n_vector_funcs': len(vfs),
            'n_backup_funcs': len(backups),
            'n_distinct_ops': len({o.name for o in module.ops}),
            'n_dialects': len(module.dialects),
            'stage': 'llvm' if by_dialect.get('llvm', 0) > by_dialect.get('func', 0)
                     else 'hir' if 'hivm' in module.dialects else 'high',
            'max_region_depth': max((o.depth for o in module.ops), default=0),
        }
        for d in TRACKED_DIALECTS:
            f[f'n_{d}_ops'] = by_dialect.get(d, 0)

        entry = max(entries, key=lambda o: len(o.text), default=None) or \
            max(funcs, key=lambda o: len(o.text), default=None)
        if entry is not None:
            m = re.search(r'@([\w$.-]+)\(', entry.text)
            f['entry_name'] = m.group(1) if m else 'unknown'
            args = re.findall(r'%arg\d+\s*:', entry.text)
            f['entry_n_args'] = len(args)
            f['entry_n_dynamic_args'] = entry.text.count('<?x')
            for key, pat in (('mix_mode', r'mix_mode\s*=\s*"(\w+)"'),
                             ('parallel_mode', r'parallel_mode\s*=\s*"(\w+)"')):
                m = re.search(pat, entry.text)
                f[key] = m.group(1) if m else 'unknown'
        return f


class LoopFeatures(FeatureSet):
    """Loop nesting from the region tree, plus constant trip counts."""
    name = 'loop'
    features = ['n_loops', 'max_depth', 'max_trip_count', 'n_known_trip_counts',
                'n_ifs', 'n_ops_in_loops', 'frac_ops_in_loops']
    default = 0

    def extract(self, module):
        loops = module.analysis(Loops)
        in_loop = sum(1 for o in module.ops if loops.in_loop(o))
        return {
            'n_loops': loops.n_loops,
            'max_depth': loops.max_depth,
            'max_trip_count': loops.max_trip_count,
            'n_known_trip_counts': len(loops.trip_counts),
            'n_ifs': module.count('scf.if', 'affine.if'),
            'n_ops_in_loops': in_loop,
            'frac_ops_in_loops': round(in_loop / max(len(module.ops), 1), 3),
        }
