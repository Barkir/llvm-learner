"""
Dialect-specific feature sets.

Each one declares the dialect it needs. A kernel with both affine and hivm gets
both sets; a kernel with only one gets zeros for the other, so the columns still
line up. This is the file to copy when you add support for a new dialect.
"""

from __future__ import annotations

import re
from collections import Counter

from ..analysis import Loops
from ..featureset import FeatureSet


class OpCountFeatures(FeatureSet):
    """
    Base for 'count these ops' sets: list `ops`, get one column each plus a total.
    Subclasses only declare data.
    """
    abstract = True
    ops: list[str] = []
    extra: list[str] = []

    def __init_subclass__(cls, **kw):
        if not cls.__dict__.get('abstract', False):
            cls.features = ['n_total'] + [f'n_{o.replace(".", "_")}' for o in cls.ops] \
                + cls.extra
        super().__init_subclass__(**kw)

    def extract(self, module):
        counts = Counter(o.name for o in module.ops)
        f = {f'n_{o.replace(".", "_")}': counts.get(o, 0) for o in self.ops}
        f['n_total'] = sum(c for n, c in counts.items()
                           if n.split('.', 1)[0] in self.requires_dialects)
        f.update(self.extra_features(module) or {})
        return f

    def extra_features(self, module) -> dict:
        return {}


class AffineFeatures(OpCountFeatures):
    """Applies only when the module uses the affine dialect."""
    name = 'affine'
    requires_dialects = {'affine'}
    ops = ['affine.apply', 'affine.for', 'affine.if', 'affine.load', 'affine.store',
           'affine.min', 'affine.max', 'affine.parallel', 'affine.yield']
    extra = ['n_maps', 'n_sets', 'max_map_symbols', 'max_map_dims',
             'max_for_depth', 'n_maps_with_floordiv']

    def extra_features(self, module):
        maps = re.findall(r'affine_map<\(([^)]*)\)(?:\[([^\]]*)\])?\s*->', module.text)
        loops = module.analysis(Loops)
        aff = [o for o in module.ops if o.name in ('affine.for', 'affine.parallel')]
        return {
            'n_maps': len(re.findall(r'affine_map<', module.text)),
            'n_sets': len(re.findall(r'affine_set<', module.text)),
            'max_map_dims': max((len([d for d in dims.split(',') if d.strip()])
                                 for dims, _ in maps), default=0),
            'max_map_symbols': max((len([s for s in (syms or '').split(',') if s.strip()])
                                    for _, syms in maps), default=0),
            'max_for_depth': max((loops.depth(o) + 1 for o in aff), default=0),
            'n_maps_with_floordiv': len(re.findall(r'floordiv|ceildiv|mod',
                                                   module.text)),
        }


class HIVMFeatures(OpCountFeatures):
    """Ascend hivm dialect: DMA, cube, synchronisation, buffer annotations."""
    name = 'hivm'
    requires_dialects = {'hivm'}
    ops = ['hivm.hir.copy', 'hivm.hir.load', 'hivm.hir.store', 'hivm.hir.mmad',
           'hivm.hir.fixpipe', 'hivm.hir.sync_block_set', 'hivm.hir.sync_block_wait',
           'hivm.hir.anchor', 'hivm.hir.pointer_cast', 'hivm.hir.set_ctrl',
           'hivm.hir.vbrc', 'hivm.hir.get_block_idx', 'hivm.hir.get_sub_block_idx']
    extra = ['n_dma', 'n_sync', 'n_multi_buffer_marks', 'max_multi_buffer',
             'n_tightly_coupled_buffers', 'n_aic_funcs', 'n_aiv_funcs',
             'n_cube_ops', 'n_vector_ops', 'has_mix']

    def extra_features(self, module):
        t = module.text
        mb = [int(x) for x in re.findall(r'multi_buffer\s*=\s*(\d+)', t)]
        return {
            'n_dma': module.count('hivm.hir.copy', 'hivm.hir.load',
                                  'hivm.hir.store', 'hivm.hir.fixpipe'),
            'n_sync': module.count('hivm.hir.sync_block_set',
                                   'hivm.hir.sync_block_wait'),
            'n_multi_buffer_marks': len(mb),
            'max_multi_buffer': max(mb, default=1),
            'n_tightly_coupled_buffers': len(set(
                re.findall(r'tightly_coupled_buffer<(\d+)>', t))),
            'n_aic_funcs': len(re.findall(r'func_core_type<AIC>', t)),
            'n_aiv_funcs': len(re.findall(r'func_core_type<AIV>', t)),
            'n_cube_ops': len(re.findall(r'tcore_type<CUBE>', t)),
            'n_vector_ops': len(re.findall(r'tcore_type<VECTOR>', t)),
            'has_mix': int('module_core_type<MIX>' in t),
        }


class RegbaseIntrinFeatures(FeatureSet):
    """
    Vector intrinsics at the LLVM stage (hivm_regbaseintrins.intr.*).
    Buckets them instead of emitting one column per intrinsic, so the column set
    stays fixed as new intrinsics appear.
    """
    name = 'intr'
    requires_dialects = {'hivm_regbaseintrins'}
    buckets = ['load', 'store', 'arith', 'math', 'reduce', 'broadcast',
               'predicate', 'other']
    features = ([f'n_{b}' for b in buckets]
                + ['n_total', 'n_distinct', 'compute_to_mem_ratio',
                   'max_vector_lanes'])
    default = 0

    MATH = ('vexp', 'vln', 'vlog', 'vsqrt', 'vrsqrt', 'vdiv', 'vrec', 'vtanh',
            'vpow', 'vsin', 'vcos', 'verf', 'vsigmoid')
    REDUCE = ('vcadd', 'vcmax', 'vcmin', 'vcgadd', 'vcgmax', 'vcgmin')

    def bucket(self, base: str) -> str:
        if base.startswith('vld'):
            return 'load'
        if base.startswith('vst'):
            return 'store'
        if base.startswith(self.REDUCE):
            return 'reduce'
        if base.startswith(self.MATH):
            return 'math'
        if base.startswith(('vdup', 'vbr')):
            return 'broadcast'
        if base.startswith('p'):
            return 'predicate'
        if base.startswith('v'):
            return 'arith'
        return 'other'

    def extract(self, module):
        counts = Counter()
        bases = set()
        for op in module.ops:
            if op.dialect != 'hivm_regbaseintrins':
                continue
            base = op.name.split('.intr.', 1)[-1].removeprefix('hivm.').split('.')[0]
            bases.add(base)
            counts[self.bucket(base)] += 1
        lanes = [int(n) for n in re.findall(r'vector<(\d+)x', module.text)]
        f = {f'n_{b}': counts.get(b, 0) for b in self.buckets}
        f['n_total'] = sum(counts.values())
        f['n_distinct'] = len(bases)
        f['compute_to_mem_ratio'] = round(
            (counts['arith'] + counts['math'] + counts['reduce'])
            / max(counts['load'] + counts['store'], 1), 3)
        f['max_vector_lanes'] = max(lanes, default=0)
        return f


class LinalgFeatures(OpCountFeatures):
    """Structured ops, before bufferization."""
    name = 'linalg'
    requires_dialects = {'linalg'}
    ops = ['linalg.matmul', 'linalg.batch_matmul', 'linalg.generic', 'linalg.reduce',
           'linalg.fill', 'linalg.broadcast', 'linalg.transpose', 'linalg.index',
           'linalg.map', 'linalg.yield']
    extra = ['n_reductions', 'is_reduce_only', 'n_matmul_like']

    def extra_features(self, module):
        mm = module.count('linalg.matmul', 'linalg.batch_matmul')
        red = module.count('linalg.reduce', 'vector.multi_reduction')
        return {
            'n_reductions': red,
            'is_reduce_only': int(red > 0 and module.count('linalg.generic') == 0),
            'n_matmul_like': mm,
        }


class TensorFeatures(OpCountFeatures):
    """Tensor-level ops that usually turn into buffers later."""
    name = 'tensor'
    requires_dialects = {'tensor'}
    ops = ['tensor.empty', 'tensor.extract_slice', 'tensor.insert_slice',
           'tensor.expand_shape', 'tensor.collapse_shape', 'tensor.reshape',
           'tensor.pad', 'tensor.cast']
    extra = ['n_bufferization_ops', 'n_materialize']

    def extra_features(self, module):
        return {
            'n_bufferization_ops': len(module.ops_in_dialect('bufferization')),
            'n_materialize': module.count(
                'bufferization.materialize_in_destination'),
        }


class LLVMStageFeatures(OpCountFeatures):
    """After lowering to the LLVM dialect."""
    name = 'llvm'
    requires_dialects = {'llvm'}
    ops = ['llvm.alloca', 'llvm.getelementptr', 'llvm.load', 'llvm.store',
           'llvm.call', 'llvm.br', 'llvm.cond_br', 'llvm.select', 'llvm.icmp',
           'llvm.insertvalue', 'llvm.extractvalue', 'llvm.inttoptr']
    extra = ['n_blocks', 'n_funcs', 'n_struct_selects', 'n_dma_calls']

    def extra_features(self, module):
        return {
            'n_blocks': len(re.findall(r'^\s*\^bb\d+', module.text, re.M)),
            'n_funcs': module.count('llvm.func'),
            'n_struct_selects': sum(1 for o in module.ops
                                    if o.name == 'llvm.select'
                                    and '!llvm.struct' in o.text),
            'n_dma_calls': len(re.findall(
                r'@(?:load|store|copy)_\w+_to_\w+_\dd', module.text)),
        }
