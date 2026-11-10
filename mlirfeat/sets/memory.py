"""Shape and memory-footprint feature sets."""

from __future__ import annotations

from ..analysis import Liveness, ShapedValues, TargetSpec
from ..featureset import FeatureSet

DTYPES = ['f32', 'f16', 'bf16', 'f8E4M3FN', 'i64', 'i32', 'i8', 'i1']

# module level: a comprehension in a class body cannot see other class attributes
SPACES = ['ub', 'cbuf', 'cc', 'ca', 'cb', 'ssbuf', 'gm', 'default']
PER_SPACE = ['n_buffers', 'n_allocs', 'n_unsized', 'peak_bytes', 'total_bytes',
             'max_buffer_bytes', 'n_in_loops', 'n_multi_buffered',
             'peak_ratio', 'total_ratio']


class ShapeFeatures(FeatureSet):
    """
    Sizes of DISTINCT SSA values, not of every textual mention of a type, and
    each value measured with its own element type.
    """
    name = 'shape'
    features = (['n_values', 'n_tensors', 'n_memrefs', 'n_vectors',
                 'n_dynamic', 'frac_dynamic', 'max_rank',
                 'total_bytes', 'max_bytes', 'median_bytes',
                 'max_elements', 'total_elements', 'max_dim', 'n_dtypes',
                 'primary_dtype'] + [f'has_{d}' for d in DTYPES])
    default = 0

    def extract(self, module):
        sv = module.analysis(ShapedValues)
        vals = sv.values
        if not vals:
            return {}
        sizes = sorted(v.nbytes for v in vals if v.nbytes is not None)
        elems = [v.shaped.num_elements for v in vals
                 if v.shaped.num_elements is not None]
        dyn = sum(1 for v in vals if not v.shaped.is_static)
        f = {
            'n_values': len(vals),
            'n_tensors': len(sv.by_kind.get('tensor', [])),
            'n_memrefs': len(sv.by_kind.get('memref', [])),
            'n_vectors': len(sv.by_kind.get('vector', [])),
            'n_dynamic': dyn,
            'frac_dynamic': round(dyn / len(vals), 3),
            'max_rank': max(v.shaped.rank for v in vals),
            'total_bytes': sum(sizes),
            'max_bytes': max(sizes, default=0),
            'median_bytes': sizes[len(sizes) // 2] if sizes else 0,
            'max_elements': max(elems, default=0),
            'total_elements': sum(elems),
            'max_dim': max((d for v in vals for d in v.shaped.dims if d), default=0),
            'n_dtypes': len(sv.dtypes),
            'primary_dtype': sv.dtypes.most_common(1)[0][0] if sv.dtypes else 'unknown',
        }
        for d in DTYPES:
            f[f'has_{d}'] = int(d in sv.dtypes)
        return f


class MemoryFeatures(FeatureSet):
    """
    Per memory space: how much is live at once, not the sum of everything.

    `spaces` is fixed so the CSV columns line up across kernels; add a space
    here (or subclass and override) if your target has more.
    """
    name = 'mem'
    spaces = SPACES
    per_space = PER_SPACE
    features = ([f'{s}_{k}' for s in SPACES for k in PER_SPACE]
                + ['n_spaces', 'n_alloc_ops', 'worst_ratio', 'worst_space'])
    default = 0

    def extract(self, module):
        live = module.analysis(Liveness)
        spec = module.analysis(TargetSpec)
        f = {'n_spaces': len(live.by_space),
             'n_alloc_ops': sum(1 for b in live.buffers.values() if b['is_alloc'])}
        worst, worst_space = 0.0, 'none'
        for space in self.spaces:
            bufs = live.by_space.get(space, [])
            if not bufs:
                continue
            peak = live.peak_bytes.get(space, 0)
            total = live.total_bytes.get(space, 0)
            limit = spec.limit_for(space)
            pr = round(peak / limit, 4) if limit else 0
            f.update({
                f'{space}_n_buffers': len(bufs),
                f'{space}_n_allocs': sum(1 for b in bufs if b['is_alloc']),
                f'{space}_n_unsized': sum(1 for b in bufs if b['bytes'] is None),
                f'{space}_peak_bytes': peak,
                f'{space}_total_bytes': total,
                f'{space}_max_buffer_bytes': live.max_buffer.get(space, 0),
                f'{space}_n_in_loops': sum(1 for b in bufs if b['in_loop']),
                f'{space}_n_multi_buffered': sum(1 for b in bufs
                                                 if b['multiplicity'] > 1),
                f'{space}_peak_ratio': pr,
                f'{space}_total_ratio': round(total / limit, 4) if limit else 0,
            })
            if pr > worst:
                worst, worst_space = pr, space
        f['worst_ratio'] = worst
        f['worst_space'] = worst_space
        return f
