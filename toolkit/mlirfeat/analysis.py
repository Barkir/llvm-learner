"""
Analyses: expensive things several feature sets want, computed once per module.

Mirrors LLVM's AnalysisManager: ask the module for one and it is cached.

    live = module.analysis(Liveness)
    live.peak_bytes['ub']

Add your own by subclassing Analysis and taking the module in __init__.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

from .ir import Op, ShapedType

LOOP_OPS = {'scf.for', 'scf.while', 'scf.parallel', 'scf.forall', 'affine.for',
            'affine.parallel'}
# ops that only re-view memory someone else allocated
ALIASING_OPS = {
    'memref.subview', 'memref.collapse_shape', 'memref.expand_shape',
    'memref.cast', 'memref.reinterpret_cast', 'memref.memory_space_cast',
    'memref.view', 'memref.transpose', 'bufferization.to_tensor',
    'tensor.extract_slice', 'tensor.collapse_shape', 'tensor.expand_shape',
    'tensor.reshape', 'tensor.cast',
}
ALLOC_OPS = {'memref.alloc', 'memref.alloca', 'memref.alloca_scope',
             'bufferization.alloc_tensor', 'gpu.alloc'}
# control flow forwards buffers created inside its regions; its results are not
# new allocations and must not be counted twice
FORWARDING_OPS = {'scf.if', 'scf.for', 'scf.while', 'scf.forall',
                  'scf.execute_region', 'scf.index_switch',
                  'affine.for', 'affine.if', 'affine.parallel',
                  'arith.select', 'scope.scope'}


class Analysis:
    """Base class. Subclasses compute everything in __init__."""
    def __init__(self, module):
        self.module = module


class Loops(Analysis):
    """Loop nesting, from the region tree — no brace counting."""

    def __init__(self, module):
        super().__init__(module)
        self.loops: list[Op] = [o for o in module.ops if o.name in LOOP_OPS]
        self.depth_of: dict[int, int] = {}
        for op in module.ops:
            d = sum(1 for a in op.ancestors() if a.name in LOOP_OPS)
            self.depth_of[id(op)] = d
        self.max_depth = max(
            (self.depth_of[id(o)] + 1 for o in self.loops), default=0)
        self.n_loops = len(self.loops)
        self.trip_counts = [tc for tc in (self._trip_count(o) for o in self.loops)
                            if tc is not None]
        self.max_trip_count = max(self.trip_counts, default=0)

    def depth(self, op: Op) -> int:
        return self.depth_of.get(id(op), 0)

    def in_loop(self, op: Op) -> bool:
        return self.depth(op) > 0

    def _trip_count(self, op: Op):
        """Constant bound if the loop uses literal constants, else None."""
        consts = self.module.analysis(Constants).values
        m = re.search(r'=\s*(%[\w$.]+)\s+to\s+(%[\w$.]+)\s+step\s+(%[\w$.]+)', op.text)
        if not m:
            return None
        lo, hi, st = (consts.get(g) for g in m.groups())
        if hi is None:
            return None
        lo = lo or 0
        st = st or 1
        return max(0, (hi - lo + st - 1) // st) if st else None


class Constants(Analysis):
    """SSA name -> integer value, for arith.constant / llvm.mlir.constant."""

    _RE = re.compile(r'^\s*(%[\w$.]+)\s*=\s*(?:arith|llvm\.mlir)\.constant\s*'
                     r'\(?\s*(-?\d+)\s*(?::|\))')

    def __init__(self, module):
        super().__init__(module)
        self.values: dict[str, int] = {}
        for op in module.ops:
            m = self._RE.match(op.text)
            if m:
                self.values[m.group(1)] = int(m.group(2))


class TargetSpec(Analysis):
    """dlti target spec on the module. Sizes are stored in BITS; converted here."""

    _RE_ENTRY = re.compile(r'#dlti\.dl_entry<"(\w+)",\s*(?:(-?\d+)\s*:\s*i\d+|"([^"]*)")>')

    def __init__(self, module):
        super().__init__(module)
        head = next((l for l in module.lines if l.lstrip().startswith('module')), '')
        self.raw: dict[str, object] = {}
        for key, ival, sval in self._RE_ENTRY.findall(head):
            self.raw[key] = int(ival) if ival else sval
        self.sizes: dict[str, int] = {}          # memory space name -> bytes
        for key, val in self.raw.items():
            if key.endswith('_SIZE') and isinstance(val, int) and 'ALIGN' not in key:
                self.sizes[key[:-5].lower()] = val // 8      # bits -> bytes
        self.align: dict[str, int] = {
            key[:-11].lower(): val // 8
            for key, val in self.raw.items()
            if key.endswith('_ALIGN_SIZE') and isinstance(val, int)
        }
        m = re.search(r'hacc\.target<"([^"]+)">', head)
        self.target = m.group(1) if m else 'unknown'
        m = re.search(r'module_core_type<(\w+)>', head)
        self.core_type = m.group(1) if m else 'unknown'

    # map a memory-space name used in types to a target-spec limit
    SPACE_TO_KEY = {'ub': 'ub', 'cbuf': 'l1', 'ca': 'l0a', 'cb': 'l0b', 'cc': 'l0c'}

    def limit_for(self, space: str):
        return self.sizes.get(self.SPACE_TO_KEY.get(space, space))

    def align_for(self, space: str) -> int:
        return self.align.get(self.SPACE_TO_KEY.get(space, space), 0)


FUNC_OPS = {'func.func', 'llvm.func', 'gpu.func'}


class Liveness(Analysis):
    """
    Peak bytes live at one point, per memory space.

    Buffers are the results of allocating ops plus any shaped value that is
    materialised (not a pure view of another buffer). An SSA value is live from
    its definition line to its last use; aliases extend the root's range.

    Peak is computed PER FUNCTION and then maximised, because each function's
    buffers occupy the same on-chip memory at different times -- summing across
    functions would double count. Backup copies of functions are skipped.
    """

    def __init__(self, module):
        super().__init__(module)
        spec = module.analysis(TargetSpec)
        loops = module.analysis(Loops)

        root: dict[str, str] = {}          # ssa name -> root buffer name
        buffers: dict[str, dict] = {}

        for op in module.ops:
            if self._skipped(op):
                continue
            if not op.results:
                continue
            res = op.results[0]
            if op.name in ALIASING_OPS and op.operands:
                src = root.get(op.operands[0], op.operands[0])
                root[res.name] = src
                continue
            if res.shaped is None or res.shaped.kind == 'vector':
                continue
            if op.name in FORWARDING_OPS:
                continue
            root[res.name] = res.name
            mult = self._multiplicity(op, module)
            align = spec.align_for(res.shaped.space)
            nb = res.shaped.nbytes
            if nb is not None and align > 1:
                nb = (nb + align - 1) // align * align
            buffers[res.name] = {
                'space': res.shaped.space,
                'bytes': None if nb is None else nb * mult,
                'start': op.line, 'end': op.line,
                'op': op.name, 'in_loop': loops.in_loop(op),
                'multiplicity': mult,
                'is_alloc': op.name in ALLOC_OPS,
                'func': self._func_of(op),
            }

        for name, v in module.values.items():
            r = root.get(name)
            if r in buffers:
                buffers[r]['end'] = max(buffers[r]['end'], v.last_use_line)

        self.buffers = buffers
        self.by_space: dict[str, list[dict]] = defaultdict(list)
        for b in buffers.values():
            self.by_space[b['space']].append(b)

        self.peak_bytes: dict[str, int] = {}
        self.total_bytes: dict[str, int] = {}
        self.max_buffer: dict[str, int] = {}
        for space, bufs in self.by_space.items():
            sized = [b for b in bufs if b['bytes'] is not None]
            per_func: dict[str, list[dict]] = defaultdict(list)
            for b in sized:
                per_func[b['func']].append(b)
            self.peak_bytes[space] = max(
                (self._peak(g) for g in per_func.values()), default=0)
            self.total_bytes[space] = max(
                (sum(b['bytes'] for b in g) for g in per_func.values()), default=0)
            self.max_buffer[space] = max((b['bytes'] for b in sized), default=0)

    @staticmethod
    def _func_of(op: Op) -> str:
        for a in op.ancestors():
            if a.name in FUNC_OPS:
                m = re.search(r'@([\w$.-]+)', a.text)
                return m.group(1) if m else str(id(a))
        return '<module>'

    @classmethod
    def _skipped(cls, op: Op) -> bool:
        """Backup copies of functions are not live at the same time as the real ones."""
        for a in op.ancestors():
            if a.name in FUNC_OPS and ('backup_function' in a.text
                                       or re.search(r'@[\w$.]*_backup\b', a.text)):
                return True
        return False

    @staticmethod
    def _multiplicity(op: Op, module) -> int:
        """hivm.multi_buffer = 2 means the buffer is allocated twice."""
        for other in module.ops:
            if other.name == 'annotation.mark' and op.results and \
                    op.results[0].name in other.operands:
                m = re.search(r'multi_buffer\s*=\s*(\d+)', other.text)
                if m:
                    return int(m.group(1))
        return 1

    @staticmethod
    def _peak(bufs) -> int:
        events = []
        for b in bufs:
            events.append((b['start'], 1, b['bytes']))
            events.append((b['end'] + 1, 0, -b['bytes']))
        cur = best = 0
        for _, _, d in sorted(events):
            cur += d
            best = max(best, cur)
        return best


class ShapedValues(Analysis):
    """All distinct shaped SSA values, deduplicated by name (not by mention)."""

    def __init__(self, module):
        super().__init__(module)
        self.values = [v for v in module.values.values() if v.shaped is not None]
        self.by_kind: dict[str, list] = defaultdict(list)
        for v in self.values:
            self.by_kind[v.shaped.kind].append(v)
        self.dtypes = Counter(v.shaped.elem for v in self.values)
