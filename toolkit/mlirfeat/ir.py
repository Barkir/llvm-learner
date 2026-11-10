"""
Backend-agnostic MLIR object model.

Two parsers build the same model, so feature sets never care which one ran:

  TextParser      pure-Python, parses the printed assembly. Works anywhere,
                  handles unregistered dialects (hivm, hacc) for free, but the
                  result-type extraction is a heuristic.
  BindingsParser  uses the real MLIR Python bindings when they are installed
                  (see backend_bindings.py). Exact, but needs a build with
                  -DMLIR_ENABLE_BINDINGS_PYTHON=ON.

Model:  Module -> Op (tree via parent/children) -> Value (SSA results)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator

DTYPE_BYTES = {
    'f64': 8, 'f32': 4, 'f16': 2, 'bf16': 2, 'tf32': 4,
    'i64': 8, 'i32': 4, 'i16': 2, 'i8': 1, 'i4': 1, 'i1': 1,
    'index': 8, 'f8E4M3FN': 1, 'f8E5M2': 1,
}

SHAPED_KINDS = ('tensor', 'memref', 'vector')


# ─────────────────────────────────────────────────────────────────────────
# types
# ─────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ShapedType:
    """tensor<16x2048xf32, #hivm.address_space<ub>> and friends."""
    kind: str                     # tensor | memref | vector
    dims: tuple                   # ints, or None for a dynamic '?' dim
    elem: str
    space: str = 'default'        # memory / address space name

    @property
    def rank(self) -> int:
        return len(self.dims)

    @property
    def is_static(self) -> bool:
        return all(d is not None for d in self.dims)

    @property
    def num_elements(self):
        if not self.is_static:
            return None
        n = 1
        for d in self.dims:
            n *= d
        return n

    @property
    def nbytes(self):
        n = self.num_elements
        return None if n is None else n * DTYPE_BYTES.get(self.elem, 4)


_RE_SPACE = re.compile(r'address_space<(\w+)>')
_RE_MEMSPACE_INT = re.compile(r',\s*(\d+)\s*>\s*$')


def _split_top(s: str, sep: str = ',') -> list[str]:
    out, depth, cur = [], 0, []
    for ch in s:
        if ch in '(<[{':
            depth += 1
        elif ch in ')>]}':
            depth -= 1
        if ch == sep and depth == 0:
            out.append(''.join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur:
        out.append(''.join(cur))
    return out


def parse_shaped_type(s: str) -> ShapedType | None:
    """Leftmost shaped type in `s`, or None."""
    hits = [(s.find(k + '<'), k) for k in SHAPED_KINDS if s.find(k + '<') >= 0]
    for _, kind in sorted(hits):
        i = s.find(kind + '<')
        j, depth = i + len(kind) + 1, 1
        while j < len(s) and depth:
            if s[j] == '<':
                depth += 1
            elif s[j] == '>':
                depth -= 1
            j += 1
        inner = s[i + len(kind) + 1: j - 1]
        parts = _split_top(inner)
        toks = parts[0].strip().split('x')
        elem = toks[-1]
        dims = []
        for t in toks[:-1]:
            t = t.strip()
            if t == '?':
                dims.append(None)
            elif t.isdigit():
                dims.append(int(t))
            else:                      # not a shape after all (e.g. memref<i32, ...>)
                dims = []
                break
        m = _RE_SPACE.search(inner)
        if m:
            space = m.group(1)
        else:
            mi = _RE_MEMSPACE_INT.search(inner)
            space = f'as{mi.group(1)}' if mi else 'default'
        return ShapedType(kind, tuple(dims), elem, space)
    return None


# ─────────────────────────────────────────────────────────────────────────
# values and ops
# ─────────────────────────────────────────────────────────────────────────
@dataclass
class Value:
    name: str                       # scope-qualified key, e.g. '3|%alloc_2'
    type_str: str = ''
    shaped: ShapedType | None = None
    def_op: 'Op | None' = None
    def_line: int = -1
    last_use_line: int = -1

    @property
    def nbytes(self):
        return self.shaped.nbytes if self.shaped else None


@dataclass
class Op:
    name: str                       # 'memref.alloc', 'scf.for', ...
    line: int                       # 0-based line index in the printed IR
    depth: int                      # region nesting depth
    text: str = ''
    results: list[Value] = field(default_factory=list)
    operands: list[str] = field(default_factory=list)
    attrs: str = ''
    parent: 'Op | None' = None
    children: list['Op'] = field(default_factory=list)

    @property
    def dialect(self) -> str:
        return self.name.split('.', 1)[0]

    def attr(self, key: str, pattern: str = r'([-\w."<>]+)'):
        m = re.search(rf'{re.escape(key)}\s*=\s*{pattern}', self.attrs or self.text)
        return m.group(1) if m else None

    def has_attr(self, key: str) -> bool:
        return key in (self.attrs or '') or key in self.text

    def walk(self) -> Iterator['Op']:
        yield self
        for c in self.children:
            yield from c.walk()

    def ancestors(self) -> Iterator['Op']:
        p = self.parent
        while p is not None:
            yield p
            p = p.parent


class Module:
    """Parsed IR plus a cache of analyses computed over it."""

    def __init__(self, text: str, ops: list[Op], values: dict[str, Value],
                 source: str = '', backend: str = 'text'):
        self.text = text
        self.lines = text.split('\n')
        self.ops = ops                     # flat, in source order
        self.values = values
        self.source = source
        self.backend = backend
        self._analyses: dict[type, object] = {}

    # -- queries feature sets rely on -------------------------------------
    @property
    def dialects(self) -> set[str]:
        if 'dialects' not in self._analyses:
            self._analyses['dialects'] = {o.dialect for o in self.ops}
        return self._analyses['dialects']

    def ops_named(self, *names: str) -> list[Op]:
        s = set(names)
        return [o for o in self.ops if o.name in s]

    def ops_in_dialect(self, dialect: str) -> list[Op]:
        return [o for o in self.ops if o.dialect == dialect]

    def count(self, *names: str) -> int:
        return len(self.ops_named(*names))

    def roots(self) -> list[Op]:
        return [o for o in self.ops if o.parent is None]

    # -- cached analyses (see analysis.py) --------------------------------
    def analysis(self, cls):
        if cls not in self._analyses:
            self._analyses[cls] = cls(self)
        return self._analyses[cls]


# ─────────────────────────────────────────────────────────────────────────
# text backend
# ─────────────────────────────────────────────────────────────────────────
_RE_LOC = re.compile(r'\s*loc\((?:[^()]|\([^()]*\))*\)\s*$')
_RE_RESULTS = re.compile(r'^\s*((?:%[\w$.]+(?::\d+)?\s*,\s*)*%[\w$.]+(?::\d+)?)\s*=\s*')
_RE_OPNAME = re.compile(r'^\s*"?([a-zA-Z_][\w]*\.[\w.$]+)"?')
_RE_BAREOP = re.compile(r'^\s*(return|yield|br|cond_br)\b')
_RE_SSA = re.compile(r'%[\w$.]+(?:#\d+)?')
_RE_ATTRS = re.compile(r'\{[^{}]*\}')
_RE_BLOCKARG = re.compile(r'^\s*\^(\w+)\s*\(([^)]*)\)\s*:')


def _strip_attrs(s: str) -> str:
    """Remove {...} attribute dicts, innermost first, so brace counting on the
    remaining text only sees region braces."""
    prev = None
    while prev != s:
        prev, s = s, _RE_ATTRS.sub('', s)
    return s


class TextParser:
    """Parses printed MLIR. Handles unregistered dialects; types are heuristic."""

    FUNC_LIKE = ('func.func', 'llvm.func', 'gpu.func', 'spirv.func')

    def parse(self, text: str, source: str = '') -> Module:
        ops: list[Op] = []
        values: dict[str, Value] = {}
        stack: list[Op] = []
        depth = 0
        # SSA names restart at %0 in every function, so keys are scope-qualified
        scope = 0

        def key(nm: str) -> str:
            return f'{scope}|{nm}'

        for i, raw in enumerate(text.split('\n')):
            line = _RE_LOC.sub('', raw.split(' // ', 1)[0]).rstrip()
            s = line.strip()
            if not s:
                continue

            # close regions before anything else on the line.
            # Strip attribute dicts first: `} {hivm.tcore_type = ...}` closes
            # ONE region, not two.
            if s.startswith('}'):
                body = _strip_attrs(s)
                for _ in range(min(body.count('}'), len(stack))):
                    stack.pop()
                    depth -= 1
                for _ in range(body.count('{')):       # "} else {"
                    stack.append(stack[-1] if stack else None)
                    depth += 1
                continue

            m = _RE_BLOCKARG.match(s)
            if m:                                       # ^bb0(%arg: t, ...):
                for part in _split_top(m.group(2)):
                    if ':' in part:
                        nm, ty = part.split(':', 1)
                        nm, ty = nm.strip(), ty.strip()
                        if nm.startswith('%'):
                            values[key(nm)] = Value(key(nm), ty,
                                                    parse_shaped_type(ty), None, i, i)
                continue

            op = self._parse_op(s, i, depth, values, key)
            if op is None:
                continue
            if op.name in self.FUNC_LIKE and depth == 0:
                scope += 1                               # new SSA namespace
                op = self._parse_op(s, i, depth, values, key)
            if stack and stack[-1] is not None:
                op.parent = stack[-1]
                stack[-1].children.append(op)
            ops.append(op)

            # mark uses
            for nm in op.operands:
                v = values.get(nm) or values.get(nm.split('#')[0])
                if v is not None:
                    v.last_use_line = max(v.last_use_line, i)

            if self._opens_region(s):
                stack.append(op)
                depth += 1

        return Module(text, ops, values, source, backend='text')

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _opens_region(s: str) -> bool:
        return _strip_attrs(s).rstrip().endswith('{')

    def _parse_op(self, s: str, line: int, depth: int,
                  values: dict[str, Value], key) -> Op | None:
        rest = s
        res_names: list[str] = []
        m = _RE_RESULTS.match(s)
        if m:
            for tok in m.group(1).split(','):
                tok = tok.strip()
                if ':' in tok:               # %84:2  -> %84#0, %84#1
                    base, n = tok.split(':')
                    res_names += [f'{base}#{k}' for k in range(int(n))]
                else:
                    res_names.append(tok)
            rest = s[m.end():]

        mo = _RE_OPNAME.match(rest)
        if mo:
            name = mo.group(1)
        elif _RE_BAREOP.match(rest):
            name = 'builtin.' + _RE_BAREOP.match(rest).group(1)
        else:
            return None

        attrs = ' '.join(_RE_ATTRS.findall(s))
        type_str = self._result_type(s)
        shaped = parse_shaped_type(type_str) if type_str else None

        op = Op(name=name, line=line, depth=depth, text=s, attrs=attrs)
        for nm in res_names:
            v = Value(key(nm), type_str, shaped, op, line, line)
            values[key(nm)] = v
            op.results.append(v)
        body = rest[mo.end():] if mo else rest
        op.operands = [key(t) for t in _RE_SSA.findall(body) if t not in res_names]
        return op

    @classmethod
    def _result_type(cls, s: str) -> str:
        """
        Heuristic for the text backend: take what follows the last top-level
        '->', else what follows the last top-level ':', then the LAST
        comma-separated element of that -- because many ops print
        `: <operand types>, <result type>` with no arrow
        (e.g. vector.transfer_read ... : memref<...>, vector<...>).
        Multi-result ops get the same type on every result.
        """
        tail = cls._tail(s)
        parts = [p for p in _split_top(tail) if p.strip()]
        return parts[-1].strip() if parts else ''

    @staticmethod
    def _tail(s: str) -> str:
        depth, arrow, colon = 0, -1, -1
        i = 0
        while i < len(s):
            ch = s[i]
            if ch in '(<[{':
                depth += 1
            elif ch in ')>]}':
                depth -= 1
            elif depth == 0 and ch == '-' and s[i:i + 2] == '->':
                arrow = i + 2
                i += 1
            elif depth == 0 and ch == ':' and s[i:i + 2] != '::':
                colon = i + 1
            i += 1
        cut = arrow if arrow >= 0 else colon
        return s[cut:].strip() if cut >= 0 else ''


def parse_text(text: str, source: str = '') -> Module:
    return TextParser().parse(text, source)


def parse_file(path: str, prefer_bindings: bool = True) -> Module:
    text = open(path, errors='replace').read()
    if prefer_bindings:
        try:
            from .backend_bindings import parse_with_bindings
            return parse_with_bindings(text, path)
        except Exception:
            pass                                  # fall back to text
    return parse_text(text, path)
