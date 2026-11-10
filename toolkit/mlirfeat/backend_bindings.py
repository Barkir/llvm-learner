"""
Optional exact backend: the real MLIR Python bindings.

Only used if `import mlir.ir` works (a build with -DMLIR_ENABLE_BINDINGS_PYTHON=ON;
your toolchain may ship it under a different module name — set MLIRFEAT_IR_MODULE).
parse_file() falls back to the text parser when this import fails, so nothing
else in the package depends on it.

NOTE: this is the one file I could not run here, because the bindings are not
installed in my sandbox. The text backend is tested.
"""

from __future__ import annotations

import os

from .ir import Module, Op, ShapedType, Value, parse_shaped_type

_IR_MODULE = os.environ.get('MLIRFEAT_IR_MODULE', 'mlir.ir')


def _import_ir():
    import importlib
    return importlib.import_module(_IR_MODULE)


def _shaped_from_type(t) -> ShapedType | None:
    """Prefer the bindings' own accessors, fall back to parsing the printed type."""
    ir = _import_ir()
    try:
        if ir.ShapedType.isinstance(t):
            st = ir.ShapedType(t)
            dims = tuple(None if st.is_dynamic_dim(i) else st.get_dim_size(i)
                         for i in range(st.rank))
            elem = str(st.element_type)
            kind = str(t).split('<', 1)[0]
            space = 'default'
            if ir.MemRefType.isinstance(t):
                ms = ir.MemRefType(t).memory_space
                if ms is not None:
                    s = str(ms)
                    space = s.split('<')[-1].rstrip('>') if '<' in s else s
            return ShapedType(kind, dims, elem, space)
    except Exception:
        pass
    return parse_shaped_type(str(t))


def parse_with_bindings(text: str, source: str = '') -> Module:
    ir = _import_ir()
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True       # hivm, hacc, ...
    with ctx, ir.Location.unknown():
        mod = ir.Module.parse(text)

    ops: list[Op] = []
    values: dict[str, Value] = {}
    counter = [0]

    def line_of(mlir_op) -> int:
        """No real line numbers from the bindings; visit order is enough for
        liveness, which only needs a consistent ordering."""
        counter[0] += 1
        return counter[0]

    def visit(mlir_op, parent: Op | None, depth: int):
        op = Op(name=str(mlir_op.operation.name), line=line_of(mlir_op),
                depth=depth, text=str(mlir_op).split('\n', 1)[0],
                attrs=str(mlir_op.operation.attributes), parent=parent)
        for i, res in enumerate(mlir_op.results):
            name = f'{id(mlir_op):x}#{i}'
            v = Value(name, str(res.type), _shaped_from_type(res.type),
                      op, op.line, op.line)
            values[name] = v
            op.results.append(v)
        for operand in mlir_op.operands:
            owner = getattr(operand, 'owner', None)
            key = None
            if owner is not None and hasattr(owner, 'results'):
                for i, r in enumerate(owner.results):
                    if r == operand:
                        key = f'{id(owner):x}#{i}'
                        break
            if key:
                op.operands.append(key)
                if key in values:
                    values[key].last_use_line = max(values[key].last_use_line, op.line)
        if parent is not None:
            parent.children.append(op)
        ops.append(op)
        for region in mlir_op.regions:
            for block in region.blocks:
                for child in block.operations:
                    visit(child, op, depth + 1)

    for region in mod.operation.regions:
        for block in region.blocks:
            for child in block.operations:
                visit(child, None, 0)

    m = Module(text, ops, values, source, backend='bindings')
    return m
