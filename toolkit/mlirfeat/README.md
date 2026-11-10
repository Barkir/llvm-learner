# mlirfeat

Pluggable feature extraction from MLIR kernels. Feature sets declare which
dialects they need, so a kernel using both `affine` and `hivm` gets both sets,
and kernels missing a dialect still produce the same CSV columns.

```bash
python3 -m mlirfeat ~/COLLECTED --file-name kernel.mlir -o kernel_features.csv
python3 -m mlirfeat ~/COLLECTED --pattern '*overflow*.mlir' --column-prefix end_ -o end_features.csv
python3 -m mlirfeat one_kernel.mlir --print      # one row, readable
python3 -m mlirfeat --list                       # what is registered
```

## Layers

```
ir.py          Module / Op / Value / ShapedType  + two parsers
   TextParser        pure Python, handles unregistered dialects (hivm, hacc)
   BindingsParser    real MLIR Python bindings, used automatically if importable
       |
analysis.py    expensive things, cached per module: Loops, Constants,
               TargetSpec, Liveness, ShapedValues
       |
featureset.py  FeatureSet base + registry + Extractor driver
       |
sets/          the feature sets themselves
```

A feature set never touches text or parsers; it asks the module and the
analyses. Swapping the text backend for the real bindings changes nothing above
`ir.py`.

## Adding a feature set

Subclass `FeatureSet` anywhere and import the module. Defining the class
registers it.

```python
from mlirfeat import FeatureSet, Loops

class MyFeatures(FeatureSet):
    name = 'mine'                      # column prefix: mine_n_anchors, ...
    requires_dialects = {'hivm'}       # omit -> always applies
    features = ['n_anchors', 'anchors_per_loop']

    def extract(self, module):
        anchors = module.count('hivm.hir.anchor')
        loops = module.analysis(Loops)          # computed once, shared
        return {'n_anchors': anchors,
                'anchors_per_loop': round(anchors / max(loops.n_loops, 1), 2)}
```

The driver handles the rest:

* **columns always line up.** A set that does not apply contributes its columns
  filled with `default`, so rows from different kernels stay joinable.
* **no name clashes.** Columns are prefixed with the set's `name`.
* **undeclared keys are an error.** Returning a key not in `features` raises,
  so the CSV header cannot drift from the code.
* **one failure does not lose the row.** An exception in one set is reported and
  that set's columns fall back to `default` (use `--strict` to raise instead).
* `applies_sets` in the output records which sets ran for each kernel.

Override `applies(module)` when dialect presence is not the right test — see
`TargetSpecFeatures` (needs a dlti spec) or `OverflowRiskFeatures` in
`my_features.py`.

For "count these ops" sets, subclass `OpCountFeatures` in `sets/dialects.py`
and just list the op names; the columns are generated for you.

## Adding an analysis

```python
from mlirfeat import Analysis

class MyAnalysis(Analysis):
    def __init__(self, module):
        super().__init__(module)
        self.result = ...           # compute here

# in a feature set:
module.analysis(MyAnalysis).result  # computed once per module, then cached
```

## What the built-in sets give you

| set | applies when | notable features |
|---|---|---|
| `target` | module has a dlti target spec | UB/L1/L0 sizes **in bytes** (the spec stores bits) |
| `struct` | always | op/function counts, dialect histogram, pipeline stage, entry signature |
| `loop` | always | nesting from the region tree, constant trip counts |
| `shape` | always | sizes of **distinct SSA values**, each with its own element type |
| `mem` | always | per memory space: peak bytes **live at once**, ratio to the limit |
| `affine` | affine dialect | maps, symbols/dims, loop depth |
| `hivm` | hivm dialect | DMA, sync, multi-buffer, tightly-coupled buffers, core types |
| `intr` | hivm_regbaseintrins | vector intrinsics bucketed by kind, compute:memory ratio |
| `linalg`, `tensor`, `llvm` | that dialect | structured ops, buffer-producing ops, LLVM-stage ops |

Two things worth knowing, because they differ from a regex extractor:

* `shape_total_bytes` counts each SSA value once, not once per textual mention
  of its type, and uses that value's own dtype.
* `mem_<space>_peak_bytes` is the peak of what is **live at the same time**,
  computed per function and maximised (backup copies of functions excluded),
  with `hivm.multi_buffer` and the space's alignment applied.

## Backends

The text parser runs everywhere and needs no MLIR build. Its one heuristic is
result-type extraction from the printed assembly: it takes the last
comma-separated type after the final top-level `->` or `:`, and gives every
result of a multi-result op the same type.

If you have MLIR built with `-DMLIR_ENABLE_BINDINGS_PYTHON=ON`,
`backend_bindings.py` is used automatically and types become exact. Set
`MLIRFEAT_IR_MODULE` if your toolchain ships the bindings under another module
name. **That file is the one part not tested here** — the bindings were not
installed in the environment where this was written. Everything else was run
against real kernels; `--no-bindings` forces the text parser.
