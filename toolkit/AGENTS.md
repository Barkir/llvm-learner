# Compiler-corpus analysis toolkit — instructions for an AI agent

This toolkit answers one question, in many disguises:

> **Which measurable property of a compilation unit predicts some outcome,
> and can that property be written down as a rule?**

"Outcome" is anything you can observe after the fact: a retry, a slowdown, a
crash, a wrong answer, a pass firing, a flag being the right choice. The method
does not change between them. Only the labels change.

Read sections 1–3 before running anything. Section 4 is a catalogue of worked
cases; find the one closest to your task and copy its slot table.

---

## 1. The four slots

Fill these in **before** touching a command. Write them down in your reply; if
you cannot fill one, you are not ready to run anything.

| slot | question | typical answer |
|---|---|---|
| **UNIT** | what is one row? | one kernel (`COLLECTED/<hash>/kernel.mlir`) |
| **FEATURES** | what can be measured *at the moment the decision is made*? | `mlirfeat` over the IR at that pipeline stage |
| **LABEL** | the outcome, as 0/1 per unit | a labels CSV, or "this unit produced a dump" |
| **GROUP** | which rows are near-copies of each other? | the source function, e.g. `struct_entry_name` |

### What GROUP means

Cross-validation hides part of the data and tests on it, to estimate accuracy
on units the model has never seen. GROUP decides what "never seen" means.

A corpus of 700 kernels is rarely 700 independent things: the same source
function appears many times with different shapes or block sizes, and those
variants have nearly identical features. If one variant is in training and
another in test, the model can recognise the family and recall its label
instead of learning anything. The score then measures memorisation.

Measured on a corpus where the outcome was a coin flip per family, so the
features contained *no* information about it:

```
750 units, 150 source functions (5 variants each)
  ungrouped   ROC-AUC 1.000 +- 0.000     <- pure memorisation
  grouped     ROC-AUC 0.608 +- 0.074
```

Set GROUP to the column whose equal values mean "near-copies": the source
function for kernels, the kernel hash when rows are *(kernel, configuration)*
pairs, the template for generated corpora. If the units are genuinely
unrelated, pass `--group-col ''`.

Check whether you need it:

```python
print(len(f), 'rows,', f.struct_entry_name.nunique(), 'distinct functions')
```

Far fewer groups than rows means grouping is mandatory. The two numbers answer
different questions — ungrouped is "a new variant of a known kernel", grouped
is "a kernel family never seen before" — and the grouped one is the one you can
defend.

**The one rule that cannot be bent: a feature must be computable before the
outcome is known.** A feature derived from a dump that only exists because the
outcome happened will score near-perfectly and predict nothing. This is the
single most common way to produce a confident wrong answer. If you are unsure
whether a feature is legitimate, ask: *could the compiler read this value at
the point where it must decide?* If no, drop it.

A second, weaker rule: features must be **cheap** if the rule is going to run
inside the compiler. A feature requiring a full lowering is fine for offline
triage, useless in a cost model.

---

## 2. The procedure

Identical for every use case. `$FEATURES`, `$LABELS`, `$LABEL`, `$GROUP` are the
slots from section 1.

```bash
# 1. features — one row per UNIT
python3 -m mlirfeat <corpus> --file-name kernel.mlir -o $FEATURES

# 2. labels — see section 4 for how each outcome is measured

# 3. is there any signal at all?
python3 rule_search.py $FEATURES --labels $LABELS --label-col $LABEL --rank-features

# 4. are the labels learnable? (section 3, pitfall 2)

# 5. simplest thing that works: a threshold rule
python3 rule_search.py $FEATURES --labels $LABELS --label-col $LABEL \
        --feature <best feature> --group-col <splitter> --list-kernels

# 6. only if a rule is not enough
python3 train_retry_type_models.py $FEATURES --labels $LABELS \
        --label-cols $LABEL ./models --group-col $GROUP

# 7. ship it
python3 rule_search.py ... --save-rule rule.json --emit-cpp rule.h
```

### Reading step 3

| best AUC | meaning | do next |
|---|---|---|
| ≥ 0.8 | one feature carries the signal | step 5, build the rule |
| 0.6–0.8 | weak, maybe a combination | step 4, then step 6 |
| ≤ 0.6 | the features do not contain the answer | **stop**; get better features (section 5). A bigger model will not help. |

### Reading step 6

Compare PR-AUC against the printed base rate, never against zero. **If the
model does not clearly beat the single-feature rule from step 5, keep the
rule**: it is inspectable, it compiles into the compiler, and it cannot drift.

---

## 3. Pitfalls

Each of these has already produced a wrong answer at least once.

1. **Leaking the outcome into the features.** See section 1.
   `train_retry_type_models.py` drops every `end_`-prefixed column for this
   reason. If you build features yourself, drop them yourself.
2. **Contradictory labels cap everything.** Units with identical feature
   vectors but different labels make a perfect model impossible. Measure it
   before blaming the model:
   ```python
   import pandas as pd
   f = pd.read_csv('$FEATURES', dtype={'kernel': str})
   lab = pd.read_csv('$LABELS', dtype={'kernel': str}).set_index('kernel')['$LABEL']
   f['y'] = f.kernel.map(lab).fillna(0).astype(int)
   cols = [c for c in f.columns if c not in ('kernel', 'y', 'file', 'applied_sets')]
   g = f.groupby(cols, dropna=False)['y'].agg(['size', 'sum'])
   mixed = g[(g['sum'] > 0) & (g['sum'] < g['size'])]
   print(f'{len(g)} distinct vectors / {len(f)} units; '
         f'{mixed["size"].sum()} units in contradictory groups')
   ```
   A large number here *is* the finding. Report it and go to section 5.
3. **Ungrouped cross-validation inflates every score.** Kernels from the same
   source function are near duplicates. Always set `--group-col`.
4. **Judge against the base rate.** With 19 positives in 700 units, accuracy
   0.97 is worse than useless and PR-AUC 0.40 is 15× better than random.
   Never report accuracy for a rare outcome.
5. **Low precision is normal for rare events.** 18 true out of 42 flagged is a
   good result at a 2.7% base rate. Phrase it as "4 in 10 flags are real".
6. **A cutoff sitting exactly on one unit's value is overfitted.**
   `rule_search.py` rounds down by default; keep that unless rounding costs
   real false alarms.
7. **Split by the thing that changes the physics.** On this target, `mix`
   kernels put matmul operands in L1/L0 while `aiv` kernels put everything in
   UB, so the same tensor size means different memory pressure. Always try
   `--group-col struct_mix_mode` before concluding a feature is weak. The
   analogous splitter exists in most tasks: core type, dtype, pipeline stage.
8. **Absence of evidence is not evidence of absence.** A missing dump, a
   missing timing, a kernel that is not in the labels CSV — confirm it means
   "did not happen" and not "did not run". `collect_pass_dataset.py` records
   `exit_code` and `status` precisely so timeouts and crashes are not counted
   as successes.
9. **Probabilities from these models are ranking scores,** not calibrated
   probabilities; the rare class is up-weighted during training.
10. **Column names differ by extractor.** `mlirfeat` prefixes everything
    (`struct_mix_mode`, `shape_total_bytes`); the older `extract_features.py`
    did not (`mix_mode`, `total_tensor_elements`). Check the CSV header before
    passing a column name.

---

## 4. Use-case catalogue

### A. Find a compiler heuristic (worked example: UB overflow)

*Want: a rule the compiler can evaluate to predict that a kernel will not fit
in on-chip memory.*

| slot | value |
|---|---|
| UNIT | one kernel |
| FEATURES | `mlirfeat` over `kernel.mlir` (the stage where the decision is made) |
| LABEL | the kernel produced an overflow dump |
| GROUP | `struct_entry_name` |

The label source here is the dumps themselves, so pass them positionally
instead of `--labels`:

```bash
python3 -m mlirfeat ~/COLLECTED --file-name kernel.mlir -o feats.csv
python3 -m mlirfeat ~/COLLECTED --pattern '*overflow*.mlir' \
        --column-prefix end_ -o dumps.csv

python3 rule_search.py feats.csv dumps.csv --type overflow --rank-features
python3 rule_search.py feats.csv dumps.csv --type overflow \
        --feature shape_total_bytes --group-col struct_mix_mode \
        --target-recall 0.95 --list-kernels --emit-cpp ub_rule.h
```

If the dump file names encode several outcomes (`uboverflow.mlir`,
`vf_stack_limit.mlir`), add an `end_retry_type` column so `--type` can select
one; both tools read it.

**Good looks like:** a cutoff that catches ≥ 90% of failures while flagging a
few percent of the healthy units, on a feature whose physical meaning explains
the failure.

---

### B. Find what causes a performance regression

*Want: which kernels got slower between two compiler builds or two flag sets,
and what they have in common.*

| slot | value |
|---|---|
| UNIT | one kernel |
| FEATURES | `mlirfeat` over the input IR |
| LABEL | `regressed` = new time is more than X% worse than old |
| GROUP | `struct_entry_name` |

The outcome is numeric, so **binarise it** — that is not a compromise, it is
the question you are actually asking:

```python
import pandas as pd
old = pd.read_csv('timings_old.csv', dtype={'kernel': str})   # kernel, seconds
new = pd.read_csv('timings_new.csv', dtype={'kernel': str})
m = old.merge(new, on='kernel', suffixes=('_old', '_new'))
m['ratio'] = m.seconds_new / m.seconds_old
m['regressed'] = (m.ratio > 1.10).astype(int)      # 10% slower
m[['kernel', 'regressed', 'ratio']].to_csv('labels.csv', index=False)
print(m.regressed.sum(), 'regressions of', len(m))
```

```bash
python3 rule_search.py feats.csv --labels labels.csv --label-col regressed \
        --rank-features
```

**Watch out:** measurement noise. Set the threshold above your run-to-run
variance, or you are modelling the noise. Check by comparing a build against
itself — the regression count should be near zero.

**Good looks like:** a feature that identifies the regressed set, pointing at
which pass or pattern the new build handles worse. The feature is the lead;
confirm it by reading the IR of two kernels that differ on it.

---

### C. Predict compiler flags

*Want: choose flags per kernel instead of compiling, failing and retrying.*

There are two framings, and they need different UNITs.

**C1 — predict the decision directly.** One row per kernel, one binary model
per flag.

| slot | value |
|---|---|
| UNIT | one kernel |
| LABEL | one 0/1 column per flag: was this flag needed? |

```bash
python3 train_retry_type_models.py feats.csv --labels flag_labels.csv \
        --label-cols flag_stack,flag_buf,flag_fusion ./models \
        --targets all --group-col struct_entry_name
```

**C2 — score configurations (a learned cost function).** One row per
*(kernel, configuration)*. Build a composite id so each row is a unit, and
**group by the real kernel** so all configurations of one kernel stay in the
same fold:

```python
cfg['kernel_hash'] = cfg.kernel                      # keep the real id
cfg['kernel'] = cfg.kernel + '|' + cfg.config_name   # composite row id
```

```bash
python3 train_retry_type_models.py cfg_features.csv --labels cfg_labels.csv \
        --label-cols retried ./models --group-col kernel_hash
```

At inference, score every configuration for a kernel and take the best. C2
handles large or changing configuration spaces; C1 is one cheap call. Use C1
when the choice is a small fixed set.

**Watch out:** if the same kernel appears under several configurations and you
do not group by `kernel_hash`, the scores are meaningless.

---

## 5. When the features are the problem

Step 3 said ≤ 0.6, or step 4 found many contradictory groups. In order of cost:

1. **Richer features from the same IR.** Add a feature set to `mlirfeat`; see
   `mlirfeat/README.md` and `my_features.py`. Liveness, per-dtype sizes and
   buffer-producing ops are already implemented. Look at what a positive unit
   has that a negative one of similar size does not — `--list-kernels` gives
   you the pairs to diff.
2. **Features from a later pipeline stage,** collected for *every* unit with
   `collect_pass_dataset.py --keep-all` (section 5.1). The information often exists
   in the IR a few passes later: on this corpus `mem_ub_peak_ratio` reaches
   1.06 on a kernel known to overflow, while the input IR shows nothing.
   Cost: you must run part of the compiler to use the rule.
3. **Information that is not in the IR at all** — runtime shapes, launch
   configuration, dynamic dimensions. Log it at compile time and join it in.
4. **More examples.** Only fixes noisy estimates, never missing information.
   Check the learning curve printed by `train_retry_type_models.py`: flat means
   more units will not help.


### 5.1 Collecting IR from the compiler

`collect_pass_dataset.py` runs the compiler over a corpus, captures the IR
printed around one pass, and writes a dataset. It is how you get both

* **features from a later pipeline stage** — the dumps become the input to
  `mlirfeat`, so you can model on IR the front end does not show; and
* **labels** — `manifest.csv` records, per kernel, whether the pass ran and
  whether it changed anything.

**Always probe one input first.** It prints the exact command, the exit code
and any dump banners found, then stops:

```bash
python3 collect_pass_dataset.py --preset bishengir \
        --pass hivm-graph-sync-solver \
        --inputs ~/.triton/dump --out /tmp/probe --probe
```

A healthy probe names the passes it parsed and how many matched `--pass`. If it
reports no banners, the cause is almost always one of: wrong flag namespace
(`--print-flag-prefix`), a C++ class name where the registered CLI name was
needed (`--list-passes <substring>`, or `--select all` to skip the lookup), or
the pass genuinely not running on that kernel.

**Then the real run:**

```bash
python3 collect_pass_dataset.py --preset bishengir \
        --pass hivm-graph-sync-solver \
        --inputs ~/.triton/dump --out ~/DATA/sync -j 16 --keep-all
```

`--keep-all` matters whenever the result feeds a model: without it you only
record the kernels the pass touched, and you cannot learn what distinguishes
them from the rest.

**Commands.** `--preset NAME` uses a built-in template (`--list-presets` to see
them; the `PRESETS` dict at the top of the script is where to add your own).
`--cmd` spells one out instead, with placeholders `{input}` `{indir}` `{stem}`
`{kernel}` `{workdir}` `{outdir}` `{target}`. Never put `--mlir-print-ir-*`
flags in either — the script appends those from `--pass`. `--dry-run` prints
the assembled command without running it.

**Flags worth knowing:**

| flag | when |
|---|---|
| `--print-flag-prefix bishengir` | driver namespaces its own IR-printing flags |
| `--after-only` | driver has no `...-print-ir-before` |
| `--select all` | you only know the C++ class name; dumps everything and filters here |
| `--changed-only` | adds `...-print-ir-after-change`, keeps only kernels the pass modified |
| `--module-scope` | print the whole module (adds `--mlir-disable-threading`) |
| `--workdir temp\|out\|cwd` | where each compile runs; `temp` (default) stops parallel jobs overwriting each other's output files |
| `--save-artifacts` | keep what the compiler produced, under `<out>/<kernel>/artifacts/` |
| `--id-mode parent` | name each row after the containing directory — the Triton hash, so it joins to your other CSVs |
| `--resume` | skip kernels already done, reusing their recorded results |
| `--limit 20 -v` | small trial run before committing to the corpus |

**Outputs.** Per kernel, `<out>/<kernel>/` holds `before_NNN.mlir`,
`after_NNN.mlir` (one pair per time the pass ran — a pass nested under a
function runs once per function) and `meta.json`. Across the run,
`<out>/manifest.csv` has one row per kernel with `ran`, `changed`, `n_before`,
`n_after`, `scopes`, `exit_code`, `seconds`, `status`.

**Feeding it back in:**

```bash
# features from the captured stage
python3 -m mlirfeat ~/DATA/sync --pattern 'before_000.mlir' \
        --column-prefix pre_ -o pre_feats.csv

# labels: which kernels the pass changed
python3 rule_search.py feats.csv --labels ~/DATA/sync/manifest.csv \
        --label-col changed --rank-features
```

Check `status` and `exit_code` before trusting any of it: a kernel that timed
out or crashed is not a kernel the pass left alone (pitfall 8).

---

## 6. How to report

In this order:

1. the four slots you chose, in one line each;
2. the base rate (`N positives / M units`);
3. the metric against that base rate, with fold-to-fold spread;
4. what the rule would do in production, in counts: "flags 42 of 700, catches
   18 of 19, 24 false alarms";
5. the feature that carries the signal **and the physical reason it should**;
6. the single biggest reason the result might be wrong.

A result you cannot explain physically is a result you should not ship.

---

## 7. Quick reference

```bash
# features
python3 -m mlirfeat --list                        # available feature sets
python3 -m mlirfeat <file.mlir> --print           # one unit, readable
python3 -m mlirfeat <root> --file-name kernel.mlir -o feats.csv
python3 -m mlirfeat <root> --pattern '*.mlir' --column-prefix end_ -o dumps.csv

# signal and rules  (labels from a CSV, or positionally from dumps)
python3 rule_search.py feats.csv --labels labels.csv --label-col Y --rank-features
python3 rule_search.py feats.csv dumps.csv --type overflow --rank-features
python3 rule_search.py feats.csv --labels labels.csv --label-col Y \
        --feature X --group-col struct_mix_mode --list-kernels \
        --save-rule r.json --emit-cpp r.h

# models
python3 train_retry_type_models.py feats.csv --labels labels.csv \
        --label-cols Y1,Y2 ./models --targets all --group-col struct_entry_name
python3 train_retry_type_models.py --predict new.csv --model-dir ./models

# second condition to cut false alarms
python3 fp_analysis.py feats.csv dumps.csv --feature X --threshold T

# corpus collection (section 5.1)
python3 collect_pass_dataset.py --list-presets              # built-in commands
python3 collect_pass_dataset.py --preset bishengir --pass NAME \
        --inputs <corpus> --out <dir> --probe               # try ONE input first
python3 collect_pass_dataset.py --preset bishengir --pass NAME \
        --inputs <corpus> --out <dir> -j 16 --keep-all
python3 collect_pass_dataset.py --cmd '<tool> {input} --pipe' --list-passes NAME
#   --print-flag-prefix bishengir   driver namespaces its own print flags
#   --after-only                    no ...-print-ir-before on this driver
#   --select all                    filter by name here instead of via the flag
#   --workdir temp|out|cwd          temp = each compile isolated (default)
#   --save-artifacts                keep the compiler's output files
#   --dry-run                       print the assembled command, run nothing
```

Useful `mlirfeat` columns:

| column | meaning |
|---|---|
| `shape_total_bytes` | sum over **distinct SSA values**, each with its own dtype |
| `mem_<space>_peak_bytes` | bytes live **at the same time**, per function, max over functions |
| `mem_<space>_peak_ratio` | that peak ÷ the target-spec limit; `> 1.0` does not fit |
| `mem_worst_space` | which memory space is tightest |
| `struct_mix_mode` | `aiv` / `mix`, from the entry function |
| `struct_stage` | `high` / `hir` / `llvm` — which pipeline stage the file is from |
| `loop_max_depth`, `loop_max_trip_count` | nesting from the region tree |
| `applied_sets` | which feature sets ran; a 0 elsewhere means "not applicable", not "measured 0" |

Target-spec sizes in the IR are in **bits**; `mlirfeat`'s `target_*_size`
columns are already bytes.
