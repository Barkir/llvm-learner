"""
The extension point.

Subclass FeatureSet, declare which dialects you need, and return a dict.
Subclasses register themselves, so importing your module is enough:

    class MyFeatures(FeatureSet):
        name = 'my'
        requires_dialects = {'hivm'}          # set() = always applies
        features = ['n_widgets', 'widget_bytes']

        def extract(self, module):
            return {'n_widgets': module.count('hivm.widget'), ...}

Guarantees the driver gives you:
  * every row has the same columns, in the same order, even when a set did not
    apply to that kernel (its columns are filled with `default`) — otherwise the
    CSV would not line up across kernels;
  * columns are prefixed with the set's name, so two sets can both emit
    'n_ops' without clashing;
  * one exception in one set does not lose the rest of the row.
"""

from __future__ import annotations

import csv
import sys
import traceback


class FeatureSet:
    """Base class for a group of related features."""

    # -- declare these in subclasses --------------------------------------
    name: str = ''                     # column prefix, e.g. 'affine'
    requires_dialects: set[str] = set()
    requires_ops: set[str] = set()     # applies only if one of these ops exists
    features: list[str] = []           # column names, without the prefix
    default = 0                        # value used when the set does not apply
    enabled: bool = True
    abstract: bool = False             # True on intermediate base classes

    registry: list[type['FeatureSet']] = []

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        # only the class that declares it is abstract; subclasses of it are not
        if not cls.__dict__.get('abstract', False):
            if not cls.name:
                raise TypeError(f'{cls.__name__} must set a name')
            FeatureSet.registry.append(cls)

    # -- override these ----------------------------------------------------
    def applies(self, module) -> bool:
        """Default: the module must contain every required dialect."""
        if self.requires_dialects and not self.requires_dialects <= module.dialects:
            return False
        if self.requires_ops and not any(module.count(o) for o in self.requires_ops):
            return False
        return True

    def extract(self, module) -> dict:
        raise NotImplementedError

    # -- used by the driver ------------------------------------------------
    def columns(self) -> list[str]:
        return [f'{self.name}_{f}' for f in self.features]

    def run(self, module) -> dict:
        """Prefix, check for undeclared keys, fill in anything missing."""
        raw = self.extract(module) or {}
        unknown = set(raw) - set(self.features)
        if unknown:
            raise KeyError(f'{type(self).__name__} returned undeclared features '
                           f'{sorted(unknown)}; add them to `features`')
        return {f'{self.name}_{f}': raw.get(f, self.default) for f in self.features}

    def blank(self) -> dict:
        return {c: self.default for c in self.columns()}


class Extractor:
    """Runs every registered feature set over a module."""

    def __init__(self, sets: list[FeatureSet] | None = None,
                 only: set[str] | None = None, skip: set[str] | None = None):
        if sets is None:
            sets = [cls() for cls in FeatureSet.registry if cls.enabled]
        if only:
            sets = [s for s in sets if s.name in only]
        if skip:
            sets = [s for s in sets if s.name not in skip]
        self.sets = sets

    @property
    def columns(self) -> list[str]:
        cols = ['backend', 'applied_sets']
        for s in self.sets:
            cols += s.columns()
        return cols

    def extract(self, module, strict: bool = False) -> dict:
        row = {'backend': module.backend}
        applied = []
        for s in self.sets:
            try:
                if s.applies(module):
                    row.update(s.run(module))
                    applied.append(s.name)
                else:
                    row.update(s.blank())
            except Exception:
                if strict:
                    raise
                print(f'  ! {s.name} failed on {module.source}:', file=sys.stderr)
                traceback.print_exc(limit=2, file=sys.stderr)
                row.update(s.blank())
        row['applied_sets'] = '+'.join(applied)
        return row

    def to_csv(self, rows: list[dict], path: str, extra_cols: list[str] = ()):
        cols = list(extra_cols) + self.columns
        with open(path, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=cols, restval=0, extrasaction='ignore')
            w.writeheader()
            w.writerows(rows)
        return cols

    def describe(self) -> str:
        out = []
        for s in self.sets:
            need = ', '.join(sorted(s.requires_dialects)) or 'always'
            out.append(f'{s.name:10s} [{need}]  {len(s.features)} features')
        return '\n'.join(out)
