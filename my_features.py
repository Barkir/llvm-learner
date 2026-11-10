"""
Adding your own feature sets — no changes to the package needed.

    python3 -c "import my_features; from mlirfeat import Extractor; print(Extractor().describe())"
    python3 -m mlirfeat ... --only mine,risk        # after importing this module

Run it as a script to extract with your sets included.
"""

from mlirfeat import Extractor, FeatureSet, Liveness, Loops, TargetSpec, parse_file


# 1 — the minimum: a name, the columns, and extract()
class MyOpFeatures(FeatureSet):
    name = 'mine'
    requires_dialects = {'hivm'}          # omit to run on every kernel
    features = ['n_anchors', 'n_marks', 'anchors_per_loop']

    def extract(self, module):
        anchors = module.count('hivm.hir.anchor')
        loops = module.analysis(Loops)                 # cached, shared with other sets
        return {
            'n_anchors': anchors,
            'n_marks': module.count('annotation.mark'),
            'anchors_per_loop': round(anchors / max(loops.n_loops, 1), 2),
        }


# 2 — derived features over an analysis, with a custom applies()
class OverflowRiskFeatures(FeatureSet):
    """Headroom against the declared hardware limits — the quantities a cost
    model would actually branch on."""
    name = 'risk'
    features = ['ub_headroom_bytes', 'ub_fill', 'over_any_limit',
                'tightest_space', 'n_spaces_over_80pct']
    default = 0

    def applies(self, module) -> bool:
        return bool(module.analysis(TargetSpec).sizes)     # needs a target spec

    def extract(self, module):
        live = module.analysis(Liveness)
        spec = module.analysis(TargetSpec)
        ratios = {}
        for space, peak in live.peak_bytes.items():
            limit = spec.limit_for(space)
            if limit:
                ratios[space] = peak / limit
        ub_limit = spec.limit_for('ub') or 0
        tightest = max(ratios, key=ratios.get) if ratios else 'none'
        return {
            'ub_headroom_bytes': ub_limit - live.peak_bytes.get('ub', 0),
            'ub_fill': round(ratios.get('ub', 0), 4),
            'over_any_limit': int(any(r > 1.0 for r in ratios.values())),
            'tightest_space': tightest,
            'n_spaces_over_80pct': sum(1 for r in ratios.values() if r > 0.8),
        }


if __name__ == '__main__':
    import sys
    ex = Extractor()
    print(ex.describe())
    for path in sys.argv[1:]:
        row = ex.extract(parse_file(path))
        print(f'\n{path}')
        for k in ('mine_n_anchors', 'risk_ub_fill', 'risk_over_any_limit',
                  'risk_tightest_space', 'risk_ub_headroom_bytes'):
            print(f'  {k:28s} {row.get(k)}')
