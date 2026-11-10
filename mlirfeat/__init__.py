"""
mlirfeat — pluggable feature extraction from MLIR kernels.

    from mlirfeat import Extractor, parse_file
    row = Extractor().extract(parse_file('kernel.mlir'))

Adding a feature set:

    from mlirfeat import FeatureSet

    class MyFeatures(FeatureSet):
        name = 'my'
        requires_dialects = {'hivm'}        # omit to always apply
        features = ['n_widgets']
        def extract(self, module):
            return {'n_widgets': module.count('hivm.widget')}

Defining the class registers it; Extractor() picks it up.
"""

from .analysis import Analysis, Constants, Liveness, Loops, ShapedValues, TargetSpec
from .featureset import Extractor, FeatureSet
from .ir import Module, Op, ShapedType, Value, parse_file, parse_text
from . import sets  # noqa: F401  registers the built-in sets

__all__ = [
    'Analysis', 'Constants', 'Liveness', 'Loops', 'ShapedValues', 'TargetSpec',
    'Extractor', 'FeatureSet',
    'Module', 'Op', 'ShapedType', 'Value', 'parse_file', 'parse_text',
]
