"""White list for the static analyzer 'vulture'."""

from tests.test_visualization import Axes3D
from weldx.transformations import LocalCoordinateSystem, WeldxAccessor

Axes3D  # unused import

WeldxAccessor  # unused class TODO: remove if class gets own tests

# unused attributes
lcs = LocalCoordinateSystem()
lcs.base.name
lcs.origin.name
