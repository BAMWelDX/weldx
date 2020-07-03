"""White list for the static analyzer 'vulture'."""

from tests.test_visualization import Axes3D

from weldx.transformations import LocalCoordinateSystem, WeldxAccessor

Axes3D  # unused import

WeldxAccessor  # unused class TODO: remove if class gets own tests

_ipython_display_

# unused attributes
lcs = LocalCoordinateSystem()
lcs.coordinates.name
lcs.orientation.name
