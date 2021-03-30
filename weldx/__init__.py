"""WeldX data format and analysis package."""
# isort:skip_file
import warnings

try:
    from ._version import __version__
except ModuleNotFoundError:  # pragma: no cover
    __version__ = None
    warnings.warn(
        "Using local weldx package files without version information.\n"
        "Consider running 'python setup.py --version' or 'pip install -e .' "
        "in the weldx root repository",
        category=UserWarning,
    )

# main modules
import weldx.transformations  # import this first to avoid circular dependencies
import weldx.util
import weldx.core
import weldx.geometry
import weldx.welding

# asdf extensions and tags
import weldx.asdf

# class imports to weldx namespace
from weldx.constants import WELDX_QUANTITY as Q_
from weldx.core import MathematicalExpression, TimeSeries
from weldx.geometry import (
    ArcSegment,
    Geometry,
    LineSegment,
    Profile,
    Shape,
    Trace,
    SpatialData,
)
from weldx.transformations import (
    CoordinateSystemManager,
    LocalCoordinateSystem,
    WXRotation,
)
from weldx.welding.processes import GmawProcess
from weldx.welding.groove.iso_9692_1 import get_groove

__all__ = (
    "ArcSegment",
    "CoordinateSystemManager",
    "Geometry",
    "GmawProcess",
    "LineSegment",
    "LocalCoordinateSystem",
    "Profile",
    "Q_",
    "Shape",
    "SpatialData",
    "Trace",
    "asdf",
    "core",
    "geometry",
    "get_groove",
    "get_groove",
    "measurement",
    "transformations",
    "util",
    "welding",
)
