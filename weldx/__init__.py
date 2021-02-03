"""
isort:skip_file
"""
import warnings

try:
    from ._version import __version__
except ModuleNotFoundError:  # pragma: no cover
    __version__ = None
    warnings.warn(
        "Using local weldx package files without version information.\n"
        "Consider running 'python setup.py --version' or 'pip install -e .' "
        "in the weldx root repository"
    )

# main modules
import weldx.utility  # import this first to avoid circular dependencies
import weldx.core
import weldx.geometry
import weldx.transformations
import weldx.welding

# asdf extensions and tags
import weldx.asdf

# class imports to weldx namespace
from weldx.constants import WELDX_QUANTITY as Q_
from weldx.core import MathematicalExpression, TimeSeries
from weldx.geometry import ArcSegment, Geometry, LineSegment, Profile, Shape, Trace
from weldx.transformations import (
    CoordinateSystemManager,
    LocalCoordinateSystem,
    WXRotation,
)
from weldx.welding.groove.iso_9692_1 import get_groove

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    Q_([])

__all__ = [
    "core",
    "geometry",
    "measurement",
    "transformations",
    "utility",
    "asdf",
    "Q_",
    "welding",
]
