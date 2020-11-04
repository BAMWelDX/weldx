import warnings

# asdf extensions and tags
import weldx.asdf

# main packages
import weldx.core
import weldx.geometry
import weldx.transformations
import weldx.utility
import weldx.welding
from weldx.constants import WELDX_QUANTITY as Q_
from weldx.core import MathematicalExpression, TimeSeries
from weldx.geometry import ArcSegment, Geometry, LineSegment, Profile, Shape, Trace
from weldx.transformations import (
    CoordinateSystemManager,
    LocalCoordinateSystem,
    WXRotation,
)

# class imports to weldx namespace
from weldx.welding.groove.iso_9692_1 import get_groove

# versioneer
from ._version import get_versions

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    Q_([])

__version__ = get_versions()["version"]
del get_versions

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
