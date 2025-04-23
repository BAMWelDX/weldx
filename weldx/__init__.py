"""
WelDX - data and quality standards for welding research data
============================================================

The weldx package provides several tools to setup an experiment, systematically process
measurement data, and archive it in a standardized fashion.

Here we quickly list the most important functions and classes.


**File handling**

Read and write weldx files.

.. autosummary::
    :toctree: _autosummary
    :caption: File handling
    :template: class-template.rst

    WeldxFile

**Welding**

These classes and functions are used to define welding processes.

.. autosummary::
    :toctree: _autosummary
    :caption: Welding

    GmawProcess
    get_groove

**Unit handling**

.. autosummary::
    :toctree: _autosummary
    :caption: Unit handling
    :template: class-template_docstring.rst

    Q_
    U_

**Data handling and transformation**

.. autosummary::
    :toctree: _autosummary
    :caption: Data handling and transformation
    :template: class-template.rst

    Time
    TimeSeries
    GenericSeries
    SpatialSeries
    MathematicalExpression
    CoordinateSystemManager
    LocalCoordinateSystem
    WXRotation

**Geometry**

These classes are used to define workpiece geometries.

.. autosummary::
    :toctree: _autosummary
    :caption: Geometry
    :template: class-template.rst

    ArcSegment
    Geometry
    DynamicBaseSegment
    DynamicShapeSegment
    LineSegment
    LinearHorizontalTraceSegment
    Profile
    Shape
    Trace
    SpatialData
    DynamicTraceSegment

**Full API Reference**

Here you find the full documentation for all sub-modules in weldx
including their classes and functions.

.. autosummary::
    :toctree: _autosummary
    :caption: Full API Reference
    :template: module-template.rst
    :recursive:

    constants
    core
    geometry
    measurement
    transformations
    util
    time
    types
    visualization
    welding


**ASDF API Reference**

Here you find the documentation of the underlying ASDF implementation. These classes
and functions are used to read and write weldx data types in the ASDF data format.

.. autosummary::
    :toctree: _autosummary/asdf
    :caption: ASDF API Reference
    :template: module-template.rst
    :recursive:

    asdf.extension
    asdf.util
    asdf.validators

"""

# isort:skip_file
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("weldx")
except PackageNotFoundError:
    # package is not installed
    pass

# constants - should be imported first, no internal weldx deps
from weldx.constants import Q_, U_

# main modules
import weldx.time  # skipcq: PY-W2000

# skipcq: PY-W2000
import weldx.util  # import this second to avoid circular dependencies
import weldx.core  # skipcq: PY-W2000
import weldx.transformations  # skipcq: PY-W2000
import weldx.config
import weldx.geometry  # skipcq: PY-W2000
import weldx.welding  # skipcq: PY-W2000

# class imports to weldx namespace
from weldx.config import Config

# skipcq: PY-W2000
from weldx.core import GenericSeries, MathematicalExpression, TimeSeries, SpatialSeries
from weldx.geometry import (
    ArcSegment,
    Geometry,
    DynamicBaseSegment,
    DynamicShapeSegment,
    LineSegment,
    LinearHorizontalTraceSegment,
    Profile,
    Shape,
    Trace,
    SpatialData,
    DynamicTraceSegment,
)
from weldx.transformations import (  # skipcq: PY-W2000
    CoordinateSystemManager,
    LocalCoordinateSystem,
    WXRotation,
)
from weldx.welding.processes import GmawProcess
from weldx.welding.groove.iso_9692_1 import get_groove
from weldx.time import Time

# tags (this will partially import weldx.asdf but not the extension)
from weldx import tags  # skipcq: PY-W2000

# asdf extensions
import weldx.asdf  # skipcq: PY-W2000
from weldx.asdf.file import WeldxFile

__all__ = (
    "ArcSegment",
    "Config",
    "CoordinateSystemManager",
    "DynamicBaseSegment",
    "DynamicShapeSegment",
    "DynamicTraceSegment",
    "Geometry",
    "GmawProcess",
    "LineSegment",
    "LinearHorizontalTraceSegment",
    "LocalCoordinateSystem",
    "MathematicalExpression",
    "Profile",
    "Q_",
    "Shape",
    "SpatialData",
    "Time",
    "TimeSeries",
    "Trace",
    "U_",
    "WeldxFile",
    "asdf",
    "core",
    "geometry",
    "get_groove",
    "measurement",
    "transformations",
    "util",
    "welding",
)

weldx.config.Config.load_installed_standards()
