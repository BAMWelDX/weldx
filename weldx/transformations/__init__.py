"""Contains methods and classes for coordinate transformations."""

from .cs_manager import CoordinateSystemManager
from .local_cs import LocalCoordinateSystem
from .rotation import WXRotation
from .util import *

__all__ = [
    "CoordinateSystemManager",
    "LocalCoordinateSystem",
    "WXRotation",
    "is_orthogonal",
    "is_orthogonal_matrix",
    "normalize",
    "orientation_point_plane",
    "orientation_point_plane_containing_origin",
    "point_left_of_line",
    "reflection_sign",
    "scale_matrix",
    "vector_points_to_left_of_vector",
]

from ..util import _patch_mod_all  # noqa

_patch_mod_all("weldx.transformations")
del _patch_mod_all
