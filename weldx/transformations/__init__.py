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


def _patch_mod_all():
    import sys

    mod = sys.modules[__name__]
    for name in __all__:
        obj = getattr(mod, name)
        obj.__module__ = __name__


#_patch_mod_all()
