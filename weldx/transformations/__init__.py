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
    """Hack the __module__ attribute of __all__ members to the current module.

    This is needed as Sphinx currently does not respect the all variable and ignores
    the contents. By simulating that the "all" attributes are belonging here, we work
    around this situation. Can be removed up this is fixed:
    https://github.com/sphinx-doc/sphinx/issues/2021
    """
    import sys

    this_mod = sys.modules[__name__]
    for name in __all__:
        obj = getattr(this_mod, name)
        obj.__module__ = __name__


_patch_mod_all()
