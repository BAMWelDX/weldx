""" Contains methods and classes for coordinate transformations.

.. currentmodule:: weldx.transformations

.. rubric:: Classes

.. autosummary::
  :toctree:
  :template: class-template.rst
  :nosignatures:

  CoordinateSystemManager
  LocalCoordinateSystem
  WXRotation

.. rubric:: Functions

.. autosummary::
   :toctree:
   :nosignatures:

   scale_matrix
   normalize
   orientation_point_plane_containing_origin
   orientation_point_plane
   is_orthogonal
   is_orthogonal_matrix
   point_left_of_line
   reflection_sign
   vector_points_to_left_of_vector

"""
from .cs_manager import CoordinateSystemManager
from .local_cs import LocalCoordinateSystem
from .rotation import WXRotation
from .util import *
