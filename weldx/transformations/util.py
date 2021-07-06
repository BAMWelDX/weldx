"""Contains functions for coordinate transformations."""

import math
from typing import Tuple

import numpy as np
import pandas as pd

from weldx import util
from weldx.transformations.types import types_timeindex

__all__ = [
    "build_time_index",
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


def build_time_index(
    time: types_timeindex = None,
    time_ref: pd.Timestamp = None,
) -> Tuple[pd.TimedeltaIndex, pd.Timestamp]:
    """Build time index used for xarray objects.

    Parameters
    ----------
    time:
        Datetime- or Timedelta-like time index.
    time_ref:
        Reference timestamp for Timedelta inputs.

    Returns
    -------
    pandas.TimedeltaIndex

    """
    if time is None:
        return time, time_ref

    time = util.to_pandas_time_index(time)

    if isinstance(time, pd.DatetimeIndex):
        if time_ref is None:
            time_ref = time[0]
        time = time - time_ref

    return time, time_ref


def scale_matrix(scale_x, scale_y, scale_z) -> np.ndarray:
    """Return a scaling matrix.

    Parameters
    ----------
    scale_x :
        Scaling factor in x direction
    scale_y :
        Scaling factor in y direction
    scale_z :
        Scaling factor in z direction

    Returns
    -------
    numpy.ndarray
        Scaling matrix

    """
    return np.diag([scale_x, scale_y, scale_z]).astype(float)


def normalize(a):
    """Normalize (l2 norm) an ndarray along the last dimension.

    Parameters
    ----------
    a :
        data in ndarray

    Returns
    -------
    numpy.ndarray
        Normalized ndarray

    """
    norm = np.linalg.norm(a, axis=(-1), keepdims=True)
    if not np.all(norm):
        raise ValueError("Length 0 encountered during normalization.")
    return a / norm


def orientation_point_plane_containing_origin(point, p_a, p_b):
    """Determine a points orientation relative to a plane containing the origin.

    The side is defined by the winding order of the triangle 'origin - A -
    B'. When looking at it from the left-hand side, the ordering is clockwise
    and counter-clockwise when looking from the right-hand side.

    The function returns 1 if the point lies left of the plane, -1 if it is
    on the right and 0 if it lies on the plane.

    Note, that this function is not appropriate to check if a point lies on
    a plane since it has no tolerance to compensate for numerical errors.

    Additional note: The points A and B can also been considered as two
    vectors spanning the plane.

    Parameters
    ----------
    point :
        Point
    p_a :
        Second point of the triangle 'origin - A - B'.
    p_b :
        Third point of the triangle 'origin - A - B'.

    Returns
    -------
    int
        1, -1 or 0 (see description)

    """
    if (
        math.isclose(np.linalg.norm(p_a), 0)
        or math.isclose(np.linalg.norm(p_b), 0)
        or math.isclose(np.linalg.norm(p_b - p_a), 0)
    ):
        raise ValueError("One or more points describing the plane are identical.")

    return np.sign(np.linalg.det([p_a, p_b, point]))


def orientation_point_plane(point, p_a, p_b, p_c):
    """Determine a points orientation relative to an arbitrary plane.

    The side is defined by the winding order of the triangle 'A - B - C'.
    When looking at it from the left-hand side, the ordering is clockwise
    and counter-clockwise when looking from the right-hand side.

    The function returns 1 if the point lies left of the plane, -1 if it is
    on the right and 0 if it lies on the plane.

    Note, that this function is not appropriate to check if a point lies on
    a plane since it has no tolerance to compensate for numerical errors.

    Parameters
    ----------
    point :
        Point
    p_a :
        First point of the triangle 'A - B - C'.
    p_b :
        Second point of the triangle 'A - B - C'.
    p_c :
        Third point of the triangle 'A - B - C'.

    Returns
    -------
    int
        1, -1 or 0 (see description)

    """
    vec_a_b = p_b - p_a
    vec_a_c = p_c - p_a
    vec_a_point = point - p_a
    return orientation_point_plane_containing_origin(vec_a_point, vec_a_b, vec_a_c)


def is_orthogonal(vec_u, vec_v, tolerance=1e-9):
    """Check if vectors are orthogonal.

    Parameters
    ----------
    vec_u :
        First vector
    vec_v :
        Second vector
    tolerance :
        Numerical tolerance (Default value = 1e-9)

    Returns
    -------
    bool
        True or False

    """
    if math.isclose(np.dot(vec_u, vec_u), 0) or math.isclose(np.dot(vec_v, vec_v), 0):
        raise ValueError("One or both vectors have zero length.")

    return math.isclose(np.dot(vec_u, vec_v), 0, abs_tol=tolerance)


def is_orthogonal_matrix(a: np.ndarray, atol=1e-9) -> bool:
    """Check if ndarray is orthogonal matrix in the last two dimensions.

    Parameters
    ----------
    a :
        Matrix to check
    atol :
        atol to pass onto np.allclose (Default value = 1e-9)

    Returns
    -------
    bool
        True if last 2 dimensions of a are orthogonal

    """
    return np.allclose(np.matmul(a, a.swapaxes(-1, -2)), np.eye(a.shape[-1]), atol=atol)


def point_left_of_line(point, line_start, line_end):
    """Determine if a point lies left of a line.

    Returns 1 if the point is left of the line and -1 if it is to the right.
    If the point is located on the line, this function returns 0.

    Parameters
    ----------
    point :
        Point
    line_start :
        Starting point of the line
    line_end :
        End point of the line

    Returns
    -------
    int
        1,-1 or 0 (see description)

    """
    vec_line_start_end = line_end - line_start
    vec_line_start_point = point - line_start
    return vector_points_to_left_of_vector(vec_line_start_point, vec_line_start_end)


def reflection_sign(matrix):
    """Get a sign indicating if the transformation is a reflection.

    Returns -1 if the transformation contains a reflection and 1 if not.

    Parameters
    ----------
    matrix :
        Transformation matrix

    Returns
    -------
    int
        1 or -1 (see description)

    """
    sign = int(np.sign(np.linalg.det(matrix)))

    if sign == 0:
        raise ValueError("Invalid transformation")

    return sign


def vector_points_to_left_of_vector(vector, vector_reference):
    """Determine if a vector points to the left of another vector.

    Returns 1 if the vector points to the left of the reference vector and
    -1 if it points to the right. In case both vectors point into the same
    or the opposite directions, this function returns 0.

    Parameters
    ----------
    vector :
        Vector
    vector_reference :
        Reference vector

    Returns
    -------
    int
        1,-1 or 0 (see description)

    """
    return int(np.sign(np.linalg.det([vector_reference, vector])))
