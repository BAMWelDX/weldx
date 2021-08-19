"""Tests the transformation package."""

import math
import random
from copy import deepcopy
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pandas import TimedeltaIndex as TDI  # noqa
from pandas import Timestamp as TS  # noqa
from pandas import date_range

import weldx.transformations as tf
import weldx.util as ut
from weldx import Q_, SpatialData
from weldx.core import MathematicalExpression, TimeSeries
from weldx.tests._helpers import get_test_name
from weldx.time import Time, types_time_like, types_timestamp_like
from weldx.transformations import LocalCoordinateSystem as LCS  # noqa
from weldx.transformations import WXRotation

# helpers for tests -----------------------------------------------------------


def check_matrix_does_not_reflect(matrix):
    """Check if a matrix does not reflect.

    Parameters
    ----------
    matrix :
        Matrix that should be checked

    """
    assert np.linalg.det(matrix) >= 0


def check_matrix_orthogonal(matrix):
    """Check if a matrix is orthogonal.

    Condition: A^-1 = A^T.

    Parameters
    ----------
    matrix :
        Matrix that should be checked

    """
    transposed = np.transpose(matrix)

    product = np.matmul(transposed, matrix)
    assert ut.matrix_is_close(product, np.identity(3))


def random_vector():
    """Get a random 3d vector.

    Returns
    -------
    np.ndarray
        Random 3d vector.

    """
    return (
        np.array([random.random(), random.random(), random.random()])
        * 10
        * random.random()
    )


def random_non_unit_vector():
    """Get a random 3d vector that is not of unit length.

    Returns
    -------
    np.ndarray
        Random 3d vector.

    """
    vec = random_vector()
    while math.isclose(np.linalg.norm(vec), 1) or math.isclose(np.linalg.norm(vec), 0):
        vec = random_vector()
    return vec


def rotated_positive_orthogonal_basis(
    angle_x=np.pi / 3, angle_y=np.pi / 4, angle_z=np.pi / 5
):
    """Get a rotated orthogonal basis.

    If X,Y,Z are the rotation matrices of the passed angles, the resulting
    basis is Z * Y * X.

    Parameters
    ----------
    angle_x :
        Rotation angle around the x-axis (Default value = np.pi / 3)
    angle_y :
        Rotation angle around the y-axis (Default value = np.pi / 4)
    angle_z :
        Rotation angle around the z-axis (Default value = np.pi / 5)

    Returns
    -------
    np.ndarray
        Rotated orthogonal basis

    """
    # rotate axes to produce a more general test case
    return WXRotation.from_euler("xyz", [angle_x, angle_y, angle_z]).as_matrix()


def check_coordinate_system_orientation(
    orientation: xr.DataArray,
    orientation_expected: np.ndarray,
    positive_orientation_expected: bool,
):
    """Check if the orientation of a local coordinate system is as expected.

    Parameters
    ----------
    orientation :
        Orientation
    orientation_expected :
        Expected orientation
    positive_orientation_expected :
        True, if the orientation is expected to be
        positive. False otherwise.

    """
    # test expected positive orientation
    det = np.linalg.det(orientation.sel(v=[2, 0, 1]))
    assert np.all((det > 0) == positive_orientation_expected)

    assert tf.is_orthogonal_matrix(orientation.values)

    orientation_expected = tf.normalize(orientation_expected)

    assert np.allclose(orientation, orientation_expected)


def check_coordinate_system(
    lcs: tf.LocalCoordinateSystem,
    orientation_expected: Union[np.ndarray, List[List[Any]], xr.DataArray],
    coordinates_expected: Union[np.ndarray, List[Any], xr.DataArray],
    positive_orientation_expected: bool = True,
    time=None,
    time_ref=None,
):
    """Check the values of a coordinate system.

    Parameters
    ----------
    lcs :
        Coordinate system that should be checked
    orientation_expected :
        Expected orientation
    coordinates_expected :
        Expected coordinates
    positive_orientation_expected :
        Expected orientation
    time :
        A pandas.DatetimeIndex object, if the coordinate system is expected to
        be time dependent. None otherwise.
    time_ref:
        The expected reference time

    """
    orientation_expected = np.array(orientation_expected)
    coordinates_expected = np.array(coordinates_expected)

    if time is not None:
        assert orientation_expected.ndim == 3 or coordinates_expected.ndim == 2
        assert np.all(lcs.time == Time(time, time_ref))
        assert lcs.reference_time == time_ref

    check_coordinate_system_orientation(
        lcs.orientation, orientation_expected, positive_orientation_expected
    )

    assert np.allclose(lcs.coordinates.values, coordinates_expected, atol=1e-9)


def check_cs_close(lcs_0, lcs_1):
    """Check if 2 coordinate systems are nearly identical.

    Parameters
    ----------
    lcs_0:
        First coordinate system.
    lcs_1
        Second coordinate system.

    """
    check_coordinate_system(
        lcs_0,
        lcs_1.orientation.data,
        lcs_1.coordinates.data,
        True,
        lcs_1.time,
        lcs_1.reference_time,
    )


# test functions --------------------------------------------------------------


def test_scaling_matrix():
    """Test the scaling matrix.

    Should be self explanatory.

    """
    mat_a = np.array([[1, 6, 2], [4, 10, 2], [3, 5, 2]], dtype=float)
    scale_mat = tf.scale_matrix(2, 0.5, 4)
    mat_b = np.matmul(scale_mat, mat_a)

    mat_b_exp = mat_a = np.array([[2, 12, 4], [2, 5, 1], [12, 20, 8]], dtype=float)
    assert ut.matrix_is_close(mat_b, mat_b_exp)


def test_normalize():
    """Test the normalize function.

    This test creates some random vectors and normalizes them. Afterwards,
    the results are checked.

    """
    for _ in range(20):
        vec = random_non_unit_vector()

        unit = tf.normalize(vec)

        # check that vector is modified
        assert not ut.vector_is_close(unit, vec)

        # check length is 1
        assert math.isclose(np.linalg.norm(unit), 1)

        # check that both vectors point into the same direction
        vec2 = unit * np.linalg.norm(vec)
        assert ut.vector_is_close(vec2, vec)

    #  exception ------------------------------------------

    # length is 0
    with pytest.raises(Exception):
        tf.normalize(np.array([0, 0, 0]))


def test_orientation_point_plane_containing_origin():
    """Test the orientation_point_plane_containing_origin function.

    This test takes the first two vectors of an orthogonal orientation matrix to
    describe the plane which contains the origin. Afterwards, several
    factors are multiplied with the normal vector of the plane (third column
    of the orientation matrix) to get some test points. Since the plane contains the
    origin, the sign returned by the orientation function must be equal to
    the sign of the factor (0 is a special case and tested at the end).
    Additionally some exceptions and special cases are tested.

    """
    [a, b, n] = rotated_positive_orthogonal_basis()
    a *= 2.3
    b /= 1.5

    for length in np.arange(-9.5, 9.51, 1):
        orientation = tf.orientation_point_plane_containing_origin(n * length, a, b)
        assert np.sign(length) == orientation

    # check exceptions
    with pytest.raises(Exception):
        tf.orientation_point_plane_containing_origin(n, a, a)
    with pytest.raises(Exception):
        tf.orientation_point_plane_containing_origin(n, np.zeros(3), b)
    with pytest.raises(Exception):
        tf.orientation_point_plane_containing_origin(n, a, np.zeros(3))

    # check special case point on plane
    a = np.array([1, 0, 0])
    b = np.array([0, 1, 0])
    orientation = tf.orientation_point_plane_containing_origin(a, a, b)
    assert orientation == 0


def test_orientation_point_plane():
    """Test the test_orientation_point_plane function.

    This test takes the first two vectors of an orthogonal orientation matrix and
    adds an offset to them to describe the plane. Afterwards, several points
    are calculated by multiplying the normal vector of the plane (third
    column of the orientation matrix) with a certain factor and shifting the result by
    the same offset as the plane. The result of the orientation function
    must be equal to the factors sign (0 is a special case and tested at the
    end).
    Additionally, some exceptions and special cases are tested.

    """
    [b, c, n] = rotated_positive_orthogonal_basis()
    a = ut.to_float_array([3.2, -2.1, 5.4])
    b = b * 6.5 + a
    c = c * 0.3 + a

    for length in np.arange(-9.5, 9.51, 1):
        orientation = tf.orientation_point_plane(n * length + a, a, b, c)
        assert np.sign(length) == orientation

    # check exceptions
    with pytest.raises(Exception):
        tf.orientation_point_plane(n, a, a, c)
    with pytest.raises(Exception):
        tf.orientation_point_plane(n, a, b, b)
    with pytest.raises(Exception):
        tf.orientation_point_plane(n, c, b, c)
    with pytest.raises(Exception):
        tf.orientation_point_plane(n, a, a, a)

    # check special case point on plane
    a = np.array([1, 0, 0])
    b = np.array([0, 1, 0])
    c = np.array([0, 0, 1])
    orientation = tf.orientation_point_plane(a, a, b, c)
    assert orientation == 0


def test_is_orthogonal():
    """Test the is_orthogonal function.

    This test creates some vectors and checks if the function returns the
    correct results.

    """
    orientation = rotated_positive_orthogonal_basis()
    x = orientation[:, 0]
    y = orientation[:, 1]
    z = orientation[:, 2]

    assert tf.is_orthogonal(x, y)
    assert tf.is_orthogonal(y, x)
    assert tf.is_orthogonal(y, z)
    assert tf.is_orthogonal(z, y)
    assert tf.is_orthogonal(z, x)
    assert tf.is_orthogonal(x, z)

    assert not tf.is_orthogonal(x, x)
    assert not tf.is_orthogonal(y, y)
    assert not tf.is_orthogonal(z, z)

    # check tolerance is working
    assert not tf.is_orthogonal(x + 0.00001, z, 1e-6)
    assert tf.is_orthogonal(x + 0.00001, z, 1e-4)

    # exceptions ------------------------------------------

    # vectors with length=0
    with pytest.raises(Exception):
        tf.is_orthogonal([0, 0, 0], z)
    with pytest.raises(Exception):
        tf.is_orthogonal(x, [0, 0, 0])
    with pytest.raises(Exception):
        tf.is_orthogonal([0, 0, 0], [0, 0, 0])


def test_vector_points_to_left_of_vector():
    """Test vector_points_to_left_of_vector function.

    Tests multiple vector combinations with known result.

    """
    assert tf.vector_points_to_left_of_vector([-0.1, 1], [0, 1]) > 0
    assert tf.vector_points_to_left_of_vector([-0.1, -1], [0, 1]) > 0
    assert tf.vector_points_to_left_of_vector([3, 5], [1, 0]) > 0
    assert tf.vector_points_to_left_of_vector([-3, 5], [1, 0]) > 0
    assert tf.vector_points_to_left_of_vector([0, -0.1], [-4, 2]) > 0
    assert tf.vector_points_to_left_of_vector([-1, -0.1], [-4, 2]) > 0

    assert tf.vector_points_to_left_of_vector([0.1, 1], [0, 1]) < 0
    assert tf.vector_points_to_left_of_vector([0.1, -1], [0, 1]) < 0
    assert tf.vector_points_to_left_of_vector([3, -5], [1, 0]) < 0
    assert tf.vector_points_to_left_of_vector([-3, -5], [1, 0]) < 0
    assert tf.vector_points_to_left_of_vector([0, 0.1], [-4, 2]) < 0
    assert tf.vector_points_to_left_of_vector([1, -0.1], [-4, 2]) < 0

    assert tf.vector_points_to_left_of_vector([4, 4], [2, 2]) == 0
    assert tf.vector_points_to_left_of_vector([-4, -4], [2, 2]) == 0


def test_point_left_of_line():
    """Test the point_left_of_line function.

    Tests multiple test cases with known results.

    """
    line_start = np.array([2, 3])
    line_end = np.array([5, 6])
    assert tf.point_left_of_line([-8, 10], line_start, line_end) > 0
    assert tf.point_left_of_line([3, 0], line_start, line_end) < 0
    assert tf.point_left_of_line(line_start, line_start, line_end) == 0

    line_start = np.array([2, 3])
    line_end = np.array([1, -4])
    assert tf.point_left_of_line([3, 0], line_start, line_end) > 0
    assert tf.point_left_of_line([-8, 10], line_start, line_end) < 0
    assert tf.point_left_of_line(line_start, line_start, line_end) == 0


def test_reflection_sign():
    """Test the reflection_sign function.

    Tests multiple test cases with known results.

    """
    assert tf.reflection_sign([[-1, 0], [0, 1]]) == -1
    assert tf.reflection_sign([[1, 0], [0, -1]]) == -1
    assert tf.reflection_sign([[0, 1], [1, 0]]) == -1
    assert tf.reflection_sign([[0, -1], [-1, 0]]) == -1
    assert tf.reflection_sign([[-4, 0], [0, 2]]) == -1
    assert tf.reflection_sign([[6, 0], [0, -4]]) == -1
    assert tf.reflection_sign([[0, 3], [8, 0]]) == -1
    assert tf.reflection_sign([[0, -3], [-2, 0]]) == -1

    assert tf.reflection_sign([[1, 0], [0, 1]]) == 1
    assert tf.reflection_sign([[-1, 0], [0, -1]]) == 1
    assert tf.reflection_sign([[0, -1], [1, 0]]) == 1
    assert tf.reflection_sign([[0, 1], [-1, 0]]) == 1
    assert tf.reflection_sign([[5, 0], [0, 6]]) == 1
    assert tf.reflection_sign([[-3, 0], [0, -7]]) == 1
    assert tf.reflection_sign([[0, -8], [9, 0]]) == 1
    assert tf.reflection_sign([[0, 3], [-2, 0]]) == 1

    with pytest.raises(Exception):
        tf.reflection_sign([[0, 0], [0, 0]])
    with pytest.raises(Exception):
        tf.reflection_sign([[1, 0], [0, 0]])
    with pytest.raises(Exception):
        tf.reflection_sign([[2, 2], [1, 1]])
