"""Test the utility functions of the transformation module."""

import math
import random

import numpy as np
import pytest

import weldx.transformations as tf

from .._helpers import matrix_is_close, vector_is_close

# --------------------------------------------------------------------------------------
# helper functions
# --------------------------------------------------------------------------------------


def _random_vector():
    """Get a random 3d vector."""
    return (
        np.array([random.random(), random.random(), random.random()])
        * 10
        * random.random()
    )


def _random_non_unit_vector():
    """Get a random 3d vector that is not of unit length."""
    vec = _random_vector()
    while math.isclose(np.linalg.norm(vec), 1) or math.isclose(np.linalg.norm(vec), 0):
        vec = _random_vector()
    return vec


# --------------------------------------------------------------------------------------
# actual tests
# --------------------------------------------------------------------------------------


def test_scaling_matrix():
    """Test the scaling matrix."""
    mat_a = np.array([[1, 6, 2], [4, 10, 2], [3, 5, 2]], dtype=float)
    scale_mat = tf.scale_matrix(2, 0.5, 4)
    mat_b = np.matmul(scale_mat, mat_a)

    mat_b_exp = np.array([[2, 12, 4], [2, 5, 1], [12, 20, 8]], dtype=float)
    assert matrix_is_close(mat_b, mat_b_exp)


def test_normalize():
    """Test the normalize function.

    This test creates some random vectors and normalizes them. Afterwards,
    the results are checked.

    """
    for _ in range(20):
        vec = _random_non_unit_vector()

        unit = tf.normalize(vec)

        # check that vector is modified
        assert not vector_is_close(unit, vec)

        # check length is 1
        assert math.isclose(np.linalg.norm(unit), 1)

        # check that both vectors point into the same direction
        vec2 = unit * np.linalg.norm(vec)
        assert vector_is_close(vec2, vec)

    #  exception ------------------------------------------

    # length is 0
    with pytest.raises(ValueError):
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
    angles = [np.pi / 3, np.pi / 4, np.pi / 5]
    [a, b, n] = tf.WXRotation.from_euler("xyz", angles).as_matrix()
    a *= 2.3
    b /= 1.5

    for length in np.arange(-9.5, 9.51, 1):
        orientation = tf.orientation_point_plane_containing_origin(n * length, a, b)
        assert np.sign(length) == orientation

    # check exceptions
    with pytest.raises(ValueError):
        tf.orientation_point_plane_containing_origin(n, a, a)
    with pytest.raises(ValueError):
        tf.orientation_point_plane_containing_origin(n, np.zeros(3), b)
    with pytest.raises(ValueError):
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
    angles = [np.pi / 3, np.pi / 4, np.pi / 5]
    [b, c, n] = tf.WXRotation.from_euler("xyz", angles).as_matrix()
    a = np.array([3.2, -2.1, 5.4], dtype=float)
    b = b * 6.5 + a
    c = c * 0.3 + a

    for length in np.arange(-9.5, 9.51, 1):
        orientation = tf.orientation_point_plane(n * length + a, a, b, c)
        assert np.sign(length) == orientation

    # check exceptions
    with pytest.raises(ValueError):
        tf.orientation_point_plane(n, a, a, c)
    with pytest.raises(ValueError):
        tf.orientation_point_plane(n, a, b, b)
    with pytest.raises(ValueError):
        tf.orientation_point_plane(n, c, b, c)
    with pytest.raises(ValueError):
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
    angles = [np.pi / 3, np.pi / 4, np.pi / 5]
    orientation = tf.WXRotation.from_euler("xyz", angles).as_matrix()
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
    with pytest.raises(ValueError):
        tf.is_orthogonal([0, 0, 0], z)
    with pytest.raises(ValueError):
        tf.is_orthogonal(x, [0, 0, 0])
    with pytest.raises(ValueError):
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

    with pytest.raises(ValueError):
        tf.reflection_sign([[0, 0], [0, 0]])
    with pytest.raises(ValueError):
        tf.reflection_sign([[1, 0], [0, 0]])
    with pytest.raises(ValueError):
        tf.reflection_sign([[2, 2], [1, 1]])
