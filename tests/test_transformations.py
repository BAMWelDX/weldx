"""Tests the transformation package."""

import weldx.transformations as tf
import weldx.utility as ut
import numpy as np
import pytest
import random
import math


# helpers for tests -----------------------------------------------------------


def check_coordinate_system(
    cs_p, basis_expected, origin_expected, positive_orientation_expected
):
    """
    Check the values of a coordinate system.

    :param cs_p: Coordinate system that should be checked
    :param basis_expected: Expected basis
    :param origin_expected: Expected origin
    :param positive_orientation_expected: Expected orientation
    :return: ---
    """
    # check orientation is as expected
    assert is_orientation_positive(cs_p) == positive_orientation_expected

    # check basis vectors are orthogonal
    assert tf.is_orthogonal(cs_p.basis[0], cs_p.basis[1])
    assert tf.is_orthogonal(cs_p.basis[1], cs_p.basis[2])
    assert tf.is_orthogonal(cs_p.basis[2], cs_p.basis[0])

    for i in range(3):
        unit_vec = tf.normalize(basis_expected[:, i])

        # check axis orientations match
        assert np.abs(np.dot(cs_p.basis[:, i], unit_vec) - 1) < 1e-9
        assert np.abs(np.dot(cs_p.orientation[:, i], unit_vec) - 1) < 1e-9

        # check origin correct
        assert np.abs(origin_expected[i] - cs_p.origin[i]) < 1e-9
        assert np.abs(origin_expected[i] - cs_p.location[i]) < 1e-9


def check_matrix_does_not_reflect(matrix):
    """
    Check if a matrix does not reflect.

    :param matrix: Matrix that should be checked
    :return: ---
    """
    assert np.linalg.det(matrix) >= 0


def check_matrix_orthogonal(matrix):
    """
    Check if a matrix is orthogonal.

    Condition: A^-1 = A^T.

    :param matrix: Matrix that should be checked
    :return: ---
    """
    transposed = np.transpose(matrix)

    product = np.matmul(transposed, matrix)
    assert ut.matrix_is_close(product, np.identity(3))


def is_orientation_positive(cs_p):
    """
    Return `True` if the coordinate system has a positive orientation.

    Otherwise, `False` is returned.

    :param cs_p: Coordinate system that should be checked
    :return: True / False
    """
    return (
        tf.orientation_point_plane_containing_origin(
            cs_p.basis[2], cs_p.basis[0], cs_p.basis[1]
        )
        > 0
    )


def random_vector():
    """
    Get a random 3d vector.

    :return: Random 3d vector.
    """
    return (
        np.array([random.random(), random.random(), random.random()])
        * 10
        * random.random()
    )


def random_non_unit_vector():
    """
    Get a random 3d vector that is not of unit length.

    :return: Random 3d vector.
    """
    vec = random_vector()
    while math.isclose(np.linalg.norm(vec), 1) or math.isclose(np.linalg.norm(vec), 0):
        vec = random_vector()
    return vec


def rotated_positive_orthogonal_basis(
    angle_x=np.pi / 3, angle_y=np.pi / 4, angle_z=np.pi / 5
):
    """
    Get a rotated orthogonal base.

    If X,Y,Z are the rotation matrices of the passed angles, the resulting
    base is Z * Y * X.

    :param angle_x: Rotation angle around the x-axis
    :param angle_y: Rotation angle around the y-axis
    :param angle_z: Rotation angle around the z-axis
    :return:
    """
    # rotate axes to produce a more general test case
    r_x = tf.rotation_matrix_x(angle_x)
    r_y = tf.rotation_matrix_y(angle_y)
    r_z = tf.rotation_matrix_z(angle_z)

    r_tot = np.matmul(r_z, np.matmul(r_y, r_x))
    return r_tot


# test functions --------------------------------------------------------------


def test_coordinate_axis_rotation_matrices():
    """
    Test the rotation matrices that rotate around one coordinate axis.

    This test creates the rotation matrices using 10 degree steps and
    multiplies them with a given vector. The result is compared to the
    expected values, which are determined using the sine and cosine.
    Additionally, some matrix properties are checked.

    :return: ---
    """
    matrix_funcs = [tf.rotation_matrix_x, tf.rotation_matrix_y, tf.rotation_matrix_z]
    vec = np.array([1, 1, 1])

    for i in range(3):
        for j in range(36):
            angle = j / 18 * np.pi
            matrix = matrix_funcs[i](angle)

            # rotation matrices are orthogonal
            check_matrix_orthogonal(matrix)

            # matrix should not reflect
            check_matrix_does_not_reflect(matrix)

            # rotate vector
            res = np.matmul(matrix, vec)

            i_1 = (i + 1) % 3
            i_2 = (i + 2) % 3

            exp_1 = np.cos(angle) - np.sin(angle)
            exp_2 = np.cos(angle) + np.sin(angle)

            assert math.isclose(res[i], 1)
            assert math.isclose(res[i_1], exp_1)
            assert math.isclose(res[i_2], exp_2)


def test_normalize():
    """
    Test the normalize function.

    This test creates some random vectors and normalizes them. Afterwards
    the results are checked.

    :return: ---
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
    """
    Test the orientation_point_plane_containing_origin function.

    This test takes the first two basis vectors of an orthogonal basis to
    describe the plane which contains the origin. Afterwards, several
    factors are multiplied with the normal vector of the plane (third column
    of the basis) to get some test points. Since the plane contains the
    origin, the sign returned by the orientation function must be equal to
    the sign of the factor (0 is a special case and tested at the end).
    Additionally some exceptions and special cases are tested.

    :return: ---
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
    """
    Test the test_orientation_point_plane function.

    This test takes the first two basis vectors of an orthogonal basis and
    adds an offset to them to describe the plane. Afterwards, several points
    are calculated by multiplying the normal vector of the plane (third
    column of the basis) with a certain factor and shifting the result by
    the same offset as the plane. The result of the orientation function
    must be equal to the factors sign (0 is a special case and tested at the
    end).
    Additionally some exceptions and special cases are tested.

    :return: ---
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
    """
    Test the is_orthogonal function.

    This test creates some vectors and checks if the function returns the
    correct results.

    :return: ---
    """
    basis = rotated_positive_orthogonal_basis()
    x = basis[:, 0]
    y = basis[:, 1]
    z = basis[:, 2]

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
    """
    Test vector_points_to_left_of_vector function.

    Tests multiple vector combinations with known result.

    :return: ---
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
    """
    Test the point_left_of_line function.

    Tests multiple test cases with known results.

    :return: ---
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
    """
    Test the reflection_sign function.

    Tests multiple test cases with known results.

    :return: ---
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


# test cartesian coordinate system class --------------------------------------


def test_coordinate_system_construction():
    """
    Test construction of coordinate system class.

    Create multiple coordinate systems with all provided methods and check
    if they are constructed correctly.

    :return: ---
    """
    # alias name for class - name is too long :)
    lcs = tf.LocalCoordinateSystem

    # setup -----------------------------------------------
    origin = [4, -2, 6]
    basis_pos = rotated_positive_orthogonal_basis()

    x = basis_pos[:, 0]
    y = basis_pos[:, 1]
    z = basis_pos[:, 2]

    basis_neg = np.transpose([x, y, -z])

    # construction with basis -----------------------------

    cs_basis_pos = lcs.construct_from_orientation(basis_pos, origin)
    cs_basis_neg = lcs.construct_from_orientation(basis_neg, origin)

    check_coordinate_system(cs_basis_pos, basis_pos, origin, True)
    check_coordinate_system(cs_basis_neg, basis_neg, origin, False)

    # construction with x,y,z-vectors ---------------------

    cs_xyz_pos = lcs.construct_from_xyz(x, y, z, origin)
    cs_xyz_neg = lcs.construct_from_xyz(x, y, -z, origin)

    check_coordinate_system(cs_xyz_pos, basis_pos, origin, True)
    check_coordinate_system(cs_xyz_neg, basis_neg, origin, False)

    # construction with x,y-vectors and orientation -------
    cs_xyo_pos = lcs.construct_from_xy_and_orientation(x, y, True, origin)
    cs_xyo_neg = lcs.construct_from_xy_and_orientation(x, y, False, origin)

    check_coordinate_system(cs_xyo_pos, basis_pos, origin, True)
    check_coordinate_system(cs_xyo_neg, basis_neg, origin, False)

    # construction with y,z-vectors and orientation -------
    cs_yzo_pos = lcs.construct_from_yz_and_orientation(y, z, True, origin)
    cs_yzo_neg = lcs.construct_from_yz_and_orientation(y, -z, False, origin)

    check_coordinate_system(cs_yzo_pos, basis_pos, origin, True)
    check_coordinate_system(cs_yzo_neg, basis_neg, origin, False)

    # construction with x,z-vectors and orientation -------
    cs_xzo_pos = lcs.construct_from_xz_and_orientation(x, z, True, origin)
    cs_xzo_neg = lcs.construct_from_xz_and_orientation(x, -z, False, origin)

    check_coordinate_system(cs_xzo_pos, basis_pos, origin, True)
    check_coordinate_system(cs_xzo_neg, basis_neg, origin, False)

    # test integers as inputs -----------------------------
    x_i = [1, 1, 0]
    y_i = [-1, 1, 0]
    z_i = [0, 0, 1]

    lcs.construct_from_xyz(x_i, y_i, z_i, origin)
    lcs.construct_from_xy_and_orientation(x_i, y_i)
    lcs.construct_from_yz_and_orientation(y_i, z_i)
    lcs.construct_from_xz_and_orientation(z_i, x_i)

    # check exceptions ------------------------------------
    with pytest.raises(Exception):
        lcs([x, y, [0, 0, 1]])


def test_coordinate_system_addition_and_subtraction():
    """
    Test the + and - operator of the coordinate system class.

    Creates some coordinate systems and uses the operators on them. Results
    are compared to expected values.

    :return: ---
    """
    lcs = tf.LocalCoordinateSystem

    orientation0 = tf.rotation_matrix_z(np.pi / 2)
    origin0 = [1, 3, 2]
    cs_0 = lcs(orientation0, origin0)

    orientation1 = tf.rotation_matrix_y(np.pi / 2)
    origin1 = [4, -2, 1]
    cs_1 = lcs(orientation1, origin1)

    orientation2 = tf.rotation_matrix_x(np.pi / 2)
    origin2 = [-3, 4, 2]
    cs_2 = lcs(orientation2, origin2)

    # addition --------------------------------------------

    # is associative
    cs_sum_0 = cs_2 + (cs_1 + cs_0)
    cs_sum_1 = (cs_2 + cs_1) + cs_0
    assert ut.matrix_is_close(cs_sum_0.basis, cs_sum_1.basis)
    assert ut.vector_is_close(cs_sum_0.origin, cs_sum_1.origin)

    expected_origin = np.array([-1, 9, 6])
    expected_orientation = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])

    assert ut.matrix_is_close(cs_sum_0.basis, expected_orientation)
    assert ut.vector_is_close(cs_sum_0.origin, expected_origin)

    # subtraction --------------------------------------------

    cs_diff_0 = cs_sum_0 - cs_0 - cs_1

    assert ut.matrix_is_close(cs_diff_0.basis, cs_2.basis)
    assert ut.vector_is_close(cs_diff_0.origin, cs_2.origin)
