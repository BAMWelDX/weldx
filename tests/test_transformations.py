"""Tests the transformation package."""

import weldx.transformations as tf
import weldx.utility as ut
import numpy as np
import pandas as pd
import xarray as xr
import pytest
import random
import math
from typing import Union, List, Any


# helpers for tests -----------------------------------------------------------


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


def test_scaling_matrix():
    """
    Test the scaling matrix.

    Should be self explanatory.

    :return: ---
    """
    mat_a = np.array([[1, 6, 2], [4, 10, 2], [3, 5, 2]], dtype=float)
    scale_mat = tf.scale_matrix(2, 0.5, 4)
    mat_b = np.matmul(scale_mat, mat_a)

    mat_b_exp = mat_a = np.array([[2, 12, 4], [2, 5, 1], [12, 20, 8]], dtype=float)
    assert ut.matrix_is_close(mat_b, mat_b_exp)


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


def check_coordinate_system_time(lcs: tf.LocalCoordinateSystem, expected_time):
    """
    Check if the time component of a LocalCoordinateSystem is as expected.

    :param lcs:Local coordinate system class
    :param expected_time: Expected time
    :return: ---
    """
    assert np.all(lcs.time == expected_time)


def check_coordinate_system_basis(
    basis: xr.DataArray, basis_expected: np.ndarray, positive_orientation_expected: bool
):
    """
    Check if the basis of a local coordinate system is as expected.

    :param basis: Basis
    :param basis_expected: Expected basis
    :param positive_orientation_expected: True, if the orientation is expected to be
    positive. False otherwise.
    :return: ---
    """
    # test expected positive orientation
    det = np.linalg.det(basis.sel(v=[2, 0, 1]))
    assert np.all((det > 0) == positive_orientation_expected)

    assert tf.is_orthogonal_matrix(basis.values)

    basis_expected = tf.normalize(basis_expected)

    assert np.allclose(basis, basis_expected)


def check_coordinate_system(
    cs_p: tf.LocalCoordinateSystem,
    basis_expected: Union[np.ndarray, List[List[Any]], xr.DataArray],
    origin_expected: Union[np.ndarray, List[Any], xr.DataArray],
    positive_orientation_expected: bool,
    time=None,
):
    # TODO: add time dependency
    """
    Check the values of a coordinate system.

    :param cs_p: Coordinate system that should be checked
    :param basis_expected: Expected basis
    :param origin_expected: Expected origin
    :param positive_orientation_expected: Expected orientation
    :param time: A pandas.DatetimeIndex object, if the coordinate system is expected to
    be time dependent. None otherwise.
    :return: ---
    """
    basis_expected = np.array(basis_expected)
    origin_expected = np.array(origin_expected)

    if time is not None:
        assert basis_expected.ndim == 3 or origin_expected.ndim == 2
        check_coordinate_system_time(cs_p, time)

    check_coordinate_system_basis(
        cs_p.basis, basis_expected, positive_orientation_expected
    )
    check_coordinate_system_basis(
        cs_p.orientation, basis_expected, positive_orientation_expected
    )

    assert np.allclose(cs_p.origin.values, origin_expected, atol=1e-9)
    assert np.allclose(cs_p.location.values, origin_expected, atol=1e-9)


def test_coordinate_system_init():
    """
    Check the __init__ method with and without time dependency.

    :return: ---
    """
    # reference data
    time_start_0 = "2042-01-01"
    time_start_1 = "2042-01-02"
    time_0 = pd.date_range(time_start_0, periods=3, freq="2D")
    time_1 = pd.date_range(time_start_1, periods=3, freq="2D")

    orientation_fix = tf.rotation_matrix_z(np.pi)
    orientation_tdp = tf.rotation_matrix_z(np.pi * np.array([0, 0.25, 0.5]))
    coordinates_fix = ut.to_float_array([3, 7, 1])
    coordinates_tdp = ut.to_float_array([[3, 7, 1], [4, -2, 8], [-5, 3, -1]])

    # numpy - no time dependency
    lcs = tf.LocalCoordinateSystem(basis=orientation_fix, origin=coordinates_fix)

    check_coordinate_system(lcs, orientation_fix, coordinates_fix, True)

    # numpy - orientation time dependent
    lcs = tf.LocalCoordinateSystem(
        basis=orientation_tdp, origin=coordinates_fix, time=time_0
    )

    check_coordinate_system(lcs, orientation_tdp, coordinates_fix, True, time_0)

    # numpy - coordinates time dependent
    lcs = tf.LocalCoordinateSystem(
        basis=orientation_fix, origin=coordinates_tdp, time=time_0
    )

    check_coordinate_system(lcs, orientation_fix, coordinates_tdp, True, time_0)

    # numpy - coordinates and orientation time dependent - only equal times
    lcs = tf.LocalCoordinateSystem(
        basis=orientation_tdp, origin=coordinates_tdp, time=time_0
    )

    check_coordinate_system(lcs, orientation_tdp, coordinates_tdp, True, time_0)

    # xarray - reference data
    xr_orientation_fix = ut.xr_3d_matrix(orientation_fix)
    xr_coordinates_fix = ut.xr_3d_vector(coordinates_fix)
    xr_orientation_tdp_0 = ut.xr_3d_matrix(orientation_tdp, time_0)
    xr_coordinates_tdp_0 = ut.xr_3d_vector(coordinates_tdp, time_0)

    # xarray - no time dependency
    lcs = tf.LocalCoordinateSystem(basis=xr_orientation_fix, origin=xr_coordinates_fix)

    check_coordinate_system(lcs, orientation_fix, coordinates_fix, True)

    # xarray - orientation time dependent
    lcs = tf.LocalCoordinateSystem(
        basis=xr_orientation_tdp_0, origin=xr_coordinates_fix
    )

    check_coordinate_system(lcs, orientation_tdp, coordinates_fix, True, time_0)

    # xarray - coordinates time dependent
    lcs = tf.LocalCoordinateSystem(
        basis=xr_orientation_fix, origin=xr_coordinates_tdp_0
    )

    check_coordinate_system(lcs, orientation_fix, coordinates_tdp, True, time_0)

    # xarray - coordinates and orientation time dependent - equal times
    lcs = tf.LocalCoordinateSystem(
        basis=xr_orientation_tdp_0, origin=xr_coordinates_tdp_0
    )

    check_coordinate_system(lcs, orientation_tdp, coordinates_tdp, True, time_0)

    # xarray - coordinates and orientation time dependent - different times
    xr_coordinates_tdp_1 = ut.xr_3d_vector(coordinates_tdp, time_1)

    lcs = tf.LocalCoordinateSystem(
        basis=xr_orientation_tdp_0, origin=xr_coordinates_tdp_1
    )

    time_exp = pd.date_range("2042-01-01", periods=6, freq="1D")
    coordinates_exp = ut.to_float_array(
        [
            [3, 7, 1],
            [3, 7, 1],
            [3.5, 2.5, 4.5],
            [4, -2, 8],
            [-0.5, 0.5, 3.5],
            [-5, 3, -1],
        ]
    )
    orientation_exp = tf.rotation_matrix_z(
        np.pi * np.array([0, 0.125, 0.25, 0.375, 0.5, 0.5])
    )
    check_coordinate_system(lcs, orientation_exp, coordinates_exp, True, time_exp)

    # exceptions --------------------------------
    # invalid inputs
    with pytest.raises(Exception):
        tf.LocalCoordinateSystem(basis="wrong", origin=coordinates_fix, time=time_0)
    with pytest.raises(Exception):
        tf.LocalCoordinateSystem(basis=orientation_fix, origin="wrong", time=time_0)
    with pytest.raises(Exception):
        tf.LocalCoordinateSystem(
            basis=orientation_fix, origin=coordinates_fix, time="wrong"
        )

    # wrong xarray format
    # TODO: implement


def test_coordinate_system_factories():
    """
    Test construction of coordinate system class.

    Create multiple coordinate systems with all provided methods and check
    if they are constructed correctly.

    :return: ---
    """
    # TODO: Time dependency
    # alias name for class - name is too long :)
    lcs = tf.LocalCoordinateSystem

    # setup -----------------------------------------------
    angle_x = np.pi / 3
    angle_y = np.pi / 4
    angle_z = np.pi / 5
    origin = [4, -2, 6]
    basis_pos = rotated_positive_orthogonal_basis(angle_x, angle_y, angle_z)

    x = basis_pos[:, 0]
    y = basis_pos[:, 1]
    z = basis_pos[:, 2]

    basis_neg = np.transpose([x, y, -z])

    # construction with basis -----------------------------

    cs_basis_pos = lcs.construct_from_orientation(basis_pos, origin)
    cs_basis_neg = lcs.construct_from_orientation(basis_neg, origin)

    check_coordinate_system(cs_basis_pos, basis_pos, origin, True)
    check_coordinate_system(cs_basis_neg, basis_neg, origin, False)

    # construction with euler -----------------------------

    angles = [angle_x, angle_y, angle_z]
    cs_euler_pos = lcs.construct_from_euler("xyz", angles, False, origin)
    check_coordinate_system(cs_euler_pos, basis_pos, origin, True)

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
    are compared to expected values. The naming pattern 'X_in_Y' is used for the
    coordinate systems to keep track of the supposed operation results.

    :return: ---
    """
    # reference data ----------------------------
    time_start_0 = "2042-01-01"
    time_start_1 = "2042-01-02"
    time_0 = pd.date_range(time_start_0, periods=3, freq="2D")
    time_1 = pd.date_range(time_start_1, periods=3, freq="2D")

    orientation_fix_0 = tf.rotation_matrix_z(np.pi * 0.5)
    orientation_fix_1 = tf.rotation_matrix_y(np.pi * 0.5)
    orientation_tdp_0 = tf.rotation_matrix_z(np.pi * np.array([0, 0.5, 1]))
    orientation_tdp_1 = tf.rotation_matrix_z(np.pi * np.array([1, 0, 0]))
    orientation_tdp_2 = tf.rotation_matrix_z(np.pi * np.array([0.75, 1.25, 0.75]))
    orientation_tdp_3 = tf.rotation_matrix_z(np.pi * np.array([1.5, 1.0, 0.75]))
    orientation_tdp_4 = tf.rotation_matrix_z(np.pi * np.array([1, 1.5, 1]))
    coordinates_fix_0 = ut.to_float_array([3, 7, 1])
    coordinates_fix_1 = ut.to_float_array([1, 4, 2])
    coordinates_tdp_0 = ut.to_float_array([[3, 7, 1], [4, -2, 8], [-5, 3, -1]])
    coordinates_tdp_1 = ut.to_float_array([[4, 2, 5], [3, -3, 2], [1, 7, -9]])

    # No time dependency ------------------------
    lcs0_in_base = tf.LocalCoordinateSystem(
        basis=orientation_fix_0, origin=coordinates_fix_0
    )
    lcs1_in_lcs_0 = tf.LocalCoordinateSystem(
        basis=orientation_fix_1, origin=coordinates_fix_1
    )

    lcs1_in_base = lcs1_in_lcs_0 + lcs0_in_base
    lcs1_in_lcs_0_calc = lcs1_in_base - lcs0_in_base

    orientation_exp = [[0, -1, 0], [0, 0, 1], [-1, 0, 0]]
    coordinates_exp = [-1, 8, 3]

    check_coordinate_system(lcs1_in_base, orientation_exp, coordinates_exp, True)
    check_coordinate_system(
        lcs1_in_lcs_0_calc, lcs1_in_lcs_0.basis, lcs1_in_lcs_0.origin, True
    )

    # orientation of left cs time dependent -----
    lcs0_in_base = tf.LocalCoordinateSystem(
        basis=orientation_fix_0, origin=coordinates_fix_0
    )
    lcs1_in_lcs_0 = tf.LocalCoordinateSystem(
        basis=orientation_tdp_0, origin=coordinates_fix_1, time=time_0
    )

    lcs1_in_base = lcs1_in_lcs_0 + lcs0_in_base
    lcs1_in_lcs_0_calc = lcs1_in_base - lcs0_in_base

    orientation_exp = tf.rotation_matrix_z(np.pi * np.array([0.5, 1, 1.5]))
    coordinates_exp = [[-1, 8, 3], [-1, 8, 3], [-1, 8, 3]]

    check_coordinate_system(
        lcs1_in_base, orientation_exp, coordinates_exp, True, time_0
    )
    check_coordinate_system(
        lcs1_in_lcs_0_calc, lcs1_in_lcs_0.basis, lcs1_in_lcs_0.origin, True, time_0
    )

    # coordinates of left cs time dependent -----
    lcs0_in_base = tf.LocalCoordinateSystem(
        basis=orientation_fix_0, origin=coordinates_fix_0
    )
    lcs1_in_lcs_0 = tf.LocalCoordinateSystem(
        basis=orientation_fix_1, origin=coordinates_tdp_0, time=time_0
    )

    lcs1_in_base = lcs1_in_lcs_0 + lcs0_in_base
    lcs1_in_lcs_0_calc = lcs1_in_base - lcs0_in_base

    orientation_exp = [
        [[0, -1, 0], [0, 0, 1], [-1, 0, 0]],
        [[0, -1, 0], [0, 0, 1], [-1, 0, 0]],
        [[0, -1, 0], [0, 0, 1], [-1, 0, 0]],
    ]
    coordinates_exp = [[-4, 10, 2], [5, 11, 9], [0, 2, 0]]
    check_coordinate_system(
        lcs1_in_base, orientation_exp, coordinates_exp, True, time_0
    )
    check_coordinate_system(
        lcs1_in_lcs_0_calc, lcs1_in_lcs_0.basis, lcs1_in_lcs_0.origin, True, time_0
    )

    # both fully time dependent, equal times ----
    lcs0_in_base = tf.LocalCoordinateSystem(
        basis=orientation_tdp_0, origin=coordinates_tdp_0, time=time_0
    )
    lcs1_in_lcs_0 = tf.LocalCoordinateSystem(
        basis=orientation_tdp_1, origin=coordinates_tdp_1, time=time_0
    )

    lcs1_in_base = lcs1_in_lcs_0 + lcs0_in_base
    lcs1_in_lcs_0_calc = lcs1_in_base - lcs0_in_base

    orientation_exp = tf.rotation_matrix_z(np.pi * np.array([1, 0.5, 1]))
    coordinates_exp = [[7, 9, 6], [7, 1, 10], [-6, -4, -10]]

    check_coordinate_system(
        lcs1_in_base, orientation_exp, coordinates_exp, True, time_0
    )
    check_coordinate_system(
        lcs1_in_lcs_0_calc, lcs1_in_lcs_0.basis, lcs1_in_lcs_0.origin, True, time_0
    )

    # both fully time dependent, different times - addition only
    """
    INFO: The subtraction can not be tested as in the previous tests by subtracting the
    added coordinate system and comparing the result to the initial one. The problem is,
    that the necessary interpolated values depend on the reference coordinate system,
    the interpolation is performed in. Since the reference systems differ between the
    addition and the subsequent subtraction, the result can not be compared to the
    initial coordinate system.
    """

    lcs0_in_base = tf.LocalCoordinateSystem(
        basis=orientation_tdp_2, origin=coordinates_tdp_0, time=time_0
    )
    lcs1_in_lcs_0 = tf.LocalCoordinateSystem(
        basis=orientation_tdp_3, origin=coordinates_tdp_1, time=time_1
    )

    lcs1_in_base = lcs1_in_lcs_0 + lcs0_in_base

    c = np.cos(np.pi * 0.75)
    s = np.sin(np.pi * 0.75)
    coordinates_exp = ut.to_float_array(
        [
            [-0.5, 0.5, 9.5],
            [-3.5, 3.5, 5.5],
            [-5 + c * 1 - s * 7, 3 + s * 1 + c * 7, -10],
        ]
    )
    orientation_exp = tf.rotation_matrix_z(np.pi * np.array([0.5, 0.0, 1.5]))

    check_coordinate_system(
        lcs1_in_base, orientation_exp, coordinates_exp, True, time_1
    )

    # both fully time dependent, different times - subtraction only
    lcs0_in_base = tf.LocalCoordinateSystem(
        basis=orientation_tdp_4, origin=coordinates_tdp_0, time=time_0
    )
    lcs1_in_base = tf.LocalCoordinateSystem(
        basis=orientation_tdp_3, origin=coordinates_tdp_1, time=time_1
    )

    lcs1_in_lcs_0 = lcs1_in_base - lcs0_in_base

    orientation_exp = tf.rotation_matrix_z(np.pi * np.array([0.25, 1.75, 1.75]))
    c = np.cos(np.pi * 0.75)
    s = np.sin(np.pi * 0.75)
    coordinates_exp = ut.to_float_array(
        [
            [0.5 + c * 4 - s * 2, 1.5 + s * 4 + c * 2, 0.5],
            [-3.5 + c * 3 - s * -3, -0.5 + s * 3 + c * -3, -1.5],
            [-6, -4, -8],
        ]
    )

    check_coordinate_system(
        lcs1_in_lcs_0, orientation_exp, coordinates_exp, True, time_1
    )

    # orientation of right cs time dependent -----
    lcs0 = tf.LocalCoordinateSystem(basis=orientation_fix_0, origin=coordinates_fix_0)
    lcs1 = tf.LocalCoordinateSystem(
        basis=orientation_tdp_0, origin=coordinates_fix_1, time=time_0
    )

    lcs_add = lcs0 + lcs1
    lcs_sub = lcs0 - lcs1

    orientation_add_exp = tf.rotation_matrix_z(np.pi * np.array([0.5, 1, 1.5]))
    coordinates_add_exp = [[4, 11, 3], [-6, 7, 3], [-2, -3, 3]]
    orientation_sub_exp = tf.rotation_matrix_z(np.pi * np.array([0.5, 0, 1.5]))
    coordinates_sub_exp = [[2, 3, -1], [3, -2, -1], [-2, -3, -1]]

    check_coordinate_system(
        lcs_add, orientation_add_exp, coordinates_add_exp, True, time_0
    )
    check_coordinate_system(
        lcs_sub, orientation_sub_exp, coordinates_sub_exp, True, time_0
    )

    # coordinates of right cs time dependent ----
    lcs0 = tf.LocalCoordinateSystem(basis=orientation_fix_0, origin=coordinates_fix_0)
    lcs1 = tf.LocalCoordinateSystem(
        basis=orientation_fix_0, origin=coordinates_tdp_0, time=time_0
    )

    lcs_add = lcs0 + lcs1
    lcs_sub = lcs0 - lcs1

    orientation_add_exp = tf.rotation_matrix_z(np.pi)
    coordinates_add_exp = [[-4, 10, 2], [-3, 1, 9], [-12, 6, 0]]
    orientation_sub_exp = tf.rotation_matrix_z(0)
    coordinates_sub_exp = [[0, 0, 0], [9, 1, -7], [4, -8, 2]]
    check_coordinate_system(
        lcs_add, orientation_add_exp, coordinates_add_exp, True, time_0
    )
    check_coordinate_system(
        lcs_sub, orientation_sub_exp, coordinates_sub_exp, True, time_0
    )

    # right cs with full time dependency --------
    lcs0 = tf.LocalCoordinateSystem(basis=orientation_fix_0, origin=coordinates_fix_0)
    lcs1 = tf.LocalCoordinateSystem(
        basis=orientation_tdp_0, origin=coordinates_tdp_0, time=time_0
    )

    lcs_add = lcs0 + lcs1
    lcs_sub = lcs0 - lcs1

    orientation_add_exp = tf.rotation_matrix_z(np.pi * np.array([0.5, 1, 1.5]))
    coordinates_add_exp = [[6, 14, 2], [-3, 1, 9], [-8, -4, 0]]
    orientation_sub_exp = tf.rotation_matrix_z(np.pi * np.array([0.5, 0, 1.5]))
    coordinates_sub_exp = [[0, 0, 0], [9, 1, -7], [-8, -4, 2]]
    print(lcs_sub.origin)
    check_coordinate_system(
        lcs_add, orientation_add_exp, coordinates_add_exp, True, time_0
    )
    check_coordinate_system(
        lcs_sub, orientation_sub_exp, coordinates_sub_exp, True, time_0
    )


def test_coordinate_system_invert():
    """
    Test the invert function.

    The test creates a coordinate system, inverts it and checks the result against the
    expected value. Afterwards, the resulting system is inverted again. This operation
    must yield the original system.

    :return: ---
    """
    # fix ---------------------------------------
    lcs0_in_lcs1 = tf.LocalCoordinateSystem.construct_from_xy_and_orientation(
        [1, 1, 0], [-1, 1, 0], origin=[2, 0, 2]
    )
    lcs1_in_lcs0 = lcs0_in_lcs1.invert()

    exp_basis = tf.rotation_matrix_z(-np.pi / 4)
    exp_origin = [-np.sqrt(2), np.sqrt(2), -2]

    check_coordinate_system(lcs1_in_lcs0, exp_basis, exp_origin, True)

    lcs0_in_lcs1_2 = lcs1_in_lcs0.invert()

    check_coordinate_system(
        lcs0_in_lcs1_2, lcs0_in_lcs1.basis, lcs0_in_lcs1.origin, True
    )

    # time dependent ----------------------------
    time = pd.date_range("2042-01-01", periods=4, freq="1D")
    orientation = tf.rotation_matrix_z(np.array([0, 0.5, 1, 0.5]) * np.pi)
    coordinates = np.array([[2, 8, 7], [4, 9, 2], [0, 2, 1], [3, 1, 2]])

    lcs0_in_lcs1 = tf.LocalCoordinateSystem(
        basis=orientation, origin=coordinates, time=time
    )

    lcs1_in_lcs0 = lcs0_in_lcs1.invert()
    orientation_exp = tf.rotation_matrix_z(np.array([0, 1.5, 1, 1.5]) * np.pi)
    coordinates_exp = np.array([[-2, -8, -7], [-9, 4, -2], [0, 2, -1], [-1, 3, -2]])

    check_coordinate_system(lcs1_in_lcs0, orientation_exp, coordinates_exp, True, time)

    lcs0_in_lcs1_2 = lcs1_in_lcs0.invert()

    check_coordinate_system(
        lcs0_in_lcs1_2, lcs0_in_lcs1.basis, lcs0_in_lcs1.origin, True, time
    )


def coordinate_system_time_interpolation_test_case(
    lcs: tf.LocalCoordinateSystem,
    time_interp: pd.DatetimeIndex,
    orientation_exp: np.ndarray,
    coordinates_exp: np.ndarray,
):
    """
    Test the time interpolation methods of the LocalCoordinateSystem class.

    :param lcs: Time dependent Local coordinate system
    :param time_interp: Target times for interpolation
    :param orientation_exp: Expected orientations
    :param coordinates_exp: Expected coordinates
    :return:
    """
    lcs_interp = lcs.interp_time(time_interp)
    check_coordinate_system(
        lcs_interp, orientation_exp, coordinates_exp, True, time_interp
    )

    lcs_interp_like = lcs.interp_time_like(lcs_interp)
    check_coordinate_system(
        lcs_interp_like, orientation_exp, coordinates_exp, True, time_interp
    )


def test_coordinate_system_time_interpolation():
    """
    Test the local coordinate systems interp_time and interp_like functions.

    :return: ---
    """
    time_0 = pd.date_range("2042-01-10", periods=4, freq="4D")
    orientation = tf.rotation_matrix_z(np.array([0, 0.5, 1, 0.5]) * np.pi)
    coordinates = np.array([[2, 8, 7], [4, 9, 2], [0, 2, 1], [3, 1, 2]])

    lcs = tf.LocalCoordinateSystem(basis=orientation, origin=coordinates, time=time_0)

    # broadcast left ----------------------------
    time_interp = pd.date_range("2042-01-01", periods=2, freq="1D")

    orientation_exp = tf.rotation_matrix_z(np.array([0, 0]) * np.pi)
    coordinates_exp = np.array([[2, 8, 7], [2, 8, 7]])

    coordinate_system_time_interpolation_test_case(
        lcs, time_interp, orientation_exp, coordinates_exp
    )

    # broadcast right ---------------------------
    time_interp = pd.date_range("2042-02-01", periods=2, freq="1D")

    orientation_exp = tf.rotation_matrix_z(np.array([0.5, 0.5]) * np.pi)
    coordinates_exp = np.array([[3, 1, 2], [3, 1, 2]])

    coordinate_system_time_interpolation_test_case(
        lcs, time_interp, orientation_exp, coordinates_exp
    )

    # pure interpolation ------------------------
    time_interp = pd.date_range("2042-01-11", periods=4, freq="3D")

    orientation_exp = tf.rotation_matrix_z(np.array([0.125, 0.5, 0.875, 0.75]) * np.pi)
    coordinates_exp = np.array(
        [[2.5, 8.25, 5.75], [4, 9, 2], [1, 3.75, 1.25], [1.5, 1.5, 1.5]]
    )

    coordinate_system_time_interpolation_test_case(
        lcs, time_interp, orientation_exp, coordinates_exp
    )

    # mixed -------------------------------------
    time_interp = pd.date_range("2042-01-06", periods=5, freq="6D")

    orientation_exp = tf.rotation_matrix_z(np.array([0, 0.25, 1, 0.5, 0.5]) * np.pi)
    coordinates_exp = np.array(
        [[2, 8, 7], [3, 8.5, 4.5], [0, 2, 1], [3, 1, 2], [3, 1, 2]]
    )

    coordinate_system_time_interpolation_test_case(
        lcs, time_interp, orientation_exp, coordinates_exp
    )

    # exceptions --------------------------------
    # wrong parameter type
    with pytest.raises(Exception):
        lcs.interp_time("wrong")
    with pytest.raises(Exception):
        lcs.interp_time_like("wrong")
    # no time component
    with pytest.raises(Exception):
        lcs.interp_time_like(tf.LocalCoordinateSystem())


def test_coordinate_system_manager_init():
    """
    Test the init method of the coordinate system manager.

    :return: ---
    """
    # default construction ----------------------
    csm = tf.CoordinateSystemManager(root_coordinate_system_name="root")
    assert csm.number_of_coordinate_systems == 1
    assert csm.number_of_neighbors("root") == 0

    # Exceptions---------------------------------
    # Invalid root system name
    with pytest.raises(Exception):
        tf.CoordinateSystemManager({})


def test_coordinate_system_manager_add_coordinate_system():
    """
    Test the add_coordinate_system function of the coordinate system manager.

    Adds some coordinate systems to a CSM and checks if the the edges and nodes
    are set as expected.

    :return: ---
    """
    # define some coordinate systems
    lcs1_in_root = tf.LocalCoordinateSystem(tf.rotation_matrix_z(np.pi / 2), [1, 2, 3])
    lcs2_in_root = tf.LocalCoordinateSystem(tf.rotation_matrix_y(np.pi / 2), [3, -3, 1])
    lcs3_in_lcs2 = tf.LocalCoordinateSystem(tf.rotation_matrix_y(np.pi / 2), [3, -3, 1])

    csm = tf.CoordinateSystemManager(root_coordinate_system_name="root")
    assert csm.number_of_coordinate_systems == 1
    assert csm.number_of_neighbors("root") == 0

    csm.add_coordinate_system("lcs1", "root", lcs1_in_root)
    assert csm.number_of_coordinate_systems == 2
    assert csm.number_of_neighbors("root") == 1
    assert csm.number_of_neighbors("lcs1") == 1
    assert csm.is_neighbor_of("root", "lcs1")
    assert csm.is_neighbor_of("lcs1", "root")

    csm.add_coordinate_system("lcs2", "root", lcs2_in_root)
    assert csm.number_of_coordinate_systems == 3
    assert csm.number_of_neighbors("root") == 2
    assert csm.number_of_neighbors("lcs1") == 1
    assert csm.number_of_neighbors("lcs2") == 1
    assert csm.is_neighbor_of("root", "lcs2")
    assert csm.is_neighbor_of("lcs2", "root")
    assert not csm.is_neighbor_of("lcs1", "lcs2")
    assert not csm.is_neighbor_of("lcs2", "lcs1")

    csm.add_coordinate_system("lcs3", "lcs2", lcs3_in_lcs2)
    assert csm.number_of_coordinate_systems == 4
    assert csm.number_of_neighbors("root") == 2
    assert csm.number_of_neighbors("lcs1") == 1
    assert csm.number_of_neighbors("lcs2") == 2
    assert csm.number_of_neighbors("lcs3") == 1
    assert not csm.is_neighbor_of("root", "lcs3")
    assert not csm.is_neighbor_of("lcs3", "root")
    assert not csm.is_neighbor_of("lcs1", "lcs3")
    assert not csm.is_neighbor_of("lcs3", "lcs1")
    assert csm.is_neighbor_of("lcs2", "lcs3")
    assert csm.is_neighbor_of("lcs3", "lcs2")

    # Exceptions---------------------------------
    # Incorrect coordinate system type
    with pytest.raises(Exception):
        csm.add_coordinate_system("lcs4", "root", "wrong")

    # Coordinate system already exists
    with pytest.raises(Exception):
        csm.add_coordinate_system("lcs3", "lcs2", lcs3_in_lcs2)

    # Invalid parent system
    with pytest.raises(Exception):
        csm.add_coordinate_system("lcs4", "something", tf.LocalCoordinateSystem())


def test_coordinate_system_manager_get_local_coordinate_system_no_time_dependence():
    """
    Test the get_local_coordinate_system function.

    This function also tests, if the internally performed transformations are correct.

    :return: ---
    """
    # define some coordinate systems
    lcs1_in_root = tf.LocalCoordinateSystem(tf.rotation_matrix_z(np.pi / 2), [1, 2, 3])
    lcs2_in_root = tf.LocalCoordinateSystem(tf.rotation_matrix_y(np.pi / 2), [3, -3, 1])
    lcs3_in_lcs2 = tf.LocalCoordinateSystem(tf.rotation_matrix_x(np.pi / 2), [1, -1, 3])

    csm = tf.CoordinateSystemManager(root_coordinate_system_name="root")
    csm.add_coordinate_system("lcs1", "root", lcs1_in_root)
    csm.add_coordinate_system("lcs2", "root", lcs2_in_root)
    csm.add_coordinate_system("lcs3", "lcs2", lcs3_in_lcs2)

    # check stored transformations
    lcs1_in_root_returned = csm.get_local_coordinate_system("lcs1", "root")
    check_coordinate_system(
        lcs1_in_root_returned, lcs1_in_root.basis, lcs1_in_root.origin, True
    )

    lcs2_in_root_returned = csm.get_local_coordinate_system("lcs2", "root")
    check_coordinate_system(
        lcs2_in_root_returned, lcs2_in_root.basis, lcs2_in_root.origin, True
    )

    lcs3_in_lcs2_returned = csm.get_local_coordinate_system("lcs3", "lcs2")
    check_coordinate_system(
        lcs3_in_lcs2_returned, lcs3_in_lcs2.basis, lcs3_in_lcs2.origin, True
    )

    # check calculated transformations
    lcs_3_in_root = csm.get_local_coordinate_system("lcs3", "root")
    expected_basis = [[0, 1, 0], [0, 0, -1], [-1, 0, 0]]
    expected_coordinates = [6, -4, 0]
    check_coordinate_system(lcs_3_in_root, expected_basis, expected_coordinates, True)

    root_in_lcs3 = csm.get_local_coordinate_system("root", "lcs3")
    expected_basis = [[0, 0, -1], [1, 0, 0], [0, -1, 0]]
    expected_coordinates = [0, -6, -4]
    check_coordinate_system(root_in_lcs3, expected_basis, expected_coordinates, True)

    lcs_3_in_lcs1 = csm.get_local_coordinate_system("lcs3", "lcs1")
    expected_basis = [[0, 0, -1], [0, -1, 0], [-1, 0, 0]]
    expected_coordinates = [-6, -5, -3]
    check_coordinate_system(lcs_3_in_lcs1, expected_basis, expected_coordinates, True)

    lcs_1_in_lcs3 = csm.get_local_coordinate_system("lcs1", "lcs3")
    expected_basis = [[0, 0, -1], [0, -1, 0], [-1, 0, 0]]
    expected_coordinates = [-3, -5, -6]
    check_coordinate_system(lcs_1_in_lcs3, expected_basis, expected_coordinates, True)


def test_coordinate_system_manager_time_union():
    """
    Test the coordinate system managers time union function.

    :return: ---
    """
    orientation = tf.rotation_matrix_z([0, 1, 2])
    coordinates = [[1, 6, 3], [8, 2, 6], [4, 4, 4]]
    lcs_0 = tf.LocalCoordinateSystem(
        basis=orientation,
        origin=coordinates,
        time=pd.date_range("2042-01-01", periods=3, freq="3D"),
    )
    lcs_1 = tf.LocalCoordinateSystem(
        basis=orientation,
        origin=coordinates,
        time=pd.date_range("2042-01-01", periods=3, freq="4D"),
    )
    lcs_2 = tf.LocalCoordinateSystem(
        basis=orientation,
        origin=coordinates,
        time=pd.date_range("2042-01-01", periods=3, freq="5D"),
    )
    lcs_3 = tf.LocalCoordinateSystem()

    csm = tf.CoordinateSystemManager("root")
    csm.add_coordinate_system("lcs_0", "root", lcs_0)
    csm.add_coordinate_system("lcs_1", "lcs_0", lcs_1)
    csm.add_coordinate_system("lcs_2", "root", lcs_2)
    csm.add_coordinate_system("lcs_3", "lcs_2", lcs_3)

    # full union --------------------------------
    expected_times = pd.DatetimeIndex(
        [
            "2042-01-01",
            "2042-01-04",
            "2042-01-05",
            "2042-01-06",
            "2042-01-07",
            "2042-01-09",
            "2042-01-11",
        ]
    )

    assert np.all(expected_times == csm.time_union())

    # selected union ------------------------------
    expected_times = pd.DatetimeIndex(
        ["2042-01-01", "2042-01-04", "2042-01-05", "2042-01-07", "2042-01-09"]
    )
    list_of_edges = [("root", "lcs_0"), ("lcs_0", "lcs_1")]

    assert np.all(expected_times == csm.time_union(list_of_edges=list_of_edges))


def test_coordinate_system_manager_interp_time():
    """
    Test the coordinate system managers interp_time and interp_like functions.

    :return: ---
    """
    # Setup -------------------------------------
    angles = ut.to_float_array([0, np.pi / 2, np.pi])
    orientation = tf.rotation_matrix_z(angles)
    coordinates = ut.to_float_array([[5, 0, 0], [1, 0, 0], [1, 4, 4]])
    lcs_0_in_root = tf.LocalCoordinateSystem(
        basis=orientation,
        origin=coordinates,
        time=pd.date_range("2042-01-01", periods=3, freq="3D"),
    )
    lcs_1_in_lcs_0 = tf.LocalCoordinateSystem(
        basis=orientation,
        origin=coordinates,
        time=pd.date_range("2042-01-01", periods=3, freq="4D"),
    )
    lcs_2_in_root = tf.LocalCoordinateSystem(
        basis=orientation,
        origin=coordinates,
        time=pd.date_range("2042-01-01", periods=3, freq="5D"),
    )
    lcs_3_in_lcs_2 = tf.LocalCoordinateSystem(
        basis=tf.rotation_matrix_y(1), origin=[4, 2, 0]
    )

    csm = tf.CoordinateSystemManager("root")
    csm.add_coordinate_system("lcs_0", "root", lcs_0_in_root)
    csm.add_coordinate_system("lcs_1", "lcs_0", lcs_1_in_lcs_0)
    csm.add_coordinate_system("lcs_2", "root", lcs_2_in_root)
    csm.add_coordinate_system("lcs_3", "lcs_2", lcs_3_in_lcs_2)

    # interp_time -------------------------------
    time_interp = pd.date_range("2042-01-01", periods=5, freq="2D")
    csm_interp = csm.interp_time(time_interp)

    assert np.all(csm_interp.time_union() == time_interp)

    lcs_0_in_root_csm = csm_interp.get_local_coordinate_system("lcs_0", "root")
    lcs_1_in_lcs_0_csm = csm_interp.get_local_coordinate_system("lcs_1", "lcs_0")
    lcs_2_in_root_csm = csm_interp.get_local_coordinate_system("lcs_2", "root")
    lcs_3_in_lcs_2_csm = csm_interp.get_local_coordinate_system("lcs_3", "lcs_2")

    assert np.all(lcs_0_in_root_csm.time == time_interp)
    assert np.all(lcs_1_in_lcs_0_csm.time == time_interp)
    assert np.all(lcs_2_in_root_csm.time == time_interp)
    assert lcs_3_in_lcs_2_csm.time is None

    coordinates_exp = [
        [5, 0, 0],
        [7 / 3, 0, 0],
        [1, 4 / 3, 4 / 3],
        [1, 4, 4],
        [1, 4, 4],
    ]
    orientation_exp = tf.rotation_matrix_z([0, np.pi / 3, 2 * np.pi / 3, np.pi, np.pi])
    check_coordinate_system(lcs_0_in_root_csm, orientation_exp, coordinates_exp, True)

    coordinates_exp = [[5, 0, 0], [3, 0, 0], [1, 0, 0], [1, 2, 2], [1, 4, 4]]
    orientation_exp = tf.rotation_matrix_z(
        [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]
    )
    check_coordinate_system(lcs_1_in_lcs_0_csm, orientation_exp, coordinates_exp, True)

    coordinates_exp = [
        [5, 0, 0],
        [17 / 5, 0, 0],
        [9 / 5, 0, 0],
        [1, 4 / 5, 4 / 5],
        [1, 12 / 5, 12 / 5],
    ]
    orientation_exp = tf.rotation_matrix_z(
        [0, np.pi / 5, 2 * np.pi / 5, 3 * np.pi / 5, 4 * np.pi / 5]
    )
    check_coordinate_system(lcs_2_in_root_csm, orientation_exp, coordinates_exp, True)

    check_coordinate_system(
        lcs_3_in_lcs_2_csm, lcs_3_in_lcs_2.basis, lcs_3_in_lcs_2.origin, True
    )

    # interp_like -------------------------------

    csm_interp_like = csm.interp_time_like(lcs_1_in_lcs_0)
    assert np.all(csm_interp_like.time_union() == lcs_1_in_lcs_0.time)

    lcs_0_in_root_csm = csm_interp_like.get_local_coordinate_system("lcs_0", "root")
    lcs_1_in_lcs_0_csm = csm_interp_like.get_local_coordinate_system("lcs_1", "lcs_0")
    lcs_2_in_root_csm = csm_interp_like.get_local_coordinate_system("lcs_2", "root")
    lcs_3_in_lcs_2_csm = csm_interp_like.get_local_coordinate_system("lcs_3", "lcs_2")

    assert np.all(lcs_0_in_root_csm.time == lcs_1_in_lcs_0.time)
    assert np.all(lcs_1_in_lcs_0_csm.time == lcs_1_in_lcs_0.time)
    assert np.all(lcs_2_in_root_csm.time == lcs_1_in_lcs_0.time)
    assert lcs_3_in_lcs_2_csm.time is None

    coordinates_exp = [[5, 0, 0], [1, 4 / 3, 4 / 3], [1, 4, 4]]
    orientation_exp = tf.rotation_matrix_z([0, 2 * np.pi / 3, np.pi])
    check_coordinate_system(lcs_0_in_root_csm, orientation_exp, coordinates_exp, True)

    check_coordinate_system(
        lcs_1_in_lcs_0_csm, lcs_1_in_lcs_0.orientation, lcs_1_in_lcs_0.origin, True
    )

    coordinates_exp = [[5, 0, 0], [9 / 5, 0, 0], [1, 12 / 5, 12 / 5]]
    orientation_exp = tf.rotation_matrix_z([0, 2 * np.pi / 5, 4 * np.pi / 5])
    check_coordinate_system(lcs_2_in_root_csm, orientation_exp, coordinates_exp, True)

    check_coordinate_system(
        lcs_3_in_lcs_2_csm, lcs_3_in_lcs_2.basis, lcs_3_in_lcs_2.origin, True
    )

    # exceptions --------------------------------
    # invalid input type
    with pytest.raises(Exception):
        csm.interp_time("wrong")
    with pytest.raises(Exception):
        csm.interp_time_like("wrong")

    # no time component
    with pytest.raises(Exception):
        csm.interp_time_like(lcs_3_in_lcs_2)


def test_coordinate_system_manager_transform_data():
    """
    Test the coordinate system managers transform_data function.

    :return: ---
    """
    # define some coordinate systems
    # TODO: test more unique rotations - not 90°
    lcs1_in_root = tf.LocalCoordinateSystem(tf.rotation_matrix_z(np.pi / 2), [1, 2, 3])
    lcs2_in_root = tf.LocalCoordinateSystem(tf.rotation_matrix_y(np.pi / 2), [3, -3, 1])
    lcs3_in_lcs2 = tf.LocalCoordinateSystem(tf.rotation_matrix_x(np.pi / 2), [1, -1, 3])

    csm = tf.CoordinateSystemManager(root_coordinate_system_name="root")
    csm.add_coordinate_system("lcs_1", "root", lcs1_in_root)
    csm.add_coordinate_system("lcs_2", "root", lcs2_in_root)
    csm.add_coordinate_system("lcs_3", "lcs_2", lcs3_in_lcs2)

    data_list = [[1, -3, -1], [2, 4, -1], [-1, 2, 3], [3, -4, 2]]
    data_exp = np.array([[-5, -2, -4], [-5, -9, -5], [-9, -7, -2], [-8, -1, -6]])

    # input list
    data_list_transformed = csm.transform_data(data_list, "lcs_3", "lcs_1")
    assert ut.matrix_is_close(data_list_transformed, data_exp)

    # input numpy array
    data_np = np.array(data_list)
    data_numpy_transformed = csm.transform_data(data_np, "lcs_3", "lcs_1")
    assert ut.matrix_is_close(data_numpy_transformed, data_exp)

    # input single numpy vector
    data_numpy_transformed = csm.transform_data(data_np[0, :], "lcs_3", "lcs_1")
    assert ut.vector_is_close(data_numpy_transformed, data_exp[0, :])

    # input xarray
    data_xr = xr.DataArray(data=data_np, dims=["n", "c"], coords={"c": ["x", "y", "z"]})
    data_xr_transformed = csm.transform_data(data_xr, "lcs_3", "lcs_1")
    assert ut.matrix_is_close(data_xr_transformed.data, data_exp)

    # TODO: Test time dependency

    # exceptions --------------------------------
    # names not in csm
    with pytest.raises(ValueError):
        csm.transform_data(data_xr, "not present", "lcs_1")
    with pytest.raises(ValueError):
        csm.transform_data(data_xr, "lcs_3", "not present")

    # data is not compatible
    with pytest.raises(Exception):
        csm.transform_data("wrong", "lcs_3", "lcs_1")


def test_coordinate_system_manager_data_assignment_and_retrieval():
    """
    Test the coordinate system managers assign_data and get_data functions.

    :return: ---
    """
    # test setup
    lcs1_in_root = tf.LocalCoordinateSystem(tf.rotation_matrix_z(np.pi / 2), [1, 2, 3])
    lcs2_in_root = tf.LocalCoordinateSystem(tf.rotation_matrix_y(np.pi / 2), [3, -3, 1])
    lcs3_in_lcs2 = tf.LocalCoordinateSystem(tf.rotation_matrix_x(np.pi / 2), [1, -1, 3])

    csm = tf.CoordinateSystemManager(root_coordinate_system_name="root")
    csm.add_coordinate_system("lcs_1", "root", lcs1_in_root)
    csm.add_coordinate_system("lcs_2", "root", lcs2_in_root)
    csm.add_coordinate_system("lcs_3", "lcs_2", lcs3_in_lcs2)

    data_xr = xr.DataArray(
        data=[[1, -3, -1], [2, 4, -1], [-1, 2, 3], [3, -4, 2]],
        dims=["n", "c"],
        coords={"c": ["x", "y", "z"]},
    )

    data_xr_exp = xr.DataArray(
        data=[[-5, -2, -4], [-5, -9, -5], [-9, -7, -2], [-8, -1, -6]],
        dims=["n", "c"],
        coords={"c": ["x", "y", "z"]},
    )

    # actual test
    assert csm.has_data("lcs_3", "my data") is False
    csm.assign_data(data_xr, "my data", "lcs_3")
    assert csm.has_data("lcs_3", "my data") is True

    assert csm.get_data("my data").equals(data_xr)
    assert csm.get_data("my data", "lcs_3").equals(data_xr)

    transformed_data = csm.get_data("my data", "lcs_1")
    assert ut.matrix_is_close(transformed_data.data, data_xr_exp.data)
    # TODO: Also check coords and dims

    # exceptions --------------------------------
    # assignment - invalid data name
    with pytest.raises(TypeError):
        csm.assign_data(data_xr, {"wrong"}, "root")
    # assignment - coordinate system does not exist
    with pytest.raises(ValueError):
        csm.assign_data(data_xr, "some data", "not there")
    # TODO: Unsupported data type ---> no spatial component

    # has_data - coordinate system does not exist
    with pytest.raises(Exception):
        csm.has_data("wrong", "not there")

    # get_data - data does not exist
    with pytest.raises(Exception):
        csm.get_data("not there", "root")
    # get_data - coordinate system does not exist
    with pytest.raises(Exception):
        csm.get_data("my data", "not there")


# TODO: Test time dependent get_local_coordinate_system
