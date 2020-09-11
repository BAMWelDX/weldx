"""Tests the transformation package."""

import math
import random
from copy import deepcopy
from typing import Any, List, Union

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pandas import TimedeltaIndex as TDI
from pandas import Timestamp as TS
from pandas import date_range

import weldx.transformations as tf
import weldx.utility as ut
from weldx.constants import WELDX_QUANTITY as Q_
from weldx.transformations import LocalCoordinateSystem as LCS

# helpers for tests -----------------------------------------------------------


# Todo: Move this to conftest.py?
def get_test_name(param):
    """Get the test name from the parameter list of a parametrized test."""
    if isinstance(param, str) and param[0] == "#":
        return param[1:]
    return ""


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
    r_x = tf.rotation_matrix_x(angle_x)
    r_y = tf.rotation_matrix_y(angle_y)
    r_z = tf.rotation_matrix_z(angle_z)

    r_tot = np.matmul(r_z, np.matmul(r_y, r_x))
    return r_tot


# test functions --------------------------------------------------------------


def test_coordinate_axis_rotation_matrices():
    """Test the rotation matrices that rotate around one coordinate axis.

    This test creates the rotation matrices using 10 degree steps and
    multiplies them with a given vector. The result is compared to the
    expected values, which are determined using the sine and cosine.
    Additionally, some matrix properties are checked.

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


# --------------------------------------------------------------------------------------
# LocalCoordinateSystem
# --------------------------------------------------------------------------------------


class TestLocalCoordinateSystem:
    """Test the 'LocalCoordinateSystem' class."""

    # support functions ----------------------------------------------------------------

    @staticmethod
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

    @classmethod
    def check_coordinate_system(
        cls,
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
            check_coordinate_system_time(lcs, time)
            assert lcs.reference_time == time_ref

        cls.check_coordinate_system_orientation(
            lcs.orientation, orientation_expected, positive_orientation_expected
        )

        assert np.allclose(lcs.coordinates.values, coordinates_expected, atol=1e-9)

    # test_init_time_formats -----------------------------------------------------------

    timestamp = TS("2000-01-01")
    time_delta = TDI([0, 1, 2], "s")
    time_quantity = Q_([0, 1, 2], "s")
    date_time = date_range("2000-01-01", periods=3, freq="s")

    @staticmethod
    @pytest.mark.parametrize(
        "time, time_ref, time_exp, time_ref_exp, datetime_exp, quantity_exp",
        [
            (time_delta, None, time_delta, None, None, time_quantity),
            (time_delta, timestamp, time_delta, timestamp, date_time, time_quantity),
            (time_quantity, None, time_delta, None, None, time_quantity),
            (time_quantity, timestamp, time_delta, timestamp, date_time, time_quantity),
            (date_time, None, time_delta, timestamp, date_time, time_quantity),
            (
                date_time,
                TS("1999-12-31"),
                TDI([86400, 86401, 86402], "s"),
                TS("1999-12-31"),
                date_time,
                Q_([86400, 86401, 86402], "s"),
            ),
        ],
    )
    def test_init_time_formats(
        time, time_ref, time_exp, time_ref_exp, datetime_exp, quantity_exp
    ):
        """Test the __init__ method with the different supported time formats.

        Parameters
        ----------
        time:
            Time object passed to the __init__ method
        time_ref:
            Reference time passed to the __init__ method
        time_exp:
            Expected return value of the 'time' property
        time_ref_exp:
            Expected return value of the 'reference_time' property
        datetime_exp:
            Expected return value of the 'datetimeindex' property
        quantity_exp:
            Expected return value of the 'time_quantity' property

        """
        # setup
        orientation = tf.rotation_matrix_z(np.array([0.5, 1.0, 1.5]) * np.pi)
        coordinates = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        lcs = tf.LocalCoordinateSystem(
            orientation, coordinates, time, time_ref=time_ref
        )

        # check results

        assert np.all(lcs.time == time_exp)
        assert lcs.reference_time == time_ref_exp
        assert np.all(lcs.datetimeindex == datetime_exp)
        assert np.all(lcs.time_quantity == quantity_exp)

    # test_init_time_dsx ---------------------------------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        "time_o,  time_c,  time_exp",
        [
            (TDI([0, 1, 2], "s"), TDI([0, 1, 2], "s"), TDI([0, 1, 2], "s")),
            (TDI([0, 2, 4], "s"), TDI([1, 3, 5], "s"), TDI([0, 1, 2, 3, 4, 5], "s"),),
        ],
    )
    @pytest.mark.parametrize("time_ref", [None, TS("2020-02-02")])
    def test_init_time_dsx(time_o, time_c, time_exp, time_ref):
        """Test if __init__ sets the internal time correctly when DataArrays are passed.

        Parameters
        ----------
        time_o:
            Time of the orientation DataArray
        time_c:
            Time of the coordinates DataArray
        time_exp:
            Expected result time
        time_ref:
            The coordinate systems reference time

        """
        orientations = tf.rotation_matrix_z(np.array(range(len(time_o))))
        coordinates = [[i, i, i] for i in range(len(time_o))]

        dax_o = ut.xr_3d_matrix(orientations, time_o)
        dax_c = ut.xr_3d_vector(coordinates, time_c)

        lcs = tf.LocalCoordinateSystem(dax_o, dax_c, time_ref=time_ref)

        # check results

        assert np.all(lcs.time == time_exp)
        assert lcs.reference_time == time_ref

    # test_reset_reference_time --------------------------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        "time, time_ref, time_ref_new, time_exp",
        [
            (
                TDI([1, 2, 3], "D"),
                TS("2020-02-02"),
                TS("2020-02-01"),
                TDI([2, 3, 4], "D"),
            ),
            (TDI([1, 2, 3], "D"), TS("2020-02-02"), "2020-02-01", TDI([2, 3, 4], "D"),),
        ],
    )
    def test_reset_reference_time(time, time_ref, time_ref_new, time_exp):
        """Test the 'reset_reference_time' function.

        Parameters
        ----------
        time:
            The time of the LCS
        time_ref:
            The reference time of the LCS
        time_ref_new:
            Reference time that should be set
        time_exp:
            Expected time of the LCS after the reset

        """
        orientation = tf.rotation_matrix_z([1, 2, 3])
        coordinates = [[i, i, i] for i in range(3)]
        lcs = tf.LocalCoordinateSystem(
            orientation, coordinates, time, time_ref=time_ref
        )

        lcs.reset_reference_time(time_ref_new)

        # check results
        assert np.all(lcs.time == time_exp)

    # test_reset_reference_time exceptions ---------------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        "time_ref, time_ref_new,  exception_type, test_name",
        [
            (TS("2020-02-02"), None, TypeError, "# invalid type #1"),
            (TS("2020-02-02"), 42, TypeError, "# invalid type #2"),
            (None, TS("2020-02-02"), TypeError, "# lcs has no reference time"),
        ],
        ids=get_test_name,
    )
    def test_delete_coordinate_system_exceptions(
        time_ref, time_ref_new, exception_type, test_name
    ):
        """Test the exceptions of the 'reset_reference_time' method.

        Parameters
        ----------
        time_ref:
            Reference time of the LCS
        time_ref_new:
            Reference time that should be set
        exception_type:
            Expected exception type
        test_name:
            Name of the test

        """
        orientation = tf.rotation_matrix_z([1, 2, 3])
        coordinates = [[i, i, i] for i in range(3)]
        time = TDI([1, 2, 3], "D")

        lcs = tf.LocalCoordinateSystem(
            orientation, coordinates, time, time_ref=time_ref
        )

        with pytest.raises(exception_type):
            lcs.reset_reference_time(time_ref_new)

    # test_interp_time -----------------------------------------------------------------

    @pytest.mark.parametrize(
        "time_ref_lcs, time,time_ref, orientation_exp, coordinates_exp",
        [
            (  # broadcast left
                TS("2020-02-10"),
                TDI([1, 2], "D"),
                TS("2020-02-10"),
                tf.rotation_matrix_z(np.array([0, 0]) * np.pi),
                np.array([[2, 8, 7], [2, 8, 7]]),
            ),
            (  # broadcast right
                TS("2020-02-10"),
                TDI([29, 30], "D"),
                TS("2020-02-10"),
                tf.rotation_matrix_z(np.array([0.5, 0.5]) * np.pi),
                np.array([[3, 1, 2], [3, 1, 2]]),
            ),
            (  # pure interpolation
                TS("2020-02-10"),
                TDI([11, 14, 17, 20], "D"),
                TS("2020-02-10"),
                tf.rotation_matrix_z(np.array([0.125, 0.5, 0.875, 0.75]) * np.pi),
                np.array(
                    [[2.5, 8.25, 5.75], [4, 9, 2], [1, 3.75, 1.25], [1.5, 1.5, 1.5]]
                ),
            ),
            (  # mixed
                TS("2020-02-10"),
                TDI([6, 12, 18, 24, 32], "D"),
                TS("2020-02-10"),
                tf.rotation_matrix_z(np.array([0, 0.25, 1, 0.5, 0.5]) * np.pi),
                np.array([[2, 8, 7], [3, 8.5, 4.5], [0, 2, 1], [3, 1, 2], [3, 1, 2]]),
            ),
            (  # different reference times
                TS("2020-02-10"),
                TDI([8, 14, 20, 26, 34], "D"),
                TS("2020-02-08"),
                tf.rotation_matrix_z(np.array([0, 0.25, 1, 0.5, 0.5]) * np.pi),
                np.array([[2, 8, 7], [3, 8.5, 4.5], [0, 2, 1], [3, 1, 2], [3, 1, 2]]),
            ),
            (  # no reference time
                None,
                TDI([6, 12, 18, 24, 32], "D"),
                None,
                tf.rotation_matrix_z(np.array([0, 0.25, 1, 0.5, 0.5]) * np.pi),
                np.array([[2, 8, 7], [3, 8.5, 4.5], [0, 2, 1], [3, 1, 2], [3, 1, 2]]),
            ),
        ],
    )
    def test_interp_time(
        self, time_ref_lcs, time, time_ref, orientation_exp, coordinates_exp
    ):
        """Test the interp_time function.

        Parameters
        ----------
        time_ref_lcs:
            Reference time of the coordinate system
        time:
            Time that is passed to the function
        time_ref:
            Reference time that is passed to the function
        orientation_exp:
            Expected orientations of the result
        coordinates_exp:
            Expected coordinates of the result

        """
        # setup
        lcs = tf.LocalCoordinateSystem(
            orientation=tf.rotation_matrix_z(np.array([0, 0.5, 1, 0.5]) * np.pi),
            coordinates=np.array([[2, 8, 7], [4, 9, 2], [0, 2, 1], [3, 1, 2]]),
            time=TDI([10, 14, 18, 22], "D"),
            time_ref=time_ref_lcs,
        )

        # test time as input
        lcs_interp = lcs.interp_time(time, time_ref)
        self.check_coordinate_system(
            lcs_interp, orientation_exp, coordinates_exp, True, time, time_ref
        )

        # test lcs as input
        lcs_interp_like = lcs.interp_time(lcs_interp)
        self.check_coordinate_system(
            lcs_interp_like, orientation_exp, coordinates_exp, True, time, time_ref
        )

    # test_addition --------------------------------------------------------------------

    # reference data ----------------------------
    time_0 = TDI([1, 3, 5], "D")
    time_1 = TDI([2, 4, 6], "D")

    time_ref_0 = TS("2020-02-02")
    time_ref_1 = TS("2020-02-03")

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

    lcs_s0 = LCS(orientation_fix_0, coordinates_fix_0)
    lcs_s1 = LCS(orientation_fix_1, coordinates_fix_1)

    lcs_to_0 = tf.LocalCoordinateSystem(
        orientation_tdp_0, coordinates_fix_1, time_0, time_ref=time_ref_0
    )

    lcs_tc_0 = tf.LocalCoordinateSystem(
        orientation_fix_1, coordinates_tdp_0, time_0, time_ref=time_ref_0
    )

    lcs_toc_0 = tf.LocalCoordinateSystem(
        orientation_tdp_0, coordinates_tdp_0, time=time_0, time_ref=time_ref_0
    )
    lcs_toc_1 = tf.LocalCoordinateSystem(
        orientation_tdp_1, coordinates_tdp_1, time=time_0, time_ref=time_ref_0
    )

    lcs_toc_2 = tf.LocalCoordinateSystem(
        orientation_tdp_2, coordinates_tdp_0, time_0, time_ref=time_ref_0
    )
    lcs_toc_3 = tf.LocalCoordinateSystem(
        orientation_tdp_3, coordinates_tdp_1, time_1, time_ref=time_ref_0
    )
    lcs_toc_4 = tf.LocalCoordinateSystem(
        orientation_tdp_3, coordinates_tdp_1, time_0, time_ref=time_ref_1
    )

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

    @pytest.mark.parametrize(
        "lcs_lhs, lcs_rhs, orientation_exp, coordinates_exp, time_exp, time_ref_exp",
        [
            (  # both static
                lcs_s1,
                lcs_s0,
                [[0, -1, 0], [0, 0, 1], [-1, 0, 0]],
                [-1, 8, 3],
                None,
                None,
            ),
            (  # left system orientation time dependent
                lcs_to_0,
                lcs_s0,
                tf.rotation_matrix_z(np.pi * np.array([0.5, 1, 1.5])),
                [[-1, 8, 3], [-1, 8, 3], [-1, 8, 3]],
                time_0,
                time_ref_0,
            ),
            (  # left system coordinates time dependent
                lcs_tc_0,
                lcs_s0,
                [[[0, -1, 0], [0, 0, 1], [-1, 0, 0]] for _ in range(3)],
                [[-4, 10, 2], [5, 11, 9], [0, 2, 0]],
                time_0,
                time_ref_0,
            ),
            (  # both fully time dependent - same time and reference time
                lcs_toc_1,
                lcs_toc_0,
                tf.rotation_matrix_z(np.pi * np.array([1, 0.5, 1])),
                [[7, 9, 6], [7, 1, 10], [-6, -4, -10]],
                time_0,
                time_ref_0,
            ),
            (  # both fully time dependent - different time but same reference time
                lcs_toc_3,
                lcs_toc_2,
                tf.rotation_matrix_z(np.pi * np.array([0.5, 0.0, 1.5])),
                [
                    [-0.5, 0.5, 9.5],
                    [-3.5, 3.5, 5.5],
                    [-5 + c * 1 - s * 7, 3 + s * 1 + c * 7, -10],
                ],
                time_1,
                time_ref_0,
            ),
            (  # both fully time dependent - different time and reference time
                lcs_toc_4,
                lcs_toc_2,
                tf.rotation_matrix_z(np.pi * np.array([0.5, 0.0, 1.5])),
                [
                    [-0.5, 0.5, 9.5],
                    [-3.5, 3.5, 5.5],
                    [-5 + c * 1 - s * 7, 3 + s * 1 + c * 7, -10],
                ],
                time_1,
                time_ref_0,
            ),
        ],
    )
    def test_addition(
        self, lcs_lhs, lcs_rhs, orientation_exp, coordinates_exp, time_exp, time_ref_exp
    ):
        self.check_coordinate_system(
            lcs_lhs + lcs_rhs,
            orientation_exp,
            coordinates_exp,
            True,
            time_exp,
            time_ref_exp,
        )


def check_coordinate_system_time(lcs: tf.LocalCoordinateSystem, expected_time):
    """Check if the time component of a LocalCoordinateSystem is as expected.

    Parameters
    ----------
    lcs :
        Local coordinate system class
    expected_time :
        Expected time

    """
    assert np.all(lcs.time == expected_time)


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
    cs_p: tf.LocalCoordinateSystem,
    orientation_expected: Union[np.ndarray, List[List[Any]], xr.DataArray],
    coordinates_expected: Union[np.ndarray, List[Any], xr.DataArray],
    positive_orientation_expected: bool = True,
    time=None,
):
    """Check the values of a coordinate system.

    Parameters
    ----------
    cs_p :
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

    """
    orientation_expected = np.array(orientation_expected)
    coordinates_expected = np.array(coordinates_expected)

    if time is not None:
        assert orientation_expected.ndim == 3 or coordinates_expected.ndim == 2
        check_coordinate_system_time(cs_p, time)

    check_coordinate_system_orientation(
        cs_p.orientation, orientation_expected, positive_orientation_expected
    )

    assert np.allclose(cs_p.coordinates.values, coordinates_expected, atol=1e-9)


def check_coordinate_systems_close(lcs_0, lcs_1):
    """Check if 2 coordinate systems are nearly identical.

    Parameters
    ----------
    lcs_0:
        First coordinate system.
    lcs_1
        Second coordinate system.

    """
    time = None
    if "time" in lcs_1.dataset:
        time = lcs_1.time
    check_coordinate_system(
        lcs_0, lcs_1.orientation.data, lcs_1.coordinates.data, True, time
    )


def test_coordinate_system_init():
    """Check the __init__ method with and without time dependency."""
    # reference data
    time_0 = TDI([1, 3, 5], "s")
    time_1 = TDI([2, 4, 6], "s")

    orientation_fix = tf.rotation_matrix_z(np.pi)
    orientation_tdp = tf.rotation_matrix_z(np.pi * np.array([0, 0.25, 0.5]))
    coordinates_fix = ut.to_float_array([3, 7, 1])
    coordinates_tdp = ut.to_float_array([[3, 7, 1], [4, -2, 8], [-5, 3, -1]])

    # numpy - no time dependency
    lcs = tf.LocalCoordinateSystem(
        orientation=orientation_fix, coordinates=coordinates_fix
    )

    check_coordinate_system(lcs, orientation_fix, coordinates_fix, True)

    # numpy - orientation time dependent
    lcs = tf.LocalCoordinateSystem(
        orientation=orientation_tdp, coordinates=coordinates_fix, time=time_0
    )

    check_coordinate_system(lcs, orientation_tdp, coordinates_fix, True, time_0)

    # numpy - coordinates time dependent
    lcs = tf.LocalCoordinateSystem(
        orientation=orientation_fix, coordinates=coordinates_tdp, time=time_0
    )

    check_coordinate_system(lcs, orientation_fix, coordinates_tdp, True, time_0)

    # numpy - coordinates and orientation time dependent - only equal times
    lcs = tf.LocalCoordinateSystem(
        orientation=orientation_tdp, coordinates=coordinates_tdp, time=time_0
    )

    check_coordinate_system(lcs, orientation_tdp, coordinates_tdp, True, time_0)

    # xarray - reference data
    xr_orientation_fix = ut.xr_3d_matrix(orientation_fix)
    xr_coordinates_fix = ut.xr_3d_vector(coordinates_fix)
    xr_orientation_tdp_0 = ut.xr_3d_matrix(orientation_tdp, time_0)
    xr_coordinates_tdp_0 = ut.xr_3d_vector(coordinates_tdp, time_0)

    # xarray - no time dependency
    lcs = tf.LocalCoordinateSystem(
        orientation=xr_orientation_fix, coordinates=xr_coordinates_fix
    )

    check_coordinate_system(lcs, orientation_fix, coordinates_fix, True)

    # xarray - orientation time dependent
    lcs = tf.LocalCoordinateSystem(
        orientation=xr_orientation_tdp_0, coordinates=xr_coordinates_fix
    )

    check_coordinate_system(lcs, orientation_tdp, coordinates_fix, True, time_0)

    # xarray - coordinates time dependent
    lcs = tf.LocalCoordinateSystem(
        orientation=xr_orientation_fix, coordinates=xr_coordinates_tdp_0
    )

    check_coordinate_system(lcs, orientation_fix, coordinates_tdp, True, time_0)

    # xarray - coordinates and orientation time dependent - equal times
    lcs = tf.LocalCoordinateSystem(
        orientation=xr_orientation_tdp_0, coordinates=xr_coordinates_tdp_0
    )

    check_coordinate_system(lcs, orientation_tdp, coordinates_tdp, True, time_0)

    # xarray - coordinates and orientation time dependent - different times
    xr_coordinates_tdp_1 = ut.xr_3d_vector(coordinates_tdp, time_1)

    lcs = tf.LocalCoordinateSystem(
        orientation=xr_orientation_tdp_0, coordinates=xr_coordinates_tdp_1
    )

    time_exp = TDI([1, 2, 3, 4, 5, 6], "s")
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

    # matrix normalization ----------------------

    # no time dependency
    orientation_exp = tf.rotation_matrix_z(np.pi / 3)
    orientation_fix_2 = deepcopy(orientation_exp)
    orientation_fix_2[:, 0] *= 10
    orientation_fix_2[:, 1] *= 3
    orientation_fix_2[:, 2] *= 4

    lcs = tf.LocalCoordinateSystem(
        orientation=orientation_fix_2, coordinates=coordinates_fix
    )

    check_coordinate_system(lcs, orientation_exp, coordinates_fix, True)

    # time dependent
    orientation_exp = tf.rotation_matrix_z(np.pi / 3 * ut.to_float_array([1, 2, 4]))
    orientation_tdp_2 = deepcopy(orientation_exp)
    orientation_tdp_2[:, :, 0] *= 10
    orientation_tdp_2[:, :, 1] *= 3
    orientation_tdp_2[:, :, 2] *= 4

    lcs = tf.LocalCoordinateSystem(
        orientation=orientation_tdp_2, coordinates=coordinates_fix, time=time_0
    )

    check_coordinate_system(lcs, orientation_exp, coordinates_fix, True, time_0)

    # exceptions --------------------------------
    # invalid inputs
    with pytest.raises(Exception):
        tf.LocalCoordinateSystem(
            orientation="wrong", coordinates=coordinates_fix, time=time_0
        )
    with pytest.raises(Exception):
        tf.LocalCoordinateSystem(
            orientation=orientation_fix, coordinates="wrong", time=time_0
        )
    with pytest.raises(Exception):
        tf.LocalCoordinateSystem(
            orientation=orientation_fix, coordinates=coordinates_fix, time="wrong"
        )

    # wrong xarray format
    # TODO: implement


def test_coordinate_system_factories_no_time_dependency():
    """Test construction of coordinate system class without time dependencies.

    Create multiple coordinate systems with all provided methods and check
    if they are constructed correctly.

    """
    # alias name for class - name is too long :)
    lcs = tf.LocalCoordinateSystem

    # setup -----------------------------------------------
    angle_x = np.pi / 3
    angle_y = np.pi / 4
    angle_z = np.pi / 5
    coordinates = [4, -2, 6]
    orientation_pos = rotated_positive_orthogonal_basis(angle_x, angle_y, angle_z)

    x = orientation_pos[:, 0]
    y = orientation_pos[:, 1]
    z = orientation_pos[:, 2]

    orientation_neg = np.transpose([x, y, -z])

    # construction with orientation -----------------------

    cs_orientation_pos = lcs.from_orientation(orientation_pos, coordinates)
    cs_orientation_neg = lcs.from_orientation(orientation_neg, coordinates)

    check_coordinate_system(cs_orientation_pos, orientation_pos, coordinates, True)
    check_coordinate_system(cs_orientation_neg, orientation_neg, coordinates, False)

    # construction with euler -----------------------------

    angles = [angle_x, angle_y, angle_z]
    cs_euler_pos = lcs.from_euler("xyz", angles, False, coordinates)
    check_coordinate_system(cs_euler_pos, orientation_pos, coordinates, True)

    # construction with x,y,z-vectors ---------------------

    cs_xyz_pos = lcs.from_xyz(x, y, z, coordinates)
    cs_xyz_neg = lcs.from_xyz(x, y, -z, coordinates)

    check_coordinate_system(cs_xyz_pos, orientation_pos, coordinates, True)
    check_coordinate_system(cs_xyz_neg, orientation_neg, coordinates, False)

    # construction with x,y-vectors and orientation -------
    cs_xyo_pos = lcs.from_xy_and_orientation(x, y, True, coordinates)
    cs_xyo_neg = lcs.from_xy_and_orientation(x, y, False, coordinates)

    check_coordinate_system(cs_xyo_pos, orientation_pos, coordinates, True)
    check_coordinate_system(cs_xyo_neg, orientation_neg, coordinates, False)

    # construction with y,z-vectors and orientation -------
    cs_yzo_pos = lcs.from_yz_and_orientation(y, z, True, coordinates)
    cs_yzo_neg = lcs.from_yz_and_orientation(y, -z, False, coordinates)

    check_coordinate_system(cs_yzo_pos, orientation_pos, coordinates, True)
    check_coordinate_system(cs_yzo_neg, orientation_neg, coordinates, False)

    # construction with x,z-vectors and orientation -------
    cs_xzo_pos = lcs.from_xz_and_orientation(x, z, True, coordinates)
    cs_xzo_neg = lcs.from_xz_and_orientation(x, -z, False, coordinates)

    check_coordinate_system(cs_xzo_pos, orientation_pos, coordinates, True)
    check_coordinate_system(cs_xzo_neg, orientation_neg, coordinates, False)

    # test integers as inputs -----------------------------
    x_i = [1, 1, 0]
    y_i = [-1, 1, 0]
    z_i = [0, 0, 1]

    lcs.from_xyz(x_i, y_i, z_i, coordinates)
    lcs.from_xy_and_orientation(x_i, y_i)
    lcs.from_yz_and_orientation(y_i, z_i)
    lcs.from_xz_and_orientation(z_i, x_i)

    # check exceptions ------------------------------------
    with pytest.raises(Exception):
        lcs([x, y, [0, 0, 1]])


def test_coordinate_system_factories_time_dependent():
    """Test construction of coordinate system class with time dependencies.

    Create multiple coordinate systems with all provided methods and check
    if they are constructed correctly.

    """
    # alias name for class - name is too long :)
    lcs = tf.LocalCoordinateSystem

    angles_x = np.array([0.5, 1, 2, 2.5]) * np.pi / 2
    angles_y = np.array([1.5, 0, 1, 0.5]) * np.pi / 2
    angles = np.array([[*angles_y], [*angles_x]]).transpose()

    rot_mat_x = tf.rotation_matrix_x(angles_x)
    rot_mat_y = tf.rotation_matrix_y(angles_y)

    time = TDI([0, 6, 12, 18], "H")
    orientations = np.matmul(rot_mat_x, rot_mat_y)
    coords = [[1, 0, 0], [-1, 0, 2], [3, 5, 7], [-4, -5, -6]]

    vec_x = orientations[:, :, 0]
    vec_y = orientations[:, :, 1]
    vec_z = orientations[:, :, 2]

    # construction with orientation -----------------------

    cs_orientation_oc = lcs.from_orientation(orientations, coords, time)
    check_coordinate_system(cs_orientation_oc, orientations, coords, time=time)

    cs_orientation_c = lcs.from_orientation(orientations[0], coords, time)
    check_coordinate_system(cs_orientation_c, orientations[0], coords, time=time)

    cs_orientation_o = lcs.from_orientation(orientations, coords[0], time)
    check_coordinate_system(cs_orientation_o, orientations, coords[0], time=time)

    # construction with euler -----------------------------

    cs_euler_oc = lcs.from_euler("yx", angles, False, coords, time)
    check_coordinate_system(cs_euler_oc, orientations, coords, time=time)

    cs_euler_c = lcs.from_euler("yx", angles[0], False, coords, time)
    check_coordinate_system(cs_euler_c, orientations[0], coords, time=time)

    cs_euler_o = lcs.from_euler("yx", angles, False, coords[0], time)
    check_coordinate_system(cs_euler_o, orientations, coords[0], time=time)

    # construction with x,y,z-vectors ---------------------

    cs_xyz_oc = lcs.from_xyz(vec_x, vec_y, vec_z, coords, time)
    check_coordinate_system(cs_xyz_oc, orientations, coords, time=time)

    cs_xyz_c = lcs.from_xyz(vec_x[0], vec_y[0], vec_z[0], coords, time)
    check_coordinate_system(cs_xyz_c, orientations[0], coords, time=time)

    cs_xyz_o = lcs.from_xyz(vec_x, vec_y, vec_z, coords[0], time)
    check_coordinate_system(cs_xyz_o, orientations, coords[0], time=time)

    # construction with x,y-vectors and orientation -------

    cs_xyo_oc = lcs.from_xy_and_orientation(vec_x, vec_y, True, coords, time)
    check_coordinate_system(cs_xyo_oc, orientations, coords, True, time=time)

    cs_xyo_c = lcs.from_xy_and_orientation(vec_x[0], vec_y[0], True, coords, time)
    check_coordinate_system(cs_xyo_c, orientations[0], coords, True, time=time)

    cs_xyo_o = lcs.from_xy_and_orientation(vec_x, vec_y, True, coords[0], time)
    check_coordinate_system(cs_xyo_o, orientations, coords[0], True, time=time)

    # construction with y,z-vectors and orientation -------

    cs_yzo_oc = lcs.from_yz_and_orientation(vec_y, vec_z, True, coords, time)
    check_coordinate_system(cs_yzo_oc, orientations, coords, True, time=time)

    cs_yzo_c = lcs.from_yz_and_orientation(vec_y[0], vec_z[0], True, coords, time)
    check_coordinate_system(cs_yzo_c, orientations[0], coords, True, time=time)

    cs_yzo_o = lcs.from_yz_and_orientation(vec_y, vec_z, True, coords[0], time)
    check_coordinate_system(cs_yzo_o, orientations, coords[0], True, time=time)

    # construction with x,z-vectors and orientation -------

    cs_xzo_oc = lcs.from_xz_and_orientation(vec_x, vec_z, True, coords, time)
    check_coordinate_system(cs_xzo_oc, orientations, coords, True, time=time)

    cs_xzo_c = lcs.from_xz_and_orientation(vec_x[0], vec_z[0], True, coords, time)
    check_coordinate_system(cs_xzo_c, orientations[0], coords, True, time=time)

    cs_xzo_o = lcs.from_xz_and_orientation(vec_x, vec_z, True, coords[0], time)
    check_coordinate_system(cs_xzo_o, orientations, coords[0], True, time=time)


def test_coordinate_system_addition_and_subtraction():
    """Test the + and - operator of the coordinate system class.

    Creates some coordinate systems and uses the operators on them. Results
    are compared to expected values. The naming pattern 'X_in_Y' is used for the
    coordinate systems to keep track of the supposed operation results.

    """
    546
    # reference data ----------------------------
    time_0 = TDI([1, 3, 5], "s")
    time_1 = TDI([2, 4, 6], "s")

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
        orientation=orientation_fix_0, coordinates=coordinates_fix_0
    )
    lcs1_in_lcs_0 = tf.LocalCoordinateSystem(
        orientation=orientation_fix_1, coordinates=coordinates_fix_1
    )

    lcs1_in_base = lcs1_in_lcs_0 + lcs0_in_base
    lcs1_in_lcs_0_calc = lcs1_in_base - lcs0_in_base

    orientation_exp = [[0, -1, 0], [0, 0, 1], [-1, 0, 0]]
    coordinates_exp = [-1, 8, 3]

    check_coordinate_system(lcs1_in_base, orientation_exp, coordinates_exp, True)
    check_coordinate_system(
        lcs1_in_lcs_0_calc, lcs1_in_lcs_0.orientation, lcs1_in_lcs_0.coordinates, True
    )

    # orientation of left cs time dependent -----
    lcs0_in_base = tf.LocalCoordinateSystem(
        orientation=orientation_fix_0, coordinates=coordinates_fix_0
    )
    lcs1_in_lcs_0 = tf.LocalCoordinateSystem(
        orientation=orientation_tdp_0, coordinates=coordinates_fix_1, time=time_0
    )

    lcs1_in_base = lcs1_in_lcs_0 + lcs0_in_base
    lcs1_in_lcs_0_calc = lcs1_in_base - lcs0_in_base

    orientation_exp = tf.rotation_matrix_z(np.pi * np.array([0.5, 1, 1.5]))
    coordinates_exp = [[-1, 8, 3], [-1, 8, 3], [-1, 8, 3]]

    check_coordinate_system(
        lcs1_in_base, orientation_exp, coordinates_exp, True, time_0
    )
    check_coordinate_system(
        lcs1_in_lcs_0_calc,
        lcs1_in_lcs_0.orientation,
        lcs1_in_lcs_0.coordinates,
        True,
        time_0,
    )

    # coordinates of left cs time dependent -----
    lcs0_in_base = tf.LocalCoordinateSystem(
        orientation=orientation_fix_0, coordinates=coordinates_fix_0
    )
    lcs1_in_lcs_0 = tf.LocalCoordinateSystem(
        orientation=orientation_fix_1, coordinates=coordinates_tdp_0, time=time_0
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
        lcs1_in_lcs_0_calc,
        lcs1_in_lcs_0.orientation,
        lcs1_in_lcs_0.coordinates,
        True,
        time_0,
    )

    # both fully time dependent, equal times ----
    lcs0_in_base = tf.LocalCoordinateSystem(
        orientation=orientation_tdp_0, coordinates=coordinates_tdp_0, time=time_0
    )
    lcs1_in_lcs_0 = tf.LocalCoordinateSystem(
        orientation=orientation_tdp_1, coordinates=coordinates_tdp_1, time=time_0
    )

    lcs1_in_base = lcs1_in_lcs_0 + lcs0_in_base
    lcs1_in_lcs_0_calc = lcs1_in_base - lcs0_in_base

    orientation_exp = tf.rotation_matrix_z(np.pi * np.array([1, 0.5, 1]))
    coordinates_exp = [[7, 9, 6], [7, 1, 10], [-6, -4, -10]]

    check_coordinate_system(
        lcs1_in_base, orientation_exp, coordinates_exp, True, time_0
    )
    check_coordinate_system(
        lcs1_in_lcs_0_calc,
        lcs1_in_lcs_0.orientation,
        lcs1_in_lcs_0.coordinates,
        True,
        time_0,
    )

    # both fully time dependent, different times - addition only
    # INFO:
    # The subtraction can not be tested as in the previous tests by subtracting the
    # added coordinate system and comparing the result to the initial one. The problem
    # is, that the necessary interpolated values depend on the reference coordinate
    # system, the interpolation is performed in. Since the reference systems differ
    # between the addition and the subsequent subtraction, the result can not be
    # compared to the initial coordinate system.

    lcs0_in_base = tf.LocalCoordinateSystem(
        orientation=orientation_tdp_2, coordinates=coordinates_tdp_0, time=time_0
    )
    lcs1_in_lcs_0 = tf.LocalCoordinateSystem(
        orientation=orientation_tdp_3, coordinates=coordinates_tdp_1, time=time_1
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
        orientation=orientation_tdp_4, coordinates=coordinates_tdp_0, time=time_0
    )
    lcs1_in_base = tf.LocalCoordinateSystem(
        orientation=orientation_tdp_3, coordinates=coordinates_tdp_1, time=time_1
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
    lcs0 = tf.LocalCoordinateSystem(
        orientation=orientation_fix_0, coordinates=coordinates_fix_0
    )
    lcs1 = tf.LocalCoordinateSystem(
        orientation=orientation_tdp_0, coordinates=coordinates_fix_1, time=time_0
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
    lcs0 = tf.LocalCoordinateSystem(
        orientation=orientation_fix_0, coordinates=coordinates_fix_0
    )
    lcs1 = tf.LocalCoordinateSystem(
        orientation=orientation_fix_0, coordinates=coordinates_tdp_0, time=time_0
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
    lcs0 = tf.LocalCoordinateSystem(
        orientation=orientation_fix_0, coordinates=coordinates_fix_0
    )
    lcs1 = tf.LocalCoordinateSystem(
        orientation=orientation_tdp_0, coordinates=coordinates_tdp_0, time=time_0
    )

    lcs_add = lcs0 + lcs1
    lcs_sub = lcs0 - lcs1

    orientation_add_exp = tf.rotation_matrix_z(np.pi * np.array([0.5, 1, 1.5]))
    coordinates_add_exp = [[6, 14, 2], [-3, 1, 9], [-8, -4, 0]]
    orientation_sub_exp = tf.rotation_matrix_z(np.pi * np.array([0.5, 0, 1.5]))
    coordinates_sub_exp = [[0, 0, 0], [9, 1, -7], [-8, -4, 2]]
    check_coordinate_system(
        lcs_add, orientation_add_exp, coordinates_add_exp, True, time_0
    )
    check_coordinate_system(
        lcs_sub, orientation_sub_exp, coordinates_sub_exp, True, time_0
    )


def test_coordinate_system_invert():
    """Test the invert function.

    The test creates a coordinate system, inverts it and checks the result against the
    expected value. Afterwards, the resulting system is inverted again. This operation
    must yield the original system.

    """
    # fix ---------------------------------------
    lcs0_in_lcs1 = tf.LocalCoordinateSystem.from_xy_and_orientation(
        [1, 1, 0], [-1, 1, 0], coordinates=[2, 0, 2]
    )
    lcs1_in_lcs0 = lcs0_in_lcs1.invert()

    exp_orientation = tf.rotation_matrix_z(-np.pi / 4)
    exp_coordinates = [-np.sqrt(2), np.sqrt(2), -2]

    check_coordinate_system(lcs1_in_lcs0, exp_orientation, exp_coordinates, True)

    lcs0_in_lcs1_2 = lcs1_in_lcs0.invert()

    check_coordinate_system(
        lcs0_in_lcs1_2, lcs0_in_lcs1.orientation, lcs0_in_lcs1.coordinates, True
    )

    # time dependent ----------------------------
    time = TDI([1, 2, 3, 4], "s")
    orientation = tf.rotation_matrix_z(np.array([0, 0.5, 1, 0.5]) * np.pi)
    coordinates = np.array([[2, 8, 7], [4, 9, 2], [0, 2, 1], [3, 1, 2]])

    lcs0_in_lcs1 = tf.LocalCoordinateSystem(
        orientation=orientation, coordinates=coordinates, time=time
    )

    lcs1_in_lcs0 = lcs0_in_lcs1.invert()
    orientation_exp = tf.rotation_matrix_z(np.array([0, 1.5, 1, 1.5]) * np.pi)
    coordinates_exp = np.array([[-2, -8, -7], [-9, 4, -2], [0, 2, -1], [-1, 3, -2]])

    check_coordinate_system(lcs1_in_lcs0, orientation_exp, coordinates_exp, True, time)

    lcs0_in_lcs1_2 = lcs1_in_lcs0.invert()

    check_coordinate_system(
        lcs0_in_lcs1_2, lcs0_in_lcs1.orientation, lcs0_in_lcs1.coordinates, True, time
    )


def coordinate_system_time_interpolation_test_case(
    lcs: tf.LocalCoordinateSystem,
    time_interp: pd.DatetimeIndex,
    orientation_exp: np.ndarray,
    coordinates_exp: np.ndarray,
):
    """Test the time interpolation methods of the LocalCoordinateSystem class.

    Parameters
    ----------
    lcs :
        Time dependent Local coordinate system
    time_interp :
        Target times for interpolation
    orientation_exp :
        Expected orientations
    coordinates_exp :
        Expected coordinates

    """
    lcs_interp = lcs.interp_time(time_interp)
    check_coordinate_system(
        lcs_interp, orientation_exp, coordinates_exp, True, time_interp
    )

    # test lcs input syntax
    lcs_interp_like = lcs.interp_time(lcs_interp)
    check_coordinate_system(
        lcs_interp_like, orientation_exp, coordinates_exp, True, time_interp
    )


def test_coordinate_system_time_interpolation():
    """Test the local coordinate systems interp_time and interp_like functions."""
    time_0 = TDI([10, 14, 18, 22], "D")
    orientation = tf.rotation_matrix_z(np.array([0, 0.5, 1, 0.5]) * np.pi)
    coordinates = np.array([[2, 8, 7], [4, 9, 2], [0, 2, 1], [3, 1, 2]])

    lcs = tf.LocalCoordinateSystem(
        orientation=orientation, coordinates=coordinates, time=time_0
    )

    # test xr_interp_orientation_in_time for single time point interpolation
    orientation = ut.xr_interp_orientation_in_time(
        lcs.orientation.isel({"time": [1]}), time_0
    )
    assert np.allclose(orientation, orientation[1, :, :])

    # exceptions --------------------------------
    # wrong parameter type
    with pytest.raises(Exception):
        lcs.interp_time("wrong")
    # no time component
    with pytest.raises(Exception):
        lcs.interp_time(tf.LocalCoordinateSystem())


# --------------------------------------------------------------------------------------
# Test CoordinateSystemManager
# --------------------------------------------------------------------------------------

# todo: Refactor old tests


class TestCoordinateSystemManager:
    """Test the CoordinateSystemManager class."""

    CSM = tf.CoordinateSystemManager
    LCS = tf.LocalCoordinateSystem

    @pytest.fixture
    def csm_fix(self):
        """Create default coordinate system fixture."""
        csm_default = self.CSM("root")
        lcs_1 = self.LCS(coordinates=[0, 1, 2])
        lcs_2 = self.LCS(coordinates=[0, -1, -2])
        lcs_3 = self.LCS(tf.rotation_matrix_y(0), [-1, -2, -3])
        lcs_4 = self.LCS(tf.rotation_matrix_y(np.pi / 2), [1, 2, 3])
        lcs_5 = self.LCS(tf.rotation_matrix_y(np.pi * 3 / 2), [2, 3, 1])
        csm_default.add_cs("lcs1", "root", lcs_1)
        csm_default.add_cs("lcs2", "root", lcs_2)
        csm_default.add_cs("lcs3", "lcs1", lcs_3)
        csm_default.add_cs("lcs4", "lcs1", lcs_4)
        csm_default.add_cs("lcs5", "lcs2", lcs_5)

        return csm_default

    @pytest.fixture()
    def list_of_csm_and_lcs_instances(self):
        """Get a list of LCS and CSM instances."""
        lcs = [self.LCS(coordinates=[i, 0, 0]) for i in range(11)]

        csm_0 = self.CSM("lcs0", "csm0")
        csm_0.add_cs("lcs1", "lcs0", lcs[1])
        csm_0.add_cs("lcs2", "lcs0", lcs[2])
        csm_0.add_cs("lcs3", "lcs2", lcs[3])

        csm_1 = self.CSM("lcs0", "csm1")
        csm_1.add_cs("lcs4", "lcs0", lcs[4])

        csm_2 = self.CSM("lcs5", "csm2")
        csm_2.add_cs("lcs3", "lcs5", lcs[5], lsc_child_in_parent=False)
        csm_2.add_cs("lcs6", "lcs5", lcs[6])

        csm_3 = self.CSM("lcs6", "csm3")
        csm_3.add_cs("lcs7", "lcs6", lcs[7])
        csm_3.add_cs("lcs8", "lcs6", lcs[8])

        csm_4 = self.CSM("lcs9", "csm4")
        csm_4.add_cs("lcs3", "lcs9", lcs[9], lsc_child_in_parent=False)

        csm_5 = self.CSM("lcs7", "csm5")
        csm_5.add_cs("lcs10", "lcs7", lcs[10])

        csm = [csm_0, csm_1, csm_2, csm_3, csm_4, csm_5]
        return [csm, lcs]

    # test_add_coordinate_system -------------------------------------------------------

    # todo
    #  add time dependent systems. The problem is, that currently something messes
    #  up the comparison. The commented version of lcs_2 somehow switches the order of
    #  how 2 coordinates are stored in the Dataset. This lets the coordinate comparison
    #  fail.
    csm_acs = CSM("root")
    time = pd.DatetimeIndex(["2000-01-01", "2000-01-04"])
    lcs_1_acs = LCS(coordinates=[0, 1, 2])
    # lcs_2_acs = LCS(coordinates=[[0, -1, -2], [8, 2, 7]], time=time)
    lcs_2_acs = LCS(coordinates=[0, -1, -2])
    lcs_3_acs = LCS(tf.rotation_matrix_y(0), [-1, -2, -3])
    lcs_4_acs = LCS(tf.rotation_matrix_y(np.pi / 2), [1, 2, 3])
    lcs_5_acs = LCS(tf.rotation_matrix_y(np.pi * 3 / 2), [2, 3, 1])

    @pytest.mark.parametrize(
        "name , parent, lcs, child_in_parent, exp_num_cs",
        [
            ("lcs1", "root", lcs_1_acs, True, 2),
            ("lcs2", "root", lcs_2_acs, False, 3),
            ("lcs3", "lcs2", lcs_4_acs, True, 4),
            ("lcs3", "lcs2", lcs_3_acs, True, 4),
            ("lcs2", "lcs3", lcs_3_acs, False, 4),
            ("lcs2", "lcs3", lcs_3_acs, True, 4),
            ("lcs4", "lcs2", lcs_1_acs, True, 5),
            ("lcs4", "lcs2", lcs_4_acs, True, 5),
            ("lcs5", "lcs1", lcs_5_acs, True, 6),
        ],
    )
    def test_add_coordinate_system(
        self, name, parent, lcs, child_in_parent, exp_num_cs
    ):
        """Test the 'add_cs' function."""
        csm = self.csm_acs
        csm.add_cs(name, parent, lcs, child_in_parent)

        assert csm.number_of_coordinate_systems == exp_num_cs
        if child_in_parent:
            assert csm.get_local_coordinate_system(name, parent) == lcs
            assert csm.get_local_coordinate_system(parent, name) == lcs.invert()
        else:
            assert csm.get_local_coordinate_system(name, parent) == lcs.invert()
            assert csm.get_local_coordinate_system(parent, name) == lcs

    # test_add_coordinate_system_exceptions --------------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        "name, parent_name, lcs, exception_type, test_name",
        [
            ("lcs", "r00t", LCS(), ValueError, "# invalid parent system"),
            ("lcs4", "root", LCS(), ValueError, "# can't update - no neighbors"),
            ("lcs", LCS(), LCS(), TypeError, "# invalid parent system name type"),
            (LCS(), "root", LCS(), TypeError, "# invalid system name type"),
            ("new_lcs", "root", "a string", TypeError, "# invalid system type"),
        ],
        ids=get_test_name,
    )
    def test_add_coordinate_system_exceptions(
        csm_fix, name, parent_name, lcs, exception_type, test_name
    ):
        """Test the exceptions of the 'add_cs' method."""
        with pytest.raises(exception_type):
            csm_fix.add_cs(name, parent_name, lcs)

    # test num_neighbors ---------------------------------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        "name, exp_num_neighbors",
        [("root", 2), ("lcs1", 3), ("lcs2", 2), ("lcs3", 1), ("lcs4", 1), ("lcs5", 1)],
    )
    def test_num_neighbors(csm_fix, name, exp_num_neighbors):
        """Test the num_neighbors function."""
        assert csm_fix.number_of_neighbors(name) == exp_num_neighbors

    # test is_neighbor_of --------------------------------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        "name1, exp_result",
        [
            ("root", [False, True, True, False, False, False]),
            ("lcs1", [True, False, False, True, True, False]),
            ("lcs2", [True, False, False, False, False, True]),
            ("lcs3", [False, True, False, False, False, False]),
            ("lcs4", [False, True, False, False, False, False]),
            ("lcs5", [False, False, True, False, False, False]),
        ],
    )
    @pytest.mark.parametrize(
        "name2, result_idx",
        [("root", 0), ("lcs1", 1), ("lcs2", 2), ("lcs3", 3), ("lcs4", 4), ("lcs5", 5)],
    )
    def test_is_neighbor_of(csm_fix, name1, name2, result_idx, exp_result):
        """Test the is_neighbor_of function."""
        assert csm_fix.is_neighbor_of(name1, name2) is exp_result[result_idx]

    # test_get_child_system_names ------------------------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        "cs_name, neighbors_only, result_exp",
        [
            ("root", True, ["lcs1", "lcs2"]),
            ("lcs1", True, ["lcs3", "lcs4"]),
            ("lcs2", True, ["lcs5"]),
            ("lcs3", True, []),
            ("lcs4", True, []),
            ("lcs5", True, []),
            ("root", False, ["lcs1", "lcs2", "lcs3", "lcs4", "lcs5"]),
            ("lcs1", False, ["lcs3", "lcs4"]),
            ("lcs2", False, ["lcs5"]),
            ("lcs3", False, []),
            ("lcs4", False, []),
            ("lcs5", False, []),
        ],
    )
    def test_get_child_system_names(csm_fix, cs_name, neighbors_only, result_exp):
        """Test the get_child_system_names function."""
        result = csm_fix.get_child_system_names(cs_name, neighbors_only)

        # check -------------------------------------------
        assert len(result) == len(result_exp)
        for name in result_exp:
            assert name in result

    # test_delete_coordinate_system ----------------------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        "lcs_del, delete_children, num_cs_exp, exp_children_deleted",
        [
            ("lcs1", True, 3, ["lcs3", "lcs4"]),
            ("lcs2", True, 4, ["lcs5"]),
            ("lcs3", True, 5, []),
            ("lcs4", True, 5, []),
            ("lcs5", True, 5, []),
            ("lcs3", False, 5, []),
            ("lcs4", False, 5, []),
            ("lcs5", False, 5, []),
            ("not included", False, 6, []),
            ("not included", True, 6, []),
        ],
    )
    def test_delete_coordinate_system(
        csm_fix, lcs_del, delete_children, exp_children_deleted, num_cs_exp
    ):
        """Test the delete function of the CSM."""
        # setup
        removed_lcs_exp = [lcs_del] + exp_children_deleted

        # delete coordinate system
        csm_fix.delete_cs(lcs_del, delete_children)

        # check
        edges = csm_fix.graph.edges

        assert csm_fix.number_of_coordinate_systems == num_cs_exp
        for lcs in removed_lcs_exp:
            assert not csm_fix.has_coordinate_system(lcs)
            for edge in edges:
                assert lcs not in edge

    # test_delete_coordinate_system_exceptions -----------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        "name, delete_children, exception_type, test_name",
        [
            ("root", True, ValueError, "# root system can't be deleted #1"),
            ("root", False, ValueError, "# root system can't be deleted #2"),
            ("lcs1", False, Exception, "# system has children"),
        ],
        ids=get_test_name,
    )
    def test_delete_coordinate_system_exceptions(
        csm_fix, name, delete_children, exception_type, test_name
    ):
        """Test the exceptions of the 'add_cs' method."""
        with pytest.raises(exception_type):
            csm_fix.delete_cs(name, delete_children)

    # test_comparison ------------------------------------------------------------------

    csm_comp_0 = CSM("root", "name")
    csm_comp_0.add_cs("lcs_0", "root", LCS(coordinates=[0, 1, 2]))
    csm_comp_0.add_cs("lcs_1", "root", LCS(coordinates=[0, -1, -2]))
    # different LCS
    csm_comp_1 = CSM("root", "name")
    csm_comp_1.add_cs("lcs_0", "root", LCS(coordinates=[1, 1, 2]))
    csm_comp_1.add_cs("lcs_1", "root", LCS(coordinates=[0, -1, -2]))
    # different nodes
    csm_comp_2 = CSM("root", "name")
    csm_comp_2.add_cs("lcs_0", "root", LCS(coordinates=[0, 1, 2]))
    csm_comp_2.add_cs("lcs_2", "root", LCS(coordinates=[0, -1, -2]))
    # different edges
    csm_comp_3 = CSM("root", "name")
    csm_comp_3.add_cs("lcs_0", "root", LCS(coordinates=[0, 1, 2]))
    csm_comp_3.add_cs("lcs_1", "lcs_0", LCS(coordinates=[0, -1, -2]))
    # merged systems
    csm_merge_0 = CSM("root", "sub 1")
    csm_merge_0.add_cs("lcs_0", "root", LCS(coordinates=[0, 1, 2]))
    csm_merge_1 = CSM("lcs_0", "sub 2")
    csm_merge_1.add_cs("lcs_2", "lcs_0", LCS(coordinates=[0, 1, 2]))
    csm_comp_4 = CSM("root", "name")
    csm_comp_4.add_cs("lcs_1", "root", LCS(coordinates=[0, -1, -2]))
    csm_comp_4.merge(csm_merge_0)
    # serial merge
    csm_comp_5 = CSM("root", "name")
    csm_comp_5.add_cs("lcs_1", "root", LCS(coordinates=[0, -1, -2]))
    csm_comp_5.merge(csm_merge_0)
    csm_comp_5.merge(csm_merge_1)
    # nested merge
    csm_comp_6 = CSM("root", "name")
    csm_comp_6.add_cs("lcs_1", "root", LCS(coordinates=[0, -1, -2]))
    csm_nest_0 = deepcopy(csm_merge_0)
    csm_nest_0.merge(csm_merge_1)
    csm_comp_6.merge(csm_nest_0)

    @pytest.mark.parametrize(
        "csm, other, result_exp",
        [
            (CSM("root", "name"), CSM("root", "name"), True),
            (CSM("root", "name"), CSM("root", "wrong name"), False),
            (CSM("root", "name"), CSM("boot", "name"), False),
            (CSM("root", "name"), "a string", False),
            (csm_comp_0, deepcopy(csm_comp_0), True),
            (csm_comp_0, CSM("root", "name"), False),
            (csm_comp_0, csm_comp_1, False),
            (csm_comp_0, csm_comp_2, False),
            (csm_comp_0, csm_comp_3, False),
            (csm_comp_4, csm_comp_4, True),
            (csm_comp_4, csm_comp_5, False),
            (csm_comp_4, csm_comp_6, False),
            (csm_comp_5, csm_comp_5, True),
            (csm_comp_5, csm_comp_6, False),
            (csm_comp_6, csm_comp_6, True),
        ],
    )
    def test_comparison(self, csm, other, result_exp):
        """Test the comparison of 2 'CoordinateSystemManager' instances."""
        assert isinstance(csm, self.CSM), "csm must be a CoordinateSystemManager"
        assert (csm == other) is result_exp
        assert (csm != other) is not result_exp

    # test_merge -----------------------------------------------------------------------

    @staticmethod
    @pytest.mark.parametrize("nested", [(True,), (False,)])
    def test_merge(list_of_csm_and_lcs_instances, nested):
        """Test the merge function."""
        # setup -------------------------------------------
        csm = list_of_csm_and_lcs_instances[0]
        lcs = list_of_csm_and_lcs_instances[1]

        # merge -------------------------------------------
        csm_mg = deepcopy(csm[0])

        if nested:
            csm_n3 = deepcopy(csm[3])
            csm_n3.merge(csm[5])
            csm_n2 = deepcopy(csm[2])
            csm_n2.merge(csm_n3)
            csm_mg.merge(csm[1])
            csm_mg.merge(csm[4])
            csm_mg.merge(csm_n2)
        else:
            csm_mg.merge(csm[1])
            csm_mg.merge(csm[2])
            csm_mg.merge(csm[3])
            csm_mg.merge(csm[4])
            csm_mg.merge(csm[5])

        # check merge results -----------------------------
        csm_0_systems = csm_mg.get_coordinate_system_names()
        assert np.all([f"lcs{i}" in csm_0_systems for i in range(len(lcs))])

        for i, cur_lcs in enumerate(lcs):
            child = f"lcs{i}"
            parent = csm_mg.get_parent_system_name(child)
            if i == 0:
                assert parent is None
                continue
            assert csm_mg.get_local_coordinate_system(child, parent) == cur_lcs
            assert csm_mg.get_local_coordinate_system(parent, child) == cur_lcs.invert()

    # test get_subsystems_merged_serially ----------------------------------------------

    @staticmethod
    def test_get_subsystems_merged_serially(list_of_csm_and_lcs_instances):
        """Test the get_subsystem method.

        In this test case, all sub systems are merged into the same target system.
        """
        # setup -------------------------------------------
        csm = list_of_csm_and_lcs_instances[0]

        csm[0].merge(csm[1])
        csm[0].merge(csm[2])
        csm[0].merge(csm[3])
        csm[0].merge(csm[4])
        csm[0].merge(csm[5])

        # get subsystems ----------------------------------
        subs = csm[0].get_sub_systems()

        # checks ------------------------------------------
        assert len(subs) == 5

        assert subs[0] == csm[1]
        assert subs[1] == csm[2]
        assert subs[2] == csm[3]
        assert subs[3] == csm[4]
        assert subs[4] == csm[5]

    # test get_subsystems_merged_nested ----------------------------------------------

    @staticmethod
    def test_get_subsystems_merged_nested(list_of_csm_and_lcs_instances):
        """Test the get_subsystem method.

        In this test case, several systems are merged together before they are merged
        to the target system. This creates a nested subsystem structure.
        """
        # setup -------------------------------------------
        csm = list_of_csm_and_lcs_instances[0]

        csm_n3 = deepcopy(csm[3])
        csm_n3.merge(csm[5])

        csm_n2 = deepcopy(csm[2])
        csm_n2.merge(csm_n3)

        csm_mg = deepcopy(csm[0])
        csm_mg.merge(csm[1])
        csm_mg.merge(csm[4])
        csm_mg.merge(csm_n2)

        # get sub systems ---------------------------------
        subs = csm_mg.get_sub_systems()

        # checks ------------------------------------------
        assert len(subs) == 3

        assert subs[0] == csm[1]
        assert subs[1] == csm[4]
        assert subs[2] == csm_n2

        # get sub sub system ------------------------------
        sub_subs = subs[2].get_sub_systems()

        # check -------------------------------------------
        assert len(sub_subs) == 1

        assert sub_subs[0] == csm_n3

        # get sub sub sub systems -------------------------
        sub_sub_subs = sub_subs[0].get_sub_systems()

        # check -------------------------------------------
        assert len(sub_sub_subs) == 1

        assert sub_sub_subs[0] == csm[5]

    # test_remove_subsystems -----------------------------------------------------------

    @staticmethod
    @pytest.mark.parametrize("nested", [(True,), (False,)])
    def test_remove_subsystems(list_of_csm_and_lcs_instances, nested):
        """Test the remove_subsystem function."""
        # setup -------------------------------------------
        csm = list_of_csm_and_lcs_instances[0]

        csm_mg = deepcopy(csm[0])

        if nested:
            csm_n3 = deepcopy(csm[3])
            csm_n3.merge(csm[5])
            csm_n2 = deepcopy(csm[2])
            csm_n2.merge(csm_n3)
            csm_mg.merge(csm[1])
            csm_mg.merge(csm[4])
            csm_mg.merge(csm_n2)
        else:
            csm_mg.merge(csm[1])
            csm_mg.merge(csm[2])
            csm_mg.merge(csm[3])
            csm_mg.merge(csm[4])
            csm_mg.merge(csm[5])

        # remove subsystems -------------------------------
        csm_mg.remove_subsystems()

        # check -------------------------------------------
        assert csm_mg == csm[0]

    # test_unmerge_merged_serially -----------------------------------------------------

    @pytest.mark.parametrize(
        "additional_cs",
        [
            ({}),
            ({"lcs0": 0}),
            ({"lcs1": 0}),
            ({"lcs2": 0}),
            ({"lcs3": 0}),
            ({"lcs4": 1}),
            ({"lcs5": 2}),
            ({"lcs6": 2}),
            ({"lcs7": 3}),
            ({"lcs8": 3}),
            ({"lcs9": 4}),
            ({"lcs10": 5}),
            ({"lcs2": 0, "lcs5": 2, "lcs7": 3, "lcs8": 3}),
            ({"lcs0": 0, "lcs3": 0, "lcs4": 1, "lcs6": 2, "lcs10": 5}),
        ],
    )
    def test_unmerge_merged_serially(
        self, list_of_csm_and_lcs_instances, additional_cs
    ):
        """Test the CSM unmerge function.

        In this test case, all sub systems are merged into the same target system.
        """
        # setup -------------------------------------------
        csm = deepcopy(list_of_csm_and_lcs_instances[0])

        csm_mg = deepcopy(csm[0])

        csm_mg.merge(csm[1])
        csm_mg.merge(csm[2])
        csm_mg.merge(csm[3])
        csm_mg.merge(csm[4])
        csm_mg.merge(csm[5])

        count = 0
        for parent_cs, target_csm in additional_cs.items():
            lcs = self.LCS(coordinates=[count, count + 1, count + 2])
            csm_mg.add_cs(f"additional_{count}", parent_cs, lcs)
            csm[target_csm].add_cs(f"additional_{count}", parent_cs, lcs)
            count += 1

        # unmerge -----------------------------------------
        subs = csm_mg.unmerge()

        # checks ------------------------------------------
        csm_res = [csm_mg] + subs
        assert len(csm_res) == 6

        for i, current_lcs in enumerate(csm_res):
            assert csm_res[i] == current_lcs

    # test_unmerge_merged_nested -------------------------------------------------------

    @pytest.mark.parametrize(
        "additional_cs",
        [
            ({}),
            ({"lcs0": 0}),
            ({"lcs1": 0}),
            ({"lcs2": 0}),
            ({"lcs3": 0}),
            ({"lcs4": 1}),
            ({"lcs5": 2}),
            ({"lcs6": 2}),
            ({"lcs7": 3}),
            ({"lcs8": 3}),
            ({"lcs9": 4}),
            ({"lcs10": 5}),
            ({"lcs2": 0, "lcs5": 2, "lcs7": 3, "lcs8": 3}),
            ({"lcs0": 0, "lcs3": 0, "lcs4": 1, "lcs6": 2, "lcs10": 5}),
        ],
    )
    def test_unmerge_merged_nested(self, list_of_csm_and_lcs_instances, additional_cs):
        """Test the CSM unmerge function.

        In this test case, several systems are merged together before they are merged
        to the target system. This creates a nested subsystem structure.
        """
        # setup -------------------------------------------
        csm = deepcopy(list_of_csm_and_lcs_instances[0])

        csm_mg = deepcopy(csm[0])

        csm_n3 = deepcopy(csm[3])
        csm_n3.merge(csm[5])
        csm_n2 = deepcopy(csm[2])
        csm_n2.merge(csm_n3)
        csm_mg.merge(csm[1])
        csm_mg.merge(csm[4])
        csm_mg.merge(csm_n2)

        count = 0
        for parent_cs, target_csm in additional_cs.items():
            lcs = self.LCS(coordinates=[count, count + 1, count + 2])
            csm_mg.add_cs(f"additional_{count}", parent_cs, lcs)
            csm[target_csm].add_cs(f"additional_{count}", parent_cs, lcs)
            if target_csm in [3, 5]:
                csm_n3.add_cs(f"additional_{count}", parent_cs, lcs)
            if target_csm in [2, 3, 5]:
                csm_n2.add_cs(f"additional_{count}", parent_cs, lcs)
            count += 1

        # unmerge -----------------------------------------
        subs = csm_mg.unmerge()

        # checks ------------------------------------------
        assert len(subs) == 3

        assert csm_mg == csm[0]
        assert subs[0] == csm[1]
        assert subs[1] == csm[4]
        assert subs[2] == csm_n2

        # unmerge sub -------------------------------------
        sub_subs = subs[2].unmerge()

        # checks ------------------------------------------
        assert len(sub_subs) == 1

        assert subs[2] == csm[2]
        assert sub_subs[0] == csm_n3

        # unmerge sub sub ---------------------------------
        sub_sub_subs = sub_subs[0].unmerge()

        # checks ------------------------------------------
        assert len(sub_sub_subs) == 1

        assert sub_subs[0] == csm[3]
        assert sub_sub_subs[0] == csm[5]

    # test_delete_cs_with_serially_merged_subsystems -----------------------------------

    @pytest.mark.parametrize(
        "name, subsystems_exp, num_cs_exp",
        [
            ("lcs1", ["csm1", "csm2", "csm3", "csm4", "csm5"], 15),
            ("lcs2", ["csm1"], 4),
            ("lcs3", ["csm1"], 6),
            ("lcs4", ["csm2", "csm3", "csm4", "csm5"], 15),
            ("lcs5", ["csm1", "csm4"], 8),
            ("lcs6", ["csm1", "csm4"], 10),
            ("lcs7", ["csm1", "csm2", "csm4"], 12),
            ("lcs8", ["csm1", "csm2", "csm4", "csm5"], 15),
            ("lcs9", ["csm1", "csm2", "csm3", "csm5"], 15),
            ("lcs10", ["csm1", "csm2", "csm3", "csm4"], 14),
            ("add0", ["csm1", "csm2", "csm3", "csm4", "csm5"], 15),
            ("add1", ["csm1", "csm2", "csm3", "csm4", "csm5"], 15),
            ("add2", ["csm1", "csm2", "csm3", "csm4", "csm5"], 15),
            ("add3", ["csm1", "csm2", "csm3", "csm4", "csm5"], 15),
            ("add4", ["csm1", "csm2", "csm3", "csm4", "csm5"], 15),
        ],
    )
    def test_delete_cs_with_serially_merged_subsystems(
        self, list_of_csm_and_lcs_instances, name, subsystems_exp, num_cs_exp
    ):
        """Test the delete_cs function with subsystems that were merged serially."""
        # setup -------------------------------------------
        csm = deepcopy(list_of_csm_and_lcs_instances[0])

        csm_mg = deepcopy(csm[0])

        csm_mg.merge(csm[1])
        csm_mg.merge(csm[2])
        csm_mg.merge(csm[3])
        csm_mg.merge(csm[4])
        csm_mg.merge(csm[5])

        # add some additional coordinate systems
        target_system_index = [0, 2, 5, 7, 10]
        for i, _ in enumerate(target_system_index):
            lcs = self.LCS(coordinates=[i, 2 * i, -i])
            csm_mg.add_cs(f"add{i}", f"lcs{target_system_index[i]}", lcs)

        # just to avoid useless tests (delete does nothing if the lcs doesn't exist)
        assert name in csm_mg.get_coordinate_system_names()

        # delete coordinate system ------------------------
        csm_mg.delete_cs(name, True)

        # check -------------------------------------------
        assert csm_mg.number_of_subsystems == len(subsystems_exp)
        assert csm_mg.number_of_coordinate_systems == num_cs_exp

        for sub_exp in subsystems_exp:
            assert sub_exp in csm_mg.subsystem_names

    # test_delete_cs_with_nested_subsystems --------------------------------------------

    @pytest.mark.parametrize(
        "name, subsystems_exp, num_cs_exp",
        [
            ("lcs1", ["csm1", "csm2", "csm4"], 17),
            ("lcs2", ["csm1"], 4),
            ("lcs3", ["csm1"], 6),
            ("lcs4", ["csm2", "csm4"], 17),
            ("lcs5", ["csm1", "csm4"], 8),
            ("lcs6", ["csm1", "csm4"], 11),
            ("lcs7", ["csm1", "csm4"], 14),
            ("lcs8", ["csm1", "csm4"], 16),
            ("lcs9", ["csm1", "csm2"], 17),
            ("lcs10", ["csm1", "csm4"], 16),
            ("add0", ["csm1", "csm2", "csm4"], 17),
            ("add1", ["csm1", "csm2", "csm4"], 17),
            ("add2", ["csm1", "csm2", "csm4"], 17),
            ("add3", ["csm1", "csm2", "csm4"], 17),
            ("add4", ["csm1", "csm2", "csm4"], 17),
            ("nes0", ["csm1", "csm4"], 17),
            ("nes1", ["csm1", "csm4"], 17),
        ],
    )
    def test_delete_cs_with_nested_subsystems(
        self, list_of_csm_and_lcs_instances, name, subsystems_exp, num_cs_exp
    ):
        """Test the delete_cs function with nested subsystems."""
        # setup -------------------------------------------
        csm = deepcopy(list_of_csm_and_lcs_instances[0])

        csm_mg = deepcopy(csm[0])

        csm_n3 = deepcopy(csm[3])
        csm_n3.add_cs("nes0", "lcs8", self.LCS(coordinates=[1, 2, 3]))
        csm_n3.merge(csm[5])
        csm_n2 = deepcopy(csm[2])
        csm_n2.add_cs("nes1", "lcs5", self.LCS(coordinates=[-1, -2, -3]))
        csm_n2.merge(csm_n3)
        csm_mg.merge(csm[1])
        csm_mg.merge(csm[4])
        csm_mg.merge(csm_n2)

        # add some additional coordinate systems
        target_system_indices = [0, 2, 5, 7, 10]
        for i, target_system_index in enumerate(target_system_indices):
            lcs = self.LCS(coordinates=[i, 2 * i, -i])
            csm_mg.add_cs(f"add{i}", f"lcs{target_system_index}", lcs)

        # just to avoid useless tests (delete does nothing if the lcs doesn't exist)
        assert name in csm_mg.get_coordinate_system_names()

        # delete coordinate system ------------------------
        csm_mg.delete_cs(name, True)

        # check -------------------------------------------
        assert csm_mg.number_of_subsystems == len(subsystems_exp)
        assert csm_mg.number_of_coordinate_systems == num_cs_exp

        for sub_exp in subsystems_exp:
            assert sub_exp in csm_mg.subsystem_names

    # test_plot ------------------------------------------------------------------------

    def test_plot(self):
        """Test if the plot function runs without problems. Output is not checked."""
        csm_global = self.CSM("root", "global coordinate systems")
        csm_global.create_cs("specimen", "root", coordinates=[1, 2, 3])
        csm_global.create_cs("robot head", "root", coordinates=[4, 5, 6])

        csm_specimen = self.CSM("specimen", "specimen coordinate systems")
        csm_specimen.create_cs("thermocouple 1", "specimen", coordinates=[1, 1, 0])
        csm_specimen.create_cs("thermocouple 2", "specimen", coordinates=[1, 4, 0])

        csm_robot = self.CSM("robot head", "robot coordinate systems")
        csm_robot.create_cs("torch", "robot head", coordinates=[0, 0, -2])
        csm_robot.create_cs("mount point 1", "robot head", coordinates=[0, 1, -1])
        csm_robot.create_cs("mount point 2", "robot head", coordinates=[0, -1, -1])

        csm_scanner = self.CSM("scanner", "scanner coordinate systems")
        csm_scanner.create_cs("mount point 1", "scanner", coordinates=[0, 0, 2])

        csm_robot.merge(csm_scanner)
        csm_global.merge(csm_robot)
        csm_global.merge(csm_specimen)

        csm_global.plot()


def test_coordinate_system_manager_init():
    """Test the init method of the coordinate system manager."""
    # default construction ----------------------
    csm = tf.CoordinateSystemManager(root_coordinate_system_name="root")
    assert csm.number_of_coordinate_systems == 1
    assert csm.number_of_neighbors("root") == 0

    # Exceptions---------------------------------
    # Invalid root system name
    with pytest.raises(Exception):
        tf.CoordinateSystemManager({})


def test_coordinate_system_manager_create_coordinate_system():
    """Test direct construction of coordinate systems in the coordinate system manager.

    Create multiple coordinate systems with all provided methods and check
    if they are constructed correctly.

    """
    angles_x = np.array([0.5, 1, 2, 2.5]) * np.pi / 2
    angles_y = np.array([1.5, 0, 1, 0.5]) * np.pi / 2
    angles = np.array([[*angles_y], [*angles_x]]).transpose()
    angles_deg = 180 / np.pi * angles

    rot_mat_x = tf.rotation_matrix_x(angles_x)
    rot_mat_y = tf.rotation_matrix_y(angles_y)

    time = TDI([0, 6, 12, 18], "H")
    orientations = np.matmul(rot_mat_x, rot_mat_y)
    coords = [[1, 0, 0], [-1, 0, 2], [3, 5, 7], [-4, -5, -6]]

    vec_x = orientations[:, :, 0]
    vec_y = orientations[:, :, 1]
    vec_z = orientations[:, :, 2]

    csm = tf.CoordinateSystemManager("root")
    lcs_default = tf.LocalCoordinateSystem()

    # orientation and coordinates -------------------------
    csm.create_cs("lcs_init_default", "root")
    check_coordinate_system(
        csm.get_local_coordinate_system("lcs_init_default"),
        lcs_default.orientation,
        lcs_default.coordinates,
        True,
    )

    csm.create_cs("lcs_init_tdp", "root", orientations, coords, time)
    check_coordinate_system(
        csm.get_local_coordinate_system("lcs_init_tdp"),
        orientations,
        coords,
        True,
        time=time,
    )

    # from euler ------------------------------------------
    csm.create_cs_from_euler("lcs_euler_default", "root", "yx", angles[0])
    check_coordinate_system(
        csm.get_local_coordinate_system("lcs_euler_default"),
        orientations[0],
        lcs_default.coordinates,
        True,
    )

    csm.create_cs_from_euler(
        "lcs_euler_tdp", "root", "yx", angles_deg, True, coords, time
    )
    check_coordinate_system(
        csm.get_local_coordinate_system("lcs_euler_tdp"),
        orientations,
        coords,
        True,
        time=time,
    )

    # from xyz --------------------------------------------
    csm.create_cs_from_xyz("lcs_xyz_default", "root", vec_x[0], vec_y[0], vec_z[0])
    check_coordinate_system(
        csm.get_local_coordinate_system("lcs_xyz_default"),
        orientations[0],
        lcs_default.coordinates,
        True,
    )

    csm.create_cs_from_xyz("lcs_xyz_tdp", "root", vec_x, vec_y, vec_z, coords, time)
    check_coordinate_system(
        csm.get_local_coordinate_system("lcs_xyz_tdp"),
        orientations,
        coords,
        True,
        time=time,
    )

    # from xy and orientation -----------------------------
    csm.create_cs_from_xy_and_orientation("lcs_xyo_default", "root", vec_x[0], vec_y[0])
    check_coordinate_system(
        csm.get_local_coordinate_system("lcs_xyo_default"),
        orientations[0],
        lcs_default.coordinates,
        True,
    )

    csm.create_cs_from_xy_and_orientation(
        "lcs_xyo_tdp", "root", vec_x, vec_y, True, coords, time
    )
    check_coordinate_system(
        csm.get_local_coordinate_system("lcs_xyo_tdp"),
        orientations,
        coords,
        True,
        time=time,
    )

    # from xz and orientation -----------------------------
    csm.create_cs_from_xz_and_orientation("lcs_xzo_default", "root", vec_x[0], vec_z[0])
    check_coordinate_system(
        csm.get_local_coordinate_system("lcs_xzo_default"),
        orientations[0],
        lcs_default.coordinates,
        True,
    )

    csm.create_cs_from_xz_and_orientation(
        "lcs_xzo_tdp", "root", vec_x, vec_z, True, coords, time
    )
    check_coordinate_system(
        csm.get_local_coordinate_system("lcs_xzo_tdp"),
        orientations,
        coords,
        True,
        time=time,
    )

    # from yz and orientation -----------------------------
    csm.create_cs_from_yz_and_orientation("lcs_yzo_default", "root", vec_y[0], vec_z[0])
    check_coordinate_system(
        csm.get_local_coordinate_system("lcs_yzo_default"),
        orientations[0],
        lcs_default.coordinates,
        True,
    )

    csm.create_cs_from_yz_and_orientation(
        "lcs_yzo_tdp", "root", vec_y, vec_z, True, coords, time
    )
    check_coordinate_system(
        csm.get_local_coordinate_system("lcs_yzo_tdp"),
        orientations,
        coords,
        True,
        time=time,
    )


def test_coordinate_system_manager_get_local_coordinate_system_no_time_dependency():
    """Test the get_local_coordinate_system function.

    This function also tests, if the internally performed transformations are correct.

    """
    # define some coordinate systems
    lcs1_in_root = tf.LocalCoordinateSystem(tf.rotation_matrix_z(np.pi / 2), [1, 2, 3])
    lcs2_in_root = tf.LocalCoordinateSystem(tf.rotation_matrix_y(np.pi / 2), [3, -3, 1])
    lcs3_in_lcs2 = tf.LocalCoordinateSystem(tf.rotation_matrix_x(np.pi / 2), [1, -1, 3])

    csm = tf.CoordinateSystemManager(root_coordinate_system_name="root")
    csm.add_cs("lcs1", "root", lcs1_in_root)
    csm.add_cs("lcs2", "root", lcs2_in_root)
    csm.add_cs("lcs3", "lcs2", lcs3_in_lcs2)

    # check stored transformations
    lcs1_in_root_returned = csm.get_local_coordinate_system("lcs1")
    check_coordinate_system(
        lcs1_in_root_returned, lcs1_in_root.orientation, lcs1_in_root.coordinates, True
    )

    lcs2_in_root_returned = csm.get_local_coordinate_system("lcs2")
    check_coordinate_system(
        lcs2_in_root_returned, lcs2_in_root.orientation, lcs2_in_root.coordinates, True
    )

    lcs3_in_lcs2_returned = csm.get_local_coordinate_system("lcs3")
    check_coordinate_system(
        lcs3_in_lcs2_returned, lcs3_in_lcs2.orientation, lcs3_in_lcs2.coordinates, True
    )

    # check calculated transformations
    lcs_3_in_root = csm.get_local_coordinate_system("lcs3", "root")
    expected_orientation = [[0, 1, 0], [0, 0, -1], [-1, 0, 0]]
    expected_coordinates = [6, -4, 0]
    check_coordinate_system(
        lcs_3_in_root, expected_orientation, expected_coordinates, True
    )

    root_in_lcs3 = csm.get_local_coordinate_system("root", "lcs3")
    expected_orientation = [[0, 0, -1], [1, 0, 0], [0, -1, 0]]
    expected_coordinates = [0, -6, -4]
    check_coordinate_system(
        root_in_lcs3, expected_orientation, expected_coordinates, True
    )

    lcs_3_in_lcs1 = csm.get_local_coordinate_system("lcs3", "lcs1")
    expected_orientation = [[0, 0, -1], [0, -1, 0], [-1, 0, 0]]
    expected_coordinates = [-6, -5, -3]
    check_coordinate_system(
        lcs_3_in_lcs1, expected_orientation, expected_coordinates, True
    )

    lcs_1_in_lcs3 = csm.get_local_coordinate_system("lcs1", "lcs3")
    expected_orientation = [[0, 0, -1], [0, -1, 0], [-1, 0, 0]]
    expected_coordinates = [-3, -5, -6]
    check_coordinate_system(
        lcs_1_in_lcs3, expected_orientation, expected_coordinates, True
    )

    # self referencing --------------------------
    lcs_1_in_lcs1 = csm.get_local_coordinate_system("lcs1", "lcs1")
    expected_orientation = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    expected_coordinates = [0, 0, 0]
    check_coordinate_system(
        lcs_1_in_lcs1, expected_orientation, expected_coordinates, True
    )

    # exceptions --------------------------------
    # system does not exist
    with pytest.raises(ValueError):
        csm.get_local_coordinate_system("not there", "root")
    with pytest.raises(ValueError):
        csm.get_local_coordinate_system("root", "not there")
    with pytest.raises(ValueError):
        csm.get_local_coordinate_system("not there", "not there")
    # no parent system
    with pytest.raises(ValueError):
        csm.get_local_coordinate_system("root")


def test_coordinate_system_manager_get_local_coordinate_system_time_dependent():
    """Test the get_local_coordinate_system function with time dependent systems.

    The point of this test is to assure that necessary time interpolations do not cause
    wrong results when transforming to the desired reference coordinate system.
    """
    # TODO: Make the test more flexible
    # Currently, the test only works with times that have specific boundaries. The
    # reason for this is how the expected results are calculated.
    lcs_0_time = TDI([i * 6 for i in range(49)], "H")
    lcs_0_coordinates = np.zeros([len(lcs_0_time), 3])
    lcs_0_in_root = tf.LocalCoordinateSystem(
        coordinates=lcs_0_coordinates, time=lcs_0_time
    )

    lcs_1_time = TDI([0, 4, 8, 12], "D")
    lcs_1_coordinates = [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]
    lcs_1_orientation = tf.rotation_matrix_z(np.array([0, 1 / 3, 2 / 3, 1]) * np.pi * 2)
    lcs_1_in_lcs0 = tf.LocalCoordinateSystem(
        coordinates=lcs_1_coordinates, orientation=lcs_1_orientation, time=lcs_1_time
    )

    lcs_2_time = TDI([0, 4, 8, 12], "D")
    lcs_2_coordinates = [1, 0, 0]
    lcs_2_orientation = tf.rotation_matrix_z(np.array([0, 1 / 3, 2 / 3, 1]) * np.pi * 2)
    lcs_2_in_lcs1 = tf.LocalCoordinateSystem(
        coordinates=lcs_2_coordinates, orientation=lcs_2_orientation, time=lcs_2_time
    )

    lcs_3_in_lcs1 = tf.LocalCoordinateSystem(coordinates=[0, 1, 0])

    csm = tf.CoordinateSystemManager("root")
    csm.add_cs("lcs_0", "root", lcs_0_in_root)
    csm.add_cs("lcs_1", "lcs_0", lcs_1_in_lcs0)
    csm.add_cs("lcs_2", "lcs_1", lcs_2_in_lcs1)
    csm.add_cs("lcs_3", "lcs_1", lcs_3_in_lcs1)

    lcs_1_in_root = csm.get_local_coordinate_system("lcs_1", "root")
    lcs_2_in_root = csm.get_local_coordinate_system("lcs_2", "root")
    lcs_3_in_root = csm.get_local_coordinate_system("lcs_3", "root")

    assert np.all(lcs_0_time == lcs_1_in_root.time)
    assert np.all(lcs_0_time == lcs_2_in_root.time)
    assert np.all(lcs_0_time == lcs_3_in_root.time)

    num_times = len(lcs_0_time)
    for i in range(num_times):
        weight = i / (num_times - 1)
        angle = weight * 2 * np.pi
        pos_x = weight * 3

        # check orientations
        lcs_1_orientation_exp = tf.rotation_matrix_z(angle)
        lcs_2_orientation_exp = tf.rotation_matrix_z(2 * angle)

        assert ut.matrix_is_close(lcs_1_in_root.orientation[i], lcs_1_orientation_exp)
        assert ut.matrix_is_close(lcs_2_in_root.orientation[i], lcs_2_orientation_exp)
        assert ut.matrix_is_close(lcs_3_in_root.orientation[i], lcs_1_orientation_exp)

        # check coordinates
        c = np.cos(angle)
        s = np.sin(angle)
        rot_p_x = c
        rot_p_y = s

        lcs_1_coordinates_exp = [pos_x, 0, 0]
        lcs_2_coordinates_exp = [rot_p_x + pos_x, rot_p_y, 0]
        lcs_3_coordinates_exp = [-rot_p_y + pos_x, rot_p_x, 0]

        assert ut.vector_is_close(lcs_1_in_root.coordinates[i], lcs_1_coordinates_exp)
        assert ut.vector_is_close(lcs_2_in_root.coordinates[i], lcs_2_coordinates_exp)
        assert ut.vector_is_close(lcs_3_in_root.coordinates[i], lcs_3_coordinates_exp)

    # define helper function --------------------

    def _validate_transformed_lcs_2(lcs_2, time):
        num_times = len(lcs_2.time)

        assert num_times == len(time)
        assert np.all(lcs_2.time == time)

        for i in range(num_times):
            weight = i / (num_times - 1)
            angle = weight * 2 * np.pi
            pos_x = weight * 3

            # check orientations
            lcs_2_orientation_exp = tf.rotation_matrix_z(2 * angle)
            assert ut.matrix_is_close(lcs_2.orientation[i], lcs_2_orientation_exp)

            # check coordinates
            c = np.cos(angle)
            s = np.sin(angle)
            rot_p_x = c
            rot_p_y = s

            lcs_2_coordinates_exp = [rot_p_x + pos_x, rot_p_y, 0]
            assert ut.vector_is_close(lcs_2.coordinates[i], lcs_2_coordinates_exp)

    # use time of another system ----------------

    lcs_2_in_root_t_lcs_1 = csm.get_local_coordinate_system("lcs_2", "root", "lcs_1")
    _validate_transformed_lcs_2(lcs_2_in_root_t_lcs_1, lcs_1_time)

    # use DatetimeIndex -------------------------
    time_custom = TDI([i * 4 for i in range(73)], "H")
    lcs_2_in_root_t_custom = csm.get_local_coordinate_system(
        "lcs_2", "root", time_custom
    )
    _validate_transformed_lcs_2(lcs_2_in_root_t_custom, time_custom)

    # self referencing --------------------------

    lcs_2_in_lcs_2 = csm.get_local_coordinate_system("lcs_2", "lcs_2", lcs_1_time)
    expected_orientation = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    expected_coordinates = [0, 0, 0]
    check_coordinate_system(
        lcs_2_in_lcs_2, expected_orientation, expected_coordinates, True
    )

    # exceptions --------------------------------
    # time reference system does not exist
    with pytest.raises(ValueError):
        csm.get_local_coordinate_system("lcs_2", "root", "not there")
    # root system is never time dependent
    with pytest.raises(ValueError):
        csm.get_local_coordinate_system("lcs_2", "root", "root")
    # time reference system is not time dependent
    with pytest.raises(ValueError):
        csm.get_local_coordinate_system("lcs_2", "root", "lcs_3")
    # invalid inputs
    with pytest.raises(TypeError):
        csm.get_local_coordinate_system("lcs_2", "root", 1)
    with pytest.raises(Exception):
        csm.get_local_coordinate_system("lcs_2", "root", ["grr", "42", "asdf"])


def test_coordinate_system_manager_time_union():
    """Test the coordinate system managers time union function."""
    orientation = tf.rotation_matrix_z([0, 1, 2])
    coordinates = [[1, 6, 3], [8, 2, 6], [4, 4, 4]]
    lcs_0 = tf.LocalCoordinateSystem(
        orientation=orientation, coordinates=coordinates, time=TDI([1, 4, 7], "D"),
    )
    lcs_1 = tf.LocalCoordinateSystem(
        orientation=orientation, coordinates=coordinates, time=TDI([1, 5, 9], "D"),
    )
    lcs_2 = tf.LocalCoordinateSystem(
        orientation=orientation, coordinates=coordinates, time=TDI([1, 6, 11], "D"),
    )
    lcs_3 = tf.LocalCoordinateSystem()

    csm = tf.CoordinateSystemManager("root")
    csm.add_cs("lcs_0", "root", lcs_0)
    csm.add_cs("lcs_1", "lcs_0", lcs_1)
    csm.add_cs("lcs_2", "root", lcs_2)
    csm.add_cs("lcs_3", "lcs_2", lcs_3)

    # full union --------------------------------
    expected_times = TDI([1, 4, 5, 6, 7, 9, 11], "D")

    assert np.all(expected_times == csm.time_union())

    # selected union ------------------------------
    expected_times = TDI([1, 4, 5, 7, 9], "D")
    list_of_edges = [("root", "lcs_0"), ("lcs_0", "lcs_1")]

    assert np.all(expected_times == csm.time_union(list_of_edges=list_of_edges))


def test_coordinate_system_manager_interp_time():
    """Test the coordinate system managers interp_time functions."""
    # Setup -------------------------------------
    angles = ut.to_float_array([0, np.pi / 2, np.pi])
    orientation = tf.rotation_matrix_z(angles)
    coordinates = ut.to_float_array([[5, 0, 0], [1, 0, 0], [1, 4, 4]])

    time_0 = TDI([1, 4, 7], "D")
    time_1 = TDI([1, 5, 9], "D")
    time_2 = TDI([1, 6, 11], "D")

    lcs_3_in_lcs_2 = tf.LocalCoordinateSystem(
        orientation=tf.rotation_matrix_y(1), coordinates=[4, 2, 0]
    )

    csm = tf.CoordinateSystemManager("root")
    csm.create_cs("lcs_0", "root", orientation, coordinates, time_0)
    csm.create_cs("lcs_1", "lcs_0", orientation, coordinates, time_1)
    csm.create_cs("lcs_2", "root", orientation, coordinates, time_2)
    csm.create_cs("lcs_3", "lcs_2", tf.rotation_matrix_y(1), [4, 2, 0])

    # interp_time -------------------------------
    time_interp = TDI([1, 3, 5, 7, 9], "D")
    csm_interp = csm.interp_time(time_interp)

    assert np.all(csm_interp.time_union() == time_interp)

    assert np.all(csm_interp.get_local_coordinate_system("lcs_0").time == time_interp)
    assert np.all(csm_interp.get_local_coordinate_system("lcs_1").time == time_interp)
    assert np.all(csm_interp.get_local_coordinate_system("lcs_2").time == time_interp)
    assert csm_interp.get_local_coordinate_system("lcs_3").time is None

    coords_0_exp = [
        [5, 0, 0],
        [7 / 3, 0, 0],
        [1, 4 / 3, 4 / 3],
        [1, 4, 4],
        [1, 4, 4],
    ]
    orient_0_exp = tf.rotation_matrix_z([0, np.pi / 3, 2 * np.pi / 3, np.pi, np.pi])
    lcs_0_in_root_exp = tf.LocalCoordinateSystem(
        orient_0_exp, coords_0_exp, time_interp
    )

    coords_1_exp = [[5, 0, 0], [3, 0, 0], [1, 0, 0], [1, 2, 2], [1, 4, 4]]
    orient_1_exp = tf.rotation_matrix_z([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
    lcs_1_in_lcs_0_exp = tf.LocalCoordinateSystem(
        orient_1_exp, coords_1_exp, time_interp
    )

    coords_2_exp = [
        [5, 0, 0],
        [17 / 5, 0, 0],
        [9 / 5, 0, 0],
        [1, 4 / 5, 4 / 5],
        [1, 12 / 5, 12 / 5],
    ]
    orient_2_exp = tf.rotation_matrix_z(
        [0, np.pi / 5, 2 * np.pi / 5, 3 * np.pi / 5, 4 * np.pi / 5]
    )
    lcs_2_in_root_exp = tf.LocalCoordinateSystem(
        orient_2_exp, coords_2_exp, time_interp
    )

    lcs_3_in_lcs_2_exp = tf.LocalCoordinateSystem(
        lcs_3_in_lcs_2.orientation, lcs_3_in_lcs_2.coordinates
    )

    for cs_name in csm_interp.graph.nodes:
        if cs_name == "root":
            continue
        ps_name = csm_interp.get_parent_system_name(cs_name)

        if cs_name == "lcs_0":
            exp = lcs_0_in_root_exp
            exp_inv = lcs_0_in_root_exp.invert()
        elif cs_name == "lcs_1":
            exp = lcs_1_in_lcs_0_exp
            exp_inv = lcs_1_in_lcs_0_exp.invert()
        elif cs_name == "lcs_2":
            exp = lcs_2_in_root_exp
            exp_inv = lcs_2_in_root_exp.invert()
        elif cs_name == "lcs_3":
            exp = lcs_3_in_lcs_2_exp
            exp_inv = lcs_3_in_lcs_2_exp.invert()
        else:
            exp = csm.get_local_coordinate_system(cs_name, ps_name)
            exp_inv = csm.get_local_coordinate_system(ps_name, cs_name)

        lcs = csm_interp.get_local_coordinate_system(cs_name, ps_name)
        lcs_inv = csm_interp.get_local_coordinate_system(ps_name, cs_name)

        check_coordinate_systems_close(lcs, exp)
        check_coordinate_systems_close(lcs_inv, exp_inv)

    # specific coordinate system (single) -----------------
    time_interp_lcs0 = TDI([3, 5], "D")
    csm_interp_single = csm.interp_time(time_interp_lcs0, "lcs_0")

    coords_0_exp = np.array(coords_0_exp)[[1, 2], :]
    orient_0_exp = np.array(orient_0_exp)[[1, 2], :]
    lcs_0_in_root_exp = tf.LocalCoordinateSystem(
        orient_0_exp, coords_0_exp, time_interp_lcs0
    )

    for cs_name in csm_interp_single.graph.nodes:
        if cs_name == "root":
            continue
        ps_name = csm_interp_single.get_parent_system_name(cs_name)

        if cs_name == "lcs_0":
            exp = lcs_0_in_root_exp
            exp_inv = lcs_0_in_root_exp.invert()
        else:
            exp = csm.get_local_coordinate_system(cs_name, ps_name)
            exp_inv = csm.get_local_coordinate_system(ps_name, cs_name)

        lcs = csm_interp_single.get_local_coordinate_system(cs_name, ps_name)
        lcs_inv = csm_interp_single.get_local_coordinate_system(ps_name, cs_name)

        check_coordinate_systems_close(lcs, exp)
        check_coordinate_systems_close(lcs_inv, exp_inv)

    # specific coordinate systems (multiple) --------------
    time_interp_multiple = TDI([5, 7, 9], "D")
    csm_interp_multiple = csm.interp_time(time_interp_multiple, ["lcs_1", "lcs_2"])

    coords_1_exp = np.array(coords_1_exp)[2:, :]
    orient_1_exp = np.array(orient_1_exp)[2:, :]
    coords_2_exp = np.array(coords_2_exp)[2:, :]
    orient_2_exp = np.array(orient_2_exp)[2:, :]

    lcs_1_in_lcs_0_exp = tf.LocalCoordinateSystem(
        orient_1_exp, coords_1_exp, time_interp_multiple
    )
    lcs_2_in_root_exp = tf.LocalCoordinateSystem(
        orient_2_exp, coords_2_exp, time_interp_multiple
    )

    for cs_name in csm_interp_multiple.graph.nodes:
        if cs_name == "root":
            continue
        ps_name = csm_interp_multiple.get_parent_system_name(cs_name)

        if cs_name == "lcs_1":
            exp = lcs_1_in_lcs_0_exp
            exp_inv = lcs_1_in_lcs_0_exp.invert()
        elif cs_name == "lcs_2":
            exp = lcs_2_in_root_exp
            exp_inv = lcs_2_in_root_exp.invert()
        else:
            exp = csm.get_local_coordinate_system(cs_name, ps_name)
            exp_inv = csm.get_local_coordinate_system(ps_name, cs_name)

        lcs = csm_interp_multiple.get_local_coordinate_system(cs_name, ps_name)
        lcs_inv = csm_interp_multiple.get_local_coordinate_system(ps_name, cs_name)

        check_coordinate_systems_close(lcs, exp)
        check_coordinate_systems_close(lcs_inv, exp_inv)


def test_coordinate_system_manager_transform_data():
    """Test the coordinate system managers transform_data function."""
    # define some coordinate systems
    # TODO: test more unique rotations - not 90°
    lcs1_in_root = tf.LocalCoordinateSystem(tf.rotation_matrix_z(np.pi / 2), [1, 2, 3])
    lcs2_in_root = tf.LocalCoordinateSystem(tf.rotation_matrix_y(np.pi / 2), [3, -3, 1])
    lcs3_in_lcs2 = tf.LocalCoordinateSystem(tf.rotation_matrix_x(np.pi / 2), [1, -1, 3])

    csm = tf.CoordinateSystemManager(root_coordinate_system_name="root")
    csm.add_cs("lcs_1", "root", lcs1_in_root)
    csm.add_cs("lcs_2", "root", lcs2_in_root)
    csm.add_cs("lcs_3", "lcs_2", lcs3_in_lcs2)

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
    """Test the coordinate system managers assign_data and get_data functions."""
    # test setup
    lcs1_in_root = tf.LocalCoordinateSystem(tf.rotation_matrix_z(np.pi / 2), [1, 2, 3])
    lcs2_in_root = tf.LocalCoordinateSystem(tf.rotation_matrix_y(np.pi / 2), [3, -3, 1])
    lcs3_in_lcs2 = tf.LocalCoordinateSystem(tf.rotation_matrix_x(np.pi / 2), [1, -1, 3])

    csm = tf.CoordinateSystemManager(root_coordinate_system_name="root")
    csm.add_cs("lcs_1", "root", lcs1_in_root)
    csm.add_cs("lcs_2", "root", lcs2_in_root)
    csm.add_cs("lcs_3", "lcs_2", lcs3_in_lcs2)

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
