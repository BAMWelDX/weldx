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
        assert np.all(lcs.time == time)
        assert lcs.reference_time == time_ref

    check_coordinate_system_orientation(
        lcs.orientation, orientation_expected, positive_orientation_expected
    )

    assert np.allclose(lcs.coordinates.values, coordinates_expected, atol=1e-9)


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


# --------------------------------------------------------------------------------------
# LocalCoordinateSystem
# --------------------------------------------------------------------------------------


def r_mat_x(factors) -> np.ndarray:
    """Get an array of rotation matrices that represent a rotation around the x-axis.

    The rotation angles are the provided factors times pi.

    Parameters
    ----------
    factors:
        List of factors that are multiplied with pi to get the rotation angles.

    Returns
    -------
    numpy.ndarray:
        An array of rotation matrices

    """
    return WXRotation.from_euler("x", np.array(factors) * np.pi).as_matrix()


def r_mat_y(factors) -> np.ndarray:
    """Get an array of rotation matrices that represent a rotation around the y-axis.

    The rotation angles are the provided factors times pi.

    Parameters
    ----------
    factors:
        List of factors that are multiplied with pi to get the rotation angles.

    Returns
    -------
    numpy.ndarray:
        An array of rotation matrices

    """
    return WXRotation.from_euler("y", np.array(factors) * np.pi).as_matrix()


def r_mat_z(factors) -> np.ndarray:
    """Get an array of rotation matrices that represent a rotation around the z-axis.

    The rotation angles are the provided factors times pi.

    Parameters
    ----------
    factors:
        List of factors that are multiplied with pi to get the rotation angles.

    Returns
    -------
    numpy.ndarray:
        An array of rotation matrices

    """
    return WXRotation.from_euler("z", np.array(factors) * np.pi).as_matrix()


class TestLocalCoordinateSystem:
    """Test the 'LocalCoordinateSystem' class."""

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
            Expected return value of the 'time_ref' property
        datetime_exp:
            Expected return value of the 'datetimeindex' property
        quantity_exp:
            Expected return value of the 'time_quantity' property

        """
        # setup
        orientation = r_mat_z([0.5, 1.0, 1.5])
        coordinates = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        lcs = tf.LocalCoordinateSystem(
            orientation, coordinates, time, time_ref=time_ref
        )

        # check results

        assert np.all(lcs.time == time_exp)
        assert lcs.reference_time == time_ref_exp
        assert np.all(lcs.datetimeindex == datetime_exp)
        assert np.all(lcs.time_quantity == quantity_exp)

    # test_time_warning ----------------------------------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        "coordinates, orientation, time, warning",
        [
            (np.zeros(3), np.eye(3, 3), TDI([0, 2], "s"), UserWarning),
            (np.zeros((2, 3)), np.eye(3, 3), TDI([0, 2], "s"), None),
            (np.zeros(3), np.eye(3, 3), None, None),
        ],
    )
    def test_time_warning(coordinates, orientation, time, warning):
        """Test that warning is emitted when time is provided to a static CS.

        Parameters
        ----------
        coordinates :
            Coordinates of the CS
        orientation :
            Orientation of the CS
        time :
            Provided time
        warning :
            Expected warning

        """
        with pytest.warns(warning):
            LCS(coordinates=coordinates, orientation=orientation, time=time)

    # test_init_time_dsx ---------------------------------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        "time_o,  time_c,  time_exp",
        [
            (TDI([0, 1, 2], "s"), TDI([0, 1, 2], "s"), TDI([0, 1, 2], "s")),
            (
                TDI([0, 2, 4], "s"),
                TDI([1, 3, 5], "s"),
                TDI([0, 1, 2, 3, 4, 5], "s"),
            ),
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
        orientations = WXRotation.from_euler("z", range(len(time_o))).as_matrix()
        coordinates = [[i, i, i] for i in range(len(time_o))]

        dax_o = ut.xr_3d_matrix(orientations, time_o)
        dax_c = ut.xr_3d_vector(coordinates, time_c)

        lcs = tf.LocalCoordinateSystem(dax_o, dax_c, time_ref=time_ref)

        # check results

        assert np.all(lcs.time == time_exp)
        assert lcs.reference_time == time_ref

    # test_init_expr_time_series_as_coord ----------------------------------------------

    @staticmethod
    @pytest.mark.parametrize("time_ref", [None, TS("2020-01-01")])
    @pytest.mark.parametrize(
        "time, angles",
        [
            (None, None),
            (Q_([1, 2, 3], "s"), None),
            (Q_([1, 2, 3], "s"), [1, 2, 3]),
        ],
    )
    def test_init_expr_time_series_as_coord(time, time_ref, angles):
        """Test if a fitting, expression based `TimeSeries` can be used as coordinates.

        Parameters
        ----------
        time :
            The time that should be passed to the `__init__` method of the LCS
        time_ref :
            The reference time that should be passed to the `__init__` method of the LCS
        angles :
            A set of angles that should be used to calculate the orientation matrices.
            Values are irrelevant, just make sure it has the same length as the `time`
            parameter

        """
        coordinates = MathematicalExpression(
            expression="a*t+b", parameters=dict(a=Q_([[1, 0, 0]], "1/s"), b=[1, 2, 3])
        )

        ts_coord = TimeSeries(data=coordinates)
        orientation = None
        if angles is not None:
            orientation = WXRotation.from_euler("x", angles, degrees=True).as_matrix()
        lcs = LCS(
            orientation=orientation, coordinates=ts_coord, time=time, time_ref=time_ref
        )
        assert lcs.has_reference_time == (time_ref is not None)

    # test_init_discrete_time_series_as_coord ------------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        "data, time, conversion_factor",
        [
            (Q_([[1, 0, 0], [1, 1, 0], [1, 1, 1]], "mm"), Q_([1, 2, 3], "s"), 1),
            (Q_([[1, 0, 0], [1, 1, 0], [1, 1, 1]], "m"), Q_([1, 2, 3], "s"), 1000),
            (Q_([[1, 2, 3]], "mm"), Q_([1], "s"), 1),
        ],
    )
    def test_init_discrete_time_series_as_coord(data, time, conversion_factor):
        """Test if a fitting, discrete `TimeSeries` can be used as coordinates.

        Parameters
        ----------
        data :
            Data of the `TimeSeries`
        time :
            Time of the `TimeSeries`
        conversion_factor :
            The conversion factor of the data's quantity to `mm`

        """
        ts_coords = TimeSeries(data, time)
        lcs = LCS(coordinates=ts_coords)

        assert np.allclose(lcs.coordinates, data.m * conversion_factor)
        if len(time) == 1:
            assert lcs.time is None
        else:
            assert np.all(lcs.time_quantity == time)

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
            (
                TDI([1, 2, 3], "D"),
                TS("2020-02-02"),
                "2020-02-01",
                TDI([2, 3, 4], "D"),
            ),
            (
                TDI([1, 2, 3], "D"),
                None,
                "2020-02-01",
                TDI([1, 2, 3], "D"),
            ),
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
        orientation = WXRotation.from_euler("z", [1, 2, 3]).as_matrix()
        coordinates = [[i, i, i] for i in range(3)]
        lcs = tf.LocalCoordinateSystem(
            orientation, coordinates, time, time_ref=time_ref
        )

        lcs.reset_reference_time(time_ref_new)

        # check results
        assert np.all(lcs.time == time_exp)

    # test_reset_reference_time_exceptions ---------------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        "time_ref, time_ref_new,  exception_type, test_name",
        [
            (TS("2020-02-02"), None, TypeError, "# invalid type #1"),
            (TS("2020-02-02"), 42, TypeError, "# invalid type #2"),
        ],
        ids=get_test_name,
    )
    def test_reset_reference_time_exceptions(
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
        orientation = WXRotation.from_euler("z", [1, 2, 3]).as_matrix()
        coordinates = [[i, i, i] for i in range(3)]
        time = TDI([1, 2, 3], "D")

        lcs = tf.LocalCoordinateSystem(
            orientation, coordinates, time, time_ref=time_ref
        )

        with pytest.raises(exception_type):
            lcs.reset_reference_time(time_ref_new)

    # test_interp_time_discrete --------------------------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        "time_ref_lcs, time,time_ref, orientation_exp, coordinates_exp",
        [
            (  # broadcast left
                TS("2020-02-10"),
                TDI([1, 2], "D"),
                TS("2020-02-10"),
                r_mat_z([0, 0]),
                np.array([[2, 8, 7], [2, 8, 7]]),
            ),
            (  # broadcast right
                TS("2020-02-10"),
                TDI([29, 30], "D"),
                TS("2020-02-10"),
                r_mat_z([0.5, 0.5]),
                np.array([[3, 1, 2], [3, 1, 2]]),
            ),
            (  # pure interpolation
                TS("2020-02-10"),
                TDI([11, 14, 17, 20], "D"),
                TS("2020-02-10"),
                r_mat_z([0.125, 0.5, 0.875, 0.75]),
                np.array(
                    [[2.5, 8.25, 5.75], [4, 9, 2], [1, 3.75, 1.25], [1.5, 1.5, 1.5]]
                ),
            ),
            (  # mixed
                TS("2020-02-10"),
                TDI([6, 12, 18, 24, 32], "D"),
                TS("2020-02-10"),
                r_mat_z([0, 0.25, 1, 0.5, 0.5]),
                np.array([[2, 8, 7], [3, 8.5, 4.5], [0, 2, 1], [3, 1, 2], [3, 1, 2]]),
            ),
            (  # different reference times
                TS("2020-02-10"),
                TDI([8, 14, 20, 26, 34], "D"),
                TS("2020-02-08"),
                r_mat_z([0, 0.25, 1, 0.5, 0.5]),
                np.array([[2, 8, 7], [3, 8.5, 4.5], [0, 2, 1], [3, 1, 2], [3, 1, 2]]),
            ),
            (  # no reference time
                None,
                TDI([6, 12, 18, 24, 32], "D"),
                None,
                r_mat_z([0, 0.25, 1, 0.5, 0.5]),
                np.array([[2, 8, 7], [3, 8.5, 4.5], [0, 2, 1], [3, 1, 2], [3, 1, 2]]),
            ),
        ],
    )
    def test_interp_time_discrete(
        time_ref_lcs, time, time_ref, orientation_exp, coordinates_exp
    ):
        """Test the interp_time function with discrete coordinates and orientations.

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
            orientation=r_mat_z([0, 0.5, 1, 0.5]),
            coordinates=np.array([[2, 8, 7], [4, 9, 2], [0, 2, 1], [3, 1, 2]]),
            time=TDI([10, 14, 18, 22], "D"),
            time_ref=time_ref_lcs,
        )

        # test time as input
        lcs_interp = lcs.interp_time(time, time_ref)
        check_coordinate_system(
            lcs_interp, orientation_exp, coordinates_exp, True, time, time_ref
        )

        # test lcs as input
        lcs_interp_like = lcs.interp_time(lcs_interp)
        check_coordinate_system(
            lcs_interp_like, orientation_exp, coordinates_exp, True, time, time_ref
        )

    # test_interp_time_timeseries_as_coords --------------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        "seconds, lcs_ref_sec, ref_sec, time_dep_orientation",
        (
            [
                ([1, 2, 3, 4, 5], None, None, False),
                ([1, 3, 5], 1, 1, False),
                ([1, 3, 5], 1, 1, True),
                ([1, 3, 5], 3, 1, False),
                ([1, 3, 5], 3, 1, True),
                ([1, 3, 5], 1, 3, False),
                ([1, 3, 5], 1, 3, True),
            ]
        ),
    )
    def test_interp_time_timeseries_as_coords(
        seconds: List[float],
        lcs_ref_sec: float,
        ref_sec: float,
        time_dep_orientation: bool,
    ):
        """Test if the `interp_time` method works with a TimeSeries as coordinates.

        The test creates a reference LCS where the coordinates x-values increase
        linearly in time.

        Parameters
        ----------
        seconds :
            The seconds (time delta) that should be interpolated
        lcs_ref_sec :
            The seconds of the reference time, that will passed to the created LCS. If
            `None`, no reference time will be passed
        ref_sec :
            The seconds of the reference time, that will passed to `interp_time`. If
            `None`, no reference time will be passed
        time_dep_orientation :
            If `True` a time dependent orientation will be passed to the LCS.
            The orientation is a rotation around the x axis. The rotation angle is 0
            degrees at t=0 seconds and increases linearly to 90 degrees over the next
            4 seconds. Before and after this period, the angle is kept constant.

        """
        # create timestamps
        lcs_ref_time = None
        ref_time = None
        time_offset = 0
        if lcs_ref_sec is not None:
            lcs_ref_time = TS(year=2017, month=1, day=1, hour=12, second=lcs_ref_sec)
        if ref_sec is not None:
            ref_time = TS(year=2017, month=1, day=1, hour=12, second=ref_sec)
            if lcs_ref_sec is not None:
                time_offset += ref_sec - lcs_ref_sec

        # create expression
        expr = "a*t+b"
        param = dict(a=Q_([[1, 0, 0]], "mm/s"), b=Q_([1, 1, 1], "mm"))
        me = MathematicalExpression(expression=expr, parameters=param)

        # create orientation and time of LCS
        orientation = None
        lcs_time = None
        if time_dep_orientation:
            lcs_time = Q_([0, 4], "s")
            orientation = WXRotation.from_euler(
                "x", lcs_time.m * 22.5, degrees=True
            ).as_matrix()

        # create LCS
        lcs = LCS(
            orientation=orientation,
            coordinates=TimeSeries(data=me),
            time=lcs_time,
            time_ref=lcs_ref_time,
        )

        # interpolate
        time = Q_(seconds, "s")
        lcs_interp = lcs.interp_time(time, time_ref=ref_time)

        # check time
        assert lcs_interp.reference_time == ref_time
        assert np.allclose(lcs_interp.time_quantity, time)

        # check coordinates
        exp_vals = [[s + time_offset + 1, 1, 1] for s in seconds]
        assert isinstance(lcs_interp.coordinates, xr.DataArray)
        assert np.allclose(lcs_interp.coordinates, exp_vals)

        # check orientation
        seconds_offset = np.array(seconds) + time_offset

        if lcs_time is not None:
            angles = []
            for sec in seconds_offset:
                if sec <= lcs_time[0].m:
                    angles.append(0)
                elif sec >= lcs_time[1].m:
                    angles.append(90)
                else:
                    angles.append(22.5 * sec)
        else:
            angles = 0
        exp_orientations = WXRotation.from_euler("x", angles, degrees=True).as_matrix()
        assert np.allclose(exp_orientations, lcs_interp.orientation)

    # test_interp_time_exceptions ------------------------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        "time_ref_lcs, time, time_ref,  exception_type, test_name",
        [
            (TS("2020-02-02"), TDI([1]), None, TypeError, "# mixed ref. times #1"),
            (None, TDI([1]), TS("2020-02-02"), TypeError, "# mixed ref. times #2"),
            (TS("2020-02-02"), "no", TS("2020-02-02"), TypeError, "# wrong type #1"),
            (TS("2020-02-02"), TDI([1]), "no", TypeError, "# wrong type #2"),
        ],
        ids=get_test_name,
    )
    def test_interp_time_exceptions(
        time_ref_lcs, time, time_ref, exception_type, test_name
    ):
        """Test the exceptions of the 'reset_reference_time' method.

        Parameters
        ----------
        time_ref_lcs:
            Reference time of the LCS
        time:
            Time that is passed to interp_time
        time_ref:
            Reference time that is passed to interp_time
        exception_type:
            Expected exception type
        test_name:
            Name of the test

        """
        orientation = r_mat_z([1, 2, 3])
        coordinates = [[i, i, i] for i in range(3)]
        time_lcs = TDI([1, 2, 3], "D")

        lcs = tf.LocalCoordinateSystem(
            orientation, coordinates, time_lcs, time_ref=time_ref_lcs
        )

        with pytest.raises(exception_type):
            lcs.interp_time(time, time_ref)

    # test_addition --------------------------------------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        "lcs_lhs, lcs_rhs, orientation_exp, coordinates_exp, time_exp, time_ref_exp",
        [
            (  # 1 - both static
                LCS(r_mat_y(0.5), [1, 4, 2]),
                LCS(r_mat_z(0.5), [3, 7, 1]),
                [[0, -1, 0], [0, 0, 1], [-1, 0, 0]],
                [-1, 8, 3],
                None,
                None,
            ),
            (  # 2 - left system orientation time dependent
                LCS(
                    r_mat_z([0, 0.5, 1]),
                    [1, 4, 2],
                    TDI([1, 3, 5], "D"),
                    TS("2020-02-02"),
                ),
                LCS(r_mat_z(0.5), [3, 7, 1]),
                r_mat_z([0.5, 1, 1.5]),
                [[-1, 8, 3], [-1, 8, 3], [-1, 8, 3]],
                TDI([1, 3, 5], "D"),
                TS("2020-02-02"),
            ),
            (  # 3 - left system coordinates time dependent
                LCS(
                    r_mat_y(0.5),
                    [[3, 7, 1], [4, -2, 8], [-5, 3, -1]],
                    TDI([1, 3, 5], "D"),
                    TS("2020-02-02"),
                ),
                LCS(r_mat_z(0.5), [3, 7, 1]),
                [[[0, -1, 0], [0, 0, 1], [-1, 0, 0]] for _ in range(3)],
                [[-4, 10, 2], [5, 11, 9], [0, 2, 0]],
                TDI([1, 3, 5], "D"),
                TS("2020-02-02"),
            ),
            (  # 4 - right system orientation time dependent
                LCS(r_mat_z(0.5), [3, 7, 1]),
                LCS(
                    r_mat_z([0, 0.5, 1]),
                    [1, 4, 2],
                    TDI([1, 3, 5], "D"),
                    TS("2020-02-02"),
                ),
                r_mat_z([0.5, 1, 1.5]),
                [[4, 11, 3], [-6, 7, 3], [-2, -3, 3]],
                TDI([1, 3, 5], "D"),
                TS("2020-02-02"),
            ),
            (  # 5 - right system coordinates time dependent
                LCS(r_mat_z(0.5), [3, 7, 1]),
                LCS(
                    r_mat_z(0.5),
                    [[3, 7, 1], [4, -2, 8], [-5, 3, -1]],
                    TDI([1, 3, 5], "D"),
                    TS("2020-02-02"),
                ),
                r_mat_z(1),
                [[-4, 10, 2], [-3, 1, 9], [-12, 6, 0]],
                TDI([1, 3, 5], "D"),
                TS("2020-02-02"),
            ),
            (  # 6 - right system fully time dependent
                LCS(r_mat_z(0.5), [3, 7, 1]),
                LCS(
                    r_mat_z([0, 0.5, 1]),
                    [[3, 7, 1], [4, -2, 8], [-5, 3, -1]],
                    TDI([1, 3, 5], "D"),
                    TS("2020-02-02"),
                ),
                r_mat_z([0.5, 1, 1.5]),
                [[6, 14, 2], [-3, 1, 9], [-8, -4, 0]],
                TDI([1, 3, 5], "D"),
                TS("2020-02-02"),
            ),
            (  # 7 - both fully time dependent - same time and reference time
                LCS(
                    r_mat_z([1, 0, 0]),
                    [[4, 2, 5], [3, -3, 2], [1, 7, -9]],
                    TDI([1, 3, 5], "D"),
                    TS("2020-02-02"),
                ),
                LCS(
                    r_mat_z([0, 0.5, 1]),
                    [[3, 7, 1], [4, -2, 8], [-5, 3, -1]],
                    TDI([1, 3, 5], "D"),
                    TS("2020-02-02"),
                ),
                r_mat_z([1, 0.5, 1]),
                [[7, 9, 6], [7, 1, 10], [-6, -4, -10]],
                TDI([1, 3, 5], "D"),
                TS("2020-02-02"),
            ),
            (  # 8 - both fully time dependent - different time but same reference time
                LCS(
                    r_mat_z([1.5, 1.0, 0.75]),
                    [[4, 2, 5], [3, -3, 2], [1, 7, -9]],
                    TDI([2, 4, 6], "D"),
                    TS("2020-02-02"),
                ),
                LCS(
                    r_mat_z([0.75, 1.25, 0.75]),
                    [[3, 7, 1], [4, -2, 8], [-5, 3, -1]],
                    TDI([1, 3, 5], "D"),
                    TS("2020-02-02"),
                ),
                r_mat_z([0.5, 0.0, 1.5]),
                [[-0.5, 0.5, 9.5], [-3.5, 3.5, 5.5], [-10.6568542, -1.242640687, -10]],
                TDI([2, 4, 6], "D"),
                TS("2020-02-02"),
            ),
            (  # 9 - both fully time dependent - different time and reference time #1
                LCS(
                    r_mat_z([1.5, 1.0, 0.75]),
                    [[4, 2, 5], [3, -3, 2], [1, 7, -9]],
                    TDI([1, 3, 5], "D"),
                    TS("2020-02-03"),
                ),
                LCS(
                    r_mat_z([0.75, 1.25, 0.75]),
                    [[3, 7, 1], [4, -2, 8], [-5, 3, -1]],
                    TDI([1, 3, 5], "D"),
                    TS("2020-02-02"),
                ),
                r_mat_z([0.5, 0.0, 1.5]),
                [[-0.5, 0.5, 9.5], [-3.5, 3.5, 5.5], [-10.6568542, -1.242640687, -10]],
                TDI([2, 4, 6], "D"),
                TS("2020-02-02"),
            ),
            (  # 10 - both fully time dependent - different time and reference time #2
                LCS(
                    r_mat_z([1.5, 1.0, 0.75]),
                    [[4, 2, 5], [3, -3, 2], [1, 7, -9]],
                    TDI([3, 5, 7], "D"),
                    TS("2020-02-01"),
                ),
                LCS(
                    r_mat_z([0.75, 1.25, 0.75]),
                    [[3, 7, 1], [4, -2, 8], [-5, 3, -1]],
                    TDI([1, 3, 5], "D"),
                    TS("2020-02-02"),
                ),
                r_mat_z([0.5, 0.0, 1.5]),
                [[-0.5, 0.5, 9.5], [-3.5, 3.5, 5.5], [-10.6568542, -1.242640687, -10]],
                TDI([3, 5, 7], "D"),
                TS("2020-02-01"),
            ),
        ],
    )
    def test_addition(
        lcs_lhs, lcs_rhs, orientation_exp, coordinates_exp, time_exp, time_ref_exp
    ):
        """Test the addition of 2 coordinate systems.

        Parameters
        ----------
        lcs_lhs:
            Left hand side coordinate system
        lcs_rhs:
            Right hand side coordinate system
        orientation_exp:
            Expected orientations of the resulting coordinate system
        coordinates_exp:
            Expected coordinates of the resulting coordinate system
        time_exp:
            Expected time of the resulting coordinate system
        time_ref_exp:
            Expected reference time of the resulting coordinate system

        """
        check_coordinate_system(
            lcs_lhs + lcs_rhs,
            orientation_exp,
            coordinates_exp,
            True,
            time_exp,
            time_ref_exp,
        )

    # test_subtraction -----------------------------------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        "lcs_lhs, lcs_rhs, orientation_exp, coordinates_exp, time_exp, time_ref_exp",
        [
            (  # 1 - both static
                LCS([[0, -1, 0], [0, 0, 1], [-1, 0, 0]], [-1, 8, 3]),
                LCS(r_mat_z(0.5), [3, 7, 1]),
                r_mat_y(0.5),
                [1, 4, 2],
                None,
                None,
            ),
            (  # 2 - left system orientation time dependent
                LCS(
                    r_mat_z([0.5, 1, 1.5]),
                    [-1, 8, 3],
                    TDI([1, 3, 5], "D"),
                    TS("2020-02-02"),
                ),
                LCS(r_mat_z(0.5), [3, 7, 1]),
                r_mat_z([0, 0.5, 1]),
                [[1, 4, 2], [1, 4, 2], [1, 4, 2]],
                TDI([1, 3, 5], "D"),
                TS("2020-02-02"),
            ),
            (  # 3 - left system coordinates time dependent
                LCS(
                    [[0, -1, 0], [0, 0, 1], [-1, 0, 0]],
                    [[-4, 10, 2], [5, 11, 9], [0, 2, 0]],
                    TDI([1, 3, 5], "D"),
                    TS("2020-02-02"),
                ),
                LCS(r_mat_z(0.5), [3, 7, 1]),
                r_mat_y([0.5, 0.5, 0.5]),
                [[3, 7, 1], [4, -2, 8], [-5, 3, -1]],
                TDI([1, 3, 5], "D"),
                TS("2020-02-02"),
            ),
            (  # 4 - right system orientation time dependent
                LCS(r_mat_z(0.5), [3, 7, 1]),
                LCS(
                    r_mat_z([0, 0.5, 1]),
                    [1, 4, 2],
                    TDI([1, 3, 5], "D"),
                    TS("2020-02-02"),
                ),
                r_mat_z([0.5, 0, 1.5]),
                [[2, 3, -1], [3, -2, -1], [-2, -3, -1]],
                TDI([1, 3, 5], "D"),
                TS("2020-02-02"),
            ),
            (  # 5 - right system coordinates time dependent
                LCS(r_mat_z(0.5), [3, 7, 1]),
                LCS(
                    r_mat_z(0.5),
                    [[3, 7, 1], [4, -2, 8], [-5, 3, -1]],
                    TDI([1, 3, 5], "D"),
                    TS("2020-02-02"),
                ),
                r_mat_z(0),
                [[0, 0, 0], [9, 1, -7], [4, -8, 2]],
                TDI([1, 3, 5], "D"),
                TS("2020-02-02"),
            ),
            (  # 6 - right system fully time dependent
                LCS(r_mat_z(0.5), [3, 7, 1]),
                LCS(
                    r_mat_z([0, 0.5, 1]),
                    [[3, 7, 1], [4, -2, 8], [-5, 3, -1]],
                    TDI([1, 3, 5], "D"),
                    TS("2020-02-02"),
                ),
                r_mat_z([0.5, 0, 1.5]),
                [[0, 0, 0], [9, 1, -7], [-8, -4, 2]],
                TDI([1, 3, 5], "D"),
                TS("2020-02-02"),
            ),
            (  # 7 - both fully time dependent - same time and reference time
                LCS(
                    r_mat_z([1, 0.5, 1]),
                    [[7, 9, 6], [7, 1, 10], [-6, -4, -10]],
                    TDI([1, 3, 5], "D"),
                    TS("2020-02-02"),
                ),
                LCS(
                    r_mat_z([0, 0.5, 1]),
                    [[3, 7, 1], [4, -2, 8], [-5, 3, -1]],
                    TDI([1, 3, 5], "D"),
                    TS("2020-02-02"),
                ),
                r_mat_z([1, 0, 0]),
                [[4, 2, 5], [3, -3, 2], [1, 7, -9]],
                TDI([1, 3, 5], "D"),
                TS("2020-02-02"),
            ),
            (  # 8 - both fully time dependent - different time but same reference time
                LCS(
                    r_mat_z([1.5, 1.0, 0.75]),
                    [[4, 2, 5], [3, -3, 2], [1, 7, -9]],
                    TDI([2, 4, 6], "D"),
                    TS("2020-02-02"),
                ),
                LCS(
                    r_mat_z([1, 1.5, 1]),
                    [[3, 7, 1], [4, -2, 8], [-5, 3, -1]],
                    TDI([1, 3, 5], "D"),
                    TS("2020-02-02"),
                ),
                r_mat_z([0.25, 1.75, 1.75]),
                [[-3.7426406, 2.9142135, 0.5], [-3.5, 3.742640, -1.5], [-6, -4, -8.0]],
                TDI([2, 4, 6], "D"),
                TS("2020-02-02"),
            ),
            (  # 9 - both fully time dependent - different time and reference time #1
                LCS(
                    r_mat_z([1.5, 1.0, 0.75]),
                    [[4, 2, 5], [3, -3, 2], [1, 7, -9]],
                    TDI([1, 3, 5], "D"),
                    TS("2020-02-03"),
                ),
                LCS(
                    r_mat_z([1, 1.5, 1]),
                    [[3, 7, 1], [4, -2, 8], [-5, 3, -1]],
                    TDI([1, 3, 5], "D"),
                    TS("2020-02-02"),
                ),
                r_mat_z([0.25, 1.75, 1.75]),
                [[-3.7426406, 2.9142135, 0.5], [-3.5, 3.742640, -1.5], [-6, -4, -8.0]],
                TDI([2, 4, 6], "D"),
                TS("2020-02-02"),
            ),
            (  # 10 - both fully time dependent - different time and reference time #2
                LCS(
                    r_mat_z([1.5, 1.0, 0.75]),
                    [[4, 2, 5], [3, -3, 2], [1, 7, -9]],
                    TDI([3, 5, 7], "D"),
                    TS("2020-02-01"),
                ),
                LCS(
                    r_mat_z([1, 1.5, 1]),
                    [[3, 7, 1], [4, -2, 8], [-5, 3, -1]],
                    TDI([1, 3, 5], "D"),
                    TS("2020-02-02"),
                ),
                r_mat_z([0.25, 1.75, 1.75]),
                [[-3.7426406, 2.9142135, 0.5], [-3.5, 3.742640, -1.5], [-6, -4, -8.0]],
                TDI([3, 5, 7], "D"),
                TS("2020-02-01"),
            ),
        ],
    )
    def test_subtraction(
        lcs_lhs, lcs_rhs, orientation_exp, coordinates_exp, time_exp, time_ref_exp
    ):
        """Test the subtraction of 2 coordinate systems.

        Parameters
        ----------
        lcs_lhs:
            Left hand side coordinate system
        lcs_rhs:
            Right hand side coordinate system
        orientation_exp:
            Expected orientations of the resulting coordinate system
        coordinates_exp:
            Expected coordinates of the resulting coordinate system
        time_exp:
            Expected time of the resulting coordinate system
        time_ref_exp:
            Expected reference time of the resulting coordinate system

        """
        check_coordinate_system(
            lcs_lhs - lcs_rhs,
            orientation_exp,
            coordinates_exp,
            True,
            time_exp,
            time_ref_exp,
        )

    # test_comparison_coords_timeseries ------------------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        "kwargs_me_other_upd, kwargs_ts_other_upd, kwargs_other_upd, exp_result",
        [
            ({}, {}, {}, True),
            (dict(expression="2*a*t"), {}, {}, False),
            (dict(parameters=dict(a=Q_([[2, 0, 0]], "1/s"))), {}, {}, False),
            ({}, dict(data=Q_(np.ones((2, 3)), "mm"), time=Q_([1, 2], "s")), {}, False),
            ({}, {}, dict(orientation=[[0, -1, 0], [1, 0, 0], [0, 0, 1]]), False),
            ({}, {}, dict(time_ref=TS("11:12")), False),
        ],
    )
    def test_comparison_coords_timeseries(
        kwargs_me_other_upd: Dict,
        kwargs_ts_other_upd: Dict,
        kwargs_other_upd: Dict,
        exp_result: bool,
    ):
        """Test the comparison operator with a TimeSeries as coordinates.

        Parameters
        ----------
        kwargs_me_other_upd :
            Set of key word arguments that should be passed to the `__init__` method of
            the `MathematicalExpression` of the `TimeSeries` that will be passed to the
            second lcs. All missing kwargs will get default values.
        kwargs_ts_other_upd :
            Set of key word arguments that should be passed to the `__init__` method of
            the `TimeSeries` of the second lcs. All missing kwargs will get default
            values.
        kwargs_other_upd :
            Set of key word arguments that should be passed to the `__init__` method of
            the second lcs. All missing kwargs will get default values.
        exp_result :
            Expected result of the comparison

        """
        me = MathematicalExpression("a*t", dict(a=Q_([[1, 0, 0]], "1/s")))
        ts = TimeSeries(data=me)
        lcs = LCS(coordinates=ts)

        kwargs_me_other = dict(expression=me.expression, parameters=me.parameters)
        kwargs_me_other.update(kwargs_me_other_upd)
        me_other = MathematicalExpression(**kwargs_me_other)

        kwargs_ts_other = dict(data=me_other, time=None, interpolation=None)
        kwargs_ts_other.update(kwargs_ts_other_upd)
        ts_other = TimeSeries(**kwargs_ts_other)

        kwargs_other = dict(
            orientation=None, coordinates=ts_other, time=None, time_ref=None
        )
        kwargs_other.update(kwargs_other_upd)
        lcs_other = LCS(**kwargs_other)

        assert (lcs == lcs_other) == exp_result


def test_coordinate_system_init():
    """Check the __init__ method with and without time dependency."""
    # reference data
    time_0 = TDI([1, 3, 5], "s")
    time_1 = TDI([2, 4, 6], "s")

    orientation_fix = r_mat_z(1)
    orientation_tdp = r_mat_z([0, 0.25, 0.5])
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
    orientation_exp = r_mat_z([0, 0.125, 0.25, 0.375, 0.5, 0.5])
    check_coordinate_system(lcs, orientation_exp, coordinates_exp, True, time_exp)

    # matrix normalization ----------------------

    # no time dependency
    orientation_exp = r_mat_z(1 / 3)
    orientation_fix_2 = deepcopy(orientation_exp)
    orientation_fix_2[:, 0] *= 10
    orientation_fix_2[:, 1] *= 3
    orientation_fix_2[:, 2] *= 4

    lcs = tf.LocalCoordinateSystem(
        orientation=orientation_fix_2, coordinates=coordinates_fix
    )

    check_coordinate_system(lcs, orientation_exp, coordinates_fix, True)

    # time dependent
    orientation_exp = r_mat_z(1 / 3 * ut.to_float_array([1, 2, 4]))
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
    # xarray time is DatetimeIndex instead of TimedeltaIndex
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

    rot_mat_x = WXRotation.from_euler("x", angles_x).as_matrix()
    rot_mat_y = WXRotation.from_euler("y", angles_y).as_matrix()

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

    exp_orientation = r_mat_z(-1 / 4)
    exp_coordinates = [-np.sqrt(2), np.sqrt(2), -2]

    check_coordinate_system(lcs1_in_lcs0, exp_orientation, exp_coordinates, True)

    lcs0_in_lcs1_2 = lcs1_in_lcs0.invert()

    check_coordinate_system(
        lcs0_in_lcs1_2, lcs0_in_lcs1.orientation, lcs0_in_lcs1.coordinates, True
    )

    # time dependent ----------------------------
    time = TDI([1, 2, 3, 4], "s")
    orientation = r_mat_z([0, 0.5, 1, 0.5])
    coordinates = np.array([[2, 8, 7], [4, 9, 2], [0, 2, 1], [3, 1, 2]])

    lcs0_in_lcs1 = tf.LocalCoordinateSystem(
        orientation=orientation, coordinates=coordinates, time=time
    )

    lcs1_in_lcs0 = lcs0_in_lcs1.invert()
    orientation_exp = r_mat_z([0, 1.5, 1, 1.5])
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
    orientation = r_mat_z([0, 0.5, 1, 0.5])
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

CSM = tf.CoordinateSystemManager
LCS = tf.LocalCoordinateSystem


class TestCoordinateSystemManager:
    """Test the CoordinateSystemManager class."""

    @staticmethod
    @pytest.fixture
    def csm_fix():
        """Create default coordinate system fixture."""
        csm_default = CSM("root")
        lcs_1 = LCS(coordinates=[0, 1, 2])
        lcs_2 = LCS(coordinates=[0, -1, -2])
        lcs_3 = LCS(coordinates=[-1, -2, -3])
        lcs_4 = LCS(r_mat_y(1 / 2), [1, 2, 3])
        lcs_5 = LCS(r_mat_y(3 / 2), [2, 3, 1])
        csm_default.add_cs("lcs1", "root", lcs_1)
        csm_default.add_cs("lcs2", "root", lcs_2)
        csm_default.add_cs("lcs3", "lcs1", lcs_3)
        csm_default.add_cs("lcs4", "lcs1", lcs_4)
        csm_default.add_cs("lcs5", "lcs2", lcs_5)

        return csm_default

    @staticmethod
    @pytest.fixture()
    def list_of_csm_and_lcs_instances():
        """Get a list of LCS and CSM instances."""
        lcs = [LCS(coordinates=[i, 0, 0]) for i in range(11)]

        csm_0 = CSM("lcs0", "csm0")
        csm_0.add_cs("lcs1", "lcs0", lcs[1])
        csm_0.add_cs("lcs2", "lcs0", lcs[2])
        csm_0.add_cs("lcs3", "lcs2", lcs[3])

        csm_1 = CSM("lcs0", "csm1")
        csm_1.add_cs("lcs4", "lcs0", lcs[4])

        csm_2 = CSM("lcs5", "csm2")
        csm_2.add_cs("lcs3", "lcs5", lcs[5], lsc_child_in_parent=False)
        csm_2.add_cs("lcs6", "lcs5", lcs[6])

        csm_3 = CSM("lcs6", "csm3")
        csm_3.add_cs("lcs7", "lcs6", lcs[7])
        csm_3.add_cs("lcs8", "lcs6", lcs[8])

        csm_4 = CSM("lcs9", "csm4")
        csm_4.add_cs("lcs3", "lcs9", lcs[9], lsc_child_in_parent=False)

        csm_5 = CSM("lcs7", "csm5")
        csm_5.add_cs("lcs10", "lcs7", lcs[10])

        csm = [csm_0, csm_1, csm_2, csm_3, csm_4, csm_5]
        return [csm, lcs]

    # test_init ------------------------------------------------------------------------

    @staticmethod
    def test_init():
        """Test the init method of the coordinate system manager."""
        # default construction ----------------------
        csm = CSM(root_coordinate_system_name="root")
        assert csm.number_of_coordinate_systems == 1
        assert csm.number_of_neighbors("root") == 0

        # Exceptions---------------------------------
        # Invalid root system name
        with pytest.raises(Exception):
            CSM({})

    # test_add_coordinate_system -------------------------------------------------------

    # todo
    #  add time dependent systems. The problem is, that currently something messes
    #  up the comparison. The commented version of lcs_2 somehow switches the order of
    #  how 2 coordinates are stored in the Dataset. This lets the coordinate comparison
    #  fail.
    csm_acs = CSM("root")
    time = pd.DatetimeIndex(["2000-01-01", "2000-01-04"])
    # lcs_2_acs = LCS(coordinates=[[0, -1, -2], [8, 2, 7]], time=time)

    @pytest.mark.parametrize(
        "name , parent, lcs, child_in_parent, exp_num_cs",
        [
            ("lcs1", "root", LCS(coordinates=[0, 1, 2]), True, 2),
            ("lcs2", "root", LCS(coordinates=[0, -1, -2]), False, 3),
            ("lcs3", "lcs2", LCS(r_mat_y(1 / 2), [1, 2, 3]), True, 4),
            ("lcs3", "lcs2", LCS(coordinates=[-1, -2, -3]), True, 4),
            ("lcs2", "lcs3", LCS(coordinates=[-1, -2, -3]), False, 4),
            ("lcs2", "lcs3", LCS(coordinates=[-1, -2, -3]), True, 4),
            ("lcs4", "lcs2", LCS(coordinates=[0, 1, 2]), True, 5),
            ("lcs4", "lcs2", LCS(r_mat_y(1 / 2), [1, 2, 3]), True, 5),
            ("lcs5", "lcs1", LCS(r_mat_y(3 / 2), [2, 3, 1]), True, 6),
            (
                "lcs5",
                "lcs1",
                LCS(
                    None,
                    TimeSeries(MathematicalExpression("a*t", dict(a=Q_(1, "1/s")))),
                ),
                True,
                6,
            ),
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
            assert csm.get_cs(name, parent) == lcs
            if not isinstance(lcs.coordinates, TimeSeries):
                assert csm.get_cs(parent, name) == lcs.invert()
        else:
            if not isinstance(lcs.coordinates, TimeSeries):
                assert csm.get_cs(name, parent) == lcs.invert()
            assert csm.get_cs(parent, name) == lcs

    # test_add_cs_reference_time -------------------------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        "has_timestamp_csm, has_timestamp_lcs_1, has_timestamp_lcs_2, exp_exception",
        [
            (True, False, False, None),
            (True, True, False, None),
            (True, False, True, None),
            (True, True, True, None),
            (False, False, False, None),
            (False, True, False, Exception),
            (False, False, True, Exception),
            (False, True, True, None),
        ],
    )
    def test_add_cs_reference_time(
        has_timestamp_csm, has_timestamp_lcs_1, has_timestamp_lcs_2, exp_exception
    ):
        """Test if reference time issues are caught while adding new coordinate systems.

        See 'Notes' section of the add_cs method documentation.

        Parameters
        ----------
        has_timestamp_csm : bool
            Set to `True` if the CoordinateSystemManager should have a reference time.
        has_timestamp_lcs_1 : bool
            Set to `True` if the first added coordinate system should have a reference
            time.
        has_timestamp_lcs_2 : bool
            Set to `True` if the second added coordinate system should have a reference
            time.
        exp_exception : Any
            Pass the expected exception type if the test should raise. Otherwise set to
            `None`

        """
        timestamp_csm = None
        timestamp_lcs_1 = None
        timestamp_lcs_2 = None

        if has_timestamp_csm:
            timestamp_csm = pd.Timestamp("2000-01-01")
        if has_timestamp_lcs_1:
            timestamp_lcs_1 = pd.Timestamp("2000-01-02")
        if has_timestamp_lcs_2:
            timestamp_lcs_2 = pd.Timestamp("2000-01-03")
        csm = tf.CoordinateSystemManager("root", time_ref=timestamp_csm)
        lcs_1 = tf.LocalCoordinateSystem(
            coordinates=[[1, 2, 3], [3, 2, 1]],
            time=pd.TimedeltaIndex([1, 2]),
            time_ref=timestamp_lcs_1,
        )
        lcs_2 = tf.LocalCoordinateSystem(
            coordinates=[[1, 5, 3], [3, 5, 1]],
            time=pd.TimedeltaIndex([0, 2]),
            time_ref=timestamp_lcs_2,
        )

        csm.add_cs("lcs_1", "root", lcs_1)

        if exp_exception is not None:
            with pytest.raises(exp_exception):
                csm.add_cs("lcs_2", "root", lcs_2)
        else:
            csm.add_cs("lcs_2", "root", lcs_2)

    # test_add_coordinate_system_timeseries --------------------------------------------

    @staticmethod
    def test_add_coordinate_system_timeseries():
        """Test if adding an LCS with a `TimeSeries` as coordinates is possible."""
        csm = CSM("r")
        me = MathematicalExpression("a*t", dict(a=Q_([[1, 0, 0]], "1/s")))
        ts = TimeSeries(me)
        lcs = LCS(coordinates=ts)

        csm.add_cs("cs1", "r", lcs)

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

    @staticmethod
    @pytest.mark.parametrize(
        "csm_data, cs_data, merge_data, csm_diffs, cs_diffs, merge_diffs, exp_results",
        [
            (  # No diff in CSM
                [("root", "csm_root", "2000-05-26")],
                [],
                [],
                [],
                [],
                [],
                [True],
            ),
            (  # Diff in CSM root system
                [("root", "csm_root", "2000-05-26")],
                [],
                [],
                [(0, ("diff", "csm_root", "2000-05-26"))],
                [],
                [],
                [False],
            ),
            (  # Diff in CSM name
                [("root", "csm_root", "2000-05-26")],
                [],
                [],
                [(0, ("root", "csm_diff", "2000-05-26"))],
                [],
                [],
                [False],
            ),
            (  # Diff in CSM reference time
                [("root", "csm_root", "2000-05-26")],
                [],
                [],
                [(0, ("root", "csm_root", "2000-01-11"))],
                [],
                [],
                [False],
            ),
            (  # No diffs in CSM with coordinate systems
                [("root", "csm_root")],
                [(0, ("cs_1", "root")), (0, ("cs_2", "root"))],
                [],
                [],
                [],
                [],
                [True],
            ),
            (  # Different number of coordinate systems
                [("root", "csm_root"), ("root", "csm_root_2")],
                [(0, ("cs_1", "root")), (0, ("cs_2", "root"))],
                [],
                [],
                [(1, (1, ("cs_2", "root")))],
                [],
                [False, False],
            ),
            (  # different coordinate systems names
                [("root", "csm_root")],
                [(0, ("cs_1", "root")), (0, ("cs_2", "root"))],
                [],
                [],
                [(1, (0, ("cs_3", "root")))],
                [],
                [False],
            ),
            (  # different coordinate system references
                [("root", "csm_root")],
                [(0, ("cs_1", "root")), (0, ("cs_2", "root"))],
                [],
                [],
                [(1, (0, ("cs_2", "cs_1")))],
                [],
                [False],
            ),
            (  # no diffs in CSM with multiple subsystems
                [("root", "csm_root"), ("cs_1", "csm_1"), ("cs_2", "csm_2")],
                [(0, ("cs_1", "root")), (1, ("cs_3", "cs_1")), (2, ("cs_1", "cs_2"))],
                [(1, 0), (2, 0)],
                [],
                [],
                [],
                [True, True, True],
            ),
            (  # different merge order
                [("root", "csm_root"), ("cs_1", "csm_1"), ("cs_2", "csm_2")],
                [(0, ("cs_1", "root")), (1, ("cs_3", "cs_1")), (2, ("cs_1", "cs_2"))],
                [(1, 0), (2, 0)],
                [],
                [],
                [(0, (2, 0)), (1, (1, 0))],
                [True, True, True],
            ),
            (  # different number of subsystems
                [("root", "csm_root"), ("cs_1", "csm_1"), ("cs_2", "csm_2")],
                [(0, ("cs_1", "root")), (1, ("cs_3", "cs_1")), (2, ("cs_1", "cs_2"))],
                [(1, 0), (2, 0)],
                [],
                [],
                [(1, None)],
                [False, True, True],
            ),
            (  # different root system name of subsystem
                [("root", "csm_root"), ("cs_1", "csm_1"), ("cs_2", "csm_2")],
                [(0, ("cs_1", "root")), (1, ("cs_3", "cs_1")), (2, ("cs_1", "cs_2"))],
                [(1, 0), (2, 0)],
                [(2, ("diff", "csm_2"))],
                [(2, (2, ("cs_1", "diff")))],
                [],
                [False, True, False],
            ),
            (  # different subsystem name
                [("root", "csm_root"), ("cs_1", "csm_1"), ("cs_2", "csm_2")],
                [(0, ("cs_1", "root")), (1, ("cs_3", "cs_1")), (2, ("cs_1", "cs_2"))],
                [(1, 0), (2, 0)],
                [(1, ("cs_1", "diff"))],
                [],
                [],
                [False, False, True],
            ),
            (  # different subsystem reference time
                [("root", "csm_root"), ("cs_1", "csm_1"), ("cs_2", "csm_2")],
                [(0, ("cs_1", "root")), (1, ("cs_3", "cs_1")), (2, ("cs_1", "cs_2"))],
                [(1, 0), (2, 0)],
                [(1, ("cs_1", "csm_1", "2000-01-01"))],
                [],
                [],
                [False, False, True],
            ),
            (  # subsystem merged at different nodes
                [("root", "csm_root"), ("cs_1", "csm_1"), ("cs_2", "csm_2")],
                [(0, ("cs_1", "root")), (1, ("cs_3", "cs_1")), (2, ("cs_1", "cs_2"))],
                [(1, 0), (2, 0)],
                [],
                [(2, (2, ("root", "cs_2")))],
                [],
                [False, True, False],
            ),
            (  # subsystem lcs name different
                [("root", "csm_root"), ("cs_1", "csm_1"), ("cs_2", "csm_2")],
                [(0, ("cs_1", "root")), (1, ("cs_3", "cs_1")), (2, ("cs_1", "cs_2"))],
                [(1, 0), (2, 0)],
                [],
                [(1, (1, ("diff", "cs_1")))],
                [],
                [False, False, True],
            ),
            (  # subsystem lcs different
                [("root", "csm_root"), ("cs_1", "csm_1"), ("cs_2", "csm_2")],
                [(0, ("cs_1", "root")), (1, ("cs_3", "cs_1")), (2, ("cs_1", "cs_2"))],
                [(1, 0), (2, 0)],
                [],
                [(1, (1, ("cs_3", "cs_1", None, [1, 0, 0])))],
                [],
                [False, False, True],
            ),
            (  # no diffs in CSM with nested subsystems
                [("root", "csm_root"), ("cs_1", "csm_1"), ("cs_2", "csm_2")],
                [(0, ("cs_1", "root")), (1, ("cs_3", "cs_1")), (2, ("cs_1", "cs_2"))],
                [(2, 1), (1, 0)],
                [],
                [],
                [],
                [True, True, True],
            ),
            (  # nested vs. not nested
                [("root", "csm_root"), ("cs_1", "csm_1"), ("cs_2", "csm_2")],
                [(0, ("cs_1", "root")), (1, ("cs_3", "cs_1")), (2, ("cs_1", "cs_2"))],
                [(2, 1), (1, 0)],
                [],
                [],
                [(0, (1, 0)), (0, (2, 0))],
                [False, False, True],
            ),
            (  # nested system has different root system
                [("root", "csm_root"), ("cs_1", "csm_1"), ("cs_2", "csm_2")],
                [(0, ("cs_1", "root")), (1, ("cs_3", "cs_1")), (2, ("cs_1", "cs_2"))],
                [(2, 1), (1, 0)],
                [(2, ("cs_4", "csm_2"))],
                [(2, (2, ("cs_1", "cs_4")))],
                [],
                [False, False, False],
            ),
            (  # nested system has different name
                [("root", "csm_root"), ("cs_1", "csm_1"), ("cs_2", "csm_2")],
                [(0, ("cs_1", "root")), (1, ("cs_3", "cs_1")), (2, ("cs_1", "cs_2"))],
                [(2, 1), (1, 0)],
                [(2, ("cs_2", "diff"))],
                [],
                [],
                [False, False, False],
            ),
            (  # nested system has different reference time
                [("root", "csm_root"), ("cs_1", "csm_1"), ("cs_2", "csm_2")],
                [(0, ("cs_1", "root")), (1, ("cs_3", "cs_1")), (2, ("cs_1", "cs_2"))],
                [(2, 1), (1, 0)],
                [(2, ("cs_2", "csm_2", "2000-04-01"))],
                [],
                [],
                [False, False, False],
            ),
            (  # nested system has lcs with different name
                [("root", "csm_root"), ("cs_1", "csm_1"), ("cs_2", "csm_2")],
                [
                    (0, ("cs_1", "root")),
                    (1, ("cs_3", "cs_1")),
                    (2, ("cs_1", "cs_2")),
                    (2, ("cs_4", "cs_2")),
                ],
                [(2, 1), (1, 0)],
                [],
                [(3, (2, ("diff", "cs_2")))],
                [],
                [False, False, False],
            ),
            (  # nested system has lcs with different reference system
                [("root", "csm_root"), ("cs_1", "csm_1"), ("cs_2", "csm_2")],
                [
                    (0, ("cs_1", "root")),
                    (1, ("cs_3", "cs_1")),
                    (2, ("cs_1", "cs_2")),
                    (2, ("cs_4", "cs_2")),
                ],
                [(2, 1), (1, 0)],
                [],
                [(3, (2, ("cs_4", "cs_1")))],
                [],
                [False, False, False],
            ),
            (  # nested system has different lcs
                [("root", "csm_root"), ("cs_1", "csm_1"), ("cs_2", "csm_2")],
                [(0, ("cs_1", "root")), (1, ("cs_3", "cs_1")), (2, ("cs_1", "cs_2"))],
                [(2, 1), (1, 0)],
                [],
                [(2, (2, ("cs_1", "cs_2", None, [1, 0, 0])))],
                [],
                [False, False, False],
            ),
        ],
    )
    def test_comparison(
        csm_data, cs_data, merge_data, csm_diffs, cs_diffs, merge_diffs, exp_results
    ):
        """Test the `__eq__` function.

        The test creates one or more CSM instances, adds coordinate systems and merges
        them. Then a second set of instances is created using a modified copy of the
        data used to create the first set of CSM instances. Afterwards, all instances
        are compared using the `==` operator and the results are checked to match the
        expectation.

        Parameters
        ----------
        csm_data :
            A list containing the arguments that should be passed to the CSM
            constructor. For each list entry a CSM instance is generated
        cs_data :
            A list containing the data to create coordinate systems. Each entry is a
            tuple containing the list index of the target CSM instance and the
            arguments that should be passed to the ``create_cs`` method
        merge_data :
            A list of tuples. Each tuple consists of two indices. The first one is the
            index of the source CSM and the second one of the target CSM. If an entry
            is `None`, it is skipped and no merge operation is performed
        csm_diffs :
            A list of modifications that should be applied to the ``csm_data`` before
            creating the second set of CSM instances. Each entry is a tuple containing
            the index and new value of the data that should be modified.
        cs_diffs :
            A list of modifications that should be applied to the ``cs_data`` before
            creating the coordinate systems of the second set of CSM instances. Each
            entry is a tuple containing the index and new value of the data that should
            be modified.
        merge_diffs :
            A list of modifications that should be applied to the ``merge_data`` before
            merging the second set of CSM instances. Each entry is a tuple containing
            the index and new value of the data that should be modified.
        exp_results :
            A list containing the expected results of each instance comparison

        """
        # define support function
        def create_csm_list(csm_data_list, cs_data_list, merge_data_list):
            """Create a list of CSM instances."""
            csm_list = []
            csm_list = [tf.CoordinateSystemManager(*args) for args in csm_data_list]

            for data in cs_data_list:
                csm_list[data[0]].create_cs(*data[1])

            for merge in merge_data_list:
                if merge is not None:
                    csm_list[merge[1]].merge(csm_list[merge[0]])

            return csm_list

        # create diff inputs
        csm_data_diff = deepcopy(csm_data)
        for diff in csm_diffs:
            csm_data_diff[diff[0]] = diff[1]

        cs_data_diff = deepcopy(cs_data)
        for diff in cs_diffs:
            cs_data_diff[diff[0]] = diff[1]

        merge_data_diff = deepcopy(merge_data)
        for diff in merge_diffs:
            merge_data_diff[diff[0]] = diff[1]

        # create CSM instances
        csm_list_1 = create_csm_list(csm_data, cs_data, merge_data)
        csm_list_2 = create_csm_list(csm_data_diff, cs_data_diff, merge_data_diff)

        # test
        for i, _ in enumerate(csm_list_1):
            assert (csm_list_1[i] == csm_list_2[i]) is exp_results[i]
            assert (csm_list_1[i] != csm_list_2[i]) is not exp_results[i]

    # test_comparison_wrong_type -------------------------------------------------------

    @staticmethod
    def test_comparison_wrong_type():
        """Test the comparison with other types."""
        csm = tf.CoordinateSystemManager("root", "csm")
        assert (csm == 4) is False
        assert (csm != 4) is True

    # test_time_union ------------------------------------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        "csm_ref_time_day, lcs_times, lcs_ref_time_days, edges,"
        "exp_time, exp_ref_time_day",
        [
            # all systems are time dependent
            ("21", [[1, 5, 6], [3, 6, 9]], ["22", "21"], None, [2, 3, 6, 7, 9], "21"),
            ("21", [[1, 5, 6], [3, 6, 9]], ["22", None], None, [2, 3, 6, 7, 9], "21"),
            ("21", [[2, 6, 7], [3, 6, 9]], [None, None], None, [2, 3, 6, 7, 9], "21"),
            (None, [[1, 5, 6], [3, 6, 9]], ["22", "21"], None, [2, 3, 6, 7, 9], "21"),
            (None, [[1, 5, 6], [3, 6, 9]], [None, None], None, [1, 3, 5, 6, 9], None),
            ("21", [[3, 4], [6, 9], [4, 8]], ["22", "20", "21"], None, [4, 5, 8], "21"),
            ("21", [[3, 4], [5, 8], [4, 8]], ["22", None, None], None, [4, 5, 8], "21"),
            ("21", [[3, 4], [3, 8], [4, 8]], [None, None, None], None, [3, 4, 8], "21"),
            (None, [[3, 4], [6, 9], [4, 8]], ["22", "20", "21"], None, [4, 5, 8], "21"),
            (None, [[3, 4], [3, 8], [4, 8]], [None, None, None], None, [3, 4, 8], None),
            # Contains static systems
            ("21", [[3, 4], None, [4, 8]], ["22", None, "21"], None, [4, 5, 8], "21"),
            ("21", [[3, 4], None, [4, 8]], ["22", None, None], None, [4, 5, 8], "21"),
            ("21", [[3, 4], None, [4, 8]], [None, None, None], None, [3, 4, 8], "21"),
            (None, [[3, 4], None, [4, 8]], ["22", None, "21"], None, [4, 5, 8], "21"),
            (None, [[3, 4], None, [4, 8]], [None, None, None], None, [3, 4, 8], None),
            # include only specific edges
            ("21", [[3, 4], None, [4, 8]], ["22", None, "21"], [0, 1], [4, 5], "21"),
            ("21", [[3, 4], None, [4, 8]], ["22", None, None], [0, 1], [4, 5], "21"),
            ("21", [[3, 4], None, [4, 8]], [None, None, None], [0, 1], [3, 4], "21"),
            (None, [[3, 4], None, [4, 8]], ["22", None, "21"], [0, 1], [4, 5], "21"),
            (None, [[3, 4], None, [4, 8]], [None, None, None], [0, 1], [3, 4], None),
            ("21", [[3, 4], None, [4, 8]], ["22", None, "21"], [0, 2], [4, 5, 8], "21"),
            ("21", [[3, 4], None, [4, 8]], ["22", None, None], [0, 2], [4, 5, 8], "21"),
            ("21", [[3, 4], None, [4, 8]], [None, None, None], [0, 2], [3, 4, 8], "21"),
            (None, [[3, 4], None, [4, 8]], ["22", None, "21"], [0, 2], [4, 5, 8], "21"),
            (None, [[3, 4], None, [4, 8]], [None, None, None], [0, 2], [3, 4, 8], None),
            ("21", [[3, 4], None, [4, 8]], ["22", None, "21"], [1, 2], [4, 8], "21"),
            ("21", [[3, 4], None, [4, 8]], ["22", None, None], [1, 2], [4, 8], "21"),
            ("21", [[3, 4], None, [4, 8]], [None, None, None], [1, 2], [4, 8], "21"),
            (None, [[3, 4], None, [4, 8]], ["22", None, "21"], [1, 2], [4, 8], "21"),
            (None, [[3, 4], None, [4, 8]], [None, None, None], [1, 2], [4, 8], None),
        ],
    )
    def test_time_union(
        csm_ref_time_day,
        lcs_times,
        lcs_ref_time_days,
        edges,
        exp_time,
        exp_ref_time_day,
    ):
        """Test the time_union function of the CSM.

        Parameters
        ----------
        csm_ref_time_day : str
            An arbitrary day number string in the range [1, 31] or `None`. The value is
            used to create the reference timestamp of the CSM
        lcs_times : List
            A list containing an arbitrary number of time delta values (days) that are
            used to create a corresponding number of `LocalCoordinateSystem` instances
            which are added to the CSM. If a value is `None`, the generated coordinate
            system will be static
        lcs_ref_time_days : List
            A list where the values are either arbitrary day number strings in the range
            [1, 31] or `None`. Those values are used to create the reference timestamps
            for the coordinate systems of the CSM. The list must have the same length as
            the one passed to the ``lcs_times`` parameter
        edges : List
            A list that specifies the indices of the ``lcs_times`` parameter that should
            be considered in the time union. If `None` is passed, all are used. Note
            that the information is used to create the correct inputs to the
            ``time_union`` function and isn't passed directly.
        exp_time : List
            A list containing time delta values (days) that are used to generate the
            expected result data
        exp_ref_time_day : str
            An arbitrary day number string in the range [1, 31] or `None`. The value is
            used as reference time to create the expected result data. If it is set to
            `None`, the expected result data type is a `pandas.TimedeltaIndex` and a
            `pandas.DatetimeIndex` otherwise

        """
        # create full time data
        csm_time_ref = None
        if csm_ref_time_day is not None:
            csm_time_ref = f"2010-03-{csm_ref_time_day}"

        lcs_time_ref = [None for _ in range(len(lcs_times))]
        for i, _ in enumerate(lcs_times):
            if lcs_times[i] is not None:
                lcs_times[i] = pd.TimedeltaIndex(lcs_times[i], "D")
            if lcs_ref_time_days[i] is not None:
                lcs_time_ref[i] = pd.Timestamp(f"2010-03-{lcs_ref_time_days[i]}")

        # create coordinate systems
        lcs = []
        for i, _ in enumerate(lcs_times):
            if isinstance(lcs_times[i], pd.TimedeltaIndex):
                coordinates = [[j, j, j] for j in range(len(lcs_times[i]))]
            else:
                coordinates = [1, 2, 3]
            lcs += [
                tf.LocalCoordinateSystem(
                    None,
                    coordinates,
                    lcs_times[i],
                    lcs_time_ref[i],
                )
            ]

        # create CSM and add coordinate systems
        csm = tf.CoordinateSystemManager("root", "base", csm_time_ref)
        for i, lcs_ in enumerate(lcs):
            csm.add_cs(f"lcs_{i}", "root", lcs_)

        # create expected data type
        exp_time = pd.TimedeltaIndex(exp_time, "D")
        if exp_ref_time_day is not None:
            exp_time = pd.Timestamp(f"2010-03-{exp_ref_time_day}") + exp_time

        # create correct list of edges
        if edges is not None:
            for i, _ in enumerate(edges):
                edges[i] = ("root", f"lcs_{edges[i]}")

        # check time_union result
        assert np.all(csm.time_union(list_of_edges=edges) == exp_time)

    # test_time_union_time_series_coords -----------------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        " tdp_orientation, add_discrete_lcs, list_of_edges, exp_time",
        [
            (False, False, None, None),
            (True, False, None, [1, 2]),
            (False, True, None, [2, 3]),
            (True, True, None, [1, 2, 3]),
            (False, True, [("tdp", "base"), ("ts", "base")], [2, 3]),
            (False, True, [("tdp", "base"), ("base", "ts")], [2, 3]),
            (False, True, [("st", "base"), ("ts", "base")], None),
            (False, True, [("st", "base"), ("base", "ts")], None),
            (False, True, [("ts", "base")], None),
            (False, True, [("base", "ts")], None),
            (False, True, [("tdp", "base"), ("st", "base")], [2, 3]),
            (False, True, [("tdp", "base"), ("st", "base"), ("ts", "base")], [2, 3]),
            (False, True, [("tdp", "base"), ("st", "base"), ("base", "ts")], [2, 3]),
            (True, True, [("tdp", "base"), ("ts", "base")], [1, 2, 3]),
            (True, True, [("tdp", "base"), ("base", "ts")], [1, 2, 3]),
            (True, True, [("st", "base"), ("ts", "base")], [1, 2]),
            (True, True, [("st", "base"), ("base", "ts")], [1, 2]),
            (True, True, [("ts", "base")], [1, 2]),
            (True, True, [("base", "ts")], [1, 2]),
            (True, True, [("tdp", "base"), ("st", "base")], [2, 3]),
            (True, True, [("tdp", "base"), ("st", "base"), ("ts", "base")], [1, 2, 3]),
            (True, True, [("tdp", "base"), ("st", "base"), ("base", "ts")], [1, 2, 3]),
        ],
    )
    def test_time_union_time_series_coords(
        tdp_orientation, add_discrete_lcs, list_of_edges, exp_time
    ):
        """Test time_union with an lcs that has a `TimeSeries` as coordinates.

        Parameters
        ----------
        tdp_orientation :
            If `True`, the LCS with the `TimeSeries` also has discrete time dependent
            orientations
        add_discrete_lcs :
            If `True`, another time dependent system with discrete values is added to
            the CSM
        list_of_edges :
            A list of edges that should be passed to `time_union`
        exp_time :
            The expected time values (in seconds)

        """
        ts = TimeSeries(MathematicalExpression("a*t", dict(a=Q_([[1, 2, 3]], "mm/s"))))
        lcs_ts_orientation = None
        lcs_ts_time = None
        if tdp_orientation:
            lcs_ts_orientation = WXRotation.from_euler("x", [0, 2]).as_matrix()
            lcs_ts_time = Q_([1, 2], "s")

        csm = CSM("base")
        csm.create_cs("st", "base", coordinates=[2, 2, 2])
        csm.create_cs("ts", "base", lcs_ts_orientation, ts, lcs_ts_time)
        if add_discrete_lcs:
            csm.create_cs(
                "tdp", "base", coordinates=[[2, 4, 5], [2, 2, 2]], time=Q_([2, 3], "s")
            )

        if exp_time is not None:
            exp_time = TDI(exp_time, unit="s")
        assert np.all(exp_time == csm.time_union(list_of_edges))

    # test_get_local_coordinate_system_no_time_dep -------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        "system_name, reference_name, exp_orientation, exp_coordinates",
        [
            ("lcs_1", None, r_mat_z(0.5), [1, 2, 3]),
            ("lcs_2", None, r_mat_y(0.5), [3, -3, 1]),
            ("lcs_3", None, r_mat_x(0.5), [1, -1, 3]),
            ("lcs_3", "root", [[0, 1, 0], [0, 0, -1], [-1, 0, 0]], [6, -4, 0]),
            ("root", "lcs_3", [[0, 0, -1], [1, 0, 0], [0, -1, 0]], [0, -6, -4]),
            ("lcs_3", "lcs_1", [[0, 0, -1], [0, -1, 0], [-1, 0, 0]], [-6, -5, -3]),
            ("lcs_1", "lcs_3", [[0, 0, -1], [0, -1, 0], [-1, 0, 0]], [-3, -5, -6]),
        ],
    )
    def test_get_local_coordinate_system_no_time_dep(
        system_name, reference_name, exp_orientation, exp_coordinates
    ):
        """Test the ``get_cs`` function without time dependencies.

        Have a look into the tests setup section to see which coordinate systems are
        defined in the CSM.

        Parameters
        ----------
        system_name : str
            Name of the system that should be returned
        reference_name : str
            Name of the reference system
        exp_orientation : List or numpy.ndarray
            The expected orientation of the returned system
        exp_coordinates
            The expected coordinates of the returned system

        """
        # setup
        csm = tf.CoordinateSystemManager(root_coordinate_system_name="root")
        csm.create_cs("lcs_1", "root", r_mat_z(0.5), [1, 2, 3])
        csm.create_cs("lcs_2", "root", r_mat_y(0.5), [3, -3, 1])
        csm.create_cs("lcs_3", "lcs_2", r_mat_x(0.5), [1, -1, 3])

        check_coordinate_system(
            csm.get_cs(system_name, reference_name),
            exp_orientation,
            exp_coordinates,
            True,
        )

    # test_get_local_coordinate_system_time_dep -------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        "function_arguments, time_refs, exp_orientation, exp_coordinates,"
        "exp_time_data, exp_failure",
        [
            # get cs in its parent system - no reference times
            (
                ("cs_1",),
                [None, None, None, None],
                [np.eye(3) for _ in range(3)],
                [[i, 0, 0] for i in [0, 0.25, 1]],
                ([0, 3, 12], None),
                False,
            ),
            # get cs in its parent system - only CSM has reference time
            (
                ("cs_1",),
                ["2000-03-03", None, None, None],
                [np.eye(3) for _ in range(3)],
                [[i, 0, 0] for i in [0, 0.25, 1]],
                ([0, 3, 12], "2000-03-03"),
                False,
            ),
            # get cs in its parent system - only system has reference time
            (
                ("cs_1",),
                [None, "2000-03-03", "2000-03-03", "2000-03-03"],
                [np.eye(3) for _ in range(3)],
                [[i, 0, 0] for i in [0, 0.25, 1]],
                ([0, 3, 12], "2000-03-03"),
                False,
            ),
            # get cs in its parent system - function and CSM have reference times
            (
                ("cs_1", None, pd.TimedeltaIndex([6, 9, 18], "D"), "2000-03-10"),
                ["2000-03-16", None, None, None],
                [np.eye(3) for _ in range(3)],
                [[i, 0, 0] for i in [0, 0.25, 1]],
                ([6, 9, 18], "2000-03-10"),
                False,
            ),
            # get cs in its parent system - system and CSM have diff. reference times
            (
                ("cs_1",),
                ["2000-03-10", "2000-03-16", None, None],
                [np.eye(3) for _ in range(3)],
                [[i, 0, 0] for i in [0, 0.25, 1]],
                ([6, 9, 18], "2000-03-10"),
                False,
            ),
            # get transformed cs - no reference times
            (
                ("cs_3", "root"),
                [None, None, None, None],
                [np.eye(3) for _ in range(7)],
                [[1, 0, 0] for _ in range(7)],
                ([0, 3, 4, 6, 8, 9, 12], None),
                False,
            ),
            # get transformed cs - only CSM has reference time
            (
                ("cs_3", "root"),
                ["2000-03-10", None, None, None],
                [np.eye(3) for _ in range(7)],
                [[1, 0, 0] for _ in range(7)],
                ([0, 3, 4, 6, 8, 9, 12], "2000-03-10"),
                False,
            ),
            # get transformed cs - CSM and two systems have a reference time
            (
                ("cs_3", "root"),
                ["2000-03-10", "2000-03-04", None, "2000-03-16"],
                r_mat_x([0, 0, 0, 2 / 3, 1, 1, 1, 1, 0.5, 0]),
                [[i, 0, 0] for i in [1, 1.25, 1.5, 1.5, 1.5, 4 / 3, 1.25, 1, 1, 1]],
                ([-6, -3, 0, 4, 6, 8, 9, 12, 15, 18], "2000-03-10"),
                False,
            ),
            # get transformed cs - CSM and all systems have a reference time
            (
                ("cs_3", "root"),
                ["2000-03-08", "2000-03-04", "2000-03-10", "2000-03-16"],
                r_mat_x([0, 0, 0, 2 / 3, 1, 1, 1, 1, 0.5, 0]),
                [[i, 0, 0] for i in [1, 1.25, 1.5, 1.5, 1.5, 4 / 3, 1.25, 1, 1, 1]],
                ([-4, -1, 2, 6, 8, 10, 11, 14, 17, 20], "2000-03-08"),
                False,
            ),
            # get transformed cs - all systems have a reference time
            (
                ("cs_3", "root"),
                [None, "2000-03-04", "2000-03-10", "2000-03-16"],
                r_mat_x([0, 0, 0, 2 / 3, 1, 1, 1, 1, 0.5, 0]),
                [[i, 0, 0] for i in [1, 1.25, 1.5, 1.5, 1.5, 4 / 3, 1.25, 1, 1, 1]],
                ([0, 3, 6, 10, 12, 14, 15, 18, 21, 24], "2000-03-04"),
                False,
            ),
            # get transformed cs at specific times - all systems and CSM have a
            # reference time
            (
                ("cs_3", "root", pd.TimedeltaIndex([-4, 8, 20], "D")),
                ["2000-03-08", "2000-03-04", "2000-03-10", "2000-03-16"],
                r_mat_x([0, 1, 0]),
                [[i, 0, 0] for i in [1, 1.5, 1]],
                ([-4, 8, 20], "2000-03-08"),
                False,
            ),
            # get transformed cs at specific times - some systems, CSM and function
            # have a reference time
            (
                ("cs_3", "root", pd.TimedeltaIndex([-4, 8, 20], "D"), "2000-03-08"),
                ["2000-03-10", "2000-03-04", None, "2000-03-16"],
                r_mat_x([0, 1, 0]),
                [[i, 0, 0] for i in [1, 1.5, 1]],
                ([-4, 8, 20], "2000-03-08"),
                False,
            ),
            # get transformed cs at specific times - all systems, CSM and function
            # have a reference time
            (
                ("cs_3", "root", pd.TimedeltaIndex([-4, 8, 20], "D"), "2000-03-08"),
                ["2000-03-02", "2000-03-04", "2000-03-10", "2000-03-16"],
                r_mat_x([0, 1, 0]),
                [[i, 0, 0] for i in [1, 1.5, 1]],
                ([-4, 8, 20], "2000-03-08"),
                False,
            ),
            # get transformed cs at specific times - all systems, and the function
            # have a reference time
            (
                ("cs_3", "root", pd.TimedeltaIndex([-4, 8, 20], "D"), "2000-03-08"),
                [None, "2000-03-04", "2000-03-10", "2000-03-16"],
                r_mat_x([0, 1, 0]),
                [[i, 0, 0] for i in [1, 1.5, 1]],
                ([-4, 8, 20], "2000-03-08"),
                False,
            ),
            # get transformed cs at specific times - the function and the CSM have a
            # reference time
            (
                ("cs_4", "root", pd.TimedeltaIndex([0, 6, 12, 18], "D"), "2000-03-08"),
                ["2000-03-14", None, None, None],
                r_mat_x([0, 0, 1, 2]),
                [[0, 1, 0], [0, 1, 0], [0, -1, 0], [0, 1, 0]],
                ([0, 6, 12, 18], "2000-03-08"),
                False,
            ),
            # get transformed cs at times of another system - no reference times
            (
                ("cs_3", "root", "cs_1"),
                [None, None, None, None],
                [np.eye(3) for _ in range(3)],
                [[1, 0, 0] for _ in range(3)],
                ([0, 3, 12], None),
                False,
            ),
            # get transformed cs at specific times - no reference times
            (
                ("cs_4", "root", pd.TimedeltaIndex([0, 3, 6, 9, 12], "D")),
                [None, None, None, None],
                r_mat_x([0, 0.5, 1, 1.5, 2]),
                [[0, 1, 0], [0, 0, 1], [0, -1, 0], [0, 0, -1], [0, 1, 0]],
                ([0, 3, 6, 9, 12], None),
                False,
            ),
            # self referencing
            (
                ("cs_3", "cs_3"),
                [None, None, None, None],
                np.eye(3),
                [0, 0, 0],
                (None, None),
                False,
            ),
            # get transformed cs at specific times using a quantity - all systems,
            # CSM and function have a reference time
            (
                ("cs_3", "root", Q_([-4, 8, 20], "day"), "2000-03-08"),
                ["2000-03-02", "2000-03-04", "2000-03-10", "2000-03-16"],
                r_mat_x([0, 1, 0]),
                [[i, 0, 0] for i in [1, 1.5, 1]],
                ([-4, 8, 20], "2000-03-08"),
                False,
            ),
            # get transformed cs at specific times using a quantity - all systems and
            # CSM have a reference time
            (
                ("cs_3", "root", Q_([-4, 8, 20], "day")),
                ["2000-03-08", "2000-03-04", "2000-03-10", "2000-03-16"],
                r_mat_x([0, 1, 0]),
                [[i, 0, 0] for i in [1, 1.5, 1]],
                ([-4, 8, 20], "2000-03-08"),
                False,
            ),
            # get transformed cs at specific times using a DatetimeIndex - all systems,
            # CSM and function have a reference time
            (
                (
                    "cs_3",
                    "root",
                    pd.DatetimeIndex(["2000-03-04", "2000-03-16", "2000-03-28"]),
                    "2000-03-08",
                ),
                ["2000-03-02", "2000-03-04", "2000-03-10", "2000-03-16"],
                r_mat_x([0, 1, 0]),
                [[i, 0, 0] for i in [1, 1.5, 1]],
                ([-4, 8, 20], "2000-03-08"),
                False,
            ),
            # get transformed cs at specific times using a DatetimeIndex - all systems,
            # and the CSM have a reference time
            (
                (
                    "cs_3",
                    "root",
                    pd.DatetimeIndex(["2000-03-04", "2000-03-16", "2000-03-28"]),
                ),
                ["2000-03-08", "2000-03-04", "2000-03-10", "2000-03-16"],
                r_mat_x([0, 1, 0]),
                [[i, 0, 0] for i in [1, 1.5, 1]],
                ([-4, 8, 20], "2000-03-08"),
                False,
            ),
            # get transformed cs at specific times using a DatetimeIndex - all systems
            # have a reference time - this is a special case since the internally used
            # mechanism will use the first value of the DatetimeIndex as reference
            # value. Using Quantities or a TimedeltaIndex will cause an exception since
            # the reference time of the time delta is undefined.
            (
                (
                    "cs_3",
                    "root",
                    pd.DatetimeIndex(["2000-03-16", "2000-03-28", "2000-03-30"]),
                ),
                [None, "2000-03-04", "2000-03-10", "2000-03-16"],
                r_mat_x([1, 0, 0]),
                [[i, 0, 0] for i in [1.5, 1, 1]],
                ([0, 12, 14], "2000-03-16"),
                False,
            ),
            # should fail - if only the coordinate systems have a reference time,
            # passing just a time delta results in an undefined reference timestamp of
            # the resulting coordinate system
            (
                ("cs_3", "root", pd.TimedeltaIndex([0, 8, 20], "D")),
                [None, "2000-03-04", "2000-03-10", "2000-03-16"],
                None,
                None,
                (None, None),
                True,
            ),
            # should fail - if neither the CSM nor its attached coordinate systems have
            # a reference time, passing one to the function results in undefined
            # behavior
            (
                ("cs_3", "root", pd.TimedeltaIndex([0, 8, 20], "D"), "2000-03-16"),
                [None, None, None, None],
                None,
                None,
                (None, None),
                True,
            ),
        ],
    )
    def test_get_local_coordinate_system_time_dep(
        function_arguments,
        time_refs,
        exp_orientation,
        exp_coordinates,
        exp_time_data,
        exp_failure,
    ):
        """Test the ``get_cs`` function with time dependencies.

        The test setup is as follows:

        - 'cs_1' moves in 12 days 1 unit along the x-axis in positive direction.
          It starts at the origin and refers to the root system
        - 'cs_2' moves in 12 days 1 unit along the x-axis in negative direction.
          In the same time it positively rotates 360 degrees around the x-axis.
          It starts at the origin and refers to 'cs_1'
        - 'cs_3' rotates in 12 days 360 degrees negatively around the x-axis.
          It remains static at the coordinate [1, 0, 0] and refers to 'cs_2'
        - 'cs_4' remains static at the coordinates [0, 1, 0] of its reference system
          'cs_2'
        - initially and after their movements, all systems have the same orientation as
          their reference system

        In case all systems have the same reference time, the following behavior can be
        observed in the root system:

        - 'cs_1' moves as described before
        - 'cs_2' remains at the origin and rotates around the x-axis
        - 'cs_3' remains completely static at the coordinates [1, 0, 0]
        - 'cs_4' rotates around the x-axis with a fixed distance of 1 unit to the origin

        Have a look into the tests setup for further details.

        Parameters
        ----------
        function_arguments : Tuple
            A tuple of values that should be passed to the function
        time_refs : List(str)
            A list of date strings. The first entry is used as reference time of the
            CSM. The others are passed as reference times to the coordinate systems that
            have the same number as the list index in their name. For example: The
            second list value with index 1 belongs to 'cs_1'.
        exp_orientation : List or numpy.ndarray
            The expected orientation of the returned system
        exp_coordinates
            The expected coordinates of the returned system
        exp_time_data : Tuple(List(int), str)
            A tuple containing the expected time data of the returned coordinate system.
            The first value is a list of the expected time deltas and the second value
            is the expected reference time as date string.
        exp_failure : bool
            Set to `True` if the function call with the given parameters should raise an
            error

        """
        # setup -------------------------------------------
        # set reference times
        for i, _ in enumerate(time_refs):
            if time_refs[i] is not None:
                time_refs[i] = pd.Timestamp(time_refs[i])

        # moves in positive x-direction
        time_1 = TDI([0, 3, 12], "D")
        time_ref_1 = time_refs[1]
        orientation_1 = None
        coordinates_1 = [[i, 0, 0] for i in [0, 0.25, 1]]

        # moves in negative x-direction and rotates positively around the x-axis
        time_2 = TDI([0, 4, 8, 12], "D")
        time_ref_2 = time_refs[2]
        coordinates_2 = [[-i, 0, 0] for i in [0, 1 / 3, 2 / 3, 1]]
        orientation_2 = r_mat_x([0, 2 / 3, 4 / 3, 2])

        # rotates negatively around the x-axis
        time_3 = TDI([0, 3, 6, 9, 12], "D")
        time_ref_3 = time_refs[3]
        coordinates_3 = [1, 0, 0]
        orientation_3 = r_mat_x([0, -0.5, -1, -1.5, -2])

        # static
        time_4 = None
        time_ref_4 = None
        orientation_4 = None
        coordinates_4 = [0, 1, 0]

        csm = tf.CoordinateSystemManager("root", "CSM", time_refs[0])
        csm.create_cs("cs_1", "root", orientation_1, coordinates_1, time_1, time_ref_1)
        csm.create_cs("cs_2", "cs_1", orientation_2, coordinates_2, time_2, time_ref_2)
        csm.create_cs("cs_3", "cs_2", orientation_3, coordinates_3, time_3, time_ref_3)
        csm.create_cs("cs_4", "cs_2", orientation_4, coordinates_4, time_4, time_ref_4)

        if not exp_failure:
            # create expected time data
            exp_time = exp_time_data[0]
            if exp_time is not None:
                exp_time = pd.TimedeltaIndex(exp_time, "D")
            exp_time_ref = exp_time_data[1]
            if exp_time_ref is not None:
                exp_time_ref = pd.Timestamp(exp_time_ref)

            check_coordinate_system(
                csm.get_cs(*function_arguments),
                exp_orientation,
                exp_coordinates,
                True,
                exp_time,
                exp_time_ref,
            )
        else:
            with pytest.raises(Exception):
                csm.get_cs(*function_arguments)

    # test_get_local_coordinate_system_timeseries --------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        "lcs, in_lcs, exp_coords, exp_time, exp_angles",
        [
            ("r", "ts", [[0, -1, -1], [0, -2, -1]], [1, 2], [0, -90]),
            ("ts", "r", [[0, 1, 1], [-2, 0, 1]], [1, 2], [0, 90]),
            ("s", "trl", [[0, 0, 0], [-1, 0, 0]], [2, 3], [0, 0]),
            ("trl", "s", [[0, 0, 0], [1, 0, 0]], [2, 3], [0, 0]),
            ("s", "r", [[1, 1, 1], [-2, 1, 1]], [1, 2], [0, 90]),
            ("r", "s", [[-1, -1, -1], [-1, -2, -1]], [1, 2], [0, -90]),
            ("trl", "r", [[1, 1, 1], [-2, 1, 1], [-3, 2, 1]], [1, 2, 3], [0, 90, 90]),
        ],
    )
    def test_get_local_coordinate_system_timeseries(
        lcs, in_lcs, exp_coords, exp_time, exp_angles
    ):
        """Test the get_cs method with one lcs having a `TimeSeries` as coordinates.

        Parameters
        ----------
        lcs :
            The lcs that should be transformed
        in_lcs :
            The target lcs
        exp_coords :
            Expected coordinates
        exp_time :
            The expected time (in seconds)
        exp_angles :
            The expected rotation angles around the z-axis

        """
        me = MathematicalExpression("a*t", {"a": Q_([[0, 1, 0]], "mm/s")})
        ts = TimeSeries(me)
        rotation = WXRotation.from_euler("z", [0, 90], degrees=True).as_matrix()
        translation = [[1, 0, 0], [2, 0, 0]]
        exp_orient = WXRotation.from_euler("z", exp_angles, degrees=True).as_matrix()

        csm = CSM("r")
        csm.create_cs("rot", "r", rotation, [0, 0, 1], time=Q_([1, 2], "s"))
        csm.create_cs("ts", "rot", coordinates=ts)
        csm.create_cs("s", "ts", coordinates=[1, 0, 0])
        csm.create_cs("trl", "ts", coordinates=translation, time=Q_([2, 3], "s"))

        result = csm.get_cs(lcs, in_lcs)
        assert np.allclose(result.orientation, exp_orient)
        assert np.allclose(result.coordinates, exp_coords)
        assert np.allclose(result.time_quantity.m, exp_time)

    # test_get_local_coordinate_system_exceptions --------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        "function_arguments, exception_type, test_name",
        [
            (("not there", "root"), ValueError, "# system does not exist"),
            (("root", "not there"), ValueError, "# reference system does not exist"),
            (("not there", "not there"), ValueError, "# both systems do not exist"),
            (("not there", None), ValueError, "# root system has no reference"),
            (("cs_4", "root", "not there"), ValueError, "# ref. system does not exist"),
            (("cs_4", "root", "cs_1"), ValueError, "# ref. system is not time dep."),
            (("cs_4", "root", 1), TypeError, "# Invalid time type #1"),
            (("cs_4", "root", ["grr", "4", "af"]), Exception, "# Invalid time type #2"),
        ],
        ids=get_test_name,
    )
    def test_get_local_coordinate_system_exceptions(
        function_arguments, exception_type, test_name
    ):
        """Test the exceptions of the ``get_cs`` function.

        Parameters
        ----------
        function_arguments : Tuple
            A tuple of values that should be passed to the function
        exception_type :
            Expected exception type
        test_name : str
            Name of the testcase

        """
        # setup
        time_1 = TDI([0, 3], "D")
        time_2 = TDI([4, 7], "D")

        csm = tf.CoordinateSystemManager(root_coordinate_system_name="root")
        csm.create_cs("cs_1", "root", r_mat_z(0.5), [1, 2, 3])
        csm.create_cs("cs_2", "root", r_mat_y(0.5), [3, -3, 1])
        csm.create_cs("cs_3", "cs_2", r_mat_x(0.5), [1, -1, 3])
        csm.create_cs("cs_4", "cs_2", r_mat_x([0, 1]), [2, -1, 2], time=time_1)
        csm.create_cs("cs_5", "root", r_mat_y([0, 1]), [1, -7, 3], time=time_2)

        # test
        with pytest.raises(exception_type):
            csm.get_cs(*function_arguments)

    # test_get_cs_exception_timeseries -------------------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        "lcs, in_lcs, exp_exception",
        [
            ("trl1", "ts", True),
            ("ts", "trl1", False),
            ("s", "trl1", True),
            ("trl1", "s", True),
            ("trl1", "trl2", False),
            ("trl2", "trl1", False),
            ("r", "trl2", False),
            ("trl2", "r", False),
            ("s", "r", False),
            ("r", "s", False),
        ],
    )
    def test_get_cs_exception_timeseries(lcs, in_lcs, exp_exception):
        """Test exceptions of get_cs method if 1 lcs has a `TimeSeries` as coordinates.

        Parameters
        ----------
        lcs :
            The lcs that should be transformed
        in_lcs :
            The target lcs
        exp_exception :
            Set to `True` if the transformation should raise

        """
        me = MathematicalExpression("a*t", {"a": Q_([[0, 1, 0]], "mm/s")})
        ts = TimeSeries(me)
        translation = [[1, 0, 0], [2, 0, 0]]

        csm = CSM("r")
        csm.create_cs("trl1", "r", coordinates=translation, time=Q_([1, 2], "s"))
        csm.create_cs("ts", "trl1", coordinates=ts)
        csm.create_cs("s", "ts", coordinates=[1, 0, 0])
        csm.create_cs("trl2", "ts", coordinates=translation, time=Q_([2, 3], "s"))
        if exp_exception:
            with pytest.raises(Exception):
                csm.get_cs(lcs, in_lcs)
        else:
            csm.get_cs(lcs, in_lcs)

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
        csm_0_systems = csm_mg.coordinate_system_names
        assert np.all([f"lcs{i}" in csm_0_systems for i in range(len(lcs))])

        for i, cur_lcs in enumerate(lcs):
            child = f"lcs{i}"
            parent = csm_mg.get_parent_system_name(child)
            if i == 0:
                assert parent is None
                continue
            assert csm_mg.get_cs(child, parent) == cur_lcs
            assert csm_mg.get_cs(parent, child) == cur_lcs.invert()

    # test_merge_reference_times -------------------------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        "time_ref_day_parent, time_ref_day_sub, is_static_parent, is_static_sub,"
        "should_fail",
        [
            # both static
            (None, None, True, True, False),
            ("01", None, True, True, False),
            ("01", "01", True, True, False),
            ("01", "03", True, True, False),
            (None, "01", True, True, False),
            # sub static
            (None, None, False, True, False),
            ("01", None, False, True, False),
            ("01", "01", False, True, False),
            ("01", "03", False, True, False),
            (None, "01", False, True, False),
            # parent static
            (None, None, True, False, False),
            ("01", None, True, False, False),
            ("01", "01", True, False, False),
            ("01", "03", True, False, True),
            (None, "01", True, False, True),
            # both dynamic
            (None, None, False, False, False),
            ("01", None, False, False, False),
            ("01", "01", False, False, False),
            ("01", "03", False, False, True),
            (None, "01", False, False, True),
        ],
    )
    def test_merge_reference_times(
        time_ref_day_parent,
        time_ref_day_sub,
        is_static_parent,
        is_static_sub,
        should_fail,
    ):
        """Test if ``merge`` raises an error for invalid reference time combinations.

        Parameters
        ----------
        time_ref_day_parent : str
            `None` or day number of the parent systems reference timestamp
        time_ref_day_sub : str
            `None` or day number of the merged systems reference timestamp
        is_static_parent : bool
            `True` if the parent system should be static, `False` otherwise
        is_static_sub : bool
            `True` if the merged system should be static, `False` otherwise
        should_fail : bool
            `True` if the merge operation should fail. `False` otherwise

        """
        # setup
        lcs_static = tf.LocalCoordinateSystem(coordinates=[1, 1, 1])
        lcs_dynamic = tf.LocalCoordinateSystem(
            coordinates=[[0, 4, 2], [7, 2, 4]], time=TDI([4, 8], "D")
        )
        time_ref_parent = None
        if time_ref_day_parent is not None:
            time_ref_parent = f"2000-01-{time_ref_day_parent}"
        csm_parent = tf.CoordinateSystemManager(
            "root", "csm_parent", time_ref=time_ref_parent
        )
        if is_static_parent:
            csm_parent.add_cs("cs_1", "root", lcs_static)
        else:
            csm_parent.add_cs("cs_1", "root", lcs_dynamic)

        time_ref_sub = None
        if time_ref_day_sub is not None:
            time_ref_sub = f"2000-01-{time_ref_day_sub}"
        csm_sub = tf.CoordinateSystemManager("base", "csm_sub", time_ref=time_ref_sub)
        if is_static_sub:
            csm_sub.add_cs("cs_1", "base", lcs_static)
        else:
            csm_sub.add_cs("cs_1", "base", lcs_dynamic)

        # test
        if should_fail:
            with pytest.raises(Exception):
                csm_parent.merge(csm_sub)
        else:
            csm_parent.merge(csm_sub)

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
        subs = csm[0].subsystems

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
        subs = csm_mg.subsystems

        # checks ------------------------------------------
        assert len(subs) == 3

        assert subs[0] == csm[1]
        assert subs[1] == csm[4]
        assert subs[2] == csm_n2

        # get sub sub system ------------------------------
        sub_subs = subs[2].subsystems

        # check -------------------------------------------
        assert len(sub_subs) == 1

        assert sub_subs[0] == csm_n3

        # get sub sub sub systems -------------------------
        sub_sub_subs = sub_subs[0].subsystems

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

    @staticmethod
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
    @pytest.mark.slow
    def test_unmerge_merged_serially(list_of_csm_and_lcs_instances, additional_cs):
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
            lcs = LCS(coordinates=[count, count + 1, count + 2])
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

    @staticmethod
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
    @pytest.mark.slow
    def test_unmerge_merged_nested(list_of_csm_and_lcs_instances, additional_cs):
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
            lcs = LCS(coordinates=[count, count + 1, count + 2])
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

    @staticmethod
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
        list_of_csm_and_lcs_instances, name, subsystems_exp, num_cs_exp
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
            lcs = LCS(coordinates=[i, 2 * i, -i])
            csm_mg.add_cs(f"add{i}", f"lcs{target_system_index[i]}", lcs)

        # just to avoid useless tests (delete does nothing if the lcs doesn't exist)
        assert name in csm_mg.coordinate_system_names

        # delete coordinate system ------------------------
        csm_mg.delete_cs(name, True)

        # check -------------------------------------------
        assert csm_mg.number_of_subsystems == len(subsystems_exp)
        assert csm_mg.number_of_coordinate_systems == num_cs_exp

        for sub_exp in subsystems_exp:
            assert sub_exp in csm_mg.subsystem_names

    # test_delete_cs_with_nested_subsystems --------------------------------------------

    @staticmethod
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
        list_of_csm_and_lcs_instances, name, subsystems_exp, num_cs_exp
    ):
        """Test the delete_cs function with nested subsystems."""
        # setup -------------------------------------------
        csm = deepcopy(list_of_csm_and_lcs_instances[0])

        csm_mg = deepcopy(csm[0])

        csm_n3 = deepcopy(csm[3])
        csm_n3.add_cs("nes0", "lcs8", LCS(coordinates=[1, 2, 3]))
        csm_n3.merge(csm[5])
        csm_n2 = deepcopy(csm[2])
        csm_n2.add_cs("nes1", "lcs5", LCS(coordinates=[-1, -2, -3]))
        csm_n2.merge(csm_n3)
        csm_mg.merge(csm[1])
        csm_mg.merge(csm[4])
        csm_mg.merge(csm_n2)

        # add some additional coordinate systems
        target_system_indices = [0, 2, 5, 7, 10]
        for i, target_system_index in enumerate(target_system_indices):
            lcs = LCS(coordinates=[i, 2 * i, -i])
            csm_mg.add_cs(f"add{i}", f"lcs{target_system_index}", lcs)

        # just to avoid useless tests (delete does nothing if the lcs doesn't exist)
        assert name in csm_mg.coordinate_system_names

        # delete coordinate system ------------------------
        csm_mg.delete_cs(name, True)

        # check -------------------------------------------
        assert csm_mg.number_of_subsystems == len(subsystems_exp)
        assert csm_mg.number_of_coordinate_systems == num_cs_exp

        for sub_exp in subsystems_exp:
            assert sub_exp in csm_mg.subsystem_names

    # test_plot ------------------------------------------------------------------------

    @staticmethod
    def test_plot():
        """Test if the plot function runs without problems. Output is not checked."""
        csm_global = CSM("root", "global coordinate systems")
        csm_global.create_cs("specimen", "root", coordinates=[1, 2, 3])
        csm_global.create_cs("robot head", "root", coordinates=[4, 5, 6])

        csm_specimen = CSM("specimen", "specimen coordinate systems")
        csm_specimen.create_cs("thermocouple 1", "specimen", coordinates=[1, 1, 0])
        csm_specimen.create_cs("thermocouple 2", "specimen", coordinates=[1, 4, 0])

        csm_robot = CSM("robot head", "robot coordinate systems")
        csm_robot.create_cs("torch", "robot head", coordinates=[0, 0, -2])
        csm_robot.create_cs("mount point 1", "robot head", coordinates=[0, 1, -1])
        csm_robot.create_cs("mount point 2", "robot head", coordinates=[0, -1, -1])

        csm_scanner = CSM("scanner", "scanner coordinate systems")
        csm_scanner.create_cs("mount point 1", "scanner", coordinates=[0, 0, 2])

        csm_robot.merge(csm_scanner)
        csm_global.merge(csm_robot)
        csm_global.merge(csm_specimen)

        csm_global.plot_graph()

    # test_assign_and_get_data ---------------------------------------------------------

    @staticmethod
    def setup_csm_test_assign_data() -> tf.CoordinateSystemManager:
        """Get a predefined CSM instance.

        Returns
        -------
        weldx.transformations.CoordinateSystemManager :
            Predefined CSM instance.

        """
        # test setup
        lcs1_in_root = tf.LocalCoordinateSystem(
            tf.WXRotation.from_euler("z", np.pi / 2).as_matrix(), [1, 2, 3]
        )
        lcs2_in_root = tf.LocalCoordinateSystem(r_mat_y(0.5), [3, -3, 1])
        lcs3_in_lcs2 = tf.LocalCoordinateSystem(r_mat_x(0.5), [1, -1, 3])

        csm = tf.CoordinateSystemManager(root_coordinate_system_name="root")
        csm.add_cs("lcs_1", "root", lcs1_in_root)
        csm.add_cs("lcs_2", "root", lcs2_in_root)
        csm.add_cs("lcs_3", "lcs_2", lcs3_in_lcs2)

        return csm

    @pytest.mark.parametrize(
        "lcs_ref, data_name, data, lcs_out, exp",
        [
            (
                "lcs_3",
                "my_data",
                [[1, -3, -1], [2, 4, -1], [-1, 2, 3], [3, -4, 2]],
                None,
                [[1, -3, -1], [2, 4, -1], [-1, 2, 3], [3, -4, 2]],
            ),
            (
                "lcs_3",
                "my_data",
                [[1, -3, -1], [2, 4, -1], [-1, 2, 3], [3, -4, 2]],
                "lcs_3",
                [[1, -3, -1], [2, 4, -1], [-1, 2, 3], [3, -4, 2]],
            ),
            (
                "lcs_3",
                "my_data",
                [[1, -3, -1], [2, 4, -1], [-1, 2, 3], [3, -4, 2]],
                "lcs_1",
                [[-5, -2, -4], [-5, -9, -5], [-9, -7, -2], [-8, -1, -6]],
            ),
            (
                "lcs_3",
                "my_data",
                SpatialData([[1, -3, -1], [2, 4, -1], [-1, 2, 3], [3, -4, 2]]),
                "lcs_1",
                [[-5, -2, -4], [-5, -9, -5], [-9, -7, -2], [-8, -1, -6]],
            ),
            (
                "lcs_3",
                "my_data",
                xr.DataArray(
                    data=[[1, -3, -1], [2, 4, -1], [-1, 2, 3], [3, -4, 2]],
                    dims=["n", "c"],
                    coords={"c": ["x", "y", "z"]},
                ),
                "lcs_1",
                [[-5, -2, -4], [-5, -9, -5], [-9, -7, -2], [-8, -1, -6]],
            ),
        ],
    )
    def test_data_functions(self, lcs_ref, data_name, data, lcs_out, exp):
        """Test the `assign_data`, `has_data` and `get_data` functions.

        Parameters
        ----------
        lcs_ref : str
            Name of the data's reference system
        data_name : str
            Name of the data
        data :
            The data that should be assigned
        lcs_out : str
            Name of the target coordinate system
        exp: List[List[float]]
            Expected return values

        """
        csm = self.setup_csm_test_assign_data()

        assert csm.has_data(lcs_ref, data_name) is False
        assert data_name not in csm.data_names

        csm.assign_data(data, data_name, lcs_ref)

        assert data_name in csm.data_names
        assert csm.get_data_system_name(data_name) == lcs_ref
        for lcs in csm.coordinate_system_names:
            assert csm.has_data(lcs, data_name) == (lcs == lcs_ref)

        transformed_data = csm.get_data(data_name, lcs_out)
        if isinstance(transformed_data, SpatialData):
            transformed_data = transformed_data.coordinates.data
        else:
            transformed_data = transformed_data.data

        assert ut.matrix_is_close(transformed_data, exp)

    # test_assign_data_exceptions ------------------------------------------------------

    @pytest.mark.parametrize(
        "arguments, exception_type, test_name",
        [
            (([[1, 2, 3]], {"wrong"}, "root"), TypeError, "# invalid data name"),
            (([[1, 2, 3]], "data", "not there"), ValueError, "# system does not exist"),
            (([[1, 2, 3]], "some data", "root"), ValueError, "# name already taken 1"),
            (([[1, 2, 3]], "some data", "lcs_1"), ValueError, "# name already taken 2"),
        ],
    )
    def test_assign_data_exceptions(self, arguments, exception_type, test_name):
        """Test exceptions of the `assign_data` method.

        Parameters
        ----------
        arguments : Tuple
            A tuple of arguments that are passed to the function
        exception_type :
            The expected exception type
        test_name : str
            A string starting with an `#` that describes the test.

        """
        csm = self.setup_csm_test_assign_data()
        csm.assign_data([[1, 2, 3], [3, 2, 1]], "some data", "root")
        with pytest.raises(exception_type):
            csm.assign_data(*arguments)

    # test_has_data_exceptions ---------------------------------------------------------

    @pytest.mark.parametrize(
        "arguments, exception_type, test_name",
        [
            (("wrong", "not there"), KeyError, "# system does not exist"),
        ],
    )
    def test_has_data_exceptions(self, arguments, exception_type, test_name):
        """Test exceptions of the `has_data` method.

        Parameters
        ----------
        arguments : Tuple
            A tuple of arguments that are passed to the function
        exception_type :
            The expected exception type
        test_name : str
            A string starting with an `#` that describes the test.

        """
        csm = self.setup_csm_test_assign_data()
        csm.assign_data([[1, 2, 3], [3, 2, 1]], "some_data", "root")
        with pytest.raises(exception_type):
            csm.has_data(*arguments)

    # test_get_data_exceptions ---------------------------------------------------------

    @pytest.mark.parametrize(
        "arguments, exception_type, test_name",
        [
            (("some data", "not there"), ValueError, "# system does not exist"),
            (("not there", "root"), KeyError, "# data does not exist"),
        ],
    )
    def test_get_data_exceptions(self, arguments, exception_type, test_name):
        """Test exceptions of the `get_data` method.

        Parameters
        ----------
        arguments : Tuple
            A tuple of arguments that are passed to the function
        exception_type :
            The expected exception type
        test_name : str
            A string starting with an `#` that describes the test.

        """
        csm = self.setup_csm_test_assign_data()
        csm.assign_data([[1, 2, 3], [3, 2, 1]], "some data", "root")
        with pytest.raises(exception_type):
            csm.get_data(*arguments)


def test_relabel():
    """Test relabeling unmerged and merged CSM nodes.

    Test covers: relabeling of child system, relabeling root system, merge
    two systems after relabeling, make sure cannot relabel after merge.
    """
    csm1 = tf.CoordinateSystemManager("A")
    csm1.add_cs("B", "A", tf.LocalCoordinateSystem())

    csm2 = tf.CoordinateSystemManager("C")
    csm2.add_cs("D", "C", tf.LocalCoordinateSystem())

    csm1.relabel({"B": "X"})
    csm2.relabel({"C": "X"})

    assert "B" not in csm1.graph.nodes
    assert "X" in csm1.graph.nodes

    assert "C" not in csm2.graph.nodes
    assert "X" in csm2.graph.nodes
    assert csm2.root_system_name == "X"

    csm1.merge(csm2)
    for n in ["A", "D", "X"]:
        assert n in csm1.graph.nodes

    with pytest.raises(NotImplementedError):
        csm1.relabel({"A": "Z"})


def test_coordinate_system_manager_create_coordinate_system():
    """Test direct construction of coordinate systems in the coordinate system manager.

    Create multiple coordinate systems with all provided methods and check
    if they are constructed correctly.

    """
    angles_x = np.array([0.5, 1, 2, 2.5]) * np.pi / 2
    angles_y = np.array([1.5, 0, 1, 0.5]) * np.pi / 2
    angles = np.array([[*angles_y], [*angles_x]]).transpose()
    angles_deg = 180 / np.pi * angles

    rot_mat_x = WXRotation.from_euler("x", angles_x).as_matrix()
    rot_mat_y = WXRotation.from_euler("y", angles_y).as_matrix()

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
        csm.get_cs("lcs_init_default"),
        lcs_default.orientation,
        lcs_default.coordinates,
        True,
    )

    csm.create_cs("lcs_init_tdp", "root", orientations, coords, time)
    check_coordinate_system(
        csm.get_cs("lcs_init_tdp"),
        orientations,
        coords,
        True,
        time=time,
    )

    # from euler ------------------------------------------
    csm.create_cs_from_euler("lcs_euler_default", "root", "yx", angles[0])
    check_coordinate_system(
        csm.get_cs("lcs_euler_default"),
        orientations[0],
        lcs_default.coordinates,
        True,
    )

    csm.create_cs_from_euler(
        "lcs_euler_tdp", "root", "yx", angles_deg, True, coords, time
    )
    check_coordinate_system(
        csm.get_cs("lcs_euler_tdp"),
        orientations,
        coords,
        True,
        time=time,
    )

    # from xyz --------------------------------------------
    csm.create_cs_from_xyz("lcs_xyz_default", "root", vec_x[0], vec_y[0], vec_z[0])
    check_coordinate_system(
        csm.get_cs("lcs_xyz_default"),
        orientations[0],
        lcs_default.coordinates,
        True,
    )

    csm.create_cs_from_xyz("lcs_xyz_tdp", "root", vec_x, vec_y, vec_z, coords, time)
    check_coordinate_system(
        csm.get_cs("lcs_xyz_tdp"),
        orientations,
        coords,
        True,
        time=time,
    )

    # from xy and orientation -----------------------------
    csm.create_cs_from_xy_and_orientation("lcs_xyo_default", "root", vec_x[0], vec_y[0])
    check_coordinate_system(
        csm.get_cs("lcs_xyo_default"),
        orientations[0],
        lcs_default.coordinates,
        True,
    )

    csm.create_cs_from_xy_and_orientation(
        "lcs_xyo_tdp", "root", vec_x, vec_y, True, coords, time
    )
    check_coordinate_system(
        csm.get_cs("lcs_xyo_tdp"),
        orientations,
        coords,
        True,
        time=time,
    )

    # from xz and orientation -----------------------------
    csm.create_cs_from_xz_and_orientation("lcs_xzo_default", "root", vec_x[0], vec_z[0])
    check_coordinate_system(
        csm.get_cs("lcs_xzo_default"),
        orientations[0],
        lcs_default.coordinates,
        True,
    )

    csm.create_cs_from_xz_and_orientation(
        "lcs_xzo_tdp", "root", vec_x, vec_z, True, coords, time
    )
    check_coordinate_system(
        csm.get_cs("lcs_xzo_tdp"),
        orientations,
        coords,
        True,
        time=time,
    )

    # from yz and orientation -----------------------------
    csm.create_cs_from_yz_and_orientation("lcs_yzo_default", "root", vec_y[0], vec_z[0])
    check_coordinate_system(
        csm.get_cs("lcs_yzo_default"),
        orientations[0],
        lcs_default.coordinates,
        True,
    )

    csm.create_cs_from_yz_and_orientation(
        "lcs_yzo_tdp", "root", vec_y, vec_z, True, coords, time
    )
    check_coordinate_system(
        csm.get_cs("lcs_yzo_tdp"),
        orientations,
        coords,
        True,
        time=time,
    )


def test_coordinate_system_manager_interp_time():
    """Test the coordinate system managers interp_time functions."""
    # Setup -------------------------------------
    angles = ut.to_float_array([0, np.pi / 2, np.pi])
    orientation = WXRotation.from_euler("z", angles).as_matrix()
    coordinates = ut.to_float_array([[5, 0, 0], [1, 0, 0], [1, 4, 4]])

    time_0 = TDI([1, 4, 7], "D")
    time_1 = TDI([1, 5, 9], "D")
    time_2 = TDI([1, 6, 11], "D")

    lcs_3_in_lcs_2 = tf.LocalCoordinateSystem(
        orientation=WXRotation.from_euler("y", 1).as_matrix(), coordinates=[4, 2, 0]
    )

    csm = tf.CoordinateSystemManager("root")
    csm.create_cs("lcs_0", "root", orientation, coordinates, time_0)
    csm.create_cs("lcs_1", "lcs_0", orientation, coordinates, time_1)
    csm.create_cs("lcs_2", "root", orientation, coordinates, time_2)
    csm.create_cs(
        "lcs_3", "lcs_2", WXRotation.from_euler("y", 1).as_matrix(), [4, 2, 0]
    )

    # interp_time -------------------------------
    time_interp = TDI([1, 3, 5, 7, 9], "D")
    csm_interp = csm.interp_time(time_interp)

    assert np.all(csm_interp.time_union() == time_interp)

    assert np.all(csm_interp.get_cs("lcs_0").time == time_interp)
    assert np.all(csm_interp.get_cs("lcs_1").time == time_interp)
    assert np.all(csm_interp.get_cs("lcs_2").time == time_interp)
    assert csm_interp.get_cs("lcs_3").time is None

    coords_0_exp = [
        [5, 0, 0],
        [7 / 3, 0, 0],
        [1, 4 / 3, 4 / 3],
        [1, 4, 4],
        [1, 4, 4],
    ]
    orient_0_exp = r_mat_z([0, 1 / 3, 2 / 3, 1, 1])
    lcs_0_in_root_exp = tf.LocalCoordinateSystem(
        orient_0_exp, coords_0_exp, time_interp
    )

    coords_1_exp = [[5, 0, 0], [3, 0, 0], [1, 0, 0], [1, 2, 2], [1, 4, 4]]
    orient_1_exp = r_mat_z([0, 1 / 4, 1 / 2, 3 / 4, 1])
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
    orient_2_exp = r_mat_z([0, 1 / 5, 2 / 5, 3 / 5, 4 / 5])
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
            exp = csm.get_cs(cs_name, ps_name)
            exp_inv = csm.get_cs(ps_name, cs_name)

        lcs = csm_interp.get_cs(cs_name, ps_name)
        lcs_inv = csm_interp.get_cs(ps_name, cs_name)

        check_coordinate_systems_close(lcs, exp)
        check_coordinate_systems_close(lcs_inv, exp_inv)

    # specific coordinate system (single) -----------------
    time_interp_lcs0 = TDI([3, 5], "D")
    csm_interp_single = csm.interp_time(time_interp_lcs0, None, "lcs_0")

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
            exp = csm.get_cs(cs_name, ps_name)
            exp_inv = csm.get_cs(ps_name, cs_name)

        lcs = csm_interp_single.get_cs(cs_name, ps_name)
        lcs_inv = csm_interp_single.get_cs(ps_name, cs_name)

        check_coordinate_systems_close(lcs, exp)
        check_coordinate_systems_close(lcs_inv, exp_inv)

    # specific coordinate system (single, scalar timestamp) -----------------
    time_interp_lcs0 = TDI([3], "D")
    csm_interp_single = csm.interp_time(time_interp_lcs0, None, "lcs_0")

    coords_0_exp = np.array(coords_0_exp)[[0], :]  # modified in prev test !
    orient_0_exp = np.array(orient_0_exp)[[0], :]  # modified in prev test !
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
            exp = csm.get_cs(cs_name, ps_name)
            exp_inv = csm.get_cs(ps_name, cs_name)

        lcs = csm_interp_single.get_cs(cs_name, ps_name)
        lcs_inv = csm_interp_single.get_cs(ps_name, cs_name)

        check_coordinate_systems_close(lcs, exp)
        check_coordinate_systems_close(lcs_inv, exp_inv)

    # specific coordinate systems (multiple) --------------
    time_interp_multiple = TDI([5, 7, 9], "D")
    csm_interp_multiple = csm.interp_time(
        time_interp_multiple, None, ["lcs_1", "lcs_2"]
    )

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
            exp = csm.get_cs(cs_name, ps_name)
            exp_inv = csm.get_cs(ps_name, cs_name)

        lcs = csm_interp_multiple.get_cs(cs_name, ps_name)
        lcs_inv = csm_interp_multiple.get_cs(ps_name, cs_name)

        check_coordinate_systems_close(lcs, exp)
        check_coordinate_systems_close(lcs_inv, exp_inv)

        # Related to pull request #275. This assures that interp time works
        # correctly if some coordinate systems have no own reference time, but the CSM
        # does.
        lcs1 = tf.LocalCoordinateSystem(
            coordinates=[[1, 0, 0], [2, 0, 0]], time=pd.TimedeltaIndex([1, 2])
        )

        lcs2 = tf.LocalCoordinateSystem(
            coordinates=[[1, 0, 0], [2, 0, 0]],
            time=pd.TimedeltaIndex([1, 2]) + pd.Timestamp("2000-01-03"),
        )

        csm_1 = tf.CoordinateSystemManager("root", time_ref=pd.Timestamp("2000-01-01"))
        csm_1.add_cs("lcs2", "root", lcs2)

        csm_2 = tf.CoordinateSystemManager("root")
        csm_2.add_cs("lcs1", "root", lcs1)

        csm_1.merge(csm_2)

        csm_1.interp_time(csm_1.time_union())


def test_coordinate_system_manager_transform_data():
    """Test the coordinate system managers transform_data function."""
    # define some coordinate systems
    # TODO: test more unique rotations - not 90°
    lcs1_in_root = tf.LocalCoordinateSystem(r_mat_z(0.5), [1, 2, 3])
    lcs2_in_root = tf.LocalCoordinateSystem(r_mat_y(0.5), [3, -3, 1])
    lcs3_in_lcs2 = tf.LocalCoordinateSystem(r_mat_x(0.5), [1, -1, 3])

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
    # data_numpy_transformed = csm.transform_data(data_np[0, :], "lcs_3", "lcs_1")
    # assert ut.vector_is_close(data_numpy_transformed, data_exp[0, :])

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
