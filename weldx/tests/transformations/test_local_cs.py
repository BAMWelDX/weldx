"""Test the `LocalCoordinateSystem` class."""

from __future__ import annotations

import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
import pint
import pytest
import xarray as xr
from pandas import Timestamp as TS
from pandas import date_range
from pint import DimensionalityError

import weldx.transformations as tf
import weldx.util as ut
from weldx.constants import Q_
from weldx.core import MathematicalExpression, TimeSeries
from weldx.tests._helpers import get_test_name
from weldx.time import Time
from weldx.transformations import LocalCoordinateSystem as LCS
from weldx.transformations import WXRotation

from ._util import check_coordinate_system, check_cs_close, r_mat_y, r_mat_z

# test_init ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "orient, coords, time, time_ref, exception",
    [
        (None, [1, 2, 3], None, None, None),
        (None, Q_([1, 2, 3], "m"), None, None, None),
        (None, np.zeros((2, 3)), Q_([1, 2], "s"), None, None),
        (None, [[1, 2, 3], [4, 5, 6]], Q_([1, 2], "s"), None, None),
        (None, Q_([1, 2, 3], "s"), None, None, DimensionalityError),
    ],
)
@pytest.mark.parametrize("data_array_coords", [True, False])
def test_init(orient, coords, time, time_ref, exception, data_array_coords):
    """Test the ´__init__´ method."""
    if not isinstance(coords, pint.Quantity):
        coords = Q_(coords, "mm")
    if data_array_coords:
        coords = ut.xr_3d_vector(coords, time)

    if exception:
        with pytest.raises(exception):
            LCS(orient, coords, time, time_ref)
        return

    LCS(orient, coords, time, time_ref)


# test_init_time_formats ---------------------------------------------------------------

timestamp = TS("2000-01-01")
time_delta = pd.to_timedelta([0, 1, 2], "s")
time_quantity = Q_([0, 1, 2], "s")
date_time = date_range("2000-01-01", periods=3, freq="s")


@pytest.mark.parametrize(
    "time, time_ref, time_exp, time_ref_exp",
    [
        (time_delta, None, time_delta, None),
        (time_delta, timestamp, time_delta, timestamp),
        (time_quantity, None, time_delta, None),
        (time_quantity, timestamp, time_delta, timestamp),
        (date_time, None, time_delta, timestamp),
        (
            date_time,
            TS("1999-12-31"),
            pd.to_timedelta([86400, 86401, 86402], "s"),
            TS("1999-12-31"),
        ),
    ],
)
def test_init_time_formats(time, time_ref, time_exp, time_ref_exp):
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

    """
    # setup
    orientation = r_mat_z([0.5, 1.0, 1.5])
    coordinates = Q_([[1, 2, 3], [4, 5, 6], [7, 8, 9]], "mm")
    lcs = tf.LocalCoordinateSystem(orientation, coordinates, time, time_ref=time_ref)

    # check results

    assert np.all(lcs.time == Time(time_exp, time_ref_exp))
    assert lcs.reference_time == time_ref_exp


# test_time_warning --------------------------------------------------------------------


@pytest.mark.parametrize(
    "coordinates, orientation, time, warning",
    [
        (
            Q_(np.zeros(3), "mm"),
            np.eye(3, 3),
            pd.to_timedelta([0, 2], "s"),
            UserWarning,
        ),
        (Q_(np.zeros((2, 3)), "mm"), np.eye(3, 3), pd.to_timedelta([0, 2], "s"), None),
        (Q_(np.zeros(3), "mm"), np.eye(3, 3), None, None),
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

    def _call():
        LCS(coordinates=coordinates, orientation=orientation, time=time)

    if warning is not None:  # pytest.warns does not allow passing None
        with pytest.warns(warning):
            _call()
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error", category=UserWarning)
            _call()


# test_init_time_dsx -------------------------------------------------------------------


@pytest.mark.parametrize(
    "time_o,  time_c,  time_exp",
    [
        (
            pd.to_timedelta([0, 1, 2], "s"),
            pd.to_timedelta([0, 1, 2], "s"),
            pd.to_timedelta([0, 1, 2], "s"),
        ),
        (
            pd.to_timedelta([0, 2, 4], "s"),
            pd.to_timedelta([1, 3, 5], "s"),
            pd.to_timedelta([0, 1, 2, 3, 4, 5], "s"),
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
    coordinates = Q_([[i, i, i] for i in range(len(time_o))], "mm")

    dax_o = ut.xr_3d_matrix(orientations, time_o)
    dax_c = ut.xr_3d_vector(coordinates, time_c)

    lcs = tf.LocalCoordinateSystem(dax_o, dax_c, time_ref=time_ref)

    # check results

    assert np.all(lcs.time == Time(time_exp, time_ref))
    assert lcs.reference_time == time_ref


# test_init_expr_time_series_as_coord --------------------------------------------------


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
        expression="a*t+b",
        parameters=dict(a=Q_([1, 0, 0], "m/s"), b=Q_([1, 2, 3], "m")),
    )

    ts_coord = TimeSeries(data=coordinates)
    orientation = None
    if angles is not None:
        orientation = WXRotation.from_euler("x", angles, degrees=True).as_matrix()
    if orientation is None and time is not None:
        with pytest.warns(UserWarning):
            lcs = LCS(orientation, ts_coord, time, time_ref)
    else:
        lcs = LCS(orientation, ts_coord, time, time_ref)
    assert lcs.has_reference_time == (time_ref is not None)


# test_init_discrete_time_series_as_coord ----------------------------------------------


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

    assert np.allclose(lcs.coordinates.data, data)
    if len(time) == 1:
        assert lcs.time is None
    else:
        assert np.all(lcs.time.as_quantity() == time)


# test_from_axis_vectors ---------------------------------------------------------------


@pytest.mark.parametrize("time_dep_orient", [True, False])
@pytest.mark.parametrize("time_dep_coord", [True, False])
@pytest.mark.parametrize("has_time_ref", [True, False])
def test_from_axis_vectors(
    time_dep_orient: bool, time_dep_coord: bool, has_time_ref: bool
):
    """Test the ``from_axis_vectors`` factory method."""
    if has_time_ref and not (time_dep_coord or time_dep_orient):
        return

    t = ["1s", "2s", "3s", "4s"] if time_dep_orient or time_dep_coord else None
    time_ref = "2011-07-22" if has_time_ref else None
    angles = [[30, 45, 60], [40, 35, 80], [1, 33, 7], [90, 180, 270]]
    o = WXRotation.from_euler("xyz", angles, degrees=True).as_matrix()
    c = Q_([[-1, 3, 2], [4, 2, 4], [5, 1, 2], [3, 3, 3]], "mm")

    if not time_dep_orient:
        o = o[0]
    if not time_dep_coord:
        c = c[0]

    x = o[..., 0] * 2
    y = o[..., 1] * 5
    z = o[..., 2] * 3
    kwargs = dict(coordinates=c, time=t, time_ref=time_ref)

    ref = LCS(o, c, t, time_ref)

    check_cs_close(LCS.from_axis_vectors(x, y, z, **kwargs), ref)
    check_cs_close(LCS.from_axis_vectors(x=x, y=y, **kwargs), ref)
    check_cs_close(LCS.from_axis_vectors(y=y, z=z, **kwargs), ref)
    check_cs_close(LCS.from_axis_vectors(x=x, z=z, **kwargs), ref)


# test_from_axis_vectors_exceptions ----------------------------------------------------


@pytest.mark.parametrize(
    "kwargs,  exception_type, test_name",
    [
        (dict(x=[1, 0, 0], y=[0, 1, 0], z=[0, 1, 1]), ValueError, "# not ortho"),
        (dict(x=[1, 0, 0], y=[1, 0, 0]), ValueError, "# not ortho 2"),
    ],
    ids=get_test_name,
)
def test_from_axis_vectors_exceptions(kwargs, exception_type, test_name):
    """Test the exceptions of the ``from_axis_vectors`` factory method."""
    with pytest.raises(exception_type):
        LCS.from_axis_vectors(**kwargs)


# test_reset_reference_time ------------------------------------------------------------


@pytest.mark.parametrize(
    "time, time_ref, time_ref_new, time_exp",
    [
        (
            pd.to_timedelta([1, 2, 3], "D"),
            TS("2020-02-02"),
            TS("2020-02-01"),
            pd.to_timedelta([2, 3, 4], "D"),
        ),
        (
            pd.to_timedelta([1, 2, 3], "D"),
            TS("2020-02-02"),
            "2020-02-01",
            pd.to_timedelta([2, 3, 4], "D"),
        ),
        (
            pd.to_timedelta([1, 2, 3], "D"),
            None,
            "2020-02-01",
            pd.to_timedelta([1, 2, 3], "D"),
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
    coordinates = Q_([[i, i, i] for i in range(3)], "mm")
    lcs = tf.LocalCoordinateSystem(orientation, coordinates, time, time_ref=time_ref)

    lcs.reset_reference_time(time_ref_new)

    # check results
    assert np.all(lcs.time == Time(time_exp, time_ref_new))
    assert lcs.reference_time == TS(time_ref_new)


# test_reset_reference_time_exceptions -------------------------------------------------


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
    coordinates = Q_([[i, i, i] for i in range(3)], "mm")
    time = pd.to_timedelta([1, 2, 3], "D")

    lcs = tf.LocalCoordinateSystem(orientation, coordinates, time, time_ref=time_ref)

    with pytest.raises(exception_type):
        lcs.reset_reference_time(time_ref_new)


# test_interp_time_discrete ------------------------------------------------------------


@pytest.mark.parametrize(
    "time_ref_lcs, time,time_ref, orientation_exp, coordinates_exp",
    [
        (  # broadcast left
            TS("2020-02-10"),
            pd.to_timedelta([1, 2, 14], "D"),
            TS("2020-02-10"),
            r_mat_z([0, 0, 0.5]),
            np.array([[2, 8, 7], [2, 8, 7], [4, 9, 2]]),
        ),
        (  # broadcast right
            TS("2020-02-10"),
            pd.to_timedelta([14, 29, 30], "D"),
            TS("2020-02-10"),
            r_mat_z([0.5, 0.5, 0.5]),
            np.array([[4, 9, 2], [3, 1, 2], [3, 1, 2]]),
        ),
        (  # pure interpolation
            TS("2020-02-10"),
            pd.to_timedelta([11, 14, 17, 20], "D"),
            TS("2020-02-10"),
            r_mat_z([0.125, 0.5, 0.875, 0.75]),
            np.array([[2.5, 8.25, 5.75], [4, 9, 2], [1, 3.75, 1.25], [1.5, 1.5, 1.5]]),
        ),
        (  # mixed
            TS("2020-02-10"),
            pd.to_timedelta([6, 12, 18, 24, 32], "D"),
            TS("2020-02-10"),
            r_mat_z([0, 0.25, 1, 0.5, 0.5]),
            np.array([[2, 8, 7], [3, 8.5, 4.5], [0, 2, 1], [3, 1, 2], [3, 1, 2]]),
        ),
        (  # different reference times
            TS("2020-02-10"),
            pd.to_timedelta([8, 14, 20, 26, 34], "D"),
            TS("2020-02-08"),
            r_mat_z([0, 0.25, 1, 0.5, 0.5]),
            np.array([[2, 8, 7], [3, 8.5, 4.5], [0, 2, 1], [3, 1, 2], [3, 1, 2]]),
        ),
        (  # no reference time
            None,
            pd.to_timedelta([6, 12, 18, 24, 32], "D"),
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
    coordinates_exp = Q_(coordinates_exp, "mm")
    # setup
    lcs = tf.LocalCoordinateSystem(
        orientation=r_mat_z([0, 0.5, 1, 0.5]),
        coordinates=Q_([[2, 8, 7], [4, 9, 2], [0, 2, 1], [3, 1, 2]], "mm"),
        time=pd.to_timedelta([10, 14, 18, 22], "D"),
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


# test_interp_time_discrete_outside_value_range ----------------------------------------


@pytest.mark.parametrize("time_dep_coords", [True, False])
@pytest.mark.parametrize("time_dep_orient", [True, False])
@pytest.mark.parametrize("all_less", [True, False])
def test_issue_289_interp_outside_time_range(
    time_dep_orient: bool, time_dep_coords: bool, all_less: bool
):
    """Test if ``interp_time`` if all interp. values are outside the value range.

    In this case it should always return a static system.

    Parameters
    ----------
    time_dep_orient :
        If `True`, the orientation is time dependent
    time_dep_coords :
        If `True`, the coordinates are time dependent
    all_less :
        If `True`, all interpolation values are less than the time values of the
        LCS. Otherwise, all values are greater.

    """
    angles = [45, 135] if time_dep_orient else 135
    orientation = WXRotation.from_euler("x", angles, degrees=True).as_matrix()
    coordinates = Q_([[0, 0, 0], [1, 1, 1]] if time_dep_coords else [1, 1, 1], "mm")
    if time_dep_coords or time_dep_orient:
        time = ["5s", "6s"] if all_less else ["0s", "1s"]
    else:
        time = None

    lcs = LCS(orientation, coordinates, time)
    lcs_interp = lcs.interp_time(["2s", "3s", "4s"])

    exp_angle = 45 if time_dep_orient and all_less else 135
    exp_orient = WXRotation.from_euler("x", exp_angle, degrees=True).as_matrix()
    exp_coords = Q_([0, 0, 0] if time_dep_coords and all_less else [1, 1, 1], "mm")

    assert lcs_interp.is_time_dependent is False
    assert lcs_interp.time is None
    assert lcs_interp.coordinates.data.shape == (3,)
    assert lcs_interp.orientation.data.shape == (3, 3)
    assert np.all(lcs_interp.coordinates.data == exp_coords)
    assert np.all(lcs_interp.orientation.data == exp_orient)


# test_interp_time_discrete_single_time ------------------------------------------------


def test_interp_time_discrete_single_time():
    """Test that single value interpolation results in a static system."""
    orientation = WXRotation.from_euler("x", [45, 135], degrees=True).as_matrix()
    coordinates = Q_([[0, 0, 0], [2, 2, 2]], "mm")
    time = ["1s", "3s"]
    lcs = LCS(orientation, coordinates, time)

    exp_coords = Q_([1, 1, 1], "mm")
    exp_orient = WXRotation.from_euler("x", 90, degrees=True).as_matrix()

    lcs_interp = lcs.interp_time("2s")
    assert lcs_interp.is_time_dependent is False
    assert lcs_interp.time.equals(Time("2s"))
    assert lcs_interp.coordinates.data.shape == (3,)
    assert lcs_interp.orientation.data.shape == (3, 3)
    assert np.all(lcs_interp.coordinates.data == exp_coords)
    assert np.allclose(lcs_interp.orientation.data, exp_orient)


# test_interp_time_discrete_outside_value_range_both_sides -----------------------------


def test_interp_time_discrete_outside_value_range_both_sides():
    """Test the interpolation is all values are outside of the LCS time range.

    In this special case there is an overlap of the time ranges and we need to
    ensure that the algorithm does not create a static system as it should if there
    is no overlap.

    """
    orientation = WXRotation.from_euler("x", [45, 135], degrees=True).as_matrix()
    coordinates = Q_([[0, 0, 0], [2, 2, 2]], "mm")
    time = ["2s", "3s"]
    lcs = LCS(orientation, coordinates, time)

    lcs_interp = lcs.interp_time(["1s", "4s"])

    assert np.all(lcs_interp.time == ["1s", "4s"])
    assert np.all(lcs_interp.coordinates.data == lcs.coordinates.data)
    assert np.allclose(lcs_interp.orientation.data, lcs.orientation.data)


# test_interp_time_timeseries_as_coords ------------------------------------------------


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
@pytest.mark.filterwarnings("ignore:Provided time is dropped")
def test_interp_time_timeseries_as_coords(
    seconds: list[float],
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
        The seconds of the reference time, that will be passed to the created LCS. If
        `None`, no reference time will be passed
    ref_sec :
        The seconds of the reference time, that will be passed to `interp_time`. If
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
    param = dict(a=Q_([1, 0, 0], "mm/s"), b=Q_([1, 1, 1], "mm"))
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
    assert np.all(lcs_interp.time == Time(time, ref_time))

    # check coordinates
    exp_vals = Q_([[s + time_offset + 1, 1, 1] for s in seconds], "mm")
    assert isinstance(lcs_interp.coordinates, xr.DataArray)
    assert np.allclose(lcs_interp.coordinates.data, exp_vals)

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


# test_interp_time_exceptions ----------------------------------------------------------


@pytest.mark.parametrize(
    "time_ref_lcs, time, time_ref,  exception_type, test_name",
    [
        (
            TS("2020-02-02"),
            pd.to_timedelta([1]),
            None,
            TypeError,
            "# mixed ref. times #1",
        ),
        (
            None,
            pd.to_timedelta([1]),
            TS("2020-02-02"),
            TypeError,
            "# mixed ref. times #2",
        ),
        (TS("2020-02-02"), "no", TS("2020-02-02"), TypeError, "# wrong type #1"),
        (TS("2020-02-02"), pd.to_timedelta([1]), "no", Exception, "# wrong type #2"),
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
    coordinates = Q_([[i, i, i] for i in range(3)], "mm")
    time_lcs = pd.to_timedelta([1, 2, 3], "D")

    lcs = tf.LocalCoordinateSystem(
        orientation, coordinates, time_lcs, time_ref=time_ref_lcs
    )

    with pytest.raises(exception_type):
        lcs.interp_time(time, time_ref)


# test_addition ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "lcs_lhs, lcs_rhs, orientation_exp, coordinates_exp, time_exp, time_ref_exp",
    [
        (  # 1 - both static
            LCS(r_mat_y(0.5), Q_([1, 4, 2], "mm")),
            LCS(r_mat_z(0.5), Q_([3, 7, 1], "mm")),
            [[0, -1, 0], [0, 0, 1], [-1, 0, 0]],
            [-1, 8, 3],
            None,
            None,
        ),
        (  # 2 - left system orientation time dependent
            LCS(
                r_mat_z([0, 0.5, 1]),
                Q_([1, 4, 2], "mm"),
                pd.to_timedelta([1, 3, 5], "D"),
                TS("2020-02-02"),
            ),
            LCS(r_mat_z(0.5), Q_([3, 7, 1], "mm")),
            r_mat_z([0.5, 1, 1.5]),
            [[-1, 8, 3], [-1, 8, 3], [-1, 8, 3]],
            pd.to_timedelta([1, 3, 5], "D"),
            TS("2020-02-02"),
        ),
        (  # 3 - left system coordinates time dependent
            LCS(
                r_mat_y(0.5),
                Q_([[3, 7, 1], [4, -2, 8], [-5, 3, -1]], "mm"),
                pd.to_timedelta([1, 3, 5], "D"),
                TS("2020-02-02"),
            ),
            LCS(r_mat_z(0.5), Q_([3, 7, 1], "mm")),
            [[[0, -1, 0], [0, 0, 1], [-1, 0, 0]] for _ in range(3)],
            [[-4, 10, 2], [5, 11, 9], [0, 2, 0]],
            pd.to_timedelta([1, 3, 5], "D"),
            TS("2020-02-02"),
        ),
        (  # 4 - right system orientation time dependent
            LCS(r_mat_z(0.5), Q_([3, 7, 1], "mm")),
            LCS(
                r_mat_z([0, 0.5, 1]),
                Q_([1, 4, 2], "mm"),
                pd.to_timedelta([1, 3, 5], "D"),
                TS("2020-02-02"),
            ),
            r_mat_z([0.5, 1, 1.5]),
            [[4, 11, 3], [-6, 7, 3], [-2, -3, 3]],
            pd.to_timedelta([1, 3, 5], "D"),
            TS("2020-02-02"),
        ),
        (  # 5 - right system coordinates time dependent
            LCS(r_mat_z(0.5), Q_([3, 7, 1], "mm")),
            LCS(
                r_mat_z(0.5),
                Q_([[3, 7, 1], [4, -2, 8], [-5, 3, -1]], "mm"),
                pd.to_timedelta([1, 3, 5], "D"),
                TS("2020-02-02"),
            ),
            r_mat_z(1),
            [[-4, 10, 2], [-3, 1, 9], [-12, 6, 0]],
            pd.to_timedelta([1, 3, 5], "D"),
            TS("2020-02-02"),
        ),
        (  # 6 - right system fully time dependent
            LCS(r_mat_z(0.5), Q_([3, 7, 1], "mm")),
            LCS(
                r_mat_z([0, 0.5, 1]),
                Q_([[3, 7, 1], [4, -2, 8], [-5, 3, -1]], "mm"),
                pd.to_timedelta([1, 3, 5], "D"),
                TS("2020-02-02"),
            ),
            r_mat_z([0.5, 1, 1.5]),
            [[6, 14, 2], [-3, 1, 9], [-8, -4, 0]],
            pd.to_timedelta([1, 3, 5], "D"),
            TS("2020-02-02"),
        ),
        (  # 7 - both fully time dependent - same time and reference time
            LCS(
                r_mat_z([1, 0, 0]),
                Q_([[4, 2, 5], [3, -3, 2], [1, 7, -9]], "mm"),
                pd.to_timedelta([1, 3, 5], "D"),
                TS("2020-02-02"),
            ),
            LCS(
                r_mat_z([0, 0.5, 1]),
                Q_([[3, 7, 1], [4, -2, 8], [-5, 3, -1]], "mm"),
                pd.to_timedelta([1, 3, 5], "D"),
                TS("2020-02-02"),
            ),
            r_mat_z([1, 0.5, 1]),
            [[7, 9, 6], [7, 1, 10], [-6, -4, -10]],
            pd.to_timedelta([1, 3, 5], "D"),
            TS("2020-02-02"),
        ),
        (  # 8 - both fully time dependent - different time but same reference time
            LCS(
                r_mat_z([1.5, 1.0, 0.75]),
                Q_([[4, 2, 5], [3, -3, 2], [1, 7, -9]], "mm"),
                pd.to_timedelta([2, 4, 6], "D"),
                TS("2020-02-02"),
            ),
            LCS(
                r_mat_z([0.75, 1.25, 0.75]),
                Q_([[3, 7, 1], [4, -2, 8], [-5, 3, -1]], "mm"),
                pd.to_timedelta([1, 3, 5], "D"),
                TS("2020-02-02"),
            ),
            r_mat_z([0.5, 0.0, 1.5]),
            [[-0.5, 0.5, 9.5], [-3.5, 3.5, 5.5], [-10.6568542, -1.242640687, -10]],
            pd.to_timedelta([2, 4, 6], "D"),
            TS("2020-02-02"),
        ),
        (  # 9 - both fully time dependent - different time and reference time #1
            LCS(
                r_mat_z([1.5, 1.0, 0.75]),
                Q_([[4, 2, 5], [3, -3, 2], [1, 7, -9]], "mm"),
                pd.to_timedelta([1, 3, 5], "D"),
                TS("2020-02-03"),
            ),
            LCS(
                r_mat_z([0.75, 1.25, 0.75]),
                Q_([[3, 7, 1], [4, -2, 8], [-5, 3, -1]], "mm"),
                pd.to_timedelta([1, 3, 5], "D"),
                TS("2020-02-02"),
            ),
            r_mat_z([0.5, 0.0, 1.5]),
            [[-0.5, 0.5, 9.5], [-3.5, 3.5, 5.5], [-10.6568542, -1.242640687, -10]],
            pd.to_timedelta([2, 4, 6], "D"),
            TS("2020-02-02"),
        ),
        (  # 10 - both fully time dependent - different time and reference time #2
            LCS(
                r_mat_z([1.5, 1.0, 0.75]),
                Q_([[4, 2, 5], [3, -3, 2], [1, 7, -9]], "mm"),
                pd.to_timedelta([3, 5, 7], "D"),
                TS("2020-02-01"),
            ),
            LCS(
                r_mat_z([0.75, 1.25, 0.75]),
                Q_([[3, 7, 1], [4, -2, 8], [-5, 3, -1]], "mm"),
                pd.to_timedelta([1, 3, 5], "D"),
                TS("2020-02-02"),
            ),
            r_mat_z([0.5, 0.0, 1.5]),
            [[-0.5, 0.5, 9.5], [-3.5, 3.5, 5.5], [-10.6568542, -1.242640687, -10]],
            pd.to_timedelta([3, 5, 7], "D"),
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
    coordinates_exp = Q_(coordinates_exp, "mm")
    check_coordinate_system(
        lcs_lhs + lcs_rhs,
        orientation_exp,
        coordinates_exp,
        True,
        time_exp,
        time_ref_exp,
    )


# test_subtraction ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "lcs_lhs, lcs_rhs, orientation_exp, coordinates_exp, time_exp, time_ref_exp",
    [
        (  # 1 - both static
            LCS([[0, -1, 0], [0, 0, 1], [-1, 0, 0]], Q_([-1, 8, 3], "mm")),
            LCS(r_mat_z(0.5), Q_([3, 7, 1], "mm")),
            r_mat_y(0.5),
            [1, 4, 2],
            None,
            None,
        ),
        (  # 2 - left system orientation time dependent
            LCS(
                r_mat_z([0.5, 1, 1.5]),
                Q_([-1, 8, 3], "mm"),
                pd.to_timedelta([1, 3, 5], "D"),
                TS("2020-02-02"),
            ),
            LCS(r_mat_z(0.5), Q_([3, 7, 1], "mm")),
            r_mat_z([0, 0.5, 1]),
            [[1, 4, 2], [1, 4, 2], [1, 4, 2]],
            pd.to_timedelta([1, 3, 5], "D"),
            TS("2020-02-02"),
        ),
        (  # 3 - left system coordinates time dependent
            LCS(
                [[0, -1, 0], [0, 0, 1], [-1, 0, 0]],
                Q_([[-4, 10, 2], [5, 11, 9], [0, 2, 0]], "mm"),
                pd.to_timedelta([1, 3, 5], "D"),
                TS("2020-02-02"),
            ),
            LCS(r_mat_z(0.5), Q_([3, 7, 1], "mm")),
            r_mat_y([0.5, 0.5, 0.5]),
            [[3, 7, 1], [4, -2, 8], [-5, 3, -1]],
            pd.to_timedelta([1, 3, 5], "D"),
            TS("2020-02-02"),
        ),
        (  # 4 - right system orientation time dependent
            LCS(r_mat_z(0.5), Q_([3, 7, 1], "mm")),
            LCS(
                r_mat_z([0, 0.5, 1]),
                Q_([1, 4, 2], "mm"),
                pd.to_timedelta([1, 3, 5], "D"),
                TS("2020-02-02"),
            ),
            r_mat_z([0.5, 0, 1.5]),
            [[2, 3, -1], [3, -2, -1], [-2, -3, -1]],
            pd.to_timedelta([1, 3, 5], "D"),
            TS("2020-02-02"),
        ),
        (  # 5 - right system coordinates time dependent
            LCS(r_mat_z(0.5), Q_([3, 7, 1], "mm")),
            LCS(
                r_mat_z(0.5),
                Q_([[3, 7, 1], [4, -2, 8], [-5, 3, -1]], "mm"),
                pd.to_timedelta([1, 3, 5], "D"),
                TS("2020-02-02"),
            ),
            r_mat_z(0),
            [[0, 0, 0], [9, 1, -7], [4, -8, 2]],
            pd.to_timedelta([1, 3, 5], "D"),
            TS("2020-02-02"),
        ),
        (  # 6 - right system fully time dependent
            LCS(r_mat_z(0.5), Q_([3, 7, 1], "mm")),
            LCS(
                r_mat_z([0, 0.5, 1]),
                Q_([[3, 7, 1], [4, -2, 8], [-5, 3, -1]], "mm"),
                pd.to_timedelta([1, 3, 5], "D"),
                TS("2020-02-02"),
            ),
            r_mat_z([0.5, 0, 1.5]),
            [[0, 0, 0], [9, 1, -7], [-8, -4, 2]],
            pd.to_timedelta([1, 3, 5], "D"),
            TS("2020-02-02"),
        ),
        (  # 7 - both fully time dependent - same time and reference time
            LCS(
                r_mat_z([1, 0.5, 1]),
                Q_([[7, 9, 6], [7, 1, 10], [-6, -4, -10]], "mm"),
                pd.to_timedelta([1, 3, 5], "D"),
                TS("2020-02-02"),
            ),
            LCS(
                r_mat_z([0, 0.5, 1]),
                Q_([[3, 7, 1], [4, -2, 8], [-5, 3, -1]], "mm"),
                pd.to_timedelta([1, 3, 5], "D"),
                TS("2020-02-02"),
            ),
            r_mat_z([1, 0, 0]),
            [[4, 2, 5], [3, -3, 2], [1, 7, -9]],
            pd.to_timedelta([1, 3, 5], "D"),
            TS("2020-02-02"),
        ),
        (  # 8 - both fully time dependent - different time but same reference time
            LCS(
                r_mat_z([1.5, 1.0, 0.75]),
                Q_([[4, 2, 5], [3, -3, 2], [1, 7, -9]], "mm"),
                pd.to_timedelta([2, 4, 6], "D"),
                TS("2020-02-02"),
            ),
            LCS(
                r_mat_z([1, 1.5, 1]),
                Q_([[3, 7, 1], [4, -2, 8], [-5, 3, -1]], "mm"),
                pd.to_timedelta([1, 3, 5], "D"),
                TS("2020-02-02"),
            ),
            r_mat_z([0.25, 1.75, 1.75]),
            [[-3.7426406, 2.9142135, 0.5], [-3.5, 3.742640, -1.5], [-6, -4, -8.0]],
            pd.to_timedelta([2, 4, 6], "D"),
            TS("2020-02-02"),
        ),
        (  # 9 - both fully time dependent - different time and reference time #1
            LCS(
                r_mat_z([1.5, 1.0, 0.75]),
                Q_([[4, 2, 5], [3, -3, 2], [1, 7, -9]], "mm"),
                pd.to_timedelta([1, 3, 5], "D"),
                TS("2020-02-03"),
            ),
            LCS(
                r_mat_z([1, 1.5, 1]),
                Q_([[3, 7, 1], [4, -2, 8], [-5, 3, -1]], "mm"),
                pd.to_timedelta([1, 3, 5], "D"),
                TS("2020-02-02"),
            ),
            r_mat_z([0.25, 1.75, 1.75]),
            [[-3.7426406, 2.9142135, 0.5], [-3.5, 3.742640, -1.5], [-6, -4, -8.0]],
            pd.to_timedelta([2, 4, 6], "D"),
            TS("2020-02-02"),
        ),
        (  # 10 - both fully time dependent - different time and reference time #2
            LCS(
                r_mat_z([1.5, 1.0, 0.75]),
                Q_([[4, 2, 5], [3, -3, 2], [1, 7, -9]], "mm"),
                pd.to_timedelta([3, 5, 7], "D"),
                TS("2020-02-01"),
            ),
            LCS(
                r_mat_z([1, 1.5, 1]),
                Q_([[3, 7, 1], [4, -2, 8], [-5, 3, -1]], "mm"),
                pd.to_timedelta([1, 3, 5], "D"),
                TS("2020-02-02"),
            ),
            r_mat_z([0.25, 1.75, 1.75]),
            [[-3.7426406, 2.9142135, 0.5], [-3.5, 3.742640, -1.5], [-6, -4, -8.0]],
            pd.to_timedelta([3, 5, 7], "D"),
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
    coordinates_exp = Q_(coordinates_exp, "mm")
    check_coordinate_system(
        lcs_lhs - lcs_rhs,
        orientation_exp,
        coordinates_exp,
        True,
        time_exp,
        time_ref_exp,
    )


# test_comparison_coords_timeseries ----------------------------------------------------


@pytest.mark.parametrize(
    "kwargs_me_other_upd, kwargs_ts_other_upd, kwargs_other_upd, exp_result",
    [
        ({}, {}, {}, True),
        (dict(expression="2*a*t"), {}, {}, False),
        (dict(parameters=dict(a=Q_([[2, 0, 0]], "m/s"))), {}, {}, False),
        ({}, dict(data=Q_(np.ones((2, 3)), "mm"), time=Q_([1, 2], "s")), {}, False),
        ({}, {}, dict(orientation=[[0, -1, 0], [1, 0, 0], [0, 0, 1]]), False),
        ({}, {}, dict(time_ref=TS("11:12")), False),
    ],
)
def test_comparison_coords_timeseries(
    kwargs_me_other_upd: dict,
    kwargs_ts_other_upd: dict,
    kwargs_other_upd: dict,
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
    me = MathematicalExpression("a*t", dict(a=Q_([[1, 0, 0]], "m/s")))
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


# --------------------------------------------------------------------------------------
# old tests --- should be rewritten or merged into the existing ones above
# --------------------------------------------------------------------------------------


def test_coordinate_system_init():
    """Check the __init__ method with and without time dependency."""
    # reference data
    time_0 = pd.to_timedelta([1, 3, 5], "s")
    time_1 = pd.to_timedelta([2, 4, 6], "s")

    orientation_fix = r_mat_z(1)
    orientation_tdp = r_mat_z([0, 0.25, 0.5])
    coordinates_fix = Q_([3, 7, 1], "mm")
    coordinates_tdp = Q_([[3, 7, 1], [4, -2, 8], [-5, 3, -1]], "mm")

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

    time_exp = pd.to_timedelta([1, 2, 3, 4, 5, 6], "s")
    coordinates_exp = Q_(
        [
            [3, 7, 1],
            [3, 7, 1],
            [3.5, 2.5, 4.5],
            [4, -2, 8],
            [-0.5, 0.5, 3.5],
            [-5, 3, -1],
        ],
        "mm",
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
    orientation_exp = r_mat_z(1 / 3 * np.array([1, 2, 4], dtype=float))
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
    with pytest.raises(ValueError):
        tf.LocalCoordinateSystem(
            orientation="wrong", coordinates=coordinates_fix, time=time_0
        )
    with pytest.raises(ValueError):
        tf.LocalCoordinateSystem(
            orientation=orientation_fix, coordinates="wrong", time=time_0
        )
    with pytest.raises(TypeError):
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

    # todo: this test is actually pretty pointless since the creation of the reference
    #       data is identical to the implementation of the tested method. We should come
    #       up with a better test (Use hardcoded results for specific inputs?)

    # setup -----------------------------------------------
    angles = [np.pi / 3, np.pi / 4, np.pi / 5]
    coordinates = Q_([4, -2, 6], "mm")
    orientation_pos = WXRotation.from_euler("xyz", angles).as_matrix()

    # construction with euler -----------------------------

    cs_euler_pos = LCS.from_euler("xyz", angles, False, coordinates)
    check_coordinate_system(cs_euler_pos, orientation_pos, coordinates, True)


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

    time = pd.to_timedelta([0, 6, 12, 18], "h")
    orientations = np.matmul(rot_mat_x, rot_mat_y)
    coords = Q_([[1, 0, 0], [-1, 0, 2], [3, 5, 7], [-4, -5, -6]], "mm")

    # construction with euler -----------------------------

    cs_euler_oc = lcs.from_euler("yx", angles, False, coords, time)
    check_coordinate_system(cs_euler_oc, orientations, coords, time=time)

    cs_euler_c = lcs.from_euler("yx", angles[0], False, coords, time)
    check_coordinate_system(cs_euler_c, orientations[0], coords, time=time)

    cs_euler_o = lcs.from_euler("yx", angles, False, coords[0], time)
    check_coordinate_system(cs_euler_o, orientations, coords[0], time=time)


def test_coordinate_system_invert():
    """Test the invert function.

    The test creates a coordinate system, inverts it and checks the result against the
    expected value. Afterwards, the resulting system is inverted again. This operation
    must yield the original system.

    """
    # fix ---------------------------------------
    lcs0_in_lcs1 = tf.LocalCoordinateSystem.from_axis_vectors(
        x=[1, 1, 0], y=[-1, 1, 0], coordinates=Q_([2, 0, 2], "mm")
    )
    lcs1_in_lcs0 = lcs0_in_lcs1.invert()

    exp_orientation = r_mat_z(-1 / 4)
    exp_coordinates = Q_([-np.sqrt(2), np.sqrt(2), -2], "mm")

    check_coordinate_system(lcs1_in_lcs0, exp_orientation, exp_coordinates, True)

    lcs0_in_lcs1_2 = lcs1_in_lcs0.invert()

    check_coordinate_system(
        lcs0_in_lcs1_2, lcs0_in_lcs1.orientation, lcs0_in_lcs1.coordinates.data, True
    )

    # time dependent ----------------------------
    time = pd.to_timedelta([1, 2, 3, 4], "s")
    orientation = r_mat_z([0, 0.5, 1, 0.5])
    coordinates = Q_([[2, 8, 7], [4, 9, 2], [0, 2, 1], [3, 1, 2]], "mm")

    lcs0_in_lcs1 = tf.LocalCoordinateSystem(
        orientation=orientation, coordinates=coordinates, time=time
    )

    lcs1_in_lcs0 = lcs0_in_lcs1.invert()
    orientation_exp = r_mat_z([0, 1.5, 1, 1.5])
    coordinates_exp = Q_([[-2, -8, -7], [-9, 4, -2], [0, 2, -1], [-1, 3, -2]], "mm")

    check_coordinate_system(lcs1_in_lcs0, orientation_exp, coordinates_exp, True, time)

    lcs0_in_lcs1_2 = lcs1_in_lcs0.invert()

    check_coordinate_system(
        lcs0_in_lcs1_2,
        lcs0_in_lcs1.orientation,
        lcs0_in_lcs1.coordinates.data,
        True,
        time,
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
    time_0 = pd.to_timedelta([10, 14, 18, 22], "D")
    orientation = r_mat_z([0, 0.5, 1, 0.5])
    coordinates = Q_([[2, 8, 7], [4, 9, 2], [0, 2, 1], [3, 1, 2]], "mm")

    lcs = tf.LocalCoordinateSystem(
        orientation=orientation, coordinates=coordinates, time=time_0
    )

    # test xr_interp_orientation_in_time for single time point interpolation
    for i, _ in enumerate(time_0):
        # test for scalar value as coordinate
        orientation_interp = ut.xr_interp_orientation_in_time(
            lcs.orientation.isel({"time": i}), time_0
        )
        assert np.allclose(orientation_interp, lcs.orientation[i])

        # test for scalar value as dimension
        orientation_interp = ut.xr_interp_orientation_in_time(
            lcs.orientation.isel({"time": [i]}), time_0
        )
        assert np.allclose(orientation_interp, lcs.orientation[i])

    # exceptions --------------------------------
    # wrong parameter type
    with pytest.raises(TypeError):
        lcs.interp_time("wrong")
    # no time component
    with pytest.raises(TypeError):
        lcs.interp_time(tf.LocalCoordinateSystem())
