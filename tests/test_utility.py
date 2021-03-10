"""Test the internal utility functions."""

import math

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pandas import DatetimeIndex as DTI
from pandas import TimedeltaIndex as TDI
from pandas import date_range
from pint.errors import DimensionalityError

import weldx.util as ut
from weldx.constants import WELDX_QUANTITY as Q_


def test_is_column_in_matrix():
    """Test the is_column_in_matrix function.

    Test should be self explanatory.

    """
    c_0 = [1, 5, 2]
    c_1 = [3, 2, 2]
    c_2 = [1, 6, 1]
    c_3 = [1, 6, 0]
    matrix = np.array([c_0, c_1, c_2, c_3]).transpose()

    assert ut.is_column_in_matrix(c_0, matrix)
    assert ut.is_column_in_matrix(c_1, matrix)
    assert ut.is_column_in_matrix(c_2, matrix)
    assert ut.is_column_in_matrix(c_3, matrix)

    assert not ut.is_column_in_matrix([1, 6], matrix)
    assert not ut.is_column_in_matrix([1, 6, 2], matrix)
    assert not ut.is_column_in_matrix([1, 1, 3, 1], matrix)


def test_is_row_in_matrix():
    """Test the is_row_in_matrix function.

    Test should be self explanatory.

    """
    c_0 = [1, 5, 2]
    c_1 = [3, 2, 2]
    c_2 = [1, 6, 1]
    c_3 = [1, 6, 0]
    matrix = np.array([c_0, c_1, c_2, c_3])

    assert ut.is_row_in_matrix(c_0, matrix)
    assert ut.is_row_in_matrix(c_1, matrix)
    assert ut.is_row_in_matrix(c_2, matrix)
    assert ut.is_row_in_matrix(c_3, matrix)

    assert not ut.is_row_in_matrix([1, 6], matrix)
    assert not ut.is_row_in_matrix([1, 6, 2], matrix)
    assert not ut.is_row_in_matrix([1, 1, 3, 1], matrix)


def test_matrix_is_close():
    """Test the matrix_is_close function.

    Test should be self explanatory.

    """
    mat_a = np.array([[0, 1, 2], [3, 4, 5]])
    mat_b = np.array([[3, 5, 1], [7, 1, 9]])

    assert ut.matrix_is_close(mat_a, mat_a)
    assert ut.matrix_is_close(mat_b, mat_b)
    assert not ut.matrix_is_close(mat_a, mat_b)
    assert not ut.matrix_is_close(mat_b, mat_a)

    # check tolerance
    mat_c = mat_a + 0.0001
    assert ut.matrix_is_close(mat_a, mat_c, abs_tol=0.00011)
    assert not ut.matrix_is_close(mat_a, mat_c, abs_tol=0.00009)

    # vectors have different size
    assert not ut.matrix_is_close(mat_a, mat_a[0:2, 0:2])


def test_vector_is_close():
    """Test the vector_is_close function.

    Test should be self explanatory.

    """
    vec_a = np.array([0, 1, 2])
    vec_b = np.array([3, 5, 1])

    assert ut.vector_is_close(vec_a, vec_a)
    assert ut.vector_is_close(vec_b, vec_b)
    assert not ut.vector_is_close(vec_a, vec_b)
    assert not ut.vector_is_close(vec_b, vec_a)

    # check tolerance
    vec_c = vec_a + 0.0001
    assert ut.vector_is_close(vec_a, vec_c, abs_tol=0.00011)
    assert not ut.vector_is_close(vec_a, vec_c, abs_tol=0.00009)

    # vectors have different size
    assert not ut.vector_is_close(vec_a, vec_a[0:2])


@pytest.mark.parametrize(
    "arg, expected",
    [
        # timedeltas
        (TDI([42]), TDI([42])),
        (pd.timedelta_range("0s", "20s", 10), pd.timedelta_range("0s", "20s", 10)),
        (np.timedelta64(42), TDI([42])),
        (np.array([-10, 0, 20]).astype(np.timedelta64), TDI([-10, 0, 20])),
        (Q_(42, "ns"), TDI([42])),
        ("10s", TDI(["10s"])),
        (["5ms", "10s", "2D"], TDI(["5 ms", "10s", "2D"])),
        # datetimes
        (np.datetime64(50, "Y"), DTI(["2020-01-01"])),
        ("2020-01-01", DTI(["2020-01-01"])),
        (
            np.array(
                ["2012-10-02", "2012-10-05", "2012-10-11"], dtype="datetime64[ns]"
            ),
            DTI(["2012-10-02", "2012-10-05", "2012-10-11"]),
        ),
    ],
)
def test_to_pandas_time_index(arg, expected):
    """Test conversion to appropriate pd.TimedeltaIndex or pd.DatetimeIndex."""
    assert np.all(ut.to_pandas_time_index(arg) == expected)


@pytest.mark.parametrize(
    "arg, exception",
    [(5, TypeError), ("string", TypeError), (Q_(10, "m"), DimensionalityError)],
)
def test_to_pandas_time_index_exceptions(arg, exception):
    """Test correct exceptions on invalid inputs."""
    with pytest.raises(exception):
        ut.to_pandas_time_index(arg)


def test_pandas_time_delta_to_quantity():
    """Test the 'pandas_time_delta_to_quantity' utility function."""
    is_close = np.vectorize(math.isclose)

    def _check_close(t1, t2):
        assert np.all(is_close(t1.magnitude, t2.magnitude))
        assert t1.units == t2.units

    time_single = pd.TimedeltaIndex([1], unit="s")

    _check_close(ut.pandas_time_delta_to_quantity(time_single), Q_(1, "s"))
    _check_close(ut.pandas_time_delta_to_quantity(time_single, "ms"), Q_(1000, "ms"))
    _check_close(ut.pandas_time_delta_to_quantity(time_single, "us"), Q_(1000000, "us"))
    _check_close(
        ut.pandas_time_delta_to_quantity(time_single, "ns"), Q_(1000000000, "ns")
    )

    time_multi = pd.TimedeltaIndex([1, 2, 3], unit="s")
    _check_close(ut.pandas_time_delta_to_quantity(time_multi), Q_([1, 2, 3], "s"))
    _check_close(
        ut.pandas_time_delta_to_quantity(time_multi, "ms"), Q_([1000, 2000, 3000], "ms")
    )
    _check_close(
        ut.pandas_time_delta_to_quantity(time_multi, "us"),
        Q_([1000000, 2000000, 3000000], "us"),
    )
    _check_close(
        ut.pandas_time_delta_to_quantity(time_multi, "ns"),
        Q_([1000000000, 2000000000, 3000000000], "ns"),
    )


def test_xr_interp_like():
    """Test behaviour of custom interpolation method for xarray Objects."""
    # basic interpolation behavior on a single coordinate
    n_a = 5  # range of "a" coordinate in da_a
    s_a = 0.5  # default steps in "a" coordinate in da_a
    da_a = xr.DataArray(
        np.arange(0, n_a + s_a, s_a),
        dims=["a"],
        coords={"a": np.arange(0, n_a + s_a, s_a)},
    )

    # interp from subset inside original data
    test = ut.xr_interp_like(da_a.loc[2:4:2], da_a)
    assert test.a[0] == da_a.a[0]
    assert test.a[-1] == da_a.a[-1]
    assert test[0] == da_a.loc[2]
    assert test[-1] == da_a.loc[4]

    # interp with overlap
    test = ut.xr_interp_like(da_a.loc[1:3], da_a.loc[2:4])
    assert test.a[0] == da_a.a.loc[2]
    assert test.a[-1] == da_a.a.loc[4]
    assert test[0] == da_a.loc[2]
    assert test[-1] == da_a.loc[3]
    assert np.all(test.loc[3:4] == da_a.loc[3])

    # overlap without fill (expecting nan values for out of range indexes)
    test = ut.xr_interp_like(da_a.loc[1:3], da_a.loc[2:4], fillna=False)
    assert test.a[0] == da_a.a.loc[2]
    assert test.a[-1] == da_a.a.loc[4]
    assert test[0] == da_a.loc[2]
    assert np.isnan(test[-1])
    assert np.all(np.isnan(test.where(test.a > 3, drop=True)))

    # outside interpolation without overlap
    test = ut.xr_interp_like(da_a.loc[0:3], da_a.loc[4:5])
    assert test.a[0] == da_a.a.loc[4]
    assert test.a[-1] == da_a.a.loc[5]
    assert np.all(test.loc[4:5] == da_a.loc[3])

    # single point to array interpolation
    # important: da_a.loc[5] for indexing would drop coordinates (unsure why)
    test = ut.xr_interp_like(da_a.loc[2:2], da_a)
    assert np.all(test.a == da_a.a)
    assert np.all(test == da_a.loc[2:2])
    with pytest.raises(ValueError):
        ut.xr_interp_like(da_a.loc[2:2], da_a, fillna=False)

    # single point to single point interpolation (different points)
    test = ut.xr_interp_like(da_a.loc[2:2], da_a.loc[3:3])
    assert np.all(test.a == da_a.a)
    assert np.all(test == da_a.loc[2:2])

    # single point to single point interpolation (matching points)
    test = ut.xr_interp_like(da_a.loc[2:2], da_a.loc[2:2])
    assert np.all(test.a == da_a.a)
    assert np.all(test == da_a.loc[2:2])

    # dict-like inputs on existing coordinate
    test = ut.xr_interp_like(da_a, {"a": np.arange(-5, 15, 0.5)})
    assert np.all(test.where(test.a < da_a.a.min(), drop=True) == da_a.min())
    assert np.all(test.where(test.a > da_a.a.max(), drop=True) == da_a[-1])
    assert np.all(test.where(test.a.isin(da_a.a), drop=True) == da_a)

    da = da_a.loc[3:3]
    test = ut.xr_interp_like(da, {"a": np.arange(-5, 15, 0.5)})
    assert np.all(test.where(test.a < da.a.min(), drop=True) == da.min())
    assert np.all(test.where(test.a > da.a.max(), drop=True) == da[-1])
    assert np.all(test.where(test.a.isin(da.a), drop=True) == da)

    test = ut.xr_interp_like(da_a, {"a": np.arange(-5, 15, 0.5)}, fillna=False)
    assert np.all(np.isnan((test.where(test.a < 0, drop=True))))
    assert np.all(np.isnan(test.where(test.a > da_a.a.max(), drop=True)))
    assert np.all(test.where(test.a.isin(da_a.a), drop=True) == da_a)

    # dict-like inputs on new coordinate with/without broadcasting
    da1 = da_a
    test = ut.xr_interp_like(da1, {"b": np.arange(3)})
    assert test.equals(da_a)

    da1 = da_a
    test = ut.xr_interp_like(da1, {"b": np.arange(3)}, broadcast_missing=True)
    assert test.equals(da1.broadcast_like(test))

    # test coordinate selection with interp_coords
    da1 = da_a
    test = ut.xr_interp_like(
        da1,
        {"b": np.arange(3), "c": np.arange(3)},
        broadcast_missing=True,
        interp_coords=["c"],
    )
    assert "b" not in test.coords
    assert "c" in test.coords

    # catch error on unsorted array
    da = xr.DataArray([0, 1, 2, 3], dims="a", coords={"a": [2, 1, 3, 0]})
    with pytest.raises(ValueError):
        test = ut.xr_interp_like(da, {"a": np.arange(6)}, assume_sorted=True)

    # basic interpolation behavior with different coordinates (broadcasting)
    n_b = 3  # range of "b" coordinate in da_b
    s_b = 1  # default steps in "b" coordinate in da_b
    da_b = xr.DataArray(
        np.arange(0, n_b + s_b, s_b) ** 2,
        dims=["b"],
        coords={"b": np.arange(0, n_b + s_b, s_b)},
    )

    assert da_a.equals(ut.xr_interp_like(da_a, da_b))
    assert da_a.broadcast_like(da_b).broadcast_equals(
        ut.xr_interp_like(da_a, da_b, broadcast_missing=True)
    )

    # coords syntax
    assert da_a.broadcast_like(da_b).broadcast_equals(
        ut.xr_interp_like(da_a, da_b.coords, broadcast_missing=True)
    )

    # sorting and interpolation with multiple dimensions
    a = np.arange(3, 6)
    b = np.arange(1, -3, -1)
    da_ab = xr.DataArray(
        a[..., np.newaxis] @ b[np.newaxis, ...],
        dims=["a", "b"],
        coords={"a": a, "b": b},
    )

    a_new = np.arange(3, 5, 0.5)
    b_new = np.arange(-1, 1, 0.5)
    test = ut.xr_interp_like(
        da_ab,
        {"a": a_new, "b": b_new, "c": np.arange(2)},
        assume_sorted=False,
        broadcast_missing=True,
    )
    assert np.all(
        test.transpose(..., "a", "b") == a_new[..., np.newaxis] @ b_new[np.newaxis, ...]
    )

    # tests with time data types
    # TODO: add more complex test examples
    t = pd.timedelta_range(start="10s", end="0s", freq="-1s", closed="left")
    da_t = xr.DataArray(np.arange(10, 0, -1), dims=["t"], coords={"t": t})

    test = ut.xr_interp_like(
        da_t,
        {"t": pd.timedelta_range(start="3s", end="7s", freq="125ms", closed="left")},
    )
    assert np.all(test == np.arange(3, 7, 0.125))


@pytest.mark.parametrize(
    "list_of_objects, time_exp",
    [
        (
            [
                date_range("2020-02-02", periods=4, freq="2D"),
                date_range("2020-02-01", periods=4, freq="2D"),
                date_range("2020-02-03", periods=2, freq="3D"),
            ],
            date_range("2020-02-01", periods=8, freq="1D"),
        ),
        ([TDI([1, 5]), TDI([2, 6, 7]), TDI([1, 3, 7])], TDI([1, 2, 3, 5, 6, 7])),
    ],
)
def test_get_time_union(list_of_objects, time_exp):
    """Test input types for get_time_union function.

    Parameters
    ----------
    list_of_objects:
        List with input objects
    time_exp:
        Expected result time

    """
    assert np.all(ut.get_time_union(list_of_objects) == time_exp)


def test_xr_fill_all():
    """Test filling along all dimensions."""
    da1 = xr.DataArray(
        np.eye(2), dims=["a", "b"], coords={"a": np.arange(2), "b": np.arange(2)}
    )
    da2 = xr.DataArray(
        dims=["a", "b"], coords={"a": np.arange(-1, 3, 0.5), "b": np.arange(-1, 3, 0.5)}
    )
    da3 = da1.broadcast_like(da2)

    da4 = ut.xr_fill_all(da3, order="fb")
    assert not np.any(np.isnan(da4))
    assert np.all(da4[0:4, 0:4] == np.ones((4, 4)))
    assert np.all(da4[0:4, 4:8] == np.zeros((4, 4)))
    assert np.all(da4[4:8, 0:4] == np.zeros((4, 4)))
    assert np.all(da4[4:8, 4:8] == np.ones((4, 4)))

    da4 = ut.xr_fill_all(da3, order="bf")
    assert not np.any(np.isnan(da4))
    assert np.all(da4[0:3, 0:3] == np.ones((3, 3)))
    assert np.all(da4[0:3, 3:8] == np.zeros((3, 5)))
    assert np.all(da4[3:8, 0:3] == np.zeros((5, 3)))
    assert np.all(da4[3:8, 3:8] == np.ones((5, 5)))

    with pytest.raises(ValueError):
        ut.xr_fill_all(da3, order="wrong")


_dax_check = xr.DataArray(
    data=np.ones((2, 2, 2, 4, 3)),
    dims=["d1", "d2", "d3", "d4", "d5"],
    coords={
        "d1": np.array([-1, 1], dtype=float),
        "d2": np.array([-1, 1], dtype=int),
        "d3": pd.DatetimeIndex(["2020-05-01", "2020-05-03"]),
        "d4": pd.TimedeltaIndex([0, 1, 2, 3], "s"),
        "d5": ["x", "y", "z"],
    },
)

_dax_ref = dict(
    d1={"values": np.array([-1, 1]), "dtype": "float"},
    d2={"values": np.array([-1, 1]), "dtype": int},
    d3={
        "values": pd.DatetimeIndex(["2020-05-01", "2020-05-03"]),
        "dtype": ["datetime64[ns]", "timedelta64[ns]"],
    },
    d4={
        "values": pd.TimedeltaIndex([0, 1, 2, 3], "s"),
        "dtype": ["datetime64[ns]", "timedelta64[ns]"],
    },
    d5={"values": ["x", "y", "z"], "dtype": "<U1"},
)


@pytest.mark.parametrize(
    "dax, ref_dict",
    [
        (_dax_check, _dax_ref),
        (_dax_check.coords, _dax_ref),
        (_dax_check, {"d1": {"dtype": ["float64", int]}}),
        (_dax_check, {"d2": {"dtype": ["float64", int]}}),
        (_dax_check, {"no_dim": {"optional": True, "dtype": float}}),
        (_dax_check, {"d5": {"dtype": str}}),
        (_dax_check, {"d5": {"dtype": [str]}}),
        (_dax_check, {"d4": {"dtype": "timedelta64"}}),
        (_dax_check, {"d3": {"dtype": ["datetime64", "timedelta64"]}}),
    ],
)
def test_xr_check_coords(dax, ref_dict):
    """Test weldx.utility.xr_check_coords function."""
    assert ut.xr_check_coords(dax, ref_dict)


@pytest.mark.parametrize(
    "dax, ref_dict, exception_type",
    [
        (_dax_check, {"d1": {"dtype": int}}, TypeError),
        (_dax_check, {"d1": {"dtype": int, "optional": True}}, TypeError),
        (_dax_check, {"no_dim": {"dtype": float}}, KeyError),
        (
            _dax_check,
            {"d5": {"values": ["x", "noty", "z"], "dtype": "str"}},
            ValueError,
        ),
        (_dax_check, {"d1": {"dtype": [int, str, bool]}}, TypeError),
        (_dax_check, {"d3": {"dtype": "timedelta64"}}, TypeError),
        (_dax_check, {"d4": {"dtype": "datetime64"}}, TypeError),
        ({"d4": np.arange(4)}, {"d4": {"dtype": "int"}}, ValueError),
    ],
)
def test_xr_check_coords_exception(dax, ref_dict, exception_type):
    """Test weldx.utility.xr_check_coords function."""
    with pytest.raises(exception_type):
        ut.xr_check_coords(dax, ref_dict)


def test_xr_time_ref():
    """Test weldx accessor functions for time handling."""
    dt = pd.TimedeltaIndex([0, 1, 2, 3], "s")
    da1 = xr.DataArray(
        data=np.ones(4),
        dims=["time"],
        coords={"time": dt},
    )

    da1.time.attrs = {"A": "B"}

    # non changing operations
    da = da1.weldx.time_ref_unset()
    assert da1.identical(da)
    da = da1.weldx.time_ref_restore()
    assert da1.identical(da)

    t0 = pd.Timestamp("2021-01-01")
    da1.weldx.time_ref = t0
    da = da1.weldx.time_ref_unset()
    assert np.all(da.time.data == (dt + t0))

    da = da.weldx.reset_reference_time(pd.Timestamp("2021-01-01 00:00:01"))
    assert np.all(da.time.data == pd.TimedeltaIndex([-1, 0, 1, 2], "s"))

    da2 = xr.DataArray(
        data=np.ones(4),
        dims=["time"],
        coords={"time": t0 + dt},
    )
    da2 = da2.weldx.time_ref_restore()
    assert np.all(da2.time.data == pd.TimedeltaIndex([0, 1, 2, 3], "s"))
    assert da2.time.attrs["time_ref"] == t0

    da2.weldx.time_ref = t0
    assert da2.time.attrs["time_ref"] == t0
