"""Test the internal utility functions."""
import copy
import math
import unittest
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy import NaN
from pandas import DatetimeIndex as DTI
from pandas import TimedeltaIndex as TDI
from pandas import date_range
from pint.errors import DimensionalityError
from xarray import DataArray

import weldx.util as ut
from weldx.constants import WELDX_QUANTITY as Q_


def test_deprecation_decorator():
    """Test that the deprecation decorator emits a warning as expected."""

    @ut.deprecated(since="3.1.0", removed="4.0.0", message="Use something else")
    def _deprecated_function():
        return "nothing"

    with pytest.warns(ut.WeldxDeprecationWarning):
        _deprecated_function()


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
        (TDI([42], unit="ns"), TDI([42], unit="ns")),
        (pd.timedelta_range("0s", "20s", 10), pd.timedelta_range("0s", "20s", 10)),
        (np.timedelta64(42), TDI([42], unit="ns")),
        (np.array([-10, 0, 20]).astype("timedelta64[ns]"), TDI([-10, 0, 20], "ns")),
        (Q_(42, "ns"), TDI([42], unit="ns")),
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


class TestXarrayInterpolation:
    """Tests all custom xarray interpolation functions."""

    @staticmethod
    @pytest.mark.parametrize("assume_sorted", [True, False])
    @pytest.mark.parametrize("use_dict_ref", [True, False])
    @pytest.mark.parametrize(
        "data, coords, coords_ref, exp_values, kwargs",
        [
            # linear interpolation
            (None, None, dict(d1=[3.3, 4.1]), [3.3, 4.1], dict(method="linear")),
            # step interpolation
            (None, None, dict(d1=[1.3, 3.9, 4.1]), [1, 3, 4], dict(method="step")),
            # interp from subset inside original data
            (None, None, dict(d1=[1, 3]), [1, 3], {}),
            # overlapping coordinates
            (None, None, dict(d1=[4, 5, 7, 8]), [4, 5, 5, 5], {}),
            # overlapping coordinates without fill
            (None, None, dict(d1=[4, 5, 7, 8]), [4, 5, NaN, NaN], dict(fillna=False)),
            # overlapping coordinates without fill
            (
                None,
                None,
                dict(d1=[4, 5, 7, 8]),
                [4, 5, NaN, NaN],
                dict(fillna=False, method="step"),
            ),
            # no overlapping coordinates
            (None, None, dict(d1=[-2, -1]), [0, 0], {}),
            # no overlapping coordinates
            (None, None, dict(d1=[-2, 7]), [0, 5], dict(method="step")),
            # single coordinate interpolation
            ([3], dict(d1=[2]), dict(d1=[4, 5, 7]), [3, 3, 3], {}),
            # single coordinate interpolation with single reference coord (identical)
            ([3], dict(d1=[2]), dict(d1=[2]), [3], {}),
            # single coordinate interpolation with single reference coord (different)
            ([3], dict(d1=[2]), dict(d1=[7]), [3], {}),
            # different dimensions without broadcasting of missing dimensions
            (None, None, dict(d2=[-2, -1]), range(6), {}),
            # different dimensions with broadcasting of missing dimensions
            (
                None,
                None,
                dict(d2=[-2, -1]),
                [range(6), range(6)],
                dict(broadcast_missing=True),
            ),
        ],
    )
    def test_xr_interp_like(
        data: List,
        coords: Dict,
        coords_ref: Dict,
        exp_values: List,
        kwargs: Dict,
        use_dict_ref: bool,
        assume_sorted: bool,
    ):
        """Test the ‘xr_interp_like‘ function.

        Parameters
        ----------
        data :
            The data of the source `DataArray`
        coords :
            The coordinates of the source `DataArray`
        coords_ref :
            The coordinates of the reference
        exp_values :
            Expected values of the interpolated `DataArray`
        kwargs :
            Key word arguments that should be passed to `xr_interp_like`
        use_dict_ref :
            If `True`, a dictionary will serve as reference. Otherwise, a `DataArray` is
            used
        assume_sorted :
            Sets the corresponding parameter of ´xr_interp_like´

        """
        # set default values for missing data
        if data is None:
            data = range(6)
        if coords is None:
            coords = dict(d1=data)
        dims = list(coords.keys())

        # create source data array
        da_data = DataArray(data=data, dims=dims, coords=coords)

        # create reference
        dims_ref = list(coords_ref.keys())
        all_dims = set(dims + dims_ref)
        if use_dict_ref:
            ref = coords_ref
        else:
            data_ref = np.zeros([len(coords_ref[dim]) for dim in dims_ref])
            ref = DataArray(data=data_ref, dims=dims_ref, coords=coords_ref)

        # perform interpolation
        da_interp = ut.xr_interp_like(
            da_data, ref, assume_sorted=assume_sorted, **kwargs
        )

        # check coordinates
        broadcast_missing = kwargs.get("broadcast_missing", False)
        for dim in all_dims:
            if dim in dims_ref and (dim in dims or broadcast_missing):
                assert np.allclose(da_interp.coords[dim], coords_ref[dim])
            elif dim in dims:
                assert np.allclose(da_interp.coords[dim], coords[dim])

        # check data
        assert da_interp.values.shape == np.array(exp_values).shape
        assert np.allclose(da_interp.values, exp_values, equal_nan=True)


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

    # single point to array interpolation
    # important: da_a.loc[5] for indexing would drop coordinates (unsure why)
    with pytest.raises(ValueError):
        ut.xr_interp_like(da_a.loc[2:2], da_a, fillna=False)

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


class TestCompareNested:
    """Test utility.compare_nested function on different objects."""

    @staticmethod
    @pytest.fixture()
    def _default_dicts():
        """Return two equivalent deeply nested structures to be modified by tests."""
        a = {
            "foo": np.arange(3),
            "x": {
                0: [1, 2, 3],
            },
            "bar": True,
        }
        b = copy.deepcopy(a)
        return a, b

    @staticmethod
    @pytest.mark.parametrize(
        argnames=["a", "b"],
        argvalues=[
            ("asdf", "foo"),
            (b"asdf", b"foo"),
            (1, 2),
        ],
    )
    def test_compare_nested_raise(a, b):  # noqa: D102
        """non-nested types should raise TypeError."""
        with pytest.raises(TypeError):
            ut.compare_nested(a, b)

    @staticmethod
    @pytest.mark.parametrize(
        argnames="a, b, expected",
        argvalues=[
            ((1, 2, 3), [1, 2, 3], True),
            ((1, 2, 3), [1, 2, 0], False),
            ((1, 2, 3), {"f": 0}, False),
            ((1, 2, 3), "bar", False),
            ({"x": [1, 2, 3, 4]}, {"x": [1, 2]}, False),
            ({"x": [1, 2]}, {"y": [1, 2]}, False),
        ],
    )
    def test_compare_nested(a, b, expected):  # noqa: D102
        assert ut.compare_nested(a, b) == expected

    @staticmethod
    def test_eq_(_default_dicts):  # noqa: D102
        a, b = _default_dicts
        assert ut.compare_nested(a, b)

    @staticmethod
    def test_missing_values(_default_dicts):  # noqa: D102
        a, b = _default_dicts
        b["x"][0].pop(-1)
        assert not ut.compare_nested(a, b)

    @staticmethod
    def test_added_value(_default_dicts):  # noqa: D102
        a, b = _default_dicts
        b["x"][0].append(4)
        assert not ut.compare_nested(a, b)

    @staticmethod
    def test_added_value_left(_default_dicts):  # noqa: D102
        a, b = _default_dicts
        a["x"][0].append(4)
        assert not ut.compare_nested(a, b)

    @staticmethod
    def test_value_changed(_default_dicts):  # noqa: D102
        a, b = _default_dicts
        b["bar"] = False
        assert not ut.compare_nested(a, b)

    @staticmethod
    def test_key_changed1(_default_dicts):  # noqa: D102
        a, b = _default_dicts
        del b["x"]
        assert not ut.compare_nested(a, b)

    @staticmethod
    def test_key_changed2(_default_dicts):  # noqa: D102
        a, b = _default_dicts
        x = b.pop("x")
        b["y"] = x
        assert not ut.compare_nested(a, b)

    @staticmethod
    def test_key_added(_default_dicts):  # noqa: D102
        a, b = (dict(a=1), dict(a=1, b=1))
        assert not ut.compare_nested(a, b)
        assert not ut.compare_nested(b, a)

    @staticmethod
    def test_array_accessible_by_two_roots():  # noqa: D102
        a = {"l1": {"l2": np.arange(5)}}
        b = {"l1": {"l2": np.arange(5)}}
        assert ut.compare_nested(a, b)

    @staticmethod
    def test_arrays_in_lists():  # noqa: D102
        a = {"l1": [np.arange(1), "foo"]}
        b = {"l1": [np.arange(2), "foo"]}
        assert not ut.compare_nested(a, b)


@pytest.mark.usefixtures("single_pass_weld_asdf")
class TestWeldxExampleCompareNested(unittest.TestCase):
    """Test case of a real world example as it compares two nested ASDF trees.

    This includes cases of xarray.DataArrays, np.ndarrays and so forth.
    """

    def setUp(self):  # noqa: D102
        self.a = self.single_pass_weld_tree
        self.b = copy.deepcopy(self.a)

    def test_equal(self):  # noqa: D102
        assert ut.compare_nested(self.a, self.b)

    def test_metadata_modified(self):  # noqa: D102
        self.b["wx_metadata"]["welder"] = "anonymous"
        assert not ut.compare_nested(self.a, self.b)

    def test_measurements_modified(self):  # noqa: D102
        from weldx import Q_

        self.b["welding_current"].data.data[-1] = Q_(500, "A")
        assert not ut.compare_nested(self.a, self.b)

    def test_equip_modified(self):  # noqa: D102
        self.b["equipment"][0].name = "broken device"
        assert not ut.compare_nested(self.a, self.b)

    def test_coordinate_systems_modified(self):  # noqa: D102
        """Manipulate one CSM and check if it gets picked up by comparison."""
        csm_org = self.a["coordinate_systems"]
        csm_copy = self.b["coordinate_systems"]

        # first ensure, that the cs exists, as delete_cs won't tell us.
        assert csm_copy.get_cs("tcp_contact")
        csm_copy.delete_cs("tcp_contact")

        assert csm_copy != csm_org
        assert not ut.compare_nested(self.a, self.b)


def test_is_interactive():
    """Assert that the Pytest session is not recognized as interactive."""
    assert not ut.is_interactive_session()
