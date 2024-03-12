"""Test the `Time` class."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pandas import DatetimeIndex as DTI
from pandas import Timedelta, Timestamp, date_range
from pint import DimensionalityError

from weldx.constants import Q_
from weldx.core import TimeSeries
from weldx.time import Time, types_time_like
from weldx.transformations.local_cs import LocalCoordinateSystem


def _initialize_delta_type(cls_type, values, unit):
    """Initialize the passed time type."""
    if cls_type is np.timedelta64:
        if isinstance(values, list):
            return np.array(values, dtype=f"timedelta64[{unit}]")
        return np.timedelta64(values, unit)
    if cls_type is Time:
        return Time(Q_(values, unit))
    if cls_type is str:
        if not isinstance(values, list):
            return f"{values}{unit}"
        return [f"{v}{unit}" for v in values]
    return cls_type(values, unit)


def _initialize_datetime_type(cls_type, values):
    """Initialize the passed datetime type."""
    if cls_type is np.datetime64:
        if isinstance(values, list):
            return np.array(values, dtype="datetime64")
        return np.datetime64(values)
    if cls_type is str:
        return values
    return cls_type(values)


def _initialize_date_time_quantity(timedelta, unit, time_ref):
    """Initialize a quantity that represents a datetime by adding a ``time_ref``."""
    quantity = Q_(timedelta, unit)
    quantity.time_ref = Timestamp(time_ref)
    return quantity


def _transform_array(data, is_array, is_scalar):
    """Transform an array into a scalar, single value array or return in unmodified."""
    if not is_array:
        return data[0]
    if is_scalar:
        return [data[0]]
    return data


def _initialize_time_type(
    input_type, delta_val, abs_val, is_timedelta, is_array, is_scalar, unit="s"
):
    """Create an instance of the desired input type."""
    val = delta_val if is_timedelta else abs_val
    if not is_timedelta and input_type is Q_:
        val = [v - delta_val[0] for v in delta_val]
    val = _transform_array(val, is_array=is_array, is_scalar=is_scalar)

    # create the time input ---------------------------------
    if is_timedelta:
        return _initialize_delta_type(input_type, val, unit)
    if input_type is not Q_:
        return _initialize_datetime_type(input_type, val)
    return _initialize_date_time_quantity(val, unit, abs_val[0])


def _is_timedelta(cls_type):
    """Return ``True`` if the passed type is a timedelta type."""
    return cls_type in [pd.to_timedelta, Timedelta, np.timedelta64] or (
        cls_type is Time and not Time.is_absolute
    )


def _is_datetime(cls_type):
    """Return ``True`` if the passed type is a datetime type."""
    return not _is_timedelta(cls_type)


class TestTime:
    """Test the time class."""

    # test_init helper functions -------------------------------------------------------

    @staticmethod
    def _parse_time_type_test_input(
        type_input,
    ) -> tuple[types_time_like | Time, bool]:
        """Return the time type and a bool that defines if the returned type is a delta.

        This is mainly used in generalized tests where a type like `Time` itself can
        represent deltas and absolute times. In this case one can use this function
        to extract the information from a tuple.

        """
        if isinstance(type_input, tuple):
            # to avoid wrong test setups due to spelling mistakes
            assert type_input[1] in ["timedelta", "datetime"]
            time_type = type_input[0]
            is_timedelta = type_input[1] == "timedelta"
        else:
            time_type = type_input
            is_timedelta = _is_timedelta(type_input)
        return time_type, is_timedelta

    @classmethod
    def _get_init_exp_values(
        cls,
        is_timedelta,
        time_ref,
        data_was_scalar,
        delta_val,
        abs_val,
    ):
        """Get the expected result values for the `__init__` test."""
        exp_is_absolute = time_ref is not None or not is_timedelta

        # expected reference time
        exp_time_ref = None
        if exp_is_absolute:
            exp_time_ref = Timestamp(time_ref if time_ref is not None else abs_val[0])

        # expected time delta values
        val = delta_val
        if exp_is_absolute:
            offset = 0
            if not is_timedelta:
                if time_ref is not None:
                    offset = Timestamp(abs_val[0]) - Timestamp(time_ref)
                    offset = offset.total_seconds()
                offset -= delta_val[0]

            val = [v + offset for v in delta_val]

        val = val[0] if data_was_scalar else val
        exp_timedelta = (
            Timedelta(val, "s") if data_was_scalar else pd.to_timedelta(val, "s")
        )

        # expected datetime
        exp_datetime = None
        if exp_is_absolute:
            time_ref = Timestamp(time_ref if time_ref is not None else abs_val[0])
            exp_datetime = time_ref + exp_timedelta

        return dict(
            is_absolute=exp_is_absolute,
            time_ref=exp_time_ref,
            timedelta=exp_timedelta,
            datetime=exp_datetime,
        )

    # test_init ------------------------------------------------------------------------

    @pytest.mark.parametrize("scl, arr", [(True, False), (True, True), (False, True)])
    @pytest.mark.parametrize("set_time_ref", [False, True])
    @pytest.mark.parametrize(
        "input_vals",
        [
            (str, "timedelta"),
            (Time, "timedelta"),
            (Q_, "timedelta"),
            pd.to_timedelta,
            Timedelta,
            np.timedelta64,
            (str, "datetime"),
            (Time, "datetime"),
            (Q_, "datetime"),
            DTI,
            Timestamp,
            np.datetime64,
        ],
    )
    def test_init(
        self,
        input_vals: type | tuple[type, str],
        set_time_ref: bool,
        scl: bool,
        arr: bool,
    ):
        """Test the `__init__` method of the time class.

        Parameters
        ----------
        input_vals :
            Either a compatible time type or a tuple of two values. The tuple is needed
            in case the tested time type can either represent relative time values as
            well as absolute ones. In this case, the first value is the type. The
            second value is a string specifying if the type represents absolute
            ("datetime") or relative ("timedelta") values.
        set_time_ref :
            If `True`, a reference time will be passed to the `__init__` method
        scl :
            If `True`, the data of the passed type consists of a single value.
        arr :
            If `True`, the data of the passed type is an array

        """
        input_type, is_timedelta = self._parse_time_type_test_input(input_vals)

        # skip matrix cases that do not work --------------------
        if arr and input_type in [Timedelta, Timestamp]:
            return
        if not arr and input_type in [DTI, pd.to_timedelta]:
            return

        # create input values -----------------------------------
        delta_val = [1, 2, 3]
        abs_val = [f"2000-01-01 16:00:0{v}" for v in delta_val]

        time = _initialize_time_type(
            input_type, delta_val, abs_val, is_timedelta, arr, scl
        )
        time_ref = "2000-01-01 15:00:00" if set_time_ref else None

        # create `Time` instance --------------------------------
        time_class_instance = Time(time, time_ref)

        # check results -----------------------------------------
        exp = self._get_init_exp_values(is_timedelta, time_ref, scl, delta_val, abs_val)

        assert time_class_instance.is_absolute == exp["is_absolute"]
        assert time_class_instance.reference_time == exp["time_ref"]
        assert np.all(time_class_instance.as_timedelta() == exp["timedelta"])
        if exp["is_absolute"]:
            assert np.all(time_class_instance.as_datetime() == exp["datetime"])
        else:
            with pytest.raises(TypeError):
                time_class_instance.as_datetime()

    # test_init_from_time_dependent_types ----------------------------------------------
    @staticmethod
    @pytest.mark.parametrize(
        "time_dep_type",
        [
            LocalCoordinateSystem(
                coordinates=Q_(np.zeros((2, 3)), "mm"), time=["2s", "3s"]
            ),
            LocalCoordinateSystem(
                coordinates=Q_(np.zeros((2, 3)), "mm"), time=["2000", "2001"]
            ),
            TimeSeries(Q_([2, 4, 1], "m"), pd.to_timedelta([1, 2, 3], "s")),
            TimeSeries(Q_([2, 4, 1], "m"), ["2001", "2002", "2003"]),
        ],
    )
    def test_init_from_time_dependent_types(time_dep_type):
        """Test initialization with types derived from `TimeDependent`."""
        t = Time(time_dep_type)
        assert np.all(t == time_dep_type.time)

    # test_init_exceptions -------------------------------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        "time, time_ref, raises",
        [
            (pd.to_timedelta([3, 2, 1]), None, ValueError),
            (DTI(["2010", "2000"]), None, ValueError),
            (["2010", "2000"], None, ValueError),
            (Q_([3, 2, 1], "s"), None, ValueError),
            (np.array([3, 2, 1], dtype="timedelta64[s]"), None, ValueError),
            (None, None, TypeError),
            (5, None, TypeError),
            ("string", None, TypeError),
            (Q_(10, "m"), None, DimensionalityError),
        ],
    )
    def test_init_exception(time, time_ref, raises):
        """Test initialization of the `Time` class with all supported types."""
        with pytest.raises(raises):
            Time(time, time_ref)

    # test_add_timedelta ---------------------------------------------------------------

    @pytest.mark.parametrize("other_on_rhs", [True, False])
    @pytest.mark.parametrize("time_class_is_array", [False, True])
    @pytest.mark.parametrize("other_is_array", [False, True])
    @pytest.mark.parametrize("unit", ["s", "h"])
    @pytest.mark.parametrize(
        "other_type",
        [
            (str, "timedelta"),
            (Time, "timedelta"),
            (Q_, "timedelta"),
            pd.to_timedelta,
            Timedelta,
            np.timedelta64,
            (str, "datetime"),
            (Time, "datetime"),
            (Q_, "datetime"),
            DTI,
            Timestamp,
            np.datetime64,
        ],
    )
    def test_add_timedelta(
        self,
        other_type,
        other_on_rhs: bool,
        unit: str,
        time_class_is_array: bool,
        other_is_array: bool,
    ):
        """Test the `__add__` method if the `Time` class represents a time delta.

        Parameters
        ----------
        other_type :
            The type of the other object
        other_on_rhs :
            If `True`, the other type is on the rhs of the + sign and on the lhs
            otherwise
        unit :
            The time unit to use
        time_class_is_array :
            If `True`, the `Time` instance contains 3 time values and 1 otherwise
        other_is_array :
            If `True`, the other time object contains 3 time values and 1 otherwise

        """
        other_type, is_timedelta = self._parse_time_type_test_input(other_type)

        # skip array cases where the type does not support arrays
        if other_type in [Timedelta, Timestamp] and other_is_array:
            return
        if not other_is_array and other_type in [DTI, pd.to_timedelta]:
            return

        # skip __radd__ cases where we got conflicts with the other types' __add__
        if not other_on_rhs and other_type in (
            Q_,
            np.ndarray,
            np.timedelta64,
            np.datetime64,
            DTI,
            pd.to_timedelta,
        ):
            return

        # setup rhs
        delta_val = [4, 6, 8]
        if unit == "s":
            abs_val = [f"2000-01-01 10:00:0{v}" for v in delta_val]
        else:
            abs_val = [f"2000-01-01 1{v}:00:00" for v in delta_val]
        other = _initialize_time_type(
            other_type,
            delta_val,
            abs_val,
            is_timedelta,
            other_is_array,
            not other_is_array,
            unit,
        )

        # setup lhs
        time_class_values = [1, 2, 3] if time_class_is_array else [1]
        time_class = Time(Q_(time_class_values, unit))

        # setup expected values
        add = delta_val if other_is_array else delta_val[0]
        exp_val = np.array(time_class_values) + add
        exp_val += 0 if is_timedelta else time_class_values[0] - exp_val[0]

        exp_time_ref = None if is_timedelta else abs_val[0]
        exp = Time(Q_(exp_val, unit), exp_time_ref)

        # calculate and evaluate result
        res = time_class + other if other_on_rhs else other + time_class

        assert res.reference_time == exp.reference_time
        assert np.all(res.as_timedelta() == exp.as_timedelta())
        assert np.all(res == exp)

    # test_add_datetime ----------------------------------------------------------------

    @staticmethod
    @pytest.mark.parametrize("other_on_rhs", [True, False])
    @pytest.mark.parametrize("time_class_is_array", [False, True])
    @pytest.mark.parametrize("other_is_array", [False, True])
    @pytest.mark.parametrize(
        "other_type",
        [
            str,
            Time,
            Q_,
            pd.to_timedelta,
            Timedelta,
            np.timedelta64,
        ],
    )
    def test_add_datetime(
        other_type,
        other_on_rhs: bool,
        time_class_is_array: bool,
        other_is_array: bool,
    ):
        """Test the `__add__` method if the `Time` class represents a datetime.

        Parameters
        ----------
        other_type :
            The type of the other object
        other_on_rhs :
            If `True`, the other type is on the rhs of the + sign and on the lhs
            otherwise
        time_class_is_array :
            If `True`, the `Time` instance contains 3 time values and 1 otherwise
        other_is_array :
            If `True`, the other time object contains 3 time values and 1 otherwise

        """
        # skip array cases where the type does not support arrays
        if other_type in [Timedelta, Timestamp] and other_is_array:
            return
        if not other_is_array and other_type in [DTI, pd.to_timedelta]:
            return

        # skip __radd__ cases where we got conflicts with the other types' __add__
        if not other_on_rhs and other_type in (
            Q_,
            np.ndarray,
            np.timedelta64,
            pd.to_timedelta,
        ):
            return

        # setup rhs
        delta_val = [4, 6, 8]

        other = _initialize_time_type(
            other_type,
            delta_val,
            None,
            True,
            other_is_array,
            not other_is_array,
            "s",
        )

        # setup lhs
        time_class_values = [1, 2, 3] if time_class_is_array else [1]
        time_class = Time(Q_(time_class_values, "s"), "2000-01-01 10:00:00")

        # setup expected values
        add = delta_val if other_is_array else delta_val[0]
        exp_val = np.array(time_class_values) + add

        exp_time_ref = time_class.reference_time
        exp = Time(Q_(exp_val, "s"), exp_time_ref)

        # calculate and evaluate result
        res = time_class + other if other_on_rhs else other + time_class

        assert res.reference_time == exp.reference_time
        assert np.all(res.as_timedelta() == exp.as_timedelta())
        assert np.all(res == exp)

    # test_sub -------------------------------------------------------------------------

    @staticmethod
    def _date_diff(date_1: str, date_2: str, unit: str) -> int:
        """Calculate the diff between two dates in the specified unit."""
        return int(Time(Timestamp(date_1) - Timestamp(date_2)).as_quantity().m_as(unit))

    @pytest.mark.parametrize("other_on_rhs", [False, True])
    @pytest.mark.parametrize("time_class_is_array", [False, True])
    @pytest.mark.parametrize("other_is_array", [False, True])
    @pytest.mark.parametrize("unit", ["s", "h"])
    @pytest.mark.parametrize("time_class_is_timedelta", [False, True])
    @pytest.mark.parametrize(
        "other_type",
        [
            (str, "timedelta"),
            (Time, "timedelta"),
            (Q_, "timedelta"),
            pd.to_timedelta,
            Timedelta,
            np.timedelta64,
            (str, "datetime"),
            (Time, "datetime"),
            (Q_, "datetime"),
            DTI,
            Timestamp,
            np.datetime64,
        ],
    )
    def test_sub(
        self,
        other_type,
        other_on_rhs: bool,
        unit: str,
        time_class_is_array: bool,
        time_class_is_timedelta: bool,
        other_is_array: bool,
    ):
        """Test the `__sub__` method of the `Time` class.

        Parameters
        ----------
        other_type :
            The type of the other object
        other_on_rhs :
            If `True`, the other type is on the rhs of the + sign and on the lhs
            otherwise
        unit :
            The time unit to use
        time_class_is_array :
            If `True`, the `Time` instance contains 3 time values and 1 otherwise
        time_class_is_timedelta :
            If `True`, the `Time` instance represents a time delta and a datetime
            otherwise
        other_is_array :
            If `True`, the other time object contains 3 time values and 1 otherwise

        """
        other_type, other_is_timedelta = self._parse_time_type_test_input(other_type)
        if other_on_rhs:
            lhs_is_array = time_class_is_array
            lhs_is_timedelta = time_class_is_timedelta
            rhs_is_array = other_is_array
            rhs_is_timedelta = other_is_timedelta
        else:
            lhs_is_array = other_is_array
            lhs_is_timedelta = other_is_timedelta
            rhs_is_array = time_class_is_array
            rhs_is_timedelta = time_class_is_timedelta

        # skip array cases where the type does not support arrays or scalars
        if other_type in [Timedelta, Timestamp] and other_is_array:
            return
        if not other_is_array and other_type in [DTI, pd.to_timedelta]:
            return

        # skip __rsub__ cases where we got conflicts with the other types' __sub__
        if not other_on_rhs and other_type in (
            Q_,
            np.ndarray,
            np.timedelta64,
            np.datetime64,
            DTI,
            pd.to_timedelta,
        ):
            return

        # skip cases where an absolute time is on the rhs, since pandas does
        # not support this case (and it does not make sense)
        if lhs_is_timedelta and not rhs_is_timedelta:
            return

        # skip cases where the lhs is a scalar and the rhs is an array because it will
        # always involve non monotonically increasing array values, which is forbidden.
        if rhs_is_array and not lhs_is_array:
            return

        # test values
        vals_lhs = [3, 5, 9] if lhs_is_array else [3]
        vals_rhs = [1, 2, 3] if rhs_is_array else [1]

        # setup rhs
        other_val = vals_rhs if other_on_rhs else vals_lhs
        if unit == "s":
            abs_val = [f"2000-01-01 10:00:0{v}" for v in other_val]
        else:
            abs_val = [f"2000-01-01 1{v}:00:00" for v in other_val]
        other = _initialize_time_type(
            other_type,
            other_val,
            abs_val,
            other_is_timedelta,
            other_is_array,
            not other_is_array,
            unit,
        )

        # setup lhs
        time_class_values = vals_lhs if other_on_rhs else vals_rhs
        time_class_time_ref = None if time_class_is_timedelta else "2000-01-01 11:00:00"
        time_class = Time(Q_(time_class_values, unit), time_class_time_ref)

        # setup expected values
        sub = vals_rhs if other_is_array else vals_rhs[0]
        exp_val = np.array(vals_lhs) - sub
        if not other_is_timedelta:
            if time_class_is_timedelta:
                exp_val -= time_class_values[0] + exp_val[0]
            else:
                d = self._date_diff(time_class_time_ref, abs_val[0], unit) + vals_rhs[0]
                exp_val += d if other_on_rhs else (d + exp_val[0]) * -1

        exp_time_ref = None
        if not other_is_timedelta and time_class_is_timedelta:
            exp_time_ref = abs_val[0]
        elif other_is_timedelta and not time_class_is_timedelta:
            exp_time_ref = time_class_time_ref
        exp = Time(Q_(exp_val, unit), exp_time_ref)

        # calculate and evaluate result
        res = time_class - other if other_on_rhs else other - time_class

        assert res.reference_time == exp.reference_time
        assert np.all(res.as_timedelta() == exp.as_timedelta())
        assert np.all(res == exp)

    # test_pandas_index ----------------------------------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        "arg, expected",
        [
            # timedeltas
            (pd.to_timedelta([42], unit="ns"), pd.to_timedelta([42], unit="ns")),
            (pd.timedelta_range("0s", "20s", 10), pd.timedelta_range("0s", "20s", 10)),
            (np.timedelta64(42), pd.to_timedelta([42], unit="ns")),
            (
                np.array([-10, 0, 20]).astype("timedelta64[ns]"),
                pd.to_timedelta([-10, 0, 20], "ns"),
            ),
            (Q_(42, "ns"), pd.to_timedelta([42], unit="ns")),
            ("10s", pd.to_timedelta(["10s"])),
            (["5ms", "10s", "2D"], pd.to_timedelta(["5 ms", "10s", "2D"])),
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
    def test_pandas_index(arg, expected):
        """Test conversion to appropriate pd.TimedeltaIndex or pd.DatetimeIndex."""
        t = Time(arg)
        assert np.all(t.as_pandas_index() == expected)
        assert np.all(t.as_pandas_index() == t.index)
        assert np.all(t.as_timedelta_index() == t.timedelta)

    # test_as_quantity -----------------------------------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        "arg, unit, expected",
        [
            ("1s", "s", 1),
            ("1s", "ms", 1000),
            ("1s", "us", 1000000),
            ("1s", "ns", 1000000000),
            (pd.to_timedelta([1, 2, 3], "s"), "s", [1, 2, 3]),
            (pd.to_timedelta([1, 2, 3], "s"), "ms", np.array([1, 2, 3]) * 1e3),
            (pd.to_timedelta([1, 2, 3], "s"), "us", np.array([1, 2, 3]) * 1e6),
            (pd.to_timedelta([1, 2, 3], "s"), "ns", np.array([1, 2, 3]) * 1e9),
            ("2020-01-01", "s", 0),
        ],
    )
    def test_quantity(arg, unit, expected):
        """Test conversion to pint.Quantity with different scales."""
        t = Time(arg)
        q = Time(arg).as_quantity(unit)
        expected = Q_(expected, unit)
        assert np.allclose(q, expected)
        if t.is_absolute:
            assert t.reference_time == q.time_ref
        if unit == "s":
            assert np.all(q == t.quantity)

    # test_convert_util ----------------------------------------------------------------

    @staticmethod
    def test_convert_util():
        """Test basic conversion functions from/to xarray/pint."""
        t = pd.date_range("2020", periods=10, freq="1s")
        ts = t[0]

        arr = xr.DataArray(
            np.arange(10),
            dims=["time"],
            coords={"time": t - ts},
        )
        arr.time.weldx.time_ref = ts
        time = Time(arr)

        assert len(time) == len(t)
        assert time.equals(Time(t))

        time_q = time.as_quantity()
        assert np.all(time_q == Q_(range(10), "s"))
        assert time_q.time_ref == ts

        arr2 = time.as_data_array()
        assert arr.time.identical(arr2.time)

    # test_duration --------------------------------------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        "values, exp_duration",
        [
            ("1s", "0s"),
            ("2000-01-01", "0s"),
            (Q_([3, 5, 7, 8], "s"), "5s"),
            (["2000-01-03", "2000-01-05", "2000-01-07", "2000-01-08"], "5days"),
        ],
    )
    def test_duration(values, exp_duration):
        """Test the duration property."""
        t = Time(values)
        assert t.duration.all_close(exp_duration)

    # test_resample --------------------------------------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        "number_or_interval, exp_values",
        [
            # test clipping
            ("10s", [3, 9]),
            # test edge case
            (2, [3, 9]),
            ("6s", [3, 9]),
            # test even time deltas
            (4, [3, 5, 7, 9]),
            ("2s", [3, 5, 7, 9]),
            # test uneven time deltas
            ("5s", [3, 8, 9]),
            ("2.2s", [3, 5.2, 7.4, 9]),
        ],
    )
    def test_resample(number_or_interval, exp_values):
        """Test resample method."""
        t_ref = "2000-01-01"
        t_delta = Time(Q_([3, 5, 8, 9], "s"))
        t_abs = t_delta + t_ref

        exp_delta = Time(Q_(exp_values, "s"))
        exp_abs = exp_delta + t_ref

        result_delta = t_delta.resample(number_or_interval)
        result_abs = t_abs.resample(number_or_interval)

        assert result_delta.all_close(exp_delta)
        assert result_abs.all_close(exp_abs)

    # test_resample_exceptions ---------------------------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        "values,number_or_interval, raises",
        [
            ("4s", 2, RuntimeError),
            ("2000-02-01", 2, RuntimeError),
            (["4s", "10s"], "no time", TypeError),
            (["4s", "10s"], 1, ValueError),
            (["4s", "10s"], "0s", ValueError),
            (["4s", "10s"], "-2s", ValueError),
        ],
    )
    def test_resample_exceptions(values, number_or_interval, raises):
        """Test possible exceptions of the resample method."""
        with pytest.raises(raises):
            Time(values).resample(number_or_interval)

    # test_union -----------------------------------------------------------------------

    @staticmethod
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
            (
                [
                    pd.to_timedelta([1, 5]),
                    pd.to_timedelta([2, 6, 7]),
                    pd.to_timedelta([1, 3, 7]),
                ],
                pd.to_timedelta([1, 2, 3, 5, 6, 7]),
            ),
        ],
    )
    @pytest.mark.parametrize("test_instance", [True, False])
    def test_union(test_instance, list_of_objects, time_exp):
        """Test input types for Time.union function.

        Parameters
        ----------
        list_of_objects:
            List with input objects
        time_exp:
            Expected result time

        """
        if test_instance:
            instance = Time(list_of_objects[0])
            assert np.all(instance.union(list_of_objects[1:]) == time_exp)
        else:
            assert np.all(Time.union(list_of_objects) == time_exp)
