"""Test the `Time` class."""

from typing import List, Tuple, Type, Union

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pandas import DatetimeIndex, Timedelta, TimedeltaIndex, Timestamp

from weldx import Q_
from weldx.time import Time


def _initialize_delta_type(cls_type, values, unit):
    """Initialize the passed time type."""
    if cls_type not in (Q_, Timedelta, np.timedelta64) and not isinstance(values, List):
        values = [values]
    if cls_type is np.timedelta64:
        if isinstance(values, List):
            return np.array(values, dtype=f"timedelta64[{unit}]")
        return np.timedelta64(values, unit)
    if cls_type is Time:
        return Time(Q_(values, unit))
    if cls_type is str:
        return [f"{v}{unit}" for v in values]
    return cls_type(values, unit)


def _initialize_datetime_type(cls_type, values):
    if cls_type is np.datetime64:
        if isinstance(values, List):
            return np.array(values, dtype="datetime64")
        return np.datetime64(values)
    if cls_type is str:
        return values
    return cls_type(values)


def _initialize_date_time_quantity(timedelta, unit, time_ref):
    quantity = Q_(timedelta, unit)
    setattr(quantity, "time_ref", Timestamp(time_ref))
    return quantity


def _is_timedelta(cls_type):
    # todo: Q_ must be checked for attribute
    return cls_type in [Q_, TimedeltaIndex, Timedelta, np.timedelta64] or (
        cls_type is Time and not Time.is_absolute
    )


def _is_datetime(cls_type):
    return not _is_timedelta(cls_type)


class TestTime:
    """Test the time class."""

    # test_init helper functions -------------------------------------------------------

    @staticmethod
    def _transform_array(data, is_array, is_scalar):
        if not is_array:
            return data[0]
        if is_scalar:
            return [data[0]]
        return data

    @classmethod
    def _create_init_input_type(
        cls, input_type, delta_val, abs_val, is_timedelta, is_array, is_scalar
    ):
        """Create an instance of the desired input type for the `__init__` test."""
        val = delta_val if is_timedelta else abs_val
        if not is_timedelta and input_type is Q_:
            val = [v - 1 for v in delta_val]
        val = cls._transform_array(val, is_array=is_array, is_scalar=is_scalar)

        # create the time input ---------------------------------
        if is_timedelta:
            return _initialize_delta_type(input_type, val, "s")
        if input_type is not Q_:
            return _initialize_datetime_type(input_type, val)
        return _initialize_date_time_quantity(val, "s", abs_val[0])

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
            Timedelta(val, "s") if data_was_scalar else TimedeltaIndex(val, "s")
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
            TimedeltaIndex,
            Timedelta,
            np.timedelta64,
            (str, "datetime"),
            (Time, "datetime"),
            (Q_, "datetime"),
            DatetimeIndex,
            Timestamp,
            np.datetime64,
        ],
    )
    def test_init(
        self,
        input_vals: Union[Type, Tuple[Type, str]],
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
        # analyze test input values -----------------------------
        if isinstance(input_vals, Tuple):
            # to avoid wrong test setups due to spelling mistakes
            assert input_vals[1] in ["timedelta", "datetime"]
            input_type = input_vals[0]
            is_timedelta = input_vals[1] == "timedelta"
        else:
            input_type = input_vals
            is_timedelta = _is_timedelta(input_vals)

        # skip matrix cases that do not work --------------------
        if arr and input_type in [Timedelta, Timestamp]:
            pytest.skip()
        if not arr and input_type in [DatetimeIndex, TimedeltaIndex]:
            pytest.skip()

        # create input values -----------------------------------
        delta_val = [1, 2, 3]
        abs_val = [f"2000-01-01 16:00:0{v}" for v in delta_val]

        time = self._create_init_input_type(
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

    # todo: issues
    #   - time parameter can be None

    # test_init_exceptions -------------------------------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        "time, time_ref, raises",
        [
            (TimedeltaIndex([3, 2, 1]), None, ValueError),
            (DatetimeIndex(["2010", "2000"]), None, ValueError),
            (["2010", "2000"], None, ValueError),
            (Q_([3, 2, 1], "s"), None, ValueError),
            (np.array([3, 2, 1], dtype="timedelta64[s]"), None, ValueError),
        ],
    )
    def test_init_exception(time, time_ref, raises):
        """Test initialization of the `Time` class with all supported types."""
        with pytest.raises(raises):
            Time(time, time_ref)

    # test_add_timedelta ---------------------------------------------------------------

    @staticmethod
    @pytest.mark.parametrize("other_on_rhs", [True, False])
    @pytest.mark.parametrize("time_class_is_array", [False, True])
    @pytest.mark.parametrize("other_is_array", [False, True])
    @pytest.mark.parametrize("unit", ["s", "h"])
    @pytest.mark.parametrize(
        "other_type", [Q_, TimedeltaIndex, Timedelta, np.timedelta64, Time]
    )
    def test_add_timedelta(
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
        # skip array cases where the type does not support arrays
        if other_type is Timedelta and other_is_array:
            pytest.skip()
        # skip __radd__ cases where we got conflicts with the other types' __add__
        if not other_on_rhs and other_type in (Q_, np.ndarray, np.timedelta64):
            pytest.skip()

        # setup rhs
        other_values = [1, 100, 10000] if other_is_array else 10
        other = _initialize_delta_type(other_type, other_values, unit)

        # setup lhs
        time_class_values = [1, 2, 3] if time_class_is_array else 1
        time_class = Time(Q_(time_class_values, unit))

        # setup expected values
        exp_val = np.array(time_class_values) + other_values
        exp = Time(Q_(exp_val, unit))

        # calculate and evaluate result
        res = time_class + other if other_on_rhs else other + time_class

        assert np.all(res.as_pandas() == exp.as_pandas())
        assert np.all(res == exp)

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

        assert time.length == len(t)
        assert time.equals(Time(t))

        time_q = time.as_quantity()
        assert np.all(time_q == Q_(range(10), "s"))
        assert time_q.time_ref == ts

        arr2 = time.as_data_array().weldx.time_ref_restore()
        assert arr.time.identical(arr2.time)
