"""Test the `Time` class."""

from typing import List, Tuple

import numpy as np
import pint
import pytest
from pandas import DatetimeIndex, Timedelta, TimedeltaIndex, Timestamp

from weldx import Q_
from weldx.time import Time


def _initialize_delta_type(cls_type, values, unit):
    """Initialize the passed time type."""
    if cls_type not in (Q_, Timedelta) and not isinstance(values, List):
        values = [values]
    if cls_type is np.timedelta64:
        return np.array(values, dtype=f"timedelta64[{unit}]")
    if cls_type is Time:
        return Time(Q_(values, unit))
    if cls_type is str:
        return [f"{v}{unit}" for v in values]
    return cls_type(values, unit)


def _initialize_datetime_type(cls_type, values):
    if cls_type is np.datetime64:
        return np.array(values, dtype="datetime64")
    if cls_type is str:
        return values
    return cls_type(values)


def _initialize_date_time_quantity(timedelta, unit, time_ref):
    quantity = Q_(timedelta, unit)
    setattr(quantity, "time_ref", Timestamp(time_ref))
    print(quantity)
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

    # test_init ------------------------------------------------------------------------

    _create_input_type():
        pass

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
            # np.datetime64,  # 2 failures since util cant convert single vals to index
        ],
    )
    def test_init(self, input_vals, scl, arr, set_time_ref):
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

        # set the values passed to the input type ---------------
        delta_val = [1, 2, 3]
        if not is_timedelta:
            abs_val = [f"2000-01-01 16:00:0{v}" for v in delta_val]

        val = delta_val if is_timedelta else abs_val
        if not is_timedelta and input_type is Q_:
            val = [v - 1 for v in delta_val]
        val = self._transform_array(val, is_array=arr, is_scalar=scl)

        # create the time input ---------------------------------
        if is_timedelta:
            time = _initialize_delta_type(input_type, val, "s")
        else:
            if input_type is not Q_:
                time = _initialize_datetime_type(input_type, val)
            else:
                time = _initialize_date_time_quantity(val, "s", abs_val[0])

        # create reference time ---------------------------------
        time_ref = f"2000-01-01 15:00:00" if set_time_ref else None

        # create `Time` instance --------------------------------
        time_class_instance = Time(time, time_ref)

        # set expected values -----------------------------------
        exp_is_absolute = set_time_ref or not is_timedelta

        exp_time_ref = None
        if exp_is_absolute:
            exp_time_ref = Timestamp(time_ref if set_time_ref else abs_val[0])

        val = delta_val
        if exp_is_absolute:
            offset = 0 if is_timedelta else 3600 if set_time_ref else -delta_val[0]
            val = [v + offset for v in delta_val]
        val = val[0] if scl else val
        exp_timedelta = Timedelta(val, "s") if scl else TimedeltaIndex(val, "s")

        exp_datetime = None
        if exp_is_absolute:
            time_ref = Timestamp(time_ref if set_time_ref else abs_val[0])
            exp_datetime = time_ref + exp_timedelta

        # check -------------------------------------------------
        assert time_class_instance.is_absolute == exp_is_absolute
        assert time_class_instance.reference_time == exp_time_ref
        assert np.all(time_class_instance.as_timedelta() == exp_timedelta)
        if exp_is_absolute:
            assert np.all(time_class_instance.as_datetime() == exp_datetime)
        else:
            with pytest.raises(TypeError):
                time_class_instance.as_datetime()

    # todo: issues
    #   - time parameter can be None

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
