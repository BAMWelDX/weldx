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
    if cls_type not in (pint.Quantity, Timedelta) and not isinstance(values, List):
        values = [values]
    if cls_type is np.timedelta64:
        return np.array(values, dtype=f"timedelta64[{unit}]")
    if cls_type is Time:
        return Time(Q_(values, unit))
    return cls_type(values, unit)


def _is_timedelta(cls_type):
    return cls_type in [Q_, TimedeltaIndex, Timedelta, np.timedelta64] or (
        cls_type is Time and not Time.is_absolute
    )


def _is_datetime(cls_type):
    return not _is_timedelta(cls_type)


class TestTime:
    """Test the time class."""

    # test_init ------------------------------------------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        "time, time_ref, exp_absolute, exp_time_ref",
        [
            (TimedeltaIndex([1, 2, 3]), None, False, None),
            (TimedeltaIndex([1, 2, 3]), Timestamp("16:00"), True, "16:00"),
            (TimedeltaIndex([1, 2, 3]), "16:00", True, "16:00"),
            (Q_(2, "s"), None, False, None),
            (Q_(1, "s"), "16:00", True, "16:00:01"),
            (Q_([1], "s"), None, False, None),
            (Q_([1], "s"), "16:00", True, "16:00:01"),
            (Q_([1, 2, 3], "s"), None, False, None),
            (Q_([1, 2, 3], "s"), "16:00", True, "16:00"),
            (np.array([1, 2, 3], dtype="timedelta64[s]"), None, False, None),
            (np.array([1, 2, 3], dtype="timedelta64[s]"), "16:00", True, "16:00"),
            (DatetimeIndex(["2000", "2010"]), None, True, "2000"),
            (DatetimeIndex(["2000", "2010"]), "2005", True, "2005"),
            (np.array(["2000", "2010"], dtype="datetime64"), None, True, "2000"),
            (np.array(["2000", "2010"], dtype="datetime64"), "2005", True, "2005"),
            (Timestamp("2000"), None, True, "2000"),
            (Timestamp("2000"), "2005", True, "2005"),
            (Timedelta(1, "d"), None, False, None),
            (Timedelta(1, "d"), "2005", True, "2005-01-02"),
            (np.datetime64("2000"), None, True, "2000"),
            (np.datetime64("2000"), "2005", True, "2005"),
            (np.timedelta64(1, "s"), None, False, None),
            (np.timedelta64(1, "s"), "16:00", True, "16:00:01"),
            ("2000", None, True, "2000"),
            ("2000", "2005", True, "2005"),
            (["2000", "2010"], None, True, "2000"),
            (["2000", "2010"], "2005", True, "2005"),
            (["1s", "2s", "3s"], None, False, None),
            (["1s", "2s", "3s"], Timestamp("16:00"), True, "16:00"),
        ],
    )
    def test_init_old(time, time_ref, exp_absolute, exp_time_ref):
        """Test initialization of the `Time` class with all supported types."""
        if exp_time_ref is not None:
            exp_time_ref = Timestamp(exp_time_ref)

        t = Time(time, time_ref)

        assert t.is_absolute == exp_absolute
        assert t.reference_time == exp_time_ref

    @staticmethod
    def _transform_array(data, is_array, is_scalar):
        if not is_array:
            return data[0]
        if is_scalar:
            return [data[0]]
        return data

    @pytest.mark.parametrize("scl, arr", [(True, False), (True, True), (False, True)])
    @pytest.mark.parametrize("set_time_ref", [False, True])
    @pytest.mark.parametrize(
        "input_vals",
        [
            Q_,
            # TimedeltaIndex,
            # Timedelta,
            # np.timedelta64,
            # (str, "timedelta"),
            # (Time, "timedelta"),
            # (Time, "datetime"),
            DatetimeIndex,
        ],
    )
    def test_init(self, input_vals, scl, arr, set_time_ref):
        # analyze test input values
        if isinstance(input_vals, Tuple):
            # to avoid wrong test setups due to spelling mistakes
            assert input_vals[1] in ["timedelta", "datetime"]
            input_type = input_vals[0]
            is_timedelta = input_vals[1] == "timedelta"
        else:
            input_type = input_vals
            is_timedelta = _is_timedelta(input_vals)

        # skip matrix cases that do not work
        if not arr and input_type in [DatetimeIndex, TimedeltaIndex]:
            pytest.skip()

        # set the values passed to the input type
        values = [1, 2, 3]
        if not is_timedelta:
            values = [f"2000-01-01 16:00:0{v}" for v in values]
        values = self._transform_array(values, is_array=arr, is_scalar=scl)

        # create the time input
        if is_timedelta:
            time = _initialize_delta_type(input_type, values, "s")
        else:
            time = input_type(values) if input_type is not str else values

        # create reference time
        time_ref = f"2000-01-01 15:00:00" if set_time_ref else None

        # create `Time` instance

        time_class_instance = Time(time, time_ref)

        # set expected values
        exp_is_absolute = set_time_ref or not is_timedelta

        if exp_is_absolute:
            exp_time_ref = Timestamp(time_ref if set_time_ref else values[0])
        else:
            exp_time_ref = None

        # check
        assert time_class_instance.is_absolute == exp_is_absolute
        assert time_class_instance.reference_time == exp_time_ref

    # todo: issues
    #   - time can be None

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
