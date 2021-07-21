"""Test the `Time` class."""

from typing import List

import numpy as np
import pint
import pytest
from pandas import DatetimeIndex, Timedelta, TimedeltaIndex, Timestamp

from weldx import Q_
from weldx.time import Time


def _initialize_type(cls_type, values, unit):
    """Initialize the passed time type."""
    if cls_type not in (pint.Quantity, Timedelta) and not isinstance(values, List):
        values = [values]
    if cls_type is np.timedelta64:
        return np.array(values, dtype=f"timedelta64[{unit}]")
    if cls_type is Time:
        return Time(Q_(values, unit))
    return cls_type(values, unit)


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
            (Q_([1, 2, 3], "s"), None, False, None),
            (Q_(2, "s"), None, False, None),
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
        ],
    )
    def test_init(time, time_ref, exp_absolute, exp_time_ref):
        """Test initialization of the `Time` class with all supported types."""
        if exp_time_ref is not None:
            exp_time_ref = Timestamp(exp_time_ref)

        t = Time(time, time_ref)

        assert t.is_absolute == exp_absolute
        assert t.reference_time == exp_time_ref

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
        other = _initialize_type(other_type, other_values, unit)

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
