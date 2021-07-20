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
            (Timestamp("2000"), "2005", True, "2000"),
            (Timedelta(1, "d"), None, False, None),
            (Timedelta(1, "d"), "2005", True, "2005-01-02"),
            (np.datetime64("2000"), None, True, "2000"),
            (np.datetime64("2000"), "2005", True, "2000"),
            (np.timedelta64(1, "s"), None, False, None),
            (np.timedelta64(1, "s"), "16:00", True, "16:00:01"),
        ],
    )
    def test_init(time, time_ref, exp_absolute, exp_time_ref):
        """Test initialization of the `Time` class with all supported types."""
        if exp_time_ref is not None:
            exp_time_ref = Timestamp(exp_time_ref)

        t = Time(time, time_ref)

        assert t.is_absolute == exp_absolute
        assert t.reference_time == exp_time_ref

    # test_add_timedelta ---------------------------------------------------------------

    @staticmethod
    @pytest.mark.parametrize("lhs_is_array", [False, True])
    @pytest.mark.parametrize("rhs_is_array", [False, True])
    @pytest.mark.parametrize("unit", ["s", "h"])
    @pytest.mark.parametrize(
        "rhs_type", [Q_, TimedeltaIndex, Timedelta, np.timedelta64]
    )
    def test_add_timedelta(rhs_type, unit, lhs_is_array, rhs_is_array):
        """Test the `__add__` method if the `Time` class represents a time delta.

        Parameters
        ----------
        rhs_type :
            The type on the right hand side
        unit :
            The time unit to use
        lhs_is_array :
            If `True`, the lhs contains 3 time values and 1 otherwise
        rhs_is_array :
            If `True`, the rhs contains 3 time values and 1 otherwise

        """
        # skip non-working matrix combination
        if rhs_type is Timedelta and rhs_is_array:
            pytest.skip()

        # setup rhs
        rhs_values = [1, 100, 10000] if rhs_is_array else 10
        rhs = Time(_initialize_type(rhs_type, rhs_values, unit))

        # setup lhs
        lhs_values = [1, 2, 3] if lhs_is_array else 1
        lhs = Time(Q_(lhs_values, unit))

        # setup expected values
        exp_val = np.array(lhs_values) + rhs_values
        exp = Time(Q_(exp_val, unit))

        res = lhs + rhs

        assert np.all(res.as_pandas() == exp.as_pandas())
        assert np.all(res == exp)
