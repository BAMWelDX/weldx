"""Test the `Time` class."""

import numpy as np
import pytest
from pandas import DatetimeIndex, Timedelta, TimedeltaIndex, Timestamp, date_range

from weldx import Q_
from weldx.time import Time


class TestTime:
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
        if exp_time_ref is not None:
            exp_time_ref = Timestamp(exp_time_ref)

        t = Time(time, time_ref)

        assert t.is_absolute == exp_absolute
        assert t.reference_time == exp_time_ref

    @staticmethod
    @pytest.mark.parametrize(
        "lhs_args, rhs_args, exp_args",
        [
            ([Q_([1, 2], "s")], [Q_("2s")], [Q_([3, 4], "s")]),
        ],
    )
    def test_add(lhs_args, rhs_args, exp_args):
        lhs = Time(*lhs_args)
        rhs = Time(*rhs_args)
        exp = Time(*exp_args)

        res = lhs + rhs

        assert np.all(res.as_pandas() == exp.as_pandas())
        assert np.all(res == exp)
        assert res.reference_time == exp.reference_time
