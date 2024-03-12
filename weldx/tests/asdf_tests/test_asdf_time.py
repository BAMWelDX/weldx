"""Test time schema implementation."""

import numpy as np
import pandas as pd
import pytest

from weldx.asdf.util import write_buffer, write_read_buffer_context
from weldx.time import Time


@pytest.mark.parametrize(
    "inputs",
    [
        pd.Timedelta("5m3ns"),
        pd.Timedelta.max - pd.Timedelta("10s"),
        pd.timedelta_range(start="-5s", end="25s", freq="3s"),
        pd.TimedeltaIndex([0, 1e9, 5e9, 3e9]),
        pd.Timestamp("2020-04-15T16:47:00.000000001"),
        pd.Timestamp("2020-04-15T16:47:00.000000001", tz="Europe/Berlin"),
        pd.date_range(start="2020-01-01", periods=5, freq="1D"),
        pd.date_range(start="2020-01-01", periods=5, freq="1D", tz="Europe/Berlin"),
        pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]),
        pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-04", "2020-01-05"]),
    ],
)
@pytest.mark.parametrize("time_ref", [None, pd.Timestamp.min + pd.Timedelta("10s")])
def test_time_classes(inputs, time_ref):
    with write_read_buffer_context({"root": inputs}) as data:
        assert np.all(data["root"] == inputs)

    if isinstance(inputs, pd.Index) and not inputs.is_monotonic_increasing:
        # this is not valid for the time class, hence cancel here
        return

    t1 = Time(inputs, time_ref)
    with write_read_buffer_context({"root": t1}) as data:
        t2 = data["root"]
        assert t1.equals(t2)


def test_time_classes_max_inline():
    # test support for 64bit literals
    dti = pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-04", "2020-01-05"])
    write_buffer(
        {"root": dti},
        write_kwargs={"all_array_storage": "inline"},
    )
