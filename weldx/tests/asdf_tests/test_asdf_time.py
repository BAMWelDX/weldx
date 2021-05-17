"""Test time schema implementation."""
import numpy as np
import pandas as pd
import pytest

from weldx.asdf.util import write_buffer, write_read_buffer


@pytest.mark.parametrize(
    "inputs",
    [
        pd.Timedelta("5m3ns"),
        pd.Timedelta("106751 days 23:47:16.854775"),
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
def test_time_classes(inputs):
    data = write_read_buffer({"root": inputs})
    assert np.all(data["root"] == inputs)


def test_time_classes_max_inline():
    # test support for 64bit literals
    dti = pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-04", "2020-01-05"])
    write_buffer(
        {"root": dti},
        write_kwargs={"all_array_storage": "inline"},
    )
