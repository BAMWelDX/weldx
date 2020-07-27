"""Test time schema implementation."""
import numpy as np
import pandas as pd
import pytest
from asdf import ValidationError

from weldx.asdf.utils import _write_buffer, _write_read_buffer


def test_time_classes():
    """Test basic implementation and serialization of pandas time datatypes."""
    # Timedelta -------------------------------------------------------
    td = pd.Timedelta("5m3ns")

    # Timedelta -------------------------------------------------------
    td_max = pd.Timedelta("106751 days 23:47:16.854775")

    # TimedeltaIndex -------------------------------------------------------
    tdi = pd.timedelta_range(start="-5s", end="25s", freq="3s")
    tdi_nofreq = pd.TimedeltaIndex([0, 1e9, 5e9, 3e9])

    # Timestamp -------------------------------------------------------
    ts = pd.Timestamp("2020-04-15T16:47:00.000000001")
    ts_tz = pd.Timestamp("2020-04-15T16:47:00.000000001", tz="Europe/Berlin")

    # DatetimeIndex -------------------------------------------------------
    dti = pd.date_range(start="2020-01-01", periods=5, freq="1D")
    dti_tz = pd.date_range(start="2020-01-01", periods=5, freq="1D", tz="Europe/Berlin")
    dti_infer = pd.DatetimeIndex(
        ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]
    )
    dti_nofreq = pd.DatetimeIndex(
        ["2020-01-01", "2020-01-02", "2020-01-04", "2020-01-05"]
    )

    tree = dict(
        td=td,
        td_max=td_max,
        tdi=tdi,
        tdi_nofreq=tdi_nofreq,
        ts=ts,
        ts_tz=ts_tz,
        dti=dti,
        dti_tz=dti_tz,
        dti_infer=dti_infer,
        dti_nofreq=dti_nofreq,
    )

    data = _write_read_buffer(tree)
    assert isinstance(data, dict)
    for k, v in tree.items():
        assert np.all(data[k] == v)

    with pytest.raises(ValidationError):
        # cannot store large ints >52 bits inline in asdf
        _write_buffer(tree, write_kwargs={"all_array_storage": "inline"})
