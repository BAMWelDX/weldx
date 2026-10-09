"""Regression tests for orientation interpolation with different time resolutions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from weldx import Q_, LocalCoordinateSystem, WXRotation


@pytest.mark.parametrize("absolute", [False, True])
@pytest.mark.parametrize("source_unit", ["s", "ms", "us", "ns"])
@pytest.mark.parametrize("query_unit", ["s", "ms", "us", "ns"])
def test_interp_time_units(absolute, source_unit, query_unit):
    """Preserve samples and interpolate correctly across mixed time units."""
    source = np.array([0, 2], dtype="timedelta64[s]")
    query = np.array([-1, 0, 1, 2, 3], dtype="timedelta64[s]")
    if absolute:
        origin = np.datetime64("2026-10-08T08:42:22", "ns")
        source = pd.DatetimeIndex(
            (origin + source).astype(f"datetime64[{source_unit}]")
        )
        query = pd.DatetimeIndex((origin + query).astype(f"datetime64[{query_unit}]"))
    else:
        source = pd.TimedeltaIndex(source.astype(f"timedelta64[{source_unit}]"))
        query = pd.TimedeltaIndex(query.astype(f"timedelta64[{query_unit}]"))

    lcs = LocalCoordinateSystem(
        orientation=WXRotation.from_euler("z", [[0], [90]], degrees=True),
        coordinates=Q_([[0, 0, 0], [20, 0, 0]], "mm"),
        time=source,
    )

    unchanged = lcs.interp_time(lcs.time)
    np.testing.assert_allclose(
        unchanged.orientation.data, lcs.orientation.data, atol=1e-12
    )
    np.testing.assert_allclose(
        unchanged.coordinates.data.m, lcs.coordinates.data.m, atol=1e-12
    )
    assert np.all(unchanged.time == lcs.time)

    result = lcs.interp_time(query)
    expected = WXRotation.from_euler("z", [[0], [0], [45], [90], [90]], degrees=True)
    np.testing.assert_allclose(
        result.orientation.data, expected.as_matrix(), atol=1e-12
    )
    np.testing.assert_allclose(
        result.coordinates.data.m,
        [[0, 0, 0], [0, 0, 0], [10, 0, 0], [20, 0, 0], [20, 0, 0]],
        atol=1e-12,
    )
    assert np.all(result.time == query)
    assert result.has_reference_time == absolute


def test_interp_time_absolute_nanosecond_precision():
    """Keep small offsets precise when interpolating recent absolute timestamps."""
    origin = np.datetime64("2026-10-08T08:42:22.631400000", "ns")
    source = pd.DatetimeIndex(origin + np.array([0, 1000], dtype="timedelta64[ns]"))
    query = pd.DatetimeIndex(
        origin + np.array([0, 100, 500, 900, 1000], dtype="timedelta64[ns]")
    )
    lcs = LocalCoordinateSystem(
        orientation=WXRotation.from_euler("z", [[0], [90]], degrees=True),
        time=source,
    )

    result = lcs.interp_time(query)
    expected = WXRotation.from_euler("z", [[0], [9], [45], [81], [90]], degrees=True)
    np.testing.assert_allclose(
        result.orientation.data, expected.as_matrix(), atol=1e-12
    )
    assert np.all(result.time == query)
