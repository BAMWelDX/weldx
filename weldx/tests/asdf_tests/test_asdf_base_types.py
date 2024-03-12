"""Tests asdf implementations of python base types."""

import uuid

import numpy as np
import pytest
import xarray as xr
from asdf.exceptions import ValidationError

from weldx.asdf.util import write_read_buffer, write_read_buffer_context


# --------------------------------------------------------------------------------------
# uuid
# --------------------------------------------------------------------------------------
def test_uuid():
    """Test uuid serialization and version 4 pattern."""
    write_read_buffer({"id": uuid.uuid4()})

    with pytest.raises(ValidationError):
        write_read_buffer({"id": uuid.uuid1()})


# --------------------------------------------------------------------------------------
# xarray
# --------------------------------------------------------------------------------------
def test_dataarray():
    da = xr.DataArray(np.random.randn(2, 3), dims=("x", "y"), coords={"x": [10, 20]})
    da.attrs["long_name"] = "random velocity"
    # add metadata to coordinate
    da.x.attrs["units"] = "x units"
    with write_read_buffer_context({"da": da}) as data:
        da2 = data["da"]
        assert da2.identical(da)


def test_dataset_children():
    da = xr.DataArray(np.arange(10), name="arr1", attrs={"name": "sample data"})
    ds = da.to_dataset()
    with write_read_buffer_context({"ds": ds}) as data:
        ds2 = data["ds"]
        assert ds2.identical(ds)
