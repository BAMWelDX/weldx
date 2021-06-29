"""Tests asdf implementations of python base types."""
import uuid
import xarray as xr
import numpy as np
import pytest
from asdf import ValidationError

from weldx.asdf.util import write_read_buffer


# --------------------------------------------------------------------------------------
# uuid
# --------------------------------------------------------------------------------------
def test_uuid():
    """Test uuid serialization and version 4 pattern."""
    write_read_buffer({"id": uuid.uuid4()})

    with pytest.raises(ValidationError):
        write_read_buffer({"id": uuid.uuid1()})


# --------------------------------------------------------------------------------------
# xarray.DataSet
# --------------------------------------------------------------------------------------
def test_dataset_children():
    da = xr.DataArray(np.arange(10), name="arr1", attrs={"name": "sample data"})
    ds = da.to_dataset()
    ds2 = write_read_buffer({"ds": ds})["ds"]
    assert ds.arr1.attrs == ds2.arr1.attrs
    assert np.all(ds.arr1 == ds2.arr1)
