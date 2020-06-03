"""Tests asdf implementations of core module."""

import numpy as np
import asdf
import xarray as xr


from weldx.asdf.extension import WeldxExtension, WeldxAsdfExtension


def get_xarray_example_data_array():
    """Get an xarray.DataArray for test purposes."""
    data = np.array([[0, 1], [2, 3]])

    dax = xr.DataArray(data=data, dims=["d1", "d2"], coords={"d1": np.array([-1, 1])})
    return dax


def test_xarray_data_array_save():
    """Test if an xarray.DataArray can be writen to an asdf file."""
    dax = get_xarray_example_data_array()
    tree = {"dax": dax}
    with asdf.AsdfFile(tree, extensions=[WeldxExtension(), WeldxAsdfExtension()]) as f:
        f.write_to("xarray.asdf")


def test_xarray_data_array_load():
    """Test if an xarray.DataArray can be restored from an asdf file."""
    f = asdf.open("xarray.asdf", extensions=[WeldxExtension(), WeldxAsdfExtension()])
    dax_file = f.tree["dax"]
    dax_exp = get_xarray_example_data_array()
    assert dax_exp.equals(dax_file)
