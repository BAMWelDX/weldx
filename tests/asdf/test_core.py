"""Tests asdf implementations of core module."""

import numpy as np
import asdf
import xarray as xr

import weldx.transformations as tf
from weldx.asdf.extension import WeldxExtension, WeldxAsdfExtension


# xarray.DataArray ---------------------------------------------------------------------


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


# weldx.transformations.LocalCoordinateSystem ------------------------------------------


def get_local_coordinate_system(time_dep_orientation, time_dep_coordinates):
    coords = [2, 5, 1]
    orientation = tf.rotation_matrix_z(np.pi / 3)

    if not time_dep_orientation and not time_dep_coordinates:
        return tf.LocalCoordinateSystem(orientation=orientation, coordinates=coords)
    raise Exception("not implemented")


def are_local_coordinate_systems_equal(
    lcs_0: tf.LocalCoordinateSystem, lcs_1: tf.LocalCoordinateSystem
):
    return lcs_0.orientation.equals(lcs_1.orientation) and lcs_0.coordinates.equals(
        lcs_1.coordinates
    )


def test_local_coordinate_system_save():
    """Test if a LocalCoordinateSystem can be writen to an asdf file."""
    lcs_static = get_local_coordinate_system(False, False)
    tree = {"lcs_static": lcs_static}
    with asdf.AsdfFile(tree, extensions=[WeldxExtension(), WeldxAsdfExtension()]) as f:
        f.write_to("local_coordinate_system.asdf")


# TODO: remove
test_local_coordinate_system_save()


def test_local_coordinate_system_load():
    """Test if an xarray.DataArray can be restored from an asdf file."""
    f = asdf.open(
        "local_coordinate_system.asdf",
        extensions=[WeldxExtension(), WeldxAsdfExtension()],
    )
    lcs_static_file = f.tree["lcs_static"]
    lcs_static_exp = get_local_coordinate_system(False, False)

    assert are_local_coordinate_systems_equal(lcs_static_file, lcs_static_exp)


# TODO: remove
test_local_coordinate_system_load()
