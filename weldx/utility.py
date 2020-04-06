"""Contains package internal utility functions."""

import math
import numpy as np
import pandas as pd
import xarray as xr
import weldx.transformations as tf
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as Rot


def is_column_in_matrix(column, matrix):
    """
    Check if a column (1d array) can be found inside of a matrix.

    :param column: Column that should be checked
    :param matrix: Matrix
    :return: True or False
    """
    return is_row_in_matrix(column, np.transpose(matrix))


def is_row_in_matrix(row, matrix):
    """
    Check if a row (1d array) can be found inside of a matrix.

    source: https://codereview.stackexchange.com/questions/193835

    :param row: Row that should be checked
    :param matrix: Matrix
    :return: True or False
    """
    if not matrix.shape[1] == np.array(row).size:
        return False
    # noinspection PyUnresolvedReferences
    return (matrix == row).all(axis=1).any()


def to_float_array(container):
    """
    Cast the passed container to a numpy array of floats.

    :param container: Container which can be cast to a numpy array
    :return:
    """
    return np.array(container, dtype=float)


def to_list(var):
    """
    Store the passed variable into a list and return it.

    If the variable is already a list, it is returned without modification.
    If 'None' is passed, the function returns an empty list.

    :param var: Arbitrary variable
    :return: List
    """
    if isinstance(var, list):
        return var
    if var is None:
        return []
    return [var]


def matrix_is_close(mat_a, mat_b, abs_tol=1e-9):
    """
    Check if a matrix is close or equal to another matrix.

    :param mat_a: First matrix
    :param mat_b: Second matrix
    :param abs_tol: Absolute tolerance
    :return: True or False
    """
    mat_a = to_float_array(mat_a)
    mat_b = to_float_array(mat_b)

    if not mat_a.shape == mat_b.shape:
        return False
    for i in range(mat_a.shape[0]):
        for j in range(mat_a.shape[1]):
            if not math.isclose(mat_a[i, j], mat_b[i, j], abs_tol=abs_tol):
                return False
    return True


def vector_is_close(vec_a, vec_b, abs_tol=1e-9):
    """
    Check if a vector is close or equal to another vector.

    :param vec_a: First vector
    :param vec_b: Second vector
    :param abs_tol: Absolute tolerance
    :return: True or False
    """
    vec_a = to_float_array(vec_a)
    vec_b = to_float_array(vec_b)

    if not vec_a.size == vec_b.size:
        return False
    for i in range(vec_a.size):
        if not math.isclose(vec_a[i], vec_b[i], abs_tol=abs_tol):
            return False

    return True


def mat_vec_mul(a, b):
    """
    Matrix x Vector multiplication using matmul with newaxis for correct broadcasting.

    :param a: Input Matrix [m, n]
    :param b: Input Vector to be multiplied [n, ]
    :return: Resulting vector [n, ]
    """
    return np.matmul(a, b[..., np.newaxis]).squeeze()


def swap_list_items(arr, i1, i2):
    """
    Swap position of two items in a list.

    :param arr: list in which to swap elements
    :param i1: element 1 in list
    :param i2: element 2 in list
    :return: copy of list with swapped elements
    """
    i = list(arr).copy()
    a, b = i.index(i1), i.index(i2)
    i[b], i[a] = i[a], i[b]
    return i


def as_xarray_dims(input):
    """
    Generate empty xarray object with coordinates for interpolation.

    :param input: xarray object, pandas TimeIndex object or dict
    :return: empty xarray object with coordinates generated from input
    """
    if isinstance(input, (xr.DataArray, xr.Dataset)):
        return input
    elif isinstance(input, (pd.DatetimeIndex, pd.TimedeltaIndex)):
        return xr.DataArray(data=None, dims=["time"], coords={"time": input})
    elif isinstance(input, dict):
        return xr.DataArray(data=None, dims=list(input.keys()), coords=input)
    return None


def get_time_union(list_of_objects):
    """
    Generate a merged union of pd.DatatimeIndex from list of inputs.

    The functions tries to merge common inputs that are "time-like" or might have time
    coordinates such as xarrray objects, tf.LocalCoordinateSystem and other time objects
    :param list_of_objects: list of input objects to merge
    :return: pd.DatetimeIndex with merge times
    """

    def _get_time(input):
        if isinstance(input, (pd.DatetimeIndex, pd.TimedeltaIndex)):
            return input
        elif isinstance(input, (xr.DataArray, xr.Dataset)):
            return pd.DatetimeIndex(input.time.data)
        elif isinstance(input, tf.LocalCoordinateSystem):
            return input.time
        else:
            return pd.DatetimeIndex(input)

    for idx, val in enumerate(list_of_objects):
        if idx == 0:
            times = _get_time(val)
        else:
            times = times.union(_get_time(val))
    return times


def xr_transpose_matrix_data(da, dim1, dim2):
    """
    Transpose data along two dimensions in an xarray.DataArray.

    :param da: xarray.DataArray to transpose
    :param dim1: name of the first dimension
    :param dim2: name of the second dimension
    :return: xarray.DataArray with transposed data at specified dimensions
    """
    i = swap_list_items(da.dims, dim1, dim2)

    return da.copy(data=da.transpose(*i).data)


def xr_matmul(
    a,
    b,
    dims_a,
    dims_b=None,
    dims_out=None,
    trans_a=False,
    trans_b=False,
    **apply_kwargs,
):
    """
    Calculate broadcasted np.matmul(a,b) for xarray objects.

    Should work for any size and shape of quadratic matrices contained in a DataArray.
    Ordering, broadcasting of dimensions should be taken care of by xarray internally.
    This can be used for both matrix * matrix and matrix * vector operations.
    :param a: xarray object containing the first matrix
    :param b: xarray object containing the second matrix
    :param dims_a: name and order of dimensions in the first object
    :param dims_b: name and order of dimensions in the second object
    (if None, use dims_a)
    :param dims_out: name and order of dimensions in the resulting object
    (if None, use dims_a unless dims_b has less items than dims_b)
    :param trans_a: flag if matrix in a should be transposed
    :param trans_b: flag if matrix in b should be transposed
    :param: **apply_kwargs: parameters to pass on to xr.apply_ufunc
    :return:
    """
    if dims_b is None:
        dims_b = dims_a
    if dims_out is None:
        if len(dims_a) <= len(dims_b):
            dims_out = dims_a
        else:
            dims_out = dims_b

    mul_func = np.matmul
    if len(dims_a) > len(dims_b):
        mul_func = mat_vec_mul

    if trans_a:
        dims_a = reversed(dims_a)
    if trans_b:
        dims_b = reversed(dims_b)

    return xr.apply_ufunc(
        mul_func,
        a,
        b,
        input_core_dims=[dims_a, dims_b],
        output_core_dims=[dims_out],
        **apply_kwargs,
    )


def xr_is_orthogonal_matrix(da, dims):
    """
    Check if  matrix along specific dimensions in a DataArray is orthogonal.

    TODO: make more general

    :param da: xarray.DataArray to test
    :param dims: list of dimensions along which to test
    :return: True if all matrices are orthogonal.
    """
    eye = np.eye(len(da.coords[dims[0]]), len(da.coords[dims[1]]))
    return np.allclose(xr_matmul(da, da, dims, trans_b=True), eye)


def xr_fill_all(da, order="bf"):
    """
    Fill NaN values along all dimensions in xarray.DataArray.

    :param da: xarray object to fill
    :param order: order in which to apply bfill/ffill operation
    :return: xarray object with NaN values filled in all dimensions
    """
    if order == "bf":
        for dim in da.dims:
            da = da.bfill(dim).ffill(dim)
    elif order == "fb":
        for dim in da.dims:
            da = da.ffill(dim).bfill(dim)
    else:
        raise ValueError(f"Order {order} is not supported (use 'bf' or 'fb)")
    return da


def xr_interp_like(
    da1: xr.DataArray,
    da2: xr.DataArray,
    broadcast_missing: bool = False,
    fillna: bool = True,
    method: str = "linear",
    assume_sorted: bool = False,
):
    """
    Interpolate DataArray along dimensions of another DataArray.

    Provides some utility options for handling out of range values and broadcasting.
    :param da1: xarray object with data to interpolate
    :param da2: xarray object along which dimensions to interpolate
    :param broadcast_missing: broadcast da1 along all additional dimensions of da2
    :param fillna: fill out of range NaN values (default = True)
    :param method: interpolation method to pass on to xarray.interp_like
    :param assume_sorted: assume_sorted flag to pass on to xarray.interp_like
    :return: interpolated DataArray
    """
    interp_coords = da2.coords  # remember original interpolation coordinates

    # make sure edge coordinate values of da1 are in new coordinate axis of da2
    if assume_sorted:
        # if all coordinates are sorted,we can use integer indexing for speedups
        edge_dict = {
            d: ([0, -1] if len(val) > 1 else [0])
            for d, val in da1.coords.items()
            if d in interp_coords
        }
        if len(edge_dict) > 0:
            da2 = da2.combine_first(da1.isel(edge_dict))
    else:
        # select, combine with min/max values if coordinates not guaranteed to be sorted
        edge_dict = {
            d: ([val.min().data, val.max().data] if len(val) > 1 else [val.min()])
            for d, val in da1.coords.items()
            if d in interp_coords
        }
        if len(edge_dict) > 0:
            da2 = da2.combine_first(da1.sel(edge_dict))

    # handle singular dimensions in da1
    # TODO: should we handle coordinates or dimensions?
    singular_dims = [d for d in da1.dims if len(da1[d]) == 1]
    for dim in singular_dims:
        if dim in da2.dims:
            exclude_dims = [d for d in da2.dims if not d == dim]
            da1 = xr_fill_all(da1.broadcast_like(da2, exclude=exclude_dims))

    # default interp will not add dimensions and fill out of range indexes with NaN
    da = da1.interp_like(da2, method=method, assume_sorted=assume_sorted)

    # fill out of range nan values for all dimensions
    if fillna:
        da = xr_fill_all(da)

    if broadcast_missing:
        da = da.broadcast_like(da2)
    else:  # careful not to select coordinates that are only in da2
        interp_coords = {d: v for d, v in interp_coords.items() if d in da1.coords}

    return da.sel(interp_coords)


def xr_3d_vector(data, times=None):
    if times is not None:
        dsx = xr.DataArray(
            data=data, dims=["time", "c"], coords={"time": times, "c": ["x", "y", "z"]}
        )
    else:
        dsx = xr.DataArray(data=data, dims=["c"], coords={"c": ["x", "y", "z"]})
    return dsx.astype(float)


def xr_3d_matrix(data, times=None):
    if times is not None:
        dsx = xr.DataArray(
            data=data,
            dims=["time", "c", "v"],
            coords={"time": times, "c": ["x", "y", "z"], "v": [0, 1, 2]},
        )
    else:
        dsx = xr.DataArray(
            data=data, dims=["c", "v"], coords={"c": ["x", "y", "z"], "v": [0, 1, 2]}
        )
    return dsx.astype(float)


def xr_interp_orientation_in_time(dsx, times):
    if "time" not in dsx.coords:
        return dsx.expand_dims({"time": times})

    # extract intersecting times and add time range boundaries of the data set
    times_ds = dsx.time.data
    if len(times_ds) > 1:
        times_ds_limits = pd.DatetimeIndex([times_ds.min(), times_ds.max()])
        times_union = times.union(times_ds_limits)
        times_intersect = times_union[
            (times_union >= times_ds_limits[0]) & (times_union <= times_ds_limits[1])
        ]

        # interpolate rotations in the intersecting time range
        rotations_key = Rot.from_matrix(dsx.data)
        times_key = dsx.time.astype(np.int64)
        rotations_interp = Slerp(times_key, rotations_key)(
            times_intersect.astype(np.int64)
        )
        dsx_out = xr_3d_matrix(rotations_interp.as_matrix(), times_intersect)
    else:
        dsx_out = dsx

    # broadcast boundary values for all times outside the intersection range
    dsx_out = dsx_out.broadcast_like(as_xarray_dims(times)).bfill("time").ffill("time")

    # Remove inserted boundaries if necessary and return
    return dsx_out.sel(time=times)


def xr_interp_coodinates_in_time(dsx, times):
    if "time" not in dsx.coords:
        return dsx.expand_dims({"time": times})

    times_ds = dsx.time.data
    if len(times_ds) > 1:
        times_ds_limits = pd.DatetimeIndex([times_ds.min(), times_ds.max()])
        times_union = times.union(times_ds_limits)
        dsx_out = dsx.interp({"time": times_union})
    else:
        dsx_out = dsx.broadcast_like(as_xarray_dims(times))
    dsx_out = dsx_out.bfill("time").ffill("time")

    return dsx_out.sel(time=times)


# weldx xarray Accessors --------------------------------------------------------


@xr.register_dataarray_accessor("weldx")
class WeldxAccessor:
    """
    Custom accessor for extending DataArray functionality.

    See http://xarray.pydata.org/en/stable/internals.html#extending-xarray for details.
    """

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def interp_like(self, da, *args, **kwargs):
        """
        Interpolate DataArray along dimensions of another DataArray.

        Provides some utility options for handling out of range values and broadcasting.
        See xr_interp_like for docstring and details.
        :return: interpolated DataArray
        """
        return xr_interp_like(self._obj, da, *args, **kwargs)
