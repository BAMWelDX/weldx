"""Contains package internal utility functions."""

import math
from collections.abc import Iterable
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import pint
import xarray as xr
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp

import weldx.transformations as tf
from weldx.constants import WELDX_QUANTITY as Q_


def is_column_in_matrix(column, matrix) -> bool:
    """Check if a column (1d array) can be found inside of a matrix.

    Parameters
    ----------
    column :
        Column that should be checked
    matrix :
        Matrix

    Returns
    -------
    bool
        True or False

    """
    return is_row_in_matrix(column, np.transpose(matrix))


def is_row_in_matrix(row, matrix) -> bool:
    """Check if a row (1d array) can be found inside of a matrix.

    source: https://codereview.stackexchange.com/questions/193835

    Parameters
    ----------
    row :
        Row that should be checked
    matrix :
        Matrix

    Returns
    -------
    bool
        True or False

    """
    if not matrix.shape[1] == np.array(row).size:
        return False
    # noinspection PyUnresolvedReferences
    return (matrix == row).all(axis=1).any()


def to_float_array(container) -> np.ndarray:
    """Cast the passed container to a numpy array of floats.

    Parameters
    ----------
    container :
        Container which can be cast to a numpy array

    Returns
    -------
    np.ndarray

    """
    return np.array(container, dtype=float)


def to_list(var) -> list:
    """Store the passed variable into a list and return it.

    If the variable is already a list, it is returned without modification.
    If 'None' is passed, the function returns an empty list.

    Parameters
    ----------
    var :
        Arbitrary variable

    Returns
    -------
    list

    """
    if isinstance(var, list):
        return var
    if var is None:
        return []
    return [var]


def to_pandas_time_index(time) -> Union[pd.TimedeltaIndex, pd.DatetimeIndex]:
    """Convert a time variable to the corresponding pandas time index type.

    Parameters
    ----------
    time :
        Variable that should be converted.

    Returns
    -------
        Variable as pandas time index

    """
    if isinstance(time, pint.Quantity):
        base = "s"  # using low base unit could cause rounding errors
        try:
            return pd.TimedeltaIndex(data=time.to(base).magnitude, unit=base)
        except TypeError:
            return pd.TimedeltaIndex(data=[time.to(base).magnitude], unit=base)

    if not isinstance(time, np.ndarray):
        if not isinstance(time, list):
            time = [time]
        time = np.array(time)

    if np.issubdtype(time.dtype, np.datetime64):
        return pd.DatetimeIndex(time)
    return pd.TimedeltaIndex(time)


def pandas_time_delta_to_quantity(
    time: pd.TimedeltaIndex, unit: str = "s"
) -> pint.Quantity:
    """Convert a 'pandas.TimedeltaIndex' into a corresponding 'pint.Quantity'.

    Parameters
    ----------
    time :
        Instance of 'pandas.TimedeltaIndex'
    unit :
        String that specifies the desired time unit.

    Returns
    -------
    pint.Quantity :
        Converted time quantity

    """
    # from pandas Timedelta documentation: "The .value attribute is always in ns."
    # https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.Timedelta.html
    nanoseconds = time.values.astype(np.int64)
    if len(nanoseconds) == 1:
        nanoseconds = nanoseconds[0]
    return Q_(nanoseconds, "ns").to(unit)


def matrix_is_close(mat_a, mat_b, abs_tol=1e-9) -> bool:
    """Check if a matrix is close or equal to another matrix.

    Parameters
    ----------
    mat_a :
        First matrix
    mat_b :
        Second matrix
    abs_tol :
        Absolute tolerance (Default value = 1e-9)

    Returns
    -------
    bool
        True or False

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


def vector_is_close(vec_a, vec_b, abs_tol=1e-9) -> bool:
    """Check if a vector is close or equal to another vector.

    Parameters
    ----------
    vec_a :
        First vector
    vec_b :
        Second vector
    abs_tol :
        Absolute tolerance (Default value = 1e-9)

    Returns
    -------
    bool
        True or False

    """
    vec_a = to_float_array(vec_a)
    vec_b = to_float_array(vec_b)

    if not vec_a.size == vec_b.size:
        return False
    for i in range(vec_a.size):
        if not math.isclose(vec_a[i], vec_b[i], abs_tol=abs_tol):
            return False

    return True


def mat_vec_mul(a, b) -> np.ndarray:
    """Matrix, Vector multiplication using matmul with newaxis for correct broadcasting.

    Parameters
    ----------
    a :
        Input Matrix [m, n]
    b :
        Input Vector to be multiplied [n, ]

    Returns
    -------
    np.ndarray
        Resulting vector [n, ]

    """
    return np.matmul(a, b[..., np.newaxis]).squeeze()


def swap_list_items(arr, i1, i2) -> list:
    """Swap position of two items in a list.

    Parameters
    ----------
    arr :
        list in which to swap elements
    i1 :
        element 1 in list
    i2 :
        element 2 in list

    Returns
    -------
    list
        copy of list with swapped elements

    """
    i = list(arr).copy()
    a, b = i.index(i1), i.index(i2)
    i[b], i[a] = i[a], i[b]
    return i


def get_time_union(list_of_objects):
    """Generate a merged union of pd.DatetimeIndex from list of inputs.

    The functions tries to merge common inputs that are "time-like" or might have time
    coordinates such as xarray objects, tf.LocalCoordinateSystem and other time objects

    Parameters
    ----------
    list_of_objects :
        list of input objects to merge

    Returns
    -------
    pd.DatetimeIndex
        pandas DatetimeIndex with merged times

    """
    # TODO: make non-nested function
    def _get_time(input_object):
        if isinstance(input_object, (pd.DatetimeIndex, pd.TimedeltaIndex)):
            return input_object
        if isinstance(input_object, (xr.DataArray, xr.Dataset)):
            return to_pandas_time_index(input_object.time.data)
        if isinstance(input_object, tf.LocalCoordinateSystem):
            return input_object.time

        return to_pandas_time_index(input_object)

    times = None
    for idx, val in enumerate(list_of_objects):
        if idx == 0:
            times = _get_time(val)
        else:
            times = times.union(_get_time(val))
    return times


def xr_transpose_matrix_data(da, dim1, dim2) -> xr.DataArray:
    """Transpose data along two dimensions in an xarray DataArray.

    Parameters
    ----------
    da :
        xarray DataArray to transpose
    dim1 :
        name of the first dimension
    dim2 :
        name of the second dimension

    Returns
    -------
    xr.DataArray
        xarray DataArray with transposed data at specified dimensions

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
) -> xr.DataArray:
    """Calculate broadcasted np.matmul(a,b) for xarray objects.

    Should work for any size and shape of quadratic matrices contained in a DataArray.
    Ordering, broadcasting of dimensions should be taken care of by xarray internally.
    This can be used for both matrix * matrix and matrix * vector operations.

    Parameters
    ----------
    a :
        xarray object containing the first matrix
    b :
        xarray object containing the second matrix
    dims_a :
        name and order of dimensions in the first object
    dims_b :
        name and order of dimensions in the second object
        (if None, use dims_a) (Default value = None)
    dims_out :
        name and order of dimensions in the resulting object
        (if None, use dims_a unless dims_b has less items than dims_b)
        (Default value = None)
    trans_a :
        flag if matrix in a should be transposed (Default value = False)
    trans_b :
        flag if matrix in b should be transposed (Default value = False)
    **apply_kwargs :
        additional kwargs passed on to ur.apply_ufunc


    Returns
    -------
    xr.DataArray

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


def xr_is_orthogonal_matrix(da: xr.DataArray, dims: List[str]) -> bool:
    """Check if  matrix along specific dimensions in a DataArray is orthogonal.

    TODO: make more general

    Parameters
    ----------
    da :
        xarray DataArray to test
    dims :
        list of dimensions along which to test

    Returns
    -------
    bool
        True if all matrices are orthogonal.

    """
    eye = np.eye(len(da.coords[dims[0]]), len(da.coords[dims[1]]))
    return np.allclose(xr_matmul(da, da, dims, trans_b=True), eye)


def xr_fill_all(da, order="bf") -> xr.DataArray:
    """Fill NaN values along all dimensions in xarray DataArray.

    Parameters
    ----------
    da :
        xarray object to fill
    order :
        order in which to apply bfill/ffill operation (Default value = "bf")

    Returns
    -------
    xr.DataArray
        xarray object with NaN values filled in all dimensions

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
    da2: Union[xr.DataArray, Dict[str, Any]],
    interp_coords: List[str] = None,
    broadcast_missing: bool = False,
    fillna: bool = True,
    method: str = "linear",
    assume_sorted: bool = False,
) -> xr.DataArray:
    """Interpolate DataArray along dimensions of another DataArray.

    Provides some utility options for handling out of range values and broadcasting.

    Parameters
    ----------
    da1 :
        xarray object with data to interpolate
    da2 :
        xarray or dict-like object along which dimensions to interpolate
    interp_coords :
        if not None, only interpolate along these coordinates of da2
        (Default value = None)
    broadcast_missing :
        broadcast da1 along all additional dimensions of da2
        (Default value = False)
    fillna :
        fill out of range NaN values
        (Default value = True)
    method :
        interpolation method to pass on to xarray.interp_like
        (Default value = "linear")
    assume_sorted :
        assume_sorted flag to pass on to xarray.interp_like
        (Default value = False)

    Returns
    -------
    xr.DataArray
        interpolated DataArray

    """
    if isinstance(da2, (xr.DataArray, xr.Dataset)):
        sel_coords = da2.coords  # remember original interpolation coordinates
    else:  # assume da2 to be dict-like
        sel_coords = {
            k: (v if isinstance(v, Iterable) else [v]) for k, v in da2.items()
        }

    # store and strip pint units at this point, since the unit is lost during
    # interpolation and because of some other conflicts. Unit is restored before
    # returning the result.
    units = None
    if isinstance(da1.data, pint.Quantity):
        units = da1.data.units
        da1 = xr.DataArray(data=da1.data.magnitude, dims=da1.dims, coords=da1.coords)

    if interp_coords is not None:
        sel_coords = {k: v for k, v in sel_coords.items() if k in interp_coords}

    # create a new (empty) temporary dataset to use for interpolation
    # we need this if da2 is passed as an existing coordinate variable like origin.time
    da_temp = xr.DataArray(dims=sel_coords.keys(), coords=sel_coords)

    # make sure edge coordinate values of da1 are in new coordinate axis of da_temp
    if assume_sorted:
        # if all coordinates are sorted,we can use integer indexing for speedups
        edge_dict = {
            d: ([0, -1] if len(val) > 1 else [0])
            for d, val in da1.coords.items()
            if d in sel_coords
        }
        if len(edge_dict) > 0:
            da_temp = da_temp.combine_first(da1.isel(edge_dict))
    else:
        # select, combine with min/max values if coordinates not guaranteed to be sorted
        # maybe switch to idxmin()/idxmax() once it available
        # TODO: handle non-numeric dtypes ! currently cannot work on unsorted str types
        edge_dict = {
            d: ([val.min().data, val.max().data] if len(val) > 1 else [val.min().data])
            for d, val in da1.coords.items()
            if d in sel_coords
        }
        if len(edge_dict) > 0:
            da_temp = da_temp.combine_first(da1.sel(edge_dict))

    # handle singular dimensions in da1
    # TODO: should we handle coordinates or dimensions?
    singular_dims = [d for d in da1.coords if len(da1[d]) == 1]
    for dim in singular_dims:
        if dim in da_temp.coords:
            if len(da_temp.coords[dim]) > 1:
                if not fillna:
                    raise ValueError(
                        "Cannot use fillna=False with single point interpolation"
                    )
                exclude_dims = [d for d in da_temp.coords if not d == dim]
                # TODO: this always fills the dimension (inconsistent with fillna=False)
                da1 = xr_fill_all(da1.broadcast_like(da_temp, exclude=exclude_dims))
            else:
                del da_temp.coords[dim]

    # default interp_like will not add dimensions and fill out of range indexes with NaN
    da = da1.interp_like(da_temp, method=method, assume_sorted=assume_sorted)

    # fill out of range nan values for all dimensions
    if fillna:
        da = xr_fill_all(da)

    if broadcast_missing:
        da = da.broadcast_like(da_temp)
    else:  # careful not to select coordinates that are only in da_temp
        sel_coords = {d: v for d, v in sel_coords.items() if d in da1.coords}

    result = da.sel(sel_coords)
    if units is not None:
        result = xr.DataArray(
            data=pint.Quantity(result.data, units),
            dims=result.dims,
            coords=result.coords,
        )

    return result


def xr_3d_vector(data, times=None) -> xr.DataArray:
    """Create an xarray 3d vector with correctly named dimensions and coordinates.

    Parameters
    ----------
    data :
        Data
    times :
        Optional time data (Default value = None)

    Returns
    -------
    xr.DataArray

    """
    if times is not None:
        dsx = xr.DataArray(
            data=data, dims=["time", "c"], coords={"time": times, "c": ["x", "y", "z"]},
        )
    else:
        dsx = xr.DataArray(data=data, dims=["c"], coords={"c": ["x", "y", "z"]})
    return dsx.astype(float)


def xr_3d_matrix(data, times=None) -> xr.DataArray:
    """Create an xarray 3d matrix with correctly named dimensions and coordinates.

    Parameters
    ----------
    data :
        Data
    times :
        Optional time data (Default value = None)

    Returns
    -------
    xr.DataArray

    """
    if times is not None:
        dsx = xr.DataArray(
            data=data,
            dims=["time", "c", "v"],
            coords={"time": times, "c": ["x", "y", "z"], "v": [0, 1, 2]},
        )
    else:
        dsx = xr.DataArray(
            data=data, dims=["c", "v"], coords={"c": ["x", "y", "z"], "v": [0, 1, 2]},
        )
    return dsx.astype(float)


def xr_interp_orientation_in_time(
    dsx: xr.DataArray, times: pd.DatetimeIndex
) -> xr.DataArray:
    """Interpolate an xarray DataArray that represents orientation data in time.

    Parameters
    ----------
    dsx :
        xarray DataArray containing the orientation as matrix
    times :
        Time data

    Returns
    -------
    xr.DataArray
        Interpolated data

    """
    if "time" not in dsx.coords:
        return dsx

    # extract intersecting times and add time range boundaries of the data set
    times_ds = dsx.time.data
    if len(times_ds) > 1:
        if isinstance(times_ds, pd.DatetimeIndex):
            times_ds_limits = pd.DatetimeIndex([times_ds.min(), times_ds.max()])
        else:
            times_ds_limits = pd.TimedeltaIndex([times_ds.min(), times_ds.max()])
        times_union = times.union(times_ds_limits)
        times_intersect = times_union[
            (times_union >= times_ds_limits[0]) & (times_union <= times_ds_limits[1])
        ]

        # interpolate rotations in the intersecting time range
        rotations_key = Rot.from_matrix(dsx.transpose(..., "c", "v").data)
        times_key = dsx.time.astype(np.int64)
        rotations_interp = Slerp(times_key, rotations_key)(
            times_intersect.astype(np.int64)
        )
        dsx_out = xr_3d_matrix(rotations_interp.as_matrix(), times_intersect)
    else:
        # TODO: this case is not really well defined, maybe avoid?
        dsx_out = dsx

    # use interp_like to select original time values and correctly fill time dimension
    dsx_out = xr_interp_like(dsx_out, {"time": times}, fillna=True)

    return dsx_out.transpose(..., "c", "v")


def xr_interp_coordinates_in_time(
    dsx: xr.DataArray, times: pd.DatetimeIndex
) -> xr.DataArray:
    """Interpolate an xarray DataArray that represents 3d coordinates in time.

    Parameters
    ----------
    dsx :
        xarray DataArray
    times :
        Time data

    Returns
    -------
    xr.DataArray
        Interpolated data

    """
    return xr_interp_like(
        dsx, {"time": times}, assume_sorted=True, broadcast_missing=False, fillna=True
    )


# weldx xarray Accessors --------------------------------------------------------


@xr.register_dataarray_accessor("weldx")
class WeldxAccessor:  # pragma: no cover
    """Custom accessor for extending DataArray functionality.

    See http://xarray.pydata.org/en/stable/internals.html#extending-xarray for details.
    """

    def __init__(self, xarray_obj):
        """Construct a WeldX xarray object."""
        self._obj = xarray_obj

    def interp_like(self, da, *args, **kwargs):
        """Interpolate DataArray along dimensions of another DataArray.

        Provides some utility options for handling out of range values and broadcasting.
        See xr_interp_like for docstring and details.

        Returns
        -------
        xr.DataArray
            interpolated DataArray

        """
        return xr_interp_like(self._obj, da, *args, **kwargs)
