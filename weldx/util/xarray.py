"""Contains xarray specific utility functions."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import pint
import xarray as xr
from pandas.api.types import is_datetime64_dtype, is_timedelta64_dtype
from pint import DimensionalityError
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp

from weldx.constants import Q_, U_
from weldx.constants import WELDX_UNIT_REGISTRY as ureg
from weldx.time import Time, types_time_like, types_timestamp_like

__all__ = [
    "WeldxAccessor",
    "mat_vec_mul",
    "xr_3d_matrix",
    "xr_3d_vector",
    "xr_check_coords",
    "xr_fill_all",
    "xr_interp_coordinates_in_time",
    "xr_interp_like",
    "xr_interp_orientation_in_time",
    "xr_is_orthogonal_matrix",
    "xr_matmul",
    "xr_transpose_matrix_data",
]


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
    numpy.ndarray
        Resulting vector [n, ]

    """
    return np.matmul(a, b[..., np.newaxis]).squeeze(axis=-1)


def _swap_list_items(arr, i1, i2) -> list:
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
    xarray.DataArray
        xarray DataArray with transposed data at specified dimensions

    """
    i = _swap_list_items(da.dims, dim1, dim2)

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
    xarray.DataArray

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
        mul_func = mat_vec_mul  # type: ignore[assignment] # irrelevant for us

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
    if not set(dims).issubset(set(da.dims)):
        raise ValueError(f"Could not find {dims=} in DataArray.")
    eye = np.eye(len(da.coords[dims[0]]), len(da.coords[dims[1]]))
    return np.allclose(xr_matmul(da, da, dims, trans_b=True), eye)


def xr_fill_all(da: xr.DataArray, order="bf") -> xr.DataArray:
    """Fill NaN values along all dimensions in xarray DataArray.

    Parameters
    ----------
    da : xarray.DataArray
        xarray object to fill
    order :
        order in which to apply bfill/ffill operation (Default value = "bf")

    Returns
    -------
    xarray.DataArray
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
    xarray.DataArray
        interpolated DataArray

    """
    da1 = da1.weldx.time_ref_unset()  # catch time formats
    if isinstance(da2, (xr.DataArray, xr.Dataset)):
        da2 = da2.weldx.time_ref_unset()  # catch time formats
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
    if method == "step":
        fill_method = "ffill" if fillna else None
        da = da1.reindex_like(da_temp, method=fill_method)
    else:
        da = da1.interp_like(da_temp, method=method, assume_sorted=assume_sorted)

    # copy original variable and coord attributes
    da.attrs = da1.attrs
    for key in da1.coords:
        da[key].attrs = da1[key].attrs

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
            data=result.data * units,
            dims=result.dims,
            coords=result.coords,
        )

    return result


def _check_dtype(var_dtype, ref_dtype: str) -> bool:
    """Check if dtype matches a reference dtype (or is subdtype).

    Parameters
    ----------
    var_dtype : numpy dtype
        A numpy-dtype to test against.
    ref_dtype : dict
        Python type or string description

    Returns
    -------
    bool
        True if dtypes matches.

    """
    if var_dtype != np.dtype(ref_dtype):
        if (
            isinstance(ref_dtype, str)
            and ("timedelta64" in ref_dtype or "datetime64" in ref_dtype)
            and np.issubdtype(var_dtype, np.dtype(ref_dtype))
        ):
            return True

        if not (
            np.issubdtype(var_dtype, np.dtype(ref_dtype)) and np.dtype(ref_dtype) == str
        ):
            return False

    return True


def xr_check_coords(dax: xr.DataArray, ref: dict) -> bool:
    """Validate the coordinates of the DataArray against a reference dictionary.

    The reference dictionary should have the dimensions as keys and those contain
    dictionaries with the following keywords (all optional):

    ``values``
        Specify exact coordinate values to match.

    ``dtype`` : str or type
        Ensure coordinate dtype matches at least one of the given dtypes.

    ``optional`` : boolean
        default ``False`` - if ``True``, the dimension has to be in the DataArray dax

    ``dimensionality`` : str or pint.Unit
        Check if ``.attrs["units"]`` is the requested dimensionality

    ``units`` : str or pint.Unit
        Check if ``.attrs["units"]`` matches the requested unit

    Parameters
    ----------
    dax : xarray.DataArray
        xarray object which should be validated
    ref : dict
        reference dictionary

    Returns
    -------
    bool
        True, if the test was a success, else an exception is raised

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import xarray as xr
    >>> import weldx as wx
    >>> dax = xr.DataArray(
    ...     data=np.ones((3, 2, 3)),
    ...     dims=["d1", "d2", "d3"],
    ...     coords={
    ...         "d1": np.array([-1, 0, 2], dtype=int),
    ...         "d2": pd.DatetimeIndex(["2020-05-01", "2020-05-03"]),
    ...         "d3": ["x", "y", "z"],
    ...     }
    ... )
    >>> ref = dict(
    ...     d1={"optional": True, "values": np.array([-1, 0, 2], dtype=int)},
    ...     d2={
    ...         "values": pd.DatetimeIndex(["2020-05-01", "2020-05-03"]),
    ...         "dtype": ["datetime64[ns]", "timedelta64[ns]"],
    ...     },
    ...     d3={"values": ["x", "y", "z"], "dtype": "<U1"},
    ... )
    >>> wx.util.xr_check_coords(dax, ref)
    True

    """
    # only process the coords of the xarray
    if isinstance(dax, (xr.DataArray, xr.Dataset)):
        coords = dax.coords
    elif isinstance(
        dax,
        (
            xr.core.coordinates.DataArrayCoordinates,
            xr.core.coordinates.DatasetCoordinates,
        ),
    ):
        coords = dax
    else:
        raise ValueError("Input variable is not an xarray object")

    for key, check in ref.items():
        # check if the optional key is set to true
        if "optional" in check and check["optional"] and key not in coords:
            # skip this key - it is not in dax
            continue

        if key not in coords:
            # Attributes not found in coords
            raise KeyError(f"Could not find required coordinate '{key}'.")

        # only if the key "values" is given do the validation
        if "values" in check and not (coords[key].values == check["values"]).all():
            raise ValueError(f"Value mismatch in DataArray and ref['{key}']")

        # only if the key "dtype" is given do the validation
        if "dtype" in check:
            dtype_list = check["dtype"]
            if not isinstance(dtype_list, list):
                dtype_list = [dtype_list]
            if not any(
                _check_dtype(coords[key].dtype, var_dtype) for var_dtype in dtype_list
            ):
                raise TypeError(
                    f"Mismatch in the dtype of the DataArray and ref['{key}']"
                )

        if "units" in check:
            units = coords[key].attrs.get("units", None)
            if not units or not U_(units) == U_(check["units"]):
                raise ValueError(
                    f"Unit mismatch in coordinate '{key}'\n"
                    f"Coordinate has unit '{units}', expected '{check['units']}'"
                )

        if "dimensionality" in check:
            units = coords[key].attrs.get("units", None)
            dim = check["dimensionality"]
            if not units or not (
                ureg.get_dimensionality(units) == ureg.get_dimensionality(dim)
            ):
                raise DimensionalityError(
                    units,
                    check["dimensionality"],
                    f"\nDimensionalit mismatch in coordinate '{key}'\n"
                    f"Coordinate has unit '{units}', expected '{dim}'",
                )

    return True


def xr_3d_vector(
    data: np.ndarray,
    time: types_time_like = None,
    add_dims: List[str] = None,
    add_coords: Dict[str, Any] = None,
) -> xr.DataArray:
    """Create an xarray 3d vector with correctly named dimensions and coordinates.

    Parameters
    ----------
    data
        Full data array.
    time
        Optional values that will fill the 'time' dimension.
    add_dims
        Addition dimensions to add between ["time", "c"].
        If either "c" or "time" are present in add_dims they are used to locate the
        dimension position in the passed array.
    add_coords
        Additional coordinates to assign to the xarray.
        ("c" and "time" coordinates will be assigned automatically)

    Returns
    -------
    xarray.DataArray

    """
    if add_dims is None:
        add_dims = []
    if add_coords is None:
        add_coords = {}

    dims = ["c"]
    coords = dict(c=["x", "y", "z"])

    # if data is static but time passed we discard time information
    if time is not None and Q_(data).ndim == 1:
        time = None

    # remove duplicates and keep order
    dims = list(dict.fromkeys(add_dims + dims))

    if time is not None:
        if "time" not in dims:  # prepend to beginning if not already set
            dims = ["time"] + dims
        coords["time"] = time  # type: ignore[assignment]

    if "time" in coords:
        coords["time"] = Time(coords["time"]).index

    coords = dict(add_coords, **coords)

    da = xr.DataArray(data=data, dims=dims, coords=coords).transpose(..., "c")

    return da.astype(float).weldx.time_ref_restore()


def xr_3d_matrix(data: np.ndarray, time: Time = None) -> xr.DataArray:
    """Create an xarray 3d matrix with correctly named dimensions and coordinates.

    Parameters
    ----------
    data :
        Data
    time :
        Optional time data (Default value = None)

    Returns
    -------
    xarray.DataArray

    """
    if time is not None and np.array(data).ndim == 3:
        if isinstance(time, Time):
            time = time.as_pandas_index()
        da = xr.DataArray(
            data=data,
            dims=["time", "c", "v"],
            coords={"time": time, "c": ["x", "y", "z"], "v": [0, 1, 2]},
        )
    else:
        da = xr.DataArray(
            data=data,
            dims=["c", "v"],
            coords={"c": ["x", "y", "z"], "v": [0, 1, 2]},
        )
    return da.astype(float).weldx.time_ref_restore()


def xr_interp_orientation_in_time(
    da: xr.DataArray, time: types_time_like
) -> xr.DataArray:
    """Interpolate an xarray DataArray that represents orientation data in time.

    Parameters
    ----------
    da :
        xarray DataArray containing the orientation as matrix
    time :
        Time data

    Returns
    -------
    xarray.DataArray
        Interpolated data

    """
    if "time" not in da.dims:
        return da
    if len(da.time) == 1:  # remove "time dimension" for static case
        return da.isel({"time": 0})

    time = Time(time).as_pandas_index()
    time_da = Time(da).as_pandas_index()
    time_ref = da.weldx.time_ref

    if not len(time_da) > 1:
        raise ValueError("Invalid time format for interpolation.")

    # extract intersecting times and add time range boundaries of the data set
    times_ds_limits = pd.Index([time_da.min(), time_da.max()])
    times_union = time.union(times_ds_limits)
    times_intersect = times_union[
        (times_union >= times_ds_limits[0]) & (times_union <= times_ds_limits[1])
    ]

    # interpolate rotations in the intersecting time range
    rotations_key = Rot.from_matrix(da.transpose(..., "time", "c", "v").data)
    times_key = time_da.view(np.int64)
    rotations_interp = Slerp(times_key, rotations_key)(times_intersect.view(np.int64))
    da = xr_3d_matrix(rotations_interp.as_matrix(), times_intersect)

    # use interp_like to select original time values and correctly fill time dimension
    da = xr_interp_like(da, {"time": time}, fillna=True)

    # resync and reset to correct format
    if time_ref:
        da.weldx.time_ref = time_ref
    da = da.weldx.time_ref_restore().transpose(..., "time", "c", "v")

    if len(da.time) == 1:  # remove "time dimension" for static case
        return da.isel({"time": 0})

    return da


def xr_interp_coordinates_in_time(
    da: xr.DataArray, times: types_time_like
) -> xr.DataArray:
    """Interpolate an xarray DataArray that represents 3d coordinates in time.

    Parameters
    ----------
    da : xarray.DataArray
        xarray DataArray
    times : pandas.TimedeltaIndex or pandas.DatetimeIndex
        Time data

    Returns
    -------
    xarray.DataArray
        Interpolated data

    """
    if "time" not in da.dims:  # not time dependent
        return da

    times = Time(times).as_pandas_index()
    da = da.weldx.time_ref_unset()
    da = xr_interp_like(
        da, {"time": times}, assume_sorted=True, broadcast_missing=False, fillna=True
    )
    da = da.weldx.time_ref_restore()

    if len(da.time) == 1:  # remove "time dimension" for static cases
        return da.isel({"time": 0})

    return da


# weldx xarray Accessors ---------------------------------------------------------------


@xr.register_dataarray_accessor("weldx")
@xr.register_dataset_accessor("weldx")
class WeldxAccessor:
    """Custom accessor for extending DataArray functionality.

    See http://xarray.pydata.org/en/stable/internals.html#extending-xarray for details.
    """

    def __init__(self, xarray_obj):
        """Construct a WelDX xarray object."""
        self._obj = xarray_obj

    def interp_like(self, da, *args, **kwargs) -> xr.DataArray:
        """Interpolate DataArray along dimensions of another DataArray.

        Provides some utility options for handling out of range values and broadcasting.
        See xr_interp_like for docstring and details.

        Returns
        -------
        xarray.DataArray
            interpolated DataArray

        """
        return xr_interp_like(self._obj, da, *args, **kwargs)

    def time_ref_unset(self) -> xr.DataArray:
        """Convert Timedelta + reference Timestamp to DatetimeIndex."""
        da = self._obj.copy()
        time_ref = da.weldx.time_ref
        if time_ref and is_timedelta64_dtype(da.time):
            da["time"] = da.time.data + time_ref
            da.time.attrs = self._obj.time.attrs  # restore old attributes !
        return da

    def time_ref_restore(self) -> xr.DataArray:
        """Convert DatetimeIndex back to TimedeltaIndex + reference Timestamp."""
        da = self._obj.copy()
        if "time" not in da.coords:
            return da

        if is_datetime64_dtype(da.time):
            time_ref = da.weldx.time_ref
            if time_ref is None:
                time_ref = pd.Timestamp(da.time.data[0])
            da["time"] = pd.DatetimeIndex(da.time.data) - time_ref
            da.time.attrs = self._obj.time.attrs  # restore old attributes !
            da.time.attrs["time_ref"] = time_ref
        return da

    def reset_reference_time(self, time_ref_new: pd.Timestamp) -> xr.DataArray:
        """Return copy with time values shifted to new reference time."""
        da = self._obj.copy()
        da = da.weldx.time_ref_restore()
        da.weldx.time_ref = time_ref_new
        return da

    @property
    def time_ref(self) -> Union[pd.Timestamp, None]:
        """Get the time_ref value or `None` if not set."""
        da = self._obj
        if "time" in da.coords and "time_ref" in da.time.attrs:
            return da.time.attrs["time_ref"]

        return None

    @time_ref.setter
    def time_ref(self, value: types_timestamp_like):
        """Convert INPLACE to new reference time.

        If no reference time exists, the new value will be assigned.
        """
        if value is None:
            raise TypeError("'None' is not allowed as value.")
        if "time" in self._obj.coords:
            value = Time(value).as_timestamp()
            if self._obj.weldx.time_ref and is_timedelta64_dtype(self._obj.time):
                if value == self._obj.weldx.time_ref:
                    return
                _attrs = self._obj.time.attrs
                time_delta = value - self._obj.weldx.time_ref
                self._obj["time"] = self._obj.time.data - time_delta
                self._obj.time.attrs = _attrs  # restore old attributes !
                self._obj.time.attrs["time_ref"] = value  # set new time_ref value
            else:
                self._obj.time.attrs["time_ref"] = value
