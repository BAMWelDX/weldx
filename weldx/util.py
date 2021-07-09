"""Contains package internal utility functions."""
import functools
import json
import sys
import warnings
from collections.abc import Iterable, Sequence
from functools import reduce, wraps
from inspect import getmembers, isfunction
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Collection, Dict, List, Mapping, Union

import numpy as np
import pandas as pd
import pint
import xarray as xr
from asdf.tags.core import NDArrayType
from boltons import iterutils
from pandas.api.types import is_datetime64_dtype, is_object_dtype, is_timedelta64_dtype
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp

from weldx.constants import WELDX_QUANTITY as Q_
from weldx.constants import WELDX_UNIT_REGISTRY as ureg
from weldx.core import MathematicalExpression, TimeSeries

if TYPE_CHECKING:  # pragma: no cover
    import weldx.transformations as tf


class WeldxDeprecationWarning(DeprecationWarning):
    """Deprecation warning type."""


def deprecated(since: str = None, removed: str = None, message: str = None) -> Callable:
    """Mark a functions as deprecated.

    This decorator emits a warning when the function is used.

    Parameters
    ----------
    since :
        The version that marked the function as deprecated
    removed :
        The version that will remove the function
    message :
        Additional information that should be added to the warning

    Returns
    -------
    Callable :
        Wrapped function

    Notes
    -----
    Original source: https://stackoverflow.com/a/30253848/6700329

    """

    def _decorator(func):
        @wraps(func)
        def _new_func(*args, **kwargs):
            wm = f"Call to deprecated function {func.__name__}.\n"
            if since is not None:
                wm += f"Deprecated since: {since}\n"
            if removed is not None:
                wm += f"Removed in: {removed}\n"
            if message is not None:
                wm += message

            warnings.warn(wm, category=WeldxDeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return _new_func

    return _decorator


def ureg_check_class(*args):
    """Decorate class :code:`__init__` function with `pint.UnitRegistry.check`.

    Useful for adding unit checks to classes created with :code:`@dataclass` decorator.

    Parameters
    ----------
    args: str or pint.util.UnitsContainer or None
        Dimensions of each of the input arguments.
        Use :code:`None` to skip argument conversion.

    Returns
    -------
    type
        The class with unit checks added to its :code:`__init__` function.

    Raises
    ------
    TypeError
        If number of given dimensions does not match the number of function parameters.
    ValueError
        If the any of the provided dimensions cannot be parsed as a dimension.

    Examples
    --------
    A simple dataclass could look like this::

        @ureg_check_class("[length]","[time]")
        @dataclass
        class A:
            a: pint.Quantity
            b: pint.Quantity

        A(Q_(3,"mm"),Q_(3,"s"))

    """

    def inner_decorator(
        original_class,
    ):
        # Make copy of original __init__, so we can call it without recursion
        orig_init = original_class.__init__

        # apply pint check decorator
        new_init = ureg.check(None, *args)(orig_init)

        # set new init
        original_class.__init__ = new_init  # Set the class' __init__ to the new one
        return original_class

    return inner_decorator


def dataclass_nested_eq(original_class):
    """Set class :code:`__eq__` using :code:`util.compare_nested` on :code:`__dict__`.

    Useful for implementing :code:`__eq__` on classes
    created with :code:`@dataclass` decorator.

    Parameters
    ----------
    original_class:
        original class to decorate

    Returns
    -------
    type
        The class with overridden :code:`__eq__` function.

    Examples
    --------
    A simple dataclass could look like this::

        @dataclass_nested_eq
        @dataclass
        class A:
            a: np.ndarray

        a = A(np.arange(3))
        b = A(np.arange(3))
        assert a==b

    """

    def new_eq(self, other):
        if not isinstance(other, type(self)):
            return False

        return compare_nested(self.__dict__, other.__dict__)

    # set new eq function
    original_class.__eq__ = new_eq  # Set the class' __eq__ to the new one
    return original_class


def _clean_notebook(file: Union[str, Path]):  # pragma: no cover
    """Clean ID metadata, output and execution count from jupyter notebook cells.

    This function overrides the existing notebook file, use with caution!

    Parameters
    ----------
    file :
        The jupyter notebook filename to clean.

    """
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for cell in data["cells"]:
        cell.pop("id", None)
        if "outputs" in cell:
            cell["outputs"] = []
        if "execution_count" in cell:
            cell["execution_count"] = None

    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
        f.write("\n")


def inherit_docstrings(cls):
    """Inherits (public) docstrings from parent classes.

    Traverses the MRO until it finds a docstring to use, or leave it blank,
    in case no parent has a docstring available.

    Parameters
    ----------
    cls: type
        The class to decorate.

    Returns
    -------
    cls: type
        The class with updated doc strings.

    """
    for name, func in getmembers(
        cls, predicate=lambda x: isfunction(x) or isinstance(x, property)
    ):
        if func.__doc__ or name.startswith("_"):
            continue
        for parent in cls.__mro__[1:]:
            if hasattr(parent, name):
                func.__doc__ = getattr(parent, name).__doc__
                if not func.__doc__:
                    warnings.warn(
                        f"could not derive docstring for {cls}.{name}", stacklevel=1
                    )
    return cls


def sine(
    f: Union[pint.Quantity, str],
    amp: Union[pint.Quantity, str],
    bias: Union[pint.Quantity, str] = None,
    phase: Union[pint.Quantity, str] = Q_(0, "rad"),
) -> TimeSeries:
    """Create a simple sine TimeSeries from quantity parameters.

    f(t) = amp*sin(f*t+phase)+bias

    Parameters
    ----------
    f :
        Frequency of the sine (in Hz)
    amp :
        Sine amplitude
    bias :
        function bias
    phase :
        phase shift

    Returns
    -------
    TimeSeries

    """
    if bias is None:
        amp = Q_(amp)
        bias = 0.0 * amp.u
    expr_string = "a*sin(o*t+p)+b"
    parameters = {"a": amp, "b": bias, "o": Q_(2 * np.pi, "rad") * Q_(f), "p": phase}
    expr = MathematicalExpression(expression=expr_string, parameters=parameters)
    return TimeSeries(expr)


def lcs_coords_from_ts(
    ts: TimeSeries, time: Union[pd.DatetimeIndex, pint.Quantity]
) -> xr.DataArray:
    """Create translation coordinates from a TimeSeries at specific timesteps.

    Parameters
    ----------
    ts:
        TimeSeries that describes the coordinate motion as a 3D vector.
    time
        Timestamps used for interpolation.
        TODO: add support for pd.DateTimeindex as well

    Returns
    -------
    xarray.DataArray :
        A DataArray with correctly labeled dimensions to be used for LCS creation.

    """
    ts_data = ts.interp_time(time=time).data_array
    # assign vector coordinates and convert to mm
    ts_data = ts_data.rename({"dim_1": "c"}).assign_coords({"c": ["x", "y", "z"]})
    ts_data.data = ts_data.data.to("mm").magnitude
    ts_data["time"] = pd.TimedeltaIndex(ts_data["time"].data)
    return ts_data


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
    numpy.ndarray

    """
    return np.array(container, dtype=float)


def to_list(var) -> list:
    """Store the passed variable into a list and return it.

    If the variable is already a list, it is returned without modification.
    If `None` is passed, the function returns an empty list.

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


def to_pandas_time_index(
    time: Union[
        pint.Quantity,
        np.ndarray,
        pd.TimedeltaIndex,
        pd.DatetimeIndex,
        xr.DataArray,
        "tf.LocalCoordinateSystem",
    ],
) -> Union[pd.TimedeltaIndex, pd.DatetimeIndex]:
    """Convert a time variable to the corresponding pandas time index type.

    Parameters
    ----------
    time :
        Variable that should be converted.

    Returns
    -------
    Union[pandas.TimedeltaIndex, pandas.DatetimeIndex] :
        Time union of all input objects

    """
    from weldx.transformations import LocalCoordinateSystem

    _input_type = type(time)

    if isinstance(time, (pd.DatetimeIndex, pd.TimedeltaIndex)):
        return time

    if isinstance(time, LocalCoordinateSystem):
        return to_pandas_time_index(time.time)

    if isinstance(time, pint.Quantity):
        base = "s"  # using low base unit could cause rounding errors
        if not np.iterable(time):  # catch zero-dim arrays
            time = np.expand_dims(time, 0)
        return pd.TimedeltaIndex(data=time.to(base).magnitude, unit=base)

    if isinstance(time, (xr.DataArray, xr.Dataset)):
        if "time" in time.coords:
            time = time.time
        time_index = pd.Index(time.values)
        if is_timedelta64_dtype(time_index) and time.weldx.time_ref:
            time_index = time_index + time.weldx.time_ref
        return time_index

    if not np.iterable(time) or isinstance(time, str):
        time = [time]
    time = pd.Index(time)

    if isinstance(time, (pd.DatetimeIndex, pd.TimedeltaIndex)):
        return time

    # try manual casting for object dtypes (i.e. strings), should avoid integers
    # warning: this allows something like ["1","2","3"] which will be ns !!
    if is_object_dtype(time):
        for func in (pd.DatetimeIndex, pd.TimedeltaIndex):
            try:
                return func(time)
            except (ValueError, TypeError):
                continue

    raise TypeError(
        f"Could not convert {_input_type} " f"to pd.DatetimeIndex or pd.TimedeltaIndex"
    )


def pandas_time_delta_to_quantity(
    time: pd.TimedeltaIndex, unit: str = "s"
) -> pint.Quantity:
    """Convert a `pandas.TimedeltaIndex` into a corresponding `pint.Quantity`.

    Parameters
    ----------
    time : pandas.TimedeltaIndex
        Instance of `pandas.TimedeltaIndex`
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

    if mat_a.shape != mat_b.shape:
        return False
    return np.all(np.isclose(mat_a, mat_b, atol=abs_tol))


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

    if vec_a.size != vec_b.size:
        return False
    return np.all(np.isclose(vec_a, vec_b, atol=abs_tol))


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


def get_time_union(
    list_of_objects: List[Union[pd.DatetimeIndex, pd.TimedeltaIndex]]
) -> Union[pd.DatetimeIndex, pd.TimedeltaIndex]:
    """Generate a merged union of `pandas.DatetimeIndex` from list of inputs.

    The functions tries to merge common inputs that are "time-like" or might have time
    coordinates such as xarray objects, `~weldx.transformations.LocalCoordinateSystem`
    and other time objects. See `to_pandas_time_index` for supported input object types.

    Parameters
    ----------
    list_of_objects :
        list of input objects to merge

    Returns
    -------
    Union[pandas.DatetimeIndex, pandas.TimedeltaIndex]
        Pandas time index class with merged times

    """
    # TODO: add tests

    # see https://stackoverflow.com/a/44762908/11242411
    return reduce(
        lambda x, y: x.union(y), (to_pandas_time_index(idx) for idx in list_of_objects)
    )


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


def _check_dtype(var_dtype, ref_dtype: dict) -> bool:
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

    return True


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
    xarray.DataArray

    """
    if times is not None:
        dsx = xr.DataArray(
            data=data,
            dims=["time", "c"],
            coords={"time": times, "c": ["x", "y", "z"]},
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
    xarray.DataArray

    """
    if times is not None:
        dsx = xr.DataArray(
            data=data,
            dims=["time", "c", "v"],
            coords={"time": times, "c": ["x", "y", "z"], "v": [0, 1, 2]},
        )
    else:
        dsx = xr.DataArray(
            data=data,
            dims=["c", "v"],
            coords={"c": ["x", "y", "z"], "v": [0, 1, 2]},
        )
    return dsx.astype(float)


def xr_interp_orientation_in_time(
    dsx: xr.DataArray, times: Union[pd.DatetimeIndex, pd.TimedeltaIndex]
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
    xarray.DataArray
        Interpolated data

    """
    if "time" not in dsx.coords:
        return dsx

    times = to_pandas_time_index(times)
    times_ds = to_pandas_time_index(dsx)
    time_ref = dsx.weldx.time_ref

    if len(times_ds) > 1:
        # extract intersecting times and add time range boundaries of the data set
        times_ds_limits = pd.Index([times_ds.min(), times_ds.max()])
        times_union = times.union(times_ds_limits)
        times_intersect = times_union[
            (times_union >= times_ds_limits[0]) & (times_union <= times_ds_limits[1])
        ]

        # interpolate rotations in the intersecting time range
        rotations_key = Rot.from_matrix(dsx.transpose(..., "c", "v").data)
        times_key = times_ds.view(np.int64)
        rotations_interp = Slerp(times_key, rotations_key)(
            times_intersect.view(np.int64)
        )
        dsx_out = xr_3d_matrix(rotations_interp.as_matrix(), times_intersect)
    else:
        # TODO: this case is not really well defined, maybe avoid?
        dsx_out = dsx

    # use interp_like to select original time values and correctly fill time dimension
    dsx_out = xr_interp_like(dsx_out, {"time": times}, fillna=True)

    # resync and reset to correct format
    if time_ref:
        dsx_out.weldx.time_ref = time_ref
    dsx_out = dsx_out.weldx.time_ref_restore()

    return dsx_out.transpose(..., "c", "v")


def xr_interp_coordinates_in_time(
    da: xr.DataArray, times: Union[pd.TimedeltaIndex, pd.DatetimeIndex]
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
    da = da.weldx.time_ref_unset()
    da = xr_interp_like(
        da, {"time": times}, assume_sorted=True, broadcast_missing=False, fillna=True
    )
    da = da.weldx.time_ref_restore()
    return da


def _as_valid_timestamp(value: Union[pd.Timestamp, np.datetime64, str]) -> pd.Timestamp:
    """Create a valid (by convention) Timestamp object or raise TypeError.

    Parameters
    ----------
    value: pandas.Timestamp, np.datetime64 or str
        Value to convert to `pd.Timestamp`.

    Returns
    -------
    pandas.Timestamp

    """
    if isinstance(value, (str, np.datetime64)):
        value = pd.Timestamp(value)
    if isinstance(value, pd.Timestamp):  # catch NaT from empty str.
        return value
    raise TypeError("Could not create a valid pandas.Timestamp.")


# geometry --------------------------------------------------------
def triangulate_geometry(geo_data):
    """Stack geometry data and add simple triangulation.

    Parameters
    ----------
    geo_data
        list of rasterized profile data along trace from geometry

    Returns
    -------
    numpy.ndarray, numpy.ndarray
        3D point cloud data and triangulation indexes

    """
    nx = geo_data.shape[2]  # Points per profile
    ny = geo_data.shape[0]  # number of profiles

    data = np.swapaxes(geo_data, 1, 2).reshape((-1, 3))
    triangle_indices = np.empty((ny - 1, nx - 1, 2, 3), dtype=int)
    r = np.arange(nx * ny).reshape(ny, nx)
    triangle_indices[:, :, 0, 0] = r[:-1, :-1]
    triangle_indices[:, :, 1, 0] = r[:-1, 1:]
    triangle_indices[:, :, 0, 1] = r[:-1, 1:]

    triangle_indices[:, :, 1, 1] = r[1:, 1:]
    triangle_indices[:, :, :, 2] = r[1:, :-1, None]
    triangle_indices.shape = (-1, 3)

    return data, triangle_indices


# weldx xarray Accessors --------------------------------------------------------


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
    def time_ref(self, value: pd.Timestamp):
        """Convert INPLACE to new reference time.

        If no reference time exists, the new value will be assigned
        TODO: should None be allowed and pass through or raise TypeError ?
        """
        if "time" in self._obj.coords:
            value = _as_valid_timestamp(value)
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


_eq_compare_nested_input_types = Union[
    Sequence,
    Mapping,
    Collection,
]


def _array_equal(a, b):
    if a.shape != b.shape:
        return False
    return np.all(a == b)


class _Eq_compare_nested:
    """Compares nested data structures like lists, sets, tuples, arrays, etc."""

    # some types need special comparison handling.
    compare_funcs = {
        (np.ndarray, NDArrayType, pint.Quantity, pd.Index): _array_equal,
        (xr.DataArray, xr.Dataset): lambda x, y: x.identical(y),
    }
    # these types will be treated as equivalent.
    _type_equalities = [
        (np.ndarray, NDArrayType),
    ]

    @staticmethod
    def _compare(x, y) -> bool:
        # 1. strict type comparison (exceptions defined in _type_equalities).
        # 2. handle special comparison cases
        if not any(
            (type(x) in e and type(y) in e) for e in _Eq_compare_nested._type_equalities
        ) and type(x) is not type(y):
            return False

        for types, func in _Eq_compare_nested.compare_funcs.items():
            if isinstance(x, types):
                return func(x, y)

        return x == y

    @staticmethod
    def _enter(path, key, value):
        # Do not traverse types defined in compare_funcs. All other types are handled
        # like in boltons.iterutils.default_enter (e.g. descend into nested structures).
        # See `boltons.iterutils.remap` for details.
        if any(isinstance(value, t) for t in _Eq_compare_nested.compare_funcs):
            return value, False

        return iterutils.default_enter(path, key, value)

    @staticmethod
    def _visit(path, key, value, a, b) -> bool:
        """Traverses all elements in `compare_nested` argument a and b...

        and tries to obtain the path `p` in `b` using boltons.iterutils.get_path.
        The following cases can occur:
        1. If the path does not exist in `b` a KeyError will be raised.
        2. If the index `k` does not exist an IndexError is raised.
        3. If the other path exists, a comparison will be made using `_compare`.
           When the elements are not equal traversing `a` will be stopped
           by raising a RuntimeError.
        """
        other_data_structure = iterutils.get_path(b, path)
        other_value = other_data_structure[key]
        if not _Eq_compare_nested._enter(None, key, value)[1]:
            # check lengths of Sequence types first and raise
            # prior starting a more expensive comparison!
            if isinstance(other_data_structure, Sequence) and len(
                other_data_structure
            ) != len(iterutils.get_path(a, path)):
                raise RuntimeError("len does not match")
            if isinstance(other_data_structure, Mapping) and any(
                other_data_structure.keys() ^ iterutils.get_path(a, path).keys()
            ):
                raise RuntimeError("keys do not match")
            if not _Eq_compare_nested._compare(value, other_value):
                raise RuntimeError("not equal")
        return True

    @staticmethod
    def compare_nested(
        a: _eq_compare_nested_input_types, b: _eq_compare_nested_input_types
    ) -> bool:
        """Deeply compares [nested] data structures combined of tuples, lists, dicts...

        Also compares non-nested data-structures.
        Arrays are compared using np.all and xr.DataArray.identical

        Parameters
        ----------
        a :
            a [nested] data structure to compare to `b`.
        b :
            a [nested] data structure to compare to `a`.

        Returns
        -------
        bool :
            True, if all elements (including dict keys) of a and b are equal.

        Raises
        ------
        TypeError
            When a or b is not a nested structure.

        """
        # we bind the input structures a, b to the visit function.
        visit = functools.partial(_Eq_compare_nested._visit, a=a, b=b)

        try:
            iterutils.remap(a, visit=visit, reraise_visit=True)
        # Key not found in b, values not equal, more elements in a than in b
        except (KeyError, RuntimeError, IndexError):
            return False
        except TypeError:
            raise TypeError("either a or b are not a nested data structure.")

        return True


compare_nested = _Eq_compare_nested.compare_nested


def is_interactive_session() -> bool:
    """Check whether this Python session is interactive, e.g. Jupyter/IPython."""
    try:
        get_ipython = sys.modules["IPython"].get_ipython
        if not get_ipython():
            return False
        if "IPKernelApp" not in get_ipython().config:  # pragma: no cover
            return False
    except KeyError:
        return False
    else:
        return True
