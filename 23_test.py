import numpy as np
import pandas as pd
import xarray as xr

from weldx.utility import xr_check_coords, xr_check_dtype

data = np.array([[0, 1], [2, 3]])

time_labels = ["2020-05-01", "2020-05-03"]
d1 = np.array([-1, 1])
d2 = pd.DatetimeIndex(time_labels)
coords = {"d1": d1, "d2": d2, "time_labels": (["d2"], time_labels)}

dax = xr.DataArray(data=data, dims=["d1", "d2"], coords=coords)

dax.attrs = {"answer": 42}

ref = {
    "d1": {"values": np.array([-1, 1])},
    "d2": {"optional": True, "dtype": ["datetime64[ns]", "timedelta64[ns]"]},
}

ref2 = {
    "d1": {"values": np.array([-1, 2])},
    "d2": {"optional": True, "dtype": ["int32"]},
}

ref_additional = {
    "d1": {"values": np.array([-1, 1])},
    "d2": {"optional": True, "dtype": ["datetime64[ns]", "timedelta64[ns]"]},
    "d3": {"values": np.array([-1, 1])},
}

# testing
print("dtype check:")
xr_check_dtype(dax, ref)
print()
print("coords check:")
xr_check_coords(dax, ref)
print()
print("dtype check - wrong dtype:")
xr_check_dtype(dax, ref2)
print()
print("coords check - wrong coords:")
xr_check_coords(dax, ref2)
print()
print("dtype check - additional entry:")
xr_check_dtype(dax, ref_additional)
print()
print("coords check - additional entry:")
xr_check_coords(dax, ref_additional)
