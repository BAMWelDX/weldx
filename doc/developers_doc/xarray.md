# xarray best practices

## units / quantities
### baseline
- as of now, using `pint.Quantity` as data for `xr.DataArray` seems to be supported for most use-cases.
- `pint.Quantity` is **not** supported in xarray coordinates. Units will get stripped (with the usual pint warning) !

### best practices
- in general, all xarray objects should always have a unit-information !\
  For dimensionless units use the pint-default of `""`.
- use `pint.Quantity` for data whenever possible
- for time-like" coordinates use numpy/pandas time-types/Indixes like `pd.DatetimeIndex`,`pd.Timedeltaindex`,`datetime64[ns]` etc.
- In cases where `pint.Quantity` cannot be used (i.e. coordinates) store the unit in `.attrs["units"]`\
  (follow the guidelines as per [https://github.com/xarray-contrib/pint-xarray](https://github.com/xarray-contrib/pint-xarray) to ensure maximum compatibility for future updates.)


## validating xarray coordinate structures
To validate the coordinate layout of a specific xarray object us `weldx.utility.xr_check_coords`.
Currently checking for dtype and specific values is supported.
Unit support should be added later.

## weldx xarray Accessors
Custom xarray Accessors should be registered under `"weldx"` (see `weldx.utility`). Currently accessors are not actively used.