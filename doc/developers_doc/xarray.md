# xarray best practices

## units / quantities
### baseline
- as of now, using `pint.Quantity` as data for `xr.DataArray` seems to be supported for most use-cases.
- `pint.Quantity` is **not** supported in xarray coordinates. Units will get stripped (with the usual pint warning) !

### best practices
- in general, all xarray objects should always have a unit-information !\
  For dimensionless units use the pint-default of `""`.
- use `pint.Quantity` for data whenever possible
- for time-like" coordinates use numpy/pandas time-types/Index-types like `pd.DatetimeIndex`,`pd.Timedeltaindex`,`datetime64[ns]` etc.
- In cases where `pint.Quantity` cannot be used (i.e. coordinates) store the unit in `.attrs["units"]`\
  (follow the guidelines as per [https://github.com/xarray-contrib/pint-xarray](https://github.com/xarray-contrib/pint-xarray) to ensure maximum compatibility for future updates.)


## validating xarray coordinate structures
To validate the coordinate layout of a specific xarray object us `weldx.utility.xr_check_coords`.
Currently checking for dtype and specific values is supported.
Unit support should be added later.

## weldx xarray Accessors
Custom xarray Accessors should be registered under `"weldx"` (see `weldx.utility`). Currently accessors are not actively used.

## naming conventions (up for discussion)
A small list of naming conventions to use throughout `weldx` with regards to internal xarray dimensions/coordinates.
- time-axis coordinates should be name `"time"`
  - dtype for time-axis should either be `datetime64` or `timedelta64`
  - when associating values stored as `timedelta64` with a reference time the reference time should be stored in `.attrs["time_ref"]`
- Cartesian-Coordinates should be named `"c"` and consist of values `["x","y","z"]`
- a (dimensionless) variable describing progression along a trace should run from 0 to 1 and be named `"s"`