# Release Notes

## 0.2.2 (unreleased)
### added
- Added `weldx.utility.ureg_check_class` class decorator to enable `pint` dimensionality checks with `@dataclass`. [[#179]](https://github.com/BAMWelDX/weldx/pull/179)
- Made coordinates and orientations optional for LCS schema. Missing values are interpreted as unity translation/rotation. An empty LCS object represents a unity transformation step. [[#177]](https://github.com/BAMWelDX/weldx/pull/177)

### changes
- refactor welding groove classes [[#181]](https://github.com/BAMWelDX/weldx/pull/181)
  - refactor groove codebase to make use of subclasses and classnames for more generic functions
  - add `_meta` attribute to subclasses that map class attributes (dataclass parameters) to common names
  - rework `get_groove` to make use of new class layout and parse function arguments
- create `weldx.welding` module (contains GMAW processes and groove definitions) [[#181]](https://github.com/BAMWelDX/weldx/pull/181)
- move `GmawProcessTypeAsdf` to `asdf.tags` folder [[#181]](https://github.com/BAMWelDX/weldx/pull/181)
- reorder module imports in `weldx.__init__` [[#181]](https://github.com/BAMWelDX/weldx/pull/181)
- support timedelta dtypes in ASDF `data_array/variable` [[#191]](https://github.com/BAMWelDX/weldx/pull/191)

### ASDF
- refactor ISO 9692-1 groove schema definitions and classes [[#181]](https://github.com/BAMWelDX/weldx/pull/181)
  - move base schema definitions in file `terms-1.0.0.yaml` to `weldx/groove`
  - split old schema into multiple files (1 per groove type) and create folder `iso_9692_1_2013_12`


## 0.2.1 (26.10.2020)
### changes
- Documentation
    - Documentation is [published on readthedocs](https://weldx.readthedocs.io/en/latest/)
    - API documentation is now available
    - New tutorial about 3 dimensional geometries [[#105]](https://github.com/BAMWelDX/weldx/pull/105)
- `CoordinateSystemManager`
    - supports multiple time formats and can get a reference time [[#162]](https://github.com/BAMWelDX/weldx/pull/162)
    - each instance can be named
    - gets a `plot` function to visualize the graph
    - coordinate systems can be updated using `add_cs`
    - supports deletion of coordinate systems
    - instances can now be merged and unmerged
- `LocalCoordinateSystem`
    - `LocalCoordinateSystem` now accepts `pd.TimedeltaIndex` and `pint.Quantity` as `time` inputs when provided with a reference `pd.Timestamp` as `time_ref` [[#97]](https://github.com/BAMWelDX/weldx/pull/97)
    - `LocalCoordinateSystem` now accepts `Rotation`-Objects as `orientation` [[#97]](https://github.com/BAMWelDX/weldx/pull/97)
    - Internal structure of `LocalCoordinateSystem` is now based on `pd.TimedeltaIndex` and a reference `pd.Timestamp` instead of `pd.DatetimeIndex`. As a consequence, providing a reference timestamp is now optional. [[#126]](https://github.com/BAMWelDX/weldx/pull/126)
- `weldx.utility.xr_interp_like` now accepts non-iterable scalar inputs for interpolation [[#97]](https://github.com/BAMWelDX/weldx/pull/97)
- add `pint` compatibility to some `geometry` classes (**experimental**)
  - when passing quantities to constructors (and some functions), values get converted to default unit `mm` and passed on as magnitude
  - old behavior is preserved
- add `weldx.utility.xr_check_coords` function to check coordinates of xarray object against dtype and value restrictions [[#125]](https://github.com/BAMWelDX/weldx/pull/125)
- add `weldx.utility._sine` to easily create sine TimeSeries [[#168]](https://github.com/BAMWelDX/weldx/pull/168)
- enable `force_ndarray_like=True` as default option when creating the global `pint.UnitRegistry` [[#167]](https://github.com/BAMWelDX/weldx/pull/167)
- `ut.xr_interp_like` keeps variable and coordinate attributes from original DataArray [[#174]](https://github.com/BAMWelDX/weldx/pull/174)
- rework `ut.to_pandas_time_index` to accept many different formats (LCS, DataArray) [[#174]](https://github.com/BAMWelDX/weldx/pull/174)
- add utility functions for handling time coordinates to "weldx" accessor [[#174]](https://github.com/BAMWelDX/weldx/pull/174)

### ASDF extension & schemas
- add `WxSyntaxError` exception for custom weldx ASDF syntax errors [[#99]](https://github.com/BAMWelDX/weldx/pull/99)
- add custom `wx_tag` validation and update `wx_property_tag` to allow new syntax [[#99]](https://github.com/BAMWelDX/weldx/pull/99) \
  the following syntax can be used:
  ```yaml
  wx_tag: http://stsci.edu/schemas/asdf/core/software-* # allow every version
  wx_tag: http://stsci.edu/schemas/asdf/core/software-1 # fix major version
  wx_tag: http://stsci.edu/schemas/asdf/core/software-1.2 # fix minor version
  wx_tag: http://stsci.edu/schemas/asdf/core/software-1.2.3 # fix patchversion
  ```
- add basic schema layout and `GmawProcess` class for arc welding process implementation [[#104]](https://github.com/BAMWelDX/weldx/pull/104)
- add example notebook and documentation for arc welding process [[#104]](https://github.com/BAMWelDX/weldx/pull/104)
- allow optional properties for validation with `wx_shape` by putting the name in brackets like `(optional_prop)`[[#176]](https://github.com/BAMWelDX/weldx/pull/176)


### fixes
- fix propagating the `name` attribute when reading an ndarray `TimeSeries` object back from ASDF files [[#104]](https://github.com/BAMWelDX/weldx/pull/104)
- fix `pint` regression in `TimeSeries` when mixing integer and float values [[#121]](https://github.com/BAMWelDX/weldx/pull/121)


## 0.2.0 (30.07.2020)
### ASDF
- add `wx_unit` and `wx_shape` validators
- add `doc/shape-validation.md` documentation for `wx_shape` [[#75]](https://github.com/BAMWelDX/weldx/pull/75)
- add `doc/unit-validation.md` documentation for `wx_unit`
- add unit validation to `iso_groove-1.0.0.yaml` 
- fixed const/enum constraints and properties in `iso_groove-1.0.0.yaml`
- add NetCDF inspired common types (`Dimension`,`Variable`) with corresponding
 asdf serialization classes
- add asdf serialization classes and schemas for `xarray.DataArray`, 
`xarray.Dataset`, `weldx.transformations.LocalCoordinateSystem` and
`weldx.transformations.CoordinateSystemManager`.
- add test for `xarray.DataArray`, `xarray.Dataset`, `weldx.transformations.LocalCoordinateSystem` and
`weldx.transformations.CoordinateSystemManager` serialization.
- allow using `pint.Quantity` coordinates in `weldx.transformations.LocalCoordinateSystem` [[#70]](https://github.com/BAMWelDX/weldx/pull/70)
- add measurement related ASDF serialization classes: [[#70]](https://github.com/BAMWelDX/weldx/pull/70)
  - `equipment/generic_equipment-1.0.0`
  - `measurement/data-1.0.0`
  - `data_transformation-1.0.0`
  - `measurement/error-1.0.0`
  - `measurement/measurement-1.0.0`
  - `measurement/measurement_chain-1.0.0`
  - `measurement/signal-1.0.0`
  - `measurement/source-1.0.0`
- add example notebook for measurement chains in tutorials [[#70]](https://github.com/BAMWelDX/weldx/pull/70)
- add support for `sympy` expressions with `weldx.core.MathematicalExpression` and ASDF serialization in `core/mathematical_expression-1.0.0` [[#70]](https://github.com/BAMWelDX/weldx/pull/70), [[#76]](https://github.com/BAMWelDX/weldx/pull/76)
- add class to describe time series - `weldx.core.TimeSeries` [[#76]](https://github.com/BAMWelDX/weldx/pull/76)
- add `wx_property_tag` validator [[#72]](https://github.com/BAMWelDX/weldx/pull/72)
 
    the `wx_property_tag` validator restricts **all** properties of an object to a single tag.
    For example the following object can have any number of properties but all must be
    of type `tag:weldx.bam.de:weldx/time/timestamp-1.0.0`
    ```yaml
    type: object
    additionalProperties: true # must be true to allow any property
    wx_property_tag: "tag:weldx.bam.de:weldx/time/timestamp-1.0.0"  
    ```
    It can be used as a "named" mapping replacement instead of YAML `arrays`.
- add `core/transformation/rotation-1.0.0` schema that implements `scipy.spatial.transform.Rotation` and `transformations.WXRotation` class to create custom tagged `Rotation` instances for custom serialization. [[#79]](https://github.com/BAMWelDX/weldx/pull/79)
- update requirements to `asdf>=2.7` [[#83]](https://github.com/BAMWelDX/weldx/pull/83)
- update `anyOf` to `oneOf` in ASDF schemas [[#83]](https://github.com/BAMWelDX/weldx/pull/83)
- add `__eq__` functions to `LocalCoordinateSystem` and `CoordinateSystemManager` [[#87]](https://github.com/BAMWelDX/weldx/pull/87)

## 0.1.0 (05.05.2020)
### ASDF
- add basic file/directory layout for asdf files
  - asdf schemas are located in `weldx/asdf/schemas/weldx.bam.de/weldx`
  - tag implementations are in `weldx/asdf/tags/weldx`
- implement support for pint quantities
- implement support for basic pandas time class
- implement base welding classes from AWS/NIST "A Welding Data Dictionary"
- add and implement ISO groove types (DIN EN ISO 9692-1:2013)
- add basic jinja templates and functions for adding simple dataclass objects
- setup package to include and install ASDF extensions and schemas (see setup.py, MANIFEST.in)
- add basic tests for writing/reading all ASDF classes (these only run code without any real checks!)

### module:
- add setup.py package configuration for install
  - required packages
  - package metadata
  - asdf extension entry points
  - version support
- update pandas, scipy, xarray and pint minimum versions (in conda env and setup.py)
- add versioneer
- update options in setup.cfg
- update tool configurations
