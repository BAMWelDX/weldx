# Release Notes

## 0.3.1 (21.03.2021)

### added

- plot function for `MeasurementChain` [[#288]](https://github.com/BAMWelDX/weldx/pull/288)

### ASDF

- remove the `additionalProperties` restriction
  from `single_pass_weld-1.0.0.schema.yaml` [[#283]](https://github.com/BAMWelDX/weldx/pull/283)
- allow scalar `integer` value in `anyOf` of `time_series-1.0.0.yaml` to
  fix [#282](https://github.com/BAMWelDX/weldx/pull/282) [[#286]](https://github.com/BAMWelDX/weldx/pull/286)
- add examples to schema files [[#274]](https://github.com/BAMWelDX/weldx/pull/274)

### changes

- `plot_graph` of the CSM now renders static and time-dependent edges differently
  [[#291]](https://github.com/BAMWelDX/weldx/pull/291)
- use `pint` compatible array syntax in `IsoBaseGroove.to_profile()`
  methods [[#189]](https://github.com/BAMWelDX/weldx/pull/189)
- CSM and LCS plot function get a `scale_vectors` parameter. It scales the plotted coordinate system vectors when using
  matplotlib as backend [[#293]](https://github.com/BAMWelDX/weldx/pull/293)

### fixes

- A warning is now emitted if a `LocalCoordinateSystem` drops a provided time during construction. This usually happens
  if the coordinates and orientation only contain a single data point.
  [[#285]](https://github.com/BAMWelDX/weldx/pull/285)

## 0.3.0 (12.03.2021)

### added

- add `weldx.transformations.CoordinateSystemManager.relabel`
  function [[#219]](https://github.com/BAMWelDX/weldx/pull/219)
- add `SpatialDate` class for storing 3D point data with optional
  triangulation [[#234]](https://github.com/BAMWelDX/weldx/pull/234)
- add `plot` function to `SpatialData`[[#251]](https://github.com/BAMWelDX/weldx/pull/251)
- add `plot` function to visualize `LocalCoordinateSystem` and `CoordinateSystemManager` instances in 3d space
  [[#231]](https://github.com/BAMWelDX/weldx/pull/231)
- add `weldx.welding.groove.iso_9692_1.IsoBaseGroove.cross_sect_area` property to compute cross sectional area between
  the workpieces [[#248]](https://github.com/BAMWelDX/weldx/pull/248).
- add `weldx.welding.util.compute_welding_speed` function [[#248]](https://github.com/BAMWelDX/weldx/pull/248).

### ASDF

- Add possibility to store meta data and content of an external file in an ASDF
  file [[#215]](https://github.com/BAMWelDX/weldx/pull/215)
    - Python class: `weldx.asdf.ExternalFile`
    - Schema: `core/file-1.0.0.yaml`
- Added support for serializing generic metadata and userdata attributes for weldx
  classes. [[#209]](https://github.com/BAMWelDX/weldx/pull/209)
    - the provisional attribute names are `wx_metadata` and `wx_user`
- `None` values are removed from the asdf tree for all `weldx`
  classes. [[#212]](https://github.com/BAMWelDX/weldx/pull/212)
- add `datamodels` directory and example `http://weldx.bam.de/schemas/weldx/datamodels/single_pass_weld-1.0.0.schema`
  schema [[#190]](https://github.com/BAMWelDX/weldx/pull/190)
    - schemas in the `datamodels` directory do not define any tags and can be referenced in other schemas and
      as `custom_schema` when reading/writing `ASDF`-files
    - the `single_pass_weld-1.0.0.schema` is an example schema for a simple, linear, single pass GMAW application
    - add `core/geometry/point_cloud-1.0.0.yaml` schema [[#234]](https://github.com/BAMWelDX/weldx/pull/234)
- add file schema describing a simple linear welding
  application `datamodels/single_pass_weld-1.0.0.schema` [[#256]](https://github.com/BAMWelDX/weldx/pull/256)

### documentation

- Simplify tutorial code and enhance plots by using newly implemented plot functions
  [[#231]](https://github.com/BAMWelDX/weldx/pull/231) [[#251]](https://github.com/BAMWelDX/weldx/pull/251)
- add AWS shielding gas descriptions to documentation [[#270]](https://github.com/BAMWelDX/weldx/pull/270)

### changes

- pass variable names as tuple to `sympy.lambdify` in `MathematicalExpression` to prevent sympy
  deprecation [[#214]](https://github.com/BAMWelDX/weldx/pull/214)
- set `conda-forge` as primary channel in `environment.yaml`
  and `build_env.yaml` [[#214]](https://github.com/BAMWelDX/weldx/pull/214)
- set minimum Python version to 3.7 [[#220]](https://github.com/BAMWelDX/weldx/pull/220)
- `geometry.Profile.rasterize` can return list of rasterized shapes instead of flat ndarray (with
  setting `stack=False`) [[#223]](https://github.com/BAMWelDX/weldx/pull/223)
- `geometry.Profile.plot` plots individual line objects for each shape (instead of a single line
  object) [[#223]](https://github.com/BAMWelDX/weldx/pull/223)
- remove jinja templates and related code [[#228]](https://github.com/BAMWelDX/weldx/pull/228)
- add `stack` option to most `geometry` classes for rasterization [[#234]](https://github.com/BAMWelDX/weldx/pull/234)
- The graph of a `CoordinateSystemManager` is now plotted with `plot_graph` instead of `plot`.
  [[#231]](https://github.com/BAMWelDX/weldx/pull/231)
- add custom `wx_shape` validation for `TimeSeries` and `Quantity` [[#256]](https://github.com/BAMWelDX/weldx/pull/256)
- refactor the `transformations` and `visualization` module into smaller
  files [[#247]](https://github.com/BAMWelDX/weldx/pull/247)
- refactor `weldx.utility` into `weldx.util` [[#247]](https://github.com/BAMWelDX/weldx/pull/247)
- refactor `weldx.asdf.utils` into `weldx.asdf.util` [[#247]](https://github.com/BAMWelDX/weldx/pull/247)
- it is now allowed to merge a time-dependent `timedelta` subsystem into another `CSM` instance if the parent instance
  has set an explicit reference time [[#268]](https://github.com/BAMWelDX/weldx/pull/268)

### fixes

- don't inline time dependent `LCS.coordinates` [[#222]](https://github.com/BAMWelDX/weldx/pull/222)
- fix "datetime64" passing for "timedelta64" in `xr_check_coords` [[#221]](https://github.com/BAMWelDX/weldx/pull/221)
- fix `time_ref_restore` not working correctly if no `time_ref` was
  set [[#221]](https://github.com/BAMWelDX/weldx/pull/221)
- fix deprecated signature in `WXRotation` [[#224]](https://github.com/BAMWelDX/weldx/pull/224)
- fix a bug with singleton dimensions in xarray
  interpolation/matmul [[#243]](https://github.com/BAMWelDX/weldx/pull/243)
- update some documentation formatting and links [[#247]](https://github.com/BAMWelDX/weldx/pull/247)
- fix `wx_shape` validation for scalar `Quantity` and `TimeSeries`
  objects [[#256]](https://github.com/BAMWelDX/weldx/pull/256)
- fix a case where `CSM.time_union()` would return with mixed `DateTimeIndex` and `TimeDeltaIndex`
  types [[#268]](https://github.com/BAMWelDX/weldx/pull/268)

### dependencies

- Add [PyFilesystem](https://docs.pyfilesystem.org/en/latest/)(`fs`) as new dependency
- Add [k3d](https://github.com/K3D-tools/K3D-jupyter) as new dependency
- restrict `scipy<1.6`
  pending [ASDF #916](https://github.com/asdf-format/asdf/issues/916) [[#224]](https://github.com/BAMWelDX/weldx/pull/224)
- set minimum Python version to
  3.8 [[#229]](https://github.com/BAMWelDX/weldx/pull/229)[[#255]](https://github.com/BAMWelDX/weldx/pull/255)
- only import some packages upon first use [[#247]](https://github.com/BAMWelDX/weldx/pull/247)
- Add [meshio](https://pypi.org/project/meshio/) as new dependency [#265](https://github.com/BAMWelDX/weldx/pull/265)

## 0.2.2 (30.11.2020)

### added

- Added `weldx.utility.ureg_check_class` class decorator to enable `pint` dimensionality checks with `@dataclass`
  . [[#179]](https://github.com/BAMWelDX/weldx/pull/179)
- Made coordinates and orientations optional for LCS schema. Missing values are interpreted as unity
  translation/rotation. An empty LCS object represents a unity transformation
  step. [[#177]](https://github.com/BAMWelDX/weldx/pull/177)
- added `weldx.utility.lcs_coords_from_ts` function [[#199]](https://github.com/BAMWelDX/weldx/pull/199)
- add a tutorial with advanced use case for combining groove interpolation with different TCP movements and distance
  calculations [[#199]](https://github.com/BAMWelDX/weldx/pull/199)

### changes

- refactor welding groove classes [[#181]](https://github.com/BAMWelDX/weldx/pull/181)
    - refactor groove codebase to make use of subclasses and classnames for more generic functions
    - add `_meta` attribute to subclasses that map class attributes (dataclass parameters) to common names
    - rework `get_groove` to make use of new class layout and parse function arguments
- create `weldx.welding` module (contains GMAW processes and groove
  definitions) [[#181]](https://github.com/BAMWelDX/weldx/pull/181)
- move `GmawProcessTypeAsdf` to `asdf.tags` folder [[#181]](https://github.com/BAMWelDX/weldx/pull/181)
- reorder module imports in `weldx.__init__` [[#181]](https://github.com/BAMWelDX/weldx/pull/181)
- support timedelta dtypes in ASDF `data_array/variable` [[#191]](https://github.com/BAMWelDX/weldx/pull/191)
- add `set_axes_equal` option to some geometry plot functions (now defaults
  to `False`) [[#199]](https://github.com/BAMWelDX/weldx/pull/199)
- make `utility.sine` public function [[#199]](https://github.com/BAMWelDX/weldx/pull/199)
- switch to setuptools_scm versioning and move package metadata to
  setup.cfg [[#206]](https://github.com/BAMWelDX/weldx/pull/206)

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
    - `LocalCoordinateSystem` now accepts `pd.TimedeltaIndex` and `pint.Quantity` as `time` inputs when provided with a
      reference `pd.Timestamp` as `time_ref` [[#97]](https://github.com/BAMWelDX/weldx/pull/97)
    - `LocalCoordinateSystem` now accepts `Rotation`-Objects
      as `orientation` [[#97]](https://github.com/BAMWelDX/weldx/pull/97)
    - Internal structure of `LocalCoordinateSystem` is now based on `pd.TimedeltaIndex` and a reference `pd.Timestamp`
      instead of `pd.DatetimeIndex`. As a consequence, providing a reference timestamp is now
      optional. [[#126]](https://github.com/BAMWelDX/weldx/pull/126)
- `weldx.utility.xr_interp_like` now accepts non-iterable scalar inputs for
  interpolation [[#97]](https://github.com/BAMWelDX/weldx/pull/97)
- add `pint` compatibility to some `geometry` classes (**experimental**)
    - when passing quantities to constructors (and some functions), values get converted to default unit `mm` and passed
      on as magnitude
    - old behavior is preserved
- add `weldx.utility.xr_check_coords` function to check coordinates of xarray object against dtype and value
  restrictions [[#125]](https://github.com/BAMWelDX/weldx/pull/125)
- add `weldx.utility._sine` to easily create sine TimeSeries [[#168]](https://github.com/BAMWelDX/weldx/pull/168)
- enable `force_ndarray_like=True` as default option when creating the
  global `pint.UnitRegistry` [[#167]](https://github.com/BAMWelDX/weldx/pull/167)
- `ut.xr_interp_like` keeps variable and coordinate attributes from original
  DataArray [[#174]](https://github.com/BAMWelDX/weldx/pull/174)
- rework `ut.to_pandas_time_index` to accept many different formats (LCS,
  DataArray) [[#174]](https://github.com/BAMWelDX/weldx/pull/174)
- add utility functions for handling time coordinates to "weldx"
  accessor [[#174]](https://github.com/BAMWelDX/weldx/pull/174)

### ASDF extension & schemas

- add `WxSyntaxError` exception for custom weldx ASDF syntax errors [[#99]](https://github.com/BAMWelDX/weldx/pull/99)
- add custom `wx_tag` validation and update `wx_property_tag` to allow new
  syntax [[#99]](https://github.com/BAMWelDX/weldx/pull/99) \
  the following syntax can be used:
  ```yaml
  wx_tag: http://stsci.edu/schemas/asdf/core/software-* # allow every version
  wx_tag: http://stsci.edu/schemas/asdf/core/software-1 # fix major version
  wx_tag: http://stsci.edu/schemas/asdf/core/software-1.2 # fix minor version
  wx_tag: http://stsci.edu/schemas/asdf/core/software-1.2.3 # fix patchversion
  ```
- add basic schema layout and `GmawProcess` class for arc welding process
  implementation [[#104]](https://github.com/BAMWelDX/weldx/pull/104)
- add example notebook and documentation for arc welding process [[#104]](https://github.com/BAMWelDX/weldx/pull/104)
- allow optional properties for validation with `wx_shape` by putting the name in brackets
  like `(optional_prop)`[[#176]](https://github.com/BAMWelDX/weldx/pull/176)

### fixes

- fix propagating the `name` attribute when reading an ndarray `TimeSeries` object back from ASDF
  files [[#104]](https://github.com/BAMWelDX/weldx/pull/104)
- fix `pint` regression in `TimeSeries` when mixing integer and float
  values [[#121]](https://github.com/BAMWelDX/weldx/pull/121)

## 0.2.0 (30.07.2020)

### ASDF

- add `wx_unit` and `wx_shape` validators
- add `doc/shape-validation.md` documentation for `wx_shape` [[#75]](https://github.com/BAMWelDX/weldx/pull/75)
- add `doc/unit-validation.md` documentation for `wx_unit`
- add unit validation to `iso_groove-1.0.0.yaml`
- fixed const/enum constraints and properties in `iso_groove-1.0.0.yaml`
- add NetCDF inspired common types (`Dimension`,`Variable`) with corresponding asdf serialization classes
- add asdf serialization classes and schemas for `xarray.DataArray`,
  `xarray.Dataset`, `weldx.transformations.LocalCoordinateSystem` and
  `weldx.transformations.CoordinateSystemManager`.
- add test for `xarray.DataArray`, `xarray.Dataset`, `weldx.transformations.LocalCoordinateSystem` and
  `weldx.transformations.CoordinateSystemManager` serialization.
- allow using `pint.Quantity` coordinates
  in `weldx.transformations.LocalCoordinateSystem` [[#70]](https://github.com/BAMWelDX/weldx/pull/70)
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
- add support for `sympy` expressions with `weldx.core.MathematicalExpression` and ASDF serialization
  in `core/mathematical_expression-1.0.0` [[#70]](https://github.com/BAMWelDX/weldx/pull/70)
  , [[#76]](https://github.com/BAMWelDX/weldx/pull/76)
- add class to describe time series - `weldx.core.TimeSeries` [[#76]](https://github.com/BAMWelDX/weldx/pull/76)
- add `wx_property_tag` validator [[#72]](https://github.com/BAMWelDX/weldx/pull/72)

  the `wx_property_tag` validator restricts **all** properties of an object to a single tag. For example the following
  object can have any number of properties but all must be of type `tag:weldx.bam.de:weldx/time/timestamp-1.0.0`
    ```yaml
    type: object
    additionalProperties: true # must be true to allow any property
    wx_property_tag: "tag:weldx.bam.de:weldx/time/timestamp-1.0.0"  
    ```
  It can be used as a "named" mapping replacement instead of YAML `arrays`.
- add `core/transformation/rotation-1.0.0` schema that implements `scipy.spatial.transform.Rotation`
  and `transformations.WXRotation` class to create custom tagged `Rotation` instances for custom
  serialization. [[#79]](https://github.com/BAMWelDX/weldx/pull/79)
- update requirements to `asdf>=2.7` [[#83]](https://github.com/BAMWelDX/weldx/pull/83)
- update `anyOf` to `oneOf` in ASDF schemas [[#83]](https://github.com/BAMWelDX/weldx/pull/83)
- add `__eq__` functions to `LocalCoordinateSystem`
  and `CoordinateSystemManager` [[#87]](https://github.com/BAMWelDX/weldx/pull/87)

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
