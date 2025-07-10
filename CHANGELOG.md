# Release Notes

## 0.7.2 (10.07.2025)

### Dependencies

- pin `pint-xarray<0.5` due to incopatibility in handling of units with xarray coordinates \[{pull}`991`\]

## 0.7.1 (23.04.2025)

### Documentation

- use `myst-nb-json` for interactive JSON mime rendering with `myst-nb` \[{pull}`978`\]
- update title in `CITATION.cff` \[{pull}`981`\]

### Dependencies

- require `ipython>=8` \[{pull}`978`\]
- retrieve version number at runtime via PEP-0566 metadata \[{pull}`981`\]

## 0.7.0 (17.04.2025)

Release `0.7.0` is a compatibility release to support major new `python`, `numpy` and `asdf` version.

Newly supported version include:

- `python=3.13`
- `numpy=2`
- `asdf=4`

Version `0.7` is planned to be the last release to support `python<3.11`, `numpy<2`and `asdf<4`.

### Fixes

- Handle `copy_arrays` in `WeldxFile` for `asdf>=3.1.0` \[{pull}`972`\].

### Dependencies

- remove upper pin for `asdf` \[{pull}`910`\].
- pin `asdf<5` \[{pull}`959`\].
- remove upper pin for `numpy` \[{pull}`959`\].
- pin `python<3.14` \[{pull}`959`\].
- pin `weldx-widgets>=0.2.5` for viz \[{pull}`959`\].

### ASDF

- update schemas for upcoming ASDF standard 1.6.0 \[{pull}`910`\].

### Deprecations

- remove `WeldxFile.show_asdf_header` \[{pull}`972`\].
- remove `lsc_child_in_parent` argument in `CoordinateSystemManager.add_cs` \[{pull}`976`\].

# Release Notes

## 0.6.9 (19.11.2024)

### Changes

- added support for homogeneous transformation matrices \[{pull}`949`\]:
  - added `create_cs_from_homogeneous_transformation` to `CoordinateSystemManager`
  - added `from_homogeneous_transformation` to `LocalCoordinateSystem`
  - added `as_homogeneous_matrix` to `LocalCoordinateSystem`

### Fixes

- rename (fix typo) argument to `lcs_child_in_parent` in `CoordinateSystemManager.add_cs` \[{pull}`936`\].
- replace usages of `pkg_resources` with `importlib.metadata` \[{pull}`941`\].
- replace usages of `copy_arrays` with `memmap` for `asdf>=3.1.0` \[{pull}`940`\].

### Dependencies

- pin `weldx-widgets>=0.2.3` for viz \[{pull}`939`\].
- pin `pint>=0.21` \[{pull}`941`\].
- pin `python<3.13` \[{pull}`896`\].
- pin `asdf<4` \[{pull}`952`\].

## 0.6.8 (07.06.2024)

### Changes

- use pandas.to_timedelta function to pass units to the TimeDeltaIndex object \[{pull}`918`\].

### Dependencies

- unpin nbval testing dependency.
- pin `python<3.12` \[{pull}`933`\].

## 0.6.7 (2023.08.24)

### Added

- added `weldx.exceptions` module with `WeldxException` \[{pull}`871`\].

### Fixes

- fix compatibility with `pint=0.21` \[{pull}`876`\].
- fix typing compatibility with `pint=0.22` \[{pull}`880`\].
- add `read_buffer_context` and `write_read_buffer_context` to `weldx.asdf.util`
  to fix tests accessing closed files. {issue}`873` \[{pull}`875`\].

### Changes

- removed keyword `dummy_inline_arrays` from function `asdf.util.write_buffer`
  as it was only used internally \[{pull}`875`\].

### ASDF

- update `PintQuantityConverter` and `PintUnitConverter` class assignments for `pint=0.22` \[{pull}`880`\].
- use `ValidationError` from `asdf` instead of `jsonschema` \[{pull}`886`\].

### Dependencies

- require `asdf>=2.15.1` \[{pull}`886`\]

## 0.6.6 (19.04.2023)

Version `0.6.6` is a compatibility release for `xarray>=2023.4.0` .

### Fixes

- explicitly keep reference time as xarray attribute in transformation and interpolation operations \[{pull}`868`\] .

### Dependencies

- compatibility with `xarray=2023.4.0` and `pandas=2` \[{pull}`868`\] .

## 0.6.5 (2023.04.06)

Version `0.6.5` is a compatibility release to support new `asdf` and `xarray` and drops support for Python 3.8.
Please see the new minimal version requirements below.

### Fixes

- fix non quantified xarray parameter inputs to `MathematicalExpression` \[{pull}`864`\].

### Dependencies

- drop Python 3.8 support \[{pull}`866`\]
- require `xarray >= 2022.9.0`, as `LocalCoordinateSystem` now handles merges correctly \[{pull}`861`\].
- require `asdf>=2.15` for new extension style validator interface \[{pull}`863`\].
- require `scipy >=1.6.2` and `networkx >=2.8.2` \[{pull}`866`\]

### ASDF

- Move validators to new extension style and remove legacy extension code \[{pull}`863`\].

## 0.6.4 (09.02.2023)

Version `0.6.4` is a small maintenance release with no changes affecting user code.

### Changes

- `WeldxFile` also shows its JSON widget header representation in a JupyterHub based environment \[{pull}`854`\].

### ASDF

- remove preliminary validators from new style ASDF extension for compatibility with upcoming `asdf` changes \[{pull}`853`\].

## 0.6.3 (02.02.2023)

Version `0.6.3` is a minor release to increase compatibility with Python 3.11 and `asdf 2.14`
with updates to the documentation structure and a new schema for video files.

### Added

- New class to handle image sequence data and videos `weldx.util.media_file.MediaFile` \[{pull}`727`\].

### Dependencies

- Pin `asdf<2.14` due to changes in the extension mechanism \[{pull}`828`\].
- Unpin `asdf` due to fix in `asdf 2.14.3` release \[{pull}`834`\].
- Unpin maximum Python version (again). \[{pull}`837`\].

### Changes

- Remove outdated calls to `weldx.asdf.util.get_highest_tag_version` in
  `TimeSeries` and `SpatialData` converters \[{pull}`831`\].
- Use Ruff in pre-commit-action \[{pull}`824`\].
- Use MyST-NB for documentation building \[{pull}`830`\].
- `WeldxFile` correctly determines whether to display the header via widgets or text \[{pull}`848`\].

### ASDF

- update to `asdf://weldx.bam.de/weldx/schemas/core/file-0.1.1` \[{pull}`727`\]
- add `asdf://weldx.bam.de/weldx/schemas/core/media_file-0.1.0` \[{pull}`727`\]

## 0.6.2 (07.11.2022)

Release `0.6.2` comes with new and updated tutorials and some minor fixes and code improvements.

### Added

- New tutorial that demonstrates the usage of the CSM in conjunction with an existing WelDX file \[{pull}`793`\]
- New tutorial about the `MeasurementChain` class using an existing WelDX file \[{pull}`796`\]

### Changes

- `weldx` now requires pip to install (previously it could be installed by directly invoking setup.py) \[{pull}`774`\].
  From a users perspective nothing changes here, as the package was always recommended to be installed via pip.
- Updated the outdated tutorial about the `LocalCoordinateSystem` \[{pull}`775`\]
- `weld_seam` is now a required field in the `multi_pass_weld` schema \[{pull}`790`\]
- Add section about time-dependent spatial data to the `CoordinateSystemManager` tutorial \[{pull}`792`\]

### Fixes

- `MathematicalExpression` now uses SciPy and NumPy in numerical function evaluation. This enables it to use
  advanced integration methods and fixes lengths computation of `DynamicShapeSegment` \[{pull}`770`\].
- Fix errors in tutorial about quality standards \[{pull}`777`\]
- Correct wrong handling of absolute times of the `TimeSeries` class \[{pull}`791`\]
- Added support for Pint 0.20 \[{pull}`818`\].

## 0.6.1 (19.05.2022)

Release `0.6.1` moves advanced plotting functions over to the `weldx-widgets` package and includes minor bugfixes.

### Changes

- `WeldxFile` now raises a `KeyError`, if the user tries to directly read or manipulate a protected ASDF keyword
  within the file. \[{pull}`759`\]
- Updated the outdated tutorial about the `CoordinateSystemManager` \[{pull}`767`\]

### Fixes

- Fix interactive `view_tree` display \[{pull}`756`\].
- Increase `mypy` coverage and update type hints and GH action \[{pull}`753`\].

### Dependencies

- `weldx` now (optionally) requires `weldx_widgets` to visualize coordinate systems/manager \[{pull}`749`\].
- NumPy is not required as a build time dependency anymore, as Bottleneck now provides binaries on PyPI \[{pull}`749`\].

## 0.6.0 (29.04.2022)

This release includes major changes to the handling and support of units in the API and ASDF schemas.
All classes now support and require quantities where appropriate. Plain numbers without units are no longer supported
and will raise an exception. If the number is truly dimensionless, you still have to wrap it with
the quantity class `weldx.Q_` like this:

```python
my_number = 42.0
my_number_wrapped = weldx.Q_(my_number, "meter")
```

Furthermore, a new class called `GenericSeries` was added. It provides a common interface to describe coordinate-based
data either by discrete values or mathematical expressions. A built-in mechanism lets you derive specialized series with
specific requirements. For more information, have a look
[at the new tutorial](https://weldx.readthedocs.io/en/v0.6.0_a/tutorials/generic_series.html) .

### Added

- `DynamicShapeSegment` \[{pull}`713`\]
- `SpatialSeries` and `DynamicTraceSegment` \[{pull}`699`\]
- first draft of the `multi_pass_weld` schema for WelDX files \[{pull}`667`\]
- add `GenericSeries` as base class supporting arrays and equations \[{pull}`618`\]
- add experimental unit support for `.weldx.interp_like` accessor \[{pull}`518`\]
- new tutorial series that introduces the most important WelDX features
  step by step based on a full example file \[{pull}`555`\]
- add `path` option to `WeldxFile.info` and `WeldxFile.show_asdf_header` \[{pull}`555`\]

### Removed

- removed access to `WeldxFile.data` \[{pull}`744`\]

### Changes

- The `wx_property_tag` validator now also accepts lists of different tags. \[{pull}`670`\]
  When multiple tags are passed, validation will fail if *none* of the supplied patterns match.
- Due to a `pandas` update, using the + operator with `Time` and either a `pandas.TimedeltaIndex` or `pandas.DatetimeIndex`
  now only works if the `Time` instance is on the left-hand side. \[{pull}`684`\]
- `LocalCoordinateSystem` and `CoordinateSystemManager` now support `pint.Quantity` as coordinates.
  Types without units are still supported but are deprecated. \[{pull}`683`\]
- Renamed show_asdf_header of `WeldxFile` to `WeldxFile.header`. \[{pull}`694`\]
- `WeldxFile.custom_schema` now accepts an optional tuple with the first element being a schema to validate upon read,
  the second upon writing the data. \[{pull}`697`\]
- Reshape `SpatialData` coordinates to `(-1, 3)` before exporting with `meshio` for compatibility. \[{pull}`723`\]
- `SpatialData`, `LocalCoordinateSystem` and `CoordinateSystemManager` now require units \[{pull}`731`\]

### Fixes

- `TimeSeries` can now be serialized correctly when using absolute times \[{pull}`677`\]

### Documentation

- update PR link format in the changelog \[{pull}`658`\]
- new tutorial that describes how to work with workpiece data from a WelDX file \[{pull}`681`\]

### ASDF

- update weldx extension and manifest version to `0.1.1` \[{pull}`655`\]
- removed legacy `weldx` tag and schema support \[{pull}`600`\]
- update `core/geometry/spatial_data` to version `0.1.1` with support for multidimensional data \[{pull}`655`\]
- add `wx_shape` validation support for `core/data_array` \[{pull}`655`\]
- update `core/time_series` schema to use `time/time` \[{pull}`677`\]
- update `core/variable` schema to allow single string as data \[{pull}`707`\]
- update the default sorting order of `select_tag` for `WeldxConverter` \[{pull}`733`\]
- add custom validation behavior to `wx_unit` \[{pull}`739`\]

### deprecations

- Coordinates without units for `LocalCoordinateSystem` and `CoordinateSystemManager`

### Dependencies

- `weldx` now works with Python-3.10. \[{pull}`696`\]
- bump to `asdf >=2.8.2` \[{pull}`668`\]
- add `pint-xarray` dependency \[{pull}`518`\]
- bump to `numpy>=1.20` (for numpy.typing) \[{pull}`656`\]
- bump to `pint >=0.18` for typing \[{pull}`664`\]
- bump to `xarray >=0.19` for array creation compatibility \[{pull}`618`\]
- add `bidict` dependency \[{pull}`618`\]
- set `networkx !=2.7` for plotting compatibility (for now) \[{pull}`714`, {pull}`722`\]

## 0.5.2 (18.11.2021)

### Added

- `CoordinateSystemManager` can now delete already assigned data with
  `CoordinateSystemManager.delete_data`. {issue}`644` \[{pull}`645`\]
- `WeldxFile` handles an `array_inline_threshold` parameter to
  indicate if short arrays will be serialized as strings, or as binary
  block. Note that this does not affect arrays, which are being shared
  across several objects in the same file. \[{pull}`643`\]

### Changes

- `WeldxFile` now raises an exception, if a warning is emitted during
  loading the weldx ASDF extension, this should prevent erroneous data
  during loading, for example missing dependencies. \[{pull}`641`\]
- `WeldxFile` now hides ASDF added fields like history and asdf_library
  from the dictionary interface. To access these, there are separate
  properties \[{pull}`625`\].
- Allow handling of `time` values as singular coordinates without
  dimensions in some classes \[{pull}`635`\].

### Fixes

- Fix wrong dimension order being passed through in `SpatialData`
  \[{pull}`635`\].

### Dependencies

- Removed `ipykernel` dependency. \[{pull}`634`\]
- The `K3D` implementation now uses the experimental
  `weldx-widgets` backend if available \[{pull}`636`\]

## 0.5.1 (04.11.2021)

### Added

- `Time.duration` to get the covered duration of the data and
  `Time.resample` to get a new `Time` instance with resampled time data
  within the same boundaries as the original object \[{pull}`603`\]
- Added `weldx.geometry.SpatialData.limits` to calculate coordinate
  boundaries. \[{pull}`604`\]
- Added `weldx.asdf.util.get_schema_tree` utility to display schema
  files. \[{pull}`610`\]

### Changes

- All public interfaces of the `weldx.geometry` module classes now
  require the usage of units and support unit strings as inputs.
  \[{pull}`588`\]
- `CoordinateSystemManager.time_union` now returns a `Time` instance
  instead of a pandas type \[{pull}`603`\]
- `SpatialData` now supports time dependent data. \[{pull}`612`\]
- Renamed the parameter `coordinate_system_name` of
  `CoordinateSystemManager.assign_data` to `reference_system` and
  added the parameter `target_system`. If the latter one is not
  `None`, the data will be transformed and stored at this coordinate
  system. \[{pull}`612`\]
- improve dimension handling of `SpatialData` \[{pull}`622`\]
- The `MathematicalExpression` now supports `xarray.DataArray` as
  parameters. Furthermore, multidimensional parameters of a
  `MathematicalExpression` that is passed to a `TimeSeries` are no
  longer required to have an extra dimension that represents time.
  \[{pull}`621`\]

### Fixes

- fix broken `Time.all_close` to now work as intended \[{pull}`603`\]
- fix `weldx.asdf.util.get_yaml_header` to work correctly with windows
  line endings. \[{pull}`609`\]

### Documentation

- move the schema documentation to [BAMWelDX/weldx-standard](https://github.com/BAMWelDX/weldx-standard) \[{pull}`594`\]

### ASDF

- fix `process` missing as required property in
  `single_pass_weld-0.1.0.yaml` \[{pull}`627`\]

### deprecations

- removed `welding.util.lcs_coords_from_ts` \[{pull}`620`\]

### Dependencies

- adjust code to support pint 0.18 unit formatting. \[{pull}`616`\]

## 0.5.0 (12.10.2021)

Release `0.5.0` brings a major rework of the `weldx` standard and many
API improvements:

### Highlights

- `weldx` now internally uses the reworked ASDF extension API. The
  schema and tag naming patterns have also changed to the recommended
  `asdf://` format.
- New `Time` class to make handling of time related functionality
  easier and consistent.
- many internal reworks to streamline the code.
- rework the [API documentation](https://weldx.readthedocs.io/en/latest/api.html) to show the most
  important classes.

### Compatibility

- the `0.5.x` versions will retain backwards compatibility with files
  generated with the `0.4.x` versions and convert them to the new
  naming schema on save. Support for the old schemas will be dropped in
  the `0.6` release.

### Added

- added "units" (exact) and "dimensionality" (dimensionality
  compatible) checking options to `util.xr_check_coords` \[{pull}`442`\]
- `Time` class that can be initialized from several other time types
  and provides time related utility functions \[{pull}`433`\]
- `TimeSeries` now supports setting a `reference_time` absolute time
  values for interpolation \[{pull}`440`\]
- `LocalCoordinateSystem.from_axis_vectors` and
  `CoordinateSystemManager.create_cs_from_axis_vectors` \[{pull}`472`\]
- added PyTest flags to use `WeldxFile` internally in
  `asdf.util.read_buffer` and `asdf.util.write_buffer` \[{pull}`469`\].
- added classes and functions at the top-level of the package to the
  documentation \[{pull}`437`\].
- added `weldx.asdf.util.get_highest_tag_version` utility function
  \[{pull}`523`\].
- added support for parsing temperature deltas with `Δ°` notation
  \[{pull}`565`\].
- `WeldxFile.info` to print a quick content overview to the stdout.
  \[{pull}`576`\].

### Removed

- removed functions now covered by `Time`:
  `pandas_time_delta_to_quantity`, `to_pandas_time_index`,
  `get_time_union` \[{pull}`448`\]
- removed custom `wx_tag` validator \[{pull}`461`\]
- attrdict dependency replaced with a custom implementation of
  recursive dicts \[{pull}`470`\].
- `from_xyz`, `from_xy_and_orientation`,
  `from_yz_and_orientation` and `from_xz_and_orientation` from
  `LocalCoordinateSystem`. Use
  `LocalCoordinateSystem.from_axis_vectors` instead. \[{pull}`472`\]
- `create_cs_from_xyz`, `create_cs_from_xy_and_orientation`,
  `create_cs_from_yz_and_orientation` and
  `create_cs_from_xz_and_orientation` from `CoordinateSystemManager`.
  Use `CoordinateSystemManager.create_cs_from_axis_vectors` instead.
  \[{pull}`472`\]
- `is_column_in_matrix`, `is_row_in_matrix`, `to_float_array`,
  `to_list`, `matrix_is_close`, `vector_is_close` and
  `triangulate_geometry` from `weldx.util` \[{pull}`490`\]
- remove the `:` syntax from `wx_shape` validation \[{pull}`537`\]

### Changes

- move `welding.util.sine` utility function to `weldx.welding.util`
  \[{pull}`439`\]
- `LocalCoordinateSystem` and `CoordinateSystemManager` function
  parameters related to time now support all types that are also
  supported by the new `Time` class \[{pull}`448`\]
- `LocalCoordinateSystem.interp_time` returns static systems if only a
  single time value is passed or if there is no overlap between the
  interpolation time range and the coordinate systems time range. This
  also affects the results of some `CoordinateSystemManager` methods
  (`CoordinateSystemManager.get_cs` ,
  `CoordinateSystemManager.interp_time`) \[{pull}`476`\]
- `util.WeldxAccessor.time_ref` setter now raises a `TypeError` if
  `None` is passed to it \[{pull}`489`\]
- move xarray related utility functions into `weldx.util.xarray` and
  all other ones into `weldx.util.util`. Content from both submodules
  can still be accessed using `weldx.util` \[{pull}`490`\]
- xarray implementations for the `LocalCoordinateSystem` now operate on
  time as a dimension instead of coordinates \[{pull}`486`\]
- `WeldxFile.copy` now creates a copy to a (optional) file. Before it
  just returned a dictionary \[{pull}`504`\].
- changed the default `pint.Unit` formatting to short notation `:~`
  \[{pull}`519`\]. (the asdf
  serialization still uses long notation (\[{pull}`560`\]))
- `welding_current` and `welding_voltage` in the single-pass weld
  schema now expect the tag
  `"asdf://weldx.bam.de/weldx/tags/core/time_series-0.1.*"` instead
  of `"asdf://weldx.bam.de/weldx/tags/measurement/signal-0.1.*"`
  \[{pull}`578`\].
- `Geometry.__init__` now also accepts an `iso.IsoBaseGroove` as
  `profile` parameter \[{pull}`583`\].
- Renamed `Geometry.__init__` parameter `trace` to
  `trace_or_length`. A `pint.Quantity` is now an accepted input. In
  this case the value will be used to create a linear trace of the
  given length \[{pull}`583`\].

### Fixes

- `WeldxFile.show_asdf_header` prints output on console, before it only
  returned the header as parsed dict and string representation. Also
  tweaked efficiency by not writing binary blocks \[{pull}`459`\], \[{pull}`469`\].
- Merging and unmerging multiple `CoordinateSystemManager` instances
  now correctly preserves all attached data. \[{pull}`494`\].
- `util.compare_nested` can compare sets \[{pull}`496`\]
- `WeldxFile` respects `mode` argument also for BytesIO and file
  handles \[{pull}`539`\].

### Documentation

- added installation guide with complete environment setup (Jupyterlab
  with extensions) and possible problems and solutions \[{pull}`450`\]
- split API documentation into user classes/functions and a full API
  reference \[{pull}`469`\].
- added citation metadata in `CITATION.cff` \[{pull}`568`\].

### ASDF

- all schema version numbers set to `0.1.0` \[{pull}`535`\].

- add `time/time` schema to support `Time` class \[{pull}`463`\].

- rework ASDF extension to new asdf 2.8 API \[{pull}`467`\]

  - move schema files to `weldx/schemas`
  - create extension manifest in `weldx/manifests`. The manifest
    also contains tag mappings for legacy tag names for backwards
    compatibility.
  - move tag module to `weldx/tags`
  - refactor all asdf uris to new `asdf://` naming convention, see
    <https://asdf.readthedocs.io/en/latest/asdf/extending/uris.html#entities-identified-by-uri>
  - replaced all referenced weldx tag versions in schemas with
    `0.1.*`
  - refactor
    `asdf://weldx.bam.de/weldx/schemas/datamodels/single_pass_weld-1.0.0.schema`
    to
    `asdf://weldx.bam.de/weldx/schemas/datamodels/single_pass_weld-0.1.0`
    and enable schema test
  - add legacy class for validators support in
    `weldx.asdf._extension.py`
  - asdf utility functions `weldx.asdf.util.uri_match`,
    `weldx.asdf.util.get_converter_for_tag` and
    `weldx.asdf.util.get_weldx_extension`
  - add `devtools/scripts/update_manifest.py` to auto update
    manifest from extension metadata
  - custom shape validation must now be implemented via staticmethod
    `weldx.asdf.types.WeldxConverter.shape_from_tagged`

- provide legacy schema support in
  `weldx/schemas/weldx.bam.de/legacy` \[{pull}`533`\]

- rewrote
  `asdf://weldx.bam.de/weldx/schemas/core/transformations/coordinate_system_hierarchy`
  schema for the `CoordinateSystemManager`. It uses the digraph schemas
  to serialize the coordinate system structure. \[{pull}`497`\]

- add `asdf://weldx.bam.de/weldx/schemas/unit/quantity` and
  `asdf://weldx.bam.de/weldx/schemas/unit/unit` schemas \[{pull}`522`\]

- use `asdf://weldx.bam.de/weldx/schemas/unit/quantity` instead of
  `tag:stsci.edu:asdf/unit/quantity-1.1.0` \[{pull}`542`\].

- refactor properties named `unit` to `units` and use `unit/unit`
  tag \[{pull}`551`\].

- reworked the optional syntax for `wx_shape` validation \[{pull}`571`\].

### Dependencies

- set `k3d!=2.10` because of conda dependency bugs \[{issue}`474`, {pull}`577`\]
- Python 3.10 is not supported in this version. \[{pull}`575`\]

## 0.4.1 (20.07.2021)

### Added

- `closed_mesh` parameter to `Geometry.spatial_data` and
  `SpatialData.from_geometry_raster` \[{pull}`414`\]
- `TimeSeries.plot` and `measurement.Signal.plot` \[{pull}`420`\]
- abstract base class `time.TimeDependent` \[{pull}`460`\]

### Changes

- `TimeSeries` `__init__` accepts `xarray.DataArray` as `data`
  parameter \[{pull}`429`\]
- The `LocalCoordinateSystem.time` and `TimeSeries.time` now return an
  instance of `Time` \[{pull}`464`\]
- Fix wrong and incomplete type-hints \[{pull}`435`\]

### ASDF

- sort `List[str]` before serialization of most `weldx` classes to
  avoid random reordering in the same file and enforce consistency.
  \[{pull}`430`\]

### deprecations

- `lcs_coords_from_ts` will be removed in version 0.5.0 \[{pull}`426`\]

## 0.4.0 (13.07.2021)

Release `0.4.0` brings many new major features to `weldx`

### Highlights

- [Quality Standards](https://weldx.readthedocs.io/en/latest/tutorials/quality_standards.html):
  Users can now create and integrate their own quality standards by
  defining new ASDF schema definitions and loading them into weldx. It
  is possible to add new definitions or modify existing schemas to
  create your own flavour of the weldx standard.
- [WeldxFile](https://weldx.readthedocs.io/en/latest/tutorials/weldxfile.html):
  Create/Load/Modify asdf files directly using `WeldxFile` with many
  helpful utility functions included.
- [TimeSeries support](https://weldx.readthedocs.io/en/latest/tutorials/welding_example_02_weaving.html#add-a-sine-wave-to-the-TCP-movement)
  for `LocalCoordinateSystem`: It is now possible to define a
  time-dependent `LocalCoordinateSystem` with a simple function by
  passing a `TimeSeries` object with a `MathematicalExpression` as
  `coordinates`. For an example, click the link above.
- [MeasurementChain](https://weldx.readthedocs.io/en/latest/tutorials/measurement_chain.html)
  The `measurement.MeasurementChain` has been reworked to be easier and
  more flexible to use.

full changelog below:

### Added

- add support for quality standards. Further information can be found
  in the corresponding new tutorial. \[{pull}`211`\]
- added `asdf.util.get_schema_path` helper function \[{pull}`325`\]
- added `util.compare_nested` to check equality of two nested data
  structures. \[{pull}`328`\]
- added `WeldxFile` wrapper to handle asdf files with history and
  schemas more easily. \[{pull}`341`\].
- add `"step"` as additional method to `util.xr_interp_like` \[{pull}`363`\]
- add `util.dataclass_nested_eq` decorator for dataclasses with
  array-like fields \[{pull}`378`\]
- adds a `asdf.util.dataclass_serialization_class` utility function
  that automatically generates the asdf serialization class for python
  dataclasses. \[{pull}`380`\]
- Added method to set the interpolation method to the `TimeSeries`
  \[{pull}`353`\]
- Add `TimeSeries.is_discrete` and `TimeSeries.is_expression`
  properties to `TimeSeries` \[{pull}`366`\]
- Add `measurement.MeasurementChain.output_signal` property that
  returns the output signal of the `measurement.MeasurementChain`
  \[{pull}`394`\]

### Changes

- `WXRotation.from_euler` now accepts a `pint.Quantity` as input.
  \[{pull}`318`\]

- move tests folder to `weldx/tests` \[{pull}`323`\]

- `asdf.util.get_yaml_header` received a new option parse, which
  optionally returns the parsed YAML header as
  `asdf.tagged.TaggedDict`. \[{pull}`338`\]

- refactor `asdf_json_repr` into `asdf.util.view_tree` \[{pull}`339`\]

- `TimeSeries.interp_time` \[{pull}`353`\]

  - now returns a new `TimeSeries` instead of a `xarray.DataArray`
  - if the data has already been interpolated before, a warning is
    emitted
  - `TimeSeries` supports now all interpolation methods supported by
    xarray

- The `measurement.MeasurementChain` is now internally based on a
  `networkx.DiGraph`. New functions are also added to the class to
  simplify its usage. \[{pull}`326`\] The following
  additional changes were applied during the update of the
  `measurement.MeasurementChain`:

  - renamed `DataTransformation` class to
    `measurement.SignalTransformation`
  - renamed `Source` to `measurement.SignalSource`
  - Added additional functionality to `measurement.Signal`,
    `measurement.SignalTransformation` and `GenericEquipment`
  - Removed `Data` class
  - Updated asdf schemas of all modified classes and the ones that
    contained references to those classes

- allow input of string quantities in `MathematicalExpression`
  parameters and a few other places \[{pull}`402`\] \[{pull}`416`\]

- `LocalCoordinateSystem` `__init__` now accepts a `TimeSeries` as
  input. All methods of the `CoordinateSystemManager` also support this
  new behavior \[{pull}`366`\]

- During the creation of a `WeldxFile` the path of a passed custom
  schema is resolved automatically \[{pull}`412`\].

### Documentation

- Add new tutorial about the `measurement.MeasurementChain` \[{pull}`326`\]
- Updated the measurement tutorial \[{pull}`326`\]

### ASDF

- fix inline array serialization for new 64bit inline limit \[{pull}`218`\]
- add `asdf.extension.WeldxExtension.yaml_tag_handles` to
  `WeldxExtension` \[{pull}`218`\]
- add `uuid-1.0.0.yaml` schema as basic version 4 UUID implementation
  \[{pull}`330`\]
- add `core/graph/di_node`, `core/graph/di_edge` &
  `core/graph/di_graph` for implementing a generic `networkx.DiGraph`
  \[{pull}`330`\]
- compatibility with ASDF-2.8 \[{pull}`355`\]
- data attached to an instance of the `CoordinateSystemManager` is now
  also stored in a WelDX file \[{pull}`364`\]
- replace references to base asdf tags with `-1.*` version wildcard
  \[{pull}`373`\]
- update `single-pass-weldx.1.0.0.schema` to allow groove types by
  wildcard \[{pull}`373`\]
- fix attributes serialization of DataSet children \[{pull}`384`\].
- update `wx_shape` syntax in `local_coordinate_system-1.0.0`
  \[{pull}`366`\]
- add custom `wx_shape` validation to `variable-1.0.0` \[{pull}`366`\]
- remove outdated `TimeSeries` shape validation code \[{pull}`399`\]
- use asdf tag validation pattern for `wx_property_tag` \[{pull}`410`\]
- update `MathematicalExpression` schema \[{pull}`410`\]

### Fixes

- added check for symmetric key difference for mappings with
  `util.compare_nested` \[{pull}`377`\]

### deprecations

- deprecate `wx_tag` validator (use default asdf uri pattern
  matching) \[{pull}`410`\]

## 0.3.3 (30.03.2021)

This is a bugfix release to correctly include the asdf schema files in
conda builds. \[{pull}`314`\]

### ASDF

- fix required welding wire metadata in
  `single-pass-weldx.1.0.0.schema` \[{pull}`316`\]

## 0.3.2 (29.03.2021)

### Added

- `util.deprecated` decorator \[{pull}`295`\]

### Removed

- `rotation_matrix_x`, `rotation_matrix_y` and
  `rotation_matrix_z` \[{pull}`317`\]

### Dependencies

- restrict `scipy!=1.6.0,scipy!=1.6.1` \[{pull}`300`\]

### ASDF

- add validators to `rotation-1.0.0.yaml` &
  `gas_component-1.0.0.yaml` \[{pull}`303`\]
- update descriptions in `single-pass-weldx.1.0.0.schema` \[{pull}`308`\]

### Fixes

- prevent creation of `welding.groove.iso_9692_1.IsoBaseGroove` with
  negative parameters \[{pull}`306`\]

## 0.3.1 (21.03.2021)

### Added

- plot function for `measurement.MeasurementChain` \[{pull}`288`\]

### ASDF

- remove the `additionalProperties` restriction from
  `single_pass_weld-1.0.0.schema.yaml` \[{pull}`283`\]
- allow scalar `integer` value in `anyOf` of
  `time_series-1.0.0.yaml` to fix \[{pull}`282`, {pull}`286`\]
- add examples to schema files \[{pull}`274`\]

### Changes

- `CoordinateSystemManager.plot_graph` now renders static and
  time-dependent edges differently \[{pull}`291`\]
- use `pint` compatible array syntax in
  `welding.groove.iso_9692_1.IsoBaseGroove.to_profile` methods \[{pull}`189`\]
- CSM and LCS plot function get a `scale_vectors` parameter. It
  scales the plotted coordinate system vectors when using matplotlib as
  backend \[{pull}`293`\]

### Fixes

- A warning is now emitted if a `LocalCoordinateSystem` drops a
  provided time during construction. This usually happens if the
  coordinates and orientation only contain a single data point. \[{pull}`285`\]

## 0.3.0 (12.03.2021)

### Added

- add `CoordinateSystemManager.relabel` function \[{pull}`219`\]
- add `SpatialData` class for storing 3D point data with optional
  triangulation \[{pull}`234`\]
- add `plot` function to `SpatialData` \[{pull}`251`\]
- add `plot` function to visualize `LocalCoordinateSystem` and
  `CoordinateSystemManager` instances in 3d space \[{pull}`231`\]
- add `weldx.welding.groove.iso_9692_1.IsoBaseGroove.cross_sect_area`
  property to compute cross sectional area between the workpieces
  \[{pull}`248`\].
- add `weldx.welding.util.compute_welding_speed` function \[{pull}`248`\].

### ASDF

- Add possibility to store meta data and content of an external file in
  an ASDF file \[{pull}`215`\]

  - Python class: `asdf.ExternalFile`
  - Schema: `core/file-1.0.0.yaml`

- Added support for serializing generic metadata and userdata
  attributes for weldx classes. \[{pull}`209`\]

  - the provisional attribute names are `wx_metadata` and
    `wx_user`

- `None` values are removed from the asdf tree for all `weldx` classes.
  \[{pull}`212`\]

- add `datamodels` directory and example
  `http://weldx.bam.de/schemas/weldx/datamodels/single_pass_weld-1.0.0.schema`
  schema \[{pull}`190`\]

  - schemas in the `datamodels` directory do not define any tags and
    can be referenced in other schemas and as `custom_schema` when
    reading/writing `ASDF`-files
  - the `single_pass_weld-1.0.0.schema` is an example schema for a
    simple, linear, single pass GMAW application
  - add `core/geometry/point_cloud-1.0.0.yaml` schema \[{pull}`234`\]

- add file schema describing a simple linear welding application
  `datamodels/single_pass_weld-1.0.0.schema` \[{pull}`256`\]

### Documentation

- Simplify tutorial code and enhance plots by using newly implemented
  plot functions \[{pull}`231`\] \[{pull}`251`\]
- add AWS shielding gas descriptions to documentation \[{pull}`270`\]

### Changes

- pass variable names as tuple to `sympy.lambdify` in
  `MathematicalExpression` to prevent sympy deprecation \[{pull}`214`\]
- set `conda-forge` as primary channel in `environment.yaml` and
  `build_env.yaml` \[{pull}`214`\]
- set minimum Python version to 3.7 \[{pull}`220`\]
- `geometry.Profile.rasterize` can return list of rasterized shapes
  instead of flat ndarray (with setting `stack=False`) \[{pull}`223`\]
- `geometry.Profile.plot` plots individual line objects for each shape
  (instead of a single line object) \[{pull}`223`\]
- remove jinja templates and related code \[{pull}`228`\]
- add `stack` option to most `geometry` classes for rasterization
  \[{pull}`234`\]
- The graph of a `CoordinateSystemManager` is now plotted with
  `CoordinateSystemManager.plot_graph` instead of
  `CoordinateSystemManager.plot`. \[{pull}`231`\]
- add custom `wx_shape` validation for `TimeSeries` and
  `pint.Quantity` \[{pull}`256`\]
- refactor the `transformations` and `visualization` module into
  smaller files \[{pull}`247`\]
- refactor `weldx.utility` into `util` \[{pull}`247`\]
- refactor `weldx.asdf.utils` into `asdf.util` \[{pull}`247`\]
- it is now allowed to merge a time-dependent `timedelta` subsystem
  into another `CoordinateSystemManager` instance if the parent
  instance has set an explicit reference time \[{pull}`268`\]

### Fixes

- don not inline time dependent `LocalCoordinateSystem.coordinates`
  \[{pull}`222`\]
- fix "datetime64" passing for "timedelta64" in `util.xr_check_coords`
  \[{pull}`221`\]
- fix `util.WeldxAccessor.time_ref_restore` not working correctly if no
  `time_ref` was set \[{pull}`221`\]
- fix deprecated signature in `WXRotation` \[{pull}`224`\]
- fix a bug with singleton dimensions in xarray interpolation/matmul
  \[{pull}`243`\]
- update some documentation formatting and links \[{pull}`247`\]
- fix `wx_shape` validation for scalar `pint.Quantity` and
  `TimeSeries` objects \[{pull}`256`\]
- fix a case where `CoordinateSystemManager.time_union` would return
  with mixed `pandas.DatetimeIndex` and `pandas.TimedeltaIndex` types
  \[{pull}`268`\]

### Dependencies

- Add [PyFilesystem](https://docs.pyfilesystem.org/en/latest/)
  (`fs`) as new dependency
- Add [k3d](https://github.com/K3D-tools/K3D-jupyter) as new
  dependency
- restrict `scipy<1.6` pending [ASDF #916](https://github.com/asdf-format/asdf/issues/916) \[{pull}`224`\]
- set minimum Python version to 3.8 \[{pull}`229`\]\[{pull}`255`\]
- only import some packages upon first use \[{pull}`247`\]
- Add [meshio](https://pypi.org/project/meshio/) as new dependency
  \[{pull}`265`\]

## 0.2.2 (30.11.2020)

### Added

- Added `util.ureg_check_class` class decorator to enable `pint`
  dimensionality checks with `@dataclass` \[{pull}`179`\].
- Made coordinates and orientations optional for LCS schema. Missing
  values are interpreted as unity translation/rotation. An empty LCS
  object represents a unity transformation step. \[{pull}`177`\]
- added `welding.util.lcs_coords_from_ts` function \[{pull}`199`\]
- add a tutorial with advanced use case for combining groove
  interpolation with different TCP movements and distance calculations
  \[{pull}`199`\]

### Changes

- refactor welding groove classes \[{pull}`181`\]

  - refactor groove codebase to make use of subclasses and classnames
    for more generic functions
  - add `_meta` attribute to subclasses that map class attributes
    (dataclass parameters) to common names
  - rework `get_groove` to make use of new class layout and parse
    function arguments

- create `welding` module (contains GMAW processes and groove
  definitions) \[{pull}`181`\]

- move `GmawProcessTypeAsdf` to `asdf/tags` folder \[{pull}`181`\]

- reorder module imports in `weldx.__init__` \[{pull}`181`\]

- support timedelta dtypes in ASDF `data_array/variable` \[{pull}`191`\]

- add `set_axes_equal` option to some geometry plot functions (now
  defaults to `False`) \[{pull}`199`\]

- make `welding.util.sine` public function \[{pull}`199`\]

- switch to setuptools_scm versioning and move package metadata to
  setup.cfg \[{pull}`206`\]

### ASDF

- refactor ISO 9692-1 groove schema definitions and classes \[{pull}`181`\]

  - move base schema definitions in file `terms-1.0.0.yaml` to
    `weldx/groove`
  - split old schema into multiple files (1 per groove type) and
    create folder `iso_9692_1_2013_12`

## 0.2.1 (26.10.2020)

### Changes

- Documentation

  - Documentation is [published on readthedocs](https://weldx.readthedocs.io/en/latest/)
  - API documentation is now available
  - New tutorial about 3 dimensional geometries \[{pull}`105`\]

- `CoordinateSystemManager`

  - supports multiple time formats and can get a reference time
    \[{pull}`162`\]
  - each instance can be named
  - gets a `CoordinateSystemManager.plot` function to visualize the
    graph
  - coordinate systems can be updated using
    `CoordinateSystemManager.add_cs`
  - supports deletion of coordinate systems
  - instances can now be merged and unmerged

- `LocalCoordinateSystem`

  - `LocalCoordinateSystem` now accepts `pandas.TimedeltaIndex` and
    `pint.Quantity` as time inputs when provided with a reference
    `pandas.Timestamp` as `time_ref` \[{pull}`97`\]
  - `LocalCoordinateSystem` now accepts `WXRotation`-objects as
    `orientation` \[{pull}`97`\]
  - Internal structure of `LocalCoordinateSystem` is now based on
    `pandas.TimedeltaIndex` and a reference `pandas.Timestamp` instead
    of `pandas.DatetimeIndex`. As a consequence, providing a reference
    timestamp is now optional. \[{pull}`126`\]

- `util.xr_interp_like` now accepts non-iterable scalar inputs for
  interpolation. \[{pull}`97`\]

- add `pint` compatibility to some `geometry` classes
  (**experimental**)

  - when passing quantities to constructors (and some functions),
    values get converted to default unit `mm` and passed on as
    magnitude.
  - old behavior is preserved.

- add `weldx.utility.xr_check_coords` function to check coordinates
  of xarray object against dtype and value restrictions \[{pull}`125`\]

- add `weldx.utility._sine` to easily create sine TimeSeries \[{pull}`168`\]

- enable `force_ndarray_like=True` as default option when creating
  the global `pint.UnitRegistry` \[{pull}`167`\]

- `util.xr_interp_like` keeps variable and coordinate attributes from
  original DataArray \[{pull}`174`\]

- rework `util.to_pandas_time_index` to accept many different formats
  (LCS, DataArray) \[{pull}`174`\]

- add utility functions for handling time coordinates to "weldx"
  accessor \[{pull}`174`\]

### ASDF extension & schemas

- add `weldx.asdf.types.WxSyntaxError` exception for custom weldx
  ASDF syntax errors \[{pull}`99`\]
- add custom
  `wx_tag`
  validation and update
  `wx_property_tag`
  to
  allow new syntax \[
  {pull}`99`
  \]
  the following syntax can be used:
  ```yaml
  wx_tag: http://stsci.edu/schemas/asdf/core/software-* # allow every version
  wx_tag: http://stsci.edu/schemas/asdf/core/software-1 # fix major version
  wx_tag: http://stsci.edu/schemas/asdf/core/software-1.2 # fix minor version
  wx_tag: http://stsci.edu/schemas/asdf/core/software-1.2.3 # fix patch version
  ```
- add basic schema layout and `GmawProcess` class for arc welding
  process implementation \[{pull}`104`\]
- add example notebook and documentation for arc welding process
  \[{pull}`104`\]
- allow optional properties for validation with `wx_shape` by putting
  the name in brackets like `(optional_prop)` \[{pull}`176`\]

### Fixes

- fix propagating the `name` attribute when reading an ndarray
  `TimeSeries` object back from ASDF files \[{pull}`104`\]
- fix `pint` regression in `TimeSeries` when mixing integer and float
  values \[{pull}`121`\]

## 0.2.0 (30.07.2020)

### ASDF

- add `wx_unit` and `wx_shape` validators

- add `doc/shape-validation.md` documentation for `wx_shape` \[{pull}`75`\]

- add `doc/unit-validation.md` documentation for `wx_unit`

- add unit validation to `iso_groove-1.0.0.yaml`

- fixed const/enum constraints and properties in
  `iso_groove-1.0.0.yaml`

- add NetCDF inspired common types (`Dimension`, `Variable`) with
  corresponding asdf serialization classes

- add asdf serialization classes and schemas for `xarray.DataArray`,
  `xarray.Dataset`, `weldx.transformations.LocalCoordinateSystem` and
  `weldx.transformations.CoordinateSystemManager`.

- add test for `xarray.DataArray`, `xarray.Dataset`,
  `weldx.transformations.LocalCoordinateSystem` and
  `weldx.transformations.CoordinateSystemManager` serialization.

- allow using `pint.Quantity` coordinates in `LocalCoordinateSystem`
  \[{pull}`70`\]

- add measurement related ASDF serialization classes: \[{pull}`70`\]

  - `equipment/generic_equipment-1.0.0`
  - `measurement/data-1.0.0`
  - `data_transformation-1.0.0`
  - `measurement/error-1.0.0`
  - `measurement/measurement-1.0.0`
  - `measurement/measurement_chain-1.0.0`
  - `measurement/signal-1.0.0`
  - `measurement/source-1.0.0`

- add example notebook for measurement chains in tutorials \[{pull}`70`\]

- add support for `sympy` expressions with
  `weldx.core.MathematicalExpression` and ASDF serialization in
  `core/mathematical_expression-1.0.0` \[{pull}`70`\], \[{pull}`76`\]

- add class to describe time series - `weldx.core.TimeSeries` \[{pull}`76`\]

- add `wx_property_tag` validator \[{pull}`72`\]

  the `wx_property_tag` validator restricts **all** properties of an
  object to a single tag. For example the following object can have any
  number of properties but all must be of type
  `tag:weldx.bam.de:weldx/time/timestamp-1.0.0`

  ```yaml
  type: object
  additionalProperties: true # must be true to allow any property
  wx_property_tag: tag:weldx.bam.de:weldx/time/timestamp-1.0.0
  ```

  It can be used as a "named" mapping replacement instead of YAML
  `arrays`.

- add `core/transformation/rotation-1.0.0` schema that implements
  `scipy.spatial.transform.Rotation` and `WXRotation` class to create
  custom tagged `Rotation` instances for custom serialization. \[{pull}`79`\]

- update requirements to `asdf>=2.7` \[{pull}`83`\]

- update `anyOf` to `oneOf` in ASDF schemas \[{pull}`83`\]

- add `__eq__` methods to `LocalCoordinateSystem` and
  `CoordinateSystemManager` \[{pull}`87`\]

## 0.1.0 (05.05.2020)

### ASDF

- add basic file/directory layout for asdf files

  - asdf schemas are located in
    `weldx/asdf/schemas/weldx.bam.de/weldx`
  - tag implementations are in `weldx/asdf/tags/weldx`

- implement support for pint quantities

- implement support for basic pandas time class

- implement base welding classes from AWS/NIST "A Welding Data
  Dictionary"

- add and implement ISO groove types (DIN EN ISO 9692-1:2013)

- add basic jinja templates and functions for adding simple dataclass
  objects

- setup package to include and install ASDF extensions and schemas (see
  setup.py, MANIFEST.in)

- add basic tests for writing/reading all ASDF classes (these only run
  code without any real checks!)

### module:

- add setup.py package configuration for install

  - required packages
  - package metadata
  - asdf extension entry points
  - version support

- update pandas, scipy, xarray and pint minimum versions (in conda env
  and setup.py)

- add versioneer

- update options in setup.cfg

- update tool configurations
