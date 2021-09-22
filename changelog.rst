Release Notes
=============

0.5.0 (unreleased)
------------------

added
~~~~~

-  added “units” (exact) and “dimensionality” (dimensionality
   compatible) checking options to `util.xr_check_coords`
   `[#442] <https://github.com/BAMWelDX/weldx/pull/442>`__
-  `Time` class that can be initialized from several other time types
   and provides time related utility functions
   `[#433] <https://github.com/BAMWelDX/weldx/pull/433>`__
-  `TimeSeries` now supports setting a `reference_time` absolute
   time values for interpolation
   `[#440] <https://github.com/BAMWelDX/weldx/pull/440>`__
-  `LocalCoordinateSystem.from_axis_vectors` and
   `CoordinateSystemManager.create_cs_from_axis_vectors`
   `[#472] <https://github.com/BAMWelDX/weldx/pulls/472>`__
-  added PyTest flags to use `WeldxFile` internally in
   `asdf.util.read_buffer` and `asdf.util.write_buffer`
   `[#469] <https://github.com/BAMWelDX/weldx/pull/469>`__.
-  added classes and functions at the top-level of the package to the
   documentation
   `[#437] <https://github.com/BAMWelDX/weldx/pulls/437>`__.
-  added `weldx.asdf.util.get_highest_tag_version` utility function
   `[#523] <https://github.com/BAMWelDX/weldx/pull/523>`__.

removed
~~~~~~~

-  removed functions now covered by `Time`:
   `pandas_time_delta_to_quantity`, `to_pandas_time_index` ,
   `get_time_union`
   `[#448] <https://github.com/BAMWelDX/weldx/pull/448>`__
-  removed custom `wx_tag` validator
   `[#461] <https://github.com/BAMWelDX/weldx/pull/461>`__
-  attrdict dependency replaced with a custom implementation of
   recursive dicts
   `[#470] <https://github.com/BAMWelDX/weldx/pulls/470>`__.
-  `from_xyz`, `from_xy_and_orientation`,
   `from_yz_and_orientation` and `from_xz_and_orientation` from
   `LocalCoordinateSystem`. Use `from_axis_vectors` instead.
   `[#472] <https://github.com/BAMWelDX/weldx/pulls/472>`__
-  `create_cs_from_xyz`, `create_cs_from_xy_and_orientation`,
   `create_cs_from_yz_and_orientation` and
   `create_cs_from_xz_and_orientation` from
   `CoordinateSystemManager`. Use `create_cs_from_axis_vectors`
   instead. `[#472] <https://github.com/BAMWelDX/weldx/pulls/472>`__
-  `is_column_in_matrix`, `is_row_in_matrix`, `to_float_array`,
   `to_list`, `matrix_is_close`, `vector_is_close` and
   `triangulate_geometry` from `weldx.util`
   `[#490] <https://github.com/BAMWelDX/weldx/pull/490>`__
-  remove the `:` syntax from `wx_shape` validation
   `[#537] <https://github.com/BAMWelDX/weldx/pull/537>`__

changes
~~~~~~~

-  move `sine` utility function to `weldx.welding.util`
   `[#439] <https://github.com/BAMWelDX/weldx/pull/439>`__
-  `LocalCoordinateSystem` and `CoordinateSystemManager` function
   parameters related to time now support all types that are also
   supported by the new `Time` class
   `[#448] <https://github.com/BAMWelDX/weldx/pull/448>`__
-  `LocalCoordinateSystem.interp_time` returns static systems if only
   a single time value is passed or if there is no overlap between the
   interpolation time range and the coordinate systems time range. This
   also affects the results of some `CoordinateSystemManager` methods
   (`get_cs` , `interp_time`)
   `[#476] <https://github.com/BAMWelDX/weldx/pull/476>`__
-  `WeldxAccessor.time_ref` setter now raises a `TypeError` if
   `None` is passed to it
   `[#489] <https://github.com/BAMWelDX/weldx/pull/489>`__
-  move xarray related utility functions into `weldx.util.xarray` and
   all other ones into `weldx.util.util`. Content from both submodules
   can still be accessed using `weldx.util`
   `[#490] <https://github.com/BAMWelDX/weldx/pull/490>`__
-  xarray implementations for the `LocalCoordinateSystem` now operate
   on time as a dimension instead of coordinates
   `[#486] <https://github.com/BAMWelDX/weldx/pull/486>`__
-  `WeldxFile.copy` now creates a copy to a (optional) file. Before it
   just returned a dictionary
   `[#504] <https://github.com/BAMWelDX/weldx/pull/504>`__.
-  changed the default `pint.Unit` formatting to short notation `:~`
   `[#519] <https://github.com/BAMWelDX/weldx/pull/519>`__.

fixes
~~~~~

-  `WeldxFile.show_asdf_header` prints output on console, before it
   only returned the header as parsed dict and string representation.
   Also tweaked efficiency by not writing binary blocks
   `[#459] <https://github.com/BAMWelDX/weldx/pull/459>`__,
   `[#469] <https://github.com/BAMWelDX/weldx/pull/469>`__.
-  Merging and unmerging multiple `CoordinateSystemManager` instances
   now correctly preserves all attached data.
   `[#494] <https://github.com/BAMWelDX/weldx/pull/494>`__.
-  `compare_nested` can compare sets
   `[#496] <https://github.com/BAMWelDX/weldx/pull/496>`__

documentation
~~~~~~~~~~~~~

-  added installation guide with complete environment setup (Jupyterlab
   with extensions) and possible problems and solutions
   `[#450] <https://github.com/BAMWelDX/weldx/pull/450>`__
-  split API documentation into user classes/functions and a full API
   reference `[#469] <https://github.com/BAMWelDX/weldx/pull/469>`__.

ASDF
~~~~

-  all schema version numbers set to `0.1.0`
   `[#535] <https://github.com/BAMWelDX/weldx/pull/535>`__.
-  add `time/time` schema to support `Time` class
   `[#463] <https://github.com/BAMWelDX/weldx/pull/463>`__.
-  rework ASDF extension to new asdf 2.8 API
   `[#467] <https://github.com/BAMWelDX/weldx/pull/467>`__

   -  move schema files to `weldx/schemas`
   -  create extension manifest in `weldx/manifests`. The manifest
      also contains tag mappings for legacy tag names for backwards
      compatibility.
   -  move tag module to `weldx/tags`
   -  refactor all asdf uris to new `asdf://` naming convention, see
      https://asdf.readthedocs.io/en/latest/asdf/extending/uris.html#entities-identified-by-uri
   -  replaced all referenced weldx tag versions in schemas with
      `0.1.*`
   -  refactor
      `asdf://weldx.bam.de/weldx/schemas/datamodels/single_pass_weld-1.0.0.schema`
      to
      `asdf://weldx.bam.de/weldx/schemas/datamodels/single_pass_weld-0.1.0`
      and enable schema test
   -  add legacy class for validators support in
      `weldx.asdf._extension.py`
   -  asdf utility functions `weldx.asdf.util.uri_match`,
      `weldx.asdf.util.get_converter_for_tag` and
      `weldx.asdf.util.get_weldx_extension`
   -  add `devtools/scripts/update_manifest.py` to auto update
      manifest from extension metadata
   -  custom shape validation must now be implemented via staticmethod
      `WeldxConverter.shape_from_tagged`

-  provide legacy schema support in
   `weldx/schemas/weldx.bam.de/legacy`
   `[#533] <https://github.com/BAMWelDX/weldx/pull/533>`__
-  rewrote
   `asdf://weldx.bam.de/weldx/schemas/core/transformations/coordinate_system_hierarchy`
   schema for the `CoordinateSystemManager`. It uses the digraph
   schemas to serialize the coordinate system structure.
   `[#497] <https://github.com/BAMWelDX/weldx/pull/497>`__
-  add `asdf://weldx.bam.de/weldx/schemas/unit/quantity` and
   `asdf://weldx.bam.de/weldx/schemas/unit/unit` schemas
   `[#522] <https://github.com/BAMWelDX/weldx/pull/522>`__

deprecations
~~~~~~~~~~~~

dependencies
~~~~~~~~~~~~

0.4.1 (20.07.2021)
------------------

.. _added-1:

added
~~~~~

-  `closed_mesh` parameter to `Geometry.spatial_data` and
   `SpatialData.from_geometry_raster`
   `[#414] <https://github.com/BAMWelDX/weldx/pull/414>`__
-  `TimeSeries.plot` and `Signal.plot`
   `[#420] <https://github.com/BAMWelDX/weldx/pull/420>`__
-  abstract base class `TimeDependent`
   `[#460] <https://github.com/BAMWelDX/weldx/pull/460>`__

.. _changes-1:

changes
~~~~~~~

-  `TimeSeries.__init__` accepts `xarray.DataArray` as `data`
   parameter `[#429] <https://github.com/BAMWelDX/weldx/pull/429>`__
-  The `LocalCoordinateSystem.time` and `TimeSeries.time` now return
   an instance of `Time`
   `[#464] <https://github.com/BAMWelDX/weldx/pull/464>`__
-  Fix wrong and incomplete type-hints
   `[#435] <https://github.com/BAMWelDX/weldx/pull/435>`__

.. _asdf-1:

ASDF
~~~~

-  sort `List[str]` before serialization of most `weldx` classes to
   avoid random reordering in the same file and enforce consistency.
   `[#430] <https://github.com/BAMWelDX/weldx/pull/430>`__

.. _deprecations-1:

deprecations
~~~~~~~~~~~~

-  `lcs_coords_from_ts` will be removed in version 0.5.0
   `[#426] <https://github.com/BAMWelDX/weldx/pull/426>`__

.. _section-1:

0.4.0 (13.07.2021)
------------------

Release `0.4.0` brings many new major features to `weldx`

Highlights
~~~~~~~~~~

-  `Quality
   Standards <https://weldx.readthedocs.io/en/latest/tutorials/quality_standards.html>`__:
   Users can now create and integrate their own quality standards by
   defining new ASDF schema definitions and loading them into weldx. It
   is possible to add new definitions or modify existing schemas to
   create your own flavour of the weldx standard.
-  `WeldxFile <https://weldx.readthedocs.io/en/latest/tutorials/weldxfile.html>`__:
   Create/Load/Modify asdf files directly using `WeldxFile` with many
   helpful utility functions included.
-  `TimeSeries
   support <https://weldx.readthedocs.io/en/latest/tutorials/welding_example_02_weaving.html#add-a-sine-wave-to-the-TCP-movement>`__
   for `LocalCoordinateSystem`: It is now possible to define a
   time-dependent `LocalCoordinateSystem` with a simple function by
   passing a `TimeSeries` object with a `MathematicalExpression` as
   `coordinates`. For an example, click the link above.
-  `MeasurementChain <https://weldx.readthedocs.io/en/latest/tutorials/measurement_chain.html>`__
   The `MeasurementChain` has been reworked to be easier and more
   flexible to use.

full changelog below:

.. _added-2:

added
~~~~~

-  add support for quality standards. Further information can be found
   in the corresponding new tutorial.
   `[#211] <https://github.com/BAMWelDX/weldx/pull/211>`__
-  added `asdf.util.get_schema_path` helper function
   `[#325] <https://github.com/BAMWelDX/weldx/pull/325>`__
-  added `util.compare_nested` to check equality of two nested data
   structures. `[#328] <https://github.com/BAMWelDX/weldx/pull/328>`__
-  added `WeldxFile` wrapper to handle asdf files with history and
   schemas more easily.
   `[#341] <https://github.com/BAMWelDX/weldx/pull/341>`__.
-  add `"step"` as additional method to `util.xr_interp_like`
   `[#363] <https://github.com/BAMWelDX/weldx/pull/363>`__
-  add `util.compare_nested_eq` decorator for dataclasses with
   array-like fields
   `[#378] <https://github.com/BAMWelDX/weldx/pull/378>`__
-  adds a `dataclass_serialization_class` utility function that
   automatically generates the asdf serialization class for python
   dataclasses. `[#380] <https://github.com/BAMWelDX/weldx/pull/380>`__
-  Added method to set the interpolation method to the `TimeSeries`
   `[#353] <https://github.com/BAMWelDX/weldx/pull/353>`__
-  Add `is_discrete` and `is_expression` properties to
   `TimeSeries`
   `[#366] <https://github.com/BAMWelDX/weldx/pull/366>`__
-  Add `MeasurementChain.output_signal` property that returns the
   output signal of the `MeasurementChain`
   `[#394] <https://github.com/BAMWelDX/weldx/pull/394>`__

.. _changes-2:

changes
~~~~~~~

-  `WXRotation.from_euler()` now accepts a `pint.Quantity` as input.
   `[#318] <https://github.com/BAMWelDX/weldx/pull/318>`__
-  move tests folder to `weldx/tests`
   `[#323] <https://github.com/BAMWelDX/weldx/pull/323>`__
-  `get_yaml_header` received a new option parse, which optionally
   returns the parsed YAML header as `asdf.tagged.TaggedDict`.
   `[#338] <https://github.com/BAMWelDX/weldx/pull/338>`__
-  refactor `asdf_json_repr` into `view_tree`
   `[#339] <https://github.com/BAMWelDX/weldx/pull/339>`__
-  `TimeSeries.interp_time`
   `[#353] <https://github.com/BAMWelDX/weldx/pull/353>`__

   -  now returns a new `TimeSeries` instead of a `xarray.DataArray`
   -  if the data has already been interpolated before, a warning is
      emitted
   -  `TimeSeries` supports now all interpolation methods supported by
      xarray

-  The `MeasurementChain` is now internally based on a
   `networkx.DiGraph`. New functions are also added to the class to
   simplify its usage.
   `[#326] <https://github.com/BAMWelDX/weldx/pull/326>`__ The following
   additional changes were applied during the update of the
   `MeasurementChain`:

   -  renamed `DataTransformation` class to `SignalTransformation`
   -  renamed `Source` to `SignalSource`
   -  Added additional functionality to `Signal`,
      `SignalTransformation` and `GenericEquipment`
   -  Removed `Data` class
   -  Updated asdf schemas of all modified classes and the ones that
      contained references to those classes

-  allow input of string quantities in `MathematicalExpression`
   parameters and a few other places
   `[#402] <https://github.com/BAMWelDX/weldx/pull/402>`__
   `[#416] <https://github.com/BAMWelDX/weldx/pull/416>`__
-  `LocalCoordinateSystem.__init__` now accepts a `TimeSeries` as
   input. All methods of the `CoordinateSystemManager` also support
   this new behavior
   `[#366] <https://github.com/BAMWelDX/weldx/pull/366>`__
-  During the creation of a `WeldxFile` the path of a passed custom
   schema is resolved automatically
   `[#412] <https://github.com/BAMWelDX/weldx/pull/412>`__.

.. _documentation-1:

documentation
~~~~~~~~~~~~~

-  Add new tutorial about the `MeasurementChain`
   `[#326] <https://github.com/BAMWelDX/weldx/pull/326>`__
-  Updated the measurement tutorial
   `[#326] <https://github.com/BAMWelDX/weldx/pull/326>`__

.. _asdf-2:

ASDF
~~~~

-  fix inline array serialization for new 64bit inline limit
   `[#218] <https://github.com/BAMWelDX/weldx/pull/218>`__
-  add `yaml_tag_handles` to `WeldxExtension`
   `[#218] <https://github.com/BAMWelDX/weldx/pull/218>`__
-  add `uuid-1.0.0.yaml` schema as basic version 4 UUID implementation
   `[#330] <https://github.com/BAMWelDX/weldx/pull/330>`__
-  add `core/graph/di_node`, `core/graph/di_edge` &
   `core/graph/di_graph` for implementing a generic
   `networkx.DiGraph`
   `[#330] <https://github.com/BAMWelDX/weldx/pull/330>`__
-  compatibility with ASDF-2.8
   `[#355] <https://github.com/BAMWelDX/weldx/pull/355>`__
-  data attached to an instance of the `CoordinateSystemManger` is now
   also stored in a WelDX file
   `[#364] <https://github.com/BAMWelDX/weldx/pull/339>`__
-  replace references to base asdf tags with `-1.*` version wildcard
   `[#373] <https://github.com/BAMWelDX/weldx/pull/373>`__
-  update `single-pass-weldx.1.0.0.schema` to allow groove types by
   wildcard `[#373] <https://github.com/BAMWelDX/weldx/pull/373>`__
-  fix attributes serialization of DataSet children
   `[#384] <https://github.com/BAMWelDX/weldx/pull/384>`__.
-  update `wx_shape` syntax in `local_coordinate_system-1.0.0`
   `[#366] <https://github.com/BAMWelDX/weldx/pull/366>`__
-  add custom `wx_shape` validation to `variable-1.0.0`
   `[#366] <https://github.com/BAMWelDX/weldx/pull/366>`__
-  remove outdated `TimeSeries` shape validation code
   `[#399] <https://github.com/BAMWelDX/weldx/pull/399>`__
-  use asdf tag validation pattern for `wx_property_tag`
   `[#410] <https://github.com/BAMWelDX/weldx/pull/410>`__
-  update `MathematicalExpression` schema
   `[#410] <https://github.com/BAMWelDX/weldx/pull/410>`__

.. _fixes-1:

fixes
~~~~~

-  added check for symmetric key difference for mappings with
   `util.compare_nested`
   `[#377] <https://github.com/BAMWelDX/weldx/pull/377>`__

.. _deprecations-2:

deprecations
~~~~~~~~~~~~

-  deprecate `wx_tag` validator (use default asdf uri pattern
   matching) `[#410] <https://github.com/BAMWelDX/weldx/pull/410>`__

.. _section-2:

0.3.3 (30.03.2021)
------------------

This is a bugfix release to correctly include the asdf schema files in
conda builds. `[#314] <https://github.com/BAMWelDX/weldx/pull/314>`__

.. _asdf-3:

ASDF
~~~~

-  fix required welding wire metadata in
   `single-pass-weldx.1.0.0.schema`
   `[#316] <https://github.com/BAMWelDX/weldx/pull/316>`__

.. _section-3:

0.3.2 (29.03.2021)
------------------

.. _added-3:

added
~~~~~

-  `weldx.util.deprecated` decorator
   `[#295] <https://github.com/BAMWelDX/weldx/pull/295>`__

.. _removed-1:

removed
~~~~~~~

-  `rotation_matrix_x`, `rotation_matrix_y` and
   `rotation_matrix_z`
   `[#317] <https://github.com/BAMWelDX/weldx/pull/317>`__

.. _dependencies-1:

dependencies
~~~~~~~~~~~~

-  restrict `scipy!=1.6.0,scipy!=1.6.1`
   `[#300] <https://github.com/BAMWelDX/weldx/pull/300>`__

.. _asdf-4:

ASDF
~~~~

-  add validators to `rotation-1.0.0.yaml` &
   `gas_component-1.0.0.yaml`
   `[#303] <https://github.com/BAMWelDX/weldx/pull/303>`__
-  update descriptions in `single-pass-weldx.1.0.0.schema`
   `[#308] <https://github.com/BAMWelDX/weldx/pull/308>`__

.. _fixes-2:

fixes
~~~~~

-  prevent creation of `IsoBaseGroove` with negative parameters
   `[#306] <https://github.com/BAMWelDX/weldx/pull/306>`__

.. _section-4:

0.3.1 (21.03.2021)
------------------

.. _added-4:

added
~~~~~

-  plot function for `MeasurementChain`
   `[#288] <https://github.com/BAMWelDX/weldx/pull/288>`__

.. _asdf-5:

ASDF
~~~~

-  remove the `additionalProperties` restriction from
   `single_pass_weld-1.0.0.schema.yaml`
   `[#283] <https://github.com/BAMWelDX/weldx/pull/283>`__
-  allow scalar `integer` value in `anyOf` of
   `time_series-1.0.0.yaml` to fix
   `#282 <https://github.com/BAMWelDX/weldx/pull/282>`__
   `[#286] <https://github.com/BAMWelDX/weldx/pull/286>`__
-  add examples to schema files
   `[#274] <https://github.com/BAMWelDX/weldx/pull/274>`__

.. _changes-3:

changes
~~~~~~~

-  `plot_graph` of the CSM now renders static and time-dependent edges
   differently `[#291] <https://github.com/BAMWelDX/weldx/pull/291>`__
-  use `pint` compatible array syntax in
   `IsoBaseGroove.to_profile()` methods
   `[#189] <https://github.com/BAMWelDX/weldx/pull/189>`__
-  CSM and LCS plot function get a `scale_vectors` parameter. It
   scales the plotted coordinate system vectors when using matplotlib as
   backend `[#293] <https://github.com/BAMWelDX/weldx/pull/293>`__

.. _fixes-3:

fixes
~~~~~

-  A warning is now emitted if a `LocalCoordinateSystem` drops a
   provided time during construction. This usually happens if the
   coordinates and orientation only contain a single data point.
   `[#285] <https://github.com/BAMWelDX/weldx/pull/285>`__

.. _section-5:

0.3.0 (12.03.2021)
------------------

.. _added-5:

added
~~~~~

-  add `weldx.transformations.CoordinateSystemManager.relabel`
   function `[#219] <https://github.com/BAMWelDX/weldx/pull/219>`__
-  add `SpatialDate` class for storing 3D point data with optional
   triangulation `[#234] <https://github.com/BAMWelDX/weldx/pull/234>`__
-  add `plot` function to
   `SpatialData`\ `[#251] <https://github.com/BAMWelDX/weldx/pull/251>`__
-  add `plot` function to visualize `LocalCoordinateSystem` and
   `CoordinateSystemManager` instances in 3d space
   `[#231] <https://github.com/BAMWelDX/weldx/pull/231>`__
-  add `weldx.welding.groove.iso_9692_1.IsoBaseGroove.cross_sect_area`
   property to compute cross sectional area between the workpieces
   `[#248] <https://github.com/BAMWelDX/weldx/pull/248>`__.
-  add `weldx.welding.util.compute_welding_speed` function
   `[#248] <https://github.com/BAMWelDX/weldx/pull/248>`__.

.. _asdf-6:

ASDF
~~~~

-  Add possibility to store meta data and content of an external file in
   an ASDF file `[#215] <https://github.com/BAMWelDX/weldx/pull/215>`__

   -  Python class: `weldx.asdf.ExternalFile`
   -  Schema: `core/file-1.0.0.yaml`

-  Added support for serializing generic metadata and userdata
   attributes for weldx classes.
   `[#209] <https://github.com/BAMWelDX/weldx/pull/209>`__

   -  the provisional attribute names are `wx_metadata` and
      `wx_user`

-  `None` values are removed from the asdf tree for all `weldx`
   classes. `[#212] <https://github.com/BAMWelDX/weldx/pull/212>`__
-  add `datamodels` directory and example
   `http://weldx.bam.de/schemas/weldx/datamodels/single_pass_weld-1.0.0.schema`
   schema `[#190] <https://github.com/BAMWelDX/weldx/pull/190>`__

   -  schemas in the `datamodels` directory do not define any tags and
      can be referenced in other schemas and as `custom_schema` when
      reading/writing `ASDF`-files
   -  the `single_pass_weld-1.0.0.schema` is an example schema for a
      simple, linear, single pass GMAW application
   -  add `core/geometry/point_cloud-1.0.0.yaml` schema
      `[#234] <https://github.com/BAMWelDX/weldx/pull/234>`__

-  add file schema describing a simple linear welding application
   `datamodels/single_pass_weld-1.0.0.schema`
   `[#256] <https://github.com/BAMWelDX/weldx/pull/256>`__

.. _documentation-2:

documentation
~~~~~~~~~~~~~

-  Simplify tutorial code and enhance plots by using newly implemented
   plot functions
   `[#231] <https://github.com/BAMWelDX/weldx/pull/231>`__
   `[#251] <https://github.com/BAMWelDX/weldx/pull/251>`__
-  add AWS shielding gas descriptions to documentation
   `[#270] <https://github.com/BAMWelDX/weldx/pull/270>`__

.. _changes-4:

changes
~~~~~~~

-  pass variable names as tuple to `sympy.lambdify` in
   `MathematicalExpression` to prevent sympy deprecation
   `[#214] <https://github.com/BAMWelDX/weldx/pull/214>`__
-  set `conda-forge` as primary channel in `environment.yaml` and
   `build_env.yaml`
   `[#214] <https://github.com/BAMWelDX/weldx/pull/214>`__
-  set minimum Python version to 3.7
   `[#220] <https://github.com/BAMWelDX/weldx/pull/220>`__
-  `geometry.Profile.rasterize` can return list of rasterized shapes
   instead of flat ndarray (with setting `stack=False`)
   `[#223] <https://github.com/BAMWelDX/weldx/pull/223>`__
-  `geometry.Profile.plot` plots individual line objects for each
   shape (instead of a single line object)
   `[#223] <https://github.com/BAMWelDX/weldx/pull/223>`__
-  remove jinja templates and related code
   `[#228] <https://github.com/BAMWelDX/weldx/pull/228>`__
-  add `stack` option to most `geometry` classes for rasterization
   `[#234] <https://github.com/BAMWelDX/weldx/pull/234>`__
-  The graph of a `CoordinateSystemManager` is now plotted with
   `plot_graph` instead of `plot`.
   `[#231] <https://github.com/BAMWelDX/weldx/pull/231>`__
-  add custom `wx_shape` validation for `TimeSeries` and
   `Quantity` `[#256] <https://github.com/BAMWelDX/weldx/pull/256>`__
-  refactor the `transformations` and `visualization` module into
   smaller files `[#247] <https://github.com/BAMWelDX/weldx/pull/247>`__
-  refactor `weldx.utility` into `weldx.util`
   `[#247] <https://github.com/BAMWelDX/weldx/pull/247>`__
-  refactor `weldx.asdf.utils` into `weldx.asdf.util`
   `[#247] <https://github.com/BAMWelDX/weldx/pull/247>`__
-  it is now allowed to merge a time-dependent `timedelta` subsystem
   into another `CSM` instance if the parent instance has set an
   explicit reference time
   `[#268] <https://github.com/BAMWelDX/weldx/pull/268>`__

.. _fixes-4:

fixes
~~~~~

-  don’t inline time dependent `LCS.coordinates`
   `[#222] <https://github.com/BAMWelDX/weldx/pull/222>`__
-  fix “datetime64” passing for “timedelta64” in `xr_check_coords`
   `[#221] <https://github.com/BAMWelDX/weldx/pull/221>`__
-  fix `time_ref_restore` not working correctly if no `time_ref` was
   set `[#221] <https://github.com/BAMWelDX/weldx/pull/221>`__
-  fix deprecated signature in `WXRotation`
   `[#224] <https://github.com/BAMWelDX/weldx/pull/224>`__
-  fix a bug with singleton dimensions in xarray interpolation/matmul
   `[#243] <https://github.com/BAMWelDX/weldx/pull/243>`__
-  update some documentation formatting and links
   `[#247] <https://github.com/BAMWelDX/weldx/pull/247>`__
-  fix `wx_shape` validation for scalar `Quantity` and
   `TimeSeries` objects
   `[#256] <https://github.com/BAMWelDX/weldx/pull/256>`__
-  fix a case where `CSM.time_union()` would return with mixed
   `DateTimeIndex` and `TimeDeltaIndex` types
   `[#268] <https://github.com/BAMWelDX/weldx/pull/268>`__

.. _dependencies-2:

dependencies
~~~~~~~~~~~~

-  Add
   `PyFilesystem <https://docs.pyfilesystem.org/en/latest/>`__\ (`fs`)
   as new dependency
-  Add `k3d <https://github.com/K3D-tools/K3D-jupyter>`__ as new
   dependency
-  restrict `scipy<1.6` pending `ASDF
   #916 <https://github.com/asdf-format/asdf/issues/916>`__
   `[#224] <https://github.com/BAMWelDX/weldx/pull/224>`__
-  set minimum Python version to 3.8
   `[#229] <https://github.com/BAMWelDX/weldx/pull/229>`__\ `[#255] <https://github.com/BAMWelDX/weldx/pull/255>`__
-  only import some packages upon first use
   `[#247] <https://github.com/BAMWelDX/weldx/pull/247>`__
-  Add `meshio <https://pypi.org/project/meshio/>`__ as new dependency
   `#265 <https://github.com/BAMWelDX/weldx/pull/265>`__

.. _section-6:

0.2.2 (30.11.2020)
------------------

.. _added-6:

added
~~~~~

-  Added `weldx.utility.ureg_check_class` class decorator to enable
   `pint` dimensionality checks with `@dataclass` .
   `[#179] <https://github.com/BAMWelDX/weldx/pull/179>`__
-  Made coordinates and orientations optional for LCS schema. Missing
   values are interpreted as unity translation/rotation. An empty LCS
   object represents a unity transformation step.
   `[#177] <https://github.com/BAMWelDX/weldx/pull/177>`__
-  added `weldx.utility.lcs_coords_from_ts` function
   `[#199] <https://github.com/BAMWelDX/weldx/pull/199>`__
-  add a tutorial with advanced use case for combining groove
   interpolation with different TCP movements and distance calculations
   `[#199] <https://github.com/BAMWelDX/weldx/pull/199>`__

.. _changes-5:

changes
~~~~~~~

-  refactor welding groove classes
   `[#181] <https://github.com/BAMWelDX/weldx/pull/181>`__

   -  refactor groove codebase to make use of subclasses and classnames
      for more generic functions
   -  add `_meta` attribute to subclasses that map class attributes
      (dataclass parameters) to common names
   -  rework `get_groove` to make use of new class layout and parse
      function arguments

-  create `weldx.welding` module (contains GMAW processes and groove
   definitions) `[#181] <https://github.com/BAMWelDX/weldx/pull/181>`__
-  move `GmawProcessTypeAsdf` to `asdf.tags` folder
   `[#181] <https://github.com/BAMWelDX/weldx/pull/181>`__
-  reorder module imports in `weldx.__init__`
   `[#181] <https://github.com/BAMWelDX/weldx/pull/181>`__
-  support timedelta dtypes in ASDF `data_array/variable`
   `[#191] <https://github.com/BAMWelDX/weldx/pull/191>`__
-  add `set_axes_equal` option to some geometry plot functions (now
   defaults to `False`)
   `[#199] <https://github.com/BAMWelDX/weldx/pull/199>`__
-  make `utility.sine` public function
   `[#199] <https://github.com/BAMWelDX/weldx/pull/199>`__
-  switch to setuptools_scm versioning and move package metadata to
   setup.cfg `[#206] <https://github.com/BAMWelDX/weldx/pull/206>`__

.. _asdf-7:

ASDF
~~~~

-  refactor ISO 9692-1 groove schema definitions and classes
   `[#181] <https://github.com/BAMWelDX/weldx/pull/181>`__

   -  move base schema definitions in file `terms-1.0.0.yaml` to
      `weldx/groove`
   -  split old schema into multiple files (1 per groove type) and
      create folder `iso_9692_1_2013_12`

.. _section-7:

0.2.1 (26.10.2020)
------------------

.. _changes-6:

changes
~~~~~~~

-  Documentation

   -  Documentation is `published on
      readthedocs <https://weldx.readthedocs.io/en/latest/>`__
   -  API documentation is now available
   -  New tutorial about 3 dimensional geometries
      `[#105] <https://github.com/BAMWelDX/weldx/pull/105>`__

-  `CoordinateSystemManager`

   -  supports multiple time formats and can get a reference time
      `[#162] <https://github.com/BAMWelDX/weldx/pull/162>`__
   -  each instance can be named
   -  gets a `plot` function to visualize the graph
   -  coordinate systems can be updated using `add_cs`
   -  supports deletion of coordinate systems
   -  instances can now be merged and unmerged

-  `LocalCoordinateSystem`

   -  `LocalCoordinateSystem` now accepts `pd.TimedeltaIndex` and
      `pint.Quantity` as `time` inputs when provided with a
      reference `pd.Timestamp` as `time_ref`
      `[#97] <https://github.com/BAMWelDX/weldx/pull/97>`__
   -  `LocalCoordinateSystem` now accepts `Rotation`-Objects as
      `orientation`
      `[#97] <https://github.com/BAMWelDX/weldx/pull/97>`__
   -  Internal structure of `LocalCoordinateSystem` is now based on
      `pd.TimedeltaIndex` and a reference `pd.Timestamp` instead of
      `pd.DatetimeIndex`. As a consequence, providing a reference
      timestamp is now optional.
      `[#126] <https://github.com/BAMWelDX/weldx/pull/126>`__

-  `weldx.utility.xr_interp_like` now accepts non-iterable scalar
   inputs for interpolation
   `[#97] <https://github.com/BAMWelDX/weldx/pull/97>`__
-  add `pint` compatibility to some `geometry` classes
   (**experimental**)

   -  when passing quantities to constructors (and some functions),
      values get converted to default unit `mm` and passed on as
      magnitude
   -  old behavior is preserved

-  add `weldx.utility.xr_check_coords` function to check coordinates
   of xarray object against dtype and value restrictions
   `[#125] <https://github.com/BAMWelDX/weldx/pull/125>`__
-  add `weldx.utility._sine` to easily create sine TimeSeries
   `[#168] <https://github.com/BAMWelDX/weldx/pull/168>`__
-  enable `force_ndarray_like=True` as default option when creating
   the global `pint.UnitRegistry`
   `[#167] <https://github.com/BAMWelDX/weldx/pull/167>`__
-  `ut.xr_interp_like` keeps variable and coordinate attributes from
   original DataArray
   `[#174] <https://github.com/BAMWelDX/weldx/pull/174>`__
-  rework `ut.to_pandas_time_index` to accept many different formats
   (LCS, DataArray)
   `[#174] <https://github.com/BAMWelDX/weldx/pull/174>`__
-  add utility functions for handling time coordinates to “weldx”
   accessor `[#174] <https://github.com/BAMWelDX/weldx/pull/174>`__

ASDF extension & schemas
~~~~~~~~~~~~~~~~~~~~~~~~

-  add `WxSyntaxError` exception for custom weldx ASDF syntax errors
   `[#99] <https://github.com/BAMWelDX/weldx/pull/99>`__

-  | add custom `wx_tag` validation and update `wx_property_tag` to
     allow new syntax
     `[#99] <https://github.com/BAMWelDX/weldx/pull/99>`__
   | the following syntax can be used:

   .. code:: yaml

      wx_tag: http://stsci.edu/schemas/asdf/core/software-* # allow every version
      wx_tag: http://stsci.edu/schemas/asdf/core/software-1 # fix major version
      wx_tag: http://stsci.edu/schemas/asdf/core/software-1.2 # fix minor version
      wx_tag: http://stsci.edu/schemas/asdf/core/software-1.2.3 # fix patchversion

-  add basic schema layout and `GmawProcess` class for arc welding
   process implementation
   `[#104] <https://github.com/BAMWelDX/weldx/pull/104>`__

-  add example notebook and documentation for arc welding process
   `[#104] <https://github.com/BAMWelDX/weldx/pull/104>`__

-  allow optional properties for validation with `wx_shape` by putting
   the name in brackets like
   `(optional_prop)`\ `[#176] <https://github.com/BAMWelDX/weldx/pull/176>`__

.. _fixes-5:

fixes
~~~~~

-  fix propagating the `name` attribute when reading an ndarray
   `TimeSeries` object back from ASDF files
   `[#104] <https://github.com/BAMWelDX/weldx/pull/104>`__
-  fix `pint` regression in `TimeSeries` when mixing integer and
   float values `[#121] <https://github.com/BAMWelDX/weldx/pull/121>`__

.. _section-8:

0.2.0 (30.07.2020)
------------------

.. _asdf-8:

ASDF
~~~~

-  add `wx_unit` and `wx_shape` validators

-  add `doc/shape-validation.md` documentation for `wx_shape`
   `[#75] <https://github.com/BAMWelDX/weldx/pull/75>`__

-  add `doc/unit-validation.md` documentation for `wx_unit`

-  add unit validation to `iso_groove-1.0.0.yaml`

-  fixed const/enum constraints and properties in
   `iso_groove-1.0.0.yaml`

-  add NetCDF inspired common types (`Dimension`,\ `Variable`) with
   corresponding asdf serialization classes

-  add asdf serialization classes and schemas for `xarray.DataArray`,
   `xarray.Dataset`, `weldx.transformations.LocalCoordinateSystem`
   and `weldx.transformations.CoordinateSystemManager`.

-  add test for `xarray.DataArray`, `xarray.Dataset`,
   `weldx.transformations.LocalCoordinateSystem` and
   `weldx.transformations.CoordinateSystemManager` serialization.

-  allow using `pint.Quantity` coordinates in
   `weldx.transformations.LocalCoordinateSystem`
   `[#70] <https://github.com/BAMWelDX/weldx/pull/70>`__

-  add measurement related ASDF serialization classes:
   `[#70] <https://github.com/BAMWelDX/weldx/pull/70>`__

   -  `equipment/generic_equipment-1.0.0`
   -  `measurement/data-1.0.0`
   -  `data_transformation-1.0.0`
   -  `measurement/error-1.0.0`
   -  `measurement/measurement-1.0.0`
   -  `measurement/measurement_chain-1.0.0`
   -  `measurement/signal-1.0.0`
   -  `measurement/source-1.0.0`

-  add example notebook for measurement chains in tutorials
   `[#70] <https://github.com/BAMWelDX/weldx/pull/70>`__

-  add support for `sympy` expressions with
   `weldx.core.MathematicalExpression` and ASDF serialization in
   `core/mathematical_expression-1.0.0`
   `[#70] <https://github.com/BAMWelDX/weldx/pull/70>`__ ,
   `[#76] <https://github.com/BAMWelDX/weldx/pull/76>`__

-  add class to describe time series - `weldx.core.TimeSeries`
   `[#76] <https://github.com/BAMWelDX/weldx/pull/76>`__

-  add `wx_property_tag` validator
   `[#72] <https://github.com/BAMWelDX/weldx/pull/72>`__

   the `wx_property_tag` validator restricts **all** properties of an
   object to a single tag. For example the following object can have any
   number of properties but all must be of type
   `tag:weldx.bam.de:weldx/time/timestamp-1.0.0`
   `yaml   type: object   additionalProperties: true # must be true to allow any property   wx_property_tag: "tag:weldx.bam.de:weldx/time/timestamp-1.0.0"`
   It can be used as a “named” mapping replacement instead of YAML
   `arrays`.

-  add `core/transformation/rotation-1.0.0` schema that implements
   `scipy.spatial.transform.Rotation` and
   `transformations.WXRotation` class to create custom tagged
   `Rotation` instances for custom serialization.
   `[#79] <https://github.com/BAMWelDX/weldx/pull/79>`__

-  update requirements to `asdf>=2.7`
   `[#83] <https://github.com/BAMWelDX/weldx/pull/83>`__

-  update `anyOf` to `oneOf` in ASDF schemas
   `[#83] <https://github.com/BAMWelDX/weldx/pull/83>`__

-  add `__eq__` functions to `LocalCoordinateSystem` and
   `CoordinateSystemManager`
   `[#87] <https://github.com/BAMWelDX/weldx/pull/87>`__

.. _section-9:

0.1.0 (05.05.2020)
------------------

.. _asdf-9:

ASDF
~~~~

-  add basic file/directory layout for asdf files

   -  asdf schemas are located in
      `weldx/asdf/schemas/weldx.bam.de/weldx`
   -  tag implementations are in `weldx/asdf/tags/weldx`

-  implement support for pint quantities
-  implement support for basic pandas time class
-  implement base welding classes from AWS/NIST “A Welding Data
   Dictionary”
-  add and implement ISO groove types (DIN EN ISO 9692-1:2013)
-  add basic jinja templates and functions for adding simple dataclass
   objects
-  setup package to include and install ASDF extensions and schemas (see
   setup.py, MANIFEST.in)
-  add basic tests for writing/reading all ASDF classes (these only run
   code without any real checks!)

module:
~~~~~~~

-  add setup.py package configuration for install

   -  required packages
   -  package metadata
   -  asdf extension entry points
   -  version support

-  update pandas, scipy, xarray and pint minimum versions (in conda env
   and setup.py)
-  add versioneer
-  update options in setup.cfg
-  update tool configurations
