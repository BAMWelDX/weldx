# WelDX CHANGELOG.md

## 0.1.1 (unreleased)
### ASDF
- add `wx_unit` and `wx_shape` validators
- add basic tests for `wx_unit` validation
- add basic tests for `wx_shape` validation
- add unit validation to `iso_groove-1.0.0.yaml` 
- fixed const/enum constraints and properties in `iso_groove-1.0.0.yaml`
- add some examples for testing `oneOf`, `anyOf` and other asdf keywords
- add NetCDF inspired common types (`Dimension`,`Variable`) with corresponding
 asdf serialization classes
- add asdf serialization classes and schemas for `xarray.DataArray`, 
`xarray.Dataset`, `weldx.transformations.LocalCoordinateSystem` and
`weldx.transformations.CoordinateSystemManager`.
- add test for `xarray.DataArray`, `xarray.Dataset`, `weldx.transformations.LocalCoordinateSystem` and
`weldx.transformations.CoordinateSystemManager` serialization.
- allow using `pint.Quantity` coordinates in `weldx.transformations.LocalCoordinateSystem` [#70](https://github.com/BAMWelDX/weldx/pull/70)
- add measurement related ASDF serialization classes: [#70](https://github.com/BAMWelDX/weldx/pull/70)
  - `equipment/generic_equipment-1.0.0`
  - `measurement/data-1.0.0`
  - `data_transformation-1.0.0`
  - `measurement/error-1.0.0`
  - `measurement/measurement-1.0.0`
  - `measurement/measurement_chain-1.0.0`
  - `measurement/signal-1.0.0`
  - `measurement/source-1.0.0`
- add example notebook for measurement chains in tutorials [#70](https://github.com/BAMWelDX/weldx/pull/70)
- add and test support for `sympy` expressions with 'weldx.core.MathematicalExpression' and ASDF serialization in '`core/mathematical_expression-1.0.0` [#70](https://github.com/BAMWelDX/weldx/pull/70), [#76](https://github.com/BAMWelDX/weldx/pull/76)
- add and test class to describe time series - 'weldx.core.TimeSeries' [#76](https://github.com/BAMWelDX/weldx/pull/76)
- add `wx_property_tag` validator [#72](https://github.com/BAMWelDX/weldx/pull/72)
 
    the `wx_property_tag` validator restricts **all** properties of an object to a single tag.
    For example the following object can have any number of properties but all must be
    of type `tag:weldx.bam.de:weldx/time/timestamp-1.0.0`
    ```yaml
    type: object
    additionalProperties: true # must be true to allow any property
    wx_property_tag: "tag:weldx.bam.de:weldx/time/timestamp-1.0.0"  
    ```
    It can be used as a "named" mapping replacement instead of YAML `arrays`.
- add `core/transformation/rotation-1.0.0` schema that implements `scipy.spatial.transform.Rotation` and `transformations.WXRotation` class to create custom tagged `Rotation` instances for custom serialization. [#79]
- update requirements to `asdf>=2.7` [#83](https://github.com/BAMWelDX/weldx/pull/83)
- update `anyOf` to `oneOf` in ASDF schemas [#83](https://github.com/BAMWelDX/weldx/pull/83)
- add `__eq__` functions to `LocalCoordinateSystem` and `CoordinateSystemManager` [#87](https://github.com/BAMWelDX/weldx/pull/87)
