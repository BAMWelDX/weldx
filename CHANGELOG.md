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
- allow using `pint.Quantity` coordinates in `weldx.transformations.LocalCoordinateSystem` [#70]
- add measurement related ASDF serialization classes: [#70]
  - `equipment/generic_equipment-1.0.0`
  - `equipment/sensor-1.0.0`
  - `measurement/data-1.0.0`
  - `data_transformation-1.0.0`
  - `measurement/error-1.0.0`
  - `measurement/measurement-1.0.0`
  - `measurement/measurement_chain-1.0.0`
  - `measurement/signal-1.0.0`
  - `measurement/source-1.0.0`
- add ASDF support for `sympy` expressions with `core/mathematical_expression-1.0.0` [#70]
- add example notebook for measurement chains in tutorials [#70]
