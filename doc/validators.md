# WelDX ADF validators overview
The `weldx` library implements a couple of custom validators in addition to existing ASDF validators.
These validators can be used by any custom schema definitions when using the WelDX-Extension.

## Naming conventions
All `weldx`-validators are prefixed with `wx_` in the ASDF-schema definitions.

## WelDX Validators
The following custom validators are provided:
*   `wx_unit`
*   `wx_shape`
*   `wx_property_tag`

## Custom implementation of existing ASDF validators
The following existing ASDF validators have extended implementations when using the `wx_` prefix:
*   `wx_tag`