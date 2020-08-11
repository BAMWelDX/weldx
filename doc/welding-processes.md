# arc welding description
This document describes the main elements and workflow when describing arc welding processes. The process description is entirely focused on the welding parameters set on the power sources and does not cover measurements.

The main layout of any GMAW process is defined in the `http://weldx.bam.de/schemas/weldx/process/terms-1.0.0` schema as `base_process`:
```yaml
# terms-1.0.0.yaml
base_process:
  description: |
    The base metadata format for all GMAW process descriptions.
  type: object
  properties:
    base_process:
      type: string
    manufacturer:
      type: string
    power_source:
      type: string
    parameters:
      type: object
      wx_property_tag: "tag:weldx.bam.de:weldx/core/time_series-*"
    meta:
      type: object
  required: [base_process,manufacturer,power_source,parameters]
```
The properties `base_process`, `manufacturer` and `power_source` are general metadata fields that must be provided to identify the process. For now these are basic string entries but will be defined more explicitly in later `weldx` iterations.

The `parameters` property is the most important aspect of the process definition.
This property is used to list all welding parameters that are set on the power source.
We use the `wx_property_tag` validator to restrict all properties of `parameters` to be a `core/time_series` object.
This means that all welding process parameters must be defined as a quantity (therefor having a unit) and with a time-dependent behavior. We will see some examples later.

Finally, the `meta` properties is an optional field that can hold additional metadata if required.

The most generic tag implementation of any arc welding process is provided by `tag:weldx.bam.de:weldx/process/GMAW-1.0.0`:
```yaml
%YAML 1.1
---
$schema: "http://stsci.edu/schemas/yaml-schema/draft-01"
id: "http://weldx.bam.de/schemas/weldx/process/GMAW-1.0.0"
tag: "tag:weldx.bam.de:weldx/process/GMAW-1.0.0"

title: |
  Generic GMAW process definition.

$ref: "./terms-1.0.0#/base_process"
...
```
This tag class can be used to define any kind of GMAW process that matches the the `base_process` layout as described above.
While possible to use it, it should only be used in circumstances where no explicit process definition exists (yet). The `process/GMAW` schema might be removed in future versions of the standard.

In addition to the `base_process` structure, the basic process variations like spray and pulsed transfer modes are also defined in `process/terms`.
These define the parameters that *must* be provided for all variations of the process.
Here are the requirements to match when describing a generic spray arc process:
```yaml
# terms-1.0.0.yaml
process:
  spray:
    type: object
    properties:
      base_process:
        type: string
        enum: [spray]
      parameters:
        type: object
        properties:
          wire_feedrate:
            $ref: "#/parameters/wire_feedrate"
          voltage:
            $ref: "#/parameters/voltage"
        required: [wire_feedrate, voltage]
```
This enforces that the `parameters` property always includes a `wire_feedrate` and `voltage` with correct unit dimensionality. Moreover it sets the `base_process` property to `spray` to correctly identify the process type.

We combine both the `terms-1.0.0#/base_process` and `terms-1.0.0#/process/spray` definitions to creates the base spray arc process template:
```yaml
  allOf:
    - $ref: "./terms-1.0.0#/base_process"
    - $ref: "./terms-1.0.0#/process/spray"
```
