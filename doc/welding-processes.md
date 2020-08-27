# arc welding description
This document describes the main elements and workflow when describing arc welding processes. The process description is entirely focused on the welding parameters set on the power sources and does not cover measurements.

## base GMAW process schema 
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
The properties `base_process`, `manufacturer`, and `power_source` are general metadata fields that must be provided to identify the process. For now, these are basic string entries but will be defined more explicitly in later `weldx` iterations.
The `meta` properties is an optional field that can hold additional metadata if required.

## power source parameters
The `parameters` property is the most important aspect of the process definition.
This property is used to list all welding parameters that are set on the power source.
We use the `wx_property_tag` validator to restrict all properties of `parameters` to be a `core/time_series` object.
This means that all welding process parameters must be defined as a quantity (therefor having a unit) and with a time-dependent behavior. We will see some examples later.

## simple generic GMAW process definition
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
This tag class can be used to define any kind of GMAW process that matches the `base_process` layout as described above.
While possible to use it, it should only be used in circumstances where no explicit process definition exists (yet). The `process/GMAW` schema might be removed in future versions of the standard.

## default spray and pulse arc process definitions
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
The `base_process` property is enforced to indicate `spray` to correctly identify the process type.
The parameters properties ensure that the `parameters` property always includes a `wire_feedrate` and `voltage` with correct unit dimensionality.
```yaml
# terms-1.0.0.yaml
parameters:
  wire_feedrate:
    description: |
      Nominal average wire feedrate.
    tag: "tag:weldx.bam.de:weldx/core/time_series-1.0.0"
    wx_unit: "m/s"

  voltage:
    description: |
      Nominal target voltage for spray arc processes.
    tag: "tag:weldx.bam.de:weldx/core/time_series-1.0.0"
    wx_unit: "V"
```
## combining process schemas
We combine both the `terms-1.0.0#/base_process` and `terms-1.0.0#/process/spray` definitions using `allOf` to create the base spray arc process template:
```yaml
  allOf:
    - $ref: "./terms-1.0.0#/base_process"
    - $ref: "./terms-1.0.0#/process/spray"
```
## example spray arc definition
The above schemas only serve as building blocks for concrete manufacturer and equipment specific definitions of welding processes.

Let's take a simple spray arc process that can be used on a CLOOS Quinto II power source as an example.
The power source settings available are the following:
- wire feed rate
- welding voltage
- impedance
- characteristics

So in addition to the default pray arc parameters `wire_feedrate` and `voltage`, both `impedance` and `characteristics` also need to be defined for the CLOOS pray arc process.
We create the new schema file as `/process/CLOOS/spray_arc-1.0.0.yaml` to imply the manufacturer.
Here is the complete schema, covering the base_process metadata requirements as well as generic spray arc and additional CLOOS specific welding parameters:

```yaml
%YAML 1.1
---
$schema: "http://stsci.edu/schemas/yaml-schema/draft-01"
id: "http://weldx.bam.de/schemas/weldx/process/CLOOS/spray_arc-1.0.0"
tag: "tag:weldx.bam.de:weldx/process/CLOOS/spray_arc-1.0.0"

title: |
  CLOOS spray arc process.

allOf:
  - $ref: "../terms-1.0.0#/base_process"
  - $ref: "../terms-1.0.0#/process/spray"
  - type: object
    properties:
      parameters:
        type: object
        properties:
          impedance:
            tag: "tag:weldx.bam.de:weldx/core/time_series-1.0.0"
            wx_unit: "percent"
          characteristic:
            tag: "tag:weldx.bam.de:weldx/core/time_series-1.0.0"
            wx_unit: "V/A"
        required: [impedance, characteristic]

...
```

## use in python API
All GMAW process definitions are handle as instances of the `weldx.welding.GmawProcess` class.
Here is how to create an example instance implementing the CLOOS GMAW spray arc process above:
```python
from weldx import Q_
from weldx.welding import GmawProcess

params_spray = dict(
    wire_feedrate=Q_(10.0, "m/min"),
    voltage=Q_(40.0, "V"),
    impedance=Q_(10.0, "percent"),
    characteristic=Q_(5,"V/A"),
)
process_spray = GmawProcess(
    "spray", "CLOOS", "Quinto", params_spray, tag="CLOOS/spray_arc"
)
```
Note that we have to manually assign the tag (without version information) that matches the `CLOOS/spray_arc-1.0.0.yaml` schema to correctly associate the tag.

And here is the resulting ASDF snippet:
```yaml
spray: !<tag:weldx.bam.de:weldx/process/CLOOS/spray_arc-1.0.0>
  base_process: spray
  manufacturer: CLOOS
  parameters:
    characteristic: !<tag:weldx.bam.de:weldx/core/time_series-1.0.0>
      unit: volt / ampere
      values: 5
    impedance: !<tag:weldx.bam.de:weldx/core/time_series-1.0.0>
      unit: percent
      values: 10.0
    voltage: !<tag:weldx.bam.de:weldx/core/time_series-1.0.0>
      unit: volt
      values: 40.0
    wire_feedrate: !<tag:weldx.bam.de:weldx/core/time_series-1.0.0>
      unit: meter / minute
      values: 10.0
  power_source: Quinto
  tag: CLOOS/spray_arc
```