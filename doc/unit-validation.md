# unit validation using `wx_unit`
When using `quantity` properties in schemas, the dimensionality of the `unit` attribute can be validated using the `wx_unit` validator.\
Similar to `wx_shape` validation, `wx_unit` assumes that the object it is attached to has a `unit` property of type `string` to validate against.

It is important to note that unit validation does not perform a literal string comparison (like can be archived with `enum`) but checks the correct dimensionality of the unit.

For example, validation with `wx_unit: "m"` will be successful against all quantities that represent a length. So this schema
```yaml
type: object
properties:
  length_prop:
    tag: tag:stsci.edu:asdf/unit/quantity-1.1.0
    wx_unit: "m"
```
allows all of the following:
```yaml
length_prop: !unit/quantity-1.1.0 {unit: millimeter, value: 1.5}
length_prop: !unit/quantity-1.1.0 {unit: meter, value: 32}
length_prop: !unit/quantity-1.1.0 {unit: inch, value: 123}
```

`wx_unit` is implemented using `pint` dimensionality-checks.
This means that for the unit syntax, every string input that gets picked up by `pint` correctly can be used. The following are all allowed and result in the same behavior:
```yaml
wx_unit: "s"
wx_unit: "seconds"
wx_unit: "ms"
```

It is also possible to validate multiple or nested properties with a single `wx_unit` entry when using an `object`-like structure.
See this more complex example for details:
```yaml
tag: "tag:weldx.bam.de:weldx/debug/test_unit_validator-1.0.0"

title: |
  simple demonstration schema for wx_unit validator
type: object
properties:
  length_prop:
    description: |
      a simple length quantity with unit validator
    tag: tag:stsci.edu:asdf/unit/quantity-1.1.0
    wx_unit: "m"

  velocity_prop:
    description: |
      a simple velocity quantity
    allOf:
      - $ref: tag:stsci.edu:asdf/unit/quantity-1.1.0
      - type: object
        wx_unit: "m/s"

  current_prop:
    description: |
      a current quantity of shape [2,2]
    $ref: tag:stsci.edu:asdf/unit/quantity-1.1.0

  nested_prop:
    description: |
      a nested object with two quantities
    type: object
    properties:
      q1:
        description: a nested length of shape [3,3]
        $ref: tag:stsci.edu:asdf/unit/quantity-1.1.0
      q2:
        description: a volume
        $ref: tag:stsci.edu:asdf/unit/quantity-1.1.0
    wx_unit:
      q1: "m"

  simple_prop:
    description: simple property without any references ref
    type: object
    properties:
      value:
        type: number
      unit:
        type: string
    wx_unit: "m"

wx_unit:
  length_prop: "m"
  velocity_prop: "m / s"
  current_prop: A
  nested_prop:
    q2: "m*mm*cm"
```