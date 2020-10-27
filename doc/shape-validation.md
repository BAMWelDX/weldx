# Shape validation using `wx_shape`
In this article we discuss how to validate the shape of objects (mostly arrays).
## Syntax definition
Let us say we have an array with 5 dimensions we want the first three have the dimension of `n=3` the fourth has
dimension 4 and the last one is optional with dimension 2. We would get this shape syntax

`expected = [n, n, n, 4, (2)]`

and would validate it with the actual shape 

`test = [3, 3, 3, 4, 2]`.
 
Through the given shape the variable `n` is defined. And any shape that does not match our requirement would not 
be accepted and throws a value error. 

Some examples that would **not** match our expected shape:

`[1, 2, 2, 4, 2]` the `n` mismatches.

`[7, 7, 7, 4, 3]` the optional 5th dimension has to be a 2.

`[1, 1, 1, 4, 2, 2]` this has more dimensions as we expect.

So what we need is a set of rules for the syntax of those shapes. The document will follow with exceptions and examples.

### Syntax

Each shape item follows these rules:
*   an ``Integer`` indicates a fix dimension for the same item

*   a ``~``, `:` or `None` indicates a single dimension of arbitrary length.

*   a ``...`` indicates an arbitrary number of dimensions of arbitrary length, which can be optional.

*   a ``n`` indicates a single dimension fixed to an alphanumeric. So a string out of letters and numbers is allowed.

*   parenthesis ``(_)`` indicate that the dimension is optional. This can be combined with the other rules.

*   the symbols ``~`` or `:` furthermore add the option to implement an interval. This string `4~` would be an open
interval that accepts all dimensions that are greater or equal to 4.

### Exceptions

This is an additional rule set which describes (un-)intuitive rules:

*   No negative Dimensions are allowed.

*   Parenthesis and `...` cannot be combined to `(...)`.

*   The addition with the interval can only be ascending. Wrong would be `5~2`

*   Parenthesis and `...` can either be at the beginning or the end of the shape syntax.

*   It is possible to have multiple optional dimensions. They must stand all be at the beginning or the end.
So ``[(1), 2, (3)]`` is not allowed.

### Examples
Example of a validator and its matches and mismatches.

**Validator**:\
`["n", "~", 2, "~6", "(n)", (3), "..."]`

**Matches**:\
`[3, 4, 2, 4, 3]`\
`[1, 3, 2, 3, 1, 3, 7, 8, 9]`\
`[1, 1, 2, 1]`

**Mismatches**:\
`[1, 4, 2, 4, 3]` mismatch of n: 1 = 3\
`[2, 4, 2, 4, 2, 2]` mismatch of optional (3) = 2\
`[2, 4, 2, 7, 2, 3]` mismatch of `~`: 7 > 6 but has to be less then or equal to 6.\
`[2, 4, 2, -3, 2, 3]` No negative dimensions allowed\

Now some examples of validators which will throw an **error**:\
`["(1)", 2, "(3)"]` Validators are only allowed at the beginning or the end.\
`["11", 22, "3(3)"]` Any character outside the parenthesis will cause an error.\
`["11", 22, "x..."]` Any character in the `...` will cause an error.\
`["11", 22, "m_1"]` Underscores are not supported in variable names. Only alphanumeric strings are allowed.

## ASDF schema usage 
Now that we know the syntax let's take a look at how to incorporate it in our ASDF schema definitions.
The validation gets triggered by the `wx_shape` keyword.

For the validation to work the validator has to be defined on a `property` that itself has a list-like `shape` property.
Take an `ndarray` property for example:
```yaml
# ASDF schema
properties:
  array_prop:
    tag: tag:stsci.edu:asdf/core/ndarray-1.0.0
```
```yaml
# ASDF file
  array_prop: !core/ndarray-1.0.0
    data: [0, 1, 2, 3, 4]
    datatype: int32
    shape: [5]
```
We would validate this to always have shape `[5]` by adding the `wx_shape` keyword to the schema definition.
```yaml
# ASDF schema
properties:
  array_prop:
    tag: tag:stsci.edu:asdf/core/ndarray-1.0.0
    wx_shape: [5]
```
The above example shows the basic usage for a single property.
We can use most of the syntax features like `()`,`~` and `...`.
But be aware that the scope of this "inline" wx_shape validation is limited to the property that it validates!
So no comparison to other shapes with alphanumerics is possible.

For example, following schema would validate and file would validate:
```yaml
# ASDF schema
properties:
  array_prop:
    tag: tag:stsci.edu:asdf/core/ndarray-1.0.0
    wx_shape: [n]
  array_prop2:
    tag: tag:stsci.edu:asdf/core/ndarray-1.0.0
    wx_shape: [n]
```
```yaml
# ASDF file
  array_prop: !core/ndarray-1.0.0
    data: [0, 1, 2, 3, 4]
    datatype: int32
    shape: [5]
  array_prop2: !core/ndarray-1.0.0
    data: [0, 1]
    datatype: int32
    shape: [2]
```
To compare and validate shapes across multiple properties we have to use a nested syntax that has all necessary properties in its scope.
To assure `array_prop` and `array_prop2` have the same shape we use the following schema:
```yaml
# ASDF schema
properties:
  array_prop:
    tag: tag:stsci.edu:asdf/core/ndarray-1.0.0
  array_prop2:
    tag: tag:stsci.edu:asdf/core/ndarray-1.0.0
wx_shape:
  array_prop: [n]
  array_prop2: [n]
```
Note the following:
*   `wx_shape` is now defined on the same level as the `properties` keyword.
*   `wx_shape` is no longer a shape-like list but itself a nested object with shape-like lists as leaves.

### optional properties
Properties that are optional (not listed as `required`) must be indicated as such for shape validation by putting the name in brackets. 
In this example, both `optional_prop` will only get validated if it exists in the tree.
```yaml
# ASDF schema
properties:
  required_prop:
    tag: tag:stsci.edu:asdf/core/ndarray-1.0.0
  optional_prop:
    tag: tag:stsci.edu:asdf/core/ndarray-1.0.0
wx_shape:
  required_prop: [n]
  (optional_prop): [n]
required: [required_prop]
```

### custom types validation
The following custom types can be validate with `wx_shape` even though the might not always define a shape property in itself.
*   `number` will validate like `shape: [1]`
*   `tag:weldx.bam.de:weldx/time/timedeltaindex-1.0.0` will validate against the length of the `TimedeltaIndex` even if no data is stored.

### complex nested example
Here is a more complex example demonstration some of the above points.
```yaml
%YAML 1.1
---
$schema: "http://stsci.edu/schemas/yaml-schema/draft-01"
id: "http://weldx.bam.de/schemas/weldx/debug/test_shape_validator-1.0.0"
tag: "tag:weldx.bam.de:weldx/debug/test_shape_validator-1.0.0"

title: |
  simple demonstration and test schema for wx_shape validator syntax
type: object
properties:
  prop1:
    tag: tag:stsci.edu:asdf/core/ndarray-1.0.0
    wx_shape: [1,2,(3),(4)]

  prop2:
    tag: tag:stsci.edu:asdf/core/ndarray-1.0.0
    wx_shape: [~,2,1]

  prop3:
    tag: tag:stsci.edu:asdf/core/ndarray-1.0.0
    wx_shape: [2,4,6,8,...]

  prop4:
    tag: tag:stsci.edu:asdf/core/ndarray-1.0.0
    wx_shape: [~,3,5,7,9]

  prop5:
    type: number
    wx_shape: [1]

  nested_prop:
    type: object
    properties:
      p1:
        tag: tag:stsci.edu:asdf/core/ndarray-1.0.0
        wx_shape: [10,8,6,4,2]
      p2:
        tag: tag:stsci.edu:asdf/core/ndarray-1.0.0
        wx_shape: [9,7,5,3,1]

  optional_prop:
    tag: tag:stsci.edu:asdf/core/ndarray-1.0.0
    wx_shape: [1,2,(3),(4)]



required: [prop1,prop2,prop3,prop4,nested_prop]
propertyOrder: [prop1,prop2,prop3,prop4,nested_prop,optional_prop]
flowStyle: block
additionalProperties: true
wx_shape:
  prop1: [(~),2,n]
  prop2: [n,2,1]
  prop3: [2,4,5~7,...]
  prop4: [a,3,5,k,m]
  prop5: [a]
  nested_prop:
    p1: [10,1~10,6,4,2]
    p2: [(m),7,5,3,1]
    (p3): [a,2,n]
  (optional_prop): [a,2,n]
``` 
