# Definition of Syntax for the Shape validation

In this article we discuss how we validate the shape of objects (mostly arrays).

Let us say we have an array with 5 dimensions we want the first three have the dimension of n=3 the fourth has
dimension 4 and the last one is optional with dimension 2. 

We would get this array 

`expected = [n, n, n, 4, (2)]` 

and would validate it with the actual array 

`test = [3, 3, 3, 4, 2]`.
 
Through the given array the variable n is defined. And any array that does not match our requirement would not 
be accepted and throws a value error. 

Some examples that would **not** match our expected array:

`test = [1, 2, 2, 4, 2]` the n mismatches.

`test = [7, 7, 7, 4, 3]` the optional dimension has to be a 2.

`test = [1, 1, 1, 4, 2, 2]` this has more dimensions as we expect.

So what we need is a set of rules for the syntax of those shapes. The document will follow with exceptions and examples.

## Syntax

Each array item follows these rules:
* an ``Integer`` indicates a fix dimension for the same item
* a ``~``, `:` or `None` indicates a single dimension of arbitrary length.
* a ``...`` indicates an arbitrary number of dimensions of arbitrary length, which can be optional.
* a ``n`` indicates a single dimension fixed to an alphanumeric. So a string out of letters and numbers is allowed.
* parenthesis ``(_)`` indicate that the dimension is optional. This can be combined with the other rules.
* the symbols ``~`` or `:` furthermore add the option to implement an interval. This string `4~` would be an open
interval that accepts all dimensions that are greater or equal to 4.

## Exceptions

This is an additional rule set which describes (un-)intuitive rules:
*  No negative Dimensions are allowed.
* Parenthesis and `...` cannot be combined to `(...)`.
* The addition with the interval can only be ascending. Wrong would be `5~2`
* Parenthesis and `...` can either be at the beginning or the end of the array.
* It is possible to have multiple optional dimensions. They must stand all be at the beginning or the end.
So ``[(1), 2, (3)]`` is not allowed.
*

## Examples

Example of a validator and its matches and mismatches.

**Validator**:

``["n", "~", 2, "~6", "(n)", (3), "..."]``

**Matches**:

``[3, 4, 2, 4, 3]``

``[1, 3, 2, 3, 1, 3, 7, 8, 9]``

``[1, 1, 2, 1]``

**Mismatches**:

``[1, 4, 2, 4, 3]`` mismatch of n: 1 = 3

``[2, 4, 2, 4, 2, 2]`` mismatch of optional (3) = 2

``[2, 4, 2, 7, 2, 3]`` mismatch of `~`: 7 > 6 but has to be less then or equal to 6.

``[2, 4, 2, -3, 2, 3]`` No negative dimensions allowed

Now some examples of validators which will throw an **error**:

``["(1)", 2, "(3)"]`` Validators are only allowed at the beginning or the end.

``["11", 22, "3(3)"]`` Any character outside the parenthesis will cause an error.

``["11", 22, "x..."]`` Any character in the `...` will cause an error.

``["11", 22, "m_1"]`` Underscores are not supported in variable names. Only alphanumeric strings are allowed.  


