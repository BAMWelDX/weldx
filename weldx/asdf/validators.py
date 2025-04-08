"""ASDF-validators for weldx types."""

from __future__ import annotations

import re
from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping
from typing import Any

from asdf.exceptions import ValidationError
from asdf.extension import Validator
from asdf.schema import _type_to_tag

from weldx.asdf.types import WxSyntaxError
from weldx.asdf.util import _get_instance_shape, _get_instance_units, uri_match
from weldx.constants import U_

__all__ = ["WxUnitValidator", "WxShapeValidator", "WxPropertyTagValidator"]


def _walk_validator(
    instance: OrderedDict,
    validator_dict: OrderedDict,
    validator_function: Callable[[Mapping, Any, str], Iterator[ValidationError]],
    position=None,
    allow_missing_keys: bool = False,
) -> Iterator[ValidationError]:
    """Walk instance and validation dict entries in parallel and apply a validator func.

    This function can be used to recursively walk both the instance dictionary and the
    custom validation dictionary in parallel. Once a leaf dictionary entry is reached,
    the validation function is applied to the selected items.

    Parameters
    ----------
    instance:
        Tree serialization (with default dtypes) of the instance
    validator_dict:
        OrderedDict representation of the validation structure.
    validator_function:
        Custom python validator function to apply along the (nested) dictionary
    position:
        String representation of the current nested schema position
    allow_missing_keys:
        If True will skip validation if the requested key to validate does not exist.

    Yields
    ------
    asdf.exceptions.ValidationError

    """
    if position is None:  # pragma: no cover
        position = []
    if isinstance(validator_dict, dict):
        for key, item in validator_dict.items():
            if isinstance(item, Mapping):
                yield from _walk_validator(
                    instance[key],
                    validator_dict[key],
                    validator_function,
                    position=position + [key],
                    allow_missing_keys=allow_missing_keys,
                )
            else:
                if key in instance:
                    yield from validator_function(instance[key], item, position + [key])
                elif allow_missing_keys:  # pragma: no cover
                    pass
                else:  # pragma: no cover
                    pass
                    # TODO: if a property is not required the key might be missing
                    # yield ValidationError(f"Missing key {key}")

    else:
        yield from validator_function(instance, validator_dict, position)


def _unit_validator(
    instance: Mapping, expected_dimensionality: str, position: list[str]
) -> Iterator[ValidationError]:
    """Validate the 'unit' key of the instance against the given string.

    Parameters
    ----------
    instance:
        Tree serialization with 'unit' key to validate.
    expected_dimensionality:
        String representation of the unit dimensionality to test against.
    position:
        Current position in nested structure for debugging

    Yields
    ------
    asdf.exceptions.ValidationError

    """
    if not position:
        position = instance

    units = _get_instance_units(instance)
    if units is None:
        yield ValidationError(
            f"Error validating unit dimension for property '{position}'. "
            f"Expected unit of dimension '{expected_dimensionality}' "
            "but found no unit information"
        )
    else:
        valid = units.is_compatible_with(U_(expected_dimensionality))
        if not valid:
            yield ValidationError(
                f"Error validating unit dimension for property '{position}'. "
                f"Expected unit of dimension '{expected_dimensionality}' "
                f"but got unit '{units}'"
            )


def _compare(_int, exp_string):
    """Compare helper of two strings for _custom_shape_validator.

    An integer and an expected string are compared so that the string either contains
    a "~" and thus describes an interval or a string consisting of numbers. So if our
    integer is within the interval or equal to the described number, True is returned.
    The interval can be open, in that there is no number left or right of the "~"
    symbol.

    Parameters
    ----------
    _int:
        Integer
    exp_string:
        String with the expected dimension

    Returns
    -------
    bool
        True or False

    Examples:
    ---------
    >>> from weldx.asdf.validators import _compare
    >>> _compare(5,"5")
    True

    >>> _compare(5,"~")
    True

    >>> _compare(5,"3~")
    True

    >>> _compare(5,"4")
    False

    open interval:
    >>> _compare(5,"~")
    True

    open interval:
    >>> _compare(5,"3~")
    True

    closed interval:
    >>> _compare(5,"4~6")
    True

    """
    if _int < 0:
        raise WxSyntaxError("Negative dimension found")

    if "~" in exp_string:
        ranges = exp_string.split("~")

        if ranges[0] == "":
            ranges[0] = 0
        elif ranges[0].isnumeric():
            ranges[0] = int(ranges[0])
        else:
            raise WxSyntaxError(f"Non numeric character in range {exp_string}")
        if ranges[1] == "":
            ranges[1] = _int
        elif ranges[1].isnumeric():
            ranges[1] = int(ranges[1])
        else:
            raise WxSyntaxError(f"Non numeric character in range {exp_string}")

        if ranges[0] > ranges[1]:
            raise WxSyntaxError(f"The range should not be descending in {exp_string}")
        return int(ranges[0]) <= _int <= int(ranges[1])

    else:
        return _int == int(exp_string)


def _prepare_list(_list, list_expected):
    """Prepare a List and an expected List for validation.

    The preparation of the lists consists in accepting all lists that contain
    white spaces.
    In addition, lists that begin with "..." or parentheses are reversed for
    validation to work.

    parameters
    ----------
    _list:
        List with values
    list_expected:
        List with expected values
    returns
    -------
    _list:
        prepared List
    list_expected:
        prepared List with expected values
    """
    # remove blank spaces in dict_test
    _list = [x.replace(" ", "") if isinstance(x, str) else x for x in _list]
    list_expected = [
        x.replace(" ", "") if isinstance(x, str) else x for x in list_expected
    ]
    # turn around the list if "..." or "(" are at the beginning.
    # because the validation is made from begin -> end.
    # like this we validate the array from end -> begin.
    if "(" in str(list_expected[0]) or "..." in str(list_expected[0]):
        list_expected = list(reversed(list_expected))
        _list = list(reversed(_list))

    return _list, list_expected


def _is_range_format_valid(format_string: str) -> bool:
    """
    Return 'True' if a string represents a valid range definition and 'False' otherwise.

    Parameters
    ----------
    format_string:
        String that should be checked.

    Returns
    -------
    bool:
        'True' if the passed string is a valid range definition, 'False' otherwise
    """
    if "~" in format_string:
        if len(format_string.split("~")) != 2:
            return False
        format_string = format_string.replace("~", "")
        return format_string.isalnum() or format_string == ""
    return format_string.isalnum()


def _validate_expected_list(list_expected):
    """Validate an expected List and raises exceptions.

    The syntax of the expected list is validated.
    Examples that will raise errors:

    Variable length should be at beginning or end.
    [1, 2, "...", 4, 5]
    [1, 2, "(3)", 4, 5]

    Additional arguments are not accepted
    [1, 2, 3, 4, "5..."]
    [1, 2, 3, 4, "5(5)"]

    Parameters
    ----------
    list_expected:
        Expected List to validate against

    Raises
    ------
    ValueError:
        ValueError will be raised if an rule violation is found
    """
    validator = 0
    for exp in list_expected:
        if validator == 1 and not ("(" in str(exp) or "..." in str(exp)):
            raise WxSyntaxError(
                "Optional dimensions in the expected "
                "shape should only stand at the end/beginning."
            )
        if validator == 2:
            raise WxSyntaxError('After "..." should not be another dimension.')
        if "..." in str(exp):
            if "..." != exp:
                raise WxSyntaxError(
                    f'"..." should not have additional properties: {exp} was found.'
                )
            validator = 2
        elif "(" in str(exp):
            val = re.search(r"\((.*)\)", exp)
            if (
                val is None
                or len(val.group(1)) + 2 != len(exp)
                or not _is_range_format_valid(val.group(1))
            ):
                raise WxSyntaxError(
                    f'Invalid optional dimension format. Correct format is "(~)", but '
                    f" {exp} was found."
                )

            validator = 1
        elif not _is_range_format_valid(str(exp)):
            raise WxSyntaxError(
                f"{exp} is an invalid range format."
                f"Consult the documentation for a list of all valid options"
            )


def _compare_lists(_list, list_expected) -> bool | dict:
    """Compare two lists.

    The two lists are interpreted as a list of dimensions. We compare the dimensions of
    these lists and in the case of a mismatch, False is returned. If the dimensions
    match and there are variables in the list_expected, they are entered into a
    dictionary and this is output. The dictionary can be empty if there are no
    variables in the list_expected.

    Parameters
    ----------
    _list:
        List of Integer
    list_expected:
        List build by the rules in _custom_shape_validator

    Returns
    -------
    bool:
        when a dimension mismatch occurs
    dict_values:
        when no dimension mismatch occurs. Can be empty {}. Dictionary - keys: variable
        names in the validation schemes. values: values of the validation schemes.

    Examples
    --------
    >>> from weldx.asdf.validators import _compare_lists
    >>> _compare_lists([1, 2, 3], [1, 2, 3])
    {}

    >>> _compare_lists([1, 2, 3], [1, "a", "b"])
    {'a': 2, 'b': 3}

    >>> _compare_lists([1, 2, 3], [1, "..."])
    {}

    >>> _compare_lists([1, 2, 3], [1, 2, 4])
    False

    """
    dict_values = dict()

    has_variable_dim_num = False
    for i, exp in enumerate(list_expected):
        if "..." in str(exp):
            has_variable_dim_num = True
            break  # all the following dimensions are accepted

        if "(" in str(exp):
            if i < len(_list):
                exp = re.search(r"\((.*)\)", exp).group(1)
            else:  # pragma: no cover
                continue  # TODO: actually covered, but not registered by codecov - bug?

        # all alphanumeric strings are OK - only numeric strings are not
        # eg: "n", "n1", "n1234", "myasdfstring1337"
        if str(exp).isalnum() and not str(exp).isnumeric():
            if exp not in dict_values:
                dict_values[exp] = _list[i]
            elif _list[i] != dict_values[exp]:
                return False

        elif i >= len(_list) or not _compare(_list[i], str(exp)):
            return False

    if (len(_list) > len(list_expected)) and not has_variable_dim_num:
        return False

    return dict_values


def _validate_instance_shape(
    dict_test: dict, shape_expected: list, optional: bool = False
) -> dict:
    """Validate a node instance against a shape definition.

    Parameters
    ----------
    dict_test
        The node to be validated
    shape_expected
        List that defines the expected shape constraints.
    optional
        flag if validation is optional

    Returns
    -------
    dict
        dictionary of shape keys (empty dictionary if no variables in shape_expected)

    Raises
    ------
    ValidationError
        If the shape does not match the requirements or no shape could be found but
        validation is not flagged as optional

    """
    shape = _get_instance_shape(dict_test)
    if shape is None:  # could not determine shape of node
        if optional:  # we are allowed to skip shape validation
            return {}
        raise ValidationError(f"Could not determine shape in instance {dict_test}.")

    list_test, list_expected = _prepare_list(shape, shape_expected)

    _validate_expected_list(list_expected)
    _dict_values = _compare_lists(list_test, list_expected)

    if _dict_values is False:
        raise ValidationError(
            f"Shape {list_test[::-1]} does not match requirement {list_expected[::-1]}"
        )

    return _dict_values


def _custom_shape_validator(
    dict_test: dict[str, Any],
    dict_expected: dict[str, Any] | list,
    optional: bool = False,
):
    """Validate dimensions which are stored in two dictionaries dict_test and
    dict_expected.

    Syntax for the dict_expected:
    -----------------------------
    Items with arrays with each value having the following Syntax:
    1)  3 : an integer indicates a fix dimension for the same item in dict_test
    2)  "~" or None : this string indicates a single dimension of arbitrary length.
    3)  "..." : this string indicates an arbitrary number of dimensions of arbitrary
            length. Can be optional.
    4)  "2~4" : this string indicates a single dimension of length 2, 3 or 4. This
            has to be ascending or you can give an unlimited interval limit like this
            "2~" which would indicate a single dimension of length greater then 1.
    5)  "n" : this indicates a single dimension fixed to a letter. Any letter or
            combination of letters should work The letter will be compared to the same
            letter in all the arrays in the dict_expected.
    6)  (x) : parenthesis indicates that the dict_test does not need to have this
            dimension. This can NOT be combined with 3) or the None from 2).

    Parameters
    ----------
    dict_test:
        dictionary to validate
    dict_expected:
        dictionary with the expected values

    Raises
    ------
    ValueError:
        when dict_expected does violate against the Syntax rules

    Returns
    -------
    False
        when any dimension mismatch occurs
    dict_values
        Dictionary - keys: variable names in the validation schemes. values: values of
        the validation schemes.

    """

    dict_values = {}

    if isinstance(dict_expected, str):
        # could be optional inline notation ([.....])
        if not (dict_expected.startswith("([") and dict_expected.endswith("])")):
            raise WxSyntaxError(f"Found an incorrect wx_shape object: {dict_expected}.")
        dict_expected = dict_expected[2:-2].split(",")
        optional = True

    # handle single shape definitions
    if isinstance(dict_expected, list):
        # we have reached a shape definition, try validation
        return _validate_instance_shape(dict_test, dict_expected, optional=optional)

    elif isinstance(dict_expected, dict):
        # we need to go deeper ...
        for key, item in dict_expected.items():
            # allow optional syntax
            _optional = False
            if key.startswith("(") and key.endswith(")"):
                key = key[1:-1]
                _optional = True
                if len(key) == 0:
                    raise WxSyntaxError("wx_shape entry undefined")

            # test shapes
            if key in dict_test:
                # go one level deeper in the dictionary
                _dict_values = _custom_shape_validator(
                    dict_test[key], item, optional=_optional
                )
            else:
                return dict_values

            for key in _dict_values:
                if key not in dict_values:
                    dict_values[key] = _dict_values[key]
                elif dict_values[key] != _dict_values[key]:
                    return False
    else:
        raise WxSyntaxError(
            f"Found an incorrect wx_shape object: {type(dict_expected)}. "
            "Should be a dict or list."
        )

    return dict_values


# VALIDATOR CLASSES --------------------------------------------------------------------


class WxUnitValidator(Validator):
    """Custom validator for checking dimensions for objects with 'unit' property."""

    schema_property = "wx_unit"
    tags = ["**"]

    def validate(self, wx_unit, node, schema):
        """Run unit validation."""

        yield from _walk_validator(
            instance=node,
            validator_dict=wx_unit,
            validator_function=_unit_validator,
            position=[],
            allow_missing_keys=False,
        )


class WxShapeValidator(Validator):
    """Custom validator for checking dimensions for objects with 'shape' property."""

    schema_property = "wx_shape"
    tags = ["**"]

    def validate(self, wx_shape, node, schema):
        """Run shape validation."""

        dim_dict = None
        try:
            dim_dict = _custom_shape_validator(node, wx_shape)
        except ValidationError:
            yield ValidationError(f"Error validating shape {wx_shape}.\nOn node {node}")

        if isinstance(dim_dict, dict):
            return None
        else:
            yield ValidationError(f"Error validating shape {wx_shape}.\nOn node {node}")


class WxPropertyTagValidator(Validator):
    """Validate list of properties against specific tags."""

    schema_property = "wx_property_tag"
    tags = ["**"]

    def validate(self, wx_property_tag, node, schema):
        """Run property tag validation."""

        def _tag_validator(tagname, node):
            """Validate against a tag string using ASDF uri match patterns."""
            if hasattr(node, "_tag"):
                node_tag = node._tag
            else:
                # Try tags for known Python builtins
                node_tag = _type_to_tag(type(node))

            if node_tag is not None:
                if not uri_match(tagname, node_tag):
                    yield ValidationError(
                        f"mismatched tags, wanted '{tagname}', got '{node_tag}'"
                    )

        if not isinstance(wx_property_tag, (str, list)):
            raise WxSyntaxError(
                f"'wx_property_tag' must be str or List[str], got {wx_property_tag}"
            )

        for _, value in node.items():
            yield from _tag_validator(tagname=wx_property_tag, node=value)
