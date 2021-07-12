import re
from typing import Any, Callable, Dict, Iterator, List, Mapping, OrderedDict, Union

import asdf
from asdf import ValidationError
from asdf.schema import _type_to_tag
from asdf.tagged import TaggedDict
from asdf.util import uri_match

from weldx.asdf.extension import WxSyntaxError
from weldx.asdf.tags.weldx.time.datetimeindex import DatetimeIndexType
from weldx.asdf.tags.weldx.time.timedeltaindex import TimedeltaIndexType
from weldx.constants import WELDX_QUANTITY as Q_
from weldx.constants import WELDX_UNIT_REGISTRY as UREG
from weldx.util import deprecated


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
    asdf.ValidationError

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
    instance: Mapping, expected_dimensionality: str, position: List[str]
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
    asdf.ValidationError

    """
    if not position:
        position = instance

    unit = instance["unit"]
    valid = Q_(unit).check(UREG.get_dimensionality(expected_dimensionality))
    if not valid:
        yield ValidationError(
            f"Error validating unit dimension for property '{position}'. "
            f"Expected unit of dimension '{expected_dimensionality}' "
            f"but got unit '{unit}'"
        )


def _compare(_int, exp_string):
    """Compare helper of two strings for _custom_shape_validator.

    An integer and an expected string are compared so that the string either contains
    a ":" and thus describes an interval or a string consisting of numbers. So if our
    integer is within the interval or equal to the described number, True is returned.
    The interval can be open, in that there is no number left or right of the ":"
    symbol.

    Examples:
    ---------

    _int = 5
    exp_string = "5"
    -> True

    _int = 5
    exp_string = ":"
    -> True

    Open interval:
    _int = 5
    exp_string = "3:"
    -> True

    _int = 5
    exp_string = "4"
    -> False

    _int = 5
    exp_string = "6:8"
    -> False

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
    """
    if _int < 0:
        raise WxSyntaxError("Negative dimension found")

    if ":" in exp_string:
        ranges = exp_string.split(":")

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
    white spaces and rewriting all lists that contain the symbols ":" as well as "~" to
    a ":". In addition, lists that begin with "..." or parentheses are reversed for
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
    # accept "~" additionally as input of ":". And remove blank spaces.
    list_expected = [
        x.replace(" ", "").replace("~", ":") if isinstance(x, str) else x
        for x in list_expected
    ]
    # turn around the list if "..." or "(" are at the beginning.
    # because the validation is made from begin -> end.
    # like this we validate the array from end -> begin.
    if "(" in str(list_expected[0]) or "..." in str(list_expected[0]):
        list_expected = list(reversed(list_expected))
        _list = list(reversed(_list))

    return _list, list_expected


def _is_range_format_valid(format_string: str):
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
    if ":" in format_string:
        if len(format_string.split(":")) != 2:
            return False
        format_string = format_string.replace(":", "")
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

    params
    ------
    list_expected:
        Expected List to validate against

    raises
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
                    f'"..." should not have additional properties:' f" {exp} was found."
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
                    f'Invalid optional dimension format. Correct format is "(_)", but '
                    f" {exp} was found."
                )

            validator = 1
        elif not _is_range_format_valid(str(exp)):
            raise WxSyntaxError(
                f"{exp} is an invalid range format."
                f"Consult the documentation for a list of all valid options"
            )


def _compare_lists(_list, list_expected):
    """Compare two lists.

    The two lists are interpreted as a list of dimensions. We compare the dimensions of
    these lists and in the case of a mismatch, False is returned. If the dimensions
    match and there are variables in the list_expected, they are entered into a
    dictionary and this is output. The dictionary can be empty if there are no
    variables in the list_expected.

    Examples:
    ---------
    _compare_lists([1, 2, 3], [1, 2, 3])
    -> {}

    _compare_lists([1, 2, 3], [1, n1, n2])
    -> {n1: 2, n2: 3}

    _compare_lists([1, 2, 3], [1, "..."])
    -> {}

    _compare_lists([1, 2, 3], [1, 2, 4])
    -> False

    params
    ------
    _list:
        List of Integer
    list_expected:
        List build by the rules in _custom_shape_validator
    returns
    -------
    False:
        when a dimension mismatch occurs
    dict_values:
        when no dimension mismatch occurs. Can be empty {}. Dictionary - keys: variable
        names in the validation schemes. values: values of the validation schemes.
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


def _get_instance_shape(instance_dict: Union[TaggedDict, Dict[str, Any]]) -> List[int]:
    """Get the shape of an ASDF instance from its tagged dict form."""
    if isinstance(instance_dict, (float, int)):  # test against [1] for scalar values
        return [1]
    elif "shape" in instance_dict:
        return instance_dict["shape"]
    elif isinstance(instance_dict, asdf.types.tagged.Tagged):
        # add custom type implementations
        if "weldx/time/timedeltaindex" in instance_dict._tag:
            return TimedeltaIndexType.shape_from_tagged(instance_dict)
        elif "weldx/time/datetimeindex" in instance_dict._tag:
            return DatetimeIndexType.shape_from_tagged(instance_dict)
        elif "weldx/core/time_series" in instance_dict._tag:
            return [1]  # scalar
        elif "asdf/unit/quantity" in instance_dict._tag:
            if isinstance(instance_dict["value"], dict):  # ndarray
                return _get_instance_shape(instance_dict["value"])
            return [1]  # scalar
        elif "weldx/core/variable" in instance_dict._tag:
            return _get_instance_shape(instance_dict["data"])

    return None


def _custom_shape_validator(dict_test: Dict[str, Any], dict_expected: Dict[str, Any]):
    """Validate dimensions which are stored in two dictionaries dict_test and
    dict_expected.

    Syntax for the dict_expected:
    -----------------------------
    Items with arrays with each value having the following Syntax:
    1)  3 : an integer indicates a fix dimension for the same item in dict_test
    2)  "~", ":" or None : this string indicates a single dimension of arbitrary length.
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
        dictionary to test against
    dict_expected:
        dictionary with the expected values

    raises
    ------
    ValueError:
        when dict_expected does violate against the Syntax rules

    returns
    -------
    False
        when any dimension mismatch occurs
    dict_values
        Dictionary - keys: variable names in the validation schemes. values: values of
        the validation schemes.
    """

    dict_values = {}

    # catch single shape definitions
    if isinstance(dict_expected, list):
        # get the shape of current
        shape = _get_instance_shape(dict_test)
        if not shape:
            raise ValidationError(f"Could not determine shape in instance {dict_test}.")

        list_test, list_expected = _prepare_list(shape, dict_expected)

        _validate_expected_list(list_expected)
        _dict_values = _compare_lists(list_test, list_expected)

        if _dict_values is False:
            raise ValidationError(
                f"Shape {list_test[::-1]} does not match requirement "
                f"{list_expected[::-1]}"
            )

        return _dict_values

    elif isinstance(dict_expected, dict):
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
                _dict_values = _custom_shape_validator(dict_test[key], item)
            elif _optional:
                _dict_values = {}
            else:
                raise ValidationError(
                    f"Could not access key '{key}'  in instance {dict_test}."
                )

            for key in _dict_values:
                if key not in dict_values:
                    dict_values[key] = _dict_values[key]
                elif dict_values[key] != _dict_values[key]:
                    return False
    else:
        raise WxSyntaxError(
            f"Found an incorrect object: {type(dict_expected)}. "
            "Should be a dict or list."
        )

    return dict_values


def wx_unit_validator(
    validator, wx_unit, instance, schema
) -> Iterator[ValidationError]:
    """Custom validator for checking dimensions for objects with 'unit' property.

    ASDF documentation:
    https://asdf.readthedocs.io/en/2.6.0/asdf/extensions.html#adding-custom-validators

    Parameters
    ----------
    validator:
        A jsonschema.Validator instance.
    wx_unit:
        Enable unit validation for this schema.
    instance:
        Tree serialization (with default dtypes) of the instance
    schema:
        Dict representing the full ASDF schema.

    Yields
    ------
    asdf.ValidationError

    """
    yield from _walk_validator(
        instance=instance,
        validator_dict=wx_unit,
        validator_function=_unit_validator,
        position=[],
        allow_missing_keys=False,
    )


def wx_shape_validator(
    validator, wx_shape, instance, schema
) -> Iterator[ValidationError]:
    """Custom validator for checking dimensions for objects with 'shape' property.

    ASDF documentation:
    https://asdf.readthedocs.io/en/2.6.0/asdf/extensions.html#adding-custom-validators

    Parameters
    ----------
    validator:
        A jsonschema.Validator instance.
    wx_shape:
        Enable shape validation for this schema.
    instance:
        Tree serialization (with default dtypes) of the instance
    schema:
        Dict representing the full ASDF schema.

    Yields
    ------
    asdf.ValidationError

    """

    dim_dict = None
    try:
        dim_dict = _custom_shape_validator(instance, wx_shape)
    except ValidationError:
        yield ValidationError(
            f"Error validating shape {wx_shape}.\nOn instance {instance}"
        )

    if isinstance(dim_dict, dict):
        return None
    else:
        yield ValidationError(
            f"Error validating shape {wx_shape}.\nOn instance {instance}"
        )


@deprecated("0.4.0", "0.5.0", " _compare_tag_version will be removed in 0.5.0")
def _compare_tag_version(instance_tag: str, tagname: str):
    """Compare ASDF tag-strings with flexible version syntax.

    Parameters
    ----------
    instance_tag:
        the full ASDF tag to validate
    tagname:
        tag string with custom version syntax to validate against

    Returns
    -------
        bool
    """
    if instance_tag is None:
        return True

    if instance_tag.startswith("tag:yaml.org"):  # test for python builtins
        return instance_tag == tagname
    instance_tag_version = [int(v) for v in instance_tag.rpartition("-")[-1].split(".")]

    tag_parts = tagname.rpartition("-")
    tag_uri = tag_parts[0]
    tag_version = [v for v in tag_parts[-1].split(".")]

    if tag_version == ["*"]:
        version_compatible = True
    elif all([vstr.isdigit() for vstr in tag_version]):
        vnum = [int(vstr) for vstr in tag_version]
        version_compatible = all(
            [v[0] == v[1] for v in zip(vnum, instance_tag_version)]
        )
    else:
        raise WxSyntaxError(f"Unknown wx_tag syntax {tagname}")

    if (not instance_tag.startswith(tag_uri)) or (not version_compatible):
        return False
    return True


def wx_tag_validator(validator, tagname, instance, schema):
    """Validate instance tag string with flexible version syntax.

    The following syntax is allowed to validate against:

    wx_tag: http://stsci.edu/schemas/asdf/core/software-* # allow every version
    wx_tag: http://stsci.edu/schemas/asdf/core/software-1 # fix major version
    wx_tag: http://stsci.edu/schemas/asdf/core/software-1.2 # fix minor version
    wx_tag: http://stsci.edu/schemas/asdf/core/software-1.2.3 # fix patch version

    Parameters
    ----------
    validator:
        A jsonschema.Validator instance.
    tagname:
        tag string with custom version syntax to validate against
    instance:
        Tree serialization (with default dtypes) of the instance
    schema:
        Dict representing the full ASDF schema.

    Returns
    -------
        bool

    """
    if hasattr(instance, "_tag"):
        instance_tag = instance._tag
    else:
        # Try tags for known Python builtins
        instance_tag = _type_to_tag(type(instance))

    if instance_tag is not None:
        if not uri_match(tagname, instance_tag):
            yield ValidationError(
                "mismatched tags, wanted '{0}', got '{1}'".format(tagname, instance_tag)
            )


def wx_property_tag_validator(
    validator, wx_property_tag: str, instance, schema
) -> Iterator[ValidationError]:
    """

    Parameters
    ----------
    validator
        A jsonschema.Validator instance.
    wx_property_tag
        The tag to test all object properties against.
    instance
        Tree serialization (with default dtypes or as tagged dict) of the instance
    schema
        Dict representing the full ASDF schema.

    Yields
    ------
    asdf.ValidationError

    """
    for _, value in instance.items():
        yield from wx_tag_validator(
            validator, tagname=wx_property_tag, instance=value, schema=None
        )
