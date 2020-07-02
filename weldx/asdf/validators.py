import re
from typing import Any, Callable, Iterator, List, Mapping, OrderedDict

from asdf import ValidationError

from weldx.constants import WELDX_QUANTITY as Q_
from weldx.constants import WELDX_UNIT_REGISTRY as UREG


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
    if position is None:
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
                elif allow_missing_keys:
                    pass
                else:
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


    Return a boolean.

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
        raise ValueError("Negative dimension found")

    if ":" in exp_string:
        ranges = exp_string.split(":")

        if ranges[0] == "":
            ranges[0] = 0
        if ranges[1] == "":
            ranges[1] = _int

        if int(ranges[0]) > int(ranges[1]):
            raise ValueError(f"The range should not be descending in {exp_string}")
        return int(ranges[0]) <= _int <= int(ranges[1])

    else:
        return _int == int(exp_string)


def _prepare_list(_list, list_expected):
    """Prepare a List and an expected List for validation.

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


def _validate_expected_list(list_expected):
    """Validate an expected List and raises exceptions.


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
        if validator == 1 and "(" not in str(exp):
            raise ValueError(
                "Optional dimensions in the expected "
                "shape should only stand at the end/beginning."
            )
        if validator == 2:
            raise ValueError('After "..." should not be another dimension.')
        if "..." in str(exp):
            if "..." != exp:
                raise ValueError(
                    f'"..." should not have additional properties:' f" {exp} was found."
                )
            validator = 2
        elif "(" in str(exp):
            validator = 1


def _compare_lists(_list, list_expected):
    """Compare two lists.

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
                comparable = re.search(r"\((.*)\)", exp).group(1)
                if comparable.isalnum() and not comparable.isnumeric():
                    if comparable not in dict_values:
                        dict_values[comparable] = _list[i]
                    elif _list[i] != dict_values[comparable]:
                        return False
                elif not _compare(_list[i], comparable):
                    return False

        # all alphanumeric strings are OK - only numeric strings are not
        # eg: "n", "n1", "n1234", "myasdfstring1337"
        elif str(exp).isalnum() and not str(exp).isnumeric():
            if exp not in dict_values:
                dict_values[exp] = _list[i]
            elif _list[i] != dict_values[exp]:
                return False

        elif i >= len(_list) or not _compare(_list[i], str(exp)):
            return False

    if (len(_list) > len(list_expected)) and not has_variable_dim_num:
        return False

    return dict_values


def _custom_shape_validator(dict_test, dict_expected):
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
    # keys have to match
    # if dict_test.keys() != dict_expected.keys():
    #     return False

    # catch single shape definitions
    if isinstance(dict_expected, list):
        if "shape" not in dict_test:
            return ValidationError(f"Could not find shape key in instance {dict_test}.")
        list_test, list_expected = _prepare_list(dict_test["shape"], dict_expected)

        _validate_expected_list(list_expected)
        _dict_values = _compare_lists(list_test, list_expected)

    elif isinstance(dict_expected, dict):
        for item in dict_expected:
            # go one level deeper in the dictionary
            _dict_values = _custom_shape_validator(dict_test[item], dict_expected[item])
    else:
        raise ValueError(
            f"Found an incorrect object: {type(dict_expected)}. "
            "Should be a dict or list."
        )

    if _dict_values is False:
        return False

    dict_values = {}
    for key in _dict_values:
        if key not in dict_values:
            dict_values[key] = _dict_values[key]
        elif dict_values[key] != _dict_values[key]:
            return False

    return dict_values


def _shape_validator(
    instance: Mapping, expected_shape: List[int], position: List[str]
) -> Iterator[ValidationError]:
    """Validate the 'shape' key of the instance against the given list of integers.

    Parameters
    ----------
    instance:
        Tree serialization with 'shape' key to validate.
    expected_shape:
        String representation of the unit dimensionality to test against.
    position:
        Current position in nested structure for debugging

    Yields
    ------
    asdf.ValidationError

    """
    if not position:
        position = instance

    shape = instance["shape"]
    valid = shape == expected_shape  # TODO: custom shape validator with "any" syntax
    if not valid:
        yield ValidationError(
            f"Error validating shape for property '{position}'. "
            f"Expected shape '{expected_shape}' but got '{shape}'"
        )


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
        Enable shape validation for this schema..
    instance:
        Tree serialization (with default dtypes) of the instance
    schema:
        Dict representing the full ASDF schema.

    Yields
    ------
    asdf.ValidationError

    """

    dim_dict = _custom_shape_validator(instance, wx_shape)

    if isinstance(dim_dict, dict):
        return None
    else:
        yield ValidationError(
            f"Error validating shape {wx_shape}.\nOn instance {instance}"
        )

    # yield from _walk_validator(
    #     instance=instance,
    #     validator_dict=wx_shape,
    #     validator_function=_shape_validator,
    #     position=[],
    #     allow_missing_keys=False,
    # )


def _run_validation(
    instance, schema, validator_function, keyword_glob, allow_missing_keys
):
    import dpath

    """Gather keywords from schema and run validation along tree instance from root."""
    schema_key_list = [k for k in dpath.util.search(schema, keyword_glob, yielded=True)]
    schema_key_list = [
        (s[0].replace("properties/", "").split("/"), s[1]) for s in schema_key_list
    ]
    for s in schema_key_list:
        if len(s[0]) > 1:
            position = s[0][:-1]
            instance_dict = dpath.util.get(instance, s[0][:-1])
        else:
            position = []
            instance_dict = instance
        yield from _walk_validator(
            instance=instance_dict,
            validator_dict=s[1],
            validator_function=validator_function,
            position=position,
            allow_missing_keys=allow_missing_keys,
        )

    # old example implementation:
    # validator_function = _shape_validator
    # keyword_glob = "**/wx_shape"
    # allow_missing_keys = False
    #
    # if isinstance(wx_shape_validate, bool):
    #     enable = wx_shape_validate
    # else:
    #     raise ValueError("validator Option 'wx_shape_validate' must be true/false")
    #
    # if enable:
    #     yield from _run_validation(
    #         instance, schema, validator_function, keyword_glob, allow_missing_keys
    #     )


def debug_validator(validator, debug_validator, instance, schema):
    """Enable simple breakpoint for validation."""
    if debug_validator:
        print(f"triggered validation on schema {schema} against instance {instance}")
