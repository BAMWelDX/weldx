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
    shape:
        Integer
    exp_string:
        String with the expected dimension

    Returns
    -------
    bool
        True or False
    """
    if ":" in exp_string:
        ranges = exp_string.split(":")
        if ranges[0] == "" and ranges[1] == "":
            return True
        else:
            if ranges[0] != "":
                return int(ranges[0]) <= _int
            if ranges[1] != "":
                return int(ranges[1]) >= _int
            if ranges[0] != "" and ranges[1] != "":
                return int(ranges[0]) <= int(ranges[1])
    else:
        return _int == int(exp_string)


def _custom_shape_validator(shape, expected_shape):
    """Validate shapes with different syntax.

    Parameters
    ----------
    shape:
        String
    expected_shape:
        String representation of the unit dimensionality to test against.

    Returs
    ------
    bool :
        True if shape matches the expected shape.
        False if shape does not match the expected shape.
    """

    # check if expected shape has right format

    # check that after one optional assign all following dimensions are optional
    # eg: "1, (1), (:), (3)" is fine, "1, (1), (:), 3" is not
    # and "(2), ..." should not be allowed too
    # replace blankspaces in strings
    expected_shape = [
        x.replace(" ", "") if isinstance(x, str) else x for x in expected_shape
    ]
    # replace None for ":" and all ~ to : in strings
    expected_shape = [
        x.replace("~", ":") if isinstance(x, str) else ":" if x is None else x
        for x in expected_shape
    ]

    # if expected shape begins with "..." or "(" string reverse expected shape and shape
    shape_length = len(shape)
    exp_shape_length = len(expected_shape)
    if str(expected_shape[0]) == "..." or "(" in str(expected_shape[0]):
        expected_shape = list(reversed(expected_shape))
        shape = list(reversed(shape))

    validator = 0
    for exp in expected_shape:
        if validator == 1:
            if "(" not in str(exp):
                raise ValueError(
                    "Optional  dimensions in the expected "
                    "shape should only stand at the end."
                )
        elif validator == 2:
            raise ValueError('After "..." should not be another dimension')
        elif "(" in str(exp):
            validator = 1
        # after "..." should not be another dimension
        elif "..." in str(exp):
            if "..." != exp:
                raise ValueError(
                    f'"..." should not have additional propterties:'
                    f" {exp} was found."
                )
            validator = 2

    for i, exp in enumerate(expected_shape):
        # if "..." is found all the following dimensions are accepted
        if "..." in str(exp):
            return True
        # if there is a parenthesis found it is an optional dimension
        elif "(" in str(exp):
            # if the shape has the optional value
            if i < shape_length:
                if isinstance(exp, str):
                    comparable = exp[exp.index("(") + 1: exp.rindex(")")]
                else:
                    comparable = str(exp)
                if not _compare(shape[i], comparable):
                    return False
        else:
            if i >= shape_length:
                return False
            if not _compare(shape[i], str(exp)):
                return False

    if shape_length > exp_shape_length:
        return False

    return True


def _another_validator(dict_test, dict_expected):
    """Validate the shape of the dictionary of two

    Parameters
    ----------
    dict_test:
        dictionary to test against
    dict_expected:
        dictionary with the expected values

    returns
    -------
    Boolean
        True or False
    """

    # keys have to match
    if dict_test.keys() != dict_expected.keys():
        return False

    dict_values = {}
    for item in dict_expected:
        # turn around the list if "..." or "(" are at the beginning
        if "(" in str(dict_expected[item][0]) or "..." in str(dict_expected[item][0]):
            dict_expected[item] = list(reversed(dict_expected[item]))
            dict_test[item] = list(reversed(dict_test[item]))
        # for loop
        for i, exp in enumerate(dict_expected[item]):
            # if "..." is found all the following dimensions are accepted
            if "..." in str(exp):
                return True

            elif "(" in str(exp):
                if i < len(dict_test[item]):
                    comparable = exp[exp.index("(") + 1: exp.rindex(")")]
                    if comparable.isalnum() and not comparable.isnumeric():
                        if comparable in dict_values:
                            if dict_test[item][i] != dict_values[comparable]:
                                return False
                        else:
                            dict_values[comparable] = dict_test[item][i]
                    elif not _compare(dict_test[item][i], comparable):
                        return False

            # all alphanumeric strings are OK - only numeric strings are not
            # eg: "n", "n1", "n1234", "myasdfstring1337"
            elif str(exp).isalnum() and not str(exp).isnumeric():
                # if value is already saved in dict_values
                if exp in dict_values:
                    # compare
                    if dict_test[item][i] != dict_values[exp]:
                        # error found
                        return False
                else:
                    # add to dict_values
                    dict_values[exp] = dict_test[item][i]
            else:
                if i >= len(dict_test[item]) or\
                        not _compare(dict_test[item][i], str(exp)):
                    return False

        if len(dict_test[item]) > len(dict_expected[item]):
            return False

    return True


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

    yield from _walk_validator(
        instance=instance,
        validator_dict=wx_shape,
        validator_function=_shape_validator,
        position=[],
        allow_missing_keys=False,
    )


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
