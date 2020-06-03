from asdf import ValidationError
from typing import Any, OrderedDict, Mapping, Iterator, Callable

from weldx.constants import WELDX_QUANTITY as Q_
from weldx.constants import WELDX_UNIT_REGISTRY as UREG


def _walk_validator(
    instance: OrderedDict,
    validator_dict: OrderedDict,
    validator_function: Callable[[Mapping, Any, str], Iterator[ValidationError]],
    position="",
):
    """Walk instance and validation dict entries in parallel and apply a validator func.

    This function can be used to recursively walk both the instance dictionary and the
    custom validation dictionary in parallel and

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

    Yields
    ------
    asdf.ValidationError

    """
    for key, item in validator_dict.items():
        if isinstance(item, Mapping):
            yield from _walk_validator(
                instance[key],
                validator_dict[key],
                validator_function,
                position=position + "/" + key,
            )
        else:
            yield from validator_function(instance[key], item, position + "/" + key)


def _unit_validator(instance: OrderedDict, expected_dimensionality: str, position: str):
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
    unit = instance["unit"]
    valid = Q_(unit).check(UREG.get_dimensionality(expected_dimensionality))
    if not valid:
        yield ValidationError(
            f"Error validating unit dimension for property '{position}'\n"
            f"expected unit of dimension '{expected_dimensionality}' but got unit '{unit}'"
        )


def _shape_validator(instance: OrderedDict, expected_shape: str, position: str):
    """Validate the 'shape' key of the instance against the given string.

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
    shape = instance["shape"]
    valid = shape == expected_shape
    if not valid:
        yield ValidationError(
            f"Error validating unit dimension for property '{position}'\n"
            f"expected unit of dimension '{expected_shape}' but got unit '{shape}'"
        )


def validate_unit_dimension(validator, wx_unit, instance, schema):
    """Custom validator for checking dimensions for objects with 'unit' property.

    ASDF documentation:
    https://asdf.readthedocs.io/en/2.6.0/asdf/extensions.html#adding-custom-validators

    Parameters
    ----------
    validator:
        A jsonschema.Validator instance.
    wx_unit:
        Dict with property keys and unit dimensions to validate.
    instance:
        Tree serialization (with default dtypes) of the instance
    schema:
        Dict representing the full ASDF schema.

    Yields
    ------
    asdf.ValidationError

    """
    yield from _walk_validator(
        instance=instance, validator_dict=wx_unit, validator_function=_unit_validator
    )


def validate_array_shape(validator, wx_shape, instance, schema):
    yield from _walk_validator(
        instance=instance, validator_dict=wx_shape, validator_function=_shape_validator,
    )
