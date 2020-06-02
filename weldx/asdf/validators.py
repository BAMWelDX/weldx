from asdf import ValidationError

from weldx.constants import WELDX_QUANTITY as Q_
from weldx.constants import WELDX_UNIT_REGISTRY as UREG


def validate_unit_dimension(validator, wx_unit, instance, schema):
    """Custom validator for checking quantity dimensions.

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

    Returns
    -------

    Yields
    ------

    """
    print(f"called {validate_unit_dimension} with {wx_unit}")
    for prop, unit_expec in wx_unit.items():
        unit = instance[prop]["unit"]
        valid = Q_(unit).check(UREG.get_dimensionality(unit_expec))
        if not valid:
            yield ValidationError(
                f"Error validating unit dimension for property {prop} \n"
                f"expected unit of dimension {unit_expec} but got unit {unit}"
            )
    return None
