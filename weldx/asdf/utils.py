from pathlib import Path
import jinja2
import pandas as pd

from .extension import SCHEMA_PATH

_DTYPE_DICT = pd.DataFrame(
    data={
        "py_type": ["str", "float", "int", "bool", "pint.Quantity"],
        "asdf_type": ["string", "number", "integer", "boolean", "unit/quantity-1.0.0"],
    }
)


def _asdf_dtype(py_type):
    if py_type.startswith("List["):
        py_type = py_type[5:-1]
    lookup = _DTYPE_DICT.py_type.isin([py_type])
    if lookup.any():
        return _DTYPE_DICT.loc[lookup].asdf_type.iloc[0]
    else:
        return f"<TODO ASDF_TYPE({py_type})>"


_DEFAULT_ASDF_DESCRIPTION = "<TODO DESCRIPTION>"

_loader = jinja2.FileSystemLoader(
    searchpath=["./asdf/templates", "./weldx/asdf/templates"]
)
_env = jinja2.Environment(loader=_loader)
_env.globals.update(zip=zip)
_env.globals.update(_asdf_dtype=_asdf_dtype)
_env.globals.update(str=str)


def make_asdf_schema_string(
    asdf_name,
    asdf_version,
    properties,
    description=None,
    property_types=None,
    required=None,
    property_order=None,
    additional_properties="false",
    schema_title=_DEFAULT_ASDF_DESCRIPTION,
    schema_description=_DEFAULT_ASDF_DESCRIPTION,
    flow_style="block",
):
    """Generate default ASDF schema."""

    if description is None:
        description = [_DEFAULT_ASDF_DESCRIPTION] * len(properties)
    description = [_DEFAULT_ASDF_DESCRIPTION if d == "" else d for d in description]

    template_file = "asdf_dataclass.yaml.jinja"
    template = _env.get_template(template_file)

    output_text = template.render(
        asdf_name=asdf_name,
        asdf_version=asdf_version,
        properties=properties,
        description=description,
        property_types=property_types,
        required=required,
        property_order=property_order,
        additional_properties=additional_properties,
        schema_title=schema_title,
        schema_description=schema_description,
        flow_style=flow_style,
    )  # this is where to put args to the template renderer

    return output_text


def make_python_class_string(
    class_name, asdf_name, asdf_version, properties, property_types
):
    """Generate default python dataclass and ASDF Type."""

    template_file = "asdf_dataclass.py.jinja"
    template = _env.get_template(template_file)

    output_text = template.render(
        class_name=class_name,
        asdf_name=asdf_name,
        asdf_version=asdf_version,
        properties=properties,
        property_types=property_types,
    )

    return output_text


def create_asdf_dataclass(
    asdf_name,
    asdf_version,
    class_name,
    properties,
    property_types=None,
    description=None,
    required=None,
    property_order=None,
    schema_title=_DEFAULT_ASDF_DESCRIPTION,
    schema_description=_DEFAULT_ASDF_DESCRIPTION,
):
    """
    Generates a ASDF schema file with corresponding python class for simple dataclasses.

    :param asdf_name: full schema name including prefixes
    :param asdf_version: schema version as string
    :param class_name: name of the Python class to generate
    :param properties: list of property names
    :param property_types: list with Python dtypes for each property
    :param description: list of property descriptions
    :param required: list of parameters that are set to required
    :param property_order: asdf schema property order
    :param schema_title: asdf schema title
    :param schema_description: asdf schema description
    :return:
    """
    asdf_file_path = Path(
        SCHEMA_PATH + f"/weldx.bam.de/weldx/{asdf_name}-{asdf_version}.yaml"
    ).resolve()
    print(asdf_file_path)

    asdf_schema_string = make_asdf_schema_string(
        asdf_name=asdf_name,
        asdf_version=asdf_version,
        properties=properties,
        description=description,
        property_types=property_types,
        required=required,
        property_order=property_order,
        schema_title=schema_title,
        schema_description=schema_description,
    )
    asdf_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(asdf_file_path), "w") as file:
        file.write(asdf_schema_string)

    python_file_path = Path(__file__ + f"/../tags/weldx/{asdf_name}.py").resolve()
    print(python_file_path)

    python_class_string = make_python_class_string(
        class_name=class_name,
        asdf_name=asdf_name,
        asdf_version=asdf_version,
        properties=properties,
        property_types=property_types,
    )
    python_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(python_file_path), "w") as file:
        file.write(python_class_string)

    return None
