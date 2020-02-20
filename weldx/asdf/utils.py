from pathlib import Path
import jinja2
import pandas as pd

from .extension import SCHEMA_PATH

_DTYPE_DICT = pd.DataFrame(
    data={
        "py_type": [None, str, float, int],
        "asdf_type": ["<TYPE>", "string", "number", "integer"],
    }
)


def _asdf_dtype(py_type):
    return _DTYPE_DICT.loc[_DTYPE_DICT.py_type.isin([py_type])].asdf_type.iloc[0]


_loader = jinja2.FileSystemLoader(searchpath=["./asdf/templates", "./weldx/asdf/templates"])
_env = jinja2.Environment(loader=_loader)
_env.globals.update(zip=zip)
_env.globals.update(_asdf_dtype=_asdf_dtype)


def make_asdf_schema_string(
    asdf_name,
    asdf_version,
    properties,
    property_types=None,
    required=None,
    property_order=None,
    additional_properties="false",
):
    """Generate default ASDF schema."""

    template_file = "asdf_dataclass.yaml.jinja"
    template = _env.get_template(template_file)

    output_text = template.render(
        asdf_name=asdf_name,
        asdf_version=asdf_version,
        properties=properties,
        property_types=property_types,
        required=required,
        property_order=property_order,
        additional_properties=additional_properties,
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
    required=None,
    property_order=None,
):
    """
    Create default files for a simple python dataclass asdf implementation
    """

    asdf_file_path = Path(
        SCHEMA_PATH + f"/weldx.bam.de/weldx/{asdf_name}-{asdf_version}.yaml"
    ).resolve()
    print(asdf_file_path)

    asdf_schema_string = make_asdf_schema_string(
        asdf_name, asdf_version, properties, property_types, required, property_order
    )
    with open(str(asdf_file_path), "w") as file:
        file.write(asdf_schema_string)

    python_file_path = Path(__file__ + f"/../tags/weldx/{asdf_name}.py").resolve()
    print(python_file_path)

    python_class_string = make_python_class_string(
        asdf_name, class_name, asdf_version, properties, property_types
    )
    with open(str(python_file_path), "w") as file:
        file.write(python_class_string)

    return None
