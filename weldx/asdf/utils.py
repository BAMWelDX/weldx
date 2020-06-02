from pathlib import Path

import jinja2
import pandas as pd

from weldx.asdf.constants import SCHEMA_PATH


def drop_none_attr(node):
    """Utility function simplify `to_tree` methods of  dataclass objects.

    The function requires the node to be convertible to dictionary via __dict__. And can
    therefor be applied to all classes created using the @dataclass operator.
    The result is "clean" dictionary with all None entries removed.
    A simple `to_tree()` function could be implemented as such:
        ```python
        def to_tree(cls, node, ctx):
            tree = drop_none_attr(node)
            return tree```

    Parameters
    ----------
    node :
        node to write to asdf-tree

    Returns
    -------
    dict
        The node dictionary with None entries removed.

    """
    tree = {k: v for (k, v) in node.__dict__.items() if v is not None}
    return tree


# jinja template functions ---------------------------------------

_DTYPE_DICT = pd.DataFrame(
    data={
        "py_type": [
            "str",
            "float",
            "int",
            "bool",
            "pint.Quantity",
            "VGroove",
            "UGroove",
        ],
        "asdf_type": [
            "string",
            "number",
            "integer",
            "boolean",
            "tag:stsci.edu:asdf/unit/quantity-1.1.0",
            "tag:weldx.bam.de:weldx/core/iso_groove-1.0.0",
            "tag:weldx.bam.de:weldx/core/iso_groove-1.0.0",
        ],
    }
)


def _asdf_dtype(py_type):
    prefix = "type: "

    # strip list
    if py_type.startswith("List["):
        py_type = py_type[5:-1]

    lookup = _DTYPE_DICT.py_type.isin([py_type])
    if lookup.any():
        asdf_type = _DTYPE_DICT.loc[lookup].asdf_type.iloc[0]
        if ("tag:" in asdf_type) or ("-" in asdf_type):
            prefix = "$ref: "
        return prefix + asdf_type

    return f"{prefix}<TODO ASDF_TYPE({py_type})>"


_DEFAULT_ASDF_DESCRIPTION = "<TODO DESCRIPTION>"

_loader = jinja2.FileSystemLoader(
    searchpath=["./asdf/templates", "./weldx/asdf/templates"]
)
_env = jinja2.Environment(loader=_loader, autoescape=True)
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
    """Generate default ASDF schema.

    Parameters
    ----------
    asdf_name :

    asdf_version :

    properties :

    description :
         (Default value = None)
    property_types :
         (Default value = None)
    required :
         (Default value = None)
    property_order :
         (Default value = None)
    additional_properties :
         (Default value = "false")
    schema_title :
         (Default value = _DEFAULT_ASDF_DESCRIPTION)
    schema_description :
         (Default value = _DEFAULT_ASDF_DESCRIPTION)
    flow_style :
         (Default value = "block")

   """

    if property_types is None:
        property_types = ["NO_TYPE"] * len(properties)

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
    class_name, asdf_name, asdf_version, properties, property_types, required
):
    """Generate default python dataclass and ASDF Type.

    Parameters
    ----------
    class_name :

    asdf_name :

    asdf_version :

    properties :

    property_types :

    required :


   """

    template_file = "asdf_dataclass.py.jinja"
    template = _env.get_template(template_file)

    lib_imports = {
        dtype.split(".")[0] for dtype in property_types if len(dtype.split(".")) > 1
    }

    output_text = template.render(
        class_name=class_name,
        asdf_name=asdf_name,
        asdf_version=asdf_version,
        properties=properties,
        property_types=property_types,
        required=required,
        lib_imports=lib_imports,
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
    """Generates a ASDF schema file with corresponding python class for simple dataclasses.

    Parameters
    ----------
    asdf_name :
        full schema name including prefixes
    asdf_version :
        schema version as string
    class_name :
        name of the Python class to generate
    properties :
        list of property names
    property_types :
        list with Python dtypes for each property (Default value = None)
    description :
        list of property descriptions (Default value = None)
    required :
        list of parameters that are set to required (Default value = None)
    property_order :
        asdf schema property order (Default value = None)
    schema_title :
        asdf schema title (Default value = _DEFAULT_ASDF_DESCRIPTION)
    schema_description :
        asdf schema description (Default value = _DEFAULT_ASDF_DESCRIPTION)

   """
    asdf_file_path = Path(
        SCHEMA_PATH + f"/weldx.bam.de/weldx/{asdf_name}-{asdf_version}.yaml"
    ).resolve()

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

    python_class_string = make_python_class_string(
        class_name=class_name,
        asdf_name=asdf_name,
        asdf_version=asdf_version,
        properties=properties,
        property_types=property_types,
        required=required,
    )
    python_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(python_file_path), "w") as file:
        file.write(python_class_string)

    return asdf_file_path, python_file_path
