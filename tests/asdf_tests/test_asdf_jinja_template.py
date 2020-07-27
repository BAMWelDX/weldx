"""Test ASDF schema file creation with jinja template."""
import os


def test_jinja_template():
    """Test jinja template compilation with basic example."""
    from weldx.asdf.utils import create_asdf_dataclass, make_asdf_schema_string

    asdf_file_path, python_file_path = create_asdf_dataclass(
        asdf_name="custom/testclass",
        asdf_version="1.0.0",
        class_name="TestClass",
        properties=[
            "prop1",
            "prop2",
            "prop3",
            "prop4",
            "list_prop",
            "pint_prop",
            "groove_prop",
            "unkonwn_prop",
        ],
        required=["prop1", "prop2", "prop3"],
        property_order=["prop1", "prop2", "prop3"],
        property_types=[
            "str",
            "int",
            "float",
            "bool",
            "List[str]",
            "pint.Quantity",
            "VGroove",
            "unknown_type",
        ],
        description=[
            "a string",
            "",
            "a float",
            "a boolean value?",
            "a list",
            "a pint quantity",
            "a groove shape",
            "some not implemented property",
        ],
    )

    os.remove(asdf_file_path)
    os.remove(python_file_path)

    make_asdf_schema_string(
        asdf_name="custom/testclass", asdf_version="1.0.0", properties=["prop"]
    )
