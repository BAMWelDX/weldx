from pathlib import Path

import asdf
from asdf.extension import ManifestExtension
from asdf.resource import DirectoryResourceMapping

from weldx.asdf.validators import (
    wx_property_tag_validator,
    wx_shape_validator,
    wx_unit_validator,
)

from .types import WeldxConverter

# RESOURCES ----------------------------------------------------------------------------
MANIFEST_MAPPING = {
    "asdf://weldx.bam.de/weldx/extensions/weldx-1.0.0": (
        Path(__file__).parent / "manifests/weldx_manifest-1.0.0.yaml"
    ).read_text()
}


def get_schema_resource_mapping():
    # Get path to schemas directory relative to this file
    schemas_path = Path(__file__).parent / "schemas/weldx.bam.de/weldx/"
    mapping = DirectoryResourceMapping(
        schemas_path,
        "asdf://weldx.bam.de/weldx/schemas/",
        recursive=True,
        filename_pattern="*.yaml",
        stem_filename=True,
    )
    return mapping


def get_resource_mappings():
    return [MANIFEST_MAPPING, get_schema_resource_mapping()]


# for mapping in get_resource_mappings():
#     asdf.get_config().add_resource_mapping(mapping)


# Extension ----------------------------------------------------------------------------
class WeldxExtension(ManifestExtension):
    """weldx extension class"""

    extension_uri = "asdf://weldx.bam.de/weldx/extensions/weldx-1.0.0"
    yaml_tag_handles = {"!weldx!": "asdf://weldx.bam.de/weldx/tags/"}
    converters = (cls() for cls in WeldxConverter.__subclasses__())
    # asdf_standard_requirement = ">= 1.2.0, < 1.5.0"
    legacy_class_names = [
        "weldx.asdf.extension.WeldxAsdfExtension",
        "weldx.asdf.extension.WeldxExtension",
    ]
    validators = {
        "wx_property_tag": wx_property_tag_validator,
        "wx_unit": wx_unit_validator,
        "wx_shape": wx_shape_validator,
    }


def get_extensions():
    return [WeldxExtension.from_uri("asdf://weldx.bam.de/weldx/extensions/weldx-1.0.0")]
