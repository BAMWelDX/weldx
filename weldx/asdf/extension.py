from pathlib import Path

import asdf
from asdf.extension import ManifestExtension

from weldx.asdf.validators import (
    wx_property_tag_validator,
    wx_shape_validator,
    wx_unit_validator,
)

from .types import WeldxConverter

# RESOURCES ----------------------------------------------------------------------------
RESOURCES = {
    "asdf://weldx.bam.de/weldx/extensions/weldx-1.0.0": (
        Path(__file__).parent / "manifests/weldx_manifest-1.0.0.yaml"
    ).read_text()
}


def get_resource_mappings():
    return [RESOURCES]


asdf.get_config().add_resource_mapping(RESOURCES)


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
