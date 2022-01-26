"""ASDF-extensions for weldx types."""
from __future__ import annotations

from asdf.extension import ManifestExtension
from asdf.resource import DirectoryResourceMapping

from .constants import (
    MANIFEST_PATH,
    SCHEMA_PATH,
    WELDX_EXTENSION_URI,
    WELDX_EXTENSION_URI_BASE,
    WELDX_SCHEMA_URI_BASE,
    WELDX_TAG_URI_BASE,
)
from .types import WeldxConverter
from .validators import wx_property_tag_validator, wx_shape_validator, wx_unit_validator


# RESOURCES ----------------------------------------------------------------------------
def get_extension_resource_mapping() -> DirectoryResourceMapping:
    """Get the weldx manifest resource mapping."""
    mapping = DirectoryResourceMapping(
        MANIFEST_PATH,
        WELDX_EXTENSION_URI_BASE,
        recursive=True,
        filename_pattern="*.yaml",
        stem_filename=True,
    )
    return mapping


def get_schema_resource_mapping() -> DirectoryResourceMapping:
    """Get the weldx schema resource mapping."""
    mapping = DirectoryResourceMapping(
        SCHEMA_PATH,
        WELDX_SCHEMA_URI_BASE,
        recursive=True,
        filename_pattern="*.yaml",
        stem_filename=True,
    )
    return mapping


def get_resource_mappings() -> list[DirectoryResourceMapping]:
    """Get list of all weldx resource mappings."""
    return [
        get_extension_resource_mapping(),
        get_schema_resource_mapping(),
    ]


# Extension ----------------------------------------------------------------------------
class WeldxExtension(ManifestExtension):
    """weldx extension class"""

    converters = (cls() for cls in WeldxConverter.__subclasses__())
    legacy_class_names = [
        "weldx.asdf.extension.WeldxAsdfExtension",
        "weldx.asdf.extension.WeldxExtension",
    ]
    yaml_tag_handles = {"!weldx!": WELDX_TAG_URI_BASE}
    validators = {  # not active yet
        "wx_property_tag": wx_property_tag_validator,
        "wx_unit": wx_unit_validator,
        "wx_shape": wx_shape_validator,
    }


def get_extensions() -> list[ManifestExtension]:
    """Get a list of all weldx extensions."""
    return [WeldxExtension.from_uri(WELDX_EXTENSION_URI)]
