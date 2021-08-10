"""Legacy asdf extension code."""

from asdf.extension import AsdfExtension, BuiltinExtension

from weldx.asdf.constants import (
    WELDX_SCHEMA_URI_BASE,
    WELDX_TAG_BASE,
    WELDX_URL_MAPPING,
)

from ._types import _weldx_asdf_types, _weldx_types

__all__ = ["WeldxExtension", "WeldxAsdfExtension"]


class WeldxExtension(AsdfExtension):
    """Extension class registering types with both tags and schemas defined by weldx."""

    @property
    def types(self):
        # There are no types yet!
        return _weldx_types

    @property
    def tag_mapping(self):
        return [(WELDX_TAG_BASE, WELDX_SCHEMA_URI_BASE + "{tag_suffix}")]

    @property
    def url_mapping(self):
        return WELDX_URL_MAPPING

    @property
    def yaml_tag_handles(self):
        return {"!weldx!": "asdf://weldx.bam.de/weldx/tags/"}


class WeldxAsdfExtension(BuiltinExtension):
    """This extension is used to register custom tag types that have schemas defined
    by ASDF, but have tag implementations defined in the weldx package

    """

    @property
    def types(self):
        return _weldx_asdf_types
