# Licensed under a 3-clause BSD style license - see LICENSE
# -*- coding: utf-8 -*-

from asdf.extension import AsdfExtension, BuiltinExtension

from weldx.asdf.constants import (
    WELDX_SCHEMA_URI_BASE,
    WELDX_TAG_BASE,
    WELDX_URL_MAPPING,
)

from .types import _weldx_asdf_types, _weldx_types

__all__ = ["WeldxExtension", "WeldxAsdfExtension"]


class WxSyntaxError(Exception):
    """Exception raising on custom weldx ASDF syntax errors."""


class WeldxExtension(AsdfExtension):
    """Extension class registering types with both tags and schemas defined by weldx."""

    @property
    def types(self):
        # There are no types yet!
        return _weldx_types

    @property
    def tag_mapping(self):
        return [(WELDX_TAG_BASE, WELDX_SCHEMA_URI_BASE + "weldx{tag_suffix}")]

    @property
    def url_mapping(self):
        return WELDX_URL_MAPPING

    @property
    def yaml_tag_handles(self):
        return {"!weldx!": "tag:weldx.bam.de:weldx/"}


class WeldxAsdfExtension(BuiltinExtension):
    """This extension is used to register custom tag types that have schemas defined
    by ASDF, but have tag implementations defined in the weldx package

    """

    @property
    def types(self):
        return _weldx_asdf_types
