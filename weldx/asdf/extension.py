# Licensed under a 3-clause BSD style license - see LICENSE
# -*- coding: utf-8 -*-

import os
from pathlib import Path

from asdf.extension import AsdfExtension, BuiltinExtension
from asdf.util import filepath_to_url

# Make sure that all tag implementations are imported by the time we create
# the extension class so that _weldx_asdf_types is populated correctly. We
# could do this using __init__ files, except it causes pytest import errors in
# the case that asdf is not installed.
from .tags.weldx.unit.pint_quantity import *  # noqa: F401,F403

from .types import _weldx_types, _weldx_asdf_types


__all__ = ["WeldxExtension", "WeldxAsdfExtension"]


WELDX_SCHEMA_URI_BASE = "http://weldx.bam.de/schemas/"
SCHEMA_PATH = str(Path(__file__).resolve().parents[2] / "weldx-standard" / "schemas")
# SCHEMA_PATH = os.path.abspath(
#     os.path.join(os.path.dirname(__file__), "data", "schemas")
# )
WELDX_URL_MAPPING = [
    (
        WELDX_SCHEMA_URI_BASE,
        filepath_to_url(os.path.join(SCHEMA_PATH, "weldx.bam.de"))
        + "/{url_suffix}.yaml",
    )
]


# This extension is used to register custom types that have both tags and
# schemas defined by weldx.
class WeldxExtension(AsdfExtension):
    @property
    def types(self):
        # Therer are no types yet!
        return _weldx_types

    @property
    def tag_mapping(self):
        return [("tag:weldx.bam.de:weldx", WELDX_SCHEMA_URI_BASE + "weldx{tag_suffix}")]

    @property
    def url_mapping(self):
        return WELDX_URL_MAPPING


# This extension is used to register custom tag types that have schemas defined
# by ASDF, but have tag implementations defined in the weldx package
class WeldxAsdfExtension(BuiltinExtension):
    @property
    def types(self):
        return _weldx_asdf_types
