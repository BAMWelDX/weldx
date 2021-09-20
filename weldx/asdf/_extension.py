"""Legacy asdf extension code to add custom validators."""

from asdf.extension import AsdfExtension
from asdf.types import CustomType
from asdf.util import filepath_to_url

from .constants import SCHEMA_PATH, WELDX_SCHEMA_URI_BASE, WELDX_TAG_URI_BASE
from .validators import wx_property_tag_validator, wx_shape_validator, wx_unit_validator

WELDX_URL_MAPPING = [
    (
        WELDX_SCHEMA_URI_BASE,
        filepath_to_url(str(SCHEMA_PATH)) + "/{url_suffix}.yaml",
    )
]


class WeldxLegacyValidatorType(CustomType):
    """Dummy legacy class to register weldx validators using legacy asdf API."""

    organization = "weldx.bam.de"
    standard = "weldx"
    name = "legacy/validators"
    version = "0.1.0"
    types = []
    validators = {
        "wx_property_tag": wx_property_tag_validator,
        "wx_unit": wx_unit_validator,
        "wx_shape": wx_shape_validator,
    }
    versioned_siblings = []


class WeldxValidatorExtension(AsdfExtension):
    """Legacy extension class registering weldx validators."""

    @property
    def types(self):
        return [WeldxLegacyValidatorType()]

    @property
    def tag_mapping(self):
        return [(WELDX_TAG_URI_BASE, WELDX_SCHEMA_URI_BASE + "{tag_suffix}")]

    @property
    def url_mapping(self):
        return WELDX_URL_MAPPING
