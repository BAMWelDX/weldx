# Licensed under a 3-clause BSD style license - see LICENSE
# -*- coding: utf-8 -*-

from asdf.types import CustomType, ExtensionTypeMeta

__all__ = ["WeldxType", "WeldxAsdfType", "_weldx_types", "_weldx_asdf_types"]

_weldx_types = set()
_weldx_asdf_types = set()


class WeldxTypeMeta(ExtensionTypeMeta):
    """Metaclass to populate _weldx_types and _weldx_asdf_types."""

    def __new__(mcls, name, bases, attrs):
        cls = super().__new__(mcls, name, bases, attrs)

        if cls.organization == "weldx.bam.de" and cls.standard == "weldx":
            _weldx_types.add(cls)
        elif cls.organization == "stsci.edu" and cls.standard == "asdf":
            _weldx_asdf_types.add(cls)

        return cls


class WeldxType(CustomType, metaclass=WeldxTypeMeta):
    """This class represents types that have schemas and tags that are defined
    within weldx.

    """

    organization = "weldx.bam.de"
    standard = "weldx"


class WeldxAsdfType(CustomType, metaclass=WeldxTypeMeta):
    """This class represents types that have schemas that are defined in the ASDF
    standard, but have tags that are implemented within weldx.

    """

    organization = "stsci.edu"
    standard = "asdf"
