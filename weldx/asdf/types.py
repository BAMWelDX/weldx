# Licensed under a 3-clause BSD style license - see LICENSE
# -*- coding: utf-8 -*-

import functools

from asdf.types import CustomType, ExtensionTypeMeta
from boltons.iterutils import remap

META_ATTR = "wx_metadata"
USER_ATTR = "wx_user"

__all__ = [
    "WeldxType",
    "WeldxAsdfType",
    "_weldx_types",
    "_weldx_asdf_types",
    "META_ATTR",
    "USER_ATTR",
]

_weldx_types = set()
_weldx_asdf_types = set()


def to_tree_metadata(func):
    """Wrapper that will add the metadata and userdata field for to_tree methods.

    Also removes all "None" values from the initial tree.
    Behavior should be similar to ASDF defaults pre v2.8 (ASDF GH #863).
    """

    @functools.wraps(func)
    def to_tree_wrapped(cls, node, ctx):  # need cls for classmethod
        """Call default to_tree method and add metadata fields."""
        tree = func(node, ctx)

        tree = remap(tree, lambda p, k, v: v is not None)  # drop all None values

        for key in [META_ATTR, USER_ATTR]:
            attr = getattr(node, key, None)
            if attr:
                tree[key] = attr
        return tree

    return to_tree_wrapped


def from_tree_metadata(func):
    """Wrapper that will add reading metadata and userdata during form_tree methods."""

    @functools.wraps(func)
    def from_tree_wrapped(cls, tree: dict, ctx):  # need cls for classmethod
        """Call default from_tree method and add metadata attributes."""
        meta_dict = {}
        for key in [META_ATTR, USER_ATTR]:
            value = tree.pop(key, None)
            if value:
                meta_dict[key] = value

        obj = func(tree, ctx)
        for key, value in meta_dict.items():
            setattr(obj, key, value)
        return obj

    return from_tree_wrapped


class WeldxTypeMeta(ExtensionTypeMeta):
    """Metaclass to populate _weldx_types and _weldx_asdf_types."""

    def __new__(mcls, name, bases, attrs):
        cls = super().__new__(mcls, name, bases, attrs)

        if cls.organization == "weldx.bam.de" and cls.standard == "weldx":
            _weldx_types.add(cls)
        elif cls.organization == "stsci.edu" and cls.standard == "asdf":
            _weldx_asdf_types.add(cls)

        # wrap original to/from_tree method to include metadata attributes
        cls.to_tree = classmethod(to_tree_metadata(cls.to_tree))
        cls.from_tree = classmethod(from_tree_metadata(cls.from_tree))

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
