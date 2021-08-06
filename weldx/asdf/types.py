# Licensed under a 3-clause BSD style license - see LICENSE
# -*- coding: utf-8 -*-

import functools

from asdf.extension import Converter
from asdf.types import CustomType, ExtensionTypeMeta
from asdf.versioning import AsdfSpec
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
    "_converters",
    "WeldxConverterMeta",
]

_weldx_types = set()
_weldx_asdf_types = set()
_converters = set()


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


def to_yaml_tree_metadata(func):
    """Wrapper that will add the metadata and userdata field for to_tree methods.

    Also removes all "None" values from the initial tree.
    Behavior should be similar to ASDF defaults pre v2.8 (ASDF GH #863).
    """

    @functools.wraps(func)
    def to_yaml_tree_wrapped(self, obj, tag, ctx):
        """Call default to_yaml_tree method and add metadata fields."""
        tree = func(obj, tag, ctx)

        tree = remap(tree, lambda p, k, v: v is not None)  # drop all None values

        for key in [META_ATTR, USER_ATTR]:
            attr = getattr(obj, key, None)
            if attr:
                tree[key] = attr
        return tree

    return to_yaml_tree_wrapped


def from_yaml_tree_metadata(func):
    """Wrapper that will add reading metadata and userdata during form_tree methods."""

    @functools.wraps(func)
    def from_yaml_tree_wrapped(cls, tree: dict, tag, ctx):
        """Call default from_yaml_tree method and add metadata attributes."""
        meta_dict = {}
        for key in [META_ATTR, USER_ATTR]:
            value = tree.pop(key, None)
            if value:
                meta_dict[key] = value

        obj = func(tree, tag, ctx)
        for key, value in meta_dict.items():
            setattr(obj, key, value)
        return obj

    return from_yaml_tree_wrapped


class WeldxConverterMeta(type(Converter)):
    """Metaclass to populate _converters and modify tree methods."""

    def __new__(mcs, name, bases, attrs):
        cls = super().__new__(mcs, name, bases, attrs)

        _converters.add(cls())  # have in instanciate class for ne api

        # if cls.organization == "weldx.bam.de" and cls.standard == "weldx":
        #     _weldx_types.add(cls)
        # elif cls.organization == "stsci.edu" and cls.standard == "asdf":
        #     _weldx_asdf_types.add(cls)

        # wrap original to/from_tree method to include metadata attributes
        cls.to_yaml_tree = classmethod(to_yaml_tree_metadata(cls.to_yaml_tree))
        cls.from_yaml_tree = classmethod(from_yaml_tree_metadata(cls.from_yaml_tree))

        return cls


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


def format_tag(tag_name, version=None, organization="weldx.bam.de", standard="weldx"):
    """
    Format a YAML tag to new style asdf:// syntax.
    """
    tag = "asdf://{0}/{1}/tags/{2}".format(organization, standard, tag_name)

    if version is None:
        return tag

    if isinstance(version, AsdfSpec):
        version = str(version.spec)

    return "{0}-{1}".format(tag, version)


class WeldxType(CustomType, metaclass=WeldxTypeMeta):
    """This class represents types that have schemas and tags that are defined
    within weldx.

    """

    organization = "weldx.bam.de"
    standard = "weldx"

    @classmethod
    def make_yaml_tag(cls, name, versioned=True):
        """
        Given the name of a type, returns a string representing its YAML tag.

        This implementation uses the new style asdf:// tag syntax seen above.

        Parameters
        ----------
        name : str
            The name of the type. In most cases this will correspond to the
            `name` attribute of the tag type. However, it is passed as a
            parameter since some tag types represent multiple custom
            types.

        versioned : bool
            If `True`, the tag will be versioned. Otherwise, a YAML tag without
            a version will be returned.

        Returns
        -------
            `str` representing the YAML tag
        """
        return format_tag(
            organization=cls.organization,
            standard=cls.standard,
            version=cls.version if versioned else None,
            tag_name=name,
        )


class WeldxAsdfType(CustomType, metaclass=WeldxTypeMeta):
    """This class represents types that have schemas that are defined in the ASDF
    standard, but have tags that are implemented within weldx.

    """

    organization = "stsci.edu"
    standard = "asdf"
