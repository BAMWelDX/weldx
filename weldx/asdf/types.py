from __future__ import annotations

import functools
import re

import asdf

if asdf.__version__ >= "3.0.0":
    from asdf.extension import SerializationContext
else:
    from asdf.asdf import SerializationContext
from asdf.extension import Converter
from boltons.iterutils import remap

from weldx.constants import META_ATTR, USER_ATTR

__all__ = [
    "WeldxConverter",
    "WxSyntaxError",
]

_new_tag_regex = re.compile(r"asdf://weldx.bam.de/weldx/tags/(.*)-(\d+.\d+.\d+|0.1.\*)")


class WxSyntaxError(Exception):
    """Exception raising on custom weldx ASDF syntax errors."""


def to_yaml_tree_metadata(func):
    """Wrapper that will add the metadata / userdata field for ``to_yaml_tree`` methods.

    Also removes all "None" values from the initial tree.
    Behavior should be similar to ASDF defaults pre v2.8 (ASDF GH #863).
    """

    @functools.wraps(func)
    def to_yaml_tree_wrapped(self, obj, tag, ctx):
        """Call default to_yaml_tree method and add metadata fields."""
        tree = func(self, obj, tag, ctx)

        for key in (META_ATTR, USER_ATTR):
            attr = getattr(obj, key, None)
            if attr:
                tree[key] = attr

        if isinstance(tree, (dict, list)):
            tree = remap(tree, lambda p, k, v: v is not None)  # drop all None values
        return tree

    return to_yaml_tree_wrapped


def from_yaml_tree_metadata(func):
    """Wrapper that will add reading metadata and userdata during form_tree methods."""

    @functools.wraps(func)
    def from_yaml_tree_wrapped(self, tree: dict | list | str, tag, ctx):
        """Call default from_yaml_tree method and add metadata attributes."""
        meta_dict = {}
        if isinstance(tree, dict):  # only valid if we serialize a dict
            for key in [META_ATTR, USER_ATTR]:
                value = tree.pop(key, None)
                if value:
                    meta_dict[key] = value

        obj = func(self, tree, tag, ctx)
        for key, value in meta_dict.items():
            setattr(obj, key, value)
        return obj

    return from_yaml_tree_wrapped


class WeldxConverterMeta(type(Converter)):
    """Metaclass to modify tree methods."""

    def __new__(mcs, name, bases, attrs):
        cls = super().__new__(mcs, name, bases, attrs)

        # legacy tag definitions
        if name := getattr(cls, "name", None):
            cls.tags = [format_tag(name, "0.1.*")]

        # wrap original to/from_yaml_tree method to include metadata attributes
        cls.to_yaml_tree = to_yaml_tree_metadata(cls.to_yaml_tree)
        cls.from_yaml_tree = from_yaml_tree_metadata(cls.from_yaml_tree)

        return cls


class WeldxConverter(Converter, metaclass=WeldxConverterMeta):
    """Base class to inherit from for custom converter classes."""

    tags: tuple[str] = None  # note: this will be updated by WeldxConverterMeta.
    types: tuple[type | str] = ()

    def to_yaml_tree(self, obj, tag: str, ctx: SerializationContext):
        raise NotImplementedError

    def from_yaml_tree(self, node: dict, tag: str, ctx: SerializationContext):
        raise NotImplementedError

    def select_tag(self, obj, tags, ctx):
        return sorted(tags)[-1]

    @classmethod
    def default_class_display_name(cls):
        """The default python class name to display for this converter."""
        if isinstance(cls.types[0], str):
            return cls.types[0].rsplit(".", 1)[-1]
        else:
            return cls.types[0].__qualname__


def format_tag(tag_name, version=None, organization="weldx.bam.de", standard="weldx"):
    """
    Format a YAML tag to new style asdf:// syntax.
    """
    tag = f"asdf://{organization}/{standard}/tags/{tag_name}"

    if version is None:
        return tag

    return f"{tag}-{version}"
