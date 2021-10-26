import functools
import re
from copy import copy
from typing import List, Union

from asdf.asdf import SerializationContext
from asdf.extension import Converter
from asdf.versioning import AsdfSpec
from boltons.iterutils import remap

META_ATTR = "wx_metadata"
USER_ATTR = "wx_user"

__all__ = [
    "META_ATTR",
    "USER_ATTR",
    "WeldxConverter",
    "WxSyntaxError",
]

_new_tag_regex = re.compile(r"asdf://weldx.bam.de/weldx/tags/(.*)-(\d+.\d+.\d+|0.1.\*)")


class WxSyntaxError(Exception):
    """Exception raising on custom weldx ASDF syntax errors."""


def to_yaml_tree_metadata(func):
    """Wrapper that will add the metadata and userdata field for ``to_yaml_tree`` methods.

    Also removes all "None" values from the initial tree.
    Behavior should be similar to ASDF defaults pre v2.8 (ASDF GH #863).
    """

    @functools.wraps(func)
    def to_yaml_tree_wrapped(self, obj, tag, ctx):
        """Call default to_yaml_tree method and add metadata fields."""
        tree = func(self, obj, tag, ctx)

        for key in [META_ATTR, USER_ATTR]:
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
    def from_yaml_tree_wrapped(self, tree: Union[dict, list, str], tag, ctx):
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
            setattr(
                cls,
                "tags",
                [format_tag(name, "0.1.*")],
            )

        # wrap original to/from_yaml_tree method to include metadata attributes
        cls.to_yaml_tree = to_yaml_tree_metadata(cls.to_yaml_tree)
        cls.from_yaml_tree = from_yaml_tree_metadata(cls.from_yaml_tree)

        for tag in copy(cls.tags):  # legacy_code
            if tag.startswith("asdf://weldx.bam.de/weldx/tags/"):
                cls.tags.append(_legacy_tag_from_new_tag(tag))

        return cls


class WeldxConverter(Converter, metaclass=WeldxConverterMeta):
    """Base class to inherit from for custom converter classes."""

    tags: List[str] = []
    types: List[Union[type, str]] = []

    def to_yaml_tree(self, obj, tag: str, ctx: SerializationContext):
        raise NotImplementedError

    def from_yaml_tree(self, node: dict, tag: str, ctx: SerializationContext):
        raise NotImplementedError

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

    if isinstance(version, AsdfSpec):
        version = str(version.spec)

    return f"{tag}-{version}"


def _legacy_tag_from_new_tag(tag: str):
    name, version = _new_tag_regex.search(tag).groups()
    version = "1.0.0"  # legacy_tag version
    return f"tag:weldx.bam.de:weldx/{name}-{version}"
