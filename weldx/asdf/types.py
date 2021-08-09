import functools

from asdf.extension import Converter
from asdf.versioning import AsdfSpec
from boltons.iterutils import remap

META_ATTR = "wx_metadata"
USER_ATTR = "wx_user"

__all__ = [
    "META_ATTR",
    "USER_ATTR",
    "WeldxConverterMeta",
    "WeldxConverter",
]


def to_yaml_tree_metadata(func):
    """Wrapper that will add the metadata and userdata field for to_tree methods.

    Also removes all "None" values from the initial tree.
    Behavior should be similar to ASDF defaults pre v2.8 (ASDF GH #863).
    """

    @functools.wraps(func)
    def to_yaml_tree_wrapped(self, obj, tag, ctx):
        """Call default to_yaml_tree method and add metadata fields."""
        tree = func(obj, tag, ctx)

        for key in [META_ATTR, USER_ATTR]:
            attr = getattr(obj, key, None)
            if attr:
                tree[key] = attr

        tree = remap(tree, lambda p, k, v: v is not None)  # drop all None values
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
    """Metaclass to modify tree methods."""

    def __new__(mcs, name, bases, attrs):
        cls = super().__new__(mcs, name, bases, attrs)

        # wrap original to/from_tree method to include metadata attributes
        cls.to_yaml_tree = classmethod(to_yaml_tree_metadata(cls.to_yaml_tree))
        cls.from_yaml_tree = classmethod(from_yaml_tree_metadata(cls.from_yaml_tree))

        return cls


class WeldxConverter(Converter, metaclass=WeldxConverterMeta):
    """Base class to inherit from for custom converter classes."""

    pass


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
