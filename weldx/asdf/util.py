"""Utilities for asdf files."""

from __future__ import annotations

import importlib.metadata
from collections.abc import Callable, Hashable, Mapping, MutableMapping, Set
from contextlib import contextmanager
from io import BytesIO, TextIOBase
from pathlib import Path
from typing import Any
from warnings import warn

import asdf
import pint

if asdf.__version__ >= "3.0.0":
    from asdf.extension import SerializationContext
else:
    from asdf.asdf import SerializationContext
from asdf.config import AsdfConfig, get_config
from asdf.extension import Extension
from asdf.tagged import TaggedDict, TaggedList, TaggedString
from asdf.util import uri_match as asdf_uri_match
from boltons.iterutils import get_path, remap
from packaging.version import Version

from weldx.asdf.constants import SCHEMA_PATH, WELDX_EXTENSION_URI
from weldx.asdf.types import WeldxConverter
from weldx.constants import U_, UNITS_KEY, WELDX_UNIT_REGISTRY
from weldx.types import (
    SupportsFileReadWrite,
    types_file_like,
    types_path_and_file_like,
    types_path_like,
)

_USE_WELDX_FILE = False
_INVOKE_SHOW_HEADER = False


__all__ = [
    "get_schema_path",
    "read_buffer",
    "read_buffer_context",
    "write_buffer",
    "write_read_buffer",
    "write_read_buffer_context",
    "get_yaml_header",
    "view_tree",
    "notebook_fileprinter",
    "dataclass_serialization_class",
]


def get_schema_path(schema: str) -> Path:  # pragma: no cover
    """Get the path to a weldx schema file.

    Parameters
    ----------
    schema :
        Name of the schema file
    Returns
    -------
    pathlib.Path
        Path to the requested schema file in the current filesystem.

    """
    schema = schema.split(".yaml")[0]

    p = SCHEMA_PATH
    schemas = list(p.glob(f"**/{schema}.yaml"))
    if len(schemas) == 0:
        raise ValueError(f"No matching schema for filename '{schema}'.")
    elif len(schemas) > 1:
        warn(
            f"Found more than one matching schema for filename '{schema}'.",
            stacklevel=1,
        )
    return schemas[0]


# asdf read/write debug tools functions ---------------------------------------


def write_buffer(
    tree: dict,
    asdffile_kwargs: dict = None,
    write_kwargs: dict = None,
    _use_weldx_file=None,
    _invoke_show_header=None,
) -> BytesIO:
    """Write ASDF file into buffer.

    Parameters
    ----------
    tree:
        Tree object to serialize.
    asdffile_kwargs
        Additional keywords to pass to `asdf.AsdfFile`
    write_kwargs
        Additional keywords to pass to `asdf.AsdfFile.write_to`
        Weldx-Extensions are always set.

    Returns
    -------
    io.BytesIO
        Bytes buffer of the ASDF file.
    """
    if asdffile_kwargs is None:
        asdffile_kwargs = {}
    if write_kwargs is None:
        write_kwargs = {}

    if _use_weldx_file is None:
        _use_weldx_file = _USE_WELDX_FILE

    if _invoke_show_header is None:
        _invoke_show_header = _INVOKE_SHOW_HEADER

    def show(wx):
        if _invoke_show_header:
            wx.header(False, False)

    if _use_weldx_file:
        from weldx.asdf.file import WeldxFile

        with WeldxFile(
            tree=tree,
            asdffile_kwargs=asdffile_kwargs,
            write_kwargs=write_kwargs,
            mode="rw",
        ) as wx:
            show(wx)
            buff = wx.file_handle
    else:
        buff = BytesIO()
        with asdf.AsdfFile(tree, extensions=None, **asdffile_kwargs) as ff:
            ff.write_to(buff, **write_kwargs)
    buff.seek(0)
    return buff


@contextmanager
def read_buffer_context(
    buffer: BytesIO,
    open_kwargs: dict = None,
    _use_weldx_file=_USE_WELDX_FILE,
):
    """Contextmanager to read ASDF file contents from buffer instance.

    Parameters
    ----------
    buffer : io.BytesIO
        Buffer containing ASDF file contents
    open_kwargs
        Additional keywords to pass to `asdf.AsdfFile.open`
        Extensions are always set, ``memmap=False`` is set by default.

    Returns
    -------
    dict
        ASDF file tree.

    """
    if open_kwargs is None:
        open_kwargs = {"memmap": False, "lazy_load": False}

    if "memmap" in open_kwargs and tuple(
        importlib.metadata.version("asdf").split(".")
    ) < ("3", "1", "0"):
        open_kwargs["copy_arrays"] = not open_kwargs["memmap"]
        del open_kwargs["memmap"]

    buffer.seek(0)

    if _use_weldx_file is None:
        _use_weldx_file = _USE_WELDX_FILE
    if _use_weldx_file:
        from weldx.asdf.file import WeldxFile

        return WeldxFile(buffer, asdffile_kwargs=open_kwargs)

    with asdf.open(
        buffer,
        extensions=None,
        **open_kwargs,
    ) as af:
        yield af.tree


def read_buffer(
    buffer: BytesIO,
    open_kwargs: dict = None,
    _use_weldx_file=_USE_WELDX_FILE,
):
    """Read ASDF file contents from buffer instance.

    Parameters
    ----------
    buffer : io.BytesIO
        Buffer containing ASDF file contents
    open_kwargs
        Additional keywords to pass to `asdf.AsdfFile.open`
        Extensions are always set, ``memmap=False`` is set by default.

    Returns
    -------
    dict
        ASDF file tree.

    """
    with read_buffer_context(buffer, open_kwargs, _use_weldx_file) as data:
        return data


@contextmanager
def write_read_buffer_context(
    tree: dict, asdffile_kwargs=None, write_kwargs=None, open_kwargs=None
):
    """Context manager to perform a buffered write/read roundtrip of a tree
    using default ASDF settings.

    Parameters
    ----------
    tree
        Tree object to serialize.
    asdffile_kwargs
        Additional keywords to pass to `asdf.AsdfFile`
    write_kwargs
        Additional keywords to pass to `asdf.AsdfFile.write_to`
        Extensions are always set.
    open_kwargs
        Additional keywords to pass to `asdf.AsdfFile.open`
        Extensions are always set, ``memmap=False`` is set by default.

    Returns
    -------
    dict

    """
    buffer = write_buffer(tree, asdffile_kwargs, write_kwargs)
    with read_buffer_context(buffer, open_kwargs) as data:
        yield data


def write_read_buffer(
    tree: dict, asdffile_kwargs=None, write_kwargs=None, open_kwargs=None
):
    """Perform a buffered write/read roundtrip of a tree using default ASDF settings.

    Parameters
    ----------
    tree
        Tree object to serialize.
    asdffile_kwargs
        Additional keywords to pass to `asdf.AsdfFile`
    write_kwargs
        Additional keywords to pass to `asdf.AsdfFile.write_to`
        Extensions are always set.
    open_kwargs
        Additional keywords to pass to `asdf.AsdfFile.open`
        Extensions are always set, ``memmap=False`` is set by default.

    Returns
    -------
    dict

    """

    with write_read_buffer_context(
        tree, asdffile_kwargs, write_kwargs, open_kwargs
    ) as data:
        return data


def get_yaml_header(file: types_path_and_file_like, parse=False) -> str | dict:
    """Read the YAML header part (excluding binary sections) of an ASDF file.

    Parameters
    ----------
    file :
        a path or file-like type pointing to a ASDF file.

    parse :
        if `True`, returns the interpreted YAML header as dict.

    Returns
    -------
    str, dict
        The YAML header as string the ASDF file, if parse is False. Or if parse is True,
        return the parsed header.

    """

    def read_header(handle):
        # reads lines until the line "...\n" is reached.
        def readline_replace_eol():
            line = handle.readline()
            if (not line) or (line in {b"...\n", b"...\r\n"}):
                raise StopIteration
            return line

        return b"".join(iter(readline_replace_eol, None))

    if isinstance(file, types_file_like.__args__):
        if isinstance(file, TextIOBase):
            raise ValueError(
                "cannot read files opened in text mode. Please open in binary mode."
            )
        if isinstance(file, SupportsFileReadWrite):
            file.seek(0)
        code = read_header(file)
    elif isinstance(file, types_path_like.__args__):
        with open(file, "rb") as f:
            code = read_header(f)
    else:
        raise TypeError(f"cannot read yaml header from {type(file)}.")

    if parse:
        return asdf.yamlutil.load_tree(code)
    return code.decode("utf-8")


def notebook_fileprinter(file: types_path_and_file_like, lexer="YAML"):
    """Print the code from file/BytesIO to notebook cell with syntax highlighting.

    Parameters
    ----------
    file :
        filename or file-like object pointing towards / containing an ASDF file.
    lexer :
        Syntax style to use

    """
    from IPython.display import HTML
    from pygments import highlight
    from pygments.formatters.html import HtmlFormatter
    from pygments.lexers import get_lexer_by_name, get_lexer_for_filename

    if isinstance(file, types_file_like.__args__):
        lexer = get_lexer_by_name(lexer)
    elif Path(file).suffix == ".asdf":
        lexer = get_lexer_by_name("YAML")
    else:
        lexer = get_lexer_for_filename(file)

    code = get_yaml_header(file, parse=False)
    formatter = HtmlFormatter()
    return HTML(
        '<style type="text/css">{}</style>{}'.format(
            formatter.get_style_defs(".highlight"),
            highlight(code, lexer, formatter),
        )
    )


def view_tree(file: types_path_and_file_like, path: tuple = None, **kwargs):
    """Display YAML header using IPython JSON display repr.

    This function works in JupyterLab.

    Parameters
    ----------
    file :
        filename or file-like object pointing towards / containing an ASDF file.
    path :
        tuple representing the lookup path in the yaml/asdf tree
    kwargs
        kwargs passed down to JSON constructor

    Returns
    -------
    IPython.display.JSON
        JSON object for rich output in JupyterLab

    Examples
    --------
    Visualize the full tree of an existing ASDF file::

        weldx.asdf.utils.view_tree("single_pass_weld_example.asdf")

    Visualize a specific element in the tree structure by proving the path::

        weldx.asdf.utils.view_tree(
            "single_pass_weld_example.asdf", path=("process",)
        )

        weldx.asdf.utils.view_tree(
            "single_pass_weld_example.asdf", path=("process", "welding_process")
        )

    """
    from IPython.display import JSON

    def _visit(p, k, v):
        """Convert tagged types to base types."""
        if isinstance(v, TaggedDict):
            return k, dict(v)
        if isinstance(v, TaggedList):
            return k, list(v)
        if isinstance(v, TaggedString):
            return k, str(v)
        return k, v

    if isinstance(file, str):
        root = file + "/"
    else:
        root = "/"

    yaml_dict = get_yaml_header(file, parse=True)
    yaml_dict = dict(remap(yaml_dict, _visit))
    if path:
        root = root + "/".join(path)
        yaml_dict = get_path(yaml_dict, path)
    kwargs["root"] = root
    return JSON(yaml_dict, **kwargs)


def _fullname(obj):
    """Get the fully qualified class name of an object."""
    if isinstance(obj, str):
        return obj

    cls = obj.__class__
    module = cls.__module__
    if module == "builtins":  # no need for builtin prefix
        return cls.__qualname__
    return module + "." + cls.__qualname__


def dataclass_serialization_class(
    class_type: type,
    class_name: str,
    version: str,
    to_yaml_tree_mod: Callable = None,
    from_yaml_tree_mod: Callable = None,
    sort_string_lists: bool = True,
) -> type:
    """Generate a asdf serialization class for a python dataclass.

    Parameters
    ----------
    class_type :
        The type of the dataclass
    class_name :
        The value that should be stored as the classes name property
    version :
        The version number
    to_yaml_tree_mod :
        A method that applies additional modifications to the tree during the
        ``to_yaml_tree`` function call
    from_yaml_tree_mod :
        A method that applies additional modifications to the tree during the
        ``from_yaml_tree`` function call
    sort_string_lists :
        Sort string lists before serialization.

    Returns
    -------
    Type :
        A new asdf serialization class.

    """
    v = version

    def _noop(tree):
        return tree

    if to_yaml_tree_mod is None:
        to_yaml_tree_mod = _noop
    if from_yaml_tree_mod is None:
        from_yaml_tree_mod = _noop

    if sort_string_lists:
        original_to_yaml_tree_mod = to_yaml_tree_mod

        def _sort_string_list(tree):
            for k, v in tree.items():
                if isinstance(v, list) and all(isinstance(item, str) for item in v):
                    tree[k] = sorted(v)
            return original_to_yaml_tree_mod(tree)

        to_yaml_tree_mod = _sort_string_list

    class _SerializationClass(WeldxConverter):
        name = class_name
        version = v
        types = [class_type]
        __module__ = class_type.__module__
        __qualname__ = class_type.__qualname__ + "Converter"

        def to_yaml_tree(
            self, obj: class_type, tag: str, ctx: SerializationContext
        ) -> dict:
            """Convert to python dict."""
            return to_yaml_tree_mod(obj.__dict__)

        def from_yaml_tree(
            self, node: dict, tag: str, ctx: SerializationContext
        ) -> class_type:
            """Reconstruct from yaml node."""
            return class_type(**from_yaml_tree_mod(node))

    return _SerializationClass


def get_weldx_extension(ctx: SerializationContext | AsdfConfig) -> Extension:
    """Grab the weldx extension from list of current active extensions."""
    if isinstance(ctx, SerializationContext):
        extensions = ctx.extension_manager.extensions
    elif isinstance(ctx, asdf.config.AsdfConfig):
        extensions = ctx.extensions
    else:
        raise TypeError(f"unsupported context {ctx=}")
    extensions = [
        ext for ext in extensions if str(ext.extension_uri) == WELDX_EXTENSION_URI
    ]
    if not len(extensions) == 1:
        raise ValueError("Could not determine correct weldx extension.")
    return extensions[0]


def uri_match(patterns: str | list[str], uri: str) -> bool:
    """Returns `True` if the ASDF URI matches any of the listed patterns.

    See Also
    --------
    asdf.util.uri_match

    """
    if isinstance(patterns, str):
        return asdf_uri_match(patterns, uri)
    return any(asdf_uri_match(p, uri) for p in patterns)


def get_converter_for_tag(tag: str) -> WeldxConverter | None:
    """Get the converter class that handles a given tag."""
    converters = [s for s in WeldxConverter.__subclasses__() if uri_match(s.tags, tag)]
    if len(converters) > 1:
        warn(
            f"Found more than one converter class for {tag=}", UserWarning, stacklevel=1
        )
    if converters:
        return converters[0]
    return None


def get_highest_tag_version(
    pattern: str | list[str], ctx: SerializationContext | AsdfConfig = None
) -> str | None:
    """Get the highest available weldx extension tag version matching a pattern.

    Parameters
    ----------
    pattern
        The tag pattern to match against.
    ctx
        The asdf context containing the extension.
        Will look in the current ``asdf_config()`` by default.

    Returns
    -------
    str
        The full tag string of the highest version match.

    Raises
    ------
    ValueError
        When the pattern matches multiple base tags in the extension.

    Examples
    --------
    >>> from weldx.asdf.util import get_highest_tag_version
    >>> get_highest_tag_version("asdf://weldx.bam.de/weldx/tags/uuid-*")
    'asdf://weldx.bam.de/weldx/tags/uuid-0.1.0'

    """
    if ctx is None:
        ctx = get_config()

    extension = get_weldx_extension(ctx)

    tags = [t._tag_uri for t in extension.tags if uri_match(pattern, t._tag_uri)]
    if not tags:  # no match found
        return None

    # we assume, that the version of the tag is separated by a right-most '-' char.
    tags.sort(key=lambda t: Version(t.rpartition("-")[-1]))
    base_tag = tags[-1].rpartition("-")[0]
    if not all(t.startswith(base_tag) for t in tags):
        raise ValueError(f"Found more than one base tag for {pattern=}.")
    return tags[-1]


def _get_instance_shape(instance_dict: TaggedDict | dict[str, Any]) -> list[int] | None:
    """Get the shape of an ASDF instance from its tagged dict form.

    Parameters
    ----------
    instance_dict
        The yaml node to evaluate.

    Returns
    -------
    List
        A numpy-style shape list or `None` if the shape could not be determined.
    """
    if isinstance(instance_dict, (float, int)):  # test against [1] for scalar values
        return [1]
    elif isinstance(instance_dict, Mapping) and "shape" in instance_dict:
        return instance_dict["shape"]
    elif isinstance(instance_dict, asdf.tagged.Tagged):
        # try calling shape_from_tagged for custom types
        converter = get_converter_for_tag(instance_dict._tag)
        if hasattr(converter, "shape_from_tagged"):
            return converter.shape_from_tagged(instance_dict)
    return None


def _get_instance_units(instance_dict: TaggedDict | dict[str, Any]) -> pint.Unit | None:
    """Get the units of an ASDF instance from its tagged dict form.

    Parameters
    ----------
    instance_dict
        The yaml node to evaluate.

    Returns
    -------
    pint.Unit
        The pint unit or `None` if no unit information is present.
    """
    if isinstance(instance_dict, (float, int)):  # base types
        return WELDX_UNIT_REGISTRY.dimensionless
    elif isinstance(instance_dict, Mapping) and UNITS_KEY in instance_dict:
        return U_(str(instance_dict[UNITS_KEY]))  # catch TaggedString as str
    elif isinstance(instance_dict, asdf.tagged.Tagged):
        # try calling units_from_tagged for custom types
        if instance_dict._tag.startswith("tag:stsci.edu:asdf/core/ndarray"):
            return WELDX_UNIT_REGISTRY.dimensionless
        converter = get_converter_for_tag(instance_dict._tag)
        if hasattr(converter, "units_from_tagged"):
            return converter.units_from_tagged(instance_dict)
    return None


class _ProtectedViewDict(MutableMapping):
    """A mutable mapping which protects given keys from manipulation."""

    def __init__(self, protected_keys, data=None):
        super().__init__()
        self.__data = data if data is not None else dict()
        self.protected_keys = protected_keys

    def _wrap_protected_non_existent(self, key, method: str):
        if key in self.protected_keys:
            self._warn_protected_keys()
            raise KeyError(f"'{key}' is protected.")
        elif key not in self.__data:
            raise KeyError(f"'{key}' not contained.")

        method_obj = getattr(self.__data, method)
        return method_obj(key)

    def __len__(self) -> int:
        return len(self.keys())

    def __getitem__(self, key):
        return self._wrap_protected_non_existent(key, "__getitem__")

    def __delitem__(self, key):
        return self._wrap_protected_non_existent(key, "__delitem__")

    def __setitem__(self, key, value):
        if key in self.protected_keys:
            self._warn_protected_keys()
            return
        self.__data[key] = value

    def keys(self) -> Set:
        return {k for k in self.__data.keys() if k not in self.protected_keys}

    def __iter__(self):
        return (k for k in self.keys())

    def __contains__(self, item):
        return item in self.keys()

    def update(self, mapping: Mapping[Hashable, Any], **kwargs: Any):  # pylint: disable=W0221
        _mapping = dict(mapping, **kwargs)  # merge mapping and kwargs
        if any(key in self.protected_keys for key in _mapping.keys()):
            self._warn_protected_keys()
            _mapping = {
                k: v for k, v in _mapping.items() if k not in self.protected_keys
            }

        self.__data.update(_mapping)

    def popitem(self) -> tuple[Hashable, Any]:
        for k in self.keys():
            if k not in self.protected_keys:
                return k, self.pop(k)

        raise KeyError("empty")

    def clear(self):
        """Clear all data except the protected keys."""
        _protected_data = {k: self.__data.pop(k) for k in self.protected_keys}
        self.__data.clear()
        self.__data.update(_protected_data)  # re-add protected data.
        assert len(self) == 0

    def _warn_protected_keys(self, stacklevel=3):
        import warnings

        warnings.warn(
            "You tried to manipulate an ASDF internal structure"
            f" (currently protected: {self.protected_keys}",
            stacklevel=stacklevel,
        )


def get_schema_tree(  # noqa: C901, MC0001, RUF100, codacy:ignore
    schemafile: str | Path, *, drop: set = None
) -> dict:
    """Get a dictionary representation of a weldx schema file with custom formatting.

    Parameters
    ----------
    schemafile
        Weldx schema file name or Path to parse.
    drop
        Set or list-like of additional keys to drop from all nested elements.
    Returns
    -------
    dict
        A reduced dictionary representation of the schema file requirements.
        The property keys are formatted to reflect the associated Python class.
        Some keys are dropped or reformatted for readability.
    """
    if drop is None:
        drop = {}
    if isinstance(schemafile, str):
        schemafile = get_schema_path(schemafile)

    contents = schemafile.read_text()
    header = asdf.yamlutil.load_tree(contents)

    remapped = [header]

    def resolve_python_classes(path, key, value):
        """Parse the tag or type information to the key string.

        This tries to resolves to python class names from 'tag' fields."""
        if not isinstance(value, dict):
            return key, value

        if "tag" in value:
            converter = get_converter_for_tag(value["tag"])
            if converter:
                tag_str = converter.default_class_display_name()
            else:
                tag_str = value["tag"].split("asdf://weldx.bam.de/weldx/tags/")[-1]
            key = f"{key} ({tag_str})"
        elif "$ref" in value:
            tag_str = value["$ref"].split("asdf://weldx.bam.de/weldx/schemas/")[-1]
            key = f"{key} (${tag_str})"
        elif value.get("type") == "object":
            key = f"{key} (dict)"
        elif value.get("type") == "array":
            key = f"{key} (list)"
        elif value.get("type") == "string":
            key = f"{key} (str)"
        elif value.get("type") == "number":
            key = f"{key} (number)"
        return key, value

    def convert_wx_shape(path, key, value):
        """Parse the list information in wx_shape into a readable string."""
        if isinstance(value, dict) and ("wx_shape" in value):
            if isinstance(value["wx_shape"], list):
                value = value.copy()
                value["wx_shape"] = f"[{','.join(str(n) for n in value['wx_shape'])}]"
        return key, value

    def mark_required(path, key, value):
        if not isinstance(value, dict):
            return key, value

        if "required" in value:
            reqs = value["required"]
            props = {
                (
                    k[:-1] + ", required" + ")"
                    if any(k.startswith(r) for r in reqs)
                    else k
                ): v
                for k, v in value["properties"].items()
            }
            value["properties"] = props

        return key, value

    def drop_meta(path, key, value):
        """Drop common metadata fields from the output."""
        default = {"examples", "description", "tag", "$ref", "type"}
        return key not in default | set(drop)

    def drop_properties(path, key, value):
        """Drop the 'properties' field."""
        if not isinstance(value, dict):
            return key, value
        if "properties" in value:
            value = value["properties"]
        return key, value

    remapped = remap(remapped, visit=convert_wx_shape)
    remapped = remap(remapped, visit=resolve_python_classes)
    remapped = remap(remapped, visit=mark_required)
    remapped = remap(remapped, visit=drop_meta)
    remapped = remap(remapped, visit=drop_properties)

    return remapped[0]
