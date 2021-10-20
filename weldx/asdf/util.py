"""Utilities for asdf files."""
from collections.abc import Mapping
from distutils.version import LooseVersion
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Type, Union
from warnings import warn

import asdf
from asdf.asdf import SerializationContext
from asdf.config import AsdfConfig, get_config
from asdf.extension._extension import Extension
from asdf.tagged import TaggedDict
from asdf.util import uri_match as asdf_uri_match
from boltons.iterutils import get_path

from weldx.asdf.constants import SCHEMA_PATH, WELDX_EXTENSION_URI
from weldx.asdf.types import WeldxConverter
from weldx.types import (
    SupportsFileReadOnly,
    SupportsFileReadWrite,
    types_file_like,
    types_path_and_file_like,
    types_path_like,
)
from weldx.util import deprecated

_USE_WELDX_FILE = False
_INVOKE_SHOW_HEADER = False


__all__ = [
    "get_schema_path",
    "read_buffer",
    "write_buffer",
    "write_read_buffer",
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

    p = SCHEMA_PATH / ".."  # legacy_code - also look for legacy schemas
    schemas = list(p.glob(f"**/{schema}.yaml"))
    if len(schemas) == 0:
        raise ValueError(f"No matching schema for filename '{schema}'.")
    elif len(schemas) > 1:
        warn(f"Found more than one matching schema for filename '{schema}'.")
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

    Notes
    -----
    In addition to the usual asdf.AsdfFile.write_to arguments in write_args you can pass
    the parameter "dummy_arrays". If set, all array data is replaced with a empty list.
    """
    if asdffile_kwargs is None:
        asdffile_kwargs = {}
    if write_kwargs is None:
        write_kwargs = {}

    dummy_inline_arrays = write_kwargs.pop("dummy_arrays", False)

    if _use_weldx_file is None:
        _use_weldx_file = _USE_WELDX_FILE

    if _invoke_show_header is None:
        _invoke_show_header = _INVOKE_SHOW_HEADER

    def show(wx):
        if _invoke_show_header:
            wx.show_asdf_header(False, False)

    if _use_weldx_file:
        write_kwargs = dict(all_array_storage="inline")
        from weldx import WeldxFile

        with WeldxFile(
            tree=tree,
            asdffile_kwargs=asdffile_kwargs,
            write_kwargs=write_kwargs,
            mode="rw",
        ) as wx:
            wx.write_to()
            show(wx)
            buff = wx.file_handle
    else:
        buff = BytesIO()
        with asdf.AsdfFile(tree, extensions=None, **asdffile_kwargs) as ff:
            if dummy_inline_arrays:  # lets store an empty list in the asdf file.
                write_kwargs["all_array_storage"] = "inline"
                from unittest.mock import patch

                with patch("asdf.tags.core.ndarray.numpy_array_to_list", lambda x: []):
                    ff.write_to(buff, **write_kwargs)
            else:
                ff.write_to(buff, **write_kwargs)
    buff.seek(0)
    return buff


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
        Extensions are always set, ``copy_arrays=True`` is set by default.

    Returns
    -------
    dict
        ASDF file tree.

    """
    if open_kwargs is None:
        open_kwargs = {"copy_arrays": True}

    buffer.seek(0)

    if _use_weldx_file is None:
        _use_weldx_file = _USE_WELDX_FILE
    if _use_weldx_file:
        from weldx import WeldxFile

        return WeldxFile(buffer, asdffile_kwargs=open_kwargs)

    with asdf.open(
        buffer,
        extensions=None,
        **open_kwargs,
    ) as af:
        data = af.tree
    return data


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
        Extensions are always set, ``copy_arrays=True`` is set by default.

    Returns
    -------
    dict

    """
    buffer = write_buffer(tree, asdffile_kwargs, write_kwargs)
    return read_buffer(buffer, open_kwargs)


def get_yaml_header(file: types_path_and_file_like, parse=False) -> Union[str, dict]:
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
        # reads lines until the line "..." is reached.
        def readline_and_strip_line_ending():
            return handle.readline()[:-1]
        return b"".join(iter(readline_and_strip_line_ending, b"..."))

    if isinstance(file, SupportsFileReadWrite):
        file.seek(0)
        code = read_header(file)
    elif isinstance(file, SupportsFileReadOnly):
        code = read_header(file)
    elif isinstance(file, types_path_like.__args__):
        with open(file, "rbU") as f:
            code = read_header(f)
    else:
        raise TypeError(f"cannot read yaml header from {type(file)}.")

    if parse:
        return asdf.yamlutil.load_tree(code)
    return code.decode("utf-8")


@deprecated("0.4.0", "0.5.0", " _write_buffer was renamed to write_buffer")
def _write_buffer(*args, **kwargs):
    return write_buffer(*args, **kwargs)


@deprecated("0.4.0", "0.5.0", " _read_buffer was renamed to read_buffer")
def _read_buffer(*args, **kwargs):
    return read_buffer(*args, **kwargs)


@deprecated("0.4.0", "0.5.0", " _write_read_buffer was renamed to write_read_buffer")
def _write_read_buffer(*args, **kwargs):
    return write_read_buffer(*args, **kwargs)


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


def view_tree(file: types_path_and_file_like, path: Tuple = None, **kwargs):
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

    if isinstance(file, str):
        root = file + "/"
    else:
        root = "/"

    yaml_dict = get_yaml_header(file, parse=True)
    if path:
        root = root + "/".join(path)
        yaml_dict = get_path(yaml_dict, path)
    kwargs["root"] = root
    return JSON(yaml_dict, **kwargs)


@deprecated("0.4.0", "0.5.0", " asdf_json_repr was renamed to view_tree")
def asdf_json_repr(file: Union[str, Path, BytesIO], path: Tuple = None, **kwargs):
    """See `view_tree` function."""
    return view_tree(file, path, **kwargs)


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
    class_type: Type,
    class_name: str,
    version: str,
    to_yaml_tree_mod: Callable = None,
    from_yaml_tree_mod: Callable = None,
    sort_string_lists: bool = True,
) -> Type:
    """Generate a asdf serialization class for a python dataclass.

    Parameters
    ----------
    class_type :
        The type of the dataclass
    class_name :
        The value that should ba stored as the classes name property
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


def get_weldx_extension(ctx: Union[SerializationContext, AsdfConfig]) -> Extension:
    """Grab the weldx extension from list of current active extensions."""
    if isinstance(ctx, asdf.asdf.SerializationContext):
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


def uri_match(patterns: Union[str, List[str]], uri: str) -> bool:
    """Returns `True` if the ASDF URI matches any of the listed patterns.

    See Also
    --------
    asdf.util.uri_match

    """
    if isinstance(patterns, str):
        return asdf_uri_match(patterns, uri)
    return any(asdf_uri_match(p, uri) for p in patterns)


def get_converter_for_tag(tag: str) -> Union[type, None]:
    """Get the converter class that handles a given tag."""
    converters = [s for s in WeldxConverter.__subclasses__() if uri_match(s.tags, tag)]
    if len(converters) > 1:
        warn(f"Found more than one converter class for {tag=}", UserWarning)
    if converters:
        return converters[0]
    return None


def get_highest_tag_version(
    pattern: str, ctx: Union[SerializationContext, AsdfConfig] = None
) -> Union[str, None]:
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
        When the pattern matches multiple base tags in in the extension.

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

    tags.sort(key=LooseVersion)
    base_tag = tags[-1].rpartition("-")[0]
    if not all(t.startswith(base_tag) for t in tags):
        raise ValueError(f"Found more than one base tag for {pattern=}.")
    return tags[-1]


def _get_instance_shape(
    instance_dict: Union[TaggedDict, Dict[str, Any]]
) -> Union[List[int], None]:
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
    elif isinstance(instance_dict, asdf.types.tagged.Tagged):
        # try calling shape_from_tagged for custom types
        converter = get_converter_for_tag(instance_dict._tag)
        if hasattr(converter, "shape_from_tagged"):
            return converter.shape_from_tagged(instance_dict)
    return None
