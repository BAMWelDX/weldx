"""Utilities for asdf files."""
from io import BytesIO
from pathlib import Path
from typing import Tuple, Union
from warnings import warn

import asdf
from boltons.iterutils import get_path

from weldx.asdf.constants import SCHEMA_PATH
from weldx.asdf.extension import WeldxAsdfExtension, WeldxExtension
from weldx.types import (
    SupportsFileReadOnly,
    SupportsFileReadWrite,
    types_file_like,
    types_path_and_file_like,
    types_path_like,
)
from weldx.util import deprecated

__all__ = [
    "get_schema_path",
    "read_buffer",
    "write_buffer",
    "write_read_buffer",
    "get_yaml_header",
    "view_tree",
    "notebook_fileprinter",
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

    p = Path(SCHEMA_PATH)
    schemas = list(p.glob(f"**/{schema}.yaml"))
    if len(schemas) == 0:
        raise ValueError(f"No matching schema for filename '{schema}'.")
    elif len(schemas) > 1:
        warn(f"Found more than one matching schema for filename '{schema}'.")
    return schemas[0]


# asdf read/write debug tools functions ---------------------------------------


def write_buffer(
    tree: dict, asdffile_kwargs: dict = None, write_kwargs: dict = None
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

    buff = BytesIO()
    with asdf.AsdfFile(
        tree, extensions=[WeldxExtension(), WeldxAsdfExtension()], **asdffile_kwargs
    ) as ff:
        ff.write_to(buff, **write_kwargs)
        buff.seek(0)
    return buff


def read_buffer(buffer: BytesIO, open_kwargs: dict = None):
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
    with asdf.open(
        buffer,
        extensions=[WeldxExtension(), WeldxAsdfExtension()],
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
        # reads lines until the byte string "...\n" is approached.
        return b"".join(iter(handle.readline, b"...\n"))

    if isinstance(file, SupportsFileReadWrite):
        file.seek(0)
        code = read_header(file)
    elif isinstance(file, SupportsFileReadOnly):
        code = read_header(file)
    elif isinstance(file, types_path_like.__args__):
        with open(file, "rb") as f:
            code = read_header(f)

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


def asdf_dataclass_serialization(original_class):
    """Add :code:`to_tree` and :code:`from_tree` methods for dataclass serialization.

    Parameters
    ----------
    original_class:
        Original data class to decorate

    Returns
    -------
    type :
        The class with added :code:`to_tree` and `from_tree` function.

    Examples
    --------
    Here is an example how to generate a simple serialization class for a given
    dataclass::

        @dataclass
        class MyClass:
            a: np.ndarray

        @asdf_dataclass_serialization
        class MyClassTypeASDF(WeldxType):
            name = "my_class"
            version = "1.0.0"
            types = [MyClass]
            requires = ["weldx"]
            handle_dynamic_subclasses = True

    """
    from copy import deepcopy

    from weldx.asdf.types import from_tree_metadata, to_tree_metadata

    def _default_to_tree(node, ctx):
        from xarray import DataArray

        tree = deepcopy(node.__dict__)
        for k, v in tree.items():
            if isinstance(v, DataArray):
                tree[k] = v.data
        return tree

    def _default_from_tree(tree, ctx):
        import numpy as np

        for k, v in tree.items():
            if isinstance(v, np.ndarray):
                tree[k] = np.asarray(v)
        return original_class.types[0](**tree)

    original_class.to_tree = classmethod(to_tree_metadata(_default_to_tree))
    original_class.from_tree = classmethod(from_tree_metadata(_default_from_tree))

    return original_class
