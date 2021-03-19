"""Utilities for asdf files."""
from io import BytesIO
from pathlib import Path
from typing import Tuple

import asdf
import yaml
from boltons.iterutils import get_path

from weldx.asdf.extension import WeldxAsdfExtension, WeldxExtension

__all__ = [
    "read_buffer",
    "write_buffer",
    "write_read_buffer",
    "get_yaml_header",
    "asdf_json_repr",
    "notebook_fileprinter",
]

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


def get_yaml_header(file) -> str:  # pragma: no cover
    """Read the YAML header part (excluding binary sections) of an ASDF file.

    Parameters
    ----------
    file
        filename, ``pathlib.Path`` or ``BytesIO`` buffer of ASDF file

    Returns
    -------
    str
        The YAML header the ASDF file

    """
    if isinstance(file, BytesIO):
        file.seek(0)
        code = file.read()
    else:
        with open(file, "rb") as f:
            code = f.read()

    parts = code.partition(b"\n...")
    code = parts[0].decode("utf-8") + parts[1].decode("utf-8")
    return code


# backward compatibility, remove when adopted to public funcs in notebooks etc.
_write_buffer = write_buffer
_read_buffer = read_buffer
_write_read_buffer = write_read_buffer


def notebook_fileprinter(file, lexer="YAML"):  # pragma: no cover
    """Print the code from file/BytesIO  to notebook cell with syntax highlighting.

    Parameters
    ----------
    file
        filename or ``BytesIO`` buffer of ASDF file
    lexer
        Syntax style to use

    """
    from IPython.display import HTML
    from pygments import highlight
    from pygments.formatters.html import HtmlFormatter
    from pygments.lexers import get_lexer_by_name, get_lexer_for_filename

    if isinstance(file, BytesIO):
        lexer = get_lexer_by_name(lexer)
    elif Path(file).suffix == ".asdf":
        lexer = get_lexer_by_name("YAML")
    else:
        lexer = get_lexer_for_filename(file)

    code = get_yaml_header(file)
    formatter = HtmlFormatter()
    return HTML(
        '<style type="text/css">{}</style>{}'.format(
            formatter.get_style_defs(".highlight"),
            highlight(code, lexer, formatter),
        )
    )


def asdf_json_repr(file, path: Tuple = None, **kwargs):  # pragma: no cover
    """Display YAML header using IPython JSON display repr.

    This function works in JupyterLab.

    Parameters
    ----------
    file
        filename or BytesIO buffer of ASDF file
    path
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

        weldx.asdf.utils.asdf_json_repr("single_pass_weld_example.asdf")

    Visualize a specific element in the tree structure by proving the path::

        weldx.asdf.utils.asdf_json_repr(
            "single_pass_weld_example.asdf", path=("process", "welding_process")
        )


    """
    from IPython.display import JSON

    if isinstance(file, str):
        root = file + "/"
    else:
        root = "/"

    code = get_yaml_header(file)
    yaml_dict = yaml.load(code, Loader=yaml.BaseLoader)
    if path:
        root = root + "/".join(path)
        yaml_dict = get_path(yaml_dict, path)
    kwargs["root"] = root
    return JSON(yaml_dict, **kwargs)
