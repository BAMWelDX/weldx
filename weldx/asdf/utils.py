from io import BytesIO
from pathlib import Path

import asdf
import yaml

from weldx.asdf.extension import WeldxAsdfExtension, WeldxExtension


# TODO: these functions be gneralized and be public
# asdf read/write debug tools functions ---------------------------------------


def _write_buffer(tree: dict, asdffile_kwargs: dict = None, write_kwargs: dict = None):
    """Write ASDF file into buffer.

    Parameters
    ----------
    tree:
        Tree object to serialize.
    asdffile_kwargs
        Additional keywords to pass to asdf.AsdfFile()
    write_kwargs
        Additional keywords to pass to asdf.AsdfFile.write_to()
        Weldx-Extensions are always set.

    Returns
    -------
    BytesIO
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


def _read_buffer(buffer: BytesIO, open_kwargs: dict = None):
    """Read ASDF file contents from buffer instance.

    Parameters
    ----------
    buffer
        Buffer containing ASDF file contents
    open_kwargs
        Additional keywords to pass to asdf.AsdfFile.open()
        Extensions are always set, copy_arrays=True is set by default.

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


def _write_read_buffer(
        tree: dict, asdffile_kwargs=None, write_kwargs=None, open_kwargs=None
):
    """Perform a buffered write/read roundtrip of a tree using default ASDF settings.
    Parameters
    ----------
    tree
        Tree object to serialize.
    asdffile_kwargs
        Additional keywords to pass to asdf.AsdfFile()
    write_kwargs
        Additional keywords to pass to asdf.AsdfFile.write_to()
        Extensions are always set.
    open_kwargs
        Additional keywords to pass to asdf.AsdfFile.open()
        Extensions are always set, copy_arrays=True is set by default.
    Returns
    -------
    dict
    """
    buffer = _write_buffer(tree, asdffile_kwargs, write_kwargs)
    return _read_buffer(buffer, open_kwargs)


def _get_yaml_header(file) -> str:
    """Read the YAML header part of an ASDF file.

    Parameters
    ----------
    file
        filename or BytesIO buffer of ASDF file

    Returns
    -------
    str

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


def notebook_fileprinter(file, lexer="YAML"):
    """Prints the code from file/BytesIO  to notebook cell with syntax highlighting.

    Parameters
    ----------
    file
        filename or BytesIO buffer of ASDF file
    lexer
        Syntax style to use

    """
    from pygments import highlight
    from pygments.lexers import get_lexer_by_name, get_lexer_for_filename
    from pygments.formatters.html import HtmlFormatter
    from IPython.display import HTML

    if isinstance(file, BytesIO):
        lexer = get_lexer_by_name(lexer)
    elif Path(file).suffix == ".asdf":
        lexer = get_lexer_by_name("YAML")
    else:
        lexer = get_lexer_for_filename(file)

    code = _get_yaml_header(file)

    formatter = HtmlFormatter()
    return HTML(
        '<style type="text/css">{}</style>{}'.format(
            formatter.get_style_defs(".highlight"),
            highlight(code, lexer, formatter),
        )
    )


def asdf_json_repr(file, **kwargs):
    """Display YAML header using IPython JSON display repr.

    This function works in JupyterLab.

    Parameters
    ----------
    file
        filename or BytesIO buffer of ASDF file
    kwargs
        kwargs passed down to JSON constructor

    Returns
    -------
    IPython.display.JSON
        JSON object for rich output in JupyterLab

    """
    from IPython.core.display import JSON
    code = _get_yaml_header(file)
    yaml_dict = yaml.load(code, Loader=yaml.BaseLoader)
    return JSON(yaml_dict, **kwargs)
