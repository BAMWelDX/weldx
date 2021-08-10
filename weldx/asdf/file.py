"""`WeldxFile` wraps creation and updating of ASDF files and underlying files."""
import pathlib
import unittest.mock
from collections import UserDict
from collections.abc import MutableMapping
from contextlib import contextmanager
from io import BytesIO, IOBase
from typing import IO, Dict, List, Mapping, Optional, Union

from asdf import AsdfFile, generic_io, info
from asdf import open as open_asdf
from asdf import util
from asdf.tags.core import Software
from asdf.util import get_file_type
from jsonschema import ValidationError

from weldx.asdf import WeldxAsdfExtension, WeldxExtension
from weldx.asdf.util import get_schema_path, view_tree
from weldx.types import SupportsFileReadWrite, types_file_like, types_path_and_file_like

__all__ = [
    "WeldxFile",
]

from weldx.util import is_interactive_session, is_jupyterlab_session


@contextmanager
def reset_file_position(fh: SupportsFileReadWrite):
    """Reset the internal position of the given file after leaving the context.

    Parameters
    ----------
    fh :
        file handle

    """
    old_pos = fh.tell()
    yield
    fh.seek(old_pos)


DEFAULT_ARRAY_COMPRESSION = "input"
"""All arrays will be compressed using this algorithm, if not specified by user."""

DEFAULT_ARRAY_COPYING = True
"""Stored Arrays will be copied to memory, or not. If False, use memory mapping."""


class WeldxFile(UserDict):
    """Expose an ASDF file as a dictionary like object and handle underlying files.

    Parameters
    ----------
    filename_or_file_like :
        A path to a weldx file or file handle like to read/write data from.
        If `None` is passed, an in-memory file will be created.
    mode :
        Reading or reading/writing mode: "r" or "rw".
    asdffile_kwargs :
        Keyword arguments to pass to asdf.open.
        See `asdf.open` for reference.
    write_kwargs :
        Keyword arguments to pass to `asdf.AsdfFile.write_to`.
        See `asdf.AsdfFile.open` for reference.
    tree :
        An optional dictionary to write to the file.
    sync :
        If True, the changes to file will be written upon closing this. This is only
        relevant, if the file has been opened in write mode.
    custom_schema :
        A path-like object to a custom schema which validates the tree. All schemas
        provided by weldx can be given by name as well.
    software_history_entry :
        An optional dictionary which will be used to add history entries upon
        modification of the file. It has to provide the following keys:
        "name", "author", "homepage", "version"
        These should be set to string typed values. The homepage should be a URL.
    compression :
        If provided, set the compression type on all binary blocks
        in the file.  Must be one of:

        - ``''`` or `None`: No compression.
        - ``zlib``: Use zlib compression.
        - ``bzp2``: Use bzip2 compression.
        - ``lz4``: Use lz4 compression.
        - ``input``: Use the same compression as in the file read.
          If there is no prior file, acts as None.
    copy_arrays :
        When `False`, when reading files, attempt to memory map (memmap) underlying data
        arrays when possible. This avoids blowing the memory when working with very
        large datasets.

    """

    def __init__(
        self,
        filename_or_file_like: Optional[
            Union[str, pathlib.Path, types_file_like]
        ] = None,
        mode: str = "r",
        asdffile_kwargs: Mapping = None,
        write_kwargs: Mapping = None,
        tree: Mapping = None,
        sync: bool = True,
        custom_schema: Union[str, pathlib.Path] = None,
        software_history_entry: Mapping = None,
        compression: str = DEFAULT_ARRAY_COMPRESSION,
        copy_arrays: bool = DEFAULT_ARRAY_COPYING,
    ):
        if write_kwargs is None:
            write_kwargs = dict(all_array_compression=compression)

        if asdffile_kwargs is None:
            asdffile_kwargs = dict(copy_arrays=copy_arrays)

        # TODO: ensure no mismatching args for compression and copy_arrays.
        self._write_kwargs = write_kwargs
        self._asdffile_kwargs = asdffile_kwargs

        if custom_schema is not None:
            _custom_schema_path = pathlib.Path(custom_schema)
            if not _custom_schema_path.exists():
                try:
                    custom_schema = get_schema_path(custom_schema)
                except ValueError:
                    raise ValueError(
                        f"provided custom_schema {custom_schema} " "does not exist."
                    )
            asdffile_kwargs["custom_schema"] = custom_schema

        if mode not in ("r", "rw"):
            raise ValueError(
                f'invalid mode "{mode}" given. Should be one of "r", "rw".'
            )
        self._mode = mode
        self.sync_upon_close = bool(sync) & (self.mode == "rw")
        self.software_history_entry = software_history_entry

        new_file_created = False
        if filename_or_file_like is None:
            filename_or_file_like = BytesIO()
            new_file_created = True
            self._in_memory = True
            self._close = False  # we want buffers to be usable later on.
        elif isinstance(filename_or_file_like, (str, pathlib.Path)):
            filename_or_file_like, new_file_created = self._handle_path(
                filename_or_file_like, mode
            )
            self._in_memory = False
            self._close = True
        elif isinstance(filename_or_file_like, types_file_like.__args__):
            if isinstance(filename_or_file_like, BytesIO):
                self._in_memory = True
            else:
                self._in_memory = False
            # the user passed a raw file handle, its their responsibility to close it.
            self._close = False
        else:
            _supported = WeldxFile.__init__.__annotations__["filename_or_file_like"]
            raise ValueError(
                f"Unsupported input type '{type(filename_or_file_like)}'."
                f" Should be one of {_supported}."
            )

        extensions = [WeldxExtension(), WeldxAsdfExtension()]
        # If we have data to write, we do it first, so a WeldxFile is always in sync.
        if tree or new_file_created:
            asdf_file = AsdfFile(
                tree=tree, extensions=extensions, custom_schema=self.custom_schema
            )
            asdf_file.write_to(filename_or_file_like, **write_kwargs)
            if isinstance(filename_or_file_like, SupportsFileReadWrite):
                filename_or_file_like.seek(0)

        asdf_file = open_asdf(
            filename_or_file_like,
            mode=self.mode,
            extensions=extensions,
            **asdffile_kwargs,
        )
        self._asdf_handle: AsdfFile = asdf_file

        # UserDict interface: we want to store a reference to the tree, but the ctor
        # of UserDict takes a copy, so we do it manually here.
        super(WeldxFile, self).__init__()
        self.data = self._asdf_handle.tree

    @property
    def mode(self) -> str:
        """Open mode of file, one of read or read/write."""
        return self._mode

    @property
    def in_memory(self) -> bool:
        """Is the underlying file an in-memory buffer.

        Returns
        -------
        True, if this file is backed by an in-memory buffer,
        False otherwise.

        """
        return self._in_memory

    @staticmethod
    def _handle_path(filename, mode) -> (IO, bool):
        # opens a file handle with given mode,
        # and returns it + if a new file has been created.
        new_file_created = False
        exists = pathlib.Path(filename).exists()
        if not exists and mode == "r":
            raise RuntimeError(f"file {filename} has be created," " but mode is 'r'.")

        if mode == "rw":
            if not exists:
                new_file_created = True
                real_mode = "bx+"  # binary, exclusive creation, e.g. raise if exists.
            else:
                generic_file = generic_io.get_file(filename, mode="r")
                file_type = get_file_type(generic_file)
                if not file_type == util.FileType.ASDF:
                    raise FileExistsError(
                        f"given file {filename} is not an ASDF file and "
                        "could be overwritten because of read/write mode!"
                    )
                real_mode = "br+"
        else:
            real_mode = "rb"

        # create file handle
        filename = open(filename, mode=real_mode)
        return filename, new_file_created

    @property
    def software_history_entry(self):
        """History entries will use this software.

        Examples
        --------
        >>> import weldx

        Define a custom softare entry:

        >>> software = dict(name="MyFancyPackage", author="Me", \
                homepage="https://startpage.com", version="1.0")
        >>> f = weldx.WeldxFile(software_history_entry=software)
        >>> f.add_history_entry("we made some change")
        >>> f.history #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
         [{'description': 'we made some change', \
           'time': datetime.datetime(...), \
           'software': {'name': 'MyFancyPackage', 'author': 'Me', \
           'homepage': 'https://startpage.com', 'version': '1.0'}}]

        We can also change the software on the fly:
        >>> software_new = dict(name="MyTool", author="MeSoft", \
                homepage="https://startpage.com", version="1.0")
        >>> f.add_history_entry("another change using mytool", software_new)

        Lets inspect the last history entry:

        >>> f.history[-1] #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        {'description': 'another change using mytool', \
         'time': datetime.datetime(...), \
         'software': {'name': 'MyTool', 'author': 'MeSoft', \
         'homepage': 'https://startpage.com', 'version': '1.0'}}

        """
        return self._DEFAULT_SOFTWARE_ENTRY

    @software_history_entry.setter
    def software_history_entry(self, value: dict):
        """Set the software used for making history entries."""
        if value is None:
            from weldx import __version__ as version

            self._DEFAULT_SOFTWARE_ENTRY = {
                "name": "weldx",
                "author": "BAM",
                "homepage": "https://www.bam.de/Content/EN/Projects/WelDX/weldx.html",
                "version": version,
            }
        else:
            if not isinstance(value, Dict):
                raise ValueError("expected a dictionary type")
            try:
                test = AsdfFile(tree=dict(software=Software(value)))
                test.validate()
            except ValidationError as ve:
                raise ValueError(f"Given value has invalid format: {ve}")
            self._DEFAULT_SOFTWARE_ENTRY = value

    def __enter__(self):
        """Enter the context."""
        self._asdf_handle.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context."""
        self.close()

    def close(self):
        """Close this file and sync it, if mode is read/write."""
        if self.mode == "rw" and self.sync_upon_close:
            self.sync(**self._write_kwargs)
        fh = self.file_handle
        self._asdf_handle.close()

        # close underlying file handle, if not already done by ASDF.
        if self._close and not fh.closed:
            fh.close()

    def sync(
        self,
        all_array_storage: str = None,
        all_array_compression: str = "input",
        pad_blocks: Union[float, bool] = False,
        include_block_index: bool = True,
        version: str = None,
        **kwargs,
    ):
        self._asdf_handle.update(
            all_array_storage=all_array_storage,
            all_array_compression=all_array_compression,
            pad_blocks=pad_blocks,
            include_block_index=include_block_index,
            version=version,
            **kwargs,
        )

    sync.__doc__ = AsdfFile.update.__doc__

    def add_history_entry(self, change_desc: str, software: dict = None) -> None:
        """Add an history_entry to the file.

        Parameters
        ----------
        change_desc :
            Description of the change made.
        software :
            Optional software used to make the change.

        See Also
        --------
        The software entry will be inferred from the constructor or, if not defined,
        from `software_history_entry`.

        """
        if software is None:
            software = self.software_history_entry
        self._asdf_handle.add_history_entry(change_desc, software)

    @property
    def history(self) -> List:
        """Return a list of all history entries in this file."""
        return self._asdf_handle.get_history_entries()

    @property
    def custom_schema(self) -> Optional[str]:
        """Return schema used to validate the structure and types of the tree."""
        return self._asdffile_kwargs.get("custom_schema", None)

    @property
    # TODO: should we actually expose this? It allows advanced operations for adults.
    # TODO: return type is also not guaranteed to be IOBase, right?
    def file_handle(self) -> IOBase:
        """Return the underlying file handle, use with CARE."""
        if self._asdf_handle._closed:
            raise RuntimeError("closed file, cannot access file handle.")
        return self._asdf_handle._fd._fd

    def as_attr(self) -> MutableMapping:
        """Return the Weldx dictionary as an attributed object.

        Returns
        -------
        attrdict :
            This dictionary wrapped such, that all of its keys can be accessed as
            properties.

        Examples
        --------
        >>> tree = dict(wx_meta={"welder": "Nikolai Nikolajewitsch Benardos"})
        >>> wf = WeldxFile(tree=tree)
        >>> wfa = wf.as_attr()
        >>> wfa.wx_meta.welder
        'Nikolai Nikolajewitsch Benardos'

        We can also change theh data easily
        >>> wfa.wx_meta.welder = "Myself"
        >>> wfa.wx_meta.welder
        'Myself'
        """

        class AttrDict(dict):
            def __init__(self, iterable, **kwargs):
                super(AttrDict, self).__init__(iterable, **kwargs)
                for key, value in self.items():
                    if isinstance(value, Mapping):
                        self.__dict__[key] = AttrDict(value)
                    else:
                        self.__dict__[key] = value

        return AttrDict(self)

    def write_to(
        self, fd: Optional[types_path_and_file_like] = None, **write_args
    ) -> Optional[BytesIO]:
        """Write current weldx file to given file name or file type.

        Parameters
        ----------
        fd :
            May be a string path to a file, or a Python file-like
            object. If a string path, the file is automatically
            closed after writing. If `None` is given, write to a new buffer.

        write_args :
            Allowed parameters:

            * all_array_storage=None
            * all_array_compression='input'
            * auto_inline=None
            * pad_blocks=False
            * include_block_index=True
            * version=None

        Returns
        -------
        The given input file name or a buffer, in case the input was omitted.

        """
        created = False
        if fd is None:
            fd = BytesIO()
            created = True

        # if no args are given, we use the specifications given in the constructor.
        if not write_args:
            write_args = self._write_kwargs

        self._asdf_handle.write_to(fd, **write_args)

        if created:
            fd.seek(0)
        return fd

    def show_asdf_header(
        self, use_widgets: bool = None, _interactive: Optional[bool] = None
    ):
        """Show the header of the ASDF serialization.

        Depending on the execution environment (plain Python interpreter, Jupyter Lab)
        and the use_widgets parameter the header will be displayed in different styles.
        By default, the display will be interactive (using widgets), but can be disabled
        if undesired.

        Parameters
        ----------
        use_widgets :
            When in an interactive session, use widgets to traverse the header or show
            a static syntax highlighted string?
            Representation is determined upon the frontend. Jupyter lab supports a
            complex widget, which does not work in plain old Jupyter notebook.
        _interactive :
            Should not be set.
        """
        return _HeaderVisualizer(self._asdf_handle).show(
            use_widgets=use_widgets, _interactive=_interactive
        )

    def _ipython_display_(self):
        # this will be called in Jupyter Lab, but not in a plain notebook.
        from IPython.core.display import display

        display(self.show_asdf_header(use_widgets=False, _interactive=True))


class _DummyBlock:
    def __init__(self):
        self.array_storage = "internal"
        self.trust_data_dtype = False
        self.data = None
        self.readonly = True

    def __len__(self):
        return 0


class _DummyBlockManager:
    array_storage = "internal"
    lazy_load = True

    def __init__(self):
        self.default_block = _DummyBlock()

    @staticmethod
    def get_source(*args, **kwargs):
        return 0

    def get_block(self, source):
        return self.default_block

    __getitem__ = get_block

    def find_or_create_block_for_array(self, *args, **kwargs):
        return self.default_block

    @staticmethod
    def get_output_compression_extensions():
        return ()

    def add(self, *args, **kwargs):
        pass

    write_internal_blocks_random_access = (
        write_block_index
    ) = (
        write_internal_blocks_serial
    ) = write_external_blocks = finalize = set_array_storage = add


@contextmanager
def _fake_block_context(asdf_handle):
    blocks_org = asdf_handle.blocks
    asdf_handle._blocks = _DummyBlockManager()
    yield asdf_handle
    asdf_handle._blocks = blocks_org


class _HeaderVisualizer:
    def __init__(self, asdf_handle):
        import copy

        self._asdf_handle = copy.copy(asdf_handle)

    def _write_to_buffer_without_blocks(self) -> BytesIO:
        """Write an asdf file with no blocks using a fake block manager.

        Returns
        -------
        buffer:
            containing the header contents.
        """
        buff = BytesIO()
        with _fake_block_context(self._asdf_handle) as h:
            h.write_to(buff, include_block_index=False, all_array_storage="internal")
        buff.seek(0)

        return buff

    def show(
        self, use_widgets=None, _interactive=None
    ) -> Union[None, "IPython.display.HTML", "IPython.display.JSON"]:  # noqa: F821
        if _interactive is None:
            _interactive = is_interactive_session()
        if use_widgets is None:
            use_widgets = is_jupyterlab_session()

        # We write the current tree to a buffer __without__ any binary data attached.
        buff = self._write_to_buffer_without_blocks()

        # automatically determine if this runs in an interactive session.
        # These methods return an IPython displayable object
        # (passed to IPython.display()).
        if _interactive:
            return self._show_interactive(use_widgets=use_widgets, buff=buff)
        else:
            self._show_non_interactive(buff=buff)

    @staticmethod
    def _show_interactive(
        use_widgets: bool, buff: BytesIO
    ) -> Union["IPython.display.HTML", "IPython.display.JSON"]:  # noqa: F821
        from weldx.asdf.util import notebook_fileprinter

        if use_widgets:
            result = view_tree(buff)
        else:
            result = notebook_fileprinter(buff)
        return result

    @staticmethod
    def _show_non_interactive(buff: BytesIO):
        with unittest.mock.patch(
            "asdf.AsdfFile.blocks", new_callable=_DummyBlockManager
        ):
            with WeldxFile(buff) as wx:
                info(wx._asdf_handle, show_values=True, max_rows=None)
