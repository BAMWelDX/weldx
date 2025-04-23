"""`WeldxFile` wraps creation and updating of ASDF files and underlying files."""

from __future__ import annotations

import copy
import importlib.metadata
import io
import pathlib
import warnings
from collections.abc import Hashable, Iterable, Mapping, MutableMapping, Set, ValuesView
from contextlib import contextmanager
from io import BytesIO, IOBase
from typing import IO, Any, get_args

import asdf
import numpy as np
from asdf import AsdfFile, config_context, generic_io
from asdf import open as open_asdf
from asdf.exceptions import AsdfWarning, ValidationError
from asdf.tags.core import Software
from asdf.util import FileType, get_file_type
from boltons.iterutils import get_path

from weldx.asdf.util import (
    _ProtectedViewDict,
    get_schema_path,
    get_yaml_header,
    view_tree,
)
from weldx.exceptions import WeldxDeprecationWarning
from weldx.types import (
    SupportsFileReadWrite,
    types_file_like,
    types_path_and_file_like,
    types_path_like,
)
from weldx.util import (
    inherit_docstrings,
    is_interactive_session,
    is_jupyterlab_session,
)

__all__ = [
    "WeldxFile",
    "DEFAULT_ARRAY_COMPRESSION",
    "DEFAULT_MEMORY_MAPPING",
    "DEFAULT_ARRAY_INLINE_THRESHOLD",
    "_PROTECTED_KEYS",
]

_asdf_version = tuple(importlib.metadata.version("asdf").split("."))


def asdf_open_memory_mapping_kwarg(memmap: bool) -> dict:
    if _asdf_version >= ("3", "1", "0"):
        return {"memmap": memmap}
    else:
        return {"copy_arrays": not memmap}


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

DEFAULT_MEMORY_MAPPING = False
"""Stored Arrays will be memory-mapped, or not. If True, use memory mapping."""

DEFAULT_ARRAY_INLINE_THRESHOLD = 10
"""Arrays with less or equal elements will be inlined (stored as string, not binary)."""

_PROTECTED_KEYS = (
    "history",
    "asdf_library",
)
"""These keys are not seen, nor can they be manipulated."""


@inherit_docstrings
class WeldxFile(_ProtectedViewDict):
    """Expose an ASDF file as a dictionary like object and handle underlying files.

    The WeldxFile class makes it easy to work with ASDF files. Creating, validating,
    and updating data is handled by just treating the WeldxFile object as a dictionary.
    Every piece of data is accessible via a key, while the keys should be strings,
    the data can be any Python object.

    Creation is being done by just creating a WeldxFile with or without a filename.
    Operation without filenames is of course non-persitant, meaning that your changes
    will be lost upon the end of your Python session.

    You can decide, whether you want to open your file with read-only (default) or
    read and write mode. This is mainly a safety precaution, in order not to overwrite
    or manipulate existing data by accident.

    For a brief introduction into the features of this class, please have a look at the
    :doc:`tutorial <../tutorials/weldxfile>` or the examples given here.

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
        Either a path-like object to a custom schema which validates the tree
        or a tuple of these objects. This tuple is allowed of lengths two and the first
        element will be used to validate the file contents upon reading, the second
        upon writing. For example:

            (None, "A")

        will not run validation upon reading, but for writing will use "A". Likewise

            ("A", "B")

        will use schema "A" to validate after reading, and "B" for writing again.

        Note that all schemas provided by weldx can be given by name as well.

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
    memmap :
        When `True`, when reading files, attempt to memory map (memmap) underlying data
        arrays when possible. This avoids blowing the memory when working with very
        large datasets.
    array_inline_threshold :
        arrays below this threshold will be serialized as string, if larger as binary
        block. Note that this does not affect arrays, which are being shared across
        several objects in the same file.

    Examples
    --------
    We define a simple data set and store it in a WeldxFile.

    >>> data = {"name": "CXCOMP", "value": 42}
    >>> wx = WeldxFile(tree=data, mode="rw")

    If we want to persist the WeldxFile to a file on hard drive we invoke:

    >>> wx2 = WeldxFile("output.wx", tree=data, mode="rw")

    Or we can store the previously created file to disk:

    >>> wx.write_to("output2.wx")
    'output2.wx'

    If we omit the filename, we receive an in-memory file. This is useful to create
    quick copies without the need for physical files.

    >>> wx.write_to()  # doctest: +ELLIPSIS
    <_io.BytesIO object at 0x...>

    We can also read a data set from disk and manipulate it in a dictionary like manner.

    >>> with WeldxFile("output.wx", mode="rw") as wx:
    ...    wx["name"] = "changed"

    If a file name is omitted, we operate only in memory.

    We can have a look at the serialized data by looking at the asdf header

    >>> wx2.header()  # doctest: +ELLIPSIS,+NORMALIZE_WHITESPACE
    #ASDF 1.0.0
    #ASDF_STANDARD ...
    %YAML 1.1
    %TAG ! tag:stsci.edu:asdf/
    --- !core/asdf-1.1.0
    asdf_library: !core/software-1.0.0 ...
    history:
      extensions:
      ...
    name: CXCOMP
    value: 42
    <BLANKLINE>

    """

    def __init__(
        self,
        filename_or_file_like: types_path_like | types_file_like | None = None,
        mode: str = "r",
        asdffile_kwargs: Mapping = None,
        write_kwargs: Mapping = None,
        tree: Mapping = None,
        sync: bool = True,
        custom_schema: None
        | (
            types_path_like,
            tuple[None, types_path_like],
        ) = None,
        software_history_entry: Mapping = None,
        compression: str = DEFAULT_ARRAY_COMPRESSION,
        memmap: bool = DEFAULT_MEMORY_MAPPING,
        array_inline_threshold: int = DEFAULT_ARRAY_INLINE_THRESHOLD,
    ):
        if write_kwargs is None:
            write_kwargs = dict(all_array_compression=compression)

        if asdffile_kwargs is None:
            asdffile_kwargs = asdf_open_memory_mapping_kwarg(memmap=memmap)

        if "copy_arrays" in asdffile_kwargs and _asdf_version >= ("3", "1", "0"):
            msg = f"""Using deprecated option `copy_arrays` for asdf version
            {_asdf_version}. Use `memmap` instead."""
            warnings.warn(
                msg,
                WeldxDeprecationWarning,
                stacklevel=2,
            )
            asdffile_kwargs["memmap"] = not asdffile_kwargs["copy_arrays"]
            del asdffile_kwargs["copy_arrays"]

        # this parameter is now (asdf-2.8) a asdf.config parameter, so we store it here.
        self._array_inline_threshold = array_inline_threshold

        # TODO: ensure no mismatching args for compression and memmap.
        self._write_kwargs = write_kwargs
        self._asdffile_kwargs = asdffile_kwargs

        if "custom_schema" in asdffile_kwargs:
            custom_schema = asdffile_kwargs.pop("custom_schema", None)
        self._handle_custom_schema(custom_schema)

        if mode not in ("r", "rw"):
            raise ValueError(
                f'invalid mode "{mode}" given. Should be one of "r", "rw".'
            )
        elif tree and mode != "rw":
            raise RuntimeError(
                "You cannot pass a tree (to be written) on a read-only file."
            )
        self._mode = mode
        self.sync_upon_close = bool(sync) & (self.mode == "rw")
        self.software_history_entry = software_history_entry

        file_like, new_file_created = self._handle_file_input(
            filename_or_file_like, mode
        )

        # If we have data to write, we do it first, so a WeldxFile is always in sync.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                category=AsdfWarning,
                action="error",
                message="asdf.extensions plugin from package weldx.*",
            )  # we turn asdf warnings about loading the weldx extension into an error.
            if tree or new_file_created:
                if self._schema_on_write:
                    asdffile_kwargs["custom_schema"] = self._schema_on_write
                asdf_file = self._write_tree(
                    file_like,
                    tree,
                    asdffile_kwargs,
                    write_kwargs,
                    new_file_created,
                )
                if isinstance(file_like, SupportsFileReadWrite):
                    file_like.seek(0)
            else:
                if self._schema_on_read:
                    asdffile_kwargs["custom_schema"] = self._schema_on_read
                asdf_file = open_asdf(
                    file_like,
                    mode=self.mode,
                    **asdffile_kwargs,
                )
        self._asdf_handle: AsdfFile = asdf_file

        # initialize protected key interface.
        super().__init__(protected_keys=_PROTECTED_KEYS, data=self._asdf_handle.tree)

    def _handle_file_input(self, filename_or_file_like, mode):
        new_file_created = False
        if filename_or_file_like is None:
            filename_or_file_like = BytesIO()
            new_file_created = True
            self._in_memory = True
            self._close = False  # we want buffers to be usable later on.
        elif isinstance(filename_or_file_like, get_args(types_path_like)):
            filename_or_file_like, new_file_created = self._handle_path(
                filename_or_file_like, mode
            )
            self._in_memory = False
            self._close = True  # close our own buffer
        elif isinstance(filename_or_file_like, get_args(types_file_like)):
            if isinstance(filename_or_file_like, BytesIO):
                self._in_memory = True
            else:
                self._in_memory = False

            if mode == "rw" and isinstance(
                filename_or_file_like, SupportsFileReadWrite
            ):
                with reset_file_position(filename_or_file_like):
                    filename_or_file_like.seek(0, io.SEEK_END)
                    new_file_created = filename_or_file_like.tell() == 0

            # the user passed a raw file handle, its their responsibility to close it.
            self._close = False
        else:
            _supported = WeldxFile.__init__.__annotations__["filename_or_file_like"]
            raise ValueError(
                f"Unsupported input type '{type(filename_or_file_like)}'."
                f" Should be one of {_supported}."
            )
        return filename_or_file_like, new_file_created

    def _handle_custom_schema(self, custom_schema):
        self._schema_on_read = self._schema_on_write = None

        def resolve_schema(schema):
            if schema is None:
                return

            _custom_schema_path = pathlib.Path(schema)
            if not _custom_schema_path.exists():
                try:
                    schema = get_schema_path(schema)
                except ValueError as ve:
                    raise ValueError(
                        f"provided custom_schema '{schema}' does not exist."
                    ) from ve
            return schema

        if isinstance(custom_schema, (list, tuple)):
            if len(custom_schema) == 2:
                self._schema_on_read = resolve_schema(custom_schema[0])
                self._schema_on_write = resolve_schema(custom_schema[1])
            else:
                raise ValueError("custom_schema should be sequence of length two.")
        elif isinstance(custom_schema, types_path_like.__args__):
            self._schema_on_read = self._schema_on_write = resolve_schema(custom_schema)

    @contextmanager
    def _config_context(self, **kwargs):
        # Temporarily set (default) options in asdf.config_context. This is useful
        # during writing/updating data.
        if (
            "array_inline_threshold" not in kwargs
            or kwargs["array_inline_threshold"] is None
        ):
            kwargs["array_inline_threshold"] = self._array_inline_threshold

        with config_context() as config:
            for k, v in kwargs.items():
                setattr(config, k, v)
            yield

    def _write_tree(
        self, filename_or_path_like, tree, asdffile_kwargs, write_kwargs, created
    ) -> AsdfFile:
        # cases:
        # 1. file is empty (use write_to)
        # 1.a empty buffer, iobase
        # 1.b path pointing to empty (new file)
        # 2. file exists, but should be updated with new tree
        if created:
            asdf_file = asdf.AsdfFile(tree=tree, **asdffile_kwargs)
            if asdf.__version__ >= "4.0.0":
                # Starting in asdf 4 passing a tree to AsdfFile will
                # not automatically validate the tree. For those versions
                # we call validate here to check the tree.
                asdf_file.validate()
            with self._config_context():
                asdf_file.write_to(filename_or_path_like, **write_kwargs)
            # Now set the file handle to the newly created AsdfFile instance.
            # That way the handle will be closed by the asdf library later on.
            generic_file = generic_io.get_file(filename_or_path_like, mode="rw")
            asdf_file._fd = generic_file
        else:
            if self._mode != "rw":
                raise RuntimeError("inconsistent mode, need to write data.")
            asdf_file = open_asdf(filename_or_path_like, **asdffile_kwargs, mode="rw")
            asdf_file.tree = tree
            with self._config_context():
                asdf_file.update(**write_kwargs)
        return asdf_file

    @property
    def mode(self) -> str:
        """File operation mode.

        This is either reading or reading/writing mode, one of "r" or "rw"."""
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
            raise RuntimeError(f"file {filename} has be created, but mode is 'r'.")

        if mode == "rw":
            if not exists:
                new_file_created = True
                real_mode = "bx+"  # binary, exclusive creation, e.g. raise if exists.
            else:
                generic_file = generic_io.get_file(filename, mode="r")
                file_type = get_file_type(generic_file)
                if not file_type == FileType.ASDF:
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
        Let us define a custom software entry and use it during file creation.

        >>> import weldx
        >>> software = dict(name="MyFancyPackage", author="Me",
        ...        homepage="https://startpage.com", version="1.0")
        >>> f = weldx.WeldxFile(software_history_entry=software)
        >>> f.add_history_entry("we made some change")
        >>> f.history #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
         [{'description': 'we made some change',
           'time': datetime.datetime(...),
           'software': {'name': 'MyFancyPackage', 'author': 'Me',
           'homepage': 'https://startpage.com', 'version': '1.0'}}]

        We can also change the software on the fly:
        >>> software_new = dict(name="MyTool", author="MeSoft",
        ...                     homepage="https://startpage.com", version="1.0")
        >>> f.add_history_entry("another change using mytool", software_new)

        Lets inspect the last history entry:

        >>> f.history[-1] #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        {'description': 'another change using mytool',
         'time': datetime.datetime(...),
         'software': {'name': 'MyTool', 'author': 'MeSoft',
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
            if not isinstance(value, dict):
                raise ValueError("expected a dictionary type")
            try:
                test = AsdfFile(tree=dict(software=Software(value)))
                test.validate()
            except ValidationError as ve:
                raise ValueError(f"Given value has invalid format: {ve}") from ve
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
        pad_blocks: float | bool = False,
        include_block_index: bool = True,
        version: str = None,
        **kwargs,
    ):
        """Get this docstring overwritten by AsdfFile.update."""
        with self._config_context(**kwargs):
            self._asdf_handle.update(
                all_array_storage=all_array_storage,
                all_array_compression=all_array_compression,
                pad_blocks=pad_blocks,
                include_block_index=include_block_index,
                version=version,
            )

    sync.__doc__ = AsdfFile.update.__doc__

    def keys(self) -> Set:
        """Return a set of keys/attributes stored in this file.

        Returns
        -------
        KeysView :
            all keys stored at the root of this file.
        """
        return super().keys()

    @classmethod
    def fromkeys(cls, iterable, default=None) -> WeldxFile:
        """Create a new file with keys from iterable and values set to value.

        Parameters
        ----------
        iterable :
            list of key names
        default :
            default value to fill the keys with.

        Examples
        --------
        >>> from collections import defaultdict
        >>> wx = WeldxFile.fromkeys(("TCP", "wx_meta", "process"))
        >>> wx["TCP"], wx["wx_meta"], wx["process"]
        (None, None, None)
        """
        tree = dict.fromkeys(iterable, default)
        return WeldxFile(tree=tree, mode="rw")

    def values(self) -> ValuesView:
        """Return a view list like object of the file content.

        Returns
        -------
        ValuesView :
            a view on the values.

        """
        return super().values()

    def get(self, key, default=None):
        """Get data attached to given key from file.

        Parameters
        ----------
        key :
            The name of the data.
        default :
            The default is being returned in case the given key cannot be found.

        Raises
        ------
        KeyError
            Raised if the given key cannot be found and no default was provided.
        """
        return super().get(key, default=default)

    def update(self, mapping: Mapping | Iterable = (), **kwargs: Any):
        """Update this file from mapping or iterable mapping and kwargs.

        Parameters
        ----------
        mapping :
            a key, value paired like structure or an iterable of keys.
        kwargs :
            any key value pair you can think of.

        Notes
        -----
        Let the mapping parameter denote E, and let the kwargs parameter denote F.
        If present and has a .keys() method, does:     for k in E: D[k] = E[k]
        If E present and lacks .keys() method, does:     for (k, v) in E: D[k] = v
        In either case, this is followed by: for k, v in F.items(): D[k] = v

        Examples
        --------
        Let us update one weldx file with another one.
        >>> a = WeldxFile(tree=dict(x=42), mode="rw")
        >>> b = WeldxFile(tree=dict(x=23, y=0), mode="rw")

        We pass all data of ``b`` to ``a`` while adding another key value pair.

        >>> a.update(b, foo="bar")

        Or we can update with an iterable of key, value tuples.
        >>> data = [('x', 0), ('y', -1)]
        >>> a.update(data)
        >>> a["foo"], a["x"], a["y"]
        ('bar', 0, -1)

        Another possibility is to directly pass keyword arguments.
        >>> a.update(x=-1, z=42)
        >>> a["x"], a["z"]
        (-1, 42)
        """
        super().update(mapping, **kwargs)

    def items(self) -> Set[tuple[Any, Any]]:
        """Return a set-like object providing a view on this file's items.

        Returns
        -------
        ItemsView :
            view on items.
        """
        return super().items()

    def setdefault(self, key, default=None):
        """Set a default for given key.

        The passed default object will be returned in case the requested key
        is not present in the file.

        Parameters
        ----------
        key :
            key name
        default :
            object to return in case key is not present in the file.
        """
        super().setdefault(key, default=default)

    def pop(self, key, default=None) -> Any:
        """Get and remove the given key from the file.

        Parameters
        ----------
        key :
            key name
        default :
            object to return in case key is not present in the file.

        Returns
        -------
        object :
            the object value of given key. If key was not found, return the default.

        Examples
        --------
        Let us remove and store some data from a weldx file.
        >>> a = WeldxFile(tree=dict(x=42, y=0), mode="rw")
        >>> x = a.pop("x")
        >>> x
        42
        >>> "x" not in a.keys()
        True
        """
        return super().pop(key, default=default)

    def popitem(self) -> tuple[Hashable, Any]:
        """Remove the item that was last inserted into the file.

        Notes
        -----
        The assumption of an ordered dictionary, that this method returns
        the key inserted lastly, does not hold necessarily. E.g. if the file has been
        written to disk and is loaded again the previous order has been lost eventually.

        Returns
        -------
        object :
            the last item.
        """
        return super().popitem()

    def add_history_entry(self, change_desc: str, software: dict = None) -> None:
        """Add an history_entry to the file.

        Parameters
        ----------
        change_desc :
            Description of the change made.
        software :
            Optional software used to make the change.

        Notes
        -----
        The software entry will be inferred from the constructor or, if not defined,
        from ``software_history_entry``.

        """
        if software is None:
            software = self.software_history_entry
        self._asdf_handle.add_history_entry(change_desc, software)

    @property
    def history(self) -> list:
        """Return a list of all history entries in this file."""
        return self._asdf_handle.get_history_entries().copy()

    @property
    def asdf_library(self) -> dict:
        """Get version information about the ASDF library lastly modifying this file."""
        return self._asdf_handle["asdf_library"].copy()

    @property
    def custom_schema(self) -> str | tuple[str | None] | None:
        """Return schema used to validate the structure and types of the tree."""
        if self._schema_on_read == self._schema_on_write:
            return self._schema_on_read

        return self._schema_on_read, self._schema_on_write

    @property
    # TODO: should we actually expose this? It allows advanced operations for adults.
    # TODO: return type is also not guaranteed to be IOBase, right?
    def file_handle(self) -> IOBase:
        """Return the underlying file handle, use with CARE."""
        if self._asdf_handle._closed:
            raise RuntimeError("closed file, cannot access file handle.")
        return self._asdf_handle._fd._fd

    def copy(
        self,
        filename_or_file_like: types_path_and_file_like | None = None,
        overwrite: bool = False,
    ) -> WeldxFile:
        """Take a copy of this file.

        Depending on the underlying file type this does several different things.

        Parameters
        ----------
        filename_or_file_like :
            The desired output file. If no file is given, an in-memory file
            will be created.
        overwrite :
            If ``filename_or_file_like`` points to a path or filename which already
            exists, this flag determines if it would be overwritten or not.

        Returns
        -------
        WeldxFile :
            The new instance with the copied content.
        """
        # check if we would overwrite an existing path
        if isinstance(filename_or_file_like, (str, pathlib.Path)):
            try:
                filename_or_file_like, _ = self._handle_path(
                    filename_or_file_like, mode="rw"
                )
            except FileExistsError:
                if not overwrite:
                    raise

        # TODO: we could try to optimize this, e.g. avoid the extra copy?
        file = self.write_to(filename_or_file_like)
        wx = WeldxFile(
            file,
            mode=self.mode,
            custom_schema=self.custom_schema,
            asdffile_kwargs=self._asdffile_kwargs,
            write_kwargs=self._write_kwargs,
            sync=self.sync_upon_close,
            software_history_entry=self.software_history_entry,
            array_inline_threshold=self._array_inline_threshold,
        )
        return wx

    def as_attr(self) -> MutableMapping:
        """Return the Weldx dictionary as an attributed object.

        Returns
        -------
        MutableMapping :
            This dictionary wrapped such, that all of its keys can be accessed as
            properties.

        Examples
        --------
        >>> tree = dict(wx_meta={"welder": "Nikolai Nikolajewitsch Benardos"})
        >>> wf = WeldxFile(tree=tree, mode="rw")
        >>> wfa = wf.as_attr()
        >>> wfa.wx_meta.welder
        'Nikolai Nikolajewitsch Benardos'

        We can also change the data easily

        >>> wfa.wx_meta.welder = "Myself"
        >>> wfa.wx_meta.welder
        'Myself'
        """

        class AttrDict(dict):
            def __init__(self, iterable, **kwargs):
                super().__init__(iterable, **kwargs)
                for key, value in self.items():
                    if isinstance(value, Mapping):
                        self.__dict__[key] = AttrDict(value)
                    else:
                        self.__dict__[key] = value

        return AttrDict(self)

    def write_to(
        self,
        fd: types_path_and_file_like | None = None,
        array_inline_threshold=None,
        **write_args,
    ) -> types_path_and_file_like | None:
        """Write current contents to given file name or file type.

        Parameters
        ----------
        fd :
            May be a string path to a file, or a Python file-like
            object. If a string path, the file is automatically
            closed after writing. If `None` is given, write to a new buffer.

        array_inline_threshold :
            arrays below this threshold will be serialized as string, if
            larger as binary block.

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
        types_path_and_file_like :
            The given input file name or a buffer, in case the input it was omitted.

        """
        if fd is None:
            fd = BytesIO()

        # if no args are given, we use the specifications given in the constructor.
        if not write_args:
            write_args = self._write_kwargs

        with self._config_context(array_inline_threshold=array_inline_threshold):
            self._asdf_handle.write_to(fd, **write_args)

        if isinstance(fd, types_file_like.__args__):
            fd.seek(0)
        return fd

    def header(
        self,
        use_widgets: bool = None,
        path: tuple = None,
        _interactive: bool | None = None,
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
        path :
            tuple representing the lookup path in the yaml/asdf tree
        _interactive :
            Should not be set.
        """
        return _HeaderVisualizer(self._asdf_handle).show(
            use_widgets=use_widgets, path=path, _interactive=_interactive
        )

    def _ipython_display_(self):
        # This will be called in Jupyter Lab, and myst-nb execution,
        # but not in a plain notebook.
        from IPython.display import display

        # Determine widget usage by runtime environment, by passing None.
        display(self.header(use_widgets=None, _interactive=True))

    def info(
        self,
        max_rows: int = None,
        max_length: int = None,
        show_values: bool = False,
        path: tuple = None,
    ):
        """Print the content to the stdout.

        Parameters
        ----------
        max_rows :
            The maximum number of rows that will be printed. If rows are cut, a
            corresponding message will be printed
        max_length :
            The maximum line length. Longer lines will be truncated
        show_values :
            Set to `True` if primitive values should be displayed
        path
            tuple representing the lookup path in the yaml/asdf tree

        """
        tree = {
            key: value
            for key, value in self._asdf_handle.tree.items()
            if key not in ["asdf_library", "history"]
        }
        if path is not None:
            tree = get_path(tree, path)
        asdf.info(tree, max_rows=max_rows, max_cols=max_length, show_values=show_values)


class _HeaderVisualizer:
    def __init__(self, asdf_handle: AsdfFile):
        # TODO: this ain't thread-safe!
        def _fake_copy(x, _):  # take a copy of the handle to avoid side effects!
            return x

        copy._deepcopy_dispatch[np.ndarray] = _fake_copy
        try:
            # asdffile takes a deepcopy by default, so we fake the deep copy method of
            # ndarray to avoid bloating memory.
            self._asdf_handle = asdf_handle.copy()
        finally:
            del copy._deepcopy_dispatch[np.ndarray]

    def show(
        self, use_widgets=None, path=None, _interactive=None
    ) -> None | IPython.display.HTML | IPython.display.JSON:  # noqa: F821
        if _interactive is None:
            _interactive = is_interactive_session()
        if use_widgets is None:
            use_widgets = is_jupyterlab_session()

        # We write the current tree to a buffer __without__ any binary data attached.
        from unittest.mock import patch

        buff = BytesIO()
        with patch("asdf.tags.core.ndarray.numpy_array_to_list", lambda x: []):
            self._asdf_handle.write_to(buff, all_array_storage="inline")
        buff.seek(0)

        # automatically determine if this runs in an interactive session.
        # These methods return an IPython displayable object
        # (passed to IPython.display()).
        if _interactive:
            return self._show_interactive(use_widgets=use_widgets, buff=buff, path=path)
        else:
            self._show_non_interactive(buff=buff)

    @staticmethod
    def _show_interactive(
        use_widgets: bool, buff: BytesIO, path: tuple = None
    ) -> IPython.display.HTML | IPython.display.JSON:  # noqa: F821
        from weldx.asdf.util import notebook_fileprinter

        if use_widgets:
            result = view_tree(buff, path=path)
        else:
            result = notebook_fileprinter(buff)
        return result

    @staticmethod
    def _show_non_interactive(buff: BytesIO):
        print(get_yaml_header(buff))  # noqa: T201
