import pathlib
from collections import UserDict
from io import BytesIO, IOBase
from typing import Optional, Protocol, Union, runtime_checkable, Mapping

import asdf
from asdf import AsdfFile

from weldx.asdf import WeldxAsdfExtension, WeldxExtension


@runtime_checkable
class SupportsFileReadOnly(Protocol):
    """Type interface for read()."""

    __slots__ = ()

    def read(self):
        raise NotImplementedError


@runtime_checkable
class SupportsFileReadWrite(Protocol):
    """Type interface for read, write and seeking."""

    __slots__ = ()

    def read(self):
        raise NotImplementedError

    def write(self, *args):
        raise NotImplementedError

    def tell(self):
        raise NotImplementedError

    def seek(self, *args):
        raise NotImplementedError


types_file_like = Union[IOBase, SupportsFileReadOnly, SupportsFileReadWrite]
types_path_and_file_like = Union[str, pathlib.Path, types_file_like]


class WeldxFile(UserDict):
    """This exposes an ASDF file as a dictionary like object.

    Parameters
    ----------
    filename_or_file_like :
        a path to a weldx file or file handle like to read/write data from.
        If None is passed, an in-memory file will be created.
    mode :
        reading or reading/writing mode: "r" or "rw"
    asdf_args :
        see `asdf.open` for reference.
    tree :
        an optional dictionary to write to the file.
    sync :
        If True, the changes to file will be written upon closing this. This is only
        relevant, if the file has been opened in write mode.
    software_history_entry :
        a optional dictionary which will be used to add history entries upon
        modification of the file. It has to provide the following keys:
        ("name", "author", "homepage", "version")

    """

    def __init__(
        self,
        filename_or_file_like: Union[None, str, pathlib.Path, types_file_like],
        mode: str = "r",
        asdf_args: Mapping = None,
        tree: Mapping = None,
        sync: bool = True,
        software_history_entry: Mapping = None,
    ):
        # TODO: default asdf_args?
        # e.g: asdf_args = {"copy_arrays": True}
        if asdf_args is None:
            asdf_args = {}

        self._quality_standard = (
            asdf_args["custom_schema"] if "custom_schema" in asdf_args else None
        )

        if "mode" in asdf_args:
            raise ValueError("mode not allowed in asdf_args, but only mode")
        self.mode = mode

        self.sync_upon_close = bool(sync)

        # let asdf.open handle/raise exceptions
        extensions = [WeldxExtension(), WeldxAsdfExtension()]

        # If we have data to write, we do it first, so a WeldxFile is always in sync.
        if tree:
            asdf_file = AsdfFile(tree=tree, extensions=extensions)
            if filename_or_file_like is None:
                filename_or_file_like = BytesIO()
            asdf_file.write_to(filename_or_file_like, **asdf_args)
            if isinstance(filename_or_file_like, SupportsFileReadWrite):
                filename_or_file_like.seek(0)

        asdf_file = asdf.open(
            filename_or_file_like,
            mode=mode,
            extensions=extensions,
            **asdf_args,
        )
        self._asdf_handle: asdf.AsdfFile = asdf_file

        # UserDict interface: we want to store a reference to the tree, but the ctor
        # of UserDict takes a copy, so we do it manually here.
        super(WeldxFile, self).__init__()
        self.data = self._asdf_handle.tree

        if software_history_entry is None:
            from weldx import __version__ as version

            self._DEFAULT_SOFTWARE_ENTRY = {
                "name": "weldx",
                "author": "BAM",
                "homepage": "https://www.bam.de/Content/EN/Projects/WelDX/weldx.html",
                "version": version,
            }
        else:
            self._DEFAULT_SOFTWARE_ENTRY = software_history_entry

    @classmethod
    def from_tree(cls, tree: Mapping, **asdf_kwargs):
        """creates a new WeldxFile from given dictionary backed by a memory file.

        Parameters
        ----------
        tree :
            a dictionary like object to write to the new WeldxFile.

        asdf_kwargs :
            additional ASDF keywords to pass to the WeldxFile file creation.

        """
        buff = BytesIO()
        asdf_file = AsdfFile(tree=tree, **asdf_kwargs)
        asdf_file.write_to(buff)
        buff.seek(0)
        return cls(
            buff,
        )

    def __enter__(self):
        self._asdf_handle.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close this file and sync it, if mode is read/write."""
        if self.mode == "rw" and self.sync_upon_close:
            self._asdf_handle.update()
        self._asdf_handle.close()

    # TODO: this updates the tree, but should it also be synched to disk directly?
    # this could be inefficient if lots of updates are performed,
    # but also adds more safety.
    #    def update(self, __m: Mapping[_KT, _VT], **kwargs: _VT) -> None: ...
    def update(
        self,
        data: Mapping = None,
        change_desc: str = None,
        software: dict = None,
    ):
        if data is None:
            return
        # first update the tree (this could fail, right?!)
        # then write the history entry
        # then update the asdf handle
        try:
            self._asdf_handle.tree.update(data)
        except BaseException as e:
            print(e)
            self.add_history_entry(change_desc, software=software)

    def sync(
        self,
        all_array_storage=None,
        all_array_compression="input",
        auto_inline=None,
        pad_blocks=False,
        include_block_index=True,
        version=None,
    ):
        """
        Update the file on disk in place.

        Parameters
        ----------
        all_array_storage : string, optional
            If provided, override the array storage type of all blocks
            in the file immediately before writing.  Must be one of:

            - ``internal``: The default.  The array data will be
              stored in a binary block in the same ASDF file.

            - ``external``: Store the data in a binary block in a
              separate ASDF file.

            - ``inline``: Store the data as YAML inline in the tree.

        all_array_compression : string, optional
            If provided, set the compression type on all binary blocks
            in the file.  Must be one of:

            - ``''`` or `None`: No compression.

            - ``zlib``: Use zlib compression.

            - ``bzp2``: Use bzip2 compression.

            - ``lz4``: Use lz4 compression.

            - ``input``: Use the same compression as in the file read.
              If there is no prior file, acts as None

        auto_inline : int, optional
            When the number of elements in an array is less than this
            threshold, store the array as inline YAML, rather than a
            binary block.  This only works on arrays that do not share
            data with other arrays.  Default is 0.

        pad_blocks : float or bool, optional
            Add extra space between blocks to allow for updating of
            the file.  If `False` (default), add no padding (always
            return 0).  If `True`, add a default amount of padding of
            10% If a float, it is a factor to multiple content_size by
            to get the new total size.

        include_block_index : bool, optional
            If `False`, don't include a block index at the end of the
            file.  (Default: `True`)  A block index is never written
            if the file has a streamed block.

        version : str, optional
            The ASDF version to write out.  If not provided, it will
            write out in the latest version supported by asdf.
        """
        self._asdf_handle.update(
            all_array_storage=all_array_storage,
            all_array_compression=all_array_compression,
            auto_inline=auto_inline,
            pad_blocks=pad_blocks,
            include_block_index=include_block_index,
            version=version,
        )

    def add_history_entry(self, change_desc: str, software: dict = None) -> None:
        """adds an history_entry to the file.

        Parameters
        ----------
        change_desc :
            description of the change made.
        software :
            software used to make the change. See `WeldxFile.__init__` software.
        """
        if software is None:
            software = self._DEFAULT_SOFTWARE_ENTRY
        self._asdf_handle.add_history_entry(change_desc, software)

    @property
    def history(self) -> list:
        """Return a list of all history entries in this file."""
        return self._asdf_handle.get_history_entries()

    @property
    def quality_standard(self) -> Optional[str]:
        return self._quality_standard

    def __setitem__(self, key, value):
        # FIXME: this only handles top level write access! So we got to wrap the ASDFEntries?
        # FIXME: is not called upon weldxfile['foo'] = 'bar' ...#
        # so better make it explicit, if users change something, they are responsible
        # to add a proper history entry.
        import textwrap

        short_value_str = textwrap.shorten(str(value), 30)
        change_desc = f"set '{key}' to '{short_value_str}'"
        self.add_history_entry(change_desc)
        self._asdf_handle[key] = value

    def __delitem__(self, item):
        del self._asdf_handle.tree[item]
        self.add_history_entry(f"deleted key {item}")
        self._asdf_handle.update()

    @property
    # TODO: should we actually expose this? It allows advanced operations for adults.
    def file_handle(self) -> IOBase:
        if self._asdf_handle._closed:
            raise RuntimeError("closed file, cannot access file handle.")
        return self._asdf_handle._fd._fd

    @property
    def as_attr(self):
        import attrdict

        return attrdict.AttrDict(self)

    def write_to(self, fd: Optional[types_path_and_file_like] = None, **asdf_args):
        """write current weldx file to given file name.

        Parameters
        ----------
        fd : str, pathlib.Path or file-like object
            May be a string path to a file, or a Python file-like
            object.  If a string path, the file is automatically
            closed after writing.

        asdf_args :
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

        # TODO: if no args are given, should we use the args given in ctor?
        self._asdf_handle.write_to(fd, **asdf_args)

        if created:
            fd.seek(0)
        return fd

    # def _repr_html_(self) -> Optional[str]:
    #    from weldx.asdf.util import notebook_fileprinter
    #    return notebook_fileprinter(self.copy_to_buffer()).__html__()

    def _repr_json_(self):
        # fake this to be an instance of dict for json repr
        from unittest.mock import patch

        from weldx.asdf.util import asdf_json_repr

        # with patch("kisa.weldx_file.WeldxFile.__class__", dict):
        return asdf_json_repr(self.copy_to_buffer())  # ._repr_json_()
