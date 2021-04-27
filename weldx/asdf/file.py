"""The WeldxFile class wraps creation and updating of ASDF files."""

import pathlib
from collections import MutableMapping, UserDict
from io import BytesIO, IOBase
from typing import Mapping, Optional, Tuple, Union

import asdf
from asdf import AsdfFile
from asdf.asdf import is_asdf_file

from weldx.types import SupportsFileReadWrite, types_file_like, types_path_and_file_like
from weldx.asdf import WeldxAsdfExtension, WeldxExtension
from weldx.asdf.util import get_yaml_header

__all__ = [
    "WeldxFile",
]


class WeldxFile(UserDict):
    """This exposes an ASDF file as a dictionary like object.

    Parameters
    ----------
    filename_or_file_like :
        A path to a weldx file or file handle like to read/write data from.
        If None is passed, an in-memory file will be created.
    mode :
        Reading or reading/writing mode: "r" or "rw".
    open_kwargs :
        Keyword arguments to pass to asdf.open.
        See `asdf.open` for reference.
    write_kwargs :
        Keyword arguments to pass to AsdfFile.write_to.
        See `asdf.AsdfFile.open` for reference.
    tree :
        An optional dictionary to write to the file.
    sync :
        If True, the changes to file will be written upon closing this. This is only
        relevant, if the file has been opened in write mode.
    custom_schema :
        a path-like object to a custom schema which validates the tree.
    software_history_entry :
        An optional dictionary which will be used to add history entries upon
        modification of the file. It has to provide the following keys:
        ("name", "author", "homepage", "version")

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
    ):
        # TODO: default asdf_args?
        # e.g: asdf_args = {"copy_arrays": True}
        if write_kwargs is None:
            write_kwargs = {}
        self._write_kwargs = write_kwargs

        if asdffile_kwargs is None:
            asdffile_kwargs = {}

        self._asdffile_kwargs = asdffile_kwargs
        self._custom_schema = custom_schema

        if mode not in ("r", "rw"):
            raise ValueError(
                f'invalid mode "{mode}" given. Should be one of "r", "rw".'
            )
        self.mode = mode
        self.sync_upon_close = bool(sync)
        self.software_history_entry = software_history_entry

        new_file_created = False
        if filename_or_file_like is None:
            filename_or_file_like = BytesIO()
            new_file_created = True
        elif isinstance(filename_or_file_like, (str, pathlib.Path)):
            filename_or_file_like, new_file_created = self._handle_path(
                filename_or_file_like, mode
            )
        elif isinstance(filename_or_file_like, types_file_like.__args__):
            pass
        else:
            _supported = WeldxFile.__init__.__annotations__["filename_or_file_like"]
            raise ValueError(
                f"Unsupported input type '{type(filename_or_file_like)}'."
                f" Should be one of {_supported}."
            )

        extensions = [WeldxExtension(), WeldxAsdfExtension()]
        # If we have data to write, we do it first, so a WeldxFile is always in sync.
        if tree or new_file_created:
            asdf_file = AsdfFile(tree=tree, extensions=extensions)
            asdf_file.write_to(filename_or_file_like, **write_kwargs)
            if isinstance(filename_or_file_like, SupportsFileReadWrite):
                filename_or_file_like.seek(0)

        asdf_file = asdf.open(
            filename_or_file_like,
            mode=mode,
            extensions=extensions,
            **asdffile_kwargs,
        )
        self._asdf_handle: asdf.AsdfFile = asdf_file

        # UserDict interface: we want to store a reference to the tree, but the ctor
        # of UserDict takes a copy, so we do it manually here.
        super(WeldxFile, self).__init__()
        self.data = self._asdf_handle.tree

    @staticmethod
    def _handle_path(filename_or_file_like, mode):
        new_file_created = False
        # TODO: simplify
        exists = pathlib.Path(filename_or_file_like).exists()
        if not exists and mode == "r":
            raise RuntimeError(
                f"file {filename_or_file_like} has be created," " but mode is 'r'."
            )

        if mode == "rw":
            if not exists:
                new_file_created = True
                real_mode = "bx+"  # binary, exclusive creation, e.g. raise if exists.
            else:
                if not is_asdf_file(filename_or_file_like):
                    raise FileExistsError(
                        f"given file {filename_or_file_like}"
                        " is not an ASDF file and could be overwritten."
                    )
                real_mode = "br+"
        else:
            real_mode = "rb"

        # create file handle
        filename_or_file_like = open(filename_or_file_like, mode=real_mode)
        return filename_or_file_like, new_file_created

    @property
    def software_history_entry(self):
        return self._DEFAULT_SOFTWARE_ENTRY

    @software_history_entry.setter
    def software_history_entry(self, value):
        if value is None:
            from weldx import __version__ as version

            self._DEFAULT_SOFTWARE_ENTRY = {
                "name": "weldx",
                "author": "BAM",
                "homepage": "https://www.bam.de/Content/EN/Projects/WelDX/weldx.html",
                "version": version,
            }
        else:
            # TODO: validate it here, or let asdf fail?
            self._DEFAULT_SOFTWARE_ENTRY = value

    def __enter__(self):
        self._asdf_handle.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close this file and sync it, if mode is read/write."""
        if self.mode == "rw" and self.sync_upon_close:
            self._asdf_handle.update(**self._write_kwargs)
        self._asdf_handle.close()

    # TODO: this updates the tree, but should it also be synced to disk directly?
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
    def custom_schema(self) -> Optional[str]:
        return self._custom_schema

    def __delitem__(self, item):
        del self._asdf_handle.tree[item]
        self.add_history_entry(f"deleted key {item}")
        self._asdf_handle.update()

    @property
    # TODO: should we actually expose this? It allows advanced operations for adults.
    # TODO: return type is also not guaranteed to be IOBase, right?
    def file_handle(self) -> IOBase:
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

        """
        import attrdict

        return attrdict.AttrDict(self)

    def write_to(self, fd: Optional[types_path_and_file_like] = None, **write_args):
        """write current weldx file to given file name.

        Parameters
        ----------
        fd : str, pathlib.Path or file-like object
            May be a string path to a file, or a Python file-like
            object.  If a string path, the file is automatically
            closed after writing.

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

        # TODO: if no args are given, should we use the args given in ctor?
        self._asdf_handle.write_to(fd, **write_args)

        if created:
            fd.seek(0)
        return fd

    def view_tree(self, path: Tuple = None):
        """Display YAML header

        Parameters
        ----------
        path :
            tuple representing the lookup path in the YAML/ASDF tree.

        Returns
        -------

        """
        from weldx.asdf.util import view_tree

        return view_tree(self.file_handle, path=path)

    def _header_as_tree(self):
        from ipytree import Node, Tree

        header = get_yaml_header(self.file_handle, parse=True)
        tree = Tree()
        for x in header:
            node = Node(x)
            tree.add_node(node)
            for child in x:
                child_node = Node(child)
                node.add_node(child_node)
        return tree

    def show_asdf_header(self, _interactive=None):
        # TODO: what if the file is out of sync? shall we enforce sync or write to temp file (which could be huge?)
        def _interactive():
            from weldx.asdf.util import notebook_fileprinter

            return notebook_fileprinter(self.file_handle)

        def _non_interactive():
            return get_yaml_header(self.file_handle, parse=True)

        # automatically determine if this runs in an interactive sesseion.
        if _interactive is None:

            def is_interactive():
                import __main__ as main

                return not hasattr(main, "__file__")

            if is_interactive():
                return _interactive()
            else:
                return self.show_asdf_header(_interactive=False)
        elif _interactive is False:
            return _non_interactive()
        else:
            return _interactive()

    def _repr_json_(self):
        return self.view_tree()
