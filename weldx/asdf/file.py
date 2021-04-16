import pathlib
from collections import UserDict
from collections.abc import MutableMapping
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


class WeldxFile(UserDict):
    """This exposes an ASDF file as a dictionary like object.

    Parameters
    ----------
    filename_or_file_like :
        a path to a weldx file or file handle like to read data from.
    mode :
        reading or reading/writing mode: "r" or "rw"
    asdf_args :
        see `asdf.open` for reference.
    sync :
        If True, the changes to file will be written upon closing this. This is only
        relevant, if the file has been opened in write mode.
    history_software_entry :
        a optional dictionary which will be used to add history entries upon
        modification of the file. It has to provide the following keys:
        ("name", "author", "homepage", "version")

    """

    def __init__(
        self,
        filename_or_file_like: Union[str, pathlib.Path, types_file_like],
        mode="r",
        asdf_args=None,
        sync=True,
        history_software_entry=None,
    ):
        if asdf_args is None:
            asdf_args = {}

        # default asdf_kwargs?
        # TODO: open_kwargs = {"copy_arrays": True}

        self._quality_standard = (
            asdf_args["custom_schema"] if "custom_schema" in asdf_args else None
        )

        if "mode" in asdf_args:
            raise ValueError("mode not allowed in asdf_args, but only mode")
        self.mode = mode

        self.sync = bool(sync)

        # let asdf.open handle/raise exceptions
        self._asdf_handle: asdf.AsdfFile = asdf.open(
            filename_or_file_like,
            mode=mode,
            extensions=[WeldxExtension(), WeldxAsdfExtension()],
            **asdf_args,
        )
        super(WeldxFile, self).__init__(dict=self._asdf_handle.tree)

        if history_software_entry is None:
            from weldx import __version__ as version

            self._DEFAULT_SOFTWARE_ENTRY = {
                "name": "weldx",
                "author": "BAM",
                "homepage": "https://www.bam.de/Content/EN/Projects/WelDX/weldx.html",
                "version": version,
            }

    @classmethod
    def from_tree(cls, tree: dict, **asdf_kwargs):
        buff = BytesIO()
        asdf_file = AsdfFile(tree, **asdf_kwargs)
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
        if self.mode == "rw" and self.sync:
            self._asdf_handle.update()
        self._asdf_handle.close()

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
            self._asdf_handle.update()

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

    # dict interface
    #def keys(self):
    #    return self._asdf_handle.keys()

    #def __getitem__(self, key):
    #    return self._asdf_handle[key]

    def __setitem__(self, key, value):
        # FIXME: this only handles top level write access! So we got to wrap the ASDFEntries?
        # FIXME: is not called upon weldxfile['foo'] = 'bar' ...
        import textwrap

        short_value_str = textwrap.shorten(str(value), 30)
        change_desc = f"set '{key}' to '{short_value_str}'"
        self.add_history_entry(change_desc)
        self._asdf_handle[key] = value

    # def __contains__(self, item):
    #     return item in self._asdf_handle

    def __delitem__(self, item):
        del self._asdf_handle.tree[item]
        self.add_history_entry(f"deleted key {item}")
        self._asdf_handle.update()

    #def __iter__(self):
    #    return iter(self._asdf_handle.tree)

    #def __len__(self):
    #    return len(self._asdf_handle.tree)

    @property
    def file_handle(self) -> IOBase:
        return self._asdf_handle._fd._fd

    def copy_to_buffer(self) -> BytesIO:
        res = BytesIO()
        self._asdf_handle.write_to(res)
        res.seek(0)

        return res

    @property
    def as_attr(self):
        import attrdict

        return attrdict.AttrDict(self)

    def write_to(self, fn: Optional[str] = None, **asdf_args):
        """write current weldx file to given file name.


        Parameters
        ----------
        fn :
            When no file name is given, write to a buffer and return it.

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
        if fn is None:
            fn = BytesIO()

        # TODO: if no args are given, should we use the args given in ctor?
        self._asdf_handle.write_to(fn, **asdf_args)

        if fn is not None:
            fn.seek(0)
        return fn

    # def _repr_html_(self) -> Optional[str]:
    #    from weldx.asdf.util import notebook_fileprinter
    #    return notebook_fileprinter(self.copy_to_buffer()).__html__()

    def _repr_json_(self):
        # fake this to be an instance of dict for json repr
        from unittest.mock import patch

        from weldx.asdf.util import asdf_json_repr

        # with patch("kisa.weldx_file.WeldxFile.__class__", dict):
        return asdf_json_repr(self.copy_to_buffer())  # ._repr_json_()
