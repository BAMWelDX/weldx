from io import BytesIO

import asdf

from weldx.asdf.extension import WeldxAsdfExtension, WeldxExtension


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
        Extensions and copy_arrays=True are always set.

    Returns
    -------
    dict

    """
    # Write the data to buffer
    if open_kwargs is None:
        open_kwargs = {}
    if write_kwargs is None:
        write_kwargs = {}
    if asdffile_kwargs is None:
        asdffile_kwargs = {}

    with asdf.AsdfFile(
        tree, extensions=[WeldxExtension(), WeldxAsdfExtension()], **asdffile_kwargs
    ) as ff:
        buff = BytesIO()
        ff.write_to(buff, **write_kwargs)
        buff.seek(0)

    # read back data from ASDF file contents in buffer
    with asdf.open(
        buff,
        extensions=[WeldxExtension(), WeldxAsdfExtension()],
        copy_arrays=True,
        **open_kwargs
    ) as af:
        data = af.tree
    return data
