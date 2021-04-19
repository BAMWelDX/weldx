import pathlib
import tempfile
from io import BytesIO, IOBase

import asdf
import pytest

from weldx import WeldxFile
from weldx.asdf.file import SupportsFileReadWrite


class ReadOnlyFile(IOBase):
    def __init__(self, tmpdir):
        fn = tempfile.mktemp(suffix='.asdf', dir=tmpdir)
        with open(fn, 'wb') as fh:
            asdf.AsdfFile(tree=dict(hi="there")).write_to(fh)
        self.file_read_only = open(fn, mode='rb')
        self.mode = "r"
        assert not self.closed

    def read(self, *args, **kwargs):
        return self.file_read_only.read(*args, **kwargs)

    def readable(self):
        return True


class WritableFile:
    """example of a class implementing SupportsFileReadWrite"""

    def __init__(self):
        self.to_wrap = BytesIO()

    def read(self, *args, **kwargs):
        return self.to_wrap.read(*args, **kwargs)

    def write(self, *args, **kwargs):
        return self.to_wrap.write(*args, **kwargs)

    def tell(self, *args, **kwargs):
        return self.to_wrap.tell(*args, **kwargs)

    def seek(self, *args, **kwargs):
        return self.to_wrap.seek(*args, **kwargs)

    def flush(self):
        """Simulate flush by rewinding to the beginning of the buffer."""
        self.seek(0)


def test_protocol_check(tmpdir):
    assert isinstance(WritableFile(), SupportsFileReadWrite)
    assert isinstance(BytesIO(), SupportsFileReadWrite)

    # real file:
    f = tempfile.mktemp(dir=tmpdir)
    assert isinstance(open(f, "w"), SupportsFileReadWrite)


# TODO: we do not need such a large test setup here!
@pytest.mark.usefixtures("single_pass_weld_asdf")
class TestWeldXFile:
    @pytest.fixture(autouse=True)
    def setUp(self, *args, **kwargs):
        copy_for_test = self.make_copy(self.single_pass_weld_file)
        self.fh = WeldxFile(copy_for_test, *args, **kwargs)

    # TODO: test all ctor arg combinations for file arg: Union[None, types_file_like]

    def test_from_physical_file(self, tmpdir):
        """tests WeldxFile() for str and pathlib.Path"""
        fn = tempfile.mktemp(suffix=".asdf", dir=tmpdir)
        self.fh.write_to(fn)
        assert WeldxFile(fn)["history"]
        path = pathlib.Path(fn)
        assert WeldxFile(path)["history"]

    def test_create_from_tree_create_buff(self):
        """tests wrapper creation from a dictionary."""
        tree = dict(foo="bar")
        # creates a buffer
        self.fh = WeldxFile(filename_or_file_like=None, tree=tree)
        new_file = self.make_copy(self.fh)
        assert WeldxFile(new_file)["foo"] == "bar"

    def test_create_from_tree_given_output_fn(self, tmpdir):
        """tests wrapper creation from a dictionary."""
        tree = dict(foo="bar")
        # should write to file
        fn = tempfile.mktemp(suffix=".asdf", dir=tmpdir)
        self.fh = WeldxFile(filename_or_file_like=fn, tree=tree)
        new_file = self.make_copy(self.fh)
        assert WeldxFile(new_file)["foo"] == "bar"

    def test_create_from_tree(self, tmpdir):
        """tests wrapper creation from a dictionary."""
        tree = dict(foo="bar")
        # TODO: actually this would be a case for pytests parameterization, but...
        # it doesn't support fixtures in parameterization yet.
        for fd in [BytesIO(), tempfile.mktemp(suffix=".asdf", dir=tmpdir)]:
            fh = WeldxFile(fd, tree=tree)
            fh["another"] = "entry"
            # sync to new file.
            new_file = self.make_copy(fh)
            # check tree changes have been written.
            fh2 = WeldxFile(new_file)
            assert fh2["foo"] == "bar"
            assert fh["another"] == "entry"

    def test_create_writable_protocol(self):
        f = WritableFile()
        WeldxFile(f, tree=dict(test="yes"))  # this should write the tree to f.
        new_file = self.make_copy(f.to_wrap)
        assert WeldxFile(new_file)["test"] == "yes"

    def test_create_readonly_protocol(self, tmpdir):
        f = ReadOnlyFile(tmpdir)
        WeldxFile(f)

    def make_copy(self, fh):
        buff = BytesIO()
        if isinstance(fh, WeldxFile):
            fh.write_to(buff)
        elif isinstance(fh, BytesIO):
            fh.seek(0)
            buff.write(fh.read())
        buff.seek(0)
        return buff

    def test_write_to(self):
        buff = self.fh.write_to()
        buff2 = self.make_copy(self.fh)
        assert buff.getvalue() == buff2.getvalue()

    def test_operation_on_closed(self):
        self.fh.close()
        assert self.fh["process"]

        # cannot access closed handles
        with pytest.raises(RuntimeError):
            self.fh.file_handle

    def test_operation_on_closed_mem_mapped(self):
        self.single_pass_weld_file.seek(0)
        fh = WeldxFile(
            self.single_pass_weld_file,
            asdf_args=dict(copy_arrays=False, lazy_load=True),
        )
        fh.close()
        # FIXME: why is the tree still valid, after closing the file?
        assert fh["process"]

    def test_update_on_close(self):
        """A WeldxFile with mode="rw" should write changes on close."""
        buff = self.make_copy(self.fh)
        fh2 = WeldxFile(buff, mode="rw", sync=True)
        fh2["test"] = True
        fh2.close()
        buff.seek(0)
        fh3 = WeldxFile(buff, mode="r")
        assert fh3["test"]

    @pytest.mark.parametrize("sync", [True, False])
    def test_context_manageable(self, sync):
        """Check the file handle gets closed."""
        copy = self.fh.write_to()
        with WeldxFile(copy, mode="rw", asdf_args=dict(), sync=sync) as fh:
            assert "something" not in fh["wx_metadata"]
            fh["wx_metadata"]["something"] = True

        copy.seek(0)
        # check if changes have been written back according to sync flag.
        with WeldxFile(copy, mode="r") as fh2:
            if sync:
                assert fh2["wx_metadata"]["something"]
            else:
                assert "something" not in fh2["wx_metadata"]

    def test_history(self):
        """test custom software specs for history entries."""
        buff = BytesIO()
        software = dict(
            name="weldx_file_test", author="marscher", homepage="http://no", version="1"
        )
        self.fh = WeldxFile(
            buff,
            tree=self.single_pass_weld_tree,
            software_history_entry=software,
            mode="rw",
        )
        self.fh["wx_metadata"]["something"] = True
        desc = "added some metadata"
        self.fh.add_history_entry(desc)
        self.fh.sync()
        buff = self.make_copy(self.fh)

        new_fh = WeldxFile(buff)
        assert new_fh["wx_metadata"]["something"]
        assert new_fh.history[-1]["description"] == desc
        assert new_fh.history[-1]["software"] == software

        del new_fh["wx_metadata"]["something"]
        other_software = dict(
            name="software name", version="42", homepage="no", author="anon"
        )
        new_fh.add_history_entry("removed some metadata", software=other_software)
        buff2 = self.make_copy(new_fh)
        fh3 = WeldxFile(buff2)
