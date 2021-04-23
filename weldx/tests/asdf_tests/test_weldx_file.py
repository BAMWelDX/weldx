import io
import pathlib
import tempfile
from io import BytesIO

import asdf
import pytest

from weldx import WeldxFile
from weldx.asdf.file import SupportsFileReadWrite
from weldx.asdf.util import get_schema_path


class ReadOnlyFile:
    def __init__(self, tmpdir):
        fn = tempfile.mktemp(suffix=".asdf", dir=tmpdir)
        with open(fn, "wb") as fh:
            asdf.AsdfFile(tree=dict(hi="there")).write_to(fh)
        self.file_read_only = open(fn, mode="rb")
        self.mode = "rb"

    def read(self, *args, **kwargs):
        return self.file_read_only.read(*args, **kwargs)

    @staticmethod
    def readable():
        return True


class WritableFile:
    """example of a class implementing SupportsFileReadWrite"""

    def __init__(self):
        self.to_wrap = BytesIO()

    def read(self, *args, **kwargs):
        return self.to_wrap.read(*args, **kwargs)

    def write(self, *args, **kwargs):
        return self.to_wrap.write(*args, **kwargs)

    def tell(self):
        return self.to_wrap.tell()

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


@pytest.fixture(scope="class")
def simple_asdf_file(request):
    f = asdf.AsdfFile(tree=dict(wx_metadata=dict(welder="anonymous")))
    buff = io.BytesIO()
    f.write_to(buff)
    request.cls.simple_asdf_file = buff


@pytest.mark.usefixtures("simple_asdf_file")
class TestWeldXFile:
    @pytest.fixture(autouse=True)
    def setUp(self, *args, **kwargs):
        copy_for_test = self.make_copy(self.simple_asdf_file)
        self.fh = WeldxFile(copy_for_test, *args, **kwargs)

    @pytest.mark.parametrize("mode", ["rb", "wb", "a"])
    def test_invalid_mode(self, mode):
        with pytest.raises(ValueError):
            WeldxFile(None, mode=mode)

    @pytest.mark.parametrize(
        "file",
        [
            b"no",
            ["no"],
            True
        ],
    )
    def test_invalid_file_like_types(self, file):
        with pytest.raises(ValueError) as e:
            WeldxFile(file)
        assert "path" in e.value.args[0]

    @pytest.mark.parametrize("dest_wrap", [str, pathlib.Path])
    def test_write_to_path_like(self, tmpdir, dest_wrap):
        """tests WeldxFile.write_to for str and pathlib.Path"""
        fn = tempfile.mktemp(suffix=".asdf", dir=tmpdir)
        wrapped = dest_wrap(fn)
        self.fh.write_to(wrapped)
        # compare
        with open(fn, "rb") as fh:
            self.fh.file_handle.seek(0)
            assert fh.read() == self.fh.file_handle.read()

    def test_write_to_buffer(self):
        """tests WeldxFile.write_to with implicit buffer creation."""
        buff = self.fh.write_to()
        buff2 = self.make_copy(self.fh)
        assert buff.getvalue() == buff2.getvalue()

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
        self.fh = WeldxFile(filename_or_file_like=fn, tree=tree, mode="rw")
        new_file = self.make_copy(self.fh)
        assert WeldxFile(new_file)["foo"] == "bar"

    def test_create_from_tree_given_output_fn_wrong_mode(self, tmpdir):
        fn = tempfile.mktemp(suffix=".asdf", dir=tmpdir)

        with pytest.raises(RuntimeError):
            WeldxFile(fn, tree=dict(foo="bar"), mode="r")

    def test_create_from_tree(self, tmpdir):
        """tests wrapper creation from a dictionary."""
        tree = dict(foo="bar")
        # TODO: actually this would be a case for pytests parameterization, but...
        # it doesn't support fixtures in parameterization yet.
        for fd in [BytesIO(), tempfile.mktemp(suffix=".asdf", dir=tmpdir)]:
            fh = WeldxFile(fd, tree=tree, mode="rw")
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

    @pytest.mark.skip("https://github.com/asdf-format/asdf/issues/975")
    def test_create_readonly_protocol(self, tmpdir):
        f = ReadOnlyFile(tmpdir)
        WeldxFile(f)

    def test_read_only_raise_on_write(self, tmpdir):
        f = ReadOnlyFile(tmpdir)
        with pytest.raises(ValueError):
            WeldxFile(f, mode="rw")

    def test_create_but_no_overwrite_existing(self, tmpdir):
        """never (accidentally) overwrite existing files!"""
        f = tempfile.mktemp(dir=tmpdir)
        with open(f, "w") as fh:
            fh.write("something")
        with pytest.raises(FileExistsError):
            WeldxFile(f, mode="rw")

    def test_update_existing_asdf_file(self, tmpdir):
        f = tempfile.mktemp(dir=tmpdir)
        self.fh.write_to(f)
        with WeldxFile(f, mode="rw") as fh:
            fh["wx_metadata"]["key"] = True

        with WeldxFile(f, mode="r") as fh:
            assert fh["wx_metadata"]["key"]

    @staticmethod
    def make_copy(fh):
        buff = BytesIO()
        if isinstance(fh, WeldxFile):
            fh.write_to(buff)
        elif isinstance(fh, BytesIO):
            fh.seek(0)
            buff.write(fh.read())
        buff.seek(0)
        return buff

    def test_operation_on_closed(self):
        self.fh.close()
        assert self.fh["wx_metadata"]

        # cannot access closed handles
        with pytest.raises(RuntimeError):
            self.fh.file_handle

    def test_operation_on_closed_mem_mapped(self, tmpdir):
        import numpy as np

        x = np.random.random(100)
        fh = WeldxFile(
            filename_or_file_like=pathlib.Path(tmpdir / "test_map"),
            tree=dict(x=x),
            asdffile_kwargs=dict(copy_arrays=False, lazy_load=True),
            mode="rw",
        )
        fh.close()
        # FIXME: why is the array still accessible, after closing the file?
        assert np.all(fh["x"] == x)

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
        with WeldxFile(copy, mode="rw", sync=sync) as fh:
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
        fh = WeldxFile(
            buff,
            tree=dict(wx_metadata={}),
            software_history_entry=software,
            mode="rw",
        )
        fh["wx_metadata"]["something"] = True
        desc = "added some metadata"
        fh.add_history_entry(desc)
        fh.sync()
        buff = self.make_copy(fh)

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
        assert "removed" in fh3.history[-1]["description"]
        assert len(fh3.history) == 2

    def test_custom_schema(self):
        from weldx.asdf.tags.weldx.debug.test_property_tag import PropertyTagTestClass

        schema = get_schema_path("test_property_tag-1.0.0")
        w = WeldxFile(
            tree={"root_node": PropertyTagTestClass()},
            asdffile_kwargs=dict(custom_schema=schema),
        )
        assert w.custom_schema == schema

    def test_jupyter_repr(self):
        pass
        # TODO: should be tested using weldxfile in notebooks, right?
