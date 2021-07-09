"""Tests for the WeldxFile class."""
import io
import pathlib
import shutil
import tempfile
from io import BytesIO

import asdf
import pytest
from jsonschema import ValidationError

from weldx import WeldxFile
from weldx.asdf.cli.welding_schema import single_pass_weld_example
from weldx.asdf.util import get_schema_path
from weldx.types import SupportsFileReadWrite


class ReadOnlyFile:
    """Simulate a read-only file."""

    def __init__(self, tmpdir):  # noqa: D107
        fn = tempfile.mktemp(suffix=".asdf", dir=tmpdir)
        with open(fn, "wb") as fh:
            asdf.AsdfFile(tree=dict(hi="there")).write_to(fh)
        self.mode = "rb"
        self.file_read_only = open(fn, mode=self.mode)

    def read(self, *args, **kwargs):  # noqa: D102
        return self.file_read_only.read(*args, **kwargs)

    def readline(self, limit=-1):
        return self.file_read_only.readline(limit)

    @staticmethod
    def readable():  # noqa: D102
        return True


class WritableFile:
    """Example of a class implementing SupportsFileReadWrite."""

    def __init__(self):  # noqa: D107
        self.to_wrap = BytesIO()

    def read(self, *args, **kwargs):  # noqa: D102
        return self.to_wrap.read(*args, **kwargs)

    def readline(self, *args, **kwargs):  # noqa: D102
        return self.to_wrap.readline(*args, **kwargs)

    def write(self, *args, **kwargs):  # noqa: D102
        return self.to_wrap.write(*args, **kwargs)

    def tell(self):  # noqa: D102
        return self.to_wrap.tell()

    def seek(self, *args, **kwargs):  # noqa: D102
        return self.to_wrap.seek(*args, **kwargs)

    def flush(self):
        """Simulate flush by rewinding to the beginning of the buffer."""
        self.seek(0)


def test_protocol_check(tmpdir):
    """Instance checks."""
    assert isinstance(WritableFile(), SupportsFileReadWrite)
    assert isinstance(BytesIO(), SupportsFileReadWrite)

    # real file:
    f = tempfile.mktemp(dir=tmpdir)
    with open(f, "w") as fh:
        assert isinstance(fh, SupportsFileReadWrite)


@pytest.fixture(scope="class")
def simple_asdf_file(request):
    """Create an ASDF file with a very simple tree and attaches it to cls."""
    f = asdf.AsdfFile(tree=dict(wx_metadata=dict(welder="anonymous")))
    buff = io.BytesIO()
    f.write_to(buff)
    request.cls.simple_asdf_file = buff


@pytest.mark.usefixtures("simple_asdf_file")
class TestWeldXFile:
    """Docstring."""

    @pytest.fixture(autouse=True)
    def setUp(self, *args, **kwargs):
        """Being called for every test. Creates a fresh copy of `simple_asdf_file`."""
        copy_for_test = self.make_copy(self.simple_asdf_file)
        self.fh = WeldxFile(copy_for_test, *args, **kwargs)

    @staticmethod
    @pytest.mark.parametrize("mode", ["rb", "wb", "a"])
    def test_invalid_mode(mode):
        """Raises on invalid modes."""
        with pytest.raises(ValueError):
            WeldxFile(None, mode=mode)

    @staticmethod
    @pytest.mark.parametrize(
        "file",
        [b"no", ["no"], True],
    )
    def test_invalid_file_like_types(file):
        """Illegal file types should raise."""
        with pytest.raises(ValueError) as e:
            WeldxFile(file)
        assert "path" in e.value.args[0]

    @pytest.mark.parametrize("dest_wrap", [str, pathlib.Path])
    def test_write_to_path_like(self, tmpdir, dest_wrap):
        """Test WeldxFile.write_to for str and pathlib.Path."""
        fn = tempfile.mktemp(suffix=".asdf", dir=tmpdir)
        wrapped = dest_wrap(fn)
        self.fh.write_to(wrapped)
        # compare
        with open(fn, "rb") as fh:
            self.fh.file_handle.seek(0)
            assert fh.read() == self.fh.file_handle.read()

    def test_write_to_buffer(self):
        """Test write_to with implicit buffer creation."""
        buff = self.fh.write_to()
        buff2 = self.make_copy(self.fh)
        assert buff.getvalue() == buff2.getvalue()

    def test_create_from_tree_create_buff(self):
        """Test wrapper creation from a dictionary."""
        tree = dict(foo="bar")
        # creates a buffer
        self.fh = WeldxFile(filename_or_file_like=None, tree=tree)
        new_file = self.make_copy(self.fh)
        assert WeldxFile(new_file)["foo"] == "bar"

    def test_create_from_tree_given_output_fn(self, tmpdir):
        """Test wrapper creation from a dictionary."""
        tree = dict(foo="bar")
        # should write to file
        fn = tempfile.mktemp(suffix=".asdf", dir=tmpdir)
        self.fh = WeldxFile(filename_or_file_like=fn, tree=tree, mode="rw")
        new_file = self.make_copy(self.fh)
        assert WeldxFile(new_file)["foo"] == "bar"

    @staticmethod
    def test_create_from_tree_given_output_fn_wrong_mode(tmpdir):
        """Passing data to be written in read-only mode should raise."""
        fn = tempfile.mktemp(suffix=".asdf", dir=tmpdir)

        with pytest.raises(RuntimeError):
            WeldxFile(fn, tree=dict(foo="bar"), mode="r")

    def test_create_from_tree(self, tmpdir):
        """Test wrapper creation from a dictionary."""
        tree = dict(foo="bar")
        # actually this would be a case for pytests parameterization, but...
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
        """Interface test for writable files."""
        f = WritableFile()
        WeldxFile(f, tree=dict(test="yes"))  # this should write the tree to f.
        new_file = self.make_copy(f.to_wrap)
        assert WeldxFile(new_file)["test"] == "yes"

    @staticmethod
    def test_create_readonly_protocol(tmpdir):
        """A read-only file should be supported by ASDF."""
        f = ReadOnlyFile(tmpdir)
        WeldxFile(f)

    @staticmethod
    def test_read_only_raise_on_write(tmpdir):
        """Read-only files cannot be written to."""
        f = ReadOnlyFile(tmpdir)
        with pytest.raises(ValueError):
            WeldxFile(f, mode="rw")

    @staticmethod
    def test_create_but_no_overwrite_existing(tmpdir):
        """Never (accidentally) overwrite existing files."""
        f = tempfile.mktemp(dir=tmpdir)
        with open(f, "w") as fh:
            fh.write("something")
        with pytest.raises(FileExistsError):
            WeldxFile(f, mode="rw")

    def test_update_existing_asdf_file(self, tmpdir):
        """Check existing files are updated."""
        f = tempfile.mktemp(dir=tmpdir)
        self.fh.write_to(f)
        with WeldxFile(f, mode="rw") as fh:
            fh["wx_metadata"]["key"] = True

        with WeldxFile(f, mode="r") as fh:
            assert fh["wx_metadata"]["key"]

    @staticmethod
    def make_copy(fh):
        """Guess what, creates a copy of fh."""
        buff = BytesIO()
        if isinstance(fh, WeldxFile):
            fh.write_to(buff)
        elif isinstance(fh, BytesIO):
            fh.seek(0)
            buff.write(fh.read())
        buff.seek(0)
        return buff

    def test_operation_on_closed(self):
        """Accessing the file_handle after closing is illegal."""
        self.fh.close()
        assert self.fh["wx_metadata"]

        # cannot access closed handles
        with pytest.raises(RuntimeError):
            self.fh.file_handle

    def test_update_on_close(self):
        """A WeldxFile with mode="rw" should write changes on close."""
        buff = self.make_copy(self.fh)
        fh2 = WeldxFile(buff, mode="rw", sync=True)
        fh2["test"] = True
        fh2.close()
        buff.seek(0)
        fh3 = WeldxFile(buff, mode="r")
        assert fh3["test"]

    @staticmethod
    def test_underlying_filehandle_closed(tmpdir):
        """Ensure file handles are being closed."""
        fn = tempfile.mktemp(suffix=".asdf", dir=tmpdir)

        with WeldxFile(fn, mode="rw") as wfile:
            wfile["updated"] = True
            fh = wfile.file_handle
            # now the context ends, and the file is being saved to disk again.
        assert fh.closed

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
        """Test custom software specs for history entries."""
        software = dict(
            name="weldx_file_test", author="marscher", homepage="http://no", version="1"
        )
        fh = WeldxFile(
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

    @staticmethod
    @pytest.mark.parametrize("schema_arg", ["custom_schema", "asdffile_kwargs"])
    def test_custom_schema(schema_arg):
        """Check the property complex_schema is being set."""
        buff, _ = single_pass_weld_example(None)
        schema = get_schema_path("datamodels/single_pass_weld-1.0.0.schema.yaml")
        kwargs = {schema_arg: schema}
        if schema_arg == "asdffile_kwargs":
            kwargs = {"asdffile_kwargs": {"custom_schema": schema}}
        w = WeldxFile(buff, **kwargs)
        assert w.custom_schema == schema

    @staticmethod
    def test_custom_schema_resolve_path():
        """Schema paths should be resolved internally."""
        schema = "single_pass_weld-1.0.0.schema"
        with pytest.raises(ValidationError) as e:
            WeldxFile(custom_schema=schema)
        assert "required property" in e.value.message

    @staticmethod
    def test_custom_schema_not_existent():
        """Non existent schema should raise."""
        with pytest.raises(ValueError):
            WeldxFile(custom_schema="no")

    @staticmethod
    def test_custom_schema_real_file(tmpdir):
        """Passing real paths."""
        assert not pathlib.Path("single_pass_weld-1.0.0.schema").exists()
        shutil.copy(get_schema_path("single_pass_weld-1.0.0.schema"), ".")
        with pytest.raises(ValueError):
            WeldxFile(custom_schema="no")

    @staticmethod
    def test_show_header_file_pos_unchanged():
        """Check displaying the header."""
        file = WeldxFile(tree={"sensor": "HKS_sensor"}, mode="rw")
        old_pos = file.file_handle.tell()
        file.show_asdf_header()
        after_pos = file.file_handle.tell()
        assert old_pos == after_pos

    def test_invalid_software_entry(self):
        """Invalid software entries should raise."""
        with pytest.raises(ValueError):
            self.fh.software_history_entry = {"invalid": None}

        with pytest.raises(ValueError):
            self.fh.software_history_entry = {"name": None}
