import unittest
from io import BytesIO
from pathlib import Path

from kisa.weldx_file import WeldxFile


# TODO: parameterize with physical file and BytesIO
class TestWeldXFile(unittest.TestCase):
    def setUp(self) -> None:
        self.filename = Path("..", "scripts", "data", "asdf_ref_00000.asdf")
        self.software = dict(
            software="weldx_file_test", author="me", homepage="no", version="1"
        )
        self.fh = WeldxFile(self.filename)

    def test_from_buffer(self):
        with open(self.filename, "rb") as fh:
            buff = BytesIO(fh.read())
        WeldxFile(buff)

    def make_copy(self, fh):
        buff = BytesIO()
        fh.write_to(buff)
        buff.seek(0)
        return buff

    def test_raise_operation_on_closed(self):
        self.fh.close()
        with self.assertRaises(IOError):
            _ = self.fh["process"]

    def test_update_on_close(self):
        """ A Weldxfile with mode="rw" should write changes on close."""
        buff = self.make_copy(self.fh)
        fh2 = WeldxFile(buff, mode="rw")
        fh2["test"] = True
        fh2.close()
        buff.seek(0)
        fh3 = WeldxFile(buff, mode="r")
        assert fh3["test"]

    def test_context_manageable(self):
        """check the file handle gets closed."""
        copy = self.fh.write_to()
        with WeldxFile(copy, mode="rw", asdf_args=dict()) as fh:
            fh["wx_metadata"]["something"] = True
            # prior closing in the context, we take another copy
            copy2 = fh.write_to()

        fh2 = WeldxFile(copy2)
        assert fh2["wx_metadata"]["something"]

    def test_history(self):
        self.fh["wx_metadata"]["something"] = True
        desc = "added some metadata"
        with self.fh:
            self.fh.add_history_entry(desc)
        buff = self.make_copy(self.fh)

        new_fh = WeldxFile(buff)
        assert new_fh["wx_metadata"]["something"]
        assert new_fh.history[-1]["description"] == desc
        assert new_fh.history[-1]["software"] == self.software

        del new_fh["wx_metadata"]["something"]
        other_software = dict(
            software="name", version="42", homepage="no", author="anon"
        )
        new_fh.add_history_entry("removed some metadata", software=other_software)
        buff2 = self.make_copy(new_fh)
        fh3 = WeldxFile(buff2)
