"""pytest configuration."""
import pytest

from weldx.asdf.cli.welding_schema import single_pass_weld_example


@pytest.fixture(scope="class")
def single_pass_weld_asdf(request):
    """Create an example of a single pass weld fulfilling the single_pass_weld schema.

    Notes
    -----
    attaches the tree and a io.BytesIO instance containing the binary output of
    the ASDF serialization to requests as following:

            request.cls.single_pass_weld_tree = tree
            request.cls.single_pass_weld_file = buff

    So it can be accessed in a unittest style class as a fixture like this:

    @pytest.mark.usefixtures("single_pass_weld_asdf")
    class MyCase(unittest.TestCase):
        def test_foo(self):
            tree = self.single_pass_weld_tree
            ...

    """
    buff, tree = single_pass_weld_example(out_file=None)
    request.cls.single_pass_weld_tree = tree
    request.cls.single_pass_weld_file = buff
