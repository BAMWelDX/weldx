import pytest
from asdf.exceptions import ValidationError

from weldx import WeldxFile
from weldx.config import enable_quality_standard
from weldx.measurement import MeasurementEquipment


def test_installable_quality_standard():
    enable_quality_standard("quality_standard_demo")

    eq = MeasurementEquipment("Equipment")

    tree = dict(equip=eq)

    with pytest.raises(ValidationError):
        WeldxFile(tree=tree, mode="rw")

    eq.wx_metadata = {"serial_number": 1234}
    WeldxFile(tree=tree, mode="rw")
