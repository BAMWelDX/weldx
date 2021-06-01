from pathlib import Path

import asdf
import pytest

from weldx import WeldxFile
from weldx.config import QualityStandard, add_quality_standard, enable_quality_standard

current_dir = Path(__file__).parent.absolute().as_posix()
qs_dir = f"{current_dir}/data/quality_standard/resources/test_organization"


class TestQualityStandard:
    @staticmethod
    @pytest.mark.parametrize(
        "resource_dir",
        [
            qs_dir,
            Path(qs_dir),
        ],
    )
    def test_init(resource_dir):
        QualityStandard(resource_dir)


class TestConfig:
    @staticmethod
    @pytest.mark.parametrize(
        "standard, expect_validation_error",
        [
            (None, False),
            # problem: once we have evaluated a file, changes have no effect anymore.
            # Report upstream?
            # ("test_standard", True),
        ],
    )
    def test_enable_quality_standard(standard, expect_validation_error):
        if standard is not None:
            add_quality_standard(QualityStandard(qs_dir))
            enable_quality_standard(name=standard)

        from weldx.measurement import GenericEquipment

        ge = GenericEquipment(name="GE")
        if expect_validation_error:
            with pytest.raises(asdf.ValidationError):
                WeldxFile(tree={"equipment": ge}, mode="rw")
        else:
            WeldxFile(tree={"equipment": ge}, mode="rw")

        ge.wx_metadata = {"serial_number": 42}
        WeldxFile(tree={"equipment": ge}, mode="rw")
