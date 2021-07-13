"""Test the `config` module."""
from pathlib import Path

import asdf
import pytest

from weldx import WeldxFile
from weldx.config import QualityStandard, add_quality_standard, enable_quality_standard
from weldx.measurement import MeasurementEquipment

current_dir = Path(__file__).parent.absolute().as_posix()
qs_dir = f"{current_dir}/data/quality_standard/resources/test_organization"


class TestQualityStandard:
    """Test the quality standard class."""

    @staticmethod
    @pytest.mark.parametrize(
        "resource_dir",
        [
            qs_dir,
            Path(qs_dir),
        ],
    )
    def test_init(resource_dir):
        """Test the class creation.

        Parameters
        ----------
        resource_dir:
            The resource directory of the quality standard

        """
        QualityStandard(resource_dir)


class TestConfig:
    """Test the weldx configuration object."""

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
    def test_enable_quality_standard(standard: str, expect_validation_error: bool):
        """Test enabling quality standards.

        The test uses the `MeasurementEquipment` class for test purposes. The standard
        created for the tests expects it to have a `wx_metadata` property with a
        `serial_number` field.

        Parameters
        ----------
        standard :
            Name of the standard that should be enabled. If `None` is provided, no
            standard is enabled.
        expect_validation_error :
            `True` if an unmodified instance of the `MeasurementEquipment` class should
            yield a `ValidationError` when validated against the specified standard.

        """
        if standard is not None:
            add_quality_standard(QualityStandard(qs_dir))
            enable_quality_standard(name=standard)

        ge = MeasurementEquipment(name="GE")
        if expect_validation_error:
            with pytest.raises(asdf.ValidationError):
                WeldxFile(tree={"equipment": ge}, mode="rw")
        else:
            WeldxFile(tree={"equipment": ge}, mode="rw")

        ge.wx_metadata = {"serial_number": 42}
        WeldxFile(tree={"equipment": ge}, mode="rw")
