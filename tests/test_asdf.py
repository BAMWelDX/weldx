"""Tests basic asdf implementations."""

import os
from io import BytesIO

import asdf
import jsonschema
import numpy as np
import pandas as pd
import pytest

from weldx.asdf.extension import WeldxAsdfExtension, WeldxExtension
# weld design -----------------------------------------------------------------
from weldx.asdf.tags.weldx.aws.design.base_metal import BaseMetal
from weldx.asdf.tags.weldx.aws.design.connection import Connection
from weldx.asdf.tags.weldx.aws.design.joint_penetration import JointPenetration
from weldx.asdf.tags.weldx.aws.design.sub_assembly import SubAssembly
from weldx.asdf.tags.weldx.aws.design.weld_details import WeldDetails
from weldx.asdf.tags.weldx.aws.design.weldment import Weldment
from weldx.asdf.tags.weldx.aws.design.workpiece import Workpiece
# welding process -----------------------------------------------------------------
from weldx.asdf.tags.weldx.aws.process.arc_welding_process import ArcWeldingProcess
from weldx.asdf.tags.weldx.aws.process.gas_component import GasComponent
from weldx.asdf.tags.weldx.aws.process.shielding_gas_for_procedure import (
    ShieldingGasForProcedure,
)
from weldx.asdf.tags.weldx.aws.process.shielding_gas_type import ShieldingGasType
# iso groove -----------------------------------------------------------------
from weldx.asdf.tags.weldx.core.iso_groove import get_groove
# validators -----------------------------------------------------------------
from weldx.asdf.tags.weldx.debug.validator_testclass import ValidatorTestClass
from weldx.asdf.validators import _custom_shape_validator as val
from weldx.constants import WELDX_QUANTITY as Q_


def _write_read_buffer(tree):
    # Write the data to buffer
    with asdf.AsdfFile(
        tree,
        extensions=[WeldxExtension(), WeldxAsdfExtension()],
        ignore_version_mismatch=False,
    ) as ff:
        buff = BytesIO()
        ff.write_to(buff, all_array_storage="inline")
        buff.seek(0)

    # read back data from ASDF file contents in buffer
    with asdf.open(
        buff, copy_arrays=True, extensions=[WeldxExtension(), WeldxAsdfExtension()]
    ) as af:
        data = af.tree
    return data


def test_aws_example():
    """Test validity of current AWS Data Dictionary standard implementation."""
    # welding process -----------------------------------------------------------------
    gas_comp = [
        GasComponent("argon", Q_(82, "percent")),
        GasComponent("carbon dioxide", Q_(18, "percent")),
    ]
    gas_type = ShieldingGasType(gas_component=gas_comp, common_name="SG")

    gas_for_procedure = ShieldingGasForProcedure(
        use_torch_shielding_gas=True,
        torch_shielding_gas=gas_type,
        torch_shielding_gas_flowrate=Q_(20, "l / min"),
    )

    arc_welding_process = ArcWeldingProcess("GMAW")
    with pytest.raises(ValueError):  # test for non viable process string
        ArcWeldingProcess("NON_EXISTENT_PROCESS")

    process = {
        "arc_welding_process": arc_welding_process,
        "shielding_gas": gas_for_procedure,
    }

    # weld design -----------------------------------------------------------------
    v_groove = get_groove(
        groove_type="VGroove",
        workpiece_thickness=Q_(9, "mm"),
        groove_angle=Q_(50, "deg"),
        root_face=Q_(4, "mm"),
        root_gap=Q_(2, "mm"),
    )
    u_groove = get_groove(
        groove_type="UGroove",
        workpiece_thickness=Q_(15, "mm"),
        bevel_angle=Q_(9, "deg"),
        bevel_radius=Q_(6, "mm"),
        root_face=Q_(3, "mm"),
        root_gap=Q_(1, "mm"),
    )

    joint_penetration = JointPenetration(
        complete_or_partial="completePenetration", root_penetration=Q_(1.0, "mm")
    )
    weld_details = WeldDetails(
        joint_design=v_groove, weld_sizes=Q_(320, "mm"), number_of_passes=1
    )
    weld_details2 = WeldDetails(
        joint_design=u_groove, weld_sizes=Q_(320, "mm"), number_of_passes=1
    )
    connection1 = Connection(
        joint_type="butt_joint",
        weld_type="singleVGroove",
        joint_penetration=joint_penetration,
        weld_details=weld_details,
    )
    connection2 = Connection(
        joint_type="butt_joint",
        weld_type="singleUGroove",
        joint_penetration=joint_penetration,
        weld_details=weld_details2,
    )
    workpieces = [Workpiece(geometry="V-Groove")]
    sub_assembly = [
        SubAssembly(workpiece=workpieces, connection=connection1),
        SubAssembly(workpiece=workpieces, connection=connection2),
    ]

    weldment = Weldment(sub_assembly)

    base_metal = BaseMetal("steel", "plate", Q_(10.3, "mm"))

    tree = dict(process=process, weldment=weldment, base_metal=base_metal)

    # Write the data to buffer
    with asdf.AsdfFile(
        tree,
        extensions=[WeldxExtension(), WeldxAsdfExtension()],
        ignore_version_mismatch=False,
    ) as ff:
        buff = BytesIO()
        ff.write_to(buff, all_array_storage="inline")
        buff.seek(0)

    # read back data from ASDF file contents in buffer
    with asdf.open(
        buff, copy_arrays=True, extensions=[WeldxExtension(), WeldxAsdfExtension()]
    ) as af:
        data = af.tree
    assert isinstance(data, dict)


def test_jinja_template():
    """Test jinja template compilation with basic example."""
    from weldx.asdf.utils import make_asdf_schema_string, create_asdf_dataclass

    asdf_file_path, python_file_path = create_asdf_dataclass(
        asdf_name="custom/testclass",
        asdf_version="1.0.0",
        class_name="TestClass",
        properties=[
            "prop1",
            "prop2",
            "prop3",
            "prop4",
            "list_prop",
            "pint_prop",
            "groove_prop",
            "unkonwn_prop",
        ],
        required=["prop1", "prop2", "prop3"],
        property_order=["prop1", "prop2", "prop3"],
        property_types=[
            "str",
            "int",
            "float",
            "bool",
            "List[str]",
            "pint.Quantity",
            "VGroove",
            "unknown_type",
        ],
        description=[
            "a string",
            "",
            "a float",
            "a boolean value?",
            "a list",
            "a pint quantity",
            "a groove shape",
            "some not implemented property",
        ],
    )

    os.remove(asdf_file_path)
    os.remove(python_file_path)

    make_asdf_schema_string(
        asdf_name="custom/testclass", asdf_version="1.0.0", properties=["prop"]
    )


def test_time_classes():
    """Test basic implementation and serialization of pandas time datatypes."""
    # Timedelta -------------------------------------------------------
    td = pd.Timedelta("5m3ns")

    # Timedelta -------------------------------------------------------
    td_max = pd.Timedelta("106751 days 23:47:16.854775")

    # TimedeltaIndex -------------------------------------------------------
    tdi = pd.timedelta_range(start="-5s", end="25s", freq="3s")
    tdi_nofreq = pd.TimedeltaIndex([0, 1e9, 5e9, 3e9])

    # Timestamp -------------------------------------------------------
    ts = pd.Timestamp("2020-04-15T16:47:00.000000001")
    ts_tz = pd.Timestamp("2020-04-15T16:47:00.000000001", tz="Europe/Berlin")

    # DatetimeIndex -------------------------------------------------------
    dti = pd.date_range(start="2020-01-01", periods=5, freq="1D")
    dti_tz = pd.date_range(start="2020-01-01", periods=5, freq="1D", tz="Europe/Berlin")
    dti_infer = pd.DatetimeIndex(
        ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]
    )
    dti_nofreq = pd.DatetimeIndex(
        ["2020-01-01", "2020-01-02", "2020-01-04", "2020-01-05"]
    )

    tree = dict(
        td=td,
        td_max=td_max,
        tdi=tdi,
        tdi_nofreq=tdi_nofreq,
        ts=ts,
        ts_tz=ts_tz,
        dti=dti,
        dti_tz=dti_tz,
        dti_infer=dti_infer,
        dti_nofreq=dti_nofreq,
    )

    # Write the data to buffer
    with asdf.AsdfFile(
        tree,
        extensions=[WeldxExtension(), WeldxAsdfExtension()],
        ignore_version_mismatch=False,
    ) as ff:
        buff = BytesIO()
        ff.write_to(buff)
        buff.seek(0)

    # read back data from ASDF file contents in buffer
    with asdf.open(
        buff, copy_arrays=True, extensions=[WeldxExtension(), WeldxAsdfExtension()]
    ) as af:
        data = af.tree

    assert isinstance(data, dict)
    for k, v in tree.items():
        assert np.all(data[k] == v)

    with pytest.raises(jsonschema.exceptions.ValidationError):
        # cannot store large ints >52 bits inline in asdf
        with asdf.AsdfFile(
            tree,
            extensions=[WeldxExtension(), WeldxAsdfExtension()],
            ignore_version_mismatch=False,
        ) as ff:
            buff = BytesIO()
            ff.write_to(buff, all_array_storage="inline")
            buff.seek(0)


def test_validators():
    """Test custom ASDF validators."""
    test = ValidatorTestClass(
        length_prop=Q_(1, "inch"),
        velocity_prop=Q_(2, "km / s"),
        current_prop=Q_(np.eye(2, 2), "mA"),
        nested_prop=dict(q1=Q_(np.eye(3, 3), "m"), q2=Q_(2, "m^3")),
        simple_prop={"value": float(3), "unit": "m"},
    )

    tree = {"root_node": test}

    data = _write_read_buffer(tree)
    test_read = data["root_node"]
    assert isinstance(data, dict)
    assert test_read.length_prop == test.length_prop
    assert test_read.velocity_prop == test.velocity_prop
    assert np.all(test_read.current_prop == test.current_prop)
    assert np.all(test_read.nested_prop["q1"] == test.nested_prop["q1"])
    assert test_read.nested_prop["q2"] == test.nested_prop["q2"]
    assert test_read.simple_prop == test.simple_prop

    with pytest.raises(jsonschema.exceptions.ValidationError):
        test = ValidatorTestClass(
            length_prop=Q_(1, "s"),  # wrong unit
            velocity_prop=Q_(2, "km / s"),
            current_prop=Q_(np.eye(2, 2), "mA"),
            nested_prop=dict(q1=Q_(np.eye(3, 3), "m"), q2=Q_(2, "m^3")),
            simple_prop={"value": float(3), "unit": "m"},
        )
        tree = {"root_node": test}
        data = _write_read_buffer(tree)

    with pytest.raises(jsonschema.exceptions.ValidationError):
        test = ValidatorTestClass(
            length_prop=Q_(1, "s"),
            velocity_prop=Q_(2, "liter"),  # wrong unit
            current_prop=Q_(np.eye(2, 2), "mA"),
            nested_prop=dict(q1=Q_(np.eye(3, 3), "m"), q2=Q_(2, "m^3")),
            simple_prop={"value": float(3), "unit": "m"},
        )
        tree = {"root_node": test}
        data = _write_read_buffer(tree)

    with pytest.raises(jsonschema.exceptions.ValidationError):
        test = ValidatorTestClass(
            length_prop=Q_(1, "inch"),
            velocity_prop=Q_(2, "km / s"),
            current_prop=Q_(np.eye(2, 2), "V"),  # wrong unit
            nested_prop=dict(q1=Q_(np.eye(3, 3), "m"), q2=Q_(2, "m^3")),
            simple_prop={"value": float(3), "unit": "m"},
        )
        tree = {"root_node": test}
        data = _write_read_buffer(tree)

    with pytest.raises(jsonschema.exceptions.ValidationError):
        test = ValidatorTestClass(
            length_prop=Q_(1, "m"),
            velocity_prop=Q_(2, "km / s"),
            current_prop=Q_(np.eye(2, 2), "mA"),
            nested_prop=dict(q1=Q_(np.eye(3, 3), "m"), q2=Q_(2, "V")),  # wrong unit
            simple_prop={"value": float(3), "unit": "m"},
        )
        tree = {"root_node": test}
        data = _write_read_buffer(tree)

    with pytest.raises(jsonschema.exceptions.ValidationError):
        test = ValidatorTestClass(
            length_prop=Q_(1, "m"),
            velocity_prop=Q_(2, "km / s"),
            current_prop=Q_(np.eye(2, 2), "mA"),
            nested_prop=dict(q1=Q_(np.eye(3, 3), "m"), q2=Q_(2, "m^3")),
            simple_prop={"value": float(3), "unit": "s"},  # wrong unit
        )
        tree = {"root_node": test}
        data = _write_read_buffer(tree)

    with pytest.raises(jsonschema.exceptions.ValidationError):
        test = ValidatorTestClass(
            length_prop=Q_(1, "m"),
            velocity_prop=Q_(2, "km / s"),
            current_prop=Q_(np.eye(2, 4), "mA"),  # wrong shape
            nested_prop=dict(q1=Q_(np.eye(3, 3), "m"), q2=Q_(2, "m^3")),
            simple_prop={"value": float(3), "unit": "m"},
        )
        tree = {"root_node": test}
        data = _write_read_buffer(tree)

    with pytest.raises(jsonschema.exceptions.ValidationError):
        test = ValidatorTestClass(
            length_prop=Q_(1, "m"),
            velocity_prop=Q_(2, "km / s"),
            current_prop=Q_(np.eye(2, 2), "mA"),
            nested_prop=dict(q1=Q_(np.eye(5, 3), "m"), q2=Q_(2, "m^3")),  # wrong shape
            simple_prop={"value": float(3), "unit": "m"},
        )
        tree = {"root_node": test}
        data = _write_read_buffer(tree)


def test_shape_validator_syntax():
    """Test handling of custom shape validation syntax in Python."""

    # correct evaluation
    assert val([3], [3])
    assert val([2, 4, 5], [2, 4, 5])
    assert val([1, 2, 3], ["..."])
    assert val([1, 2], [1, 2, "..."])
    assert val([1, 2], ["...", 1, 2])
    assert val([1, 2, 3], [1, 2, None])
    assert val([1, 2, 3], [None, 2, 3])
    assert val([1], [1, "..."])
    assert val([1, 2, 3, 4, 5], [1, "..."])
    assert val([1, 2, 3, 4, 5], ["...", 4, 5])
    assert val([1, 2], [1, 2, "(3)"])
    assert val([2, 3], ["(1)", 2, 3])
    assert val([1, 2, 3], [1, "1~3", 3])
    assert val([1, 2, 3], [1, "1~", 3])
    assert val([1, 2, 3], [1, "~3", 3])

    # shape mismatch
    assert not val([2, 2, 3], [1, "..."])
    assert not val([2, 2, 3], ["...", 1])
    assert not val([1], [1, 2])
    assert not val([1, 2], [1])
    assert not val([1, 2], [3, 2])
    assert not val([1], [1, "~"])
    assert not val([1], ["~", 1])
    assert not val([1, 2, 3], [1, 2, "(4)"])
    assert not val([1, 2, 3], ["(2)", 2, 3])
    assert not val([1, 2], [1, "4~8"])

    # syntax errors, these should throw a ValueError
    with pytest.raises(ValueError):
        val([1, 2], [1, "~", "(...)"])  # value error?
    with pytest.raises(ValueError):
        val([1, 2], [1, "(2)", 3])
    with pytest.raises(ValueError):
        val([1, 2], [1, "...", 2])  # should this be allowed? syntax/value error?
    with pytest.raises(ValueError):
        val([1, 2], ["(1)", "..."])
    with pytest.raises(ValueError):
        val([1, 2], [1, "4~1"])
    with pytest.raises(ValueError):  # no negative shape numbers allowed in syntax
        val([-1, -2], [-1, -2])
