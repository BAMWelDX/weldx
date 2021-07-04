"""Test ASDF serialization of AWS schema definitions."""
import pytest

# weld design -----------------------------------------------------------------
from weldx.asdf.tags.weldx.aws.design.base_metal import BaseMetal
from weldx.asdf.tags.weldx.aws.design.connection import Connection

# weld design -----------------------------------------------------------------
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
from weldx.asdf.util import write_read_buffer
from weldx.constants import WELDX_QUANTITY as Q_
from weldx.util import compare_nested

# iso groove -----------------------------------------------------------------
from weldx.welding.groove.iso_9692_1 import get_groove


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

    data = write_read_buffer(tree)
    data.pop("asdf_library", None)
    data.pop("history", None)
    assert compare_nested(data, tree)
