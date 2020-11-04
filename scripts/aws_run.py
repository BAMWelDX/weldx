"""Run the AWS Data Dictionary debug example."""

import pprint

import asdf

from weldx.asdf.extension import WeldxAsdfExtension, WeldxExtension
from weldx.asdf.tags.weldx.aws.design.base_metal import BaseMetal
from weldx.asdf.tags.weldx.aws.design.connection import Connection

# weld design -----------------------------------------------------------------
from weldx.asdf.tags.weldx.aws.design.joint_penetration import JointPenetration
from weldx.asdf.tags.weldx.aws.design.sub_assembly import SubAssembly
from weldx.asdf.tags.weldx.aws.design.weld_details import WeldDetails
from weldx.asdf.tags.weldx.aws.design.weldment import Weldment
from weldx.asdf.tags.weldx.aws.design.workpiece import Workpiece
from weldx.asdf.tags.weldx.aws.process.arc_welding_process import ArcWeldingProcess

# welding process -----------------------------------------------------------------
from weldx.asdf.tags.weldx.aws.process.gas_component import GasComponent
from weldx.asdf.tags.weldx.aws.process.shielding_gas_for_procedure import (
    ShieldingGasForProcedure,
)
from weldx.asdf.tags.weldx.aws.process.shielding_gas_type import ShieldingGasType
from weldx.constants import WELDX_QUANTITY as Q_
from weldx.welding.groove.iso_9692_1 import get_groove

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

joint_penetration = JointPenetration(
    complete_or_partial="completePenetration", root_penetration=Q_(1.0, "mm")
)
weld_details = WeldDetails(
    joint_design=v_groove, weld_sizes=Q_(320, "mm"), number_of_passes=1
)
connection = Connection(
    joint_type="butt_joint",
    weld_type="singleVGroove",
    joint_penetration=joint_penetration,
    weld_details=weld_details,
)
workpieces = [Workpiece(geometry="V-Groove")]
sub_assembly = [SubAssembly(workpiece=workpieces, connection=connection)]

weldment = Weldment(sub_assembly)

base_metal = BaseMetal("steel", "plate", Q_(10.3, "mm"))

filename = "aws_demo.yaml"
tree = dict(process=process, weldment=weldment, base_metal=base_metal)

# Write the data to a new file
with asdf.AsdfFile(
    tree,
    extensions=[WeldxExtension(), WeldxAsdfExtension()],
    ignore_version_mismatch=False,
) as ff:
    ff.write_to(filename, all_array_storage="inline")

# read back data from ASDF file
with asdf.open(
    filename, copy_arrays=True, extensions=[WeldxExtension(), WeldxAsdfExtension()]
) as af:
    data = af.tree
    pprint.pprint(data["process"])
    pprint.pprint(data["weldment"].__dict__)
    pprint.pprint(data["base_metal"].__dict__)
