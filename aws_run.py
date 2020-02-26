"""Run the AWS Data Dictionary debug example."""

import asdf
import pprint

from weldx.constants import WELDX_QUANTITY as Q_

from weldx.asdf.extension import WeldxExtension, WeldxAsdfExtension

# welding process -----------------------------------------------------------------
from weldx.asdf.tags.weldx.aws.process.gas_component import GasComponent
from weldx.asdf.tags.weldx.aws.process.shielding_gas_type import ShieldingGasType
from weldx.asdf.tags.weldx.aws.process.shielding_gas_for_procedure import (
    ShieldingGasForProcedure,
)

# weld design -----------------------------------------------------------------
from weldx.asdf.tags.weldx.aws.design.joint_penetration import JointPenetration
from weldx.asdf.tags.weldx.aws.design.weld_details import WeldDetails
from weldx.asdf.tags.weldx.aws.design.connection import Connection
from weldx.asdf.tags.weldx.aws.design.workpiece import Workpiece
from weldx.asdf.tags.weldx.aws.design.sub_assembly import SubAssembly
from weldx.asdf.tags.weldx.aws.design.weldment import Weldment
from weldx.asdf.tags.weldx.core.groove import get_groove


# welding process -----------------------------------------------------------------
gas_comp = [GasComponent("Argon", 82.0), GasComponent("Carbon Dioxide", 18.0)]
gas_type = ShieldingGasType(gas_component=gas_comp, common_name="SG")

gas_for_procedure = ShieldingGasForProcedure(
    use_torch_shielding_gas=True,
    torch_shielding_gas=gas_type,
    torch_shielding_gas_flowrate=Q_(20, "l / min"),
)

# weld design -----------------------------------------------------------------
v_groove = get_groove(
    groove_type="VGroove",
    **dict(t=Q_(8, "mm"), alpha=Q_(60, "deg"), c=Q_(4, "mm"), b=Q_(2, "mm")),
)

joint_penetration = JointPenetration(
    complete_or_partial="complete", units="mm", root_penetration=1.0
)
weld_details = WeldDetails(
    joint_design=v_groove, weld_sizes=Q_(320, "mm"), number_of_passes=1
)
connection = Connection(
    joint_type="Butt-Joint",
    weld_type="full",
    joint_penetration=joint_penetration,
    weld_details=weld_details,
)
workpieces = [Workpiece(geometry="V-Groove")]
sub_assembly = [SubAssembly(workpiece=workpieces, connection=connection)]

weldment = Weldment(sub_assembly)

filename = "aws_demo.yaml"
tree = dict(process=gas_for_procedure, weldment=weldment)

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
    pprint.pprint(data["process"].__dict__)
    pprint.pprint(data["weldment"].__dict__)
