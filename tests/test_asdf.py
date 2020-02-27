"""Tests basic asdf implementations."""

import os
from io import BytesIO
import asdf

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

    # weld design -----------------------------------------------------------------
    v_groove = get_groove(
        groove_type="VGroove",
        **dict(t=Q_(8, "mm"), alpha=Q_(60, "deg"), c=Q_(4, "mm"), b=Q_(2, "mm")),
    )
    u_groove = get_groove(
        groove_type="UGroove",
        **dict(
            t=Q_(15, "mm"),
            beta=Q_(9, "deg"),
            R=Q_(6, "mm"),
            c=Q_(3, "mm"),
            b=Q_(1, "mm"),
        ),
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

    filename = "aws_demo.yaml"
    tree = dict(process=gas_for_procedure, weldment=weldment)

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

    asdf_schema_string = make_asdf_schema_string(
        asdf_name="custom/testclass", asdf_version="1.0.0", properties=["prop"]
    )
