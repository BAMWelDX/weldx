# from weldx.asdf.utils import create_asdf_dataclass
#
# create_asdf_dataclass(
#     asdf_name="debug/testclass",
#     asdf_version="1.0.0",
#     class_name="TestClass",
#     properties=["prop1", "prop2", "prop3",],
#     required=["prop1", "prop2", "prop3"],
#     property_order=["prop1", "prop2", "prop3"],
#     property_types=["pint.Quantity", "pint.Quantity", "pint.Quantity",],
#     description=["a length", "a time", "a current"],
# )

import asdf

import weldx

from weldx.constants import WELDX_QUANTITY as Q_
from weldx.constants import WELDX_UNIT_REGISTRY as UREG
from weldx.asdf.extension import WeldxAsdfExtension, WeldxExtension

from weldx.asdf.tags.weldx.debug.unit_val_testclass import UnitValTestClass

Q1 = Q_(1, "inch")
Q2 = Q_(2, "km / s")
Q3 = Q_(3, "mA")
nested_prop = dict(q1=Q_(1, "m"), q2=Q_(2, "m^3"))

test = UnitValTestClass(Q1, Q2, Q3, nested_prop)

filename = "asdf_unit_validator.yaml"
tree = {"obj": test}

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
    # print(data["obj"])
