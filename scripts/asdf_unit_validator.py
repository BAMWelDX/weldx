from io import BytesIO

import asdf
import numpy as np

from weldx.asdf.extension import WeldxAsdfExtension, WeldxExtension
from weldx.asdf.tags.weldx.core.iso_groove import get_groove
from weldx.asdf.tags.weldx.debug.anyof_class import AnyOfClass
from weldx.asdf.tags.weldx.debug.validator_testclass import ValidatorTestClass
from weldx.constants import WELDX_QUANTITY as Q_

# ---- DEBUG
obj_a = AnyOfClass({"value": 1})
obj_b = AnyOfClass({"file": "filename.txt", "meta": "csv"})

filename = "anyof_class.asdf"
tree = {"obj_a": obj_a, "obj_b": obj_b}

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

# ------------ UNIT / SHAPE
Q1 = Q_(1, "inch")
Q2 = Q_(2, "km / s")
Q3 = Q_(np.eye(2, 2), "mA")
nested_prop = dict(q1=Q_(np.eye(3, 3), "m"), q2=Q_(2, "m^3"))

test = ValidatorTestClass(
    Q1, Q2, Q3, nested_prop, simple_prop={"value": float(3), "unit": "m"}
)

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


# -------- GROOVE
groove = get_groove(
    groove_type="VGroove",
    workpiece_thickness=Q_(9, "mm"),
    groove_angle=Q_(50, "deg"),
    root_face=Q_(4, "mm"),
    root_gap=Q_(2, "mm"),
)

dv_groove2 = get_groove(
    groove_type="DoubleVGroove",
    workpiece_thickness=Q_(19, "mm"),
    groove_angle=Q_(40, "deg"),
    groove_angle2=Q_(60, "deg"),
    root_gap=Q_(2, "mm"),
    root_face=Q_(5, "mm"),
)

# Write the data to buffer
with asdf.AsdfFile(
    {"groove": dv_groove2},
    extensions=[WeldxExtension(), WeldxAsdfExtension()],
    ignore_version_mismatch=False,
) as ff:
    buff = BytesIO()
    ff.write_to(buff)
    buff.seek(0)
