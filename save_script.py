"""Debug asdf save of groove implementation."""

import asdf

from weldx.constants import WELDX_QUANTITY as Q_
from weldx.asdf.extension import WeldxExtension, WeldxAsdfExtension
from weldx.asdf.tags.weldx.core.iso_groove import get_groove

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
i_groove = get_groove(
    groove_type="IGroove", workpiece_thickness=Q_(4, "mm"), root_gap=Q_(4, "mm")
)
uv_groove = get_groove(
    groove_type="UVGroove",
    workpiece_thickness=Q_(12, "mm"),
    groove_angle=Q_(60, "deg"),
    bevel_angle=Q_(11, "deg"),
    bevel_radius=Q_(6, "mm"),
    root_face=Q_(4, "mm"),
    root_gap=Q_(2, "mm"),
)
vv_groove = get_groove(
    groove_type="VVGroove",
    workpiece_thickness=Q_(12, "mm"),
    groove_angle=Q_(70, "deg"),
    bevel_angle=Q_(13, "deg"),
    root_gap=Q_(3, "mm"),
    root_face=Q_(1, "mm"),
    root_face2=Q_(5, "mm"),
)
hv_groove = get_groove(
    groove_type="HVGroove",
    workpiece_thickness=Q_(9, "mm"),
    bevel_angle=Q_(55, "deg"),
    root_gap=Q_(2, "mm"),
    root_face=Q_(1, "mm"),
)
hu_groove = get_groove(
    groove_type="HUGroove",
    workpiece_thickness=Q_(18, "mm"),
    bevel_angle=Q_(15, "deg"),
    bevel_radius=Q_(8, "mm"),
    root_gap=Q_(2, "mm"),
    root_face=Q_(3, "mm"),
)
dv_groove = get_groove(
    groove_type="DoubleVGroove",
    workpiece_thickness=Q_(19, "mm"),
    groove_angle=Q_(40, "deg"),
    groove_angle2=Q_(60, "deg"),
    root_gap=Q_(2, "mm"),
    root_face=Q_(5, "mm"),
    root_face2=Q_(7, "mm"),
    root_face3=Q_(7, "mm"),
)
du_groove = get_groove(
    groove_type="DoubleUGroove",
    workpiece_thickness=Q_(33, "mm"),
    bevel_angle=Q_(8, "deg"),
    bevel_angle2=Q_(12, "deg"),
    bevel_radius=Q_(6, "mm"),
    bevel_radius2=Q_(6, "mm"),
    root_face=Q_(3, "mm"),
    root_face2=Q_(15, "mm"),
    root_gap=Q_(2, "mm"),
)
dhv_groove = get_groove(
    groove_type="DoubleHVGroove",
    workpiece_thickness=Q_(11, "mm"),
    bevel_angle=Q_(35, "deg"),
    bevel_angle2=Q_(60, "deg"),
    root_face2=Q_(5, "mm"),
    root_face=Q_(1, "mm"),
    root_gap=Q_(3, "mm"),
)
dhu_groove = get_groove(
    groove_type="DoubleHUGroove",
    workpiece_thickness=Q_(32, "mm"),
    bevel_angle=Q_(10, "deg"),
    bevel_angle2=Q_(20, "deg"),
    bevel_radius=Q_(8, "mm"),
    bevel_radius2=Q_(8, "mm"),
    root_face2=Q_(15, "mm"),
    root_face=Q_(2, "mm"),
    root_gap=Q_(2, "mm"),
)
ff_groove0 = get_groove(
    groove_type="FrontalFaceGroove", workpiece_thickness=Q_(5, "mm"), code_number="1.12"
)
ff_groove1 = get_groove(
    groove_type="FrontalFaceGroove",
    workpiece_thickness=Q_(2, "mm"),
    workpiece_thickness2=Q_(5, "mm"),
    groove_angle=Q_(80, "deg"),
    root_gap=Q_(1, "mm"),
    code_number="3.1.1",
)
ff_groove2 = get_groove(
    groove_type="FrontalFaceGroove",
    workpiece_thickness=Q_(2, "mm"),
    workpiece_thickness2=Q_(5, "mm"),
    root_gap=Q_(1, "mm"),
    code_number="3.1.2",
)
ff_groove3 = get_groove(
    groove_type="FrontalFaceGroove",
    workpiece_thickness=Q_(2, "mm"),
    workpiece_thickness2=Q_(5, "mm"),
    groove_angle=Q_(80, "deg"),
    root_gap=Q_(1, "mm"),
    code_number="3.1.3",
)
ff_groove4 = get_groove(
    groove_type="FrontalFaceGroove",
    workpiece_thickness=Q_(2, "mm"),
    workpiece_thickness2=Q_(5, "mm"),
    groove_angle=Q_(80, "deg"),
    special_depth=Q_(4, "mm"),
    code_number="4.1.2",
)
ff_groove5 = get_groove(
    groove_type="FrontalFaceGroove",
    workpiece_thickness=Q_(2, "mm"),
    workpiece_thickness2=Q_(5, "mm"),
    root_gap=Q_(1, "mm"),
    code_number="4.1.3",
)

target = asdf.AsdfFile(
    dict(
        test001=v_groove,
        test002=u_groove,
        test003=i_groove,
        test004=uv_groove,
        test005=vv_groove,
        test006=hv_groove,
        test007=hu_groove,
        test008=dv_groove,
        test009=du_groove,
        test010=dhv_groove,
        test011=dhu_groove,
        test012=ff_groove0,
        test013=ff_groove1,
        test014=ff_groove2,
        test015=ff_groove3,
        test016=ff_groove4,
        test017=ff_groove5,
    ),
    extensions=[WeldxAsdfExtension(), WeldxExtension()],
)
target.write_to("testfile.yml", all_array_storage="inline")
