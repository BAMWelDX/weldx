"""Test script"""

import asdf

from weldx import Q_
from weldx.asdf.extension import WeldxExtension, WeldxAsdfExtension
from weldx.asdf.tags.weldx.core.groove import get_groove

v_groove = get_groove(groove_type="VGroove",
                      workpiece_thickness=Q_(9, "mm"),
                      groove_angle=Q_(50, "deg"),
                      root_face=Q_(4, "mm"),
                      root_gap=Q_(2, "mm"))
u_groove = get_groove(groove_type="UGroove",
                      workpiece_thickness=Q_(15, "mm"),
                      bevel_angle=Q_(9, "deg"),
                      bevel_radius=Q_(6, "mm"),
                      root_face=Q_(3, "mm"),
                      root_gap=Q_(1, "mm"))
i_groove = get_groove(groove_type="IGroove",
                      workpiece_thickness=Q_(4, 'mm'),
                      root_gap=Q_(4, 'mm'))
uv_groove = get_groove(groove_type="UVGroove",
                       workpiece_thickness=Q_(12, "mm"),
                       groove_angle=Q_(60, "deg"),
                       bevel_angle=Q_(11, "deg"),
                       bevel_radius=Q_(6, "mm"),
                       root_face=Q_(4, "mm"),
                       root_gap=Q_(2, "mm"))
vv_groove = get_groove(groove_type="VVGroove",
                       workpiece_thickness=Q_(12, "mm"),
                       groove_angle=Q_(70, "deg"),
                       bevel_angle=Q_(13, "deg"),
                       root_gap=Q_(3, "mm"),
                       root_face=Q_(1, "mm"))
hv_groove = get_groove(groove_type="HVGroove",
                       workpiece_thickness=Q_(9, "mm"),
                       bevel_angle=Q_(55, "deg"),
                       root_gap=Q_(2, "mm"),
                       root_face=Q_(1, "mm"))
hu_groove = get_groove(groove_type="HUGroove",
                       workpiece_thickness=Q_(18, "mm"),
                       bevel_angle=Q_(15, "deg"),
                       bevel_radius=Q_(8, "mm"),
                       root_gap=Q_(2, "mm"),
                       root_face=Q_(3, "mm"))
dv_groove = get_groove(groove_type="DoubleVGroove",
                       workpiece_thickness=Q_(15, "mm"),
                       groove_angle=Q_(40, "deg"),
                       groove_angle2=Q_(60, "deg"),
                       root_gap=Q_(2, "mm"),
                       root_face=Q_(5, "mm"))
du_groove = get_groove(groove_type="DoubleUGroove",
                       workpiece_thickness=Q_(33, "mm"),
                       bevel_angle=Q_(8, "deg"),
                       bevel_angle2=Q_(12, "deg"),
                       root_face=Q_(3, "mm"),
                       root_face2=Q_(15, "mm"),
                       root_gap=Q_(2, "mm"))
dhv_groove = get_groove(groove_type="DoubleHVGroove",
                        workpiece_thickness=Q_(11, "mm"),
                        bevel_angle=Q_(35, "deg"),
                        bevel_angle2=Q_(60, "deg"),
                        root_face2=Q_(5, "mm"),
                        root_face=Q_(1, "mm"),
                        root_gap=Q_(3, "mm"))
dhu_groove = get_groove(groove_type="DoubleHUGroove",
                        workpiece_thickness=Q_(32, "mm"),
                        bevel_angle=Q_(10, "deg"),
                        bevel_angle2=Q_(20, "deg"),
                        bevel_radius=Q_(8, "mm"),
                        root_face2=Q_(15, "mm"),
                        root_face=Q_(2, "mm"),
                        root_gap=Q_(2, "mm"))
ff_groove = get_groove(groove_type="FrontalFaceGroove",
                       workpiece_thickness=Q_(2, "mm"),
                       workpiece_thickness2=Q_(5, "mm"),
                       groove_angle=Q_(80, "deg"),
                       root_gap=Q_(0, "mm"),
                       special_depth=Q_(4, "mm"))

target = asdf.AsdfFile(dict(
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
    test012=ff_groove,
),
    extensions=[WeldxAsdfExtension(), WeldxExtension()]
)
target.write_to("testfile.yml", all_array_storage="inline")
