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

target = asdf.AsdfFile(dict(
    test001=v_groove,
    test002=u_groove,
    test003=i_groove,
    test004=uv_groove,
),
    extensions=[WeldxAsdfExtension(), WeldxExtension()]
)
target.write_to("testfile.yml", all_array_storage="inline")
