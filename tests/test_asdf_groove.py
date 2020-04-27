"""Test all asdf groove implementations."""

# import pytest

from io import BytesIO
import asdf

from weldx.geometry import Profile
from weldx.constants import WELDX_QUANTITY as Q_
from weldx.asdf.extension import WeldxExtension, WeldxAsdfExtension
from weldx.asdf.tags.weldx.core.groove import (
    get_groove,
    VGroove,
    UGroove,
    IGroove,
    UVGroove,
    VVGroove,
    HVGroove,
    HUGroove,
    DVGroove,
    DUGroove,
    DHVGroove,
    DHUGroove,
    FFGroove,
)


def test_asdf_groove():
    """"
    Test ASDF functionality for all groves.
    """
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
        groove_type="FrontalFaceGroove",
        workpiece_thickness=Q_(5, "mm"),
        code_number="1.12",
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

    tree = dict(
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
    )

    with asdf.AsdfFile(
        tree,
        extensions=[WeldxExtension(), WeldxAsdfExtension()],
        ignore_version_mismatch=False,
    ) as ff:
        buff = BytesIO()
        ff.write_to(buff, all_array_storage="inline")
        buff.seek(0)

    with asdf.open(
        buff, copy_arrays=True, extensions=[WeldxExtension(), WeldxAsdfExtension()]
    ) as af:
        data = af.tree

    ERRMSG01 = "This is not a "
    ERRMSG02 = "Wrong dict content in "

    assert isinstance(data["test001"], VGroove), ERRMSG01 + f"{VGroove}"
    assert v_groove.__dict__ == data["test001"].__dict__, ERRMSG02 + f"{VGroove}"
    assert isinstance(data["test002"], UGroove), ERRMSG01 + f"{UGroove}"
    assert u_groove.__dict__ == data["test002"].__dict__, ERRMSG02 + f"{UGroove}"
    assert isinstance(data["test003"], IGroove), ERRMSG01 + f"{IGroove}"
    assert i_groove.__dict__ == data["test003"].__dict__, ERRMSG02 + f"{IGroove}"
    assert isinstance(data["test004"], UVGroove), ERRMSG01 + f"{UVGroove}"
    assert uv_groove.__dict__ == data["test004"].__dict__, ERRMSG02 + f"{UVGroove}"
    assert isinstance(data["test005"], VVGroove), ERRMSG01 + f"{VVGroove}"
    assert vv_groove.__dict__ == data["test005"].__dict__, ERRMSG02 + f"{VVGroove}"
    assert isinstance(data["test006"], HVGroove), ERRMSG01 + f"{HVGroove}"
    assert hv_groove.__dict__ == data["test006"].__dict__, ERRMSG02 + f"{HVGroove}"
    assert isinstance(data["test007"], HUGroove), ERRMSG01 + f"{HUGroove}"
    assert hu_groove.__dict__ == data["test007"].__dict__, ERRMSG02 + f"{HUGroove}"
    assert isinstance(data["test008"], DVGroove), ERRMSG01 + f"{DVGroove}"
    assert dv_groove.__dict__ == data["test008"].__dict__, ERRMSG02 + f"{DVGroove}"
    assert isinstance(data["test009"], DUGroove), ERRMSG01 + f"{DUGroove}"
    assert du_groove.__dict__ == data["test009"].__dict__, ERRMSG02 + f"{DUGroove}"
    assert isinstance(data["test010"], DHVGroove), ERRMSG01 + f"{DHVGroove}"
    assert dhv_groove.__dict__ == data["test010"].__dict__, ERRMSG02 + f"{DHVGroove}"
    assert isinstance(data["test011"], DHUGroove), ERRMSG01 + f"{DHUGroove}"
    assert dhu_groove.__dict__ == data["test011"].__dict__, ERRMSG02 + f"{DHUGroove}"
    assert isinstance(data["test012"], FFGroove), ERRMSG01 + f"{FFGroove}"
    assert ff_groove0.__dict__ == data["test012"].__dict__, ERRMSG02 + f"{FFGroove}"
    assert isinstance(data["test013"], FFGroove), ERRMSG01 + f"{FFGroove}"
    assert ff_groove1.__dict__ == data["test013"].__dict__, ERRMSG02 + f"{FFGroove}"
    assert isinstance(data["test014"], FFGroove), ERRMSG01 + f"{FFGroove}"
    assert ff_groove2.__dict__ == data["test014"].__dict__, ERRMSG02 + f"{FFGroove}"
    assert isinstance(data["test015"], FFGroove), ERRMSG01 + f"{FFGroove}"
    assert ff_groove3.__dict__ == data["test015"].__dict__, ERRMSG02 + f"{FFGroove}"
    assert isinstance(data["test016"], FFGroove), ERRMSG01 + f"{FFGroove}"
    assert ff_groove4.__dict__ == data["test016"].__dict__, ERRMSG02 + f"{FFGroove}"
    assert isinstance(data["test017"], FFGroove), ERRMSG01 + f"{FFGroove}"
    assert ff_groove5.__dict__ == data["test017"].__dict__, ERRMSG02 + f"{FFGroove}"

    # test to_profile
    assert isinstance(data["test001"].to_profile(), Profile)
    assert isinstance(data["test002"].to_profile(), Profile)
    assert isinstance(data["test003"].to_profile(), Profile)
    assert isinstance(data["test004"].to_profile(), Profile)
    assert isinstance(data["test005"].to_profile(), Profile)
    assert isinstance(data["test006"].to_profile(), Profile)
    assert isinstance(data["test007"].to_profile(), Profile)
    assert isinstance(data["test008"].to_profile(), Profile)
    assert isinstance(data["test009"].to_profile(), Profile)
    assert isinstance(data["test010"].to_profile(), Profile)
    assert isinstance(data["test011"].to_profile(), Profile)
    assert isinstance(data["test012"].to_profile(), Profile)
    assert isinstance(data["test013"].to_profile(), Profile)
    assert isinstance(data["test014"].to_profile(), Profile)
    assert isinstance(data["test015"].to_profile(), Profile)
    assert isinstance(data["test016"].to_profile(), Profile)
    assert isinstance(data["test017"].to_profile(), Profile)
