"""Test all ASDF groove implementations."""

from io import BytesIO
import asdf
import pytest

from weldx.geometry import Profile
from weldx.constants import WELDX_QUANTITY as Q_
from weldx.asdf.extension import WeldxExtension, WeldxAsdfExtension
from weldx.asdf.tags.weldx.core.groove import (
    get_groove,
    BaseGroove,
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
    """Test ASDF functionality for all grooves."""
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
    dv_groove2 = get_groove(
        groove_type="DoubleVGroove",
        workpiece_thickness=Q_(19, "mm"),
        groove_angle=Q_(40, "deg"),
        groove_angle2=Q_(60, "deg"),
        root_gap=Q_(2, "mm"),
        root_face=Q_(5, "mm"),
    )
    dv_groove3 = get_groove(
        groove_type="DoubleVGroove",
        workpiece_thickness=Q_(19, "mm"),
        groove_angle=Q_(40, "deg"),
        groove_angle2=Q_(60, "deg"),
        root_gap=Q_(2, "mm"),
        root_face=Q_(5, "mm"),
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
        dv_groove2=dv_groove2,
        dv_groove3=dv_groove3,
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

    _key_to_type = dict(
        test001=VGroove,
        test002=UGroove,
        test003=IGroove,
        test004=UVGroove,
        test005=VVGroove,
        test006=HVGroove,
        test007=HUGroove,
        test008=DVGroove,
        dv_groove2=DVGroove,
        dv_groove3=DVGroove,
        test009=DUGroove,
        test010=DHVGroove,
        test011=DHUGroove,
        test012=FFGroove,
        test013=FFGroove,
        test014=FFGroove,
        test015=FFGroove,
        test016=FFGroove,
        test017=FFGroove,
    )

    for k, v in tree.items():
        # test class
        assert isinstance(
            data[k], _key_to_type[k]
        ), f"Item {k} did not match expected type {_key_to_type[k]}"
        # test content equality using dataclass built-in functions
        assert v == data[k], f"Could not correctly reconstruct groove of type {type(v)}"
        # test to_profile
        assert isinstance(
            v.to_profile(), Profile
        ), f"Error calling plot function of {type(v)} "

    # test parameter string generation
    assert set(v_groove.param_strings()) == {
        "alpha=50 deg",
        "b=2 mm",
        "c=4 mm",
        "t=9 mm",
    }

    # test exceptions
    with pytest.raises(ValueError):
        v_groove = get_groove(
            groove_type="WrongGrooveString",
            workpiece_thickness=Q_(9, "mm"),
            groove_angle=Q_(50, "deg"),
        )

    with pytest.raises(NotImplementedError):
        BaseGroove().to_profile()
