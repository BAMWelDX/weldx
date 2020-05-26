"""Test all ASDF groove implementations."""

from io import BytesIO
import asdf
import matplotlib.pyplot as plt
import pytest

from weldx.geometry import Profile
from weldx.constants import WELDX_QUANTITY as Q_
from weldx.asdf.extension import WeldxExtension, WeldxAsdfExtension
from weldx.asdf.tags.weldx.core.iso_groove import (
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


def _create_test_grooves():
    """Create dictionary with examples for all groove variations."""
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
    # DU grooves
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
    du_groove2 = get_groove(
        groove_type="DoubleUGroove",
        workpiece_thickness=Q_(33, "mm"),
        bevel_angle=Q_(8, "deg"),
        bevel_angle2=Q_(12, "deg"),
        bevel_radius=Q_(6, "mm"),
        bevel_radius2=Q_(6, "mm"),
        root_face=Q_(3, "mm"),
        root_gap=Q_(2, "mm"),
    )
    du_groove3 = get_groove(
        groove_type="DoubleUGroove",
        workpiece_thickness=Q_(33, "mm"),
        bevel_angle=Q_(8, "deg"),
        bevel_angle2=Q_(12, "deg"),
        bevel_radius=Q_(6, "mm"),
        bevel_radius2=Q_(6, "mm"),
        root_face=Q_(3, "mm"),
        root_face2=Q_(15, "mm"),
        root_face3=Q_(15, "mm"),
        root_gap=Q_(2, "mm"),
    )
    du_groove4 = get_groove(
        groove_type="DoubleUGroove",
        workpiece_thickness=Q_(33, "mm"),
        bevel_angle=Q_(8, "deg"),
        bevel_angle2=Q_(12, "deg"),
        bevel_radius=Q_(6, "mm"),
        bevel_radius2=Q_(6, "mm"),
        root_face=Q_(3, "mm"),
        root_face3=Q_(15, "mm"),
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
        workpiece_thickness=Q_(5, "mm"),
        workpiece_thickness2=Q_(7, "mm"),
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

    test_data = dict(
        v_groove=(v_groove, VGroove),
        u_groove=(u_groove, UGroove),
        i_groove=(i_groove, IGroove),
        uv_groove=(uv_groove, UVGroove),
        vv_groove=(vv_groove, VVGroove),
        hv_groove=(hv_groove, HVGroove),
        hu_groove=(hu_groove, HUGroove),
        dv_groove=(dv_groove, DVGroove),
        dv_groove2=(dv_groove2, DVGroove),
        dv_groove3=(dv_groove3, DVGroove),
        du_groove=(du_groove, DUGroove),
        du_groove2=(du_groove2, DUGroove),
        du_groove3=(du_groove3, DUGroove),
        du_groove4=(du_groove4, DUGroove),
        dhv_groove=(dhv_groove, DHVGroove),
        dhu_groove=(dhu_groove, DHUGroove),
        ff_groove0=(ff_groove0, FFGroove),
        ff_groove1=(ff_groove1, FFGroove),
        ff_groove2=(ff_groove2, FFGroove),
        ff_groove3=(ff_groove3, FFGroove),
        ff_groove4=(ff_groove4, FFGroove),
        ff_groove5=(ff_groove5, FFGroove),
    )

    return test_data


test_params = _create_test_grooves()


@pytest.mark.parametrize(
    "groove, expected_dtype", test_params.values(), ids=test_params.keys()
)
def test_asdf_groove(groove: BaseGroove, expected_dtype):
    """Test ASDF functionality for all grooves.

    Parameters
    ----------
    groove:
       Groove instance to be tested.

    expected_dtype:
       Expected type of the groove to be tested.

    """
    k = "groove"
    tree = {k: groove}

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
        assert isinstance(
            data[k], expected_dtype
        ), f"Did not match expected type {expected_dtype} on item {data[k]}"
        # test content equality using dataclass built-in functions
        assert (
            groove == data[k]
        ), f"Could not correctly reconstruct groove of type {type(groove)}"
        # test to_profile
        assert isinstance(
            groove.to_profile(), Profile
        ), f"Error calling plot function of {type(groove)} "

        # call plot function
        fig, ax = plt.subplots()
        groove.plot(ax=ax)
        plt.close(fig)


def test_asdf_groove_exceptions():
    """Test special cases and exceptions of groove classes."""
    # test parameter string generation
    v_groove = get_groove(
        groove_type="VGroove",
        workpiece_thickness=Q_(9, "mm"),
        groove_angle=Q_(50, "deg"),
        root_face=Q_(4, "mm"),
        root_gap=Q_(2, "mm"),
    )

    assert set(v_groove.param_strings()) == {
        "alpha=50 deg",
        "b=2 mm",
        "c=4 mm",
        "t=9 mm",
    }

    # test custom groove axis labels
    fig, _ = plt.subplots()
    v_groove.plot(axis_label=["x", "y"])
    plt.close(fig)

    # test exceptions
    with pytest.raises(ValueError):
        get_groove(
            groove_type="WrongGrooveString",
            workpiece_thickness=Q_(9, "mm"),
            groove_angle=Q_(50, "deg"),
        )

    with pytest.raises(NotImplementedError):
        BaseGroove().to_profile()

    with pytest.raises(ValueError):
        get_groove(
            groove_type="FrontalFaceGroove",
            workpiece_thickness=Q_(2, "mm"),
            workpiece_thickness2=Q_(5, "mm"),
            groove_angle=Q_(80, "deg"),
            root_gap=Q_(1, "mm"),
            code_number="6.1.1",
        ).to_profile()
