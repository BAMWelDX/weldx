"""Test all ASDF groove implementations."""

import matplotlib.pyplot as plt
import pytest
from decorator import contextmanager

from weldx.asdf.util import write_read_buffer
from weldx.constants import WELDX_QUANTITY as Q_
from weldx.geometry import Profile
from weldx.welding.groove.iso_9692_1 import (
    IsoBaseGroove,
    _create_test_grooves,
    get_groove,
)

test_params = _create_test_grooves()


@pytest.mark.parametrize(
    "groove, expected_dtype", test_params.values(), ids=test_params.keys()
)
def test_asdf_groove(groove: IsoBaseGroove, expected_dtype):
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

    data = write_read_buffer(tree)
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
    with pytest.raises(KeyError):
        get_groove(
            groove_type="WrongGrooveString",
            workpiece_thickness=Q_(9, "mm"),
            groove_angle=Q_(50, "deg"),
        )

    with pytest.raises(ValueError):
        get_groove(
            groove_type="FFGroove",
            workpiece_thickness=Q_(2, "mm"),
            workpiece_thickness2=Q_(5, "mm"),
            groove_angle=Q_(80, "deg"),
            root_gap=Q_(1, "mm"),
            code_number="6.1.1",
        ).to_profile()

    # negative parameter value
    with pytest.raises(ValueError):
        get_groove(
            groove_type="VGroove",
            workpiece_thickness=Q_(9, "mm"),
            groove_angle=Q_(50, "deg"),
            root_face=Q_(-4, "mm"),
            root_gap=Q_(2, "mm"),
        )


@pytest.mark.parametrize("groove", test_params.values(), ids=test_params.keys())
def test_cross_section(groove):  # noqa
    @contextmanager
    def temp_attr(obj, attr, new_value):
        old_value = getattr(obj, attr)
        setattr(obj, attr, new_value)
        yield
        setattr(obj, attr, old_value)

    groove_obj, groove_cls = groove
    # make rasterization for U-based grooves rather rough.
    with temp_attr(groove_obj, "_AREA_RASTER_WIDTH", 0.75):  # skipcq: PYL-E1129
        try:
            A = groove_obj.cross_sect_area
        except NotImplementedError:
            return
        except Exception as ex:
            raise ex

    # check docstring got inherited.
    assert groove_cls.cross_sect_area.__doc__ is not None

    assert hasattr(A, "units")
    assert A.units == Q_("mm²")
    assert A > 0


def test_igroove_area():  # noqa
    groove, _ = test_params["i_groove"]
    A = groove.cross_sect_area
    assert A == groove.t * groove.b
