"""Tests the geometry package."""

import copy
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Union

import numpy as np
import pint
import pytest
from xarray import DataArray

import weldx.geometry as geo
import weldx.tests._helpers as helpers
import weldx.transformations as tf
import weldx.util as ut
from weldx import Q_
from weldx.geometry import SpatialData
from weldx.transformations import WXRotation

# helpers ---------------------------------------------------------------------


def check_segments_identical(seg_a, seg_b):
    """Check if 2 segments are identical within floating point tolerance.

    Parameters
    ----------
    seg_a :
        First segment
    seg_b :
        Second segment

    """
    assert isinstance(seg_a, type(seg_b))
    assert ut.matrix_is_close(seg_a.points, seg_b.points)
    if isinstance(seg_a, geo.ArcSegment):
        assert seg_a.arc_winding_ccw == seg_b.arc_winding_ccw
        assert ut.vector_is_close(seg_a.point_center, seg_b.point_center)


def check_shapes_identical(shp_a, shp_b):
    """Check if 2 shapes are identical within floating point tolerance.

    Parameters
    ----------
    shp_a :
        First profile
    shp_b :
        Second profile

    """
    assert shp_a.num_segments == shp_b.num_segments
    for i in range(shp_a.num_segments):
        check_segments_identical(shp_a.segments[i], shp_b.segments[i])


def check_profiles_identical(pro_a, pro_b):
    """Check if 2 profiles are identical within floating point tolerance.

    Parameters
    ----------
    pro_a :
        First profile
    pro_b :
        Second profile

    """
    assert pro_a.num_shapes == pro_b.num_shapes
    for i in range(pro_a.num_shapes):
        check_shapes_identical(pro_a.shapes[i], pro_b.shapes[i])


def check_variable_profiles_identical(vp_a, vp_b):
    """Check if 2 variable profiles are identical within floating point tolerance.

    Parameters
    ----------
    vp_a :
        First variable profile
    vp_b :
        Second variable profile

    """
    assert vp_a.num_profiles == vp_b.num_profiles
    assert vp_a.num_locations == vp_b.num_locations
    assert vp_a.num_interpolation_schemes == vp_b.num_interpolation_schemes

    for i in range(vp_a.num_profiles):
        check_profiles_identical(vp_a.profiles[i], vp_b.profiles[i])
    for i in range(vp_a.num_locations):
        assert math.isclose(vp_a.locations[i], vp_b.locations[i])
    for i in range(vp_a.num_interpolation_schemes):
        assert isinstance(
            vp_a.interpolation_schemes[i], type(vp_b.interpolation_schemes[i])
        )


def check_trace_segments_identical(seg_a, seg_b):
    """Check if 2 trace segments are identical within floating point tolerance.

    Parameters
    ----------
    seg_a :
        First segment
    seg_b :
        Second segment

    """
    assert isinstance(seg_a, type(seg_b))
    if isinstance(seg_a, geo.LinearHorizontalTraceSegment):
        assert seg_a.length == seg_b.length
    else:
        assert seg_a.is_clockwise == seg_b.is_clockwise
        assert math.isclose(seg_a.angle, seg_b.angle)
        assert math.isclose(seg_a.length, seg_b.length)
        assert math.isclose(seg_a.radius, seg_b.radius)


def check_traces_identical(trc_a, trc_b):
    """Check if 2 traces are identical within floating point tolerance.

    Parameters
    ----------
    trc_a :
        First trace
    trc_b :
        Second trace

    """
    assert trc_a.num_segments == trc_b.num_segments
    for i in range(trc_a.num_segments):
        check_trace_segments_identical(trc_a.segments[i], trc_b.segments[i])


def check_coordinate_systems_identical(lcs_a, lcs_b, abs_tol=1e-9):
    """Check if 2 local coordinate systems are identical within a tolerance.

    Parameters
    ----------
    lcs_a :
        First local coordinate system
    lcs_b :
        Second local coordinate system
    abs_tol :
        Absolute tolerance (Default value = 1e-9)

    """
    assert ut.matrix_is_close(lcs_a.orientation, lcs_b.orientation, abs_tol)
    assert ut.vector_is_close(lcs_a.coordinates, lcs_b.coordinates, abs_tol)


def get_default_profiles() -> List:
    """Get 2 profiles.

    Returns
    -------
    list
        List containing 2 profiles

    """
    a_0 = [0, 0]
    a_1 = [8, 16]
    a_2 = [16, 0]
    shape_a01 = geo.Shape().add_line_segments([a_0, a_1])
    shape_a12 = geo.Shape().add_line_segments([a_1, a_2])
    profile_a = geo.Profile([shape_a01, shape_a12])

    b_0 = [-4, 8]
    b_1 = [0, 8]
    b_2 = [16, -16]
    shape_b01 = geo.Shape().add_line_segments([b_0, b_1])
    shape_b12 = geo.Shape().add_line_segments([b_1, b_2])
    profile_b = geo.Profile([shape_b01, shape_b12])
    return [profile_a, profile_b]


# helper for segment tests ----------------------------------------------------


def default_segment_rasterization_tests(
    segment: Union[geo.ArcSegment, geo.LineSegment], raster_width
):
    """Perform some default checks for a passed segment's rasterization method.

    The segment is rasterized and tested afterwards. The purpose of every
    test is explained by a comment in the code.

    Parameters
    ----------
    segment :
        Instance of a segment class
    raster_width :
        Raster width

    """
    data = segment.rasterize(raster_width)

    # check dimensions are correct
    assert len(data.shape) == 2

    point_dimension = data.shape[0]
    num_points = data.shape[1]
    assert point_dimension == 2

    # Check if first and last point of the data are identical to the segment
    # start and end
    assert ut.vector_is_close(data[:, 0], segment.point_start)
    assert ut.vector_is_close(data[:, -1], segment.point_end)

    for i in range(num_points - 1):
        point = data[:, i]
        next_point = data[:, i + 1]

        raster_width_eff = np.linalg.norm(next_point - point)

        # effective raster width is close to specified one
        assert np.abs(raster_width_eff - raster_width) < 0.1 * raster_width

        # effective raster width is constant (equidistant points)
        assert math.isclose(raster_width_eff, np.linalg.norm(data[:, 1] - data[:, 0]))

    # check that there are no duplicate points
    assert helpers.are_all_columns_unique(data)

    # check that rasterization with too large raster width still works
    data_200 = segment.rasterize(200)

    num_points_200 = data_200.shape[1]
    assert num_points_200 == 2

    # only 2 points must be segment start and end
    assert ut.vector_is_close(segment.point_start, data_200[:, 0])
    assert ut.vector_is_close(segment.point_end, data_200[:, 1])

    # exceptions ------------------------------------------

    # raster width <= 0
    with pytest.raises(ValueError):
        segment.rasterize(0)
    with pytest.raises(ValueError):
        segment.rasterize(-3)


# test LineSegment ------------------------------------------------------------


def test_line_segment_construction():
    """Test constructor and factories."""
    # class constructor -----------------------------------
    segment = geo.LineSegment([[3, 5], [3, 4]])
    assert math.isclose(segment.length, np.sqrt(5))

    # exceptions ------------------------------------------
    # length = 0
    with pytest.raises(ValueError):
        geo.LineSegment([[0, 0], [1, 1]])
    # not 2x2
    with pytest.raises(ValueError):
        geo.LineSegment([[3, 5], [3, 4], [3, 2]])
    # not a 2d array
    with pytest.raises(ValueError):
        geo.LineSegment([[[3, 5], [3, 4]]])

    # factories -------------------------------------------
    segment = geo.LineSegment.construct_with_points([3, 3], [4, 5])
    assert math.isclose(segment.length, np.sqrt(5))


def test_line_segment_rasterization():
    """Test line segment rasterization.

    This test checks, if every rasterized point lies on the line that
    connects the start and the end of the segment. It also checks that those
    points lie between the segments start and end point.

    """
    raster_width = 0.1

    point_start = np.array([3, 3])
    point_end = np.array([4, 5])
    segment = geo.LineSegment.construct_with_points(point_start, point_end)

    # perform default tests
    default_segment_rasterization_tests(segment, raster_width)

    # rasterize data
    raster_data = segment.rasterize(raster_width)
    num_points = raster_data.shape[1]

    # check that points lie between start and end
    vec_start_end = point_end - point_start
    unit_vec_start_end = tf.normalize(vec_start_end)
    length_start_end = np.linalg.norm(vec_start_end)
    for i in np.arange(1, num_points - 1, 1):
        vec_start_point = raster_data[:, i] - point_start
        unit_vec_start_point = tf.normalize(vec_start_point)
        length_start_point = np.linalg.norm(vec_start_point)

        assert ut.vector_is_close(unit_vec_start_point, unit_vec_start_end)
        assert length_start_point < length_start_end


def line_segment_transformation_test_case(
    point_start,
    point_end,
    exp_start,
    exp_end,
    exp_length,
    translation=None,
    transformation=None,
):
    """Perform a single transformation test on a line segment.

    The test applies a transformation and compares the result to the
    expected values.

    Parameters
    ----------
    point_start :
        Start point of the line segment
    point_end :
        End point of the line segment
    exp_start :
        Expected start point of the transformed line segment
    exp_end :
        Expected end point of the transformed line segment
    exp_length :
        Expected length of the transformed line segment
    translation :
        Translation that should be applied (optional) (Default value = None)
    transformation :
        Transformation that should be applied (optional) (Default value = None)

    """
    if translation is not None:
        assert transformation is None, "No mixed test cases supported"

    segment = geo.LineSegment.construct_with_points(point_start, point_end)

    if translation is not None:
        segment_trans = segment.translate(translation)
    else:
        segment_trans = segment.transform(transformation)

    # original segment not modified
    assert ut.vector_is_close(segment.point_start, point_start)
    assert ut.vector_is_close(segment.point_end, point_end)

    # check new segment
    assert ut.vector_is_close(segment_trans.point_start, exp_start)
    assert ut.vector_is_close(segment_trans.point_end, exp_end)
    assert math.isclose(segment_trans.length, exp_length)

    # apply same transformation in place
    if translation is not None:
        segment.apply_translation(translation)
    else:
        segment.apply_transformation(transformation)
    check_segments_identical(segment, segment_trans)


def test_line_segment_transformations():
    """Test line segment transformations.

    This test tests all relevant transformations and exceptions.

    """
    # translation -----------------------------------------

    line_segment_transformation_test_case(
        point_start=[3, 3],
        point_end=[4, 5],
        translation=[-1, 4],
        exp_start=[2, 7],
        exp_end=[3, 9],
        exp_length=np.sqrt(5),
    )

    # 45 degree rotation ----------------------------------
    s = np.sin(np.pi / 4.0)
    c = np.cos(np.pi / 4.0)
    rotation_matrix = [[c, -s], [s, c]]

    line_segment_transformation_test_case(
        point_start=[2, 2],
        point_end=[3, 6],
        transformation=rotation_matrix,
        exp_start=[0, np.sqrt(8)],
        exp_end=np.matmul(rotation_matrix, [3, 6]),
        exp_length=np.sqrt(17),
    )

    # reflection at 45 degree line ------------------------
    v = np.array([-1, 1], dtype=float)
    reflection_matrix = np.identity(2) - 2 / np.dot(v, v) * np.outer(v, v)

    line_segment_transformation_test_case(
        point_start=[-1, 3],
        point_end=[6, 1],
        transformation=reflection_matrix,
        exp_start=[3, -1],
        exp_end=[1, 6],
        exp_length=np.sqrt(53),
    )

    # scaling ---------------------------------------------
    scale_matrix = [[4, 0], [0, 0.5]]

    line_segment_transformation_test_case(
        point_start=[-2, 2],
        point_end=[1, 4],
        transformation=scale_matrix,
        exp_start=[-8, 1],
        exp_end=[4, 2],
        exp_length=np.sqrt(145),
    )

    # exceptions ------------------------------------------

    # transformation results in length = 0
    zero_matrix = np.zeros((2, 2))
    segment = geo.LineSegment.construct_with_points([0, 0], [1, 2])
    with pytest.raises(Exception):
        segment.apply_transformation(zero_matrix)
    with pytest.raises(Exception):
        segment.transform(zero_matrix)


def test_line_segment_interpolation():
    """Test the line segments linear interpolation function.

    Two segments are created and interpolated using different weights. The
    result is compared to the expected values.

    """
    segment_a = geo.LineSegment.construct_with_points([1, 3], [7, -3])
    segment_b = geo.LineSegment.construct_with_points([5, -5], [-1, 13])

    for i in range(5):
        weight = i / 4
        segment_c = geo.LineSegment.linear_interpolation(segment_a, segment_b, weight)
        exp_point_start = [1 + i, 3 - 2 * i]
        exp_point_end = [7 - 2 * i, -3 + 4 * i]
        assert ut.vector_is_close(segment_c.point_start, exp_point_start)
        assert ut.vector_is_close(segment_c.point_end, exp_point_end)

    # check weight clipped to valid range -----------------

    segment_c = geo.LineSegment.linear_interpolation(segment_a, segment_b, -3)
    assert ut.vector_is_close(segment_c.point_start, segment_a.point_start)
    assert ut.vector_is_close(segment_c.point_end, segment_a.point_end)

    segment_c = geo.LineSegment.linear_interpolation(segment_a, segment_b, 6)
    assert ut.vector_is_close(segment_c.point_start, segment_b.point_start)
    assert ut.vector_is_close(segment_c.point_end, segment_b.point_end)

    # exceptions ------------------------------------------

    # wrong types
    arc_segment = geo.ArcSegment.construct_with_points([0, 0], [1, 1], [1, 0])
    with pytest.raises(TypeError):
        geo.LineSegment.linear_interpolation(segment_a, arc_segment, 0.5)
    with pytest.raises(TypeError):
        geo.LineSegment.linear_interpolation(arc_segment, segment_a, 0.5)
    with pytest.raises(TypeError):
        geo.LineSegment.linear_interpolation(arc_segment, arc_segment, 0.5)


# test ArcSegment ------------------------------------------------------------


def check_arc_segment_values(
    segment,
    point_start,
    point_end,
    point_center,
    winding_ccw,
    radius,
    arc_angle,
    arc_length,
):
    """Check if the internal values are identical with the expected values.

    Parameters
    ----------
    segment :
        Arc segment that should be checked
    point_start :
        Expected start point of the segment
    point_end :
        Expected end point of the segment
    point_center :
        Expected center point of the segment
    winding_ccw :
        Expected winding bool (see ArcSegment documentation)
    radius :
        Expected radius
    arc_angle :
        Expected arc angle
    arc_length :
        Expected arc length

    """
    assert ut.vector_is_close(segment.point_start, point_start)
    assert ut.vector_is_close(segment.point_end, point_end)
    assert ut.vector_is_close(segment.point_center, point_center)

    assert segment.arc_winding_ccw is winding_ccw
    assert math.isclose(segment.radius, radius)
    assert math.isclose(segment.arc_angle, arc_angle)
    assert math.isclose(segment.arc_length, arc_length)


def arc_segment_rasterization_test(
    point_center,
    point_start,
    point_end,
    raster_width,
    arc_winding_ccw,
    is_point_location_valid_func,
):
    """Test the arc segment's rasterize function.

    Performs the default segment rasterization test and some additional ones
    specific to the arc segment.

    Parameters
    ----------
    point_center :
        Center point of the segment
    point_start :
        Start point of the segment
    point_end :
        End point of the segment
    raster_width :
        Raster width
    arc_winding_ccw :
        Bool that determines the winding order
    is_point_location_valid_func :
        Function that returns a bool which
        specifies whether a point is valid or not. Interface: (point,
        point_center_arc) -> bool

    """
    point_center = np.array(point_center)
    point_start = np.array(point_start)
    point_end = np.array(point_end)

    radius_arc = np.linalg.norm(point_start - point_center)

    arc_segment = geo.ArcSegment.construct_with_points(
        point_start, point_end, point_center, arc_winding_ccw
    )

    # Perform standard segment rasterization tests
    default_segment_rasterization_tests(arc_segment, raster_width)

    # rasterize segment
    data = arc_segment.rasterize(raster_width)

    num_points = data.shape[1]
    for i in range(num_points):
        point = data[:, i]

        # Check that winding is correct
        assert is_point_location_valid_func(point, point_center)

        # Check that points have the correct distance to the arcs center
        distance_center_point = np.linalg.norm(point - point_center)
        assert math.isclose(distance_center_point, radius_arc, abs_tol=1e-6)


def test_arc_segment_constructor():
    """Test the arc segment constructor."""
    points = [[3, 6, 6], [3, 6, 3]]
    segment_cw = geo.ArcSegment(points, False)
    segment_ccw = geo.ArcSegment(points, True)

    check_arc_segment_values(
        segment=segment_cw,
        point_start=[3, 3],
        point_end=[6, 6],
        point_center=[6, 3],
        winding_ccw=False,
        radius=3,
        arc_angle=1 / 2 * np.pi,
        arc_length=3 / 2 * np.pi,
    )

    check_arc_segment_values(
        segment=segment_ccw,
        point_start=[3, 3],
        point_end=[6, 6],
        point_center=[6, 3],
        winding_ccw=True,
        radius=3,
        arc_angle=3 / 2 * np.pi,
        arc_length=9 / 2 * np.pi,
    )

    # check exceptions ------------------------------------

    # radius differs
    points = [[3, 6, 6], [3, 10, 3]]
    with pytest.raises(Exception):
        geo.ArcSegment(points, False)

    # radius is zero
    points = [[3, 3, 3], [3, 3, 3]]
    with pytest.raises(Exception):
        geo.ArcSegment(points, False)

    # arc length zero
    points = [[3, 3, 6], [3, 3, 3]]
    with pytest.raises(Exception):
        geo.ArcSegment(points, False)
    with pytest.raises(Exception):
        geo.ArcSegment(points, True)

    # not 2x3
    points = [[3, 3], [3, 3]]
    with pytest.raises(ValueError):
        geo.ArcSegment(points)

    # not a 2d array
    with pytest.raises(ValueError):
        geo.ArcSegment([[[3, 5], [3, 4]]])


def test_arc_segment_factories():
    """Test the arc segment's factory functions.

    Creates arc segments using the factory functions and checks if they are
    constructed as expected.

    """
    # construction with center point ----------------------
    point_start = [3, 3]
    point_end = [6, 6]
    point_center_left = [3, 6]
    point_center_right = [6, 3]

    # expected results
    radius = 3
    angle_small = np.pi * 0.5
    angle_large = np.pi * 1.5
    arc_length_small = np.pi * 1.5
    arc_length_large = np.pi * 4.5

    segment_cw = geo.ArcSegment.construct_with_points(
        point_start, point_end, point_center_right, False
    )
    segment_ccw = geo.ArcSegment.construct_with_points(
        point_start, point_end, point_center_right, True
    )

    check_arc_segment_values(
        segment_cw,
        point_start,
        point_end,
        point_center_right,
        False,
        radius,
        angle_small,
        arc_length_small,
    )
    check_arc_segment_values(
        segment_ccw,
        point_start,
        point_end,
        point_center_right,
        True,
        radius,
        angle_large,
        arc_length_large,
    )

    # construction with radius ----------------------

    # center left of line
    segment_cw = geo.ArcSegment.construct_with_radius(
        point_start, point_end, radius, True, False
    )
    segment_ccw = geo.ArcSegment.construct_with_radius(
        point_start, point_end, radius, True, True
    )

    check_arc_segment_values(
        segment_cw,
        point_start,
        point_end,
        point_center_left,
        False,
        radius,
        angle_large,
        arc_length_large,
    )
    check_arc_segment_values(
        segment_ccw,
        point_start,
        point_end,
        point_center_left,
        True,
        radius,
        angle_small,
        arc_length_small,
    )

    # center right of line
    segment_cw = geo.ArcSegment.construct_with_radius(
        point_start, point_end, radius, False, False
    )
    segment_ccw = geo.ArcSegment.construct_with_radius(
        point_start, point_end, radius, False, True
    )

    check_arc_segment_values(
        segment_cw,
        point_start,
        point_end,
        point_center_right,
        False,
        radius,
        angle_small,
        arc_length_small,
    )
    check_arc_segment_values(
        segment_ccw,
        point_start,
        point_end,
        point_center_right,
        True,
        radius,
        angle_large,
        arc_length_large,
    )

    # check that too small radii will be clipped to minimal radius
    segment_cw = geo.ArcSegment.construct_with_radius(
        point_start, point_end, 0.1, False, False
    )
    segment_ccw = geo.ArcSegment.construct_with_radius(
        point_start, point_end, 0.1, False, True
    )

    check_arc_segment_values(
        segment_cw,
        point_start,
        point_end,
        [4.5, 4.5],
        False,
        np.sqrt(18) / 2,
        np.pi,
        np.pi * np.sqrt(18) / 2,
    )
    check_arc_segment_values(
        segment_ccw,
        point_start,
        point_end,
        [4.5, 4.5],
        True,
        np.sqrt(18) / 2,
        np.pi,
        np.pi * np.sqrt(18) / 2,
    )


def point_in_second_quadrant(p, c):
    """Return True if a point is inside a circle's second quadrant.

    A point that lies directly on the boundary is considered as being inside.

    Parameters
    ----------
    p :
        Point that should be checked
    c :
        Center point of the circle

    Returns
    -------
    bool
        True or False

    """
    return p[0] - 1e-9 <= c[0] and p[1] >= c[1] - 1e-9


def point_not_in_second_quadrant(p, c):
    """Return True if a point is not inside a circle's second quadrant.

    A point that lies directly on the boundary is considered as being outside.

    Parameters
    ----------
    p :
        Point that should be checked
    c :
        Center point of the circle

    Returns
    -------
    bool
        True or False

    """
    return not (p[0] + 1e-9 < c[0] and p[1] > c[1] + 1e-9)


def point_not_below_center(p, c):
    """Return True if a point lies not below (y-value) a circle's center point.

    Parameters
    ----------
    p :
        Point that should be checked
    c :
        Center point of the circle

    Returns
    -------
    bool
        True or False

    """
    return p[1] >= c[1] - 1e-9


def point_not_above_center(p, c):
    """Return True if a point lies not above (y-value) a circle's center point.

    Parameters
    ----------
    p :
        Point that should be checked
    c :
        Center point of the circle

    Returns
    -------
    bool
        True or False

    """
    return p[1] - 1e-9 <= c[1]


def test_arc_segment_rasterization():
    """Test the arc segment's rasterize function.

    Creates some simple arc segments (semi-circle and quadrant) and test the
    rasterization results.

    """
    # center right of line point_start -> point_end
    # ---------------------------------------------

    point_center = [3, 2]
    point_start = [1, 2]
    point_end = [3, 4]
    raster_width = 0.2

    arc_segment_rasterization_test(
        point_center,
        point_start,
        point_end,
        raster_width,
        False,
        point_in_second_quadrant,
    )
    arc_segment_rasterization_test(
        point_center,
        point_start,
        point_end,
        raster_width,
        True,
        point_not_in_second_quadrant,
    )

    # center left of line point_start -> point_end
    # --------------------------------------------

    point_center = [-4, -7]
    point_start = [-4, -2]
    point_end = [-9, -7]
    raster_width = 0.1

    arc_segment_rasterization_test(
        point_center,
        point_start,
        point_end,
        raster_width,
        False,
        point_not_in_second_quadrant,
    )
    arc_segment_rasterization_test(
        point_center,
        point_start,
        point_end,
        raster_width,
        True,
        point_in_second_quadrant,
    )

    # center on line point_start -> point_end
    # ---------------------------------------

    point_center = [3, 2]
    point_start = [2, 2]
    point_end = [4, 2]
    raster_width = 0.1

    arc_segment_rasterization_test(
        point_center,
        point_start,
        point_end,
        raster_width,
        False,
        point_not_below_center,
    )
    arc_segment_rasterization_test(
        point_center, point_start, point_end, raster_width, True, point_not_above_center
    )


def arc_segment_transformation_test_case(
    point_start,
    point_end,
    point_center,
    exp_start,
    exp_end,
    exp_center,
    exp_is_winding_changed,
    exp_radius,
    exp_angle_ccw,
    translation=None,
    transformation=None,
):
    """Perform a single transformation test on an arc segment.

    The test applies a transformation and compares the result to the
    expected values.

    Parameters
    ----------
    point_start :
        Start point of the arc segment
    point_end :
        End point of the arc segment
    point_center :
        End point of the arc segment
    exp_start :
        Expected start point of the transformed arc segment
    exp_end :
        Expected end point of the transformed arc segment
    exp_center :
        Expected center point of the transformed arc segment
    exp_is_winding_changed :
        Bool that specifies if the transformation
        should change the winding order
    exp_radius :
        Expected radius of the transformed arc segment
    exp_angle_ccw :
        Expected angle of the transformed counter
        clockwise winding arc segment (refers to the winding before transformation)
    translation :
        Translation that should be applied (optional) (Default value = None)
    transformation :
        Transformation that should be applied (optional) (Default value = None)

    """
    if translation is not None:
        assert transformation is None, "No mixed test cases supported"

    segment_cw = geo.ArcSegment.construct_with_points(
        point_start, point_end, point_center, False
    )
    segment_ccw = geo.ArcSegment.construct_with_points(
        point_start, point_end, point_center, True
    )

    # store some values
    radius_original = segment_cw.radius
    arc_angle_cw_original = segment_cw.arc_angle
    arc_angle_ccw_original = segment_ccw.arc_angle
    arc_length_cw_original = segment_cw.arc_length
    arc_length_ccw_original = segment_ccw.arc_length

    if translation is not None:
        segment_cw_trans = segment_cw.translate(translation)
        segment_ccw_trans = segment_ccw.translate(translation)
    else:
        segment_cw_trans = segment_cw.transform(transformation)
        segment_ccw_trans = segment_ccw.transform(transformation)

    # original segments not modified
    check_arc_segment_values(
        segment_cw,
        point_start,
        point_end,
        point_center,
        False,
        radius_original,
        arc_angle_cw_original,
        arc_length_cw_original,
    )
    check_arc_segment_values(
        segment_ccw,
        point_start,
        point_end,
        point_center,
        True,
        radius_original,
        arc_angle_ccw_original,
        arc_length_ccw_original,
    )

    # check new segment
    exp_angle_cw = 2 * np.pi - exp_angle_ccw
    exp_arc_length_cw = exp_angle_cw * exp_radius
    exp_arc_length_ccw = exp_angle_ccw * exp_radius

    check_arc_segment_values(
        segment_cw_trans,
        exp_start,
        exp_end,
        exp_center,
        exp_is_winding_changed,
        exp_radius,
        exp_angle_cw,
        exp_arc_length_cw,
    )
    check_arc_segment_values(
        segment_ccw_trans,
        exp_start,
        exp_end,
        exp_center,
        not exp_is_winding_changed,
        exp_radius,
        exp_angle_ccw,
        exp_arc_length_ccw,
    )

    # apply same transformation in place
    if translation is not None:
        segment_cw.apply_translation(translation)
        segment_ccw.apply_translation(translation)
    else:
        segment_cw.apply_transformation(transformation)
        segment_ccw.apply_transformation(transformation)

    check_segments_identical(segment_cw, segment_cw_trans)
    check_segments_identical(segment_ccw, segment_ccw_trans)


def test_arc_segment_transformations():
    """Test the arc segments transformation functions."""
    # translation -----------------------------------------

    arc_segment_transformation_test_case(
        point_start=[3, 3],
        point_end=[5, 5],
        point_center=[5, 3],
        exp_start=[2, 7],
        exp_end=[4, 9],
        exp_center=[4, 7],
        exp_is_winding_changed=False,
        exp_radius=2,
        exp_angle_ccw=1.5 * np.pi,
        translation=[-1, 4],
    )

    # 45 degree rotation ----------------------------------
    s = np.sin(np.pi / 4.0)
    c = np.cos(np.pi / 4.0)
    rotation_matrix = [[c, -s], [s, c]]

    arc_segment_transformation_test_case(
        point_start=[3, 3],
        point_end=[5, 5],
        point_center=[5, 3],
        exp_start=[0, np.sqrt(18)],
        exp_end=[0, np.sqrt(50)],
        exp_center=np.matmul(rotation_matrix, [5, 3]),
        exp_is_winding_changed=False,
        exp_radius=2,
        exp_angle_ccw=1.5 * np.pi,
        transformation=rotation_matrix,
    )

    # reflection at 45 degree line ------------------------
    v = np.array([-1, 1], dtype=float)
    reflection_matrix = np.identity(2) - 2 / np.dot(v, v) * np.outer(v, v)

    arc_segment_transformation_test_case(
        point_start=[3, 2],
        point_end=[5, 4],
        point_center=[5, 2],
        exp_start=[2, 3],
        exp_end=[4, 5],
        exp_center=[2, 5],
        exp_is_winding_changed=True,
        exp_radius=2,
        exp_angle_ccw=1.5 * np.pi,
        transformation=reflection_matrix,
    )

    # scaling both coordinates equally --------------------
    scaling_matrix = [[4, 0], [0, 4]]

    arc_segment_transformation_test_case(
        point_start=[3, 2],
        point_end=[5, 4],
        point_center=[5, 2],
        exp_start=[12, 8],
        exp_end=[20, 16],
        exp_center=[20, 8],
        exp_is_winding_changed=False,
        exp_radius=8,
        exp_angle_ccw=1.5 * np.pi,
        transformation=scaling_matrix,
    )

    # non-uniform scaling which results in a valid arc ----
    scaling_matrix = [[0.25, 0], [0, 2]]

    exp_angle_ccw = 2 * np.pi - 2 * np.arcsin(3 / 5)

    arc_segment_transformation_test_case(
        point_start=[8, 4],
        point_end=[32, 4],
        point_center=[20, 2],
        exp_start=[2, 8],
        exp_end=[8, 8],
        exp_center=[5, 4],
        exp_is_winding_changed=False,
        exp_radius=5,
        exp_angle_ccw=exp_angle_ccw,
        transformation=scaling_matrix,
    )

    # exceptions ------------------------------------------

    # transformation distorts arc
    segment = geo.ArcSegment.construct_with_points([3, 2], [5, 4], [5, 2], False)
    with pytest.raises(Exception):
        segment.transform(scaling_matrix)
    with pytest.raises(Exception):
        segment.apply_transformation(scaling_matrix)

    # transformation results in length = 0
    segment = geo.ArcSegment.construct_with_points([3, 2], [5, 4], [5, 2], False)
    zero_matrix = np.zeros((2, 2))
    with pytest.raises(Exception):
        segment.transform(zero_matrix)
    with pytest.raises(Exception):
        segment.apply_transformation(zero_matrix)


def test_arc_segment_interpolation():
    """Test the arc segment interpolation.

    Since it is not implemented, check if an exception is raised.

    """
    segment_a = geo.ArcSegment.construct_with_points([0, 0], [1, 1], [1, 0])
    segment_b = geo.ArcSegment.construct_with_points([0, 0], [2, 2], [0, 2])

    # not implemented yet
    with pytest.raises(Exception):
        geo.ArcSegment.linear_interpolation(segment_a, segment_b, 1)


# test Shape ------------------------------------------------------------------


def test_shape_construction():
    """Test the constructor of the shape.

    Constructs some shapes in various ways and checks the results.

    """
    line_segment = geo.LineSegment.construct_with_points([1, 1], [1, 2])
    arc_segment = geo.ArcSegment.construct_with_points([0, 0], [1, 1], [0, 1])

    # Empty construction
    shape = geo.Shape()
    assert shape.num_segments == 0

    # Single element construction shape
    shape = geo.Shape(line_segment)
    assert shape.num_segments == 1

    # Multi segment construction
    shape = geo.Shape([arc_segment, line_segment])
    assert shape.num_segments == 2
    assert isinstance(shape.segments[0], geo.ArcSegment)
    assert isinstance(shape.segments[1], geo.LineSegment)

    # exceptions ------------------------------------------

    # segments not connected
    with pytest.raises(Exception):
        shape = geo.Shape([line_segment, arc_segment])


def test_shape_segment_addition():
    """Test the add_segments function of the shape.

    Test should be self explanatory.

    """
    # Create shape and add segments
    line_segment = geo.LineSegment.construct_with_points([1, 1], [0, 0])
    arc_segment = geo.ArcSegment.construct_with_points([0, 0], [1, 1], [0, 1])
    arc_segment2 = geo.ArcSegment.construct_with_points([1, 1], [0, 0], [0, 1])

    shape = geo.Shape()
    shape.add_segments(line_segment)
    assert shape.num_segments == 1

    shape.add_segments([arc_segment, arc_segment2])
    assert shape.num_segments == 3
    assert isinstance(shape.segments[0], geo.LineSegment)
    assert isinstance(shape.segments[1], geo.ArcSegment)
    assert isinstance(shape.segments[2], geo.ArcSegment)

    # exceptions ------------------------------------------

    # new segment are not connected to already included segments
    with pytest.raises(Exception):
        shape.add_segments(arc_segment2)
    assert shape.num_segments == 3  # ensure shape is unmodified

    with pytest.raises(Exception):
        shape.add_segments([arc_segment2, arc_segment])
    assert shape.num_segments == 3  # ensure shape is unmodified

    with pytest.raises(Exception):
        shape.add_segments([arc_segment, arc_segment])
    assert shape.num_segments == 3  # ensure shape is unmodified


def test_shape_line_segment_addition():
    """Test the shape's add_line_segments function.

    Test should be self explanatory.

    """
    shape_0 = geo.Shape()
    shape_0.add_line_segments([[0, 0], [1, 0]])
    assert shape_0.num_segments == 1

    shape_1 = geo.Shape()
    shape_1.add_line_segments([[0, 0], [1, 0], [2, 0]])
    assert shape_1.num_segments == 2

    # test possible formats to add single line segment ----

    shape_0.add_line_segments([2, 0])
    assert shape_0.num_segments == 2
    shape_0.add_line_segments([[3, 0]])
    assert shape_0.num_segments == 3
    shape_0.add_line_segments(np.array([4, 0]))
    assert shape_0.num_segments == 4
    shape_0.add_line_segments(np.array([[5, 0]]))
    assert shape_0.num_segments == 5

    # add multiple segments -------------------------------

    shape_0.add_line_segments([[6, 0], [7, 0], [8, 0]])
    assert shape_0.num_segments == 8
    shape_0.add_line_segments(np.array([[9, 0], [10, 0], [11, 0]]))
    assert shape_0.num_segments == 11

    for i in range(11):
        expected_segment = geo.LineSegment.construct_with_points([i, 0], [i + 1, 0])
        check_segments_identical(shape_0.segments[i], expected_segment)
        if i < 2:
            check_segments_identical(shape_1.segments[i], expected_segment)

    # exceptions ------------------------------------------

    shape_2 = geo.Shape()
    # invalid inputs
    with pytest.raises(Exception):
        shape_2.add_line_segments([])
    assert shape_2.num_segments == 0

    with pytest.raises(Exception):
        shape_2.add_line_segments(None)
    assert shape_2.num_segments == 0

    # single point with empty shape
    with pytest.raises(Exception):
        shape_2.add_line_segments([0, 1])
    assert shape_2.num_segments == 0

    # invalid point format
    with pytest.raises(Exception):
        shape_2.add_line_segments([[0, 1, 2], [1, 2, 3]])
    assert shape_2.num_segments == 0


def test_shape_rasterization():
    """Test rasterization function of the shape.

    The test uses three line segment of equal length, making it easy to
    check the rasterized points. Every step of the test is documented
    with comments.

    """
    points = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])

    shape = geo.Shape().add_line_segments(points)

    # rasterize shape
    raster_width = 0.2
    data = shape.rasterize(raster_width)

    # no duplications
    assert helpers.are_all_columns_unique(data)

    # check each data point
    num_data_points = data.shape[1]
    for i in range(num_data_points):
        if i < 6:
            assert ut.vector_is_close([0, i * 0.2], data[:, i])
        elif i < 11:
            assert ut.vector_is_close([(i - 5) * 0.2, 1], data[:, i])
        else:
            assert ut.vector_is_close([1, 1 - (i - 10) * 0.2], data[:, i])

    # Test with too large raster width --------------------
    # The shape does not clip large values to the valid range itself. The
    # added segments do the clipping. If a custom segment does not do that,
    # there is currently no mechanism to correct it.
    # However, this test somewhat ensures, that each segment is rasterized
    # individually.

    data = shape.rasterize(10)

    for point in points:
        assert ut.is_column_in_matrix(point, data)

    assert data.shape[1] == 4

    # no duplication if shape is closed -------------------

    shape.add_line_segments(points[0])

    data = shape.rasterize(10)

    assert data.shape[1] == 4
    assert helpers.are_all_columns_unique(data)

    # exceptions ------------------------------------------
    with pytest.raises(Exception):
        shape.rasterize(0)
    with pytest.raises(Exception):
        shape.rasterize(-3)
    # empty shape
    shape_empty = geo.Shape()
    with pytest.raises(Exception):
        shape_empty.rasterize(0.2)


def default_test_shape():
    """Get a default shape for tests.

    Returns
    -------
    weldx.geometry.Shape
        Default shape for tests

    """
    # create shape
    arc_segment = geo.ArcSegment.construct_with_points([3, 4], [5, 0], [6, 3])
    line_segment = geo.LineSegment.construct_with_points([5, 0], [11, 3])
    return geo.Shape([arc_segment, line_segment])


def default_translation_vector():
    """Get a default translation for transformation tests.

    Returns
    -------
    numpy.ndarray
        Translation vector

    """
    return ut.to_float_array([3, 4])


def check_point_translation(point_trans, point_original):
    """Check if a point is translated by the default translation test vector.

    Parameters
    ----------
    point_trans :
        Translated point
    point_original :
        Original point

    """
    assert ut.vector_is_close(
        point_trans - default_translation_vector(), point_original
    )


def check_point_rotation_90_degree(point_trans, point_original):
    """Check if a point is rotated by 90 degrees.

    Parameters
    ----------
    point_trans :
        Transformed point
    point_original :
        Original point

    """
    assert point_trans[0] == point_original[1]
    assert point_trans[1] == -point_original[0]


def check_point_reflection_at_line_with_slope_1(point_trans, point_original):
    """Check if a point is reflected at a line through the origin with slope 1.

    Parameters
    ----------
    point_trans :
        Transformed point
    point_original :
        Original point

    """
    assert point_trans[0] == point_original[1]
    assert point_trans[1] == point_original[0]


def shape_transformation_test_case(
    check_point_func, exp_winding_change, translation=None, transformation=None
):
    """Test a shape transformation.

    Parameters
    ----------
    check_point_func :
        Function that checks if a point is transformed
        correctly. Interface: (point_transformed, point_original) -> None
    exp_winding_change :
        Bool that specifies if the transformation
        should change the winding order of arc segments.
    translation :
        Translation vector (optional) (Default value = None)
    transformation :
        Transformation matrix (optional) (Default value = None)

    """
    if translation is not None:
        assert transformation is None, "No mixed test cases supported"

    shape = default_test_shape()

    if translation is not None:
        shape_trans = shape.translate(translation)
    else:
        shape_trans = shape.transform(transformation)

    # original shape unchanged
    check_shapes_identical(shape, default_test_shape())

    # extract segments
    arc_segment = shape.segments[0]
    line_segment = shape.segments[1]
    arc_segment_trans = shape_trans.segments[0]
    line_segment_trans = shape_trans.segments[1]

    # check transformed arc segment's winding order
    assert arc_segment_trans.arc_winding_ccw is not exp_winding_change

    # check segment points
    check_point_func(arc_segment_trans.point_start, arc_segment.point_start)
    check_point_func(arc_segment_trans.point_end, arc_segment.point_end)
    check_point_func(arc_segment_trans.point_center, arc_segment.point_center)

    check_point_func(line_segment_trans.point_start, line_segment.point_start)
    check_point_func(line_segment_trans.point_end, line_segment.point_end)

    # apply same transformation in place
    if translation is not None:
        shape.apply_translation(translation)
    else:
        shape.apply_transformation(transformation)

    check_shapes_identical(shape_trans, shape)


def test_shape_transformation():
    """Test the shapes transformation functions.

    Dedicated reflection functions are tested separately.

    """
    # translation -----------------------------------------
    shape_transformation_test_case(
        check_point_func=check_point_translation,
        exp_winding_change=False,
        translation=default_translation_vector(),
    )

    # transformation without reflection -------------------
    rotation_matrix = np.array([[0, 1], [-1, 0]])

    shape_transformation_test_case(
        check_point_func=check_point_rotation_90_degree,
        exp_winding_change=False,
        transformation=rotation_matrix,
    )

    # transformation with reflection ----------------------
    reflection_matrix = np.array([[0, 1], [1, 0]])

    shape_transformation_test_case(
        check_point_func=check_point_reflection_at_line_with_slope_1,
        exp_winding_change=True,
        transformation=reflection_matrix,
    )


def check_reflected_point(
    point_original, point_reflected, reflection_axis_offset, reflection_axis_direction
):
    """Check if a point is reflected correctly.

    The function determines if the midpoint of the line
    point->reflected_point lies on the reflection axis. The reflection axis
    is specified by a normal and an offset.

    Parameters
    ----------
    point_original :
        Original point
    point_reflected :
        Reflected point
    reflection_axis_offset :
        Offset vector of the reflection axis
        towards the origin.
    reflection_axis_direction :
        Direction vector of the reflection axis.

    """
    vec_original_reflected = point_reflected - point_original
    midpoint = point_original + 0.5 * vec_original_reflected
    shifted_mid_point = midpoint - reflection_axis_offset

    determinant = np.linalg.det([shifted_mid_point, reflection_axis_direction])
    assert math.isclose(determinant, 0, abs_tol=1e-9)


def shape_reflection_test_case(normal, distance_to_origin):
    """Test the shape's reflection functions.

    Only the functions that use a normal and a distance to the origin to
    specify the reflection axis are tested by this test.

    Parameters
    ----------
    normal :
        Normal of the reflection axis
    distance_to_origin :
        Distance to the origin of the reflection axis.

    """
    direction_reflection_axis = np.array([normal[1], -normal[0]])
    normal_length = np.linalg.norm(normal)
    unit_normal = np.array(normal) / normal_length
    offset = distance_to_origin * unit_normal

    shape = default_test_shape()

    # create reflected shape
    shape_reflected = shape.reflect(normal, distance_to_origin)

    # original shape is not modified
    check_shapes_identical(shape, default_test_shape())

    arc_segment = shape.segments[0]
    arc_segment_ref = shape_reflected.segments[0]
    line_segment = shape.segments[1]
    line_segment_ref = shape_reflected.segments[1]

    # check reflected points
    check_reflected_point(
        arc_segment.point_start,
        arc_segment_ref.point_start,
        offset,
        direction_reflection_axis,
    )
    check_reflected_point(
        arc_segment.point_end,
        arc_segment_ref.point_end,
        offset,
        direction_reflection_axis,
    )
    check_reflected_point(
        arc_segment.point_center,
        arc_segment_ref.point_center,
        offset,
        direction_reflection_axis,
    )

    check_reflected_point(
        line_segment.point_start,
        line_segment_ref.point_start,
        offset,
        direction_reflection_axis,
    )
    check_reflected_point(
        line_segment.point_end,
        line_segment_ref.point_end,
        offset,
        direction_reflection_axis,
    )

    # apply same reflection in place
    shape.apply_reflection(normal, distance_to_origin)
    check_shapes_identical(shape, shape_reflected)


def test_shape_reflection():
    """Test multiple reflections."""
    shape_reflection_test_case([2, 1], np.linalg.norm([2, 1]))
    shape_reflection_test_case([0, 1], 5)
    shape_reflection_test_case([1, 0], 3)
    shape_reflection_test_case([1, 0], -3)
    shape_reflection_test_case([-7, 2], 4.12)
    shape_reflection_test_case([-7, -2], 4.12)
    shape_reflection_test_case([7, -2], 4.12)

    # exceptions ------------------------------------------
    shape = default_test_shape()

    with pytest.raises(Exception):
        shape.reflect([0, 0], 2)
    with pytest.raises(Exception):
        shape.apply_reflection([0, 0])


def check_point_reflected_across_line(
    point_original, point_reflected, point_start, point_end
):
    """Check if a point is reflected correctly.

    The function determines if the midpoint of the line
    point->reflected_point lies on the reflection axis. The reflection axis
    is specified by 2 points.

    Parameters
    ----------
    point_original :
        Original point
    point_reflected :
        Reflected point
    point_start :
        First point of the reflection axis
    point_end :
        Second point of the reflection axis

    """
    vec_original_reflected = point_reflected - point_original
    mid_point = point_original + 0.5 * vec_original_reflected

    vec_start_mid = mid_point - point_start
    vec_start_end = point_end - point_start

    determinant = np.linalg.det([vec_start_end, vec_start_mid])
    assert math.isclose(determinant, 0, abs_tol=1e-9)


def shape_reflection_across_line_test_case(point_start, point_end):
    """Test the shape's reflection functions.

    Only the functions that use 2 points to specify the reflection axis are
    tested by this test.

    Parameters
    ----------
    point_start :
        First point of the reflection axis
    point_end :
        Second point of the reflection axis

    """
    point_start = np.array(point_start, float)
    point_end = np.array(point_end, float)

    shape = default_test_shape()

    # create reflected shape
    shape_reflected = shape.reflect_across_line(point_start, point_end)

    # original shape is not modified
    check_shapes_identical(shape, default_test_shape())

    arc_segment = shape.segments[0]
    arc_segment_ref = shape_reflected.segments[0]
    line_segment = shape.segments[1]
    line_segment_ref = shape_reflected.segments[1]

    # check reflected points
    check_point_reflected_across_line(
        arc_segment.point_start, arc_segment_ref.point_start, point_start, point_end
    )
    check_point_reflected_across_line(
        arc_segment.point_end, arc_segment_ref.point_end, point_start, point_end
    )
    check_point_reflected_across_line(
        arc_segment.point_center, arc_segment_ref.point_center, point_start, point_end
    )

    check_point_reflected_across_line(
        line_segment.point_start, line_segment_ref.point_start, point_start, point_end
    )
    check_point_reflected_across_line(
        line_segment.point_end, line_segment_ref.point_end, point_start, point_end
    )

    # apply same reflection in place
    shape.apply_reflection_across_line(point_start, point_end)
    check_shapes_identical(shape, shape_reflected)


def test_shape_reflection_across_line():
    """Test multiple reflections."""
    shape_reflection_across_line_test_case([0, 0], [0, 1])
    shape_reflection_across_line_test_case([0, 0], [1, 0])
    shape_reflection_across_line_test_case([-3, 2.5], [31.53, -23.44])
    shape_reflection_across_line_test_case([7, 8], [9, 10])
    shape_reflection_across_line_test_case([-4.26, -23.1], [-8, -0.12])
    shape_reflection_across_line_test_case([-2, 1], [2, -4.5])

    # exceptions ------------------------------------------
    shape = default_test_shape()

    with pytest.raises(Exception):
        shape.reflect_across_line([2, 5], [2, 5])
    with pytest.raises(Exception):
        shape.apply_reflection_across_line([-3, 2], [-3, 2])


def segment_interpolation_nearest(segment_a, segment_b, weight):
    """Interpolate 2 segments by taking the nearest one.

    Parameters
    ----------
    segment_a :
        First segment
    segment_b :
        Second segment
    weight :
        Interpolation weight

    Returns
    -------
    weldx.geometry.LineSegment
        Nearest segment

    """
    if weight > 0.5:
        return segment_b
    return segment_a


def test_shape_interpolation_general():
    """Test the shapes interpolation function.

    Creates 2 shapes, each containing 2 segments. Different segment
    interpolations are used. Afterwards, the shapes are interpolated using
    different weights and the results are compared to the expected values.

    """
    # create shapes
    shape_a = geo.Shape().add_line_segments([[-1, -1], [1, 1], [3, -1]])
    shape_b = geo.Shape().add_line_segments([[-1, 4], [1, 1], [3, 4]])

    # define interpolation schemes
    interpolations = [
        geo.LineSegment.linear_interpolation,
        segment_interpolation_nearest,
    ]

    for i in range(6):
        # interpolate shapes
        weight = i / 5.0
        shape_c = geo.Shape.interpolate(shape_a, shape_b, weight, interpolations)

        # check result
        if weight > 0.5:
            last_point_exp = [3, 4]
        else:
            last_point_exp = [3, -1]

        points_exp = [[-1, -1 + 5 * weight], [1, 1], last_point_exp]
        shape_c_exp = geo.Shape().add_line_segments(points_exp)

        check_shapes_identical(shape_c, shape_c_exp)

    # check weight clipped to valid range -----------------

    shape_d = geo.Shape.linear_interpolation(shape_a, shape_b, -3)
    check_shapes_identical(shape_d, shape_a)

    shape_e = geo.Shape.linear_interpolation(shape_a, shape_b, 100)
    check_shapes_identical(shape_e, shape_b)

    # exceptions ------------------------------------------

    # interpolation destroys shape continuity
    shape_f = geo.Shape().add_line_segments([[-1, 4], [2, 2], [3, 4]])
    with pytest.raises(Exception):
        geo.Shape.interpolate(shape_a, shape_f, 0.5, interpolations)

    # number of segments differ
    shape_a.add_line_segments([2, 2])
    with pytest.raises(Exception):
        geo.Shape.linear_interpolation(shape_a, shape_b, 0.25)


def test_shape_linear_interpolation():
    """Test the shapes linear interpolation function.

    Creates 2 shapes, each containing 2 segments. Afterwards, the shapes are
    interpolated using different weights and the results are compared to the
    expected values.

    """
    # create shapes
    shape_a = geo.Shape().add_line_segments([[0, 0], [1, 1], [2, 0]])
    shape_b = geo.Shape().add_line_segments([[1, 1], [2, -1], [3, 5]])

    for i in range(5):
        # interpolate shapes
        weight = i / 4.0
        shape_c = geo.Shape.linear_interpolation(shape_a, shape_b, weight)

        # check result
        points_exp = [
            [weight, weight],
            [1 + weight, 1 - 2 * weight],
            [2 + weight, 5 * weight],
        ]
        shape_c_exp = geo.Shape().add_line_segments(points_exp)

        check_shapes_identical(shape_c, shape_c_exp)

    # check weight clipped to valid range -----------------

    shape_d = geo.Shape.linear_interpolation(shape_a, shape_b, -3)
    check_shapes_identical(shape_d, shape_a)

    shape_e = geo.Shape.linear_interpolation(shape_a, shape_b, 100)
    check_shapes_identical(shape_e, shape_b)

    # exceptions ------------------------------------------

    # number of segments differ
    shape_a.add_line_segments([2, 2])
    with pytest.raises(Exception):
        geo.Shape.linear_interpolation(shape_a, shape_b, 0.25)


# Test profile class ----------------------------------------------------------


def test_profile_construction_and_shape_addition():
    """Test profile construction and addition of shapes.

    Test details are explained by comments.

    """
    arc_segment = geo.ArcSegment.construct_with_radius([-2, -2], [-1, -1], 1)
    shape = geo.Shape(arc_segment)
    shape.add_line_segments([[0, 0], [1, 0], [2, -1], [0, -1]])

    # Check invalid types
    with pytest.raises(TypeError):
        geo.Profile(3)
    with pytest.raises(TypeError):
        geo.Profile("This is not right")
    with pytest.raises(TypeError):
        geo.Profile([2, 8, 1])

    # Check valid types
    profile = geo.Profile(shape)
    assert profile.num_shapes == 1
    profile = geo.Profile([shape, shape])
    assert profile.num_shapes == 2

    # Check invalid addition
    with pytest.raises(TypeError):
        profile.add_shapes([shape, 0.1])
    with pytest.raises(TypeError):
        profile.add_shapes(["shape"])
    with pytest.raises(TypeError):
        profile.add_shapes(0.1)

    # Check that invalid calls only raise an exception and do not invalidate
    # the internal data
    assert profile.num_shapes == 2

    # Check valid addition
    profile.add_shapes(shape)
    assert profile.num_shapes == 3
    profile.add_shapes([shape, shape])
    assert profile.num_shapes == 5

    # Check shapes
    shapes_profile = profile.shapes
    for shape_profile in shapes_profile:
        check_shapes_identical(shape, shape_profile)


def test_profile_rasterization():
    """Test the profile's rasterize function.

    The test creates a profile where all its shapes lie on the y axis. The
    gaps between each shape are identical to the raster width and they are
    added in ascending order to the profile. Therefore, all raster points
    are equidistant and can be checked easily.

    """
    raster_width = 0.1

    # create shapes
    shape0 = geo.Shape().add_line_segments([[-1, 0], [-raster_width, 0]])
    shape1 = geo.Shape().add_line_segments([[0, 0], [1, 0]])
    shape2 = geo.Shape().add_line_segments([[1 + raster_width, 0], [2, 0]])

    # create profile
    profile = geo.Profile([shape0, shape1, shape2])

    # rasterize
    data = profile.rasterize(raster_width)

    # no duplications
    assert helpers.are_all_columns_unique(data)

    # check raster data size
    expected_number_raster_points = int(round(3 / raster_width)) + 1
    assert data.shape[1] == expected_number_raster_points

    # Check that all shapes are rasterized correct
    for i in range(int(round(3 / raster_width)) + 1):
        assert ut.vector_is_close(data[:, i], [i * raster_width - 1, 0])

    # exceptions
    with pytest.raises(Exception):
        profile.rasterize(0)
    with pytest.raises(Exception):
        profile.rasterize(-3)


# Test trace segment classes --------------------------------------------------


def check_trace_segment_length(segment, tolerance=1e-9):
    """Check if a trace segment returns the correct length.

    The check calculates the segment length numerically and compares it to
    the length returned by the segment.
    The numerical algorithm calculates the distances between several points
    on the trace and sums them up. The number of points is increased until
    the difference of the sum between two iterations is way below the
    specified tolerance.

    Parameters
    ----------
    segment :
        Trace segment (any type)
    tolerance :
        Numerical tolerance (Default value = 1e-9)

    """
    lcs = segment.local_coordinate_system(1)
    length_numeric_prev = np.linalg.norm(lcs.coordinates)

    # calculate numerical length by linearization
    num_segments = 2.0
    num_iterations = 20

    # calculate numerical length with increasing number of segments until
    # the rate of change between 2 calculations is small enough
    for i in range(num_iterations):
        length_numeric = 0
        increment = 1.0 / num_segments

        cs_0 = segment.local_coordinate_system(0)
        for rel_pos in np.arange(increment, 1.0 + increment / 2, increment):
            cs_1 = segment.local_coordinate_system(rel_pos)
            length_numeric += np.linalg.norm(cs_1.coordinates - cs_0.coordinates)
            cs_0 = copy.deepcopy(cs_1)

        relative_change = length_numeric / length_numeric_prev

        length_numeric_prev = copy.deepcopy(length_numeric)
        num_segments *= 2

        if math.isclose(relative_change, 1, abs_tol=tolerance / 10):
            break
        assert i < num_iterations - 1, (
            "Segment length could not be " "determined numerically"
        )

    assert math.isclose(length_numeric, segment.length, abs_tol=tolerance)


def check_trace_segment_orientation(segment):
    """Test if the segment's local coordinate system is always oriented correctly.

    The orientation of the trace is determined numerically. A small delta is
    applied to the tested location to approximate the local direction of the
    trace. The result is compared to the local coordinate systems x-axis,
    which should always point into the trace's direction.

    Parameters
    ----------
    segment :
        Trace segment (any type)

    """
    # The initial orientation of a segment must be [1, 0, 0]
    lcs = segment.local_coordinate_system(0)
    assert ut.vector_is_close(lcs.orientation[:, 0], np.array([1, 0, 0]))

    delta = 1e-9
    for rel_pos in np.arange(0.1, 1.01, 0.1):
        lcs = segment.local_coordinate_system(rel_pos)
        lcs_d = segment.local_coordinate_system(rel_pos - delta)
        trace_direction_approx = tf.normalize(lcs.coordinates - lcs_d.coordinates)

        # Check if the x-axis is aligned with the approximate trace direction
        assert ut.vector_is_close(lcs.orientation[:, 0], trace_direction_approx, 1e-6)


def default_trace_segment_tests(segment, tolerance_length=1e-9):
    """Perform some default tests on trace segment.

    Parameters
    ----------
    segment :
        Trace segment (any type)
    tolerance_length :
        Tolerance for the length test (Default value = 1e-9)

    """
    lcs = segment.local_coordinate_system(0)

    # test that function actually returns a coordinate system class
    assert isinstance(lcs, tf.LocalCoordinateSystem)

    # check that coordinates for weight 0 are at [0, 0, 0]
    assert ut.vector_is_close(lcs.coordinates, [0, 0, 0])

    # length and orientation tests
    check_trace_segment_length(segment, tolerance_length)
    check_trace_segment_orientation(segment)


def test_linear_horizontal_trace_segment():
    """Test the linear horizontal trace segment.

    Each sub test is documented by comments.

    """
    length = 7.13
    segment = geo.LinearHorizontalTraceSegment(length)

    # default tests
    default_trace_segment_tests(segment)

    # getter tests
    assert math.isclose(segment.length, length)

    # invalid inputs
    with pytest.raises(ValueError):
        geo.LinearHorizontalTraceSegment(0)
    with pytest.raises(ValueError):
        geo.LinearHorizontalTraceSegment(-4.61)


@pytest.mark.slow
def test_radial_horizontal_trace_segment():
    """Test the radial horizontal trace segment.

    Each sub test is documented by comments.

    """
    radius = 4.74
    angle = np.pi / 1.23
    segment_cw = geo.RadialHorizontalTraceSegment(radius, angle, True)
    segment_ccw = geo.RadialHorizontalTraceSegment(radius, angle, False)

    # default tests
    default_trace_segment_tests(segment_cw, 1e-4)
    default_trace_segment_tests(segment_ccw, 1e-4)

    # getter tests
    assert math.isclose(segment_cw.angle, angle)
    assert math.isclose(segment_ccw.angle, angle)
    assert math.isclose(segment_cw.radius, radius)
    assert math.isclose(segment_ccw.radius, radius)
    assert segment_cw.is_clockwise
    assert not segment_ccw.is_clockwise

    # check positions
    for weight in np.arange(0.1, 1, 0.1):
        current_angle = angle * weight
        x_exp = np.sin(current_angle) * radius
        y_exp = (1 - np.cos(current_angle)) * radius

        lcs_cw = segment_cw.local_coordinate_system(weight)
        lcs_ccw = segment_ccw.local_coordinate_system(weight)

        assert ut.vector_is_close(lcs_cw.coordinates, [x_exp, -y_exp, 0])
        assert ut.vector_is_close(lcs_ccw.coordinates, [x_exp, y_exp, 0])

    # invalid inputs
    with pytest.raises(ValueError):
        geo.RadialHorizontalTraceSegment(0, np.pi)
    with pytest.raises(ValueError):
        geo.RadialHorizontalTraceSegment(-0.53, np.pi)
    with pytest.raises(ValueError):
        geo.RadialHorizontalTraceSegment(1, 0)
    with pytest.raises(ValueError):
        geo.RadialHorizontalTraceSegment(1, -np.pi)


# Test trace class ------------------------------------------------------------


class CustomSegment:
    """Custom trace segment for tests."""

    def __init__(self):
        """Construct a custom segment."""
        self.length = None

    @staticmethod
    def local_coordinate_system(*_args):
        """Get the local coordinate system.

        Parameters
        ----------
        _args :
            Unused parameters

        Returns
        -------
        weldx.transformations.LocalCoordinateSystem
            Local coordinate system

        """
        return tf.LocalCoordinateSystem()


def test_trace_construction():
    """Test the trace's construction."""
    linear_segment = geo.LinearHorizontalTraceSegment(1)
    radial_segment = geo.RadialHorizontalTraceSegment(1, np.pi)
    cs_coordinates = np.array([2, 3, -2])
    cs_initial = helpers.rotated_coordinate_system(coordinates=cs_coordinates)

    # test single segment construction --------------------
    trace = geo.Trace(linear_segment, cs_initial)
    assert math.isclose(trace.length, linear_segment.length)
    assert trace.num_segments == 1

    segments = trace.segments
    assert len(segments) == 1

    check_trace_segments_identical(trace.segments[0], linear_segment)

    check_coordinate_systems_identical(trace.coordinate_system, cs_initial)

    # test multi segment construction ---------------------
    trace = geo.Trace([radial_segment, linear_segment])
    assert math.isclose(trace.length, linear_segment.length + radial_segment.length)
    assert trace.num_segments == 2

    check_trace_segments_identical(trace.segments[0], radial_segment)
    check_trace_segments_identical(trace.segments[1], linear_segment)

    check_coordinate_systems_identical(
        trace.coordinate_system, tf.LocalCoordinateSystem()
    )

    # check invalid inputs --------------------------------
    with pytest.raises(TypeError):
        geo.Trace(radial_segment, linear_segment)
    with pytest.raises(TypeError):
        geo.Trace(radial_segment, 2)
    with pytest.raises(Exception):
        geo.Trace(None)

    # check construction with custom segment --------------
    custom_segment = CustomSegment()
    custom_segment.length = 3
    geo.Trace(custom_segment)

    # trace length <= 0
    with pytest.raises(Exception):
        custom_segment.length = -12
        geo.Trace(custom_segment)
    with pytest.raises(Exception):
        custom_segment.length = 0
        geo.Trace(custom_segment)


@pytest.mark.slow
def test_trace_local_coordinate_system():
    """Test the trace's local coordinate system function.

    The tested trace starts with a semicircle of radius 1 turning to the left
    and continues with a straight line of length 1.

    """
    radial_segment = geo.RadialHorizontalTraceSegment(1, np.pi)
    linear_segment = geo.LinearHorizontalTraceSegment(1)

    # check with default coordinate system ----------------
    trace = geo.Trace([radial_segment, linear_segment])

    # check first segment (radial)
    for i in range(11):
        weight = i / 10
        position = radial_segment.length * weight
        cs_trace = trace.local_coordinate_system(position)
        cs_segment = radial_segment.local_coordinate_system(weight)

        check_coordinate_systems_identical(cs_trace, cs_segment)

    # check second segment (linear)
    expected_orientation = radial_segment.local_coordinate_system(1).orientation
    for i in range(11):
        weight = i / 10
        position_on_segment = linear_segment.length * weight
        position = radial_segment.length + position_on_segment

        expected_coordinates = np.array([-position_on_segment, 2, 0])
        cs_expected = tf.LocalCoordinateSystem(
            orientation=expected_orientation, coordinates=expected_coordinates
        )
        cs_trace = trace.local_coordinate_system(position)

        check_coordinate_systems_identical(cs_trace, cs_expected)

    # check with arbitrary coordinate system --------------
    orientation = WXRotation.from_euler("x", np.pi / 2).as_matrix()
    coordinates = np.array([-3, 2.5, 5])
    cs_base = tf.LocalCoordinateSystem(orientation, coordinates)

    trace = geo.Trace([radial_segment, linear_segment], cs_base)

    # check first segment
    for i in range(11):
        weight = i / 10
        position = radial_segment.length * weight
        cs_trace = trace.local_coordinate_system(position)
        cs_segment = radial_segment.local_coordinate_system(weight)

        cs_expected = cs_segment + cs_base

        check_coordinate_systems_identical(cs_trace, cs_expected)

    # check second segment
    cs_start_seg2 = radial_segment.local_coordinate_system(1) + cs_base
    for i in range(11):
        weight = i / 10
        position_on_segment = linear_segment.length * weight
        position = radial_segment.length + position_on_segment
        lcs_coordinates = [position_on_segment, 0, 0]

        cs_exp = tf.LocalCoordinateSystem(coordinates=lcs_coordinates) + cs_start_seg2
        cs_trace = trace.local_coordinate_system(position)

        check_coordinate_systems_identical(cs_trace, cs_exp)


@pytest.mark.slow
def test_trace_rasterization():
    """Test the trace's rasterize function.

    The tested trace starts with a line segment of length 1 and continues
    with a radial segment of radius 1 and counter clockwise winding.

    """
    radial_segment = geo.RadialHorizontalTraceSegment(1, np.pi)
    linear_segment = geo.LinearHorizontalTraceSegment(1)

    # check with default coordinate system ----------------
    trace = geo.Trace([linear_segment, radial_segment])
    data = trace.rasterize(0.1)

    # no duplications
    assert helpers.are_all_columns_unique(data)

    raster_width_eff = trace.length / (data.shape[1] - 1)
    for i in range(data.shape[1]):
        trace_location = i * raster_width_eff
        if trace_location <= 1:
            assert ut.vector_is_close([trace_location, 0, 0], data[:, i])
        else:
            arc_length = trace_location - 1
            angle = arc_length  # radius 1 -> arc_length = arc_angle * radius
            x = np.sin(angle) + 1  # radius 1 -> sin(arc_angle) = x / radius
            y = 1 - np.cos(angle)

            assert ut.vector_is_close([x, y, 0], data[:, i])

    # check with arbitrary coordinate system --------------
    orientation = WXRotation.from_euler("y", np.pi / 2).as_matrix()
    coordinates = np.array([-3, 2.5, 5])
    cs_base = tf.LocalCoordinateSystem(orientation, coordinates)

    trace = geo.Trace([linear_segment, radial_segment], cs_base)
    data = trace.rasterize(0.1)

    raster_width_eff = trace.length / (data.shape[1] - 1)

    for i in range(data.shape[1]):
        trace_location = i * raster_width_eff
        if trace_location <= 1:
            x = coordinates[0]
            y = coordinates[1]
            z = coordinates[2] - trace_location
        else:
            arc_length = trace_location - 1
            angle = arc_length  # radius 1 -> arc_length = arc_angle * radius
            x = coordinates[0]
            y = coordinates[1] + 1 - np.cos(angle)
            z = coordinates[2] - 1 - np.sin(angle)

        assert ut.vector_is_close([x, y, z], data[:, i])

    # check if raster width is clipped to valid range -----
    data = trace.rasterize(1000)

    assert data.shape[1] == 2
    assert ut.vector_is_close([-3, 2.5, 5], data[:, 0])
    assert ut.vector_is_close([-3, 4.5, 4], data[:, 1])

    # exceptions ------------------------------------------
    with pytest.raises(Exception):
        trace.rasterize(0)
    with pytest.raises(Exception):
        trace.rasterize(-23.1)


# Profile interpolation classes -----------------------------------------------


def check_interpolated_profile_points(profile, c_0, c_1, c_2):
    """Check the points of an interpolated profile from the interpolation test.

    Parameters
    ----------
    profile :
        Interpolated profile.
    c_0 :
        First expected point
    c_1 :
        Second expected point
    c_2 :
        Third expected point

    """
    assert ut.vector_is_close(profile.shapes[0].segments[0].point_start, c_0)
    assert ut.vector_is_close(profile.shapes[0].segments[0].point_end, c_1)
    assert ut.vector_is_close(profile.shapes[1].segments[0].point_start, c_1)
    assert ut.vector_is_close(profile.shapes[1].segments[0].point_end, c_2)


def test_linear_profile_interpolation_sbs():
    """Test linear profile interpolation.

    Uses the default profiles which consist of two shapes. Each shape
    contains just a single line segment.

    """
    [profile_a, profile_b] = get_default_profiles()

    for i in range(5):
        weight = i / 4.0
        profile_c = geo.linear_profile_interpolation_sbs(profile_a, profile_b, weight)
        check_interpolated_profile_points(
            profile_c, [-i, 2 * i], [8 - 2 * i, 16 - 2 * i], [16, -4 * i]
        )

    # check weight clipped to valid range -----------------
    a_0 = profile_a.shapes[0].segments[0].point_start
    a_1 = profile_a.shapes[1].segments[0].point_start
    a_2 = profile_a.shapes[1].segments[0].point_end

    profile_c = geo.linear_profile_interpolation_sbs(profile_a, profile_b, -3)

    check_interpolated_profile_points(profile_c, a_0, a_1, a_2)

    profile_c = geo.linear_profile_interpolation_sbs(profile_a, profile_b, 42)

    b_0 = profile_b.shapes[0].segments[0].point_start
    b_1 = profile_b.shapes[1].segments[0].point_start
    b_2 = profile_b.shapes[1].segments[0].point_end

    check_interpolated_profile_points(profile_c, b_0, b_1, b_2)

    # exceptions ------------------------------------------

    shape_a12 = profile_a.shapes[1]
    shape_b01 = profile_b.shapes[0]
    shape_b12 = profile_b.shapes[1]

    # number of shapes differ
    profile_d = geo.Profile([shape_b01, shape_b12, shape_a12])
    with pytest.raises(Exception):
        geo.linear_profile_interpolation_sbs(profile_d, profile_b, 0.5)

    # number of segments differ
    shape_b012 = geo.Shape(
        [
            geo.LineSegment.construct_with_points(b_0, b_1),
            geo.LineSegment.construct_with_points(b_1, b_2),
        ]
    )

    profile_b2 = geo.Profile([shape_b01, shape_b012])
    with pytest.raises(Exception):
        geo.linear_profile_interpolation_sbs(profile_a, profile_b2, 0.2)


# test variable profile -------------------------------------------------------


def check_variable_profile_state(variable_profile, profiles_exp, locations_exp):
    """Check the state of a variable profile.

    Parameters
    ----------
    variable_profile :
        Variable profile that should be checked.
    profiles_exp :
        Expected stored profiles
    locations_exp :
        Expected stored locations

    """
    num_profiles = len(locations_exp)
    assert variable_profile.num_interpolation_schemes == num_profiles - 1
    assert variable_profile.num_locations == num_profiles
    assert variable_profile.num_profiles == num_profiles

    for i in range(num_profiles):
        assert math.isclose(variable_profile.locations[i], locations_exp[i])
        check_profiles_identical(variable_profile.profiles[i], profiles_exp[i])


def test_variable_profile_construction():
    """Test construction of variable profiles."""
    interpol = geo.linear_profile_interpolation_sbs

    profile_a, profile_b = get_default_profiles()

    # construction with single location and interpolation
    variable_profile = geo.VariableProfile([profile_a, profile_b], 1, interpol)
    check_variable_profile_state(variable_profile, [profile_a, profile_b], [0, 1])

    variable_profile = geo.VariableProfile([profile_a, profile_b], [1], [interpol])
    check_variable_profile_state(variable_profile, [profile_a, profile_b], [0, 1])

    # construction with location list
    variable_profile = geo.VariableProfile([profile_a, profile_b], [0, 1], interpol)
    check_variable_profile_state(variable_profile, [profile_a, profile_b], [0, 1])

    variable_profile = geo.VariableProfile(
        [profile_a, profile_b, profile_a], [1, 2], [interpol, interpol]
    )
    check_variable_profile_state(
        variable_profile, [profile_a, profile_b, profile_a], [0, 1, 2]
    )

    variable_profile = geo.VariableProfile(
        [profile_a, profile_b, profile_a], [0, 1, 2], [interpol, interpol]
    )
    check_variable_profile_state(
        variable_profile, [profile_a, profile_b, profile_a], [0, 1, 2]
    )

    # exceptions ------------------------------------------

    # first location is not 0
    with pytest.raises(Exception):
        geo.VariableProfile([profile_a, profile_b], [1, 2], interpol)

    # number of locations is not correct
    with pytest.raises(Exception):
        geo.VariableProfile(
            [profile_a, profile_b, profile_a], [1], [interpol, interpol]
        )
    with pytest.raises(Exception):
        geo.VariableProfile([profile_a, profile_b], [0, 1, 2], interpol)

    # number of interpolations is not correct
    with pytest.raises(Exception):
        geo.VariableProfile([profile_a, profile_b, profile_a], [0, 1, 2], [interpol])
    with pytest.raises(Exception):
        geo.VariableProfile(
            [profile_a, profile_b, profile_a], [0, 1, 2], [interpol, interpol, interpol]
        )

    # locations not ordered
    with pytest.raises(Exception):
        geo.VariableProfile(
            [profile_a, profile_b, profile_a], [0, 2, 1], [interpol, interpol]
        )


def test_variable_profile_local_profile():
    """Test if the local profiles of a variable profile are calculated correctly."""
    interpol = geo.linear_profile_interpolation_sbs

    profile_a, profile_b = get_default_profiles()
    variable_profile = geo.VariableProfile(
        [profile_a, profile_b, profile_a], [0, 1, 2], [interpol, interpol]
    )

    for i in range(5):
        # first segment
        location = i / 4.0
        profile = variable_profile.local_profile(location)
        check_interpolated_profile_points(
            profile, [-i, 2 * i], [8 - 2 * i, 16 - 2 * i], [16, -4 * i]
        )
        # second segment
        location += 1
        profile = variable_profile.local_profile(location)
        check_interpolated_profile_points(
            profile, [-4 + i, 8 - 2 * i], [2 * i, 8 + 2 * i], [16, -16 + 4 * i]
        )

    # check if values are clipped to valid range ----------

    profile = variable_profile.local_profile(177)
    check_interpolated_profile_points(profile, [0, 0], [8, 16], [16, 0])

    profile = variable_profile.local_profile(-2)
    check_interpolated_profile_points(profile, [0, 0], [8, 16], [16, 0])


# test geometry class ---------------------------------------------------------


def test_geometry_construction():
    """Test construction of the geometry class."""
    profile_a, profile_b = get_default_profiles()
    variable_profile = geo.VariableProfile(
        [profile_a, profile_b], [0, 1], geo.linear_profile_interpolation_sbs
    )

    radial_segment = geo.RadialHorizontalTraceSegment(1, np.pi)
    linear_segment = geo.LinearHorizontalTraceSegment(1)
    trace = geo.Trace([radial_segment, linear_segment])

    # single profile construction
    geometry = geo.Geometry(profile_a, trace)
    check_profiles_identical(geometry.profile, profile_a)
    check_traces_identical(geometry.trace, trace)

    # variable profile construction
    geometry = geo.Geometry(variable_profile, trace)
    check_variable_profiles_identical(geometry.profile, variable_profile)
    check_traces_identical(geometry.trace, trace)

    # exceptions ------------------------------------------

    # wrong types
    with pytest.raises(TypeError):
        geo.Geometry(variable_profile, profile_b)
    with pytest.raises(TypeError):
        geo.Geometry(trace, trace)
    with pytest.raises(TypeError):
        geo.Geometry(trace, profile_b)
    with pytest.raises(TypeError):
        geo.Geometry(variable_profile, "a")
    with pytest.raises(TypeError):
        geo.Geometry("42", trace)


@pytest.mark.slow
def test_geometry_rasterization_trace():
    """Test if the rasterized geometry data follows the trace.

    The utilized trace starts with a line segment of length 1 and continues
    with a radial segment of radius 1 and counter clockwise winding. Each
    individual step is documented by comments.

    """
    a0 = [1, 0]
    a1 = [1, 1]
    a2 = [0, 1]
    a3 = [-1, 1]
    a4 = [-1, 0]
    profile_points = ut.to_float_array([a0, a1, a2, a2, a3, a4]).transpose()

    # create profile
    shape_a012 = geo.Shape().add_line_segments([a0, a1, a2])
    shape_a234 = geo.Shape().add_line_segments([a2, a3, a4])
    profile_a = geo.Profile([shape_a012, shape_a234])

    # create trace
    radial_segment = geo.RadialHorizontalTraceSegment(1, np.pi / 2, False)
    linear_segment = geo.LinearHorizontalTraceSegment(1)
    trace = geo.Trace([linear_segment, radial_segment])

    # create geometry
    geometry = geo.Geometry(profile_a, trace)

    # rasterize
    # Note, if the raster width is larger than the segment, it is automatically
    # adjusted to the segment width. Hence, each rasterized profile has 6
    # points, which were defined at the beginning of the test (a2 is
    # included twice)
    data = geometry.rasterize(7, 0.1)

    # calculate the number of rasterized profiles
    num_raster_profiles = int(np.round(data.shape[1] / 6))

    # calculate effective raster width
    eff_raster_width = trace.length / (data.shape[1] / 6 - 1)
    arc_point_distance_on_trace = 2 * np.sin(eff_raster_width / 2)

    for i in range(num_raster_profiles):
        # get index of the current profiles first point
        idx_0 = i * 6

        # check first segment (line)
        if data[0, idx_0 + 2] <= 1:
            for j in range(6):
                point_exp = [
                    eff_raster_width * i,
                    profile_points[0, j],
                    profile_points[1, j],
                ]
                assert ut.vector_is_close(data[:, idx_0 + j], point_exp)
        # check second segment (arc)
        else:
            # first 2 profile points lie on the arcs center point
            assert ut.vector_is_close(data[:, idx_0], [1, a0[0], a0[1]])
            assert ut.vector_is_close(data[:, idx_0 + 1], [1, a1[0], a1[1]])

            # z-values are constant
            for j in np.arange(2, 6, 1):
                assert math.isclose(data[2, idx_0 + j], profile_points[1, j])

            # all profile points in a common x-y plane
            exp_radius = np.array([1, 1, 2, 2])

            vec_02 = data[0:2, idx_0 + 2] - data[0:2, idx_0]
            assert math.isclose(np.linalg.norm(vec_02), exp_radius[0])

            for j in np.arange(3, 6, 1):
                vec_0j = data[0:2, idx_0 + j] - data[0:2, idx_0]
                assert math.isclose(np.linalg.norm(vec_0j), exp_radius[j - 2])
                unit_vec_0j = tf.normalize(vec_0j)
                assert math.isclose(np.dot(unit_vec_0j, vec_02), 1)

            # check point distance between profiles
            if data[1, idx_0 - 4] > 1:
                exp_point_distance = arc_point_distance_on_trace * exp_radius
                for j in np.arange(2, 6, 1):
                    point_distance = np.linalg.norm(
                        data[:, idx_0 + j] - data[:, idx_0 + j - 6]
                    )
                    assert math.isclose(exp_point_distance[j - 2], point_distance)

    # check if raster width is clipped to valid range -----
    data = geometry.rasterize(7, 1000)

    assert data.shape[1] == 12

    for i in range(12):
        if i < 6:
            math.isclose(data[0, i], 0)
        else:
            assert math.isclose(data[1, i], 1)

    # exceptions ------------------------------------------
    with pytest.raises(Exception):
        geometry.rasterize(0, 1)
    with pytest.raises(Exception):
        geometry.rasterize(1, 0)
    with pytest.raises(Exception):
        geometry.rasterize(0, 0)
    with pytest.raises(Exception):
        geometry.rasterize(-2.3, 1)
    with pytest.raises(Exception):
        geometry.rasterize(1, -4.6)
    with pytest.raises(Exception):
        geometry.rasterize(-2.3, -4.6)


@pytest.mark.slow
def test_geometry_rasterization_profile_interpolation():
    """Check if the rasterized geometry interpolates profiles correctly."""
    interpol = geo.linear_profile_interpolation_sbs

    a0 = [1, 0]
    a1 = [1, 1]
    a2 = [0, 1]
    a3 = [-1, 1]
    a4 = [-1, 0]

    # create shapes
    shape_a012 = geo.Shape().add_line_segments([a0, a1, a2])
    shape_a234 = geo.Shape().add_line_segments([a2, a3, a4])

    shape_b012 = copy.deepcopy(shape_a012)
    shape_b234 = copy.deepcopy(shape_a234)
    shape_b012.apply_transformation([[2, 0], [0, 2]])
    shape_b234.apply_transformation([[2, 0], [0, 2]])

    # create variable profile
    profile_a = geo.Profile([shape_a012, shape_a234])
    profile_b = geo.Profile([shape_b012, shape_b234])

    variable_profile = geo.VariableProfile(
        [profile_a, profile_b, profile_a], [0, 2, 6], [interpol, interpol]
    )

    linear_segment_l1 = geo.LinearHorizontalTraceSegment(1)
    linear_segment_l2 = geo.LinearHorizontalTraceSegment(2)
    # Note: The profile in the middle of the variable profile is not located
    # at the start of the second trace segment
    trace = geo.Trace([linear_segment_l2, linear_segment_l1])

    geometry = geo.Geometry(variable_profile, trace)

    # Note: If the raster width is larger than the segment, it is automatically
    # adjusted to the segment width. Hence each rasterized profile has 6
    # points, which were defined at the beginning of the test (a2 is
    # included twice)
    data = geometry.rasterize(7, 0.1)
    assert data.shape[1] == 186

    profile_points = np.array([a0, a1, a2, a2, a3, a4]).transpose()

    # check first profile interpolation
    for i in range(11):
        idx_0 = i * 6
        for j in range(6):
            point_exp = np.array(
                [
                    i * 0.1,
                    profile_points[0, j] * (1 + i * 0.1),
                    profile_points[1, j] * (1 + i * 0.1),
                ]
            )
            assert ut.vector_is_close(data[:, idx_0 + j], point_exp)

    # check second profile interpolation
    for i in range(20):
        idx_0 = (30 - i) * 6
        for j in range(6):
            point_exp = np.array(
                [
                    3 - i * 0.1,
                    profile_points[0, j] * (1 + i * 0.05),
                    profile_points[1, j] * (1 + i * 0.05),
                ]
            )
            assert ut.vector_is_close(data[:, idx_0 + j], point_exp)


def get_test_profile() -> geo.Profile:
    """Create a `weldx.geometry.Profile` for tests.

    Returns
    -------
    weldx.geometry.Profile :
        `weldx.geometry.Profile` for tests.

    """
    shape_0 = geo.Shape().add_line_segments(Q_([[1, 0], [1, 1], [3, 1]], "cm"))
    shape_1 = geo.Shape().add_line_segments(Q_([[-1, 0], [-1, 1]], "cm"))
    return geo.Profile([shape_0, shape_1])


def get_test_geometry_constant_profile() -> geo.Geometry:
    """Create a `weldx.geometry.Geometry` with constant profile for tests.

    Returns
    -------
    weldx.geometry.Geometry :
        `weldx.geometry.Geometry` with constant profile for tests.

    """
    profile = get_test_profile()
    trace = geo.Trace([geo.LinearHorizontalTraceSegment(Q_(1, "cm"))])
    return geo.Geometry(profile=profile, trace=trace)


def get_test_geometry_variable_profile():
    """Create a `weldx.geometry.Geometry` with variable profile for tests.

    Returns
    -------
    weldx.geometry.Geometry :
        `weldx.geometry.Geometry` with constant profile for tests.

    """
    profile = get_test_profile()
    variable_profile = geo.VariableProfile(
        [profile, profile], [0, 1], [geo.linear_profile_interpolation_sbs]
    )
    trace = geo.Trace([geo.LinearHorizontalTraceSegment(Q_(1, "cm"))])
    return geo.Geometry(profile=variable_profile, trace=trace)


class TestGeometry:
    """Test the geometry class."""

    @staticmethod
    @pytest.mark.parametrize(
        "geometry, p_rw, t_rw, exp_num_points, exp_num_triangles",
        [
            (get_test_geometry_constant_profile(), Q_(1, "cm"), Q_(1, "cm"), 12, 8),
            (get_test_geometry_variable_profile(), Q_(1, "cm"), Q_(1, "cm"), 12, 0),
        ],
    )
    def test_spatial_data(
        geometry: geo.Geometry,
        p_rw: pint.Quantity,
        t_rw: pint.Quantity,
        exp_num_points: int,
        exp_num_triangles: int,
    ):
        """Test the `spatial_data` function.

        Parameters
        ----------
        geometry : weldx.geometry.Geometry
            Geometry that should be tested
        p_rw : pint.Quantity
            Profile raster width that is passed to the function
        t_rw : pint.Quantity
            Trace raster width that is passed to the function
        exp_num_points : int
            Expected number of points of the returned `weldx.geometry.SpatialData`
            instance
        exp_num_triangles : int
            Expected number of triangles of the returned `weldx.geometry.SpatialData`
            instance

        """
        spatial_data = geometry.spatial_data(p_rw, t_rw)
        assert len(spatial_data.coordinates.data) == exp_num_points

        num_triangles = 0
        if spatial_data.triangles is not None:
            num_triangles = len(spatial_data.triangles)
        assert num_triangles == exp_num_triangles


# --------------------------------------------------------------------------------------
# SpatialData
# --------------------------------------------------------------------------------------


class TestSpatialData:
    """Test the functionality of the `SpatialData` class."""

    @staticmethod
    @pytest.mark.parametrize(
        "arguments",
        [
            (np.ones((5, 3)),),
            (np.ones((5, 3)), [[0, 1, 2], [0, 2, 3]]),
            (np.ones((5, 3)), [[0, 1, 2], [0, 2, 3]], {}),
            (np.ones((5, 3)), None, {}),
        ],
    )
    def test_class_creation(arguments):
        """Test creation of a `SpatialData` instance.

        Parameters
        ----------
        arguments :
            Tuple of arguments that are passed to the `__init__` method

        """
        pc = SpatialData(*arguments)
        assert isinstance(pc.coordinates, DataArray)
        assert np.allclose(pc.coordinates.data, arguments[0])

        if len(arguments) > 1 and arguments[1] is not None:
            np.all(arguments[1] == pc.triangles)

    # test_class_creation_exceptions ---------------------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        "arguments, exception_type, test_name",
        [
            ((np.ones((5, 3)), [[0, 1], [2, 3]]), ValueError, "# inv. triangulation 1"),
            ((np.ones((5, 3)), [[0, 1, 2, 3]]), ValueError, "# inv. triangulation 2"),
            ((np.ones((5, 3)), [0, 1, 2]), ValueError, "# inv. triangulation 3"),
        ],
    )
    def test_class_creation_exceptions(arguments, exception_type, test_name):
        """Test exceptions during creation of a `SpatialData` instance.

        Parameters
        ----------
        arguments :
            Tuple of arguments that are passed to the `__init__` method
        exception_type :
            Expected exception type
        test_name : str
            A string starting with an `#` that describes the test.

        """
        with pytest.raises(exception_type):
            SpatialData(*arguments)

    # test_comparison ------------------------------------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        "kwargs_mod, expected_result",
        [
            ({}, True),
            (dict(coordinates=[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 1]]), False),
            (dict(coordinates=[[0, 0, 0], [1, 0, 0], [1, 1, 0]]), False),
            (dict(triangles=[[0, 1, 2], [2, 3, 1]]), False),
            (dict(triangles=[[0, 1, 2], [2, 3, 1], [2, 3, 1]]), False),
            (dict(triangles=[[0, 1, 2]]), False),
            (dict(triangles=None), False),
            (dict(attributes=dict(data=[2, 2, 3])), False),
            (dict(attributes=dict(dat=[1, 2, 3])), False),
            # uncomment once issue #376 is resolved
            # (dict(attributes=dict(data=[1, 2, 3], more=[1, 2, 5])), False),
            (dict(attributes={}), False),
            (dict(attributes=None), False),
        ],
    )
    def test_comparison(kwargs_mod: Dict, expected_result: bool):
        """Test the comparison operator by comparing two instances.

        Parameters
        ----------
        kwargs_mod :
            A dictionary of key word arguments that is used to overwrite the default
            values in the RHS `SpatialData`. If an empty dict is passed, LHS and RHS
            are constructed with the same values.
        expected_result :
            Expected result of the comparison

        """
        from copy import deepcopy

        default_kwargs = dict(
            coordinates=[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
            triangles=[[0, 1, 2], [2, 3, 0]],
            attributes=dict(data=[1, 2, 3]),
        )
        reference = SpatialData(**default_kwargs)

        kwargs_other = deepcopy(default_kwargs)
        kwargs_other.update(kwargs_mod)
        other = SpatialData(**kwargs_other)

        assert (reference == other) == expected_result

    # test_read_write_file -------------------------------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        "filename",
        ["test.ply", "test.stl", "test.vtk", Path("test.stl")],
    )
    def test_read_write_file(filename: Union[str, Path]):
        """Test the `from_file` and `write_to_file` functions.

        The test simply creates a `SpatialData` instance, writes it to a file and reads
        it back. The result is compared to the original object.

        Parameters
        ----------
        filename :
            Name of the file

        """
        points = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
        triangles = [[0, 1, 2], [2, 3, 0]]

        data = SpatialData(points, triangles)
        with TemporaryDirectory(dir=Path(__file__).parent) as tmpdirname:
            filepath = f"{tmpdirname}/{filename}"
            if isinstance(filename, Path):
                filepath = Path(filepath)
            data.write_to_file(filepath)
            data_read = SpatialData.from_file(filepath)

        assert np.allclose(data.coordinates, data_read.coordinates)
        assert np.allclose(data.triangles, data_read.triangles)
