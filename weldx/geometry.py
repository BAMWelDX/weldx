"""Provides classes to define lines and surfaces."""
from __future__ import annotations

import copy
import math
from dataclasses import InitVar, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Tuple, Union

import meshio
import numpy as np
import pint
from xarray import DataArray

import weldx.transformations as tf
import weldx.util as ut
from weldx.constants import Q_
from weldx.constants import WELDX_UNIT_REGISTRY as UREG
from weldx.time import Time

_DEFAULT_LEN_UNIT = UREG.millimeters
_DEFAULT_ANG_UNIT = UREG.rad

# only import heavy-weight packages on type checking
if TYPE_CHECKING:  # pragma: no cover
    import matplotlib.axes

    import weldx.visualization.types as vs_types
    import weldx.welding.groove.iso_9692_1 as iso

# helper -------------------------------------------------------------------------------


def has_cw_ordering(points: np.ndarray):
    """Return `True` if a set of points has clockwise ordering, `False` otherwise.

    Notes
    -----
        The algorithm was taken from the following Stack Overflow answer:
        https://stackoverflow.com/a/1165943/6700329

    """
    if sum((points[1:, 1] - points[:-1, 1]) * (points[1:, 2] + points[:-1, 2])) < 0:
        return False
    return True


# todo: Note that this is a copy of the weldx.tests._helpers.py function.
def _vector_is_close(vec_a, vec_b, abs_tol=1e-9) -> bool:
    """Check if a vector is close or equal to another vector.

    Parameters
    ----------
    vec_a :
        First vector
    vec_b :
        Second vector
    abs_tol :
        Absolute tolerance (Default value = 1e-9)

    Returns
    -------
    bool
        True or False

    """
    if isinstance(vec_a, pint.Quantity):
        vec_a = vec_a.m
    if isinstance(vec_b, pint.Quantity):
        vec_b = vec_b.m
    vec_a = np.array(vec_a, dtype=float)
    vec_b = np.array(vec_b, dtype=float)

    if vec_a.size != vec_b.size:
        return False
    return np.all(np.isclose(vec_a, vec_b, atol=abs_tol)).__bool__()


def _to_list(var) -> list:
    """Store the passed variable into a list and return it.

    If the variable is already a list, it is returned without modification.
    If `None` is passed, the function returns an empty list.

    Parameters
    ----------
    var :
        Arbitrary variable

    Returns
    -------
    list

    """
    if isinstance(var, list):
        return var
    if var is None:
        return []
    return [var]


# LineSegment -----------------------------------------------------------------


class LineSegment:
    """Line segment."""

    @UREG.wraps(None, (None, _DEFAULT_LEN_UNIT), strict=True)
    def __init__(self, points: pint.Quantity):
        """Construct line segment.

        Parameters
        ----------
        points :
            2x2 matrix of points. The first column is the
            starting point and the second column the end point.

        Returns
        -------
        LineSegment

        """
        if not len(points.shape) == 2:
            raise ValueError("'points' must be a 2d array/matrix.")
        if not (points.shape[0] == 2 and points.shape[1] == 2):
            raise ValueError("'points' is not a 2x2 matrix.")
        self._points = points.astype(float)
        self._calculate_length()

    def __repr__(self):
        """Output representation of a LineSegment."""
        return f"LineSegment('points'={self._points!r}, 'length'={self._length!r})"

    def __str__(self):
        """Output simple string representation of a LineSegment."""
        p1 = np.array2string(self.points[:, 0].m, precision=2, separator=",")
        p2 = np.array2string(self.points[:, 1].m, precision=2, separator=",")
        return f"Line: {p1} -> {p2}"

    def _calculate_length(self):
        """Calculate the segment length from its points."""
        self._length = np.linalg.norm(self._points[:, 1] - self._points[:, 0])
        if math.isclose(self._length, 0):
            raise ValueError("Segment length is 0.")

    @classmethod
    @UREG.check(None, _DEFAULT_LEN_UNIT, _DEFAULT_LEN_UNIT)
    def construct_with_points(
        cls, point_start: pint.Quantity, point_end: pint.Quantity
    ) -> LineSegment:
        """Construct a line segment with two points.

        Parameters
        ----------
        point_start :
            Starting point of the segment
        point_end :
            End point of the segment

        Returns
        -------
        LineSegment
            Line segment

        """
        points = np.transpose(np.array([point_start.m, point_end.m], dtype=float))
        return cls(Q_(points, _DEFAULT_LEN_UNIT))

    @classmethod
    def linear_interpolation(
        cls, segment_a: LineSegment, segment_b: LineSegment, weight: float
    ):
        """Interpolate two line segments linearly.

        Parameters
        ----------
        segment_a :
            First segment
        segment_b :
            Second segment
        weight :
            Weighting factor in the range [0 .. 1] where 0 is
            segment a and 1 is segment b

        Returns
        -------
        LineSegment
            Interpolated segment

        """
        if not isinstance(segment_a, cls) or not isinstance(segment_b, cls):
            raise TypeError("Parameters a and b must both be line segments.")

        weight = np.clip(weight, 0, 1)
        points = (1 - weight) * segment_a.points.m + weight * segment_b.points.m
        return cls(Q_(points, _DEFAULT_LEN_UNIT))

    @property
    @UREG.wraps(_DEFAULT_LEN_UNIT, (None,), strict=True)
    def length(self) -> pint.Quantity:
        """Get the segment length.

        Returns
        -------
        pint.Quantity
            Segment length

        """
        return self._length

    @property
    @UREG.wraps(_DEFAULT_LEN_UNIT, (None,), strict=True)
    def point_end(self) -> pint.Quantity:
        """Get the end point of the segment.

        Returns
        -------
        pint.Quantity
            End point

        """
        return self._points[:, 1]

    @property
    @UREG.wraps(_DEFAULT_LEN_UNIT, (None,), strict=True)
    def point_start(self) -> pint.Quantity:
        """Get the starting point of the segment.

        Returns
        -------
        pint.Quantity
            Starting point

        """
        return self._points[:, 0]

    @property
    @UREG.wraps(_DEFAULT_LEN_UNIT, (None,), strict=True)
    def points(self) -> pint.Quantity:
        """Get the segments points in form of a 2x2 matrix.

        The first column represents the starting point and the second one the end point.

        Returns
        -------
        pint.Quantity
            2x2 matrix containing the segments points

        """
        return self._points

    def apply_transformation(self, matrix):
        """Apply a transformation matrix to the segment.

        Parameters
        ----------
        matrix :
            Transformation matrix

        """
        self._points = np.matmul(matrix, self._points)
        self._calculate_length()

    @UREG.wraps(None, (None, _DEFAULT_LEN_UNIT), strict=True)
    def apply_translation(self, vector: pint.Quantity):
        """Apply a translation to the segment.

        Parameters
        ----------
        vector :
            Translation vector

        """
        self._points += np.ndarray((2, 1), float, np.array(vector, float))

    @UREG.wraps(_DEFAULT_LEN_UNIT, (None, _DEFAULT_LEN_UNIT), strict=True)
    def rasterize(self, raster_width: pint.Quantity) -> pint.Quantity:
        """Create an array of points that describe the segments contour.

        The effective raster width may vary from the specified one,
        since the algorithm enforces constant distances between two
        raster points.

        Parameters
        ----------
        raster_width :
            The desired distance between two raster points

        Returns
        -------
        pint.Quantity
            Array of contour points

        """
        if not raster_width > 0:
            raise ValueError("'raster_width' must be > 0")
        raster_width = np.min([raster_width, self.length.m])

        num_raster_segments = np.round(self.length.m / raster_width)

        # normalized effective raster width
        nerw = 1.0 / num_raster_segments

        multiplier = np.arange(0, 1 + 0.5 * nerw, nerw)
        weight_matrix = np.array([1 - multiplier, multiplier])

        return np.matmul(self._points, weight_matrix)

    def transform(self, matrix: np.ndarray) -> LineSegment:
        """Get a transformed copy of the segment.

        Parameters
        ----------
        matrix :
            Transformation matrix

        Returns
        -------
        LineSegment
            Transformed copy

        """
        new_segment = copy.deepcopy(self)
        new_segment.apply_transformation(matrix)
        return new_segment

    @UREG.check(None, _DEFAULT_LEN_UNIT)
    def translate(self, vector: pint.Quantity) -> LineSegment:
        """Get a translated copy of the segment.

        Parameters
        ----------
        vector :
            Translation vector

        Returns
        -------
        LineSegment
            Transformed copy

        """
        new_segment = copy.deepcopy(self)
        new_segment.apply_translation(vector)
        return new_segment


# ArcSegment ------------------------------------------------------------------


class ArcSegment:
    """Arc segment."""

    @UREG.wraps(None, (None, _DEFAULT_LEN_UNIT, None), strict=True)
    def __init__(self, points: pint.Quantity, arc_winding_ccw: bool = True):
        """Construct arc segment.

        Parameters
        ----------
        points :
            2x3 matrix of points. The first column is the starting point,
            the second column the end point and the last the center point.
        arc_winding_ccw :
            Specifies if the arcs winding order is counter-clockwise

        Returns
        -------
        ArcSegment

        """
        if not len(points.shape) == 2:
            raise ValueError("'points' must be a 2d array/matrix.")
        if not (points.shape[0] == 2 and points.shape[1] == 3):
            raise ValueError("'points' is not a 2x3 matrix.")

        if arc_winding_ccw:
            self._sign_arc_winding = 1
        else:
            self._sign_arc_winding = -1
        self._points = points

        self._arc_angle = None
        self._arc_length = None
        self._radius = None
        self._calculate_arc_parameters()

    def __repr__(self):
        """Output representation of an ArcSegment."""
        return (
            f"ArcSegment('points': {self._points!r}, 'arc_angle': {self._arc_angle!r}, "
            f"'radius': {self._radius!r}, "
            f"'sign_arc_winding': {self._sign_arc_winding!r}, "
            f"'arc_length': {self._arc_length!r})"
        )

    def __str__(self):
        """Output simple string representation of an ArcSegment."""
        values = np.array(
            [self._radius, self._arc_angle / np.pi * 180, self._arc_length]
        )
        return f"Arc : {np.array2string(values, precision=2, separator=',')}"

    def _calculate_arc_angle(self):
        """Calculate the arc angle."""
        point_start = self.point_start.m
        point_end = self.point_end.m
        point_center = self.point_center.m

        # Calculate angle between vectors (always the smaller one)
        unit_center_start = tf.normalize(point_start - point_center)
        unit_center_end = tf.normalize(point_end - point_center)

        dot_unit = np.dot(unit_center_start, unit_center_end)
        angle_vecs = np.arccos(np.clip(dot_unit, -1, 1))

        sign_winding_points = tf.vector_points_to_left_of_vector(
            unit_center_end, unit_center_start
        )

        if np.abs(sign_winding_points + self._sign_arc_winding) > 0:
            self._arc_angle = angle_vecs
        else:
            self._arc_angle = 2 * np.pi - angle_vecs

    def _calculate_arc_parameters(self):
        """Calculate radius, arc length and arc angle from the segments points."""
        self._radius = np.linalg.norm(self._points[:, 0] - self._points[:, 2])
        self._calculate_arc_angle()
        self._arc_length = self._arc_angle * self._radius

        self._check_valid()

    def _check_valid(self):
        """Check if the segments data is valid."""
        point_start = self.point_start.m
        point_end = self.point_end.m
        point_center = self.point_center.m

        radius_start_center = np.linalg.norm(point_start - point_center)
        radius_end_center = np.linalg.norm(point_end - point_center)
        radius_diff = radius_end_center - radius_start_center

        if not math.isclose(radius_diff, 0, abs_tol=1e-9):
            raise ValueError("Radius is not constant.")
        if math.isclose(self._arc_length, 0):
            raise ValueError("Arc length is 0.")

    @classmethod
    @UREG.wraps(
        None,
        (None, _DEFAULT_LEN_UNIT, _DEFAULT_LEN_UNIT, _DEFAULT_LEN_UNIT, None),
        strict=True,
    )
    def construct_with_points(
        cls,
        point_start: pint.Quantity,
        point_end: pint.Quantity,
        point_center: pint.Quantity,
        arc_winding_ccw: bool = True,
    ) -> ArcSegment:
        """Construct an arc segment with three points (start, end, center).

        Parameters
        ----------
        point_start :
            Starting point of the segment
        point_end :
            End point of the segment
        point_center :
            Center point of the arc
        arc_winding_ccw :
            Specifies if the arcs winding order is
            counter-clockwise (Default value = True)

        Returns
        -------
        ArcSegment
            Arc segment

        """
        points = np.transpose(
            np.array([point_start, point_end, point_center], dtype=float)
        )
        return cls(Q_(points, _DEFAULT_LEN_UNIT), arc_winding_ccw)

    @classmethod
    @UREG.wraps(
        None,
        (None, _DEFAULT_LEN_UNIT, _DEFAULT_LEN_UNIT, _DEFAULT_LEN_UNIT, None, None),
        strict=True,
    )
    def construct_with_radius(
        cls,
        point_start: pint.Quantity,
        point_end: pint.Quantity,
        radius: pint.Quantity,
        center_left_of_line: bool = True,
        arc_winding_ccw: bool = True,
    ) -> ArcSegment:
        """Construct an arc segment with a radius and the start and end points.

        Parameters
        ----------
        point_start :
            Starting point of the segment
        point_end :
            End point of the segment
        radius :
            Radius
        center_left_of_line :
            Specifies if the center point is located
            to the left of the vector point_start -> point_end (Default value = True)
        arc_winding_ccw :
            Specifies if the arcs winding order is
            counter-clockwise (Default value = True)

        Returns
        -------
        ArcSegment
            Arc segment

        """
        vec_start_end = point_end - point_start
        if center_left_of_line:
            vec_normal = np.array([-vec_start_end[1], vec_start_end[0]])
        else:
            vec_normal = np.array([vec_start_end[1], -vec_start_end[0]])

        squared_length = np.dot(vec_start_end, vec_start_end)
        squared_radius = radius * radius

        normal_scaling = np.sqrt(
            np.clip(squared_radius / squared_length - 0.25, 0, None)
        )

        vec_start_center = 0.5 * vec_start_end + vec_normal * normal_scaling
        point_center = point_start + vec_start_center

        return cls.construct_with_points(
            Q_(point_start, _DEFAULT_LEN_UNIT),
            Q_(point_end, _DEFAULT_LEN_UNIT),
            Q_(point_center, _DEFAULT_LEN_UNIT),
            arc_winding_ccw,
        )

    @classmethod
    def linear_interpolation(
        cls, segment_a: ArcSegment, segment_b: ArcSegment, weight: float
    ) -> ArcSegment:
        """Interpolate two arc segments linearly.

        This function is not implemented, since linear interpolation of an
        arc segment is not unique. The 'Shape' class requires succeeding
        segments to be connected through a common point. Therefore two
        connected segments must interpolate the connecting point in the same
        way. Connecting an arc segment to two line segments would enforce a
        linear interpolation of the start and end points. If the centre
        point is also interpolated in a linear way, might (or might not)
        result in different distances of start and end point to the center,
        which invalidates the arc segment. Alternatively, one can
        interpolate the radius linearly which guarantees a valid arc
        segment, but this can cause the center point to vary even though it
        is the same in both interpolated segments. To ensure the desired
        interpolation behavior, you have to provide a custom interpolation.

        Parameters
        ----------
        segment_a :
            First segment
        segment_b :
            Second segment
        weight :
            Weighting factor in the range [0 .. 1] where 0 is
            segment a and 1 is segment b

        Returns
        -------
        ArcSegment
            Interpolated segment

        """
        raise NotImplementedError(
            "Linear interpolation of an arc segment is not unique (see "
            "docstring). You need to provide a custom interpolation."
        )

    @property
    @UREG.wraps(_DEFAULT_ANG_UNIT, (None,), strict=True)
    def arc_angle(self) -> pint.Quantity:
        """Get the arc angle.

        Returns
        -------
        pint.Quantity
            Arc angle

        """
        return self._arc_angle

    @property
    @UREG.wraps(_DEFAULT_LEN_UNIT, (None,), strict=True)
    def arc_length(self) -> pint.Quantity:
        """Get the arc length.

        Returns
        -------
        pint.Quantity
            Arc length

        """
        return self._arc_length

    @property
    def arc_winding_ccw(self) -> bool:
        """Get True if the winding order is counter-clockwise. False if clockwise.

        Returns
        -------
        bool
            True or False

        """
        return self._sign_arc_winding > 0

    @property
    @UREG.wraps(_DEFAULT_LEN_UNIT, (None,), strict=True)
    def point_center(self) -> pint.Quantity:
        """Get the center point of the segment.

        Returns
        -------
        pint.Quantity
            Center point

        """
        return self._points[:, 2]

    @property
    @UREG.wraps(_DEFAULT_LEN_UNIT, (None,), strict=True)
    def point_end(self) -> pint.Quantity:
        """Get the end point of the segment.

        Returns
        -------
        pint.Quantity
            End point

        """
        return self._points[:, 1]

    @property
    @UREG.wraps(_DEFAULT_LEN_UNIT, (None,), strict=True)
    def point_start(self) -> pint.Quantity:
        """Get the starting point of the segment.

        Returns
        -------
        pint.Quantity
            Starting point

        """
        return self._points[:, 0]

    @property
    @UREG.wraps(_DEFAULT_LEN_UNIT, (None,), strict=True)
    def points(self) -> pint.Quantity:
        """Get the segments points in form of a 2x3 matrix.

        The first column represents the starting point, the second one the
        end and the third one the center.

        Returns
        -------
        pint.Quantity
            2x3 matrix containing the segments points

        """
        return self._points

    @property
    @UREG.wraps(_DEFAULT_LEN_UNIT, (None,), strict=True)
    def radius(self) -> pint.Quantity:
        """Get the radius.

        Returns
        -------
        pint.Quantity
            Radius

        """
        return self._radius

    def apply_transformation(self, matrix: np.ndarray):
        """Apply a transformation to the segment.

        Parameters
        ----------
        matrix :
            Transformation matrix

        """
        self._points = np.matmul(matrix, self._points)
        self._sign_arc_winding *= tf.reflection_sign(matrix)
        self._calculate_arc_parameters()

    @UREG.wraps(None, (None, _DEFAULT_LEN_UNIT), strict=True)
    def apply_translation(self, vector: pint.Quantity):
        """Apply a translation to the segment.

        Parameters
        ----------
        vector :
            Translation vector

        """
        self._points += np.ndarray((2, 1), float, np.array(vector, float))

    @UREG.wraps(_DEFAULT_LEN_UNIT, (None, _DEFAULT_LEN_UNIT), strict=True)
    def rasterize(self, raster_width: pint.Quantity) -> pint.Quantity:
        """Create an array of points that describe the segments contour.

        The effective raster width may vary from the specified one,
        since the algorithm enforces constant distances between two
        raster points.

        Parameters
        ----------
        raster_width :
            The desired distance between two raster points

        Returns
        -------
        pint.Quantity
            Array of contour points

        """
        point_start = self.point_start.m
        point_center = self.point_center.m
        vec_center_start = point_start - point_center
        if not raster_width > 0:
            raise ValueError("'raster_width' must be > 0")
        raster_width = np.clip(raster_width, None, self.arc_length.m)

        num_raster_segments = int(np.round(self._arc_length / raster_width))
        delta_angle = self._arc_angle / num_raster_segments

        max_angle = self._sign_arc_winding * (self._arc_angle + 0.5 * delta_angle)
        angles = np.arange(0, max_angle, self._sign_arc_winding * delta_angle)

        rotation_matrices = tf.WXRotation.from_euler("z", angles).as_matrix()[
            :, 0:2, 0:2
        ]

        data = np.matmul(rotation_matrices, vec_center_start) + point_center

        return data.transpose()

    def transform(self, matrix) -> ArcSegment:
        """Get a transformed copy of the segment.

        Parameters
        ----------
        matrix :
            Transformation matrix

        Returns
        -------
        ArcSegment
            Transformed copy

        """
        new_segment = copy.deepcopy(self)
        new_segment.apply_transformation(matrix)
        return new_segment

    @UREG.check(None, "[length]")
    def translate(self, vector) -> ArcSegment:
        """Get a translated copy of the segment.

        Parameters
        ----------
        vector :
            Translation vector

        Returns
        -------
        ArcSegment
            Transformed copy

        """
        new_segment = copy.deepcopy(self)
        new_segment.apply_translation(vector)
        return new_segment


# Shape class -----------------------------------------------------------------

segment_types = Union[LineSegment, ArcSegment]


class Shape:
    """Defines a shape in 2 dimensions."""

    def __init__(self, segments: Union[segment_types, List[segment_types]] = None):
        """Construct a shape.

        Parameters
        ----------
        segments :
            Single segment or list of segments

        Returns
        -------
        Shape

        """
        segments = _to_list(segments)
        self._check_segments_connected(segments)
        self._segments = segments

    def __repr__(self):
        """Output representation of a Shape."""
        return f"Shape('segments': {self.segments!r})"

    def __str__(self):
        """Output simple string representation of a Shape (listing segments)."""
        shape_str = "\n".join(str(s) for s in self.segments)
        return f"{shape_str}"

    @staticmethod
    def _check_segments_connected(segments: Union[segment_types, List[segment_types]]):
        """Check if all segments are connected to each other.

        The start point of a segment must be identical to the end point of
        the previous segment.

        Parameters
        ----------
        segments :
            List of segments

        """
        for i in range(len(segments) - 1):
            if not _vector_is_close(segments[i].point_end, segments[i + 1].point_start):
                raise ValueError("Segments are not connected.")

    @classmethod
    def interpolate(
        cls, shape_a: Shape, shape_b: Shape, weight: float, interpolation_schemes
    ) -> Shape:
        """Interpolate 2 shapes.

        Parameters
        ----------
        shape_a :
            First shape
        shape_b :
            Second shape
        weight :
            Weighting factor in the range [0 .. 1] where 0 is
            shape a and 1 is shape b
        interpolation_schemes :
            List of interpolation schemes for each
            segment of the shape.

        Returns
        -------
        Shape
            Interpolated shape

        """
        if not shape_a.num_segments == shape_b.num_segments:
            raise ValueError("Number of segments differ.")

        weight = np.clip(weight, 0, 1)

        segments_c = []
        for i in range(shape_a.num_segments):
            segments_c += [
                interpolation_schemes[i](
                    shape_a.segments[i], shape_b.segments[i], weight
                )
            ]
        return cls(segments_c)

    @classmethod
    def linear_interpolation(
        cls, shape_a: Shape, shape_b: Shape, weight: float
    ) -> Shape:
        """Interpolate 2 shapes linearly.

        Each segment is interpolated individually, using the corresponding
        linear segment interpolation.

        Parameters
        ----------
        shape_a :
            First shape
        shape_b :
            Second shape
        weight :
            Weighting factor in the range [0 .. 1] where 0 is
            shape a and 1 is shape b

        Returns
        -------
        Shape
            Interpolated shape

        """
        interpolation_schemes = []
        for i in range(shape_a.num_segments):
            interpolation_schemes += [shape_a.segments[i].linear_interpolation]

        return cls.interpolate(shape_a, shape_b, weight, interpolation_schemes)

    @property
    def num_segments(self) -> int:
        """Get the number of segments of the shape.

        Returns
        -------
        int
            number of segments

        """
        return len(self._segments)

    @property
    def segments(self) -> List[segment_types]:
        """Get the shape's segments.

        Returns
        -------
        list
            List of segments

        """
        return self._segments

    @UREG.wraps(None, (None, _DEFAULT_LEN_UNIT), strict=True)
    def add_line_segments(self, points: pint.Quantity):
        """Add line segments to the shape.

        The line segments are constructed from the provided points.

        Parameters
        ----------
        points :
            List of points / Matrix Nx2 matrix

        Returns
        -------
        Shape
            self

        """
        dimension = len(points.shape)
        if dimension == 1:
            points = points[np.newaxis, :]
        elif not dimension == 2:
            raise ValueError("Invalid input parameter")

        if not points.shape[1] == 2:
            raise ValueError("Invalid point format")

        if len(self.segments) > 0:
            points = np.vstack((self.segments[-1].point_end.m, points))
        elif points.shape[0] <= 1:
            raise ValueError("Insufficient number of points provided.")

        num_new_segments = len(points) - 1
        line_segments = []

        points = Q_(points, _DEFAULT_LEN_UNIT)
        for i in range(num_new_segments):
            line_segments += [
                LineSegment.construct_with_points(points[i], points[i + 1])
            ]
        self.add_segments(line_segments)

        return self

    def add_segments(self, segments: Union[segment_types, List[segment_types]]):
        """Add segments to the shape.

        Parameters
        ----------
        segments :
            Single segment or list of segments

        """
        segments = _to_list(segments)
        if self.num_segments > 0:
            self._check_segments_connected([self.segments[-1], segments[0]])
        self._check_segments_connected(segments)
        self._segments += segments

    def apply_transformation(self, transformation_matrix: np.ndarray):
        """Apply a transformation to the shape.

        Parameters
        ----------
        transformation_matrix :
            Transformation matrix

        """
        for i in range(self.num_segments):
            self._segments[i].apply_transformation(transformation_matrix)

    @UREG.wraps(None, (None, _DEFAULT_LEN_UNIT, _DEFAULT_LEN_UNIT), strict=True)
    def apply_reflection(
        self,
        reflection_normal: pint.Quantity,
        distance_to_origin: pint.Quantity = "0mm",
    ):
        """Apply a reflection at the given axis to the shape.

        Parameters
        ----------
        reflection_normal :
            Normal of the line of reflection
        distance_to_origin :
            Distance of the line of reflection to the origin (Default value = 0)

        """
        normal = reflection_normal
        if _vector_is_close(normal, np.array([0, 0], dtype=float)):
            raise ValueError("Normal has no length.")

        dot_product = np.dot(normal, normal)
        outer_product = np.outer(normal, normal)
        householder_matrix = np.identity(2) - 2 / dot_product * outer_product

        offset = Q_(
            normal / np.sqrt(dot_product) * distance_to_origin, _DEFAULT_LEN_UNIT
        )

        self.apply_translation(-offset)
        self.apply_transformation(householder_matrix)
        self.apply_translation(offset)

    @UREG.wraps(None, (None, _DEFAULT_LEN_UNIT, _DEFAULT_LEN_UNIT), strict=True)
    def apply_reflection_across_line(
        self, point_start: pint.Quantity, point_end: pint.Quantity
    ):
        """Apply a reflection across a line.

        Parameters
        ----------
        point_start :
            Line of reflection's start point
        point_end :
            Line of reflection's end point

        """
        if _vector_is_close(point_start, point_end):
            raise ValueError("Line start and end point are identical.")

        vector = point_end - point_start
        length_vector = np.linalg.norm(vector)

        line_distance_origin = (
            np.abs(point_start[1] * point_end[0] - point_start[0] * point_end[1])
            / length_vector
        )

        if tf.point_left_of_line([0, 0], point_start, point_end) > 0:
            normal = np.array([vector[1], -vector[0]], dtype=float)
        else:
            normal = np.array([-vector[1], vector[0]], dtype=float)

        self.apply_reflection(
            Q_(normal, _DEFAULT_LEN_UNIT), Q_(line_distance_origin, _DEFAULT_LEN_UNIT)
        )

    @UREG.check(None, "[length]")
    def apply_translation(self, vector: pint.Quantity):
        """Apply a translation to the shape.

        Parameters
        ----------
        vector :
            Translation vector

        """
        for i in range(self.num_segments):
            self._segments[i].apply_translation(vector)

    @UREG.wraps(_DEFAULT_LEN_UNIT, (None, _DEFAULT_LEN_UNIT), strict=True)
    def rasterize(self, raster_width: pint.Quantity) -> pint.Quantity:
        """Create an array of points that describe the shapes contour.

        The effective raster width may vary from the specified one,
        since the algorithm enforces constant distances between two
        raster points inside of each segment.

        Parameters
        ----------
        raster_width :
            The desired distance between two raster points

        Returns
        -------
        numpy.ndarray
            Array of contour points (2d)

        """
        if self.num_segments == 0:
            raise Exception("Can't rasterize empty shape.")
        if not raster_width > 0:
            raise ValueError("'raster_width' must be > 0")

        raster_width = Q_(raster_width, _DEFAULT_LEN_UNIT)
        raster_data = []
        for segment in self.segments:
            raster_data.append(segment.rasterize(raster_width).m[:, :-1])
        raster_data = np.hstack(raster_data)

        last_point = self.segments[-1].point_end.m[:, np.newaxis]
        if not _vector_is_close(last_point, self.segments[0].point_start.m):
            raster_data = np.hstack((raster_data, last_point))
        return raster_data

    @UREG.check(None, _DEFAULT_LEN_UNIT, _DEFAULT_LEN_UNIT)
    def reflect(
        self,
        reflection_normal: pint.Quantity,
        distance_to_origin: pint.Quantity = Q_("0mm"),
    ) -> Shape:
        """Get a reflected copy of the shape.

        Parameters
        ----------
        reflection_normal :
            Normal of the line of reflection
        distance_to_origin :
            Distance of the line of reflection to the
            origin (Default value = 0)

        Returns
        -------
        Shape

        """
        new_shape = copy.deepcopy(self)
        new_shape.apply_reflection(reflection_normal, distance_to_origin)
        return new_shape

    @UREG.check(None, "[length]", "[length]")
    def reflect_across_line(
        self, point_start: pint.Quantity, point_end: pint.Quantity
    ) -> Shape:
        """Get a reflected copy across a line.

        Parameters
        ----------
        point_start :
            Line of reflection's start point
        point_end :
            Line of reflection's end point

        Returns
        -------
        Shape

        """
        new_shape = copy.deepcopy(self)
        new_shape.apply_reflection_across_line(point_start, point_end)
        return new_shape

    def transform(self, matrix: np.ndarray) -> Shape:
        """Get a transformed copy of the shape.

        Parameters
        ----------
        matrix :
            Transformation matrix

        Returns
        -------
        Shape
            Transformed copy

        """
        new_shape = copy.deepcopy(self)
        new_shape.apply_transformation(matrix)
        return new_shape

    @UREG.check(None, "[length]")
    def translate(self, vector: pint.Quantity) -> Shape:
        """Get a translated copy of the shape.

        Parameters
        ----------
        vector :
            Translation vector

        Returns
        -------
        Shape
            Transformed copy

        """
        new_shape = copy.deepcopy(self)
        new_shape.apply_translation(vector)
        return new_shape


# Profile class ---------------------------------------------------------------


class Profile:
    """Defines a 2d profile."""

    def __init__(self, shapes: Union[Shape, List[Shape]], units: pint.Unit = None):
        """Construct profile class.

        Parameters
        ----------
        shapes :
            Instance or list of geo.Shape class(es)
        units :
            Associated units.

        Returns
        -------
        Profile

        """
        self._shapes = []
        self.attrs = {}
        if units is not None:
            self.attrs["units"] = units
        self.add_shapes(shapes)

    def __repr__(self):
        """Output representation of a Profile."""
        return f"Profile('shapes': {self._shapes!r})"

    def __str__(self):
        """Output simple string representation of a Profile for users."""
        repr_str = f"Profile with {len(self.shapes)} shape(s)\n"
        repr_str = repr_str + "\n\n".join(
            f"Shape {i}:\n{s!s}" for i, s in enumerate(self.shapes)
        )
        return repr_str

    def _ipython_display_(self):
        """Use __str__ output in notebooks."""
        print(str(self))

    @property
    def num_shapes(self) -> int:
        """Get the number of shapes of the profile.

        Returns
        -------
        int
            Number of shapes

        """
        return len(self._shapes)

    def add_shapes(self, shapes: Union[Shape, List[Shape]]):
        """Add shapes to the profile.

        Parameters
        ----------
        shapes :
            Instance or list of geo.Shape class(es)

        """
        if not isinstance(shapes, list):
            shapes = [shapes]

        if not all(isinstance(shape, Shape) for shape in shapes):
            raise TypeError("Only instances or lists of Shape objects are accepted.")

        self._shapes += shapes

    @UREG.wraps(None, (None, _DEFAULT_LEN_UNIT, None), strict=True)
    def rasterize(
        self, raster_width: pint.Quantity, stack: bool = True
    ) -> Union[pint.Quantity, List[pint.Quantity]]:
        """Rasterize the profile.

        Parameters
        ----------
        raster_width :
            Distance between points for rasterization.
        stack :
            hstack data into a single output array (default = True)

        Returns
        -------
        Union[pint.Quantity, List[pint.Quantity]]
            Raster data

        """
        raster_width = Q_(raster_width, _DEFAULT_LEN_UNIT)
        raster_data = []
        for shape in self._shapes:
            raster_data.append(shape.rasterize(raster_width).m)
        if stack:
            return Q_(np.hstack(raster_data), _DEFAULT_LEN_UNIT)
        return [Q_(item, _DEFAULT_LEN_UNIT) for item in raster_data]

    @UREG.check(None, None, "[length]", None, None, None, None, None, None, None)
    def plot(
        self,
        title: str = None,
        raster_width: pint.Quantity = Q_(0.5, _DEFAULT_LEN_UNIT),
        label: str = None,
        axis: str = "equal",
        axis_labels: List[str] = None,
        grid: bool = True,
        line_style: str = ".-",
        ax=None,
        color: str = "k",
    ):
        """Plot the profile.

        Parameters
        ----------
        title :
            Matplotlib plot title. (Default value = None)
        raster_width :
            Distance between Points to plot (Default value = 0.5)
        label :
            Matplotlib plot label. (Default value = None)
        axis :
            Matplotlib axis setting. (Default value = "equal")
        axis_labels :
            List of Matplotlib axis labels. (Default value = None)
        grid :
            Matplotlib grid setting. (Default value = True)
        line_style :
            Matplotlib line style. (Default value = ".-")
        ax :
            Axis to plot to. (Default value = None)
        color:
            Color of plot lines

        """
        raster_data = self.rasterize(raster_width, stack=False)
        raster_data = [q.m for q in raster_data]
        if ax is None:  # pragma: no cover
            from matplotlib.pyplot import subplots

            _, ax = subplots()
        ax.grid(grid)
        if not ax.name == "3d":
            ax.axis(axis)
        ax.set_title(title, loc="center", wrap=True)
        if axis_labels is not None:
            ax.set_xlabel(axis_labels[0])
            ax.set_ylabel(axis_labels[1])
        elif u := self.attrs.get("units"):
            ax.set_xlabel(f"y in {u}")
            ax.set_ylabel(f"z in {u}")

        if isinstance(color, str):  # single color
            color = [color] * len(raster_data)

        for segment, c in zip(raster_data, color):
            ax.plot(segment[0], segment[1], line_style, label=label, color=c)

    @property
    def shapes(self) -> List[Shape]:
        """Get the profiles shapes.

        Returns
        -------
        list
            Shapes

        """
        return self._shapes


# Trace segment classes -------------------------------------------------------


class LinearHorizontalTraceSegment:
    """Trace segment with a linear path and constant z-component."""

    @UREG.wraps(None, (None, _DEFAULT_LEN_UNIT), strict=True)
    def __init__(self, length: pint.Quantity):
        """Construct linear horizontal trace segment.

        Parameters
        ----------
        length :
            Length of the segment

        Returns
        -------
        LinearHorizontalTraceSegment

        """
        if length <= 0:
            raise ValueError("'length' must have a positive value.")
        self._length = float(length)

    def __repr__(self):
        """Output representation of a LinearHorizontalTraceSegment."""
        return f"LinearHorizontalTraceSegment('length': {self.length!r})"

    @property
    @UREG.wraps(_DEFAULT_LEN_UNIT, (None,), strict=True)
    def length(self):
        """Get the length of the segment.

        Returns
        -------
        pint.Quantity
            Length of the segment

        """
        return self._length

    def local_coordinate_system(
        self, relative_position: float
    ) -> tf.LocalCoordinateSystem:
        """Calculate a local coordinate system along the trace segment.

        Parameters
        ----------
        relative_position :
            Relative position on the trace [0 .. 1]

        Returns
        -------
        weldx.transformations.LocalCoordinateSystem
            Local coordinate system

        """
        relative_position = np.clip(relative_position, 0, 1)

        coordinates = np.array([1, 0, 0]) * relative_position * self._length
        return tf.LocalCoordinateSystem(coordinates=coordinates)


class RadialHorizontalTraceSegment:
    """Trace segment describing an arc with constant z-component."""

    @UREG.wraps(None, (None, _DEFAULT_LEN_UNIT, _DEFAULT_ANG_UNIT, None), strict=True)
    def __init__(
        self, radius: pint.Quantity, angle: pint.Quantity, clockwise: bool = False
    ):
        """Construct radial horizontal trace segment.

        Parameters
        ----------
        radius :
            Radius of the arc
        angle :
            Angle of the arc
        clockwise :
            If True, the rotation is clockwise. Otherwise it is counter-clockwise.

        Returns
        -------
        RadialHorizontalTraceSegment

        """
        if radius <= 0:
            raise ValueError("'radius' must have a positive value.")
        if angle <= 0:
            raise ValueError("'angle' must have a positive value.")
        self._radius = float(radius)
        self._angle = float(angle)
        self._length = self._arc_length(self._radius, self._angle)
        if clockwise:
            self._sign_winding = -1
        else:
            self._sign_winding = 1

    def __repr__(self):
        """Output representation of a RadialHorizontalTraceSegment."""
        return (
            f"RadialHorizontalTraceSegment('radius': {self._radius!r}, "
            f"'angle': {self._angle!r}, "
            f"'length': {self._length!r}, "
            f"'sign_winding': {self._sign_winding!r})"
        )

    @staticmethod
    def _arc_length(radius, angle) -> float:
        """Calculate the arc length.

        Parameters
        ----------
        radius :
            Radius
        angle :
            Angle (rad)

        Returns
        -------
        float
            Arc length

        """
        return angle * radius

    @property
    @UREG.wraps(_DEFAULT_ANG_UNIT, (None,), strict=True)
    def angle(self) -> pint.Quantity:
        """Get the angle of the segment.

        Returns
        -------
        pint.Quantity
            Angle of the segment (rad)

        """
        return self._angle

    @property
    @UREG.wraps(_DEFAULT_LEN_UNIT, (None,), strict=True)
    def length(self) -> pint.Quantity:
        """Get the length of the segment.

        Returns
        -------
        pint.Quantity
            Length of the segment

        """
        return self._length

    @property
    @UREG.wraps(_DEFAULT_LEN_UNIT, (None,), strict=True)
    def radius(self) -> pint.Quantity:
        """Get the radius of the segment.

        Returns
        -------
        pint.Quantity
            Radius of the segment

        """
        return self._radius

    @property
    def is_clockwise(self) -> bool:
        """Get True, if the segments winding is clockwise, False otherwise.

        Returns
        -------
        bool
            True or False

        """
        return self._sign_winding < 0

    def local_coordinate_system(
        self, relative_position: float
    ) -> tf.LocalCoordinateSystem:
        """Calculate a local coordinate system along the trace segment.

        Parameters
        ----------
        relative_position :
            Relative position on the trace [0 .. 1]

        Returns
        -------
        weldx.transformations.LocalCoordinateSystem
            Local coordinate system

        """
        relative_position = np.clip(relative_position, 0, 1)

        orientation = tf.WXRotation.from_euler(
            "z", self._angle * relative_position * self._sign_winding
        ).as_matrix()
        translation = np.array([0, -1, 0]) * self._radius * self._sign_winding

        coordinates = np.matmul(orientation, translation) - translation
        return tf.LocalCoordinateSystem(orientation, coordinates)


# Trace class -----------------------------------------------------------------

trace_segment_types = Union[LinearHorizontalTraceSegment, RadialHorizontalTraceSegment]


class Trace:
    """Defines a 3d trace."""

    def __init__(
        self,
        segments: Union[trace_segment_types, List[trace_segment_types]],
        coordinate_system: tf.LocalCoordinateSystem = None,
    ):
        """Construct trace.

        Parameters
        ----------
        segments :
            Single segment or list of segments
        coordinate_system :
            Coordinate system of the trace

        Returns
        -------
        Trace

        """
        if coordinate_system is None:
            coordinate_system = tf.LocalCoordinateSystem()

        if not isinstance(coordinate_system, tf.LocalCoordinateSystem):
            raise TypeError(
                "'coordinate_system' must be of type "
                "'transformations.LocalCoordinateSystem'"
            )

        self._segments = _to_list(segments)
        self._create_lookups(coordinate_system)

        if self.length.m <= 0:
            raise ValueError("Trace has no length.")

    def __repr__(self):
        """Output representation of a Trace."""
        return (
            f"Trace('segments': {self._segments!r}, "
            f"'coordinate_system_lookup': {self._coordinate_system_lookup!r}, "
            f"'total_length_lookup': {self._total_length_lookup!r}, "
            f"'segment_length_lookup': {self._segment_length_lookup!r})"
        )

    def _create_lookups(self, coordinate_system_start: tf.LocalCoordinateSystem):
        """Create lookup tables.

        Parameters
        ----------
        coordinate_system_start :
            Coordinate system at the start of
            the trace.

        """
        self._coordinate_system_lookup = [coordinate_system_start]
        self._total_length_lookup = [Q_("0mm")]
        self._segment_length_lookup = []

        segments = self._segments

        total_length = Q_(0.0, "mm")
        for i, segment in enumerate(segments):
            # Fill coordinate system lookup
            lcs_segment_end = segments[i].local_coordinate_system(1)
            lcs = lcs_segment_end + self._coordinate_system_lookup[i]
            self._coordinate_system_lookup += [lcs]

            # Fill length lookups
            segment_length = segment.length
            total_length += segment_length

            self._segment_length_lookup += [segment_length]
            self._total_length_lookup += [total_length.copy()]

    def _get_segment_index(self, position: float) -> int:
        """Get the segment index for a certain position.

        Parameters
        ----------
        position :
            Position

        Returns
        -------
        int
            Segment index

        """
        position = np.clip(position, 0, self.length)
        for i in range(len(self._total_length_lookup) - 2):
            if position <= self._total_length_lookup[i + 1]:
                return i
        return self.num_segments - 1

    @property
    def coordinate_system(self) -> tf.LocalCoordinateSystem:
        """Get the trace's coordinate system.

        Returns
        -------
        weldx.transformations.LocalCoordinateSystem
            Coordinate system of the trace

        """
        return self._coordinate_system_lookup[0]

    @property
    @UREG.wraps(_DEFAULT_LEN_UNIT, (None,), strict=True)
    def length(self) -> pint.Quantity:
        """Get the length of the trace.

        Returns
        -------
        pint.Quantity
            Length of the trace.

        """
        return self._total_length_lookup[-1].m

    @property
    def segments(self) -> List[trace_segment_types]:
        """Get the trace's segments.

        Returns
        -------
        list
            Segments of the trace

        """
        return self._segments

    @property
    def num_segments(self) -> int:
        """Get the number of segments.

        Returns
        -------
        int
            Number of segments

        """
        return len(self._segments)

    @UREG.check(None, "[length]")
    def local_coordinate_system(
        self, position: pint.Quantity
    ) -> tf.LocalCoordinateSystem:
        """Get the local coordinate system at a specific position on the trace.

        Parameters
        ----------
        position :
            Position

        Returns
        -------
        weldx.transformations.LocalCoordinateSystem
            Local coordinate system

        """
        idx = self._get_segment_index(position)

        total_length_start = self._total_length_lookup[idx]
        segment_length = self._segment_length_lookup[idx]
        weight = (position - total_length_start) / segment_length

        local_segment_cs = self.segments[idx].local_coordinate_system(weight)
        segment_start_cs = self._coordinate_system_lookup[idx]

        return local_segment_cs + segment_start_cs

    @UREG.wraps(_DEFAULT_LEN_UNIT, (None, _DEFAULT_LEN_UNIT), strict=True)
    def rasterize(self, raster_width: pint.Quantity) -> pint.Quantity:
        """Rasterize the trace.

        Parameters
        ----------
        raster_width :
           Distance between points for rasterization.

        Returns
        -------
        pint.Quantity
            Raster data


        """
        if not raster_width > 0:
            raise ValueError("'raster_width' must be > 0")

        raster_width = Q_(raster_width, _DEFAULT_LEN_UNIT)

        raster_width = np.clip(raster_width, Q_("0mm"), self.length)
        num_raster_segments = int(np.round(self.length / raster_width))
        raster_width_eff = self.length / num_raster_segments

        idx = 0
        raster_data = np.empty((3, 0))
        for i in range(num_raster_segments):
            location = i * raster_width_eff

            while not location <= self._total_length_lookup[idx + 1]:
                idx += 1

            segment_location = location - self._total_length_lookup[idx]
            weight = segment_location / self._segment_length_lookup[idx]
            local_segment_cs = self.segments[idx].local_coordinate_system(weight)
            segment_start_cs = self._coordinate_system_lookup[idx]

            local_cs = local_segment_cs + segment_start_cs

            data_point = local_cs.coordinates.data[:, np.newaxis]
            raster_data = np.hstack([raster_data, data_point])

        last_point = self._coordinate_system_lookup[-1].coordinates.data[:, np.newaxis]
        return np.hstack([raster_data.m, last_point])

    @UREG.check(None, "[length]", None, None, None)
    def plot(
        self,
        raster_width: pint.Quantity = "1mm",
        axes=None,
        fmt: str = None,
        axes_equal: bool = False,
    ):
        """Plot the trace.

        Parameters
        ----------
        raster_width : float, int
            The target distance between two raster points
        axes : matplotlib.axes.Axes
            The target `matplotlib.axes.Axes` object of the plot. If 'None' is passed, a
            new figure will be created
        fmt : str
            Format string that is passed to matplotlib.pyplot.plot.
        axes_equal : bool
            Set plot axes to equal scaling (Default = False).

        """
        data = self.rasterize(raster_width).to(_DEFAULT_LEN_UNIT)
        if fmt is None:
            fmt = "x-"
        if axes is None:
            from matplotlib.pyplot import subplots

            _, axes = subplots(subplot_kw=dict(projection="3d", proj_type="ortho"))
            axes.plot(data[0].m, data[1].m, data[2].m, fmt)
            axes.set_xlabel(f"x / {_DEFAULT_LEN_UNIT}")
            axes.set_ylabel(f"y / {_DEFAULT_LEN_UNIT}")
            axes.set_zlabel(f"z / {_DEFAULT_LEN_UNIT}")
            if axes_equal:
                import weldx.visualization as vs

                vs.axes_equal(axes)
        else:
            axes.plot(data[0].m, data[1].m, data[2].m, fmt)


# Linear profile interpolation class ------------------------------------------


def linear_profile_interpolation_sbs(
    profile_a: Profile, profile_b: Profile, weight: float
):
    """Interpolate 2 profiles linearly, segment by segment.

    Parameters
    ----------
    profile_a : Profile
        First profile
    profile_b : Profile
        Second profile
    weight : float
        Weighting factor [0 .. 1]. If 0, the profile is identical
        to 'a' and if 1, it is identical to b.

    Returns
    -------
    Profile
        Interpolated profile

    """
    weight = np.clip(weight, 0, 1)
    if not len(profile_a.shapes) == len(profile_b.shapes):
        raise ValueError("Number of profile shapes do not match.")

    shapes_c = []
    for i in range(profile_a.num_shapes):
        shapes_c += [
            Shape.linear_interpolation(profile_a.shapes[i], profile_b.shapes[i], weight)
        ]

    return Profile(shapes_c)


# Varying profile class -------------------------------------------------------


class VariableProfile:
    """Class to define a profile of variable shape."""

    @UREG.wraps(None, (None, None, _DEFAULT_LEN_UNIT, None), strict=True)
    def __init__(
        self, profiles: List[Profile], locations: pint.Quantity, interpolation_schemes
    ):
        """Construct variable profile.

        Parameters
        ----------
        profiles :
            List of profiles.
        locations :
            Ascending list of profile locations. Since the first location needs to be 0,
             it can be omitted.
        interpolation_schemes :
            List of interpolation schemes to define the interpolation between
            two locations.

        Returns
        -------
        VariableProfile

        """
        locations = (
            locations.tolist() if isinstance(locations, np.ndarray) else [locations]
        )
        interpolation_schemes = _to_list(interpolation_schemes)

        if not locations[0] == 0:
            locations = [0] + locations

        if not len(profiles) == len(locations):
            raise ValueError("Invalid list of locations. See function description.")

        if not len(interpolation_schemes) == len(profiles) - 1:
            raise ValueError(
                "Number of interpolations must be 1 less than number of " "profiles."
            )

        for i in range(len(profiles) - 1):
            if locations[i] >= locations[i + 1]:
                raise ValueError("Locations need to be sorted in ascending order.")

        self._profiles = profiles
        self._locations = locations
        self._interpolation_schemes = interpolation_schemes

    def __repr__(self):
        """Output representation of a VariableProfile."""
        return (
            f"VariableProfile('profiles': {self._profiles!r}, "
            f"'locations' {self._locations!r}, "
            f"'interpolation_schemes' {self._interpolation_schemes!r})"
        )

    def _segment_index(self, location: float):
        """Get the index of the segment at a certain location.

        Parameters
        ----------
        location :
            Location

        Returns
        -------
        int
            Segment index

        """
        idx = 0
        while location > self._locations[idx + 1]:
            idx += 1
        return idx

    @property
    def interpolation_schemes(self) -> List:
        """Get the interpolation schemes.

        Returns
        -------
        list
            List of interpolation schemes

        """
        return self._interpolation_schemes

    @property
    @UREG.wraps(_DEFAULT_LEN_UNIT, (None,), strict=True)
    def locations(self) -> pint.Quantity:
        """Get the locations.

        Returns
        -------
        pint.Quantity
            List of locations

        """
        return self._locations

    @property
    @UREG.wraps(_DEFAULT_LEN_UNIT, (None,), strict=True)
    def max_location(self) -> pint.Quantity:
        """Get the maximum location.

        Returns
        -------
        pint.Quantity
            Maximum location

        """
        return self._locations[-1]

    @property
    def num_interpolation_schemes(self) -> int:
        """Get the number of interpolation schemes.

        Returns
        -------
        int
            Number of interpolation schemes

        """
        return len(self._interpolation_schemes)

    @property
    def num_locations(self) -> int:
        """Get the number of profile locations.

        Returns
        -------
        int
            Number of profile locations

        """
        return len(self._locations)

    @property
    def num_profiles(self) -> int:
        """Get the number of profiles.

        Returns
        -------
        int
            Number of profiles

        """
        return len(self._profiles)

    @property
    def profiles(self) -> List[Profile]:
        """Get the profiles.

        Returns
        -------
        list
            List of profiles

        """
        return self._profiles

    @UREG.wraps(None, (None, _DEFAULT_LEN_UNIT), strict=True)
    def local_profile(self, location: pint.Quantity) -> Profile:
        """Get the profile at the specified location.

        Parameters
        ----------
        location :
            Location

        Returns
        -------
        Profile
            Local profile.

        """
        location = np.clip(location, 0, self.max_location.m)

        idx = self._segment_index(location)
        segment_length = self._locations[idx + 1] - self._locations[idx]
        weight = (location - self._locations[idx]) / segment_length
        return self._interpolation_schemes[idx](
            self._profiles[idx], self._profiles[idx + 1], weight
        )


#  Geometry class -------------------------------------------------------------


class Geometry:
    """Defines a 3 dimensional geometry by extrusion of a 2 dimensional profile.

    The path of the extrusion can be chosen arbitrarily. It is also possible to vary
    the profile along the extrusion path.

    """

    def __init__(
        self,
        profile: Union[Profile, VariableProfile, iso.IsoBaseGroove],
        trace_or_length: Union[Trace, pint.Quantity],
        width: pint.Quantity = Q_(10, "mm"),
    ):
        """Construct a geometry.

        Parameters
        ----------
        profile :
            Profile that is used as cross section along the specified trace
        trace_or_length :
            The path that is used to extrude the given profile or a quantity that
            specifies the length of a linear, horizontal extrusion
        width :
            If a groove type is passed as ``profile`` this parameter determines the
            width of the generated cross-section. For all other types it has no effect.

        Returns
        -------
        Geometry :
            A Geometry class instance

        """
        from weldx.welding.groove.iso_9692_1 import IsoBaseGroove

        if isinstance(profile, IsoBaseGroove):
            profile = profile.to_profile(width)
        if not isinstance(trace_or_length, Trace):
            trace_or_length = Trace(LinearHorizontalTraceSegment(Q_(trace_or_length)))
        self._check_inputs(profile, trace_or_length)
        self._profile = profile
        self._trace = trace_or_length

    def __repr__(self):
        """Output representation of a Geometry class."""
        return f"Geometry('profile': {self._profile!r}, 'trace': {self._trace!r})"

    @staticmethod
    def _check_inputs(profile: Union[Profile, VariableProfile], trace: Trace):
        """Check the inputs to the constructor.

        Parameters
        ----------
        profile :
            Constant or variable profile.
        trace :
            Trace

        """
        if not isinstance(profile, (Profile, VariableProfile)):
            raise TypeError("'profile' must be a 'Profile' or 'VariableProfile' class")

        if not isinstance(trace, Trace):
            raise TypeError("'trace' must be a 'Trace' class")

    def _get_local_profile_data(self, trace_location: float, raster_width: float):
        """Get a rasterized profile at a certain location on the trace.

        Parameters
        ----------
        trace_location :
            Location on the trace
        raster_width :
            Raster width

        """
        relative_location = trace_location / self._trace.length
        profile_location = relative_location * self._profile.max_location
        profile = self._profile.local_profile(profile_location)
        return self._profile_raster_data_3d(profile, raster_width)

    def _rasterize_trace(self, raster_width: float) -> np.ndarray:
        """Rasterize the trace.

        Parameters
        ----------
        raster_width :
            Raster width

        Returns
        -------
        numpy.ndarray
            Raster data

        """
        if not raster_width > 0:
            raise ValueError("'raster_width' must be > 0")
        raster_width = np.clip(raster_width, None, self._trace.length)

        num_raster_segments = int(np.round(self._trace.length / raster_width))
        raster_width_eff = self._trace.length / num_raster_segments
        locations = np.arange(
            0, (self._trace.length - raster_width_eff / 2).m, raster_width_eff.m
        )
        return Q_(np.hstack([locations, self._trace.length.m]), _DEFAULT_LEN_UNIT)

    def _get_transformed_profile_data(self, profile_raster_data, location):
        """Transform a profiles data to a specified location on the trace.

        Parameters
        ----------
        profile_raster_data :
            Rasterized profile
        location :
            Location on the trace

        Returns
        -------
        numpy.ndarray
            Transformed profile data

        """
        local_cs = self._trace.local_coordinate_system(location)
        local_data = np.matmul(local_cs.orientation.data, profile_raster_data)
        coords = local_cs.coordinates.data[:, np.newaxis]
        if isinstance(coords, pint.Quantity):
            coords = coords.m
        return local_data + coords

    @staticmethod
    def _profile_raster_data_3d(profile: Profile, raster_width, stack: bool = True):
        """Get the rasterized profile in 3d.

        The profile is located in the x-z-plane.

        Parameters
        ----------
        profile :
            Profile
        raster_width :
            Raster width
        stack :
            hstack data into a single output array, else return list (default = True)

        Returns
        -------
        numpy.ndarray
            Rasterized profile in 3d

        """
        profile_data = profile.rasterize(raster_width, stack=stack)
        if stack:
            return np.insert(profile_data, 0, 0, axis=0)
        return [np.insert(p, 0, 0, axis=0) for p in profile_data]

    def _rasterize_constant_profile(
        self, profile_raster_width, trace_raster_width, stack: bool = True
    ):
        """Rasterize the geometry with a constant profile.

        Parameters
        ----------
        profile_raster_width :
            Raster width of the profiles
        trace_raster_width :
            Distance between two profiles
        stack :
            hstack data into a single output array (default = True)

        Returns
        -------
        numpy.ndarray
            Raster data

        """
        locations = self._rasterize_trace(Q_(trace_raster_width, _DEFAULT_LEN_UNIT))

        if stack:  # old behavior for 3d point cloud
            profile_data = self._profile_raster_data_3d(
                self._profile, profile_raster_width, stack=True
            )
            raster_data = np.empty([3, 0])
            for _, location in enumerate(locations):
                local_data = self._get_transformed_profile_data(
                    profile_data.m, location
                )
                raster_data = np.hstack([raster_data, local_data])
            raster_data = Q_(raster_data, _DEFAULT_LEN_UNIT)
        else:
            profile_data = self._profile_raster_data_3d(
                self._profile, profile_raster_width, stack=False
            )

            raster_data = []
            for data in profile_data:
                raster_data.append(
                    np.stack(
                        [
                            self._get_transformed_profile_data(data.m, location)
                            for location in locations
                        ],
                        0,
                    )
                )

        return raster_data

    def _rasterize_variable_profile(self, profile_raster_width, trace_raster_width):
        """Rasterize the geometry with a variable profile.

        Parameters
        ----------
        profile_raster_width :
            Raster width of the profiles
        trace_raster_width :
            Distance between two profiles

        Returns
        -------
        numpy.ndarray
            Raster data

        """
        locations = self._rasterize_trace(Q_(trace_raster_width, _DEFAULT_LEN_UNIT))
        raster_data = np.empty([3, 0])
        for _, location in enumerate(locations):
            profile_data = self._get_local_profile_data(location, profile_raster_width)

            local_data = self._get_transformed_profile_data(profile_data.m, location)
            raster_data = np.hstack([raster_data, local_data])

        return raster_data

    @property
    def profile(self) -> Union[Profile, VariableProfile]:
        """Get the geometry's profile.

        Returns
        -------
        Profile

        """
        return self._profile

    @property
    def trace(self) -> Trace:
        """Get the geometry's trace.

        Returns
        -------
        Trace

        """
        return self._trace

    @UREG.wraps(
        None,
        (None, _DEFAULT_LEN_UNIT, _DEFAULT_LEN_UNIT, None),
        strict=True,
    )
    def rasterize(
        self,
        profile_raster_width: pint.Quantity,
        trace_raster_width: pint.Quantity,
        stack: bool = True,
    ) -> pint.Quantity:
        """Rasterize the geometry.

        Parameters
        ----------
        profile_raster_width :
            Raster width of the profiles
        trace_raster_width :
            Distance between two profiles
        stack :
            hstack data into a single output array (default = True)

        Returns
        -------
        numpy.ndarray
            Raster data

        """
        profile_raster_width = Q_(profile_raster_width, _DEFAULT_LEN_UNIT)
        if isinstance(self._profile, Profile):
            return self._rasterize_constant_profile(
                profile_raster_width, trace_raster_width, stack=stack
            )
        return Q_(
            self._rasterize_variable_profile(profile_raster_width, trace_raster_width),
            _DEFAULT_LEN_UNIT,
        )

    @UREG.check(None, "[length]", "[length]", None, None, None, None, None, None)
    def plot(
        self,
        profile_raster_width: pint.Quantity = Q_("1mm"),
        trace_raster_width: pint.Quantity = Q_("50mm"),
        axes: matplotlib.axes.Axes = None,
        color: Union[int, Tuple[int, int, int], Tuple[float, float, float]] = None,
        label: str = None,
        limits: vs_types.types_limits = None,
        show_wireframe: bool = True,
        backend: str = "mpl",
    ) -> matplotlib.axes.Axes:
        """Plot the geometry.

        Parameters
        ----------
        profile_raster_width : pint.Quantity
            Target distance between the individual points of a profile
        trace_raster_width : pint.Quantity
            Target distance between the individual profiles on the trace
        axes : matplotlib.axes.Axes
            The target `matplotlib.axes.Axes` object of the plot. If 'None' is passed, a
            new figure will be created
        color : Union[int, Tuple[int, int, int], Tuple[float, float, float]]
            A 24 bit integer, a triplet of integers with a value range of 0-255
            or a triplet of floats with a value range of 0.0-1.0 that represent an RGB
            color
        label : str
            Label of the plotted geometry
        limits :
            Either a single tuple of two float values that specifies the minimum and
            maximum value of all 3 axis or a list containing 3 tuples to specify the
            limits of each axis individually. If `None` is passed, the limits will be
            set automatically.
        show_wireframe : bool
            (matplotlib only) If `True`, the mesh is plotted as wireframe. Otherwise
            only the raster points are visualized. Currently, the wireframe can't be
            visualized if a `weldx.geometry.VariableProfile` is used.
        backend :
            Select the rendering backend of the plot. The options are:

            - ``k3d`` to get an interactive plot using `k3d <https://k3d-jupyter.org/>`_
            - ``mpl`` for static plots using `matplotlib <https://matplotlib.org/>`_

            Note that k3d only works inside jupyter notebooks

        Returns
        -------
        matplotlib.axes.Axes :
            The utilized matplotlib axes, if matplotlib was used as rendering backend

        """
        data = self.spatial_data(profile_raster_width, trace_raster_width)
        return data.plot(
            axes=axes,
            color=color,
            label=label,
            limits=limits,
            show_wireframe=show_wireframe,
            backend=backend,
        )

    @UREG.check(None, "[length]", "[length]", None)
    def spatial_data(
        self,
        profile_raster_width: pint.Quantity,
        trace_raster_width: pint.Quantity,
        closed_mesh: bool = True,
    ) -> SpatialData:
        """Rasterize the geometry and get it as `SpatialData` instance.

        If no `weldx.geometry.VariableProfile` is used, a triangulation
        is added automatically.

        Parameters
        ----------
        profile_raster_width :
            Target distance between the individual points of a profile
        trace_raster_width :
            Target distance between the individual profiles on the trace
        closed_mesh :
            If `True`, the surface of the 3d geometry will be closed

        Returns
        -------
        SpatialData :
            The rasterized geometry data

        """
        # Todo: This branch is a "dirty" fix for the fact that there is no "stackable"
        #       rasterization for geometries with a VariableProfile. The stacked
        #       rasterization is needed for the triangulation performed in
        #       `from_geometry_raster`.
        if isinstance(self._profile, VariableProfile):
            rasterization = self.rasterize(profile_raster_width, trace_raster_width)
            return SpatialData(np.swapaxes(rasterization.m, 0, 1))

        rasterization = self.rasterize(
            profile_raster_width, trace_raster_width, stack=False
        )
        return SpatialData.from_geometry_raster(rasterization, closed_mesh)

    @UREG.wraps(None, (None, None, _DEFAULT_LEN_UNIT, _DEFAULT_LEN_UNIT), strict=True)
    def to_file(
        self,
        file_name: str,
        profile_raster_width: pint.Quantity,
        trace_raster_width: pint.Quantity,
    ):
        """Write the ``Geometry`` data into a CAD file.

        The geometry is rasterized and triangulated before the export. All file formats
        supported by ``meshio`` that are based on points and triangles can be used
        (For example ``.stl`` or ``.ply``). Just add the corresponding extension to the
        file name. For further information about supported file formats refer to the
        [``meshio`` documentation](https://pypi.org/project/meshio/).

        Parameters
        ----------
        file_name :
            Name of the file. Add the extension of the desired file format.
        profile_raster_width :
            Target distance between the individual points of a profile
        trace_raster_width :
            Target distance between the individual profiles on the trace

        """
        if isinstance(self._profile, VariableProfile):
            raise NotImplementedError

        raster_data = self._rasterize_constant_profile(
            profile_raster_width=profile_raster_width,
            trace_raster_width=trace_raster_width,
            stack=False,
        )

        SpatialData.from_geometry_raster(raster_data, True).to_file(file_name)


# SpatialData --------------------------------------------------------------------------


@ut.dataclass_nested_eq
@dataclass
class SpatialData:
    """Represent 3D point cloud data with optional triangulation."""

    coordinates: DataArray
    """3D array of point data.
        The expected array dimension order is [("time"), "n", "c"]."""
    triangles: np.ndarray = None
    """3D Array of triangulation connectivity.
        Shape should be [time * n, 3]."""
    attributes: Dict[str, np.ndarray] = None
    """optional dictionary with additional attributes to store alongside data."""
    time: InitVar[Time] = None
    """Time axis if data is time dependent."""

    def __post_init__(self, time):
        """Convert and check input values."""
        if not isinstance(self.coordinates, DataArray):
            self.coordinates = ut.xr_3d_vector(
                data=np.array(self.coordinates),
                time=time,
                add_dims=["n"],
            )

        # make sure we have correct dimension order
        self.coordinates = self.coordinates.transpose(..., "n", "c")

        if self.triangles is not None:
            if not isinstance(self.triangles, np.ndarray):
                self.triangles = np.array(self.triangles, dtype="uint")
            if not self.triangles.shape[-1] == 3:
                raise ValueError(
                    "SpatialData triangulation vertices must connect 3 points."
                )
            if not self.triangles.ndim == 2:
                raise ValueError("SpatialData triangulation must be a 2d array")

    @staticmethod
    def from_file(file_name: Union[str, Path]) -> SpatialData:
        """Create an instance from a file.

        Parameters
        ----------
        file_name :
            Name of the source file.

        Returns
        -------
        SpatialData:
            New `SpatialData` instance

        """
        mesh = meshio.read(file_name)
        triangles = mesh.cells_dict.get("triangle")

        return SpatialData(mesh.points, triangles)

    @staticmethod
    def _shape_raster_points(shape_raster_data: np.ndarray) -> List[List[int]]:
        """Extract all points from a shapes raster data."""
        return shape_raster_data.reshape(
            (shape_raster_data.shape[0] * shape_raster_data.shape[1], 3)
        ).tolist()

    @staticmethod
    def _shape_profile_triangles(
        num_profiles: int, num_profile_points: int, offset: int, cw_ordering: bool
    ) -> List[List[int]]:
        """Create the profile main surface triangles for ``_shape_triangles``."""
        tri_base = []
        for i in range(num_profile_points):
            idx_0 = i
            idx_1 = (i + 1) % num_profile_points
            idx_2 = idx_0 + num_profile_points
            idx_3 = idx_1 + num_profile_points

            if cw_ordering:
                tri_base += [[idx_0, idx_2, idx_1], [idx_1, idx_2, idx_3]]
            else:
                tri_base += [[idx_0, idx_1, idx_2], [idx_1, idx_3, idx_2]]
        tri_base = np.array(tri_base, dtype=int)

        triangles = np.array(
            [
                tri_base + i * num_profile_points + offset
                for i in range(num_profiles - 1)
            ],
            dtype=int,
        )

        return triangles.reshape((tri_base.shape[0] * (num_profiles - 1), 3)).tolist()

    @staticmethod
    def _shape_front_back_triangles(
        num_profiles: int, num_profile_points: int, offset: int, cw_ordering: bool
    ) -> List[List[int]]:
        """Create the front and back surface triangles for ``_shape_triangles``."""
        tri_cw = []
        tri_ccw = []
        i_0 = 0
        i_1 = 0

        while i_0 + i_1 < num_profile_points - 2:
            p_0 = i_0 + offset
            if i_1 == i_0:
                p_1 = p_0 + 1
                p_2 = num_profile_points + offset - i_1 - 1
                i_0 += 1
            else:
                p_1 = num_profile_points + offset - i_1 - 2
                p_2 = p_1 + 1
                i_1 += 1
            tri_cw += [[p_0, p_1, p_2]]
            tri_ccw += [[p_0, p_2, p_1]]

        if cw_ordering:
            front = tri_cw
            back = tri_ccw
        else:
            front = tri_ccw
            back = tri_cw
        return [
            *front,
            *(np.array(back, int) + (num_profiles - 1) * num_profile_points).tolist(),
        ]

    @classmethod
    def _shape_triangles(
        cls, shape_raster_data: np.ndarray, offset: int, closed_mesh: bool
    ) -> List[List[int]]:
        """Get the triangles of a shape from its raster data.

        The triangle data are just indices referring to a list of points.

        Parameters
        ----------
        shape_raster_data :
            Raster data of the shape
        offset :
            An offset that will be added to all indices.
        closed_mesh :
            If `True`, the front and back faces of the geometry will also be
            triangulated.

        Returns
        -------
        List[List[int]] :
            The list of triangles

        """
        n_prf = shape_raster_data.shape[0]
        n_prf_pts = shape_raster_data.shape[1]
        cw_ord = has_cw_ordering(shape_raster_data[0])
        if not closed_mesh:
            return cls._shape_profile_triangles(n_prf, n_prf_pts, offset, cw_ord)
        return [
            *cls._shape_profile_triangles(n_prf, n_prf_pts, offset, cw_ord),
            *cls._shape_front_back_triangles(n_prf, n_prf_pts, offset, cw_ord),
        ]

    @classmethod
    def from_geometry_raster(
        cls, geometry_raster: np.ndarray, closed_mesh: bool = True
    ) -> SpatialData:
        """Triangulate rasterized Geometry Profile.

        Parameters
        ----------
        geometry_raster : numpy.ndarray
            A single unstacked geometry rasterization.
        closed_mesh :
            If `True`, the surface of the 3d geometry will be closed

        Returns
        -------
        SpatialData:
            New `SpatialData` instance

        """
        points = []
        triangles = []
        for shape_data in geometry_raster:
            shape_data = shape_data.swapaxes(1, 2)
            triangles += cls._shape_triangles(shape_data, len(points), closed_mesh)
            points += cls._shape_raster_points(shape_data)

        return SpatialData(points, triangles)

    def limits(self) -> np.ndarray:
        """Get the xyz limits of the coordinates.

        Array format:
        [[x0,y0,z0],
        [x1,y1,z1]]
        """
        dims = self.additional_dims
        mins = self.coordinates.min(dim=dims)
        maxs = self.coordinates.max(dim=dims)

        return np.vstack([mins, maxs])

    def plot(
        self,
        axes: matplotlib.axes.Axes = None,
        color: Union[int, Tuple[int, int, int], Tuple[float, float, float]] = None,
        label: str = None,
        show_wireframe: bool = True,
        limits: vs_types.types_limits = None,
        backend: str = "mpl",
    ) -> matplotlib.axes.Axes:
        """Plot the spatial data.

        Parameters
        ----------
        axes : matplotlib.axes.Axes
            The target `matplotlib.axes.Axes` object of the plot. If 'None' is passed, a
            new figure will be created
        color : Union[int, Tuple[int, int, int], Tuple[float, float, float]]
            A 24 bit integer, a triplet of integers with a value range of 0-255
            or a triplet of floats with a value range of 0.0-1.0 that represent an RGB
            color
        label : str
            Label of the plotted geometry
        limits :
            Either a single tuple of two float values that specifies the minimum and
            maximum value of all 3 axis or a list containing 3 tuples to specify the
            limits of each axis individually. If `None` is passed, the limits will be
            set automatically.
        show_wireframe : bool
            (Matplotlib only) If `True`, the mesh is plotted as wireframe. Otherwise
            only the raster points are visualized.
        backend :
            Select the rendering backend of the plot. The options are:

            - ``k3d`` to get an interactive plot using `k3d <https://k3d-jupyter.org/>`_
            - ``mpl`` for static plots using `matplotlib <https://matplotlib.org/>`_

            Note that k3d only works inside jupyter notebooks

        Returns
        -------
        matplotlib.axes.Axes :
            The utilized matplotlib axes, if matplotlib was used as rendering backend

        """
        if backend not in ("mpl", "k3d"):
            raise ValueError(
                f"backend has to be one of ('mpl', 'k3d'), but was {backend}"
            )

        import weldx.visualization as vs

        if backend == "k3d":
            import k3d

            limits = tuple(self.limits().flatten())
            plot = k3d.plot(grid=limits)
            vs.SpatialDataVisualizer(
                self, name=None, reference_system=None, color=color, plot=plot
            )
            return plot

        return vs.plot_spatial_data_matplotlib(
            data=self,
            axes=axes,
            color=color,
            label=label,
            limits=limits,
            show_wireframe=show_wireframe,
        )

    def to_file(self, file_name: Union[str, Path]):
        """Write spatial data into a file.

        The extension prescribes the output format.

        Parameters
        ----------
        file_name :
            Name of the file

        """
        mesh = meshio.Mesh(
            points=self.coordinates.data, cells={"triangle": self.triangles}
        )
        mesh.write(file_name)

    @property
    def is_time_dependent(self) -> bool:
        """Return `True` if the coordinates are time dependent."""
        return "time" in self.coordinates.dims

    @property
    def additional_dims(self) -> List[str]:
        """Return the list of array dimension besides the required 'c' dimension."""
        return [str(d) for d in self.coordinates.dims if d != "c"]
