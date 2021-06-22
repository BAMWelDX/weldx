"""Provides classes to define lines and surfaces."""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Tuple, Union

import meshio
import numpy as np
import pint
from xarray import DataArray

import weldx.transformations as tf
import weldx.util as ut
from weldx.constants import WELDX_UNIT_REGISTRY as UREG

_DEFAULT_LEN_UNIT = UREG.millimeters
_DEFAULT_ANG_UNIT = UREG.rad

# only import heavy-weight packages on type checking
if TYPE_CHECKING:  # pragma: no cover
    import matplotlib.axes

# LineSegment -----------------------------------------------------------------


class LineSegment:
    """Line segment."""

    @UREG.wraps(None, (None, _DEFAULT_LEN_UNIT), strict=False)
    def __init__(self, points):
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
        points = ut.to_float_array(points)
        if not len(points.shape) == 2:
            raise ValueError("'points' must be a 2d array/matrix.")
        if not (points.shape[0] == 2 and points.shape[1] == 2):
            raise ValueError("'points' is not a 2x2 matrix.")

        self._points = points
        self._calculate_length()

    def __repr__(self):
        """Output representation of a LineSegment."""
        return f"LineSegment('points'={self._points!r}, 'length'={self._length!r})"

    def __str__(self):
        """Output simple string representation of a LineSegment."""
        p1 = np.array2string(self.points[:, 0], precision=2, separator=",")
        p2 = np.array2string(self.points[:, 1], precision=2, separator=",")
        return f"Line: {p1} -> {p2}"

    def _calculate_length(self):
        """Calculate the segment length from its points."""
        self._length = np.linalg.norm(self._points[:, 1] - self._points[:, 0])
        if math.isclose(self._length, 0):
            raise ValueError("Segment length is 0.")

    @classmethod
    @UREG.wraps(None, (None, _DEFAULT_LEN_UNIT, _DEFAULT_LEN_UNIT), strict=False)
    def construct_with_points(cls, point_start, point_end) -> "LineSegment":
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
        points = np.transpose(np.array([point_start, point_end], dtype=float))
        return cls(points)

    @classmethod
    @UREG.wraps(None, (None, None, None, ""), strict=False)
    def linear_interpolation(cls, segment_a, segment_b, weight):
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
        points = (1 - weight) * segment_a.points + weight * segment_b.points
        return cls(points)

    @property
    def length(self):
        """Get the segment length.

        Returns
        -------
        float
            Segment length

        """
        return self._length

    @property
    def point_end(self):
        """Get the end point of the segment.

        Returns
        -------
        numpy.ndarray
            End point

        """
        return self._points[:, 1]

    @property
    def point_start(self):
        """Get the starting point of the segment.

        Returns
        -------
        numpy.ndarray
            Starting point

        """
        return self._points[:, 0]

    @property
    def points(self):
        """Get the segments points in form of a 2x2 matrix.

        The first column represents the starting point and the second one the end point.

        Returns
        -------
        numpy.ndarray
            2x2 matrix containing the segments points

        """
        return self._points

    @UREG.wraps(None, (None, ""), strict=False)
    def apply_transformation(self, matrix):
        """Apply a transformation matrix to the segment.

        Parameters
        ----------
        matrix :
            Transformation matrix

        """
        self._points = np.matmul(matrix, self._points)
        self._calculate_length()

    @UREG.wraps(None, (None, _DEFAULT_LEN_UNIT), strict=False)
    def apply_translation(self, vector):
        """Apply a translation to the segment.

        Parameters
        ----------
        vector :
            Translation vector

        """
        self._points += np.ndarray((2, 1), float, np.array(vector, float))

    @UREG.wraps(None, (None, _DEFAULT_LEN_UNIT), strict=False)
    def rasterize(self, raster_width) -> np.ndarray:
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
        numpy.ndarray
            Array of contour points

        """
        if not raster_width > 0:
            raise ValueError("'raster_width' must be > 0")
        raster_width = np.min([raster_width, self.length])

        num_raster_segments = np.round(self.length / raster_width)

        # normalized effective raster width
        nerw = 1.0 / num_raster_segments

        multiplier = np.arange(0, 1 + 0.5 * nerw, nerw)
        weight_matrix = np.array([1 - multiplier, multiplier])

        return np.matmul(self._points, weight_matrix)

    @UREG.wraps(None, (None, ""), strict=False)
    def transform(self, matrix):
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

    @UREG.wraps(None, (None, _DEFAULT_LEN_UNIT), strict=False)
    def translate(self, vector):
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

    @UREG.wraps(None, (None, _DEFAULT_LEN_UNIT, None), strict=False)
    def __init__(self, points, arc_winding_ccw=True):
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
        points = ut.to_float_array(points)
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
        point_start = self.point_start
        point_end = self.point_end
        point_center = self.point_center

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
        point_start = self.point_start
        point_end = self.point_end
        point_center = self.point_center

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
        strict=False,
    )
    def construct_with_points(
        cls, point_start, point_end, point_center, arc_winding_ccw=True
    ):
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
        return cls(points, arc_winding_ccw)

    @classmethod
    @UREG.wraps(
        None,
        (None, _DEFAULT_LEN_UNIT, _DEFAULT_LEN_UNIT, _DEFAULT_LEN_UNIT, None, None),
        strict=False,
    )
    def construct_with_radius(
        cls,
        point_start,
        point_end,
        radius,
        center_left_of_line=True,
        arc_winding_ccw=True,
    ):
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
        point_start = ut.to_float_array(point_start)
        point_end = ut.to_float_array(point_end)

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
            point_start, point_end, point_center, arc_winding_ccw
        )

    @classmethod
    @UREG.wraps(None, (None, None, None, ""), strict=False)
    def linear_interpolation(cls, segment_a, segment_b, weight):
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
    def arc_angle(self):
        """Get the arc angle.

        Returns
        -------
        float
            Arc angle

        """
        return self._arc_angle

    @property
    def arc_length(self):
        """Get the arc length.

        Returns
        -------
        float
            Arc length

        """
        return self._arc_length

    @property
    def arc_winding_ccw(self):
        """Get True if the winding order is counter-clockwise. False if clockwise.

        Returns
        -------
        bool
            True or False

        """
        return self._sign_arc_winding > 0

    @property
    def point_center(self):
        """Get the center point of the segment.

        Returns
        -------
        numpy.ndarray
            Center point

        """
        return self._points[:, 2]

    @property
    def point_end(self):
        """Get the end point of the segment.

        Returns
        -------
        numpy.ndarray
            End point

        """
        return self._points[:, 1]

    @property
    def point_start(self):
        """Get the starting point of the segment.

        Returns
        -------
        numpy.ndarray
            Starting point

        """
        return self._points[:, 0]

    @property
    def points(self):
        """Get the segments points in form of a 2x3 matrix.

        The first column represents the starting point, the second one the
        end and the third one the center.

        Returns
        -------
        numpy.ndarray
            2x3 matrix containing the segments points

        """
        return self._points

    @property
    def radius(self):
        """Get the radius.

        Returns
        -------
        float
            Radius

        """
        return self._radius

    @UREG.wraps(None, (None, ""), strict=False)
    def apply_transformation(self, matrix):
        """Apply a transformation to the segment.

        Parameters
        ----------
        matrix :
            Transformation matrix

        """
        self._points = np.matmul(matrix, self._points)
        self._sign_arc_winding *= tf.reflection_sign(matrix)
        self._calculate_arc_parameters()

    @UREG.wraps(None, (None, _DEFAULT_LEN_UNIT), strict=False)
    def apply_translation(self, vector):
        """Apply a translation to the segment.

        Parameters
        ----------
        vector :
            Translation vector

        """
        self._points += np.ndarray((2, 1), float, np.array(vector, float))

    @UREG.wraps(None, (None, _DEFAULT_LEN_UNIT), strict=False)
    def rasterize(self, raster_width) -> np.ndarray:
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
        numpy.ndarray
            Array of contour points

        """
        point_start = self.point_start
        point_center = self.point_center
        vec_center_start = point_start - point_center

        if not raster_width > 0:
            raise ValueError("'raster_width' must be > 0")
        raster_width = np.clip(raster_width, None, self.arc_length)

        num_raster_segments = int(np.round(self._arc_length / raster_width))
        delta_angle = self._arc_angle / num_raster_segments

        max_angle = self._sign_arc_winding * (self._arc_angle + 0.5 * delta_angle)
        angles = np.arange(0, max_angle, self._sign_arc_winding * delta_angle)

        rotation_matrices = tf.WXRotation.from_euler("z", angles).as_matrix()[
            :, 0:2, 0:2
        ]

        data = np.matmul(rotation_matrices, vec_center_start) + point_center

        return data.transpose()

    @UREG.wraps(None, (None, ""), strict=False)
    def transform(self, matrix):
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

    @UREG.wraps(None, (None, _DEFAULT_LEN_UNIT), strict=False)
    def translate(self, vector):
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


class Shape:
    """Defines a shape in 2 dimensions."""

    def __init__(self, segments=None):
        """Construct a shape.

        Parameters
        ----------
        segments :
            Single segment or list of segments

        Returns
        -------
        Shape

        """
        segments = ut.to_list(segments)
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
    def _check_segments_connected(segments):
        """Check if all segments are connected to each other.

        The start point of a segment must be identical to the end point of
        the previous segment.

        Parameters
        ----------
        segments :
            List of segments

        """
        for i in range(len(segments) - 1):
            if not ut.vector_is_close(
                segments[i].point_end, segments[i + 1].point_start
            ):
                raise ValueError("Segments are not connected.")

    @classmethod
    def interpolate(cls, shape_a, shape_b, weight, interpolation_schemes):
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
    def linear_interpolation(cls, shape_a, shape_b, weight):
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
    def num_segments(self):
        """Get the number of segments of the shape.

        Returns
        -------
        int
            number of segments

        """
        return len(self._segments)

    @property
    def segments(self):
        """Get the shape's segments.

        Returns
        -------
        list
            List of segments

        """
        return self._segments

    @UREG.wraps(None, (None, _DEFAULT_LEN_UNIT), strict=False)
    def add_line_segments(self, points):
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
        points = ut.to_float_array(points)
        dimension = len(points.shape)
        if dimension == 1:
            points = points[np.newaxis, :]
        elif not dimension == 2:
            raise ValueError("Invalid input parameter")

        if not points.shape[1] == 2:
            raise ValueError("Invalid point format")

        if len(self.segments) > 0:
            points = np.vstack((self.segments[-1].point_end, points))
        elif points.shape[0] <= 1:
            raise ValueError("Insufficient number of points provided.")

        num_new_segments = len(points) - 1
        line_segments = []
        for i in range(num_new_segments):
            line_segments += [
                LineSegment.construct_with_points(points[i], points[i + 1])
            ]
        self.add_segments(line_segments)

        return self

    def add_segments(self, segments):
        """Add segments to the shape.

        Parameters
        ----------
        segments :
            Single segment or list of segments

        """
        segments = ut.to_list(segments)
        if self.num_segments > 0:
            self._check_segments_connected([self.segments[-1], segments[0]])
        self._check_segments_connected(segments)
        self._segments += segments

    def apply_transformation(self, transformation_matrix):
        """Apply a transformation to the shape.

        Parameters
        ----------
        transformation_matrix :
            Transformation matrix

        """
        for i in range(self.num_segments):
            self._segments[i].apply_transformation(transformation_matrix)

    def apply_reflection(self, reflection_normal, distance_to_origin=0):
        """Apply a reflection at the given axis to the shape.

        Parameters
        ----------
        reflection_normal :
            Normal of the line of reflection
        distance_to_origin :
            Distance of the line of reflection to the origin (Default value = 0)

        """
        normal = ut.to_float_array(reflection_normal)
        if ut.vector_is_close(normal, ut.to_float_array([0, 0])):
            raise ValueError("Normal has no length.")

        dot_product = np.dot(normal, normal)
        outer_product = np.outer(normal, normal)
        householder_matrix = np.identity(2) - 2 / dot_product * outer_product

        offset = normal / np.sqrt(dot_product) * distance_to_origin

        self.apply_translation(-offset)
        self.apply_transformation(householder_matrix)
        self.apply_translation(offset)

    @UREG.wraps(None, (None, _DEFAULT_LEN_UNIT, _DEFAULT_LEN_UNIT), strict=False)
    def apply_reflection_across_line(self, point_start, point_end):
        """Apply a reflection across a line.

        Parameters
        ----------
        point_start :
            Line of reflection's start point
        point_end :
            Line of reflection's end point

        """
        point_start = ut.to_float_array(point_start)
        point_end = ut.to_float_array(point_end)

        if ut.vector_is_close(point_start, point_end):
            raise ValueError("Line start and end point are identical.")

        vector = point_end - point_start
        length_vector = np.linalg.norm(vector)

        line_distance_origin = (
            np.abs(point_start[1] * point_end[0] - point_start[0] * point_end[1])
            / length_vector
        )

        if tf.point_left_of_line([0, 0], point_start, point_end) > 0:
            normal = ut.to_float_array([vector[1], -vector[0]])
        else:
            normal = ut.to_float_array([-vector[1], vector[0]])

        self.apply_reflection(normal, line_distance_origin)

    @UREG.wraps(None, (None, _DEFAULT_LEN_UNIT), strict=False)
    def apply_translation(self, vector):
        """Apply a translation to the shape.

        Parameters
        ----------
        vector :
            Translation vector

        """
        for i in range(self.num_segments):
            self._segments[i].apply_translation(vector)

    @UREG.wraps(None, (None, _DEFAULT_LEN_UNIT), strict=False)
    def rasterize(self, raster_width) -> np.ndarray:
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

        raster_data = []
        for segment in self.segments:
            raster_data.append(segment.rasterize(raster_width)[:, :-1])
        raster_data = np.hstack(raster_data)

        last_point = self.segments[-1].point_end[:, np.newaxis]
        if not ut.vector_is_close(last_point, self.segments[0].point_start):
            raster_data = np.hstack((raster_data, last_point))
        return raster_data

    def reflect(self, reflection_normal, distance_to_origin=0):
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

    @UREG.wraps(None, (None, _DEFAULT_LEN_UNIT, _DEFAULT_LEN_UNIT), strict=False)
    def reflect_across_line(self, point_start, point_end):
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

    def transform(self, matrix):
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

    @UREG.wraps(None, (None, _DEFAULT_LEN_UNIT), strict=False)
    def translate(self, vector):
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

    def __init__(self, shapes, units=None):
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
    def num_shapes(self):
        """Get the number of shapes of the profile.

        Returns
        -------
        int
            Number of shapes

        """
        return len(self._shapes)

    def add_shapes(self, shapes):
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

    @UREG.wraps(None, (None, _DEFAULT_LEN_UNIT, None), strict=False)
    def rasterize(
        self, raster_width, stack: bool = True
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Rasterize the profile.

        Parameters
        ----------
        raster_width :
            Distance between points for rasterization.
        stack :
            hstack data into a single output array (default = True)

        Returns
        -------
        numpy.ndarray or List[numpy.ndarray]
            Raster data

        """
        raster_data = []
        for shape in self._shapes:
            raster_data.append(shape.rasterize(raster_width))
        if stack:
            return np.hstack(raster_data)
        return raster_data

    def plot(
        self,
        title=None,
        raster_width=0.5,
        label=None,
        axis="equal",
        axis_labels=None,
        grid=True,
        line_style=".-",
        ax=None,
        color="k",
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
        elif "units" in self.attrs:
            ax.set_xlabel("y in " + self.attrs["units"])
            ax.set_ylabel("z in " + self.attrs["units"])

        if isinstance(color, str):  # single color
            color = [color] * len(raster_data)

        for segment, c in zip(raster_data, color):
            ax.plot(segment[0], segment[1], line_style, label=label, color=c)

    @property
    def shapes(self):
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

    @UREG.wraps(None, (None, _DEFAULT_LEN_UNIT), strict=False)
    def __init__(self, length):
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
    def length(self):
        """Get the length of the segment.

        Returns
        -------
        float
            Length of the segment

        """
        return self._length

    def local_coordinate_system(self, relative_position) -> tf.LocalCoordinateSystem:
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

    @UREG.wraps(None, (None, _DEFAULT_LEN_UNIT, _DEFAULT_ANG_UNIT, None), strict=False)
    def __init__(self, radius, angle, clockwise=False):
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
        self._length = self._arc_length(self.radius, self.angle)
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
    def _arc_length(radius, angle):
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
    def angle(self):
        """Get the angle of the segment.

        Returns
        -------
        float
            Angle of the segment (rad)

        """
        return self._angle

    @property
    def length(self):
        """Get the length of the segment.

        Returns
        -------
        float
            Length of the segment

        """
        return self._length

    @property
    def radius(self):
        """Get the radius of the segment.

        Returns
        -------
        float
            Radius of the segment

        """
        return self._radius

    @property
    def is_clockwise(self):
        """Get True, if the segments winding is clockwise, False otherwise.

        Returns
        -------
        bool
            True or False

        """
        return self._sign_winding < 0

    def local_coordinate_system(self, relative_position) -> tf.LocalCoordinateSystem:
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


class Trace:
    """Defines a 3d trace."""

    def __init__(self, segments, coordinate_system=None):
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

        self._segments = ut.to_list(segments)
        self._create_lookups(coordinate_system)

        if self.length <= 0:
            raise ValueError("Trace has no length.")

    def __repr__(self):
        """Output representation of a Trace."""
        return (
            f"Trace('segments': {self._segments!r}, "
            f"'coordinate_system_lookup': {self._coordinate_system_lookup!r}, "
            f"'total_length_lookup': {self._total_length_lookup!r}, "
            f"'segment_length_lookup': {self._segment_length_lookup!r})"
        )

    def _create_lookups(self, coordinate_system_start):
        """Create lookup tables.

        Parameters
        ----------
        coordinate_system_start :
            Coordinate system at the start of
            the trace.

        """
        self._coordinate_system_lookup = [coordinate_system_start]
        self._total_length_lookup = [0]
        self._segment_length_lookup = []

        segments = self._segments

        total_length = 0
        for i, segment in enumerate(segments):
            # Fill coordinate system lookup
            lcs_segment_end = segments[i].local_coordinate_system(1)
            lcs = lcs_segment_end + self._coordinate_system_lookup[i]
            self._coordinate_system_lookup += [lcs]

            # Fill length lookups
            segment_length = segment.length
            total_length += segment_length
            self._segment_length_lookup += [segment_length]
            self._total_length_lookup += [total_length]

    def _get_segment_index(self, position):
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
    def coordinate_system(self):
        """Get the trace's coordinate system.

        Returns
        -------
        weldx.transformations.LocalCoordinateSystem
            Coordinate system of the trace

        """
        return self._coordinate_system_lookup[0]

    @property
    def length(self):
        """Get the length of the trace.

        Returns
        -------
        float
            Length of the trace.

        """
        return self._total_length_lookup[-1]

    @property
    def segments(self):
        """Get the trace's segments.

        Returns
        -------
        list
            Segments of the trace

        """
        return self._segments

    @property
    def num_segments(self):
        """Get the number of segments.

        Returns
        -------
        int
            Number of segments

        """
        return len(self._segments)

    def local_coordinate_system(self, position) -> tf.LocalCoordinateSystem:
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

    @UREG.wraps(None, (None, _DEFAULT_LEN_UNIT), strict=False)
    def rasterize(self, raster_width):
        """Rasterize the trace.

        Parameters
        ----------
        raster_width :
           Distance between points for rasterization.

        Returns
        -------
        numpy.ndarray
            Raster data


        """
        if not raster_width > 0:
            raise ValueError("'raster_width' must be > 0")

        raster_width = np.clip(raster_width, 0, self.length)
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
        return np.hstack([raster_data, last_point])

    @UREG.wraps(None, (None, _DEFAULT_LEN_UNIT, None, None, None), strict=False)
    def plot(self, raster_width=1, axes=None, fmt=None, axes_equal=False):
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
        data = self.rasterize(raster_width)
        if fmt is None:
            fmt = "x-"
        if axes is None:
            from matplotlib.pyplot import figure

            fig = figure()
            axes = fig.gca(projection="3d", proj_type="ortho")
            axes.plot(data[0], data[1], data[2], fmt)
            axes.set_xlabel("x")
            axes.set_ylabel("y")
            axes.set_zlabel("z")
            if axes_equal:
                import weldx.visualization as vs

                vs.axes_equal(axes)
        else:
            axes.plot(data[0], data[1], data[2], fmt)


# Linear profile interpolation class ------------------------------------------


def linear_profile_interpolation_sbs(profile_a, profile_b, weight):
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

    def __init__(self, profiles, locations, interpolation_schemes):
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
        locations = ut.to_list(locations)
        interpolation_schemes = ut.to_list(interpolation_schemes)

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

    def _segment_index(self, location):
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
    def interpolation_schemes(self):
        """Get the interpolation schemes.

        Returns
        -------
        list
            List of interpolation schemes

        """
        return self._interpolation_schemes

    @property
    def locations(self):
        """Get the locations.

        Returns
        -------
        list
            List of locations

        """
        return self._locations

    @property
    def max_location(self):
        """Get the maximum location.

        Returns
        -------
        float
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
    def profiles(self):
        """Get the profiles.

        Returns
        -------
        list
            List of profiles

        """
        return self._profiles

    def local_profile(self, location):
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
        location = np.clip(location, 0, self.max_location)

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

    def __init__(self, profile, trace):
        """Construct a geometry.

        Parameters
        ----------
        profile : Profile, VariableProfile
            Constant or variable profile that is used as cross section along the
            specified trace
        trace : Trace
            The path that is used to extrude the given profile

        Returns
        -------
        Geometry :
            A Geometry class instance

        """
        self._check_inputs(profile, trace)
        self._profile = profile
        self._trace = trace

    def __repr__(self):
        """Output representation of a Geometry class."""
        return f"Geometry('profile': {self._profile!r}, 'trace': {self._trace!r})"

    @staticmethod
    def _check_inputs(profile, trace):
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

    def _get_local_profile_data(self, trace_location, raster_width):
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

    @UREG.wraps(None, (None, _DEFAULT_LEN_UNIT), strict=False)
    def _rasterize_trace(self, raster_width) -> np.ndarray:
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
            0, self._trace.length - raster_width_eff / 2, raster_width_eff
        )
        return np.hstack([locations, self._trace.length])

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
        return local_data + local_cs.coordinates.data[:, np.newaxis]

    @staticmethod
    @UREG.wraps(None, (None, _DEFAULT_LEN_UNIT, None), strict=False)
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

    @UREG.wraps(None, (None, _DEFAULT_LEN_UNIT, _DEFAULT_LEN_UNIT, None), strict=False)
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
        locations = self._rasterize_trace(trace_raster_width)

        if stack:  # old behavior for 3d point cloud
            profile_data = self._profile_raster_data_3d(
                self._profile, profile_raster_width, stack=True
            )
            raster_data = np.empty([3, 0])
            for _, location in enumerate(locations):
                local_data = self._get_transformed_profile_data(profile_data, location)
                raster_data = np.hstack([raster_data, local_data])

        else:
            profile_data = self._profile_raster_data_3d(
                self._profile, profile_raster_width, stack=False
            )

            raster_data = []
            for data in profile_data:
                raster_data.append(
                    np.stack(
                        [
                            self._get_transformed_profile_data(data, location)
                            for location in locations
                        ],
                        0,
                    )
                )

        return raster_data

    @UREG.wraps(None, (None, _DEFAULT_LEN_UNIT, _DEFAULT_LEN_UNIT), strict=False)
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
        locations = self._rasterize_trace(trace_raster_width)
        raster_data = np.empty([3, 0])
        for _, location in enumerate(locations):
            profile_data = self._get_local_profile_data(location, profile_raster_width)

            local_data = self._get_transformed_profile_data(profile_data, location)
            raster_data = np.hstack([raster_data, local_data])

        return raster_data

    @property
    def profile(self):
        """Get the geometry's profile.

        Returns
        -------
        Profile

        """
        return self._profile

    @property
    def trace(self):
        """Get the geometry's trace.

        Returns
        -------
        Trace

        """
        return self._trace

    @UREG.wraps(None, (None, _DEFAULT_LEN_UNIT, _DEFAULT_LEN_UNIT, None), strict=False)
    def rasterize(self, profile_raster_width, trace_raster_width, stack: bool = True):
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
        if isinstance(self._profile, Profile):
            return self._rasterize_constant_profile(
                profile_raster_width, trace_raster_width, stack=stack
            )
        return self._rasterize_variable_profile(
            profile_raster_width, trace_raster_width
        )

    @UREG.wraps(
        None,
        (
            None,
            _DEFAULT_LEN_UNIT,
            _DEFAULT_LEN_UNIT,
            None,
            None,
            None,
            None,
        ),
        strict=False,
    )
    def plot(
        self,
        profile_raster_width: pint.Quantity,
        trace_raster_width: pint.Quantity,
        axes: matplotlib.axes.Axes = None,
        color: Union[int, Tuple[int, int, int], Tuple[float, float, float]] = None,
        label: str = None,
        show_wireframe: bool = True,
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
        show_wireframe : bool
            If `True`, the mesh is plotted as wireframe. Otherwise only the raster
            points are visualized. Currently, the wireframe can't be visualized if a
            `VariableProfile` is used.

        Returns
        -------
        matplotlib.axes.Axes :
            The utilized matplotlib axes, if matplotlib was used as rendering backend

        """
        data = self.spatial_data(profile_raster_width, trace_raster_width)
        return data.plot(
            axes=axes, color=color, label=label, show_wireframe=show_wireframe
        )

    @UREG.wraps(
        None,
        (None, _DEFAULT_LEN_UNIT, _DEFAULT_LEN_UNIT),
        strict=False,
    )
    def spatial_data(
        self, profile_raster_width: pint.Quantity, trace_raster_width: pint.Quantity
    ):
        """Rasterize the geometry and get it as `SpatialData` instance.

        If no `VariableProfile` is used, a triangulation is added automatically.

        Parameters
        ----------
        profile_raster_width : pint.Quantity
            Target distance between the individual points of a profile
        trace_raster_width : pint.Quantity
            Target distance between the individual profiles on the trace

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
            return SpatialData(np.swapaxes(rasterization, 0, 1))

        rasterization = self.rasterize(
            profile_raster_width, trace_raster_width, stack=False
        )
        return SpatialData.from_geometry_raster(rasterization)


# SpatialData --------------------------------------------------------------------------


@ut.dataclass_nested_eq
@dataclass
class SpatialData:
    """Represent 3D point cloud data with optional triangulation.

    Parameters
    ----------
    coordinates
        3D array of point data.
    triangles
        3D Array of triangulation connectivity
    attributes
        optional dictionary with additional attributes to store alongside data

    """

    coordinates: np.ndarray
    triangles: np.ndarray = None
    attributes: Dict[str, np.ndarray] = None

    def __post_init__(self):
        """Convert and check input values."""
        if not isinstance(self.coordinates, DataArray):
            self.coordinates = DataArray(
                self.coordinates, dims=["n", "c"], coords={"c": ["x", "y", "z"]}
            )

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
    def from_file(file_name: Union[str, Path]) -> "SpatialData":
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
    def from_geometry_raster(geometry_raster: np.ndarray) -> "SpatialData":
        """Triangulate rasterized Geometry Profile.

        Parameters
        ----------
        geometry_raster : numpy.ndarray
            A single unstacked geometry rasterization.

        Returns
        -------
        SpatialData:
            New `SpatialData` instance

        """
        # todo: this needs a test
        # todo: workaround ... fix the real problem
        # if not isinstance(geometry_raster, np.ndarray):
        #    geometry_raster = np.array(geometry_raster)
        if geometry_raster[0].ndim == 2:
            return SpatialData(*ut.triangulate_geometry(geometry_raster))

        part_data = [ut.triangulate_geometry(part) for part in geometry_raster]

        total_points = []
        total_triangles = []
        for points, triangulation in part_data:
            total_triangles += (triangulation + len(total_points)).tolist()
            total_points += points.tolist()
        return SpatialData(total_points, total_triangles)

    def plot(
        self,
        axes: matplotlib.axes.Axes = None,
        color: Union[int, Tuple[int, int, int], Tuple[float, float, float]] = None,
        label: str = None,
        show_wireframe: bool = True,
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
        show_wireframe : bool
            If `True`, the mesh is plotted as wireframe. Otherwise only the raster
            points are visualized. Currently, the wireframe can't be visualized if a
            `VariableProfile` is used.

        Returns
        -------
        matplotlib.axes.Axes :
            The utilized matplotlib axes, if matplotlib was used as rendering backend

        """
        import weldx.visualization as vs

        return vs.plot_spatial_data_matplotlib(
            data=self,
            axes=axes,
            color=color,
            label=label,
            show_wireframe=show_wireframe,
        )

    def write_to_file(self, file_name: Union[str, Path]):
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
