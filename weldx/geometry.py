"""Provides classes to define lines and surfaces."""

import weldx.utility as ut
import weldx.transformations as tf

import copy
import math
import numpy as np


# LineSegment -----------------------------------------------------------------


class LineSegment:
    """Line segment."""

    def __init__(self, points):
        """
        Constructor.

        :param points: 2x2 matrix of points. The first column is the
        starting point and the second column the end point.
        """
        points = ut.to_float_array(points)
        if not len(points.shape) == 2:
            raise ValueError("'points' must be a 2d array/matrix.")
        if not (points.shape[0] == 2 and points.shape[1] == 2):
            raise ValueError("'points' is not a 2x2 matrix.")

        self._points = points
        self._calculate_length()

    def _calculate_length(self):
        """
        Calculate the segment length from its points.

        :return: ---
        """
        self._length = np.linalg.norm(self._points[:, 1] - self._points[:, 0])
        if math.isclose(self._length, 0):
            raise ValueError("Segment length is 0.")

    @classmethod
    def construct_with_points(cls, point_start, point_end):
        """
        Construct a line segment with two points.

        :param point_start: Starting point of the segment
        :param point_end: End point of the segment
        :return: Line segment
        """
        points = np.transpose(np.array([point_start, point_end], dtype=float))
        return cls(points)

    @classmethod
    def linear_interpolation(cls, segment_a, segment_b, weight):
        """
        Interpolate two line segments linearly.

        :param segment_a: First segment
        :param segment_b: Second segment
        :param weight: Weighting factor in the range [0 .. 1] where 0 is
        segment a and 1 is segment b
        :return: Interpolated segment
        """
        if not isinstance(segment_a, cls) or not isinstance(segment_b, cls):
            raise TypeError("Parameters a and b must both be line segments.")

        weight = np.clip(weight, 0, 1)
        points = (1 - weight) * segment_a.points + weight * segment_b.points
        return cls(points)

    @property
    def length(self):
        """
        Get the segment length.

        :return: Segment length
        """
        return self._length

    @property
    def point_end(self):
        """
        Get the end point of the segment.

        :return: End point
        """
        return self._points[:, 1]

    @property
    def point_start(self):
        """
        Get the starting point of the segment.

        :return: Starting point
        """
        return self._points[:, 0]

    @property
    def points(self):
        """
        Get the segments points in form of a 2x2 matrix.

        The first column represents the starting point and the second one
        the end point.

        :return: 2x2 matrix containing the segments points
        """
        return self._points

    def apply_transformation(self, matrix):
        """
        Apply a transformation matrix to the segment.

        :param matrix: Transformation matrix
        :return: ---
        """
        self._points = np.matmul(matrix, self._points)
        self._calculate_length()

    def apply_translation(self, vector):
        """
        Apply a translation to the segment.

        :param vector: Translation vector
        :return: ---
        """
        self._points += np.ndarray((2, 1), float, np.array(vector, float))

    def rasterize(self, raster_width):
        """
        Create an array of points that describe the segments contour.

        The effective raster width may vary from the specified one,
        since the algorithm enforces constant distances between two
        raster points.

        :param raster_width: The desired distance between two raster points
        :return: Array of contour points
        """
        if not raster_width > 0:
            raise ValueError("'raster_width' must be > 0")
        raster_width = np.clip(np.abs(raster_width), None, self.length)

        num_raster_segments = np.round(self.length / raster_width)

        # normalized effective raster width
        nerw = 1.0 / num_raster_segments

        multiplier = np.arange(0, 1 + 0.5 * nerw, nerw)
        weight_matrix = np.array([1 - multiplier, multiplier])

        return np.matmul(self._points, weight_matrix)

    def transform(self, matrix):
        """
        Get a transformed copy of the segment.

        :param matrix: Transformation matrix
        :return: Transformed copy
        """
        new_segment = copy.deepcopy(self)
        new_segment.apply_transformation(matrix)
        return new_segment

    def translate(self, vector):
        """
        Get a translated copy of the segment.

        :param vector: Translation vector
        :return: Transformed copy
        """
        new_segment = copy.deepcopy(self)
        new_segment.apply_translation(vector)
        return new_segment


# ArcSegment ------------------------------------------------------------------


class ArcSegment:
    """Arc segment."""

    def __init__(self, points, arc_winding_ccw=True):
        """
        Constructor.

        :param points: 2x3 matrix of points. The first column is the
        starting point, the second column the end point and the last the
        center point.
        :param: arc_winding_ccw: Specifies if the arcs winding order is
        counter-clockwise
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

    def _calculate_arc_angle(self):
        """
        Calculate the arc angle.

        :return: ---
        """
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
        """
        Calculate radius, arc length and arc angle from the segments points.

        :return: ---
        """
        self._radius = np.linalg.norm(self._points[:, 0] - self._points[:, 2])
        self._calculate_arc_angle()
        self._arc_length = self._arc_angle * self._radius

        self._check_valid()

    def _check_valid(self):
        """
        Check if the segments data is valid.

        :return: ---
        """
        point_start = self.point_start
        point_end = self.point_end
        point_center = self.point_center

        radius_start_center = np.linalg.norm(point_start - point_center)
        radius_end_center = np.linalg.norm(point_end - point_center)
        radius_diff = radius_end_center - radius_start_center

        if not math.isclose(radius_diff, 0, abs_tol=1e-9):
            raise ValueError("Radius is not constant.")
        if math.isclose(self._arc_length, 0):
            raise Exception("Arc length is 0.")

    @classmethod
    def construct_with_points(
        cls, point_start, point_end, point_center, arc_winding_ccw=True
    ):
        """
        Construct an arc segment with three points (start, end, center).

        :param point_start: Starting point of the segment
        :param point_end: End point of the segment
        :param point_center: Center point of the arc
        :param arc_winding_ccw: Specifies if the arcs winding order is
        counter-clockwise
        :return: Arc segment
        """
        points = np.transpose(
            np.array([point_start, point_end, point_center], dtype=float)
        )
        return cls(points, arc_winding_ccw)

    @classmethod
    def construct_with_radius(
        cls,
        point_start,
        point_end,
        radius,
        center_left_of_line=True,
        arc_winding_ccw=True,
    ):
        """
        Construct an arc segment with a radius and the start and end points.

        :param point_start: Starting point of the segment
        :param point_end: End point of the segment
        :param radius: Radius
        :param center_left_of_line: Specifies if the center point is located
        to the left of the vector point_start -> point_end
        :param arc_winding_ccw: Specifies if the arcs winding order is
        counter-clockwise
        :return: Arc segment
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
    def linear_interpolation(cls, segment_a, segment_b, weight):
        """
        Interpolate two arc segments linearly.

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

        :param segment_a: First segment
        :param segment_b: Second segment
        :param weight: Weighting factor in the range [0 .. 1] where 0 is
        segment a and 1 is segment b
        :return: Interpolated segment
        """
        raise Exception(
            "Linear interpolation of an arc segment is not unique (see "
            "docstring). You need to provide a custom interpolation."
        )

    @property
    def arc_angle(self):
        """
        Get the arc angle.

        :return: Arc angle
        """
        return self._arc_angle

    @property
    def arc_length(self):
        """
        Get the arc length.

        :return: Arc length
        """
        return self._arc_length

    @property
    def arc_winding_ccw(self):
        """
        Get True if the winding order is counter-clockwise. False if clockwise.

        :return: True or False
        """
        return self._sign_arc_winding > 0

    @property
    def point_center(self):
        """
        Get the center point of the segment.

        :return: Center point
        """
        return self._points[:, 2]

    @property
    def point_end(self):
        """
        Get the end point of the segment.

        :return: End point
        """
        return self._points[:, 1]

    @property
    def point_start(self):
        """
        Get the starting point of the segment.

        :return: Starting point
        """
        return self._points[:, 0]

    @property
    def points(self):
        """
        Get the segments points in form of a 2x3 matrix.

        The first column represents the starting point, the second one the
        end and the third one the center.

        :return: 2x3 matrix containing the segments points
        """
        return self._points

    @property
    def radius(self):
        """
        Get the radius.

        :return: Radius
        """
        return self._radius

    def apply_transformation(self, matrix):
        """
        Apply a transformation to the segment.

        :param matrix: Transformation matrix
        :return: ---
        """
        self._points = np.matmul(matrix, self._points)
        self._sign_arc_winding *= tf.reflection_sign(matrix)
        self._calculate_arc_parameters()

    def apply_translation(self, vector):
        """
        Apply a translation to the segment.

        :param vector: Translation vector
        :return: ---
        """
        self._points += np.ndarray((2, 1), float, np.array(vector, float))

    def rasterize(self, raster_width):
        """
        Create an array of points that describe the segments contour.

        The effective raster width may vary from the specified one,
        since the algorithm enforces constant distances between two
        raster points.

        :param raster_width: The desired distance between two raster points
        :return: Array of contour points
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

        rotation_matrices = tf.rotation_matrix_z(angles)[:, 0:2, 0:2]

        data = np.matmul(rotation_matrices, vec_center_start) + point_center

        return data.transpose()

    def transform(self, matrix):
        """
        Get a transformed copy of the segment.

        :param matrix: Transformation matrix
        :return: Transformed copy
        """
        new_segment = copy.deepcopy(self)
        new_segment.apply_transformation(matrix)
        return new_segment

    def translate(self, vector):
        """
        Get a translated copy of the segment.

        :param vector: Translation vector
        :return: Transformed copy
        """
        new_segment = copy.deepcopy(self)
        new_segment.apply_translation(vector)
        return new_segment


# Shape class -----------------------------------------------------------------


class Shape:
    """Defines a shape in 2 dimensions."""

    def __init__(self, segments=None):
        """
        Constructor.

        :param segments: Single segment or list of segments
        """
        segments = ut.to_list(segments)
        self._check_segments_connected(segments)
        self._segments = segments

    @staticmethod
    def _check_segments_connected(segments):
        """
        Check if all segments are connected to each other.

        The start point of a segment must be identical to the end point of
        the previous segment.

        :param segments: List of segments
        :return: ---
        """
        for i in range(len(segments) - 1):
            if not ut.vector_is_close(
                segments[i].point_end, segments[i + 1].point_start
            ):
                raise Exception("Segments are not connected.")

    @classmethod
    def interpolate(cls, shape_a, shape_b, weight, interpolation_schemes):
        """
        Interpolate 2 shapes.

        :param shape_a: First shape
        :param shape_b: Second shape
        :param weight: Weighting factor in the range [0 .. 1] where 0 is
        shape a and 1 is shape b
        :param interpolation_schemes: List of interpolation schemes for each
        segment of the shape.
        :return: Interpolated shape
        """
        if not shape_a.num_segments == shape_b.num_segments:
            raise Exception("Number of segments differ.")

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
        """
        Interpolate 2 shapes linearly.

        Each segment is interpolated individually, using the corresponding
        linear segment interpolation.

        :param shape_a: First shape
        :param shape_b: Second shape
        :param weight: Weighting factor in the range [0 .. 1] where 0 is
        shape a and 1 is shape b
        :return: Interpolated shape
        """
        interpolation_schemes = []
        for i in range(shape_a.num_segments):
            interpolation_schemes += [shape_a.segments[i].linear_interpolation]

        return cls.interpolate(shape_a, shape_b, weight, interpolation_schemes)

    @property
    def num_segments(self):
        """
        Get the number of segments of the shape.

        :return: number of segments
        """
        return len(self._segments)

    @property
    def segments(self):
        """
        Get the shape's segments.

        :return: List of segments
        """
        return self._segments

    def add_line_segments(self, points):
        """
        Add line segments to the shape.

        The line segments are constructed from the provided points.

        :param points:  List of points / Matrix Nx2 matrix
        :return: self
        """
        points = ut.to_float_array(points)
        dimension = len(points.shape)
        if dimension == 1:
            points = points[np.newaxis, :]
        elif not dimension == 2:
            raise Exception("Invalid input parameter")

        if not points.shape[1] == 2:
            raise Exception("Invalid point format")

        if len(self.segments) > 0:
            points = np.vstack((self.segments[-1].point_end, points))
        elif points.shape[0] <= 1:
            raise Exception("Insufficient number of points provided.")

        num_new_segments = len(points) - 1
        line_segments = []
        for i in range(num_new_segments):
            line_segments += [
                LineSegment.construct_with_points(points[i], points[i + 1])
            ]
        self.add_segments(line_segments)

        return self

    def add_segments(self, segments):
        """
        Add segments to the shape.

        :param segments: Single segment or list of segments
        :return: ---
        """
        segments = ut.to_list(segments)
        if self.num_segments > 0:
            self._check_segments_connected([self.segments[-1], segments[0]])
        self._check_segments_connected(segments)
        self._segments += segments

    def apply_transformation(self, transformation_matrix):
        """
        Apply a transformation to the shape.

        :param transformation_matrix: Transformation matrix
        :return: ---
        """
        for i in range(self.num_segments):
            self._segments[i].apply_transformation(transformation_matrix)

    def apply_reflection(self, reflection_normal, distance_to_origin=0):
        """
        Apply a reflection at the given axis to the shape.

        :param reflection_normal: Normal of the line of reflection
        :param distance_to_origin: Distance of the line of reflection to the
        origin
        :return: ---
        """
        normal = ut.to_float_array(reflection_normal)
        if ut.vector_is_close(normal, ut.to_float_array([0, 0])):
            raise Exception("Normal has no length.")

        dot_product = np.dot(normal, normal)
        outer_product = np.outer(normal, normal)
        householder_matrix = np.identity(2) - 2 / dot_product * outer_product

        offset = normal / np.sqrt(dot_product) * distance_to_origin

        self.apply_translation(-offset)
        self.apply_transformation(householder_matrix)
        self.apply_translation(offset)

    def apply_reflection_across_line(self, point_start, point_end):
        """
        Apply a reflection across a line.

        :param point_start: Line of reflection's start point
        :param point_end: Line of reflection's end point
        :return: ---
        """
        point_start = ut.to_float_array(point_start)
        point_end = ut.to_float_array(point_end)

        if ut.vector_is_close(point_start, point_end):
            raise Exception("Line start and end point are identical.")

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

    def apply_translation(self, vector):
        """
        Apply a translation to the shape.

        :param vector: Translation vector
        :return: ---
        """
        for i in range(self.num_segments):
            self._segments[i].apply_translation(vector)

    def rasterize(self, raster_width):
        """
        Create an array of points that describe the shapes contour.

        The effective raster width may vary from the specified one,
        since the algorithm enforces constant distances between two
        raster points inside of each segment.

        :param raster_width: The desired distance between two raster points
        :return: Array of contour points (3d)
        """
        if self.num_segments == 0:
            raise Exception("Can't rasterize empty shape.")
        if not raster_width > 0:
            raise ValueError("'raster_width' must be > 0")

        raster_data = np.empty([2, 0])
        for i in range(self.num_segments):
            segment_data = self.segments[i].rasterize(raster_width)
            raster_data = np.hstack((raster_data, segment_data[:, :-1]))

        last_point = self.segments[-1].point_end[:, np.newaxis]
        if not ut.vector_is_close(last_point, self.segments[0].point_start):
            raster_data = np.hstack((raster_data, last_point))
        return raster_data

    def reflect(self, reflection_normal, distance_to_origin=0):
        """
        Get a reflected copy of the shape.

        :param reflection_normal: Normal of the line of reflection
        :param distance_to_origin: Distance of the line of reflection to the
        origin
        :return: ---
        """
        new_shape = copy.deepcopy(self)
        new_shape.apply_reflection(reflection_normal, distance_to_origin)
        return new_shape

    def reflect_across_line(self, point_start, point_end):
        """
        Get a reflected copy across a line.

        :param point_start: Line of reflection's start point
        :param point_end: Line of reflection's end point
        :return
        """
        new_shape = copy.deepcopy(self)
        new_shape.apply_reflection_across_line(point_start, point_end)
        return new_shape

    def transform(self, matrix):
        """
        Get a transformed copy of the shape.

        :param matrix: Transformation matrix
        :return: Transformed copy
        """
        new_shape = copy.deepcopy(self)
        new_shape.apply_transformation(matrix)
        return new_shape

    def translate(self, vector):
        """
        Get a translated copy of the shape.

        :param vector: Translation vector
        :return: Transformed copy
        """
        new_shape = copy.deepcopy(self)
        new_shape.apply_translation(vector)
        return new_shape


# Profile class ---------------------------------------------------------------


class Profile:
    """Defines a 2d profile."""

    def __init__(self, shapes):
        """
        Construct profile class.

        :param: shapes: Instance or list of geo.Shape class(es)
        """
        self._shapes = []
        self.add_shapes(shapes)

    @property
    def num_shapes(self):
        """
        Get the number of shapes of the profile.

        :return: Number of shapes
        """
        return len(self._shapes)

    def add_shapes(self, shapes):
        """
        Add shapes to the profile.

        :param shapes: Instance or list of geo.Shape class(es)
        :return: ---
        """
        if not isinstance(shapes, list):
            shapes = [shapes]

        if not all(isinstance(shape, Shape) for shape in shapes):
            raise TypeError("Only instances or lists of Shape objects are accepted.")

        self._shapes += shapes

    def rasterize(self, raster_width):
        """
        Rasterize the profile.

        :param: raster_width: Raster width
        :return: Raster data
        """
        raster_data = np.empty([2, 0])
        for shape in self._shapes:
            raster_data = np.hstack((raster_data, shape.rasterize(raster_width)))

        return raster_data

    @property
    def shapes(self):
        """
        Get the profiles shapes.

        :return: Shapes
        """
        return self._shapes


# Trace segment classes -------------------------------------------------------


class LinearHorizontalTraceSegment:
    """Trace segment with a linear path and constant z-component."""

    def __init__(self, length):
        """
        Constructor.

        :param length: Length of the segment
        """
        if length <= 0:
            raise ValueError("'length' must have a positive value.")
        self._length = float(length)

    @property
    def length(self):
        """
        Get the length of the segment.

        :return: Length of the segment
        """
        return self._length

    def local_coordinate_system(self, relative_position):
        """
        Calculate a local coordinate system along the trace segment.

        :param relative_position: Relative position on the trace [0 .. 1]
        :return: Local coordinate system
        """
        relative_position = np.clip(relative_position, 0, 1)

        origin = np.array([1, 0, 0]) * relative_position * self._length
        return tf.LocalCoordinateSystem(origin=origin)


class RadialHorizontalTraceSegment:
    """Trace segment describing an arc with constant z-component."""

    def __init__(self, radius, angle, clockwise=False):
        """
        Constructor.

        :param radius: Radius of the arc
        :param angle: Angle of the arc
        :param clockwise: If True, the rotation is clockwise. Otherwise it
        is counter-clockwise.
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

    @staticmethod
    def _arc_length(radius, angle):
        """
        Calculate the arc length.

        :param radius: Radius
        :param angle: Angle (rad)
        :return: Arc length
        """
        return angle * radius

    @property
    def angle(self):
        """
        Get the angle of the segment.

        :return: Angle of the segment
        """
        return self._angle

    @property
    def length(self):
        """
        Get the length of the segment.

        :return: Length of the segment
        """
        return self._length

    @property
    def radius(self):
        """
        Get the radius of the segment.

        :return: Radius of the segment
        """
        return self._radius

    @property
    def is_clockwise(self):
        """
        Get True, if the segments winding is clockwise, False otherwise.

        :return: True or False
        """
        return self._sign_winding < 0

    def local_coordinate_system(self, relative_position):
        """
        Calculate a local coordinate system along the trace segment.

        :param relative_position: Relative position on the trace [0 .. 1]
        :return: Local coordinate system
        """
        relative_position = np.clip(relative_position, 0, 1)

        basis = tf.rotation_matrix_z(
            self._angle * relative_position * self._sign_winding
        )
        translation = np.array([0, -1, 0]) * self._radius * self._sign_winding

        origin = np.matmul(basis, translation) - translation
        return tf.LocalCoordinateSystem(basis, origin)


# Trace class -----------------------------------------------------------------


class Trace:
    """Defines a 3d trace."""

    def __init__(self, segments, coordinate_system=tf.LocalCoordinateSystem()):
        """
        Constructor.

        :param segments: Single segment or list of segments
        :param coordinate_system: Coordinate system of the trace
        """
        if not isinstance(coordinate_system, tf.LocalCoordinateSystem):
            raise TypeError(
                "'coordinate_system' must be of type "
                "'transformations.LocalCoordinateSystem'"
            )

        self._segments = ut.to_list(segments)
        self._create_lookups(coordinate_system)

        if self.length <= 0:
            raise Exception("Trace has no length.")

    def _create_lookups(self, coordinate_system_start):
        """
        Create lookup tables.

        :param coordinate_system_start: Coordinate system at the start of
        the trace.
        :return: ---
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
        """
        Get the segment index for a certain position.

        :param position: Position
        :return: Segment index
        """
        position = np.clip(position, 0, self.length)
        for i in range(len(self._total_length_lookup) - 2):
            if position <= self._total_length_lookup[i + 1]:
                return i
        return self.num_segments - 1

    @property
    def coordinate_system(self):
        """
        Get the trace's coordinate system.

        :return: Coordinate system of the trace
        """
        return self._coordinate_system_lookup[0]

    @property
    def length(self):
        """
        Get the length of the trace.

        :return: Length of the trace.
        """
        return self._total_length_lookup[-1]

    @property
    def segments(self):
        """
        Get the trace's segments.

        :return: Segments of the trace
        """
        return self._segments

    @property
    def num_segments(self):
        """
        Get the number of segments.

        :return: Number of segments
        """
        return len(self._segments)

    def local_coordinate_system(self, position):
        """
        Get the local coordinate system at a specific position on the trace.

        :param position: Position
        :return: Local coordinate system
        """
        idx = self._get_segment_index(position)

        total_length_start = self._total_length_lookup[idx]
        segment_length = self._segment_length_lookup[idx]
        weight = (position - total_length_start) / segment_length

        local_segment_cs = self.segments[idx].local_coordinate_system(weight)
        segment_start_cs = self._coordinate_system_lookup[idx]

        return local_segment_cs + segment_start_cs

    def rasterize(self, raster_width):
        """
        Rasterize the trace.

        :return: Raster data
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

            data_point = local_cs.origin[:, np.newaxis]
            raster_data = np.hstack([raster_data, data_point])

        last_point = self._coordinate_system_lookup[-1].origin[:, np.newaxis]
        return np.hstack([raster_data, last_point])


# Linear profile interpolation class ------------------------------------------


def linear_profile_interpolation_sbs(profile_a, profile_b, weight):
    """
    Interpolate 2 profiles linearly, segment by segment.

    :param profile_a: First profile
    :param profile_b: Second profile
    :param weight: Weighting factor [0 .. 1]. If 0, the profile is identical
    to 'a' and if 1, it is identical to b.
    :return: Interpolated profile
    """
    weight = np.clip(weight, 0, 1)
    if not len(profile_a.shapes) == len(profile_b.shapes):
        raise Exception("Number of profile shapes do not match.")

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
        """
        Constructor.

        :param profiles: List of profiles.
        :param locations: Ascending list of profile locations. Since the
        first location needs to be 0, it can be omitted.
        :param interpolation_schemes: List of interpolation schemes to
        define the interpolation between two locations.
        """
        locations = ut.to_list(locations)
        interpolation_schemes = ut.to_list(interpolation_schemes)

        if not locations[0] == 0:
            locations = [0] + locations

        if not len(profiles) == len(locations):
            raise Exception("Invalid list of locations. See function description.")

        if not len(interpolation_schemes) == len(profiles) - 1:
            raise Exception(
                "Number of interpolations must be 1 less than number of " "profiles."
            )

        for i in range(len(profiles) - 1):
            if locations[i] >= locations[i + 1]:
                raise Exception("Locations need to be sorted in ascending order.")

        self._profiles = profiles
        self._locations = locations
        self._interpolation_schemes = interpolation_schemes

    def _segment_index(self, location):
        """
        Get the index of the segment at a certain location.

        :param location: Location
        :return: Segment index
        """
        idx = 0
        while location > self._locations[idx + 1]:
            idx += 1
        return idx

    @property
    def interpolation_schemes(self):
        """
        Get the interpolation schemes.

        :return: List of interpolation schemes
        """
        return self._interpolation_schemes

    @property
    def locations(self):
        """
        Get the locations.

        :return: List of locations
        """
        return self._locations

    @property
    def max_location(self):
        """
        Get the maximum location.

        :return: Maximum location
        """
        return self._locations[-1]

    @property
    def num_interpolation_schemes(self):
        """
        Get the number of interpolation schemes.

        :return: Number of interpolation schemes
        """
        return len(self._interpolation_schemes)

    @property
    def num_locations(self):
        """
        Get the number of profile locations.

        :return: Number of profile locations
        """
        return len(self._locations)

    @property
    def num_profiles(self):
        """
        Get the number of profiles.

        :return: Number of profiles
        """
        return len(self._profiles)

    @property
    def profiles(self):
        """
        Get the profiles.

        :return: List of profiles
        """
        return self._profiles

    def local_profile(self, location):
        """
        Get the profile at the specified location.

        :param location: Location
        :return: Local profile.
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
    """Define the experimental geometry."""

    def __init__(self, profile, trace):
        """
        Constructor.

        :param profile: Constant or variable profile.
        :param trace: Trace
        """
        self._check_inputs(profile, trace)
        self._profile = profile
        self._trace = trace

    @staticmethod
    def _check_inputs(profile, trace):
        """
        Check the inputs to the constructor.

        :param profile: Constant or variable profile.
        :param trace: Trace
        :return: ---
        """
        if not isinstance(profile, (Profile, VariableProfile)):
            raise TypeError("'profile' must be a 'Profile' or 'VariableProfile' class")

        if not isinstance(trace, Trace):
            raise TypeError("'trace' must be a 'Trace' class")

    def _get_local_profile_data(self, trace_location, raster_width):
        """
        Get a rasterized profile at a certain location on the trace.

        :param trace_location: Location on the trace
        :param raster_width: Raster width
        :return:
        """
        relative_location = trace_location / self._trace.length
        profile_location = relative_location * self._profile.max_location
        profile = self._profile.local_profile(profile_location)
        return self._profile_raster_data_3d(profile, raster_width)

    def _rasterize_trace(self, raster_width):
        """
        Rasterize the trace.

        :param raster_width: Raster width
        :return: Raster data
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
        """
        Transform a profiles data to a specified location on the trace.

        :param profile_raster_data: Rasterized profile
        :param location: Location on the trace
        :return: Transformed profile data
        """
        local_cs = self._trace.local_coordinate_system(location)
        local_data = np.matmul(local_cs.basis, profile_raster_data)
        return local_data + local_cs.origin[:, np.newaxis]

    @staticmethod
    def _profile_raster_data_3d(profile, raster_width):
        """
        Get the rasterized profile in 3d.

        The profile is located in the x-z-plane.

        :param profile: Profile
        :param raster_width: Raster width
        :return: Rasterized profile in 3d
        """
        profile_data = profile.rasterize(raster_width)
        return np.insert(profile_data, 0, 0, axis=0)

    def _rasterize_constant_profile(self, profile_raster_width, trace_raster_width):
        """
        Rasterize the geometry with a constant profile.

        :param profile_raster_width: Raster width of the profiles
        :param trace_raster_width: Distance between two profiles
        :return: Raster data
        """
        profile_data = self._profile_raster_data_3d(self._profile, profile_raster_width)

        locations = self._rasterize_trace(trace_raster_width)
        raster_data = np.empty([3, 0])
        for _, location in enumerate(locations):
            local_data = self._get_transformed_profile_data(profile_data, location)
            raster_data = np.hstack([raster_data, local_data])

        return raster_data

    def _rasterize_variable_profile(self, profile_raster_width, trace_raster_width):
        """
        Rasterize the geometry with a variable profile.

        :param profile_raster_width: Raster width of the profiles
        :param trace_raster_width: Distance between two profiles
        :return: Raster data
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
        """
        Get the geometry's profile.

        :return: Profile
        """
        return self._profile

    @property
    def trace(self):
        """
        Get the geometry's trace.

        :return: Trace
        """
        return self._trace

    def rasterize(self, profile_raster_width, trace_raster_width):
        """
        Rasterize the geometry.

        :param profile_raster_width: Raster width of the profiles
        :param trace_raster_width: Distance between two profiles
        :return: Raster data
        """
        if isinstance(self._profile, Profile):
            return self._rasterize_constant_profile(
                profile_raster_width, trace_raster_width
            )
        return self._rasterize_variable_profile(
            profile_raster_width, trace_raster_width
        )
