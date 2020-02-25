"""Contains methods and classes for coordinate transformations."""

import weldx.utility as ut
import numpy as np
import math
from scipy.spatial.transform import Rotation as Rot


# functions -------------------------------------------------------------------


def rotation_matrix_x(angle):
    """
    Create a rotation matrix that rotates around the x-axis.

    :param angle: Rotation angle
    :return: Rotation matrix
    """
    return Rot.from_euler("x", angle).as_matrix()


def rotation_matrix_y(angle):
    """
    Create a rotation matrix that rotates around the y-axis.

    :param angle: Rotation angle
    :return: Rotation matrix
    """
    return Rot.from_euler("y", angle).as_matrix()


def rotation_matrix_z(angle):
    """
    Create a rotation matrix that rotates around the z-axis.

    :param angle: Rotation angle
    :return: Rotation matrix
    """
    return Rot.from_euler("z", angle).as_matrix()


def normalize(vec):
    """
    Normalize a vector.

    :param vec: Vector
    :return: Normalized vector
    """
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        raise Exception("Vector length is 0.")
    return vec / norm


def orientation_point_plane_containing_origin(point, p_a, p_b):
    """
    Determine a points orientation relative to a plane containing the origin.

    The side is defined by the winding order of the triangle 'origin - A -
    B'. When looking at it from the left-hand side, the ordering is clockwise
    and counter-clockwise when looking from the right-hand side.

    The function returns 1 if the point lies left of the plane, -1 if it is
    on the right and 0 if it lies on the plane.

    Note, that this function is not appropriate to check if a point lies on
    a plane since it has no tolerance to compensate for numerical errors.

    Additional note: The points A and B can also been considered as two
    vectors spanning the plane.

    :param point: Point
    :param p_a: Second point of the triangle 'origin - A - B'.
    :param p_b: Third point of the triangle 'origin - A - B'.
    :return: 1, -1 or 0 (see description)
    """
    if (
        math.isclose(np.linalg.norm(p_a), 0)
        or math.isclose(np.linalg.norm(p_b), 0)
        or math.isclose(np.linalg.norm(p_b - p_a), 0)
    ):
        raise Exception("One or more points describing the plane are identical.")

    return np.sign(np.linalg.det([p_a, p_b, point]))


def orientation_point_plane(point, p_a, p_b, p_c):
    """
    Determine a points orientation relative to an arbitrary plane.

    The side is defined by the winding order of the triangle 'A - B - C'.
    When looking at it from the left-hand side, the ordering is clockwise
    and counter-clockwise when looking from the right-hand side.

    The function returns 1 if the point lies left of the plane, -1 if it is
    on the right and 0 if it lies on the plane.

    Note, that this function is not appropriate to check if a point lies on
    a plane since it has no tolerance to compensate for numerical errors.

    :param point: Point
    :param p_a: First point of the triangle 'A - B - C'.
    :param p_b: Second point of the triangle 'A - B - C'.
    :param p_c: Third point of the triangle 'A - B - C'.
    :return: 1, -1 or 0 (see description)
    """
    vec_a_b = p_b - p_a
    vec_a_c = p_c - p_a
    vec_a_point = point - p_a
    return orientation_point_plane_containing_origin(vec_a_point, vec_a_b, vec_a_c)


def is_orthogonal(vec_u, vec_v, tolerance=1e-9):
    """
    Check if vectors are orthogonal.

    :param vec_u: First vector
    :param vec_v: Second vector
    :param tolerance: Numerical tolerance
    :return: True or False
    """
    if math.isclose(np.dot(vec_u, vec_u), 0) or math.isclose(np.dot(vec_v, vec_v), 0):
        raise Exception("One or both vectors have zero length.")

    return math.isclose(np.dot(vec_u, vec_v), 0, abs_tol=tolerance)


def point_left_of_line(point, line_start, line_end):
    """
    Determine if a point lies left of a line.

    Returns 1 if the point is left of the line and -1 if it is to the right.
    If the point is located on the line, this function returns 0.

    :param point: Point
    :param line_start: Starting point of the line
    :param line_end: End point of the line
    :return: 1,-1 or 0 (see description)
    """
    vec_line_start_end = line_end - line_start
    vec_line_start_point = point - line_start
    return vector_points_to_left_of_vector(vec_line_start_point, vec_line_start_end)


def reflection_sign(matrix):
    """
    Get a sign indicating if the transformation is a reflection.

    Returns -1 if the transformation contains a reflection and 1 if not.

    :param matrix: Transformation matrix
    :return: 1 or -1 (see description)
    """
    sign = int(np.sign(np.linalg.det(matrix)))

    if sign == 0:
        raise Exception("Invalid transformation")

    return sign


def vector_points_to_left_of_vector(vector, vector_reference):
    """
    Determine if a vector points to the left of another vector.

    Returns 1 if the vector points to the left of the reference vector and
    -1 if it points to the right. In case both vectors point into the same
    or the opposite directions, this function returns 0.

    :param vector: Vector
    :param vector_reference: Reference vector
    :return: 1,-1 or 0 (see description)
    """
    return int(np.sign(np.linalg.det([vector_reference, vector])))


# cartesian coordinate system class -------------------------------------------


class LocalCoordinateSystem:
    """Defines a local cartesian coordinate system in 3d."""

    def __init__(
        self,
        basis=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        origin=np.array([0, 0, 0]),
    ):
        """
        Construct a cartesian coordinate system.

        :param basis: Matrix of 3 orthogonal column vectors which represent
        the coordinate systems basis. Keep in mind, that the columns of the
        corresponding orientation matrix is equal to the normalized basis
        vectors. So each orthogonal transformation matrix can also be
        provided as basis.
        :param origin: Position of the origin
        :return: Cartesian coordinate system
        """
        basis = ut.to_float_array(basis)
        basis[:, 0] = normalize(basis[:, 0])
        basis[:, 1] = normalize(basis[:, 1])
        basis[:, 2] = normalize(basis[:, 2])

        if not (
            is_orthogonal(basis[:, 0], basis[:, 1])
            and is_orthogonal(basis[:, 1], basis[:, 2])
            and is_orthogonal(basis[:, 2], basis[:, 0])
        ):
            raise Exception("Basis vectors must be orthogonal")

        self._orientation = basis

        self._location = ut.to_float_array(origin)

    def __add__(self, rhs_cs):
        """
        Add 2 coordinate systems.

        Generates a new coordinate system by treating the left-hand side
        coordinate system as being defined in the right hand-side coordinate
        system.
        The transformations from the base coordinate system to the new
        coordinate system are equivalent to the combination of the
        transformations from both added coordinate systems:

        R_n = R_r * R_l
        T_n = R_r * T_l + T_r

        R_r and T_r are rotation matrix and translation vector of the
        right-hand side coordinate system, R_l and T_l of the left-hand side
        coordinate system and R_n and T_n of the resulting coordinate system.

        :param rhs_cs: Right-hand side coordinate system
        :return: Resulting coordinate system.
        """
        basis = np.matmul(rhs_cs.basis, self.basis)
        origin = np.matmul(rhs_cs.basis, self.origin) + rhs_cs.origin
        return LocalCoordinateSystem(basis, origin)

    def __sub__(self, rhs_cs):
        """
        Subtract 2 coordinate systems.

        Generates a new coordinate system from two local coordinate systems
        with the same reference coordinate system. The resulting system is
        equivalent to the left-hand side system but with the right-hand side
        as reference coordinate system.
        This is achieved by the following transformations:

        R_n = R_r^(-1) * R_l
        T_n = R_r^(-1) * (T_l - T_r)

        R_r and T_r are rotation matrix and translation vector of the
        right-hand side coordinate system, R_l and T_l of the left-hand side
        coordinate system and R_n and T_n of the resulting coordinate system.

        :param rhs_cs: Right-hand side coordinate system
        :return: Resulting coordinate system.
        """
        transformation_matrix = rhs_cs.basis.transpose()
        basis = np.matmul(transformation_matrix, self.basis)
        origin = np.matmul(transformation_matrix, self.origin - rhs_cs.origin)
        return LocalCoordinateSystem(basis, origin)

    @classmethod
    def construct_from_orientation(cls, orientation, origin=np.array([0, 0, 0])):
        """
        Construct a cartesian coordinate system from orientation matrix.

        :param orientation: Orthogonal transformation matrix
        :param origin: Position of the origin
        :return: Cartesian coordinate system
        """
        return cls(orientation, origin=origin)

    @classmethod
    def construct_from_xyz(cls, vec_x, vec_y, vec_z, origin=np.array([0, 0, 0])):
        """
        Construct a cartesian coordinate system from 3 basis vectors.

        :param vec_x: Vector defining the x-axis
        :param vec_y: Vector defining the y-axis
        :param vec_z: Vector defining the z-axis
        :param origin: Position of the origin
        :return: Cartesian coordinate system
        """
        basis = np.transpose([vec_x, vec_y, vec_z])
        return cls(basis, origin=origin)

    @classmethod
    def construct_from_xy_and_orientation(
        cls, vec_x, vec_y, positive_orientation=True, origin=np.array([0, 0, 0])
    ):
        """
        Construct a coordinate system from 2 vectors and an orientation.

        :param vec_x: Vector defining the x-axis
        :param vec_y: Vector defining the y-axis
        :param positive_orientation: Set to True if the orientation should
        be positive and to False if not
        :param origin: Position of the origin
        :return: Cartesian coordinate system
        """
        vec_z = cls._calculate_orthogonal_axis(vec_x, vec_y) * cls._sign_orientation(
            positive_orientation
        )

        basis = np.transpose([vec_x, vec_y, vec_z])
        return cls(basis, origin=origin)

    @classmethod
    def construct_from_yz_and_orientation(
        cls, vec_y, vec_z, positive_orientation=True, origin=np.array([0, 0, 0])
    ):
        """
        Construct a coordinate system from 2 vectors and an orientation.

        :param vec_y: Vector defining the y-axis
        :param vec_z: Vector defining the z-axis
        :param positive_orientation: Set to True if the orientation should
        be positive and to False if not
        :param origin: Position of the origin
        :return: Cartesian coordinate system
        """
        vec_x = cls._calculate_orthogonal_axis(vec_y, vec_z) * cls._sign_orientation(
            positive_orientation
        )

        basis = np.transpose(np.array([vec_x, vec_y, vec_z]))
        return cls(basis, origin=origin)

    @classmethod
    def construct_from_xz_and_orientation(
        cls, vec_x, vec_z, positive_orientation=True, origin=np.array([0, 0, 0])
    ):
        """
        Construct a coordinate system from 2 vectors and an orientation.

        :param vec_x: Vector defining the x-axis
        :param vec_z: Vector defining the z-axis
        :param positive_orientation: Set to True if the orientation should
        be positive and to False if not
        :param origin: Position of the origin
        :return: Cartesian coordinate system
        """
        vec_y = cls._calculate_orthogonal_axis(vec_z, vec_x) * cls._sign_orientation(
            positive_orientation
        )

        basis = np.transpose([vec_x, vec_y, vec_z])
        return cls(basis, origin=origin)

    @staticmethod
    def _sign_orientation(positive_orientation):
        """
        Get -1 or 1 depending on the coordinate systems orientation.

        :param positive_orientation: Set to True if the orientation should
        be positive and to False if not
        :return: 1 if the coordinate system has positive orientation,
        -1 otherwise
        """
        if positive_orientation:
            return 1
        return -1

    @staticmethod
    def _calculate_orthogonal_axis(a_0, a_1):
        """
        Calculate an axis which is orthogonal to two other axes.

        The calculated axis has a positive orientation towards the other 2
        axes.

        :param a_0: First axis
        :param a_1: Second axis
        :return: Orthogonal axis
        """
        return np.cross(a_0, a_1)

    @property
    def basis(self):
        """
        Get the normalizes basis as matrix of 3 column vectors.

        This function is identical to the 'orientation' function.

        :return: Basis of the coordinate system
        """
        return self._orientation

    @property
    def orientation(self):
        """
        Get the coordinate systems orientation matrix.

        This function is identical to the 'basis' function.

        :return: Orientation matrix
        """
        return self._orientation

    @property
    def origin(self):
        """
        Get the coordinate systems origin.

        This function is identical to the 'location' function.

        :return: Origin of the coordinate system
        """
        return self._location

    @property
    def location(self):
        """
        Get the coordinate systems location.

        This function is identical to the 'origin' function.

        :return: Location of the coordinate system.
        """
        return self._location
