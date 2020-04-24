"""Contains methods and classes for coordinate transformations."""

import collections.abc as cl
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Hashable, List, Union

import networkx as nx
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial.transform import Rotation as Rot

import weldx.utility as ut

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


def scale_matrix(scale_x, scale_y, scale_z):
    """
    Return a scaling matrix.

    :param scale_x: Scaling factor in x direction
    :param scale_y: Scaling factor in y direction
    :param scale_z: Scaling factor in z direction
    :return: Scaling matrix
    """
    return np.array([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, scale_z]], dtype=float)


def normalize(vec):
    """
    Normalize a vector.

    :param vec: Vector
    :return: Normalized vector
    """
    norm = np.linalg.norm(vec, axis=(-1))
    if not np.all(norm):
        raise ValueError("Vector length is 0.")
    if vec.ndim > 1:
        return vec / norm[..., np.newaxis]
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
        raise ValueError("One or more points describing the plane are identical.")

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
        raise ValueError("One or both vectors have zero length.")

    return math.isclose(np.dot(vec_u, vec_v), 0, abs_tol=tolerance)


def is_orthogonal_matrix(a: np.ndarray, atol=1e-9) -> bool:
    """
    Check if ndarray is orthogonal matrix in the last two dimensions.

    :param a: Matrix to check
    :param atol: atol to pass onto np.allclose
    :return: True if last 2 dimensions of a are orthogonal
    """
    return np.allclose(np.matmul(a, a.swapaxes(-1, -2)), np.eye(a.shape[-1]), atol=atol)


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
        raise ValueError("Invalid transformation")

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


# local coordinate system class --------------------------------------------------------


class LocalCoordinateSystem:
    """Defines a local cartesian coordinate system in 3d."""

    # TODO: Add option to ctors to create time dependent lcs
    def __init__(
        self,
        basis: Union[xr.DataArray, np.ndarray, List[List]] = None,
        origin: Union[xr.DataArray, np.ndarray, List] = None,
        time: pd.DatetimeIndex = None,
        construction_checks: bool = True,
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
        if construction_checks:
            if basis is None:
                basis = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

            if origin is None:
                origin = np.array([0, 0, 0])

            if time is not None and not isinstance(time, pd.DatetimeIndex):
                raise TypeError("time must be an instance of pandas.DateTimeIndex")

            if not isinstance(basis, xr.DataArray):
                if not isinstance(basis, np.ndarray):
                    basis = np.array(basis)
                time_basis = None
                if basis.ndim == 3:
                    time_basis = time

                basis = ut.xr_3d_matrix(basis, time_basis)
            else:
                # TODO: Test if xarray has correct format
                pass

            if not isinstance(origin, xr.DataArray):
                if not isinstance(origin, np.ndarray):
                    origin = np.array(origin)
                time_origin = None
                if origin.ndim == 2:
                    time_origin = time
                origin = ut.xr_3d_vector(origin, time_origin)
            else:
                # TODO: Test if xarray has correct format
                pass

            basis = xr.apply_ufunc(
                normalize,
                basis,
                input_core_dims=[["c", "v"]],
                output_core_dims=[["c", "v"]],
            )

            # unify time axis
            if ("time" in basis.coords) and ("time" in origin.coords):
                if not np.all(basis.time.data == origin.time.data):
                    time_union = ut.get_time_union([basis.time, origin.time])
                    basis = ut.xr_interp_orientation_in_time(basis, time_union)
                    origin = ut.xr_interp_coodinates_in_time(origin, time_union)

            # vectorize test if orthogonal
            if not ut.xr_is_orthogonal_matrix(basis, dims=["c", "v"]):
                raise ValueError("Basis vectors must be orthogonal")

        origin.name = "origin"
        basis.name = "basis"

        self._dataset = xr.merge([origin, basis], join="exact")

    def __repr__(self):
        """Give __repr_ output in xarray format."""
        return self._dataset.__repr__().replace(
            "<xarray.Dataset", "<LocalCoordinateSystem"
        )

    def __add__(self, rhs_cs: "LocalCoordinateSystem") -> "LocalCoordinateSystem":
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

        If the left-hand side system has a time component, the data of the right-hand
        side system will be interpolated to the same times, before the previously shown
        operations are performed per point in time. In case, that the left-hand side
        system has no time component, but the right-hand side does, the resulting system
        has the same time components as the right-hand side system.

        :param rhs_cs: Right-hand side coordinate system
        :return: Resulting coordinate system.
        """
        if self.time is not None:
            rhs_cs = rhs_cs.interp_time(self.time)

        basis = ut.xr_matmul(rhs_cs.basis, self.basis, dims_a=["c", "v"])
        origin = (
            ut.xr_matmul(rhs_cs.basis, self.origin, ["c", "v"], ["c"]) + rhs_cs.origin
        )
        return LocalCoordinateSystem(basis, origin)

    def __sub__(self, rhs_cs: "LocalCoordinateSystem") -> "LocalCoordinateSystem":
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

        If the left-hand side system has a time component, the data of the right-hand
        side system will be interpolated to the same times, before the previously shown
        operations are performed per point in time. In case, that the left-hand side
        system has no time component, but the right-hand side does, the resulting system
        has the same time components as the right-hand side system.

        :param rhs_cs: Right-hand side coordinate system
        :return: Resulting coordinate system.
        """
        rhs_cs_inv = rhs_cs.invert()
        return self + rhs_cs_inv

    @classmethod
    def construct_from_euler(
        cls, sequence, angles, degrees=False, origin=None, time=None
    ) -> "LocalCoordinateSystem":
        """
        Construct a cartesian coordinate system from an euler sequence.

        This function uses scipy.spatial.transform.Rotation.from_euler method to define
        the coordinate systems orientation. Take a look at it's documentation, if some
        information is missing here. The related parameter docs are a copy of the scipy
        documentation.
        :param sequence: Specifies sequence of axes for rotations. Up to 3 characters
        belonging to the set {‘X’, ‘Y’, ‘Z’} for intrinsic rotations, or {‘x’, ‘y’, ‘z’}
        for extrinsic rotations. Extrinsic and intrinsic rotations cannot be mixed in
        one function call.
        :param angles: Euler angles specified in radians (degrees is False) or degrees
        (degrees is True). For a single character seq, angles can be:
        - a single value
        - array_like with shape (N,), where each angle[i] corresponds to a single
          rotation
        - array_like with shape (N, 1), where each angle[i, 0] corresponds to a single
          rotation
        For 2- and 3-character wide seq, angles can be:
        - array_like with shape (W,) where W is the width of seq, which corresponds to a
          single rotation with W axes
        - array_like with shape (N, W) where each angle[i] corresponds to a sequence of
          Euler angles describing a single rotation
        :param degrees: If True, then the given angles are assumed to be in degrees.
        Default is False.
        :param origin: Position of the origin
        :return: Local coordinate system
        :return:
        """
        orientation = Rot.from_euler(sequence, angles, degrees).as_matrix()
        return cls(orientation, origin=origin, time=time)

    @classmethod
    def construct_from_orientation(
        cls, orientation, origin=None, time=None
    ) -> "LocalCoordinateSystem":
        """
        Construct a cartesian coordinate system from orientation matrix.

        :param orientation: Orthogonal transformation matrix
        :param origin: Position of the origin
        :return: Local coordinate system
        """
        return cls(orientation, origin=origin, time=time)

    @classmethod
    def construct_from_xyz(
        cls, vec_x, vec_y, vec_z, origin=None, time=None
    ) -> "LocalCoordinateSystem":
        """
        Construct a cartesian coordinate system from 3 basis vectors.

        :param vec_x: Vector defining the x-axis
        :param vec_y: Vector defining the y-axis
        :param vec_z: Vector defining the z-axis
        :param origin: Position of the origin
        :return: Cartesian coordinate system
        """
        basis = np.transpose([vec_x, vec_y, vec_z])
        return cls(basis, origin=origin, time=time)

    @classmethod
    def construct_from_xy_and_orientation(
        cls, vec_x, vec_y, positive_orientation=True, origin=None, time=None
    ) -> "LocalCoordinateSystem":
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
        return cls(basis, origin=origin, time=time)

    @classmethod
    def construct_from_yz_and_orientation(
        cls, vec_y, vec_z, positive_orientation=True, origin=None, time=None
    ) -> "LocalCoordinateSystem":
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
        return cls(basis, origin=origin, time=time)

    @classmethod
    def construct_from_xz_and_orientation(
        cls, vec_x, vec_z, positive_orientation=True, origin=None, time=None
    ) -> "LocalCoordinateSystem":
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
        return cls(basis, origin=origin, time=time)

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
    def basis(self) -> xr.DataArray:
        """
        Get the normalizes basis as matrix of 3 column vectors.

        This function is identical to the 'orientation' function.

        :return: Basis of the coordinate system
        """
        return self.dataset.basis

    @property
    def orientation(self) -> xr.DataArray:
        """
        Get the coordinate systems orientation matrix.

        This function is identical to the 'basis' function.

        :return: Orientation matrix
        """
        return self.dataset.basis

    @property
    def origin(self) -> xr.DataArray:
        """
        Get the coordinate systems origin.

        This function is identical to the 'location' function.

        :return: Origin of the coordinate system
        """
        return self.dataset.origin

    @property
    def location(self) -> xr.DataArray:
        """
        Get the coordinate systems location.

        This function is identical to the 'origin' function.

        :return: Location of the coordinate system.
        """
        return self.dataset.origin

    @property
    def time(self) -> pd.DatetimeIndex:
        """
        Get the time union of the local coordinate system (or None if system is static).

        :return: DateTimeIndex-like time union
        """
        if "time" in self._dataset.coords:
            return pd.DatetimeIndex(self._dataset.time.data)
        return None

    @property
    def dataset(self) -> xr.Dataset:
        """
        Get the underlying xarray.Dataset with ordered dimensions.

        :return: xarray Dataset with origin and basis as DataVariables.
        """
        return self._dataset.transpose(..., "c", "v")

    def interp_time(
        self, time: Union[pd.DatetimeIndex, List[pd.Timestamp], "LocalCoordinateSystem"]
    ) -> "LocalCoordinateSystem":
        """
        Interpolates the data in time.

        :param time: Series of times.
        :return: Coordinate system with interpolated data
        """
        if isinstance(time, LocalCoordinateSystem):
            time = time.time

        try:
            time = pd.DatetimeIndex(time)
        except Exception as err:
            print(
                "Unable to convert input argument to pd.DatetimeIndex. "
                + "If passing single values convert to list first (like [pd.Timestamp])"
            )
            raise err
        basis = ut.xr_interp_orientation_in_time(self.basis, time)
        origin = ut.xr_interp_coodinates_in_time(self.origin, time)

        return LocalCoordinateSystem(basis, origin)

    def invert(self) -> "LocalCoordinateSystem":
        """
        Get a local coordinate system defining the parent in the child system.

        Inverse is defined as basis_new=basis.T, origin_new=basis.T*(-origin)
        :return: Inverted coordinate system.
        """
        basis = ut.xr_transpose_matrix_data(self.basis, dim1="c", dim2="v")
        origin = ut.xr_matmul(
            self.basis, -self.origin, dims_a=["c", "v"], dims_b=["c"], trans_a=True
        )
        return LocalCoordinateSystem(basis, origin)


# coordinate system manager class ------------------------------------------------------


class CoordinateSystemManager:
    """Manages multiple coordinate systems and the transformations between them."""

    @dataclass
    class CoordinateSystemData:
        """Class that stores data and the coordinate system, the data is assigned to."""

        coordinate_system_name: Hashable
        data: xr.DataArray

    def __init__(self, root_coordinate_system_name: Hashable):
        """
        Construct a coordinate system manager.

        :param root_coordinate_system_name: Name of the root coordinate system. This can
        be any hashable type, but it is recommended to use strings.
        """
        self._graph = nx.DiGraph()
        self._data = dict()
        self._add_coordinate_system_node(root_coordinate_system_name)

    def _add_coordinate_system_node(self, coordinate_system_name):
        self._check_new_coordinate_system_name(coordinate_system_name)
        self._graph.add_node(coordinate_system_name, data=[])

    def _add_edges(
        self, node_from: Hashable, node_to: Hashable, lcs: LocalCoordinateSystem
    ):
        """
        Add an edge to the internal graph.

        :param node_from: Start node of the edge
        :param node_to: End node of the edge
        :param lcs: Local coordinate system
        :return: ---
        """
        self._graph.add_edge(node_from, node_to, lcs=lcs)
        self._graph.add_edge(node_to, node_from, lcs=lcs.invert())

    def _check_coordinate_system_exists(self, coordinate_system_name: Hashable):
        """
        Raise an exception if the specified coordinate system does not exist.

        :param coordinate_system_name: Name of the coordinate system, that should be
        checked.
        :return: ---
        """
        if not self.has_coordinate_system(coordinate_system_name):
            raise ValueError(
                "There is no coordinate system with name " + str(coordinate_system_name)
            )

    def _check_new_coordinate_system_name(self, coordinate_system_name: Hashable):
        """
        Raise an exception if the new coordinate systems' name is invalid.

        :param coordinate_system_name: Name of the new coordinate system, that should be
        checked.
        :return: ---
        """
        if not isinstance(coordinate_system_name, cl.Hashable):
            raise TypeError("The coordinate system name must be a hashable type.")
        if self.has_coordinate_system(coordinate_system_name):
            raise ValueError(
                "There already is a coordinate system with name "
                + str(coordinate_system_name)
            )

    def add_coordinate_system(
        self,
        coordinate_system_name: Hashable,
        reference_system_name: Hashable,
        local_coordinate_system: LocalCoordinateSystem,
    ):
        """
        Add a coordinate system to the coordinate system manager.

        :param coordinate_system_name: Name of the new coordinate system. This can be
        any hashable type, but it is recommended to use strings.
        :param reference_system_name: Name of the parent system. This must have been
        already added.
        :param local_coordinate_system: An instance of
        weldx.transformations.LocalCoordinateSystem that describes how the new
        coordinate system is oriented in its parent system.
        :return: ---
        """
        if not isinstance(local_coordinate_system, LocalCoordinateSystem):
            raise TypeError(
                "'local_coordinate_system' must be an instance of "
                + "weldx.transformations.LocalCoordinateSystem"
            )
        self._check_coordinate_system_exists(reference_system_name)

        self._add_coordinate_system_node(coordinate_system_name)
        self._add_edges(
            coordinate_system_name, reference_system_name, local_coordinate_system
        )

    def assign_data(
        self, data: xr.DataArray, data_name: Hashable, coordinate_system_name: Hashable
    ):
        """
        Assign spatial data to a coordinate system.

        :param data: Spatial data
        :param data_name: Name of the data. Can be any hashable type, but strings are
        recommended.
        :param coordinate_system_name: Name of the coordinate system the data should be
        assigned to.
        :return: ---
        """
        # TODO: How to handle time dependent data? some things to think about:
        # - times of coordinate system and data are not equal
        # - which time is taken as reference? (probably the one of the data)
        # - what happens during cal of time interpolation functions with data? Also
        #   interpolated or not?
        if not isinstance(data_name, cl.Hashable):
            raise TypeError("The data name must be a hashable type.")
        self._check_coordinate_system_exists(coordinate_system_name)

        self._data[data_name] = self.CoordinateSystemData(coordinate_system_name, data)
        self._graph.nodes[coordinate_system_name]["data"].append(data_name)

    def get_local_coordinate_system(
        self, coordinate_system_name: Hashable, reference_system_name: Hashable
    ) -> LocalCoordinateSystem:
        """
        Get a coordinate system in relation to another reference system.

        :param coordinate_system_name: Name of the coordinate system
        :param reference_system_name: Name of the reference coordinate system
        :return: Local coordinate system
        """
        # TODO: Add time parameter
        # TODO: Treat static separately
        # TODO: What if coordinate system and reference are the same?
        self._check_coordinate_system_exists(coordinate_system_name)
        self._check_coordinate_system_exists(reference_system_name)

        path = nx.shortest_path(
            self.graph, coordinate_system_name, reference_system_name
        )
        lcs = self.graph.edges[path[0], path[1]]["lcs"]
        length_path = len(path) - 1
        if length_path > 1:
            for i in np.arange(1, length_path):
                lcs = lcs + self.graph.edges[path[i], path[i + 1]]["lcs"]
        return lcs

    def has_coordinate_system(self, coordinate_system_name: Hashable) -> bool:
        """
        Return 'True' if a coordinate system with specified name is part of the class.

        :param coordinate_system_name: Name of the coordinate system, that should be
        checked.
        :return: 'True' or 'False'
        """
        return coordinate_system_name in self._graph.nodes

    def has_data(self, coordinate_system_name: Hashable, data_name: Hashable) -> bool:
        """
        Return 'True' if the desired coordinate system owns the specified data.

        :param coordinate_system_name: Name of the coordinate system
        :param data_name: Name of the data
        :return: 'True' or 'False'
        """
        return data_name in self._graph.nodes[coordinate_system_name]["data"]

    def get_data(self, data_name, target_coordinate_system_name=None):
        """
        Get the specified data, optionally transformed into any coordinate system.

        :param data_name: Name of the data
        :param target_coordinate_system_name: Name of the target coordinate system. If
        it is not None or not identical to the owning coordinate system name, the data
        will be transformed to the desired system.
        :return: (Transformed) data
        """
        data_struct = self._data[data_name]
        if (
            target_coordinate_system_name is None
            or target_coordinate_system_name == data_struct.coordinate_system_name
        ):
            return data_struct.data
        else:
            return self.transform_data(
                data_struct.data,
                data_struct.coordinate_system_name,
                target_coordinate_system_name,
            )

    def transform_data(
        self,
        data: Union[xr.DataArray, np.ndarray, List],
        source_coordinate_system_name: Hashable,
        target_coordinate_system_name: Hashable,
    ):
        """
        Transform spatial data from one coordinate system to another.

        :param data: Pointcloud input as array-like with cartesian x,y,z-data stored in
        the last dimension. When using xarray objects, the vector dimension is expected
        to be named "c" and have coordinates "x","y","z"
        :param source_coordinate_system_name: Name of the coordinate system the data is
        defined in
        :param target_coordinate_system_name: Name of the coordinate system the data
        should be transformed to
        :return: Transformed data
        """
        lcs = self.get_local_coordinate_system(
            source_coordinate_system_name, target_coordinate_system_name
        )
        if isinstance(data, xr.DataArray):
            mul = ut.xr_matmul(
                lcs.orientation, data, dims_a=["c", "v"], dims_b=["c"], dims_out=["c"]
            )
            return mul + lcs.location
        else:
            data = ut.to_float_array(data)
            rotation = lcs.orientation.data
            translation = lcs.location.data
            return ut.mat_vec_mul(rotation, data) + translation

    @property
    def graph(self) -> nx.DiGraph:
        """
        Get the internal graph.

        :return: networkx.DiGraph
        """
        return self._graph

    @property
    def number_of_coordinate_systems(self) -> int:
        """
        Get the number of coordinate systems inside the coordinate system manager.

        :return: Number of coordinate systems
        """
        return self._graph.number_of_nodes()

    def neighbors(self, coordinate_system_name: Hashable) -> List:
        """
        Get a list of neighbors of a certain coordinate system.

        :param coordinate_system_name: Name of the coordinate system
        :return: List of neighbors
        """
        self._check_coordinate_system_exists(coordinate_system_name)
        return list(self._graph.neighbors(coordinate_system_name))

    def number_of_neighbors(self, coordinate_system_name) -> int:
        """
        Get the number of neighbors  of a certain coordinate system.

        :param coordinate_system_name: Name of the coordinate system
        :return: Number of neighbors
        """
        return len(self.neighbors(coordinate_system_name))

    def is_neighbor_of(
        self, coordinate_system_name_0: Hashable, coordinate_system_name_1: Hashable
    ) -> bool:
        """
        Get a boolean result, specifying if 2 coordinate systems are neighbors.

        :param coordinate_system_name_0: Name of the first coordinate system
        :param coordinate_system_name_1: Name of the second coordinate system
        :return:
        """
        self._check_coordinate_system_exists(coordinate_system_name_0)
        self._check_coordinate_system_exists(coordinate_system_name_1)

        return coordinate_system_name_1 in self.neighbors(coordinate_system_name_0)

    def interp_time(
        self,
        time: Union[pd.DatetimeIndex, List[pd.Timestamp], "LocalCoordinateSystem"],
        inplace: bool = False,
    ) -> "CoordinateSystemManager":
        """
        Interpolates the coordinate systems in time.

        :param time: Time data.
        :param inplace: If 'True' the interpolation is performed in place, otherwise a
        new instance is returned.
        :return: Coordinate system manager with interpolated data
        """
        if inplace:
            for edge in self._graph.edges:
                self._graph.edges[edge]["lcs"] = self._graph.edges[edge][
                    "lcs"
                ].interp_time(time)
            return self
        else:
            return deepcopy(self).interp_time(time, inplace=True)

    def time_union(self, list_of_edges: List = None) -> pd.DatetimeIndex:
        """
        Get the time union of all or selected local coordinate systems.

        :param list_of_edges: If not None, the union is only calculated from the
        specified edges
        :return: Time union
        """
        edges = self.graph.edges
        if list_of_edges is None:
            list_of_edges = edges
        time_union = None
        for edge in list_of_edges:
            time_edge = edges[edge]["lcs"].time
            if time_union is None:
                time_union = time_edge
            elif time_edge is not None:
                time_union = time_union.union(time_edge)

        return time_union
