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
    """Create a rotation matrix that rotates around the x-axis.

    Parameters
    ----------
    angle :
        Rotation angle

    Returns
    -------
    np.ndarray
        Rotation matrix

    """
    return Rot.from_euler("x", angle).as_matrix()


def rotation_matrix_y(angle):
    """Create a rotation matrix that rotates around the y-axis.

    Parameters
    ----------
    angle :
        Rotation angle

    Returns
    -------
    np.ndarray
        Rotation matrix

    """
    return Rot.from_euler("y", angle).as_matrix()


def rotation_matrix_z(angle) -> np.ndarray:
    """Create a rotation matrix that rotates around the z-axis.

    Parameters
    ----------
    angle :
        Rotation angle

    Returns
    -------
    np.ndarray
        Rotation matrix

    """
    return Rot.from_euler("z", angle).as_matrix()


def scale_matrix(scale_x, scale_y, scale_z) -> np.ndarray:
    """Return a scaling matrix.

    Parameters
    ----------
    scale_x :
        Scaling factor in x direction
    scale_y :
        Scaling factor in y direction
    scale_z :
        Scaling factor in z direction

    Returns
    -------
    np.ndarray
        Scaling matrix

    """
    return np.diag([scale_x, scale_y, scale_z]).astype(float)


def normalize(a):
    """Normalize (l2 norm) an ndarray along the last dimension.

    Parameters
    ----------
    a :
        data in ndarray

    Returns
    -------
    np.ndarray
        Normalized ndarray

    """
    norm = np.linalg.norm(a, axis=(-1), keepdims=True)
    if not np.all(norm):
        raise ValueError("Length 0 encountered during normalization.")
    return a / norm


def orientation_point_plane_containing_origin(point, p_a, p_b):
    """Determine a points orientation relative to a plane containing the origin.

    The side is defined by the winding order of the triangle 'origin - A -
    B'. When looking at it from the left-hand side, the ordering is clockwise
    and counter-clockwise when looking from the right-hand side.

    The function returns 1 if the point lies left of the plane, -1 if it is
    on the right and 0 if it lies on the plane.

    Note, that this function is not appropriate to check if a point lies on
    a plane since it has no tolerance to compensate for numerical errors.

    Additional note: The points A and B can also been considered as two
    vectors spanning the plane.

    Parameters
    ----------
    point :
        Point
    p_a :
        Second point of the triangle 'origin - A - B'.
    p_b :
        Third point of the triangle 'origin - A - B'.

    Returns
    -------
    int
        1, -1 or 0 (see description)

    """
    if (
        math.isclose(np.linalg.norm(p_a), 0)
        or math.isclose(np.linalg.norm(p_b), 0)
        or math.isclose(np.linalg.norm(p_b - p_a), 0)
    ):
        raise ValueError("One or more points describing the plane are identical.")

    return np.sign(np.linalg.det([p_a, p_b, point]))


def orientation_point_plane(point, p_a, p_b, p_c):
    """Determine a points orientation relative to an arbitrary plane.

    The side is defined by the winding order of the triangle 'A - B - C'.
    When looking at it from the left-hand side, the ordering is clockwise
    and counter-clockwise when looking from the right-hand side.

    The function returns 1 if the point lies left of the plane, -1 if it is
    on the right and 0 if it lies on the plane.

    Note, that this function is not appropriate to check if a point lies on
    a plane since it has no tolerance to compensate for numerical errors.

    Parameters
    ----------
    point :
        Point
    p_a :
        First point of the triangle 'A - B - C'.
    p_b :
        Second point of the triangle 'A - B - C'.
    p_c :
        Third point of the triangle 'A - B - C'.

    Returns
    -------
    int
        1, -1 or 0 (see description)

    """
    vec_a_b = p_b - p_a
    vec_a_c = p_c - p_a
    vec_a_point = point - p_a
    return orientation_point_plane_containing_origin(vec_a_point, vec_a_b, vec_a_c)


def is_orthogonal(vec_u, vec_v, tolerance=1e-9):
    """Check if vectors are orthogonal.

    Parameters
    ----------
    vec_u :
        First vector
    vec_v :
        Second vector
    tolerance :
        Numerical tolerance (Default value = 1e-9)

    Returns
    -------
    bool
        True or False

    """
    if math.isclose(np.dot(vec_u, vec_u), 0) or math.isclose(np.dot(vec_v, vec_v), 0):
        raise ValueError("One or both vectors have zero length.")

    return math.isclose(np.dot(vec_u, vec_v), 0, abs_tol=tolerance)


def is_orthogonal_matrix(a: np.ndarray, atol=1e-9) -> bool:
    """Check if ndarray is orthogonal matrix in the last two dimensions.

    Parameters
    ----------
    a :
        Matrix to check
    atol :
        atol to pass onto np.allclose (Default value = 1e-9)

    Returns
    -------
    bool
        True if last 2 dimensions of a are orthogonal

    """
    return np.allclose(np.matmul(a, a.swapaxes(-1, -2)), np.eye(a.shape[-1]), atol=atol)


def point_left_of_line(point, line_start, line_end):
    """Determine if a point lies left of a line.

    Returns 1 if the point is left of the line and -1 if it is to the right.
    If the point is located on the line, this function returns 0.

    Parameters
    ----------
    point :
        Point
    line_start :
        Starting point of the line
    line_end :
        End point of the line

    Returns
    -------
    int
        1,-1 or 0 (see description)

    """
    vec_line_start_end = line_end - line_start
    vec_line_start_point = point - line_start
    return vector_points_to_left_of_vector(vec_line_start_point, vec_line_start_end)


def reflection_sign(matrix):
    """Get a sign indicating if the transformation is a reflection.

    Returns -1 if the transformation contains a reflection and 1 if not.

    Parameters
    ----------
    matrix :
        Transformation matrix

    Returns
    -------
    int
        1 or -1 (see description)

    """
    sign = int(np.sign(np.linalg.det(matrix)))

    if sign == 0:
        raise ValueError("Invalid transformation")

    return sign


def vector_points_to_left_of_vector(vector, vector_reference):
    """Determine if a vector points to the left of another vector.

    Returns 1 if the vector points to the left of the reference vector and
    -1 if it points to the right. In case both vectors point into the same
    or the opposite directions, this function returns 0.

    Parameters
    ----------
    vector :
        Vector
    vector_reference :
        Reference vector

    Returns
    -------
    int
        1,-1 or 0 (see description)

    """
    return int(np.sign(np.linalg.det([vector_reference, vector])))


# local coordinate system class --------------------------------------------------------


class LocalCoordinateSystem:
    """Defines a local cartesian coordinate system in 3d."""

    def __init__(
        self,
        orientation: Union[xr.DataArray, np.ndarray, List[List], Rot] = None,
        coordinates: Union[xr.DataArray, np.ndarray, List] = None,
        time: pd.DatetimeIndex = None,
        construction_checks: bool = True,
    ):
        """Construct a cartesian coordinate system.

        Parameters
        ----------
        orientation :
            Matrix of 3 orthogonal column vectors which represent
            the coordinate systems orientation. Keep in mind, that the columns of the
            corresponding orientation matrix is equal to the normalized orientation
            vectors. So each orthogonal transformation matrix can also be
            provided as orientation.
            Passing a scipy.spatial.transform.Rotation object is also supported.
        coordinates :
            Coordinates of the origin
        time :
            Time data for time dependent coordinate systems
        construction_checks :
            If 'True', the validity of the data will be verified

        Returns
        -------
        LocalCoordinateSystem
            Cartesian coordinate system

        """
        if isinstance(orientation, Rot):
            orientation = orientation.as_matrix()

        if construction_checks:
            if orientation is None:
                orientation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

            if coordinates is None:
                coordinates = np.array([0, 0, 0])

            if time is not None:
                try:
                    time = pd.DatetimeIndex(time)
                except Exception as err:
                    print(
                        "Unable to convert input argument to pd.DatetimeIndex. "
                        + "If passing single values convert to list first (like ["
                        "pd.Timestamp])"
                    )
                    raise err

            if not isinstance(orientation, xr.DataArray):
                if not isinstance(orientation, np.ndarray):
                    orientation = np.array(orientation)
                time_orientation = None
                if orientation.ndim == 3:
                    time_orientation = time

                orientation = ut.xr_3d_matrix(orientation, time_orientation)
            else:
                # TODO: Test if xarray has correct format
                pass

            if not isinstance(coordinates, xr.DataArray):
                if not isinstance(coordinates, np.ndarray):
                    coordinates = np.array(coordinates)
                time_coordinates = None
                if coordinates.ndim == 2:
                    time_coordinates = time
                coordinates = ut.xr_3d_vector(coordinates, time_coordinates)
            else:
                # TODO: Test if xarray has correct format
                pass

            orientation = xr.apply_ufunc(
                normalize,
                orientation,
                input_core_dims=[["c"]],
                output_core_dims=[["c"]],
            )

            # unify time axis
            if ("time" in orientation.coords) and ("time" in coordinates.coords):
                if not np.all(orientation.time.data == coordinates.time.data):
                    time_union = ut.get_time_union([orientation.time, coordinates.time])
                    orientation = ut.xr_interp_orientation_in_time(
                        orientation, time_union
                    )
                    coordinates = ut.xr_interp_coordinates_in_time(
                        coordinates, time_union
                    )

            # vectorize test if orthogonal
            if not ut.xr_is_orthogonal_matrix(orientation, dims=["c", "v"]):
                raise ValueError("Orientation vectors must be orthogonal")

        coordinates.name = "coordinates"
        orientation.name = "orientation"

        self._dataset = xr.merge([coordinates, orientation], join="exact")

    def __repr__(self):
        """Give __repr_ output in xarray format."""
        return self._dataset.__repr__().replace(
            "<xarray.Dataset", "<LocalCoordinateSystem"
        )

    def __add__(self, rhs_cs: "LocalCoordinateSystem") -> "LocalCoordinateSystem":
        """Add 2 coordinate systems.

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

        Parameters
        ----------
        rhs_cs :
            Right-hand side coordinate system

        Returns
        -------
        LocalCoordinateSystem
            Resulting coordinate system.

        """
        rhs_cs = rhs_cs.interp_time(self.time)

        orientation = ut.xr_matmul(
            rhs_cs.orientation, self.orientation, dims_a=["c", "v"]
        )
        coordinates = (
            ut.xr_matmul(rhs_cs.orientation, self.coordinates, ["c", "v"], ["c"])
            + rhs_cs.coordinates
        )
        return LocalCoordinateSystem(orientation, coordinates)

    def __sub__(self, rhs_cs: "LocalCoordinateSystem") -> "LocalCoordinateSystem":
        """Subtract 2 coordinate systems.

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

        Parameters
        ----------
        rhs_cs :
            Right-hand side coordinate system

        Returns
        -------
        LocalCoordinateSystem
            Resulting coordinate system.

        """
        rhs_cs_inv = rhs_cs.invert()
        return self + rhs_cs_inv

    @classmethod
    def construct_from_euler(
        cls, sequence, angles, degrees=False, coordinates=None, time=None
    ) -> "LocalCoordinateSystem":
        """Construct a local coordinate system from an euler sequence.

        This function uses scipy.spatial.transform.Rotation.from_euler method to define
        the coordinate systems orientation. Take a look at it's documentation, if some
        information is missing here. The related parameter docs are a copy of the scipy
        documentation.

        Parameters
        ----------
        sequence :
            Specifies sequence of axes for rotations. Up to 3 characters
            belonging to the set {‘X’, ‘Y’, ‘Z’} for intrinsic rotations,
            or {‘x’, ‘y’, ‘z’} for extrinsic rotations.
            Extrinsic and intrinsic rotations cannot be mixed in one function call.
        angles :
            Euler angles specified in radians (degrees is False) or degrees
            (degrees is True). For a single character seq, angles can be:
            - a single value
            - array_like with shape (N,), where each angle[i] corresponds to a single
            rotation
            - array_like with shape (N, 1), where each angle[i, 0] corresponds to a
            single rotation
            For 2- and 3-character wide seq, angles can be:
            - array_like with shape (W,) where W is the width of seq, which corresponds
            to a single rotation with W axes
            - array_like with shape (N, W) where each angle[i] corresponds to a sequence
            of Euler angles describing a single rotation
        degrees :
            If True, then the given angles are assumed to be in degrees.
            Default is False.
        coordinates :
            Coordinates of the origin (Default value = None)
        time :
            Time data for time dependent coordinate systems (Default value = None)

        Returns
        -------
        LocalCoordinateSystem
            Local coordinate system

        """
        orientation = Rot.from_euler(sequence, angles, degrees)
        return cls(orientation, coordinates=coordinates, time=time)

    @classmethod
    def construct_from_orientation(
        cls, orientation, coordinates=None, time=None
    ) -> "LocalCoordinateSystem":
        """Construct a local coordinate system from orientation matrix.

        Parameters
        ----------
        orientation :
            Orthogonal transformation matrix
        coordinates :
            Coordinates of the origin (Default value = None)
        time :
            Time data for time dependent coordinate systems (Default value = None)

        Returns
        -------
        LocalCoordinateSystem
            Local coordinate system

        """
        return cls(orientation, coordinates=coordinates, time=time)

    @classmethod
    def construct_from_xyz(
        cls, vec_x, vec_y, vec_z, coordinates=None, time=None
    ) -> "LocalCoordinateSystem":
        """Construct a local coordinate system from 3 vectors defining the orientation.

        Parameters
        ----------
        vec_x :
            Vector defining the x-axis
        vec_y :
            Vector defining the y-axis
        vec_z :
            Vector defining the z-axis
        coordinates :
            Coordinates of the origin (Default value = None)
        time :
            Time data for time dependent coordinate systems (Default value = None)

        Returns
        -------
        LocalCoordinateSystem
            Local coordinate system

        """
        orientation = np.transpose([vec_x, vec_y, vec_z])
        return cls(orientation, coordinates=coordinates, time=time)

    @classmethod
    def construct_from_xy_and_orientation(
        cls, vec_x, vec_y, positive_orientation=True, coordinates=None, time=None
    ) -> "LocalCoordinateSystem":
        """Construct a coordinate system from 2 vectors and an orientation.

        Parameters
        ----------
        vec_x :
            Vector defining the x-axis
        vec_y :
            Vector defining the y-axis
        positive_orientation :
            Set to True if the orientation should
            be positive and to False if not (Default value = True)
        coordinates :
            Coordinates of the origin (Default value = None)
        time :
            Time data for time dependent coordinate systems (Default value = None)

        Returns
        -------
        LocalCoordinateSystem
            Local coordinate system

        """
        vec_z = cls._calculate_orthogonal_axis(vec_x, vec_y) * cls._sign_orientation(
            positive_orientation
        )

        orientation = np.transpose([vec_x, vec_y, vec_z])
        return cls(orientation, coordinates=coordinates, time=time)

    @classmethod
    def construct_from_yz_and_orientation(
        cls, vec_y, vec_z, positive_orientation=True, coordinates=None, time=None
    ) -> "LocalCoordinateSystem":
        """Construct a coordinate system from 2 vectors and an orientation.

        Parameters
        ----------
        vec_y :
            Vector defining the y-axis
        vec_z :
            Vector defining the z-axis
        positive_orientation :
            Set to True if the orientation should
            be positive and to False if not (Default value = True)
        coordinates :
            Coordinates of the origin (Default value = None)
        time :
            Time data for time dependent coordinate systems (Default value = None)

        Returns
        -------
        LocalCoordinateSystem
            Local coordinate system

        """
        vec_x = cls._calculate_orthogonal_axis(vec_y, vec_z) * cls._sign_orientation(
            positive_orientation
        )

        orientation = np.transpose(np.array([vec_x, vec_y, vec_z]))
        return cls(orientation, coordinates=coordinates, time=time)

    @classmethod
    def construct_from_xz_and_orientation(
        cls, vec_x, vec_z, positive_orientation=True, coordinates=None, time=None
    ) -> "LocalCoordinateSystem":
        """Construct a coordinate system from 2 vectors and an orientation.

        Parameters
        ----------
        vec_x :
            Vector defining the x-axis
        vec_z :
            Vector defining the z-axis
        positive_orientation :
            Set to True if the orientation should
            be positive and to False if not (Default value = True)
        coordinates :
            Coordinates of the origin (Default value = None)
        time :
            Time data for time dependent coordinate systems (Default value = None)

        Returns
        -------
        LocalCoordinateSystem
            Local coordinate system

        """
        vec_y = cls._calculate_orthogonal_axis(vec_z, vec_x) * cls._sign_orientation(
            positive_orientation
        )

        orientation = np.transpose([vec_x, vec_y, vec_z])
        return cls(orientation, coordinates=coordinates, time=time)

    @staticmethod
    def _sign_orientation(positive_orientation):
        """Get -1 or 1 depending on the coordinate systems orientation.

        Parameters
        ----------
        positive_orientation :
            Set to True if the orientation should
            be positive and to False if not

        Returns
        -------
        int
            1 if the coordinate system has positive orientation,
            -1 otherwise

        """
        if positive_orientation:
            return 1
        return -1

    @staticmethod
    def _calculate_orthogonal_axis(a_0, a_1):
        """Calculate an axis which is orthogonal to two other axes.

        The calculated axis has a positive orientation towards the other 2
        axes.

        Parameters
        ----------
        a_0 :
            First axis
        a_1 :
            Second axis

        Returns
        -------
        np.ndarray
            Orthogonal axis

        """
        return np.cross(a_0, a_1)

    @property
    def orientation(self) -> xr.DataArray:
        """Get the coordinate systems orientation matrix.

        Returns
        -------
        xr.DataArray
            Orientation matrix

        """
        return self.dataset.orientation

    @property
    def coordinates(self) -> xr.DataArray:
        """Get the coordinate systems coordinates.

        Returns
        -------
        xr.DataArray
            Coordinates of the coordinate system

        """
        return self.dataset.coordinates

    @property
    def time(self) -> Union[pd.DatetimeIndex, None]:
        """Get the time union of the local coordinate system (None if system is static).

        Returns
        -------
        pd.DatetimeIndex
            DateTimeIndex-like time union

        """
        if "time" in self._dataset.coords:
            return pd.DatetimeIndex(self._dataset.time.data)
        return None

    @property
    def dataset(self) -> xr.Dataset:
        """Get the underlying xarray.Dataset with ordered dimensions.

        Returns
        -------
        xr.Dataset
            xarray Dataset with coordinates and orientation as DataVariables.

        """
        return self._dataset.transpose(..., "c", "v")

    def interp_time(
        self,
        time: Union[
            pd.DatetimeIndex, List[pd.Timestamp], "LocalCoordinateSystem", None
        ],
    ) -> "LocalCoordinateSystem":
        """Interpolates the data in time.

        Parameters
        ----------
        time :
            Series of times.
            If passing "None" no interpolation will be performed.

        Returns
        -------
        LocalCoordinateSystem
            Coordinate system with interpolated data

        """
        if time is None:
            return self

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
        orientation = ut.xr_interp_orientation_in_time(self.orientation, time)
        coordinates = ut.xr_interp_coordinates_in_time(self.coordinates, time)

        return LocalCoordinateSystem(orientation, coordinates)

    def invert(self) -> "LocalCoordinateSystem":
        """Get a local coordinate system defining the parent in the child system.

        Inverse is defined as orientation_new=orientation.T,
        coordinates_new=orientation.T*(-coordinates)

        Returns
        -------
        LocalCoordinateSystem
            Inverted coordinate system.

        """
        orientation = ut.xr_transpose_matrix_data(self.orientation, dim1="c", dim2="v")
        coordinates = ut.xr_matmul(
            self.orientation,
            -self.coordinates,
            dims_a=["c", "v"],
            dims_b=["c"],
            trans_a=True,
        )
        return LocalCoordinateSystem(orientation, coordinates)

    def as_rotation(self) -> Rot:  # pragma: no cover
        """Get a scipy.Rotation object from the coordinate system orientation.

        Returns
        -------
        scipy.spatial.transform.Rotation
            Scipy rotation object representing the orientation.

        """
        return Rot.from_matrix(self.orientation.values)

    def as_euler(
        self, seq: str = "xyz", degrees: bool = False
    ) -> np.ndarray:  # pragma: no cover
        """Return Euler angle representation of the coordinate system orientation.

        Parameters
        ----------
        seq :
            Euler rotation sequence as described in
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.as_euler.html
        degrees :
            Returned angles are in degrees if True, else they are in radians.
            Default is False.

        Returns
        -------
        numpy.ndarray
            Array of euler angles.

        """
        return self.as_rotation().as_euler(seq=seq, degrees=degrees)


# coordinate system manager class ------------------------------------------------------


class CoordinateSystemManager:
    """Manages multiple coordinate systems and the transformations between them."""

    @dataclass
    class CoordinateSystemData:
        """Class that stores data and the coordinate system, the data is assigned to."""

        coordinate_system_name: Hashable
        data: xr.DataArray

    def __init__(self, root_coordinate_system_name: Hashable):
        """Construct a coordinate system manager.

        Parameters
        ----------
        root_coordinate_system_name :
            Name of the root coordinate system. This can be any hashable type, but it is
            recommended to use strings.

        Returns
        -------
        CoordinateSystemManager

        """
        self._graph = nx.DiGraph()
        self._data = {}
        self._add_coordinate_system_node(root_coordinate_system_name)

    def __repr__(self):
        """Output representation of a CoordinateSystemManager class."""
        return (
            f"CoordinateSystemManager('graph': {self._graph!r}, 'data': {self._data!r})"
        )

    def _add_coordinate_system_node(self, coordinate_system_name):
        self._check_new_coordinate_system_name(coordinate_system_name)
        self._graph.add_node(coordinate_system_name, data=[])

    def _add_edges(
        self, node_from: Hashable, node_to: Hashable, lcs: LocalCoordinateSystem
    ):
        """Add an edge to the internal graph.

        Parameters
        ----------
        node_from :
            Start node of the edge
        node_to :
            End node of the edge
        lcs :
            Local coordinate system

        """
        self._graph.add_edge(node_from, node_to, lcs=lcs)
        self._graph.add_edge(node_to, node_from, lcs=lcs.invert())

    def _check_coordinate_system_exists(self, coordinate_system_name: Hashable):
        """Raise an exception if the specified coordinate system does not exist.

        Parameters
        ----------
        coordinate_system_name :
            Name of the coordinate system, that should be
            checked.

        """
        if not self.has_coordinate_system(coordinate_system_name):
            raise ValueError(
                "There is no coordinate system with name " + str(coordinate_system_name)
            )

    def _check_new_coordinate_system_name(self, coordinate_system_name: Hashable):
        """Raise an exception if the new coordinate systems' name is invalid.

        Parameters
        ----------
        coordinate_system_name :
            Name of the new coordinate system, that should be checked.

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
        """Add a coordinate system to the coordinate system manager.

        Parameters
        ----------
        coordinate_system_name :
            Name of the new coordinate system. This can be
            any hashable type, but it is recommended to use strings.
        reference_system_name :
            Name of the parent system. This must have been
            already added.
        local_coordinate_system :
            An instance of
            weldx.transformations.LocalCoordinateSystem that describes how the new
            coordinate system is oriented in its parent system.

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
        """Assign spatial data to a coordinate system.

        Parameters
        ----------
        data :
            Spatial data
        data_name :
            Name of the data. Can be any hashable type, but strings are
            recommended.
        coordinate_system_name :
            Name of the coordinate system the data should be
            assigned to.

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
        """Get a coordinate system in relation to another reference system.

        If any coordinate system that is involved in the coordinate transformation has
        a time dependency, the union of all coordinate systems' timestamps is
        determined. Subsequently, all coordinate systems are interpolated in time to
        match the time union before they are used in the transformation. The purpose is
        to prevent interpolation errors when rotations are involved. In result, the
        returned 'LocalCoordinateSystem' timestamps equals the time union of all
        coordinate systems that were used during the transformation.

        Without the preceding time union interpolation, multiple subsequent
        transformations can cause changes in the rotation direction and transform
        circular movements into linear ones. The reason for this is that each individual
        transformation using the 'LocalCoordinateSystem's + operator generates a new
        ''set of keyframes'' for the next transformation. Those ''keyframes'' do not
        preserve the rotational trajectory of a coordinate system attached to a rotating
        parent system. Additionally, the rotation angle between two keyframes after a
        transformation might exceed 180°. Since the employed SLERP method always uses
        the smaller rotation angle, subsequent interpolations will cause a change of the
        rotation order.

        Parameters
        ----------
        coordinate_system_name :
            Name of the coordinate system
        reference_system_name :
            Name of the reference coordinate system

        Returns
        -------
        LocalCoordinateSystem
            Local coordinate system

        """
        # TODO: Add time parameter
        # TODO: Treat static separately
        # TODO: What if coordinate system and reference are the same?
        self._check_coordinate_system_exists(coordinate_system_name)
        self._check_coordinate_system_exists(reference_system_name)

        path = nx.shortest_path(
            self.graph, coordinate_system_name, reference_system_name
        )
        path_edges = [edge for edge in zip(path[:-1], path[1:])]

        time_union = self.time_union(path_edges)

        lcs = self.graph.edges[path_edges[0]]["lcs"]

        lcs = lcs.interp_time(time_union)
        for edge in path_edges[1:]:
            lcs = lcs + self.graph.edges[edge]["lcs"].interp_time(time_union)

        return lcs

    def has_coordinate_system(self, coordinate_system_name: Hashable) -> bool:
        """Return 'True' if a coordinate system with specified name already exists.

        Parameters
        ----------
        coordinate_system_name :
            Name of the coordinate system, that should be checked.

        Returns
        -------
        bool
            'True' or 'False'

        """
        return coordinate_system_name in self._graph.nodes

    def has_data(self, coordinate_system_name: Hashable, data_name: Hashable) -> bool:
        """Return 'True' if the desired coordinate system owns the specified data.

        Parameters
        ----------
        coordinate_system_name :
            Name of the coordinate system
        data_name :
            Name of the data

        Returns
        -------
        bool
            'True' or 'False'

        """
        return data_name in self._graph.nodes[coordinate_system_name]["data"]

    def get_data(
        self, data_name, target_coordinate_system_name=None
    ) -> Union[np.ndarray, xr.DataArray]:
        """Get the specified data, optionally transformed into any coordinate system.

        Parameters
        ----------
        data_name :
            Name of the data
        target_coordinate_system_name :
            Name of the target coordinate system. If it is not None or not identical to
            the owning coordinate system name, the data will be transformed to the
            desired system. (Default value = None)

        Returns
        -------
        np.ndarray
            Transformed data

        """
        data_struct = self._data[data_name]
        if (
            target_coordinate_system_name is None
            or target_coordinate_system_name == data_struct.coordinate_system_name
        ):
            return data_struct.data

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
        """Transform spatial data from one coordinate system to another.

        Parameters
        ----------
        data :
            Point cloud input as array-like with cartesian x,y,z-data stored in
            the last dimension. When using xarray objects, the vector dimension is
            expected to be named "c" and have coordinates "x","y","z"
        source_coordinate_system_name :
            Name of the coordinate system the data is
            defined in
        target_coordinate_system_name :
            Name of the coordinate system the data
            should be transformed to

        Returns
        -------
        np.ndarray
            Transformed data

        """
        lcs = self.get_local_coordinate_system(
            source_coordinate_system_name, target_coordinate_system_name
        )
        if isinstance(data, xr.DataArray):
            mul = ut.xr_matmul(
                lcs.orientation, data, dims_a=["c", "v"], dims_b=["c"], dims_out=["c"]
            )
            return mul + lcs.coordinates

        data = ut.to_float_array(data)
        rotation = lcs.orientation.data
        translation = lcs.coordinates.data
        return ut.mat_vec_mul(rotation, data) + translation

    @property
    def graph(self) -> nx.DiGraph:
        """Get the internal graph.

        Returns
        -------
        networkx.DiGraph

        """
        return self._graph

    @property
    def number_of_coordinate_systems(self) -> int:
        """Get the number of coordinate systems inside the coordinate system manager.

        Returns
        -------
        int
            Number of coordinate systems

        """
        return self._graph.number_of_nodes()

    def neighbors(self, coordinate_system_name: Hashable) -> List:
        """Get a list of neighbors of a certain coordinate system.

        Parameters
        ----------
        coordinate_system_name :
            Name of the coordinate system

        Returns
        -------
        list
            List of neighbors

        """
        self._check_coordinate_system_exists(coordinate_system_name)
        return list(self._graph.neighbors(coordinate_system_name))

    def number_of_neighbors(self, coordinate_system_name) -> int:
        """Get the number of neighbors  of a certain coordinate system.

        Parameters
        ----------
        coordinate_system_name :
            Name of the coordinate system

        Returns
        -------
        int
            Number of neighbors

        """
        return len(self.neighbors(coordinate_system_name))

    def is_neighbor_of(
        self, coordinate_system_name_0: Hashable, coordinate_system_name_1: Hashable
    ) -> bool:
        """Get a boolean result, specifying if 2 coordinate systems are neighbors.

        Parameters
        ----------
        coordinate_system_name_0 :
            Name of the first coordinate system
        coordinate_system_name_1 :
            Name of the second coordinate system

        """
        self._check_coordinate_system_exists(coordinate_system_name_0)
        self._check_coordinate_system_exists(coordinate_system_name_1)

        return coordinate_system_name_1 in self.neighbors(coordinate_system_name_0)

    def interp_time(
        self,
        time: Union[pd.DatetimeIndex, List[pd.Timestamp], "LocalCoordinateSystem"],
        inplace: bool = False,
    ) -> "CoordinateSystemManager":
        """Interpolates the coordinate systems in time.

        Parameters
        ----------
        time :
            Time data.
        inplace :
            If 'True' the interpolation is performed in place, otherwise a
            new instance is returned. (Default value = False)

        Returns
        -------
        CoordinateSystemManager
            Coordinate system manager with interpolated data

        """
        if inplace:
            for edge in self._graph.edges:
                self._graph.edges[edge]["lcs"] = self._graph.edges[edge][
                    "lcs"
                ].interp_time(time)
            return self

        return deepcopy(self).interp_time(time, inplace=True)

    def time_union(self, list_of_edges: List = None) -> pd.DatetimeIndex:
        """Get the time union of all or selected local coordinate systems.

        Parameters
        ----------
        list_of_edges :
            If not None, the union is only calculated from the
            specified edges (Default value = None)

        Returns
        -------
        pd.DatetimeIndex
            Time union

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
