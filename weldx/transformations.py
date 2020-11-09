"""Contains methods and classes for coordinate transformations."""


import itertools
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pint
import xarray as xr
from scipy.spatial.transform import Rotation as Rot

import weldx.utility as ut
from weldx.constants import WELDX_UNIT_REGISTRY as UREG

_DEFAULT_LEN_UNIT = UREG.millimeters
_DEFAULT_ANG_UNIT = UREG.rad
__all__ = ["LocalCoordinateSystem", "CoordinateSystemManager", "WXRotation"]

# functions -----------------------------------------------------------------------


def _build_time_index(
    time: Union[pd.DatetimeIndex, pd.TimedeltaIndex, pint.Quantity] = None,
    time_ref: pd.Timestamp = None,
) -> pd.TimedeltaIndex:
    """Build time index used for xarray objects.

    Parameters
    ----------
    time:
        Datetime- or Timedelta-like time index.
    time_ref:
        Reference timestamp for Timedelta inputs.

    Returns
    -------
    pandas.TimedeltaIndex

    """
    if time is None:
        # time_ref = None
        return time, time_ref

    time = ut.to_pandas_time_index(time)

    if isinstance(time, pd.DatetimeIndex):
        if time_ref is None:
            time_ref = time[0]
        time = time - time_ref

    return time, time_ref


@UREG.wraps(None, (_DEFAULT_ANG_UNIT), strict=False)
def rotation_matrix_x(angle):
    """Create a rotation matrix that rotates around the x-axis.

    Parameters
    ----------
    angle :
        Rotation angle

    Returns
    -------
    numpy.ndarray
        Rotation matrix

    """
    return Rot.from_euler("x", angle).as_matrix()


@UREG.wraps(None, (_DEFAULT_ANG_UNIT), strict=False)
def rotation_matrix_y(angle):
    """Create a rotation matrix that rotates around the y-axis.

    Parameters
    ----------
    angle :
        Rotation angle

    Returns
    -------
    numpy.ndarray
        Rotation matrix

    """
    return Rot.from_euler("y", angle).as_matrix()


@UREG.wraps(None, (_DEFAULT_ANG_UNIT), strict=False)
def rotation_matrix_z(angle) -> np.ndarray:
    """Create a rotation matrix that rotates around the z-axis.

    Parameters
    ----------
    angle :
        Rotation angle

    Returns
    -------
    numpy.ndarray
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
    numpy.ndarray
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
    numpy.ndarray
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


# WXRotation ---------------------------------------------------------------------------


class WXRotation(Rot):
    """Wrapper for creating meta-tagged Scipy.Rotation objects."""

    @classmethod
    def from_quat(cls, quat: np.ndarray, normalized=None) -> "WXRotation":
        """Initialize from quaternions.

        scipy.spatial.transform.Rotation docs for details.
        """
        rot = super().from_quat(quat, normalized)
        rot.wx_meta = {"constructor": "from_quat"}
        return rot

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> "WXRotation":
        """Initialize from matrix.

        scipy.spatial.transform.Rotation docs for details.
        """
        rot = super().from_matrix(matrix)
        rot.wx_meta = {"constructor": "from_matrix"}
        return rot

    @classmethod
    def from_rotvec(cls, rotvec: np.ndarray) -> "WXRotation":
        """Initialize from rotation vector.

        scipy.spatial.transform.Rotation docs for details.
        """
        rot = Rot.from_rotvec(rotvec)
        rot.wx_meta = {"constructor": "from_rotvec"}
        return rot

    @classmethod
    def from_euler(cls, seq: str, angles, degrees: bool = False) -> "WXRotation":
        """Initialize from euler angles.

        scipy.spatial.transform.Rotation docs for details.
        """
        rot = Rot.from_euler(seq=seq, angles=angles, degrees=degrees)
        rot.wx_meta = {"constructor": "from_euler", "seq": seq, "degrees": degrees}
        return rot


# LocalCoordinateSystem ----------------------------------------------------------------


class LocalCoordinateSystem:
    """Defines a local cartesian coordinate system in 3d.

    Notes
    -----
    Learn how to use this class by reading the
    :doc:`Tutorial <../tutorials/transformations_01_coordinate_systems>`.

    """

    def __init__(
        self,
        orientation: Union[xr.DataArray, np.ndarray, List[List], Rot] = None,
        coordinates: Union[xr.DataArray, np.ndarray, List] = None,
        time: Union[pd.DatetimeIndex, pd.TimedeltaIndex, pint.Quantity] = None,
        time_ref: pd.Timestamp = None,
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
        time_ref :
            Reference Timestamp to use if time is Timedelta or pint.Quantity.
        construction_checks :
            If 'True', the validity of the data will be verified

        Returns
        -------
        LocalCoordinateSystem
            Cartesian coordinate system

        """
        if orientation is None:
            orientation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if coordinates is None:
            coordinates = np.array([0, 0, 0])

        time, time_ref = _build_time_index(time, time_ref)
        orientation = self._build_orientation(orientation, time)
        coordinates = self._build_coordinates(coordinates, time)

        if construction_checks:
            ut.xr_check_coords(
                coordinates,
                dict(
                    c={"values": ["x", "y", "z"]},
                    time={"dtype": "timedelta64", "optional": True},
                ),
            )

            ut.xr_check_coords(
                orientation,
                dict(
                    c={"values": ["x", "y", "z"]},
                    v={"values": [0, 1, 2]},
                    time={"dtype": "timedelta64", "optional": True},
                ),
            )

            orientation = xr.apply_ufunc(
                normalize,
                orientation,
                input_core_dims=[["c"]],
                output_core_dims=[["c"]],
            )

            # vectorize test if orthogonal
            if not ut.xr_is_orthogonal_matrix(orientation, dims=["c", "v"]):
                raise ValueError("Orientation vectors must be orthogonal")

        # unify time axis
        if ("time" in orientation.coords) and ("time" in coordinates.coords):
            if not np.all(orientation.time.data == coordinates.time.data):
                time_union = ut.get_time_union([orientation, coordinates])
                orientation = ut.xr_interp_orientation_in_time(orientation, time_union)
                coordinates = ut.xr_interp_coordinates_in_time(coordinates, time_union)

        coordinates.name = "coordinates"
        orientation.name = "orientation"

        self._dataset = xr.merge([coordinates, orientation], join="exact")
        if "time" in self._dataset and time_ref is not None:
            self._dataset.weldx.time_ref = time_ref

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
        lhs_cs = self
        if (
            lhs_cs.reference_time != rhs_cs.reference_time
            and lhs_cs.has_reference_time
            and rhs_cs.has_reference_time
        ):
            if lhs_cs.reference_time < rhs_cs.reference_time:
                time_ref = lhs_cs.reference_time
                rhs_cs = deepcopy(rhs_cs)
                rhs_cs.reset_reference_time(time_ref)
            else:
                time_ref = rhs_cs.reference_time
                lhs_cs = deepcopy(lhs_cs)
                lhs_cs.reset_reference_time(time_ref)
        elif not lhs_cs.has_reference_time:
            time_ref = rhs_cs.reference_time
        else:
            time_ref = lhs_cs.reference_time

        rhs_cs = rhs_cs.interp_time(lhs_cs.time, time_ref)

        orientation = ut.xr_matmul(
            rhs_cs.orientation, lhs_cs.orientation, dims_a=["c", "v"]
        )
        coordinates = (
            ut.xr_matmul(rhs_cs.orientation, lhs_cs.coordinates, ["c", "v"], ["c"])
            + rhs_cs.coordinates
        )
        return LocalCoordinateSystem(orientation, coordinates, time_ref=time_ref)

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

    def __eq__(self: "LocalCoordinateSystem", other: "LocalCoordinateSystem") -> bool:
        """Check equality of LocalCoordinateSystems."""
        return (
            self.orientation.identical(other.orientation)
            and self.coordinates.identical(other.coordinates)
            and self.reference_time == other.reference_time
        )

    @staticmethod
    def _build_orientation(
        orientation: Union[xr.DataArray, np.ndarray, List[List], Rot],
        time: pd.DatetimeIndex = None,
    ):
        """Create xarray orientation from different formats and time-inputs.

        Parameters
        ----------
        orientation :
            Orientation object or data.
        time :
            Valid time index formatted with `_build_time_index`.

        Returns
        -------
        xarray.DataArray

        """
        if not isinstance(orientation, xr.DataArray):
            time_orientation = None
            if isinstance(orientation, Rot):
                orientation = orientation.as_matrix()
            elif not isinstance(orientation, np.ndarray):
                orientation = np.array(orientation)

            if orientation.ndim == 3:
                time_orientation = time
            orientation = ut.xr_3d_matrix(orientation, time_orientation)

        # make sure we have correct "time" format
        orientation = orientation.weldx.time_ref_restore()

        return orientation

    @staticmethod
    def _build_coordinates(coordinates, time: pd.DatetimeIndex = None):
        """Create xarray coordinates from different formats and time-inputs.

        Parameters
        ----------
        coordinates:
            Coordinates data.
        time:
            Valid time index formatted with `_build_time_index`.

        Returns
        -------
        xarray.DataArray

        """
        if not isinstance(coordinates, xr.DataArray):
            time_coordinates = None
            if not isinstance(coordinates, (np.ndarray, pint.Quantity)):
                coordinates = np.array(coordinates)
            if coordinates.ndim == 2:
                time_coordinates = time
            coordinates = ut.xr_3d_vector(coordinates, time_coordinates)

        # make sure we have correct "time" format
        coordinates = coordinates.weldx.time_ref_restore()

        return coordinates

    @classmethod
    def from_euler(
        cls, sequence, angles, degrees=False, coordinates=None, time=None, time_ref=None
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
        time_ref :
            Reference Timestamp to use if time is Timedelta or pint.Quantity.

        Returns
        -------
        LocalCoordinateSystem
            Local coordinate system

        """
        orientation = Rot.from_euler(sequence, angles, degrees)
        return cls(orientation, coordinates=coordinates, time=time, time_ref=time_ref)

    @classmethod
    def from_orientation(
        cls, orientation, coordinates=None, time=None, time_ref=None
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
        time_ref :
            Reference Timestamp to use if time is Timedelta or pint.Quantity.

        Returns
        -------
        LocalCoordinateSystem
            Local coordinate system

        """
        return cls(orientation, coordinates=coordinates, time=time, time_ref=time_ref)

    @classmethod
    def from_xyz(
        cls, vec_x, vec_y, vec_z, coordinates=None, time=None, time_ref=None
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
        time_ref :
            Reference Timestamp to use if time is Timedelta or pint.Quantity.

        Returns
        -------
        LocalCoordinateSystem
            Local coordinate system

        """
        vec_x = ut.to_float_array(vec_x)
        vec_y = ut.to_float_array(vec_y)
        vec_z = ut.to_float_array(vec_z)

        orientation = np.concatenate((vec_x, vec_y, vec_z), axis=vec_x.ndim - 1)
        orientation = np.reshape(orientation, (*vec_x.shape, 3))
        orientation = orientation.swapaxes(orientation.ndim - 1, orientation.ndim - 2)
        return cls(orientation, coordinates=coordinates, time=time, time_ref=time_ref)

    @classmethod
    def from_xy_and_orientation(
        cls,
        vec_x,
        vec_y,
        positive_orientation=True,
        coordinates=None,
        time=None,
        time_ref=None,
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
        time_ref :
            Reference Timestamp to use if time is Timedelta or pint.Quantity.

        Returns
        -------
        LocalCoordinateSystem
            Local coordinate system

        """
        vec_z = cls._calculate_orthogonal_axis(vec_x, vec_y) * cls._sign_orientation(
            positive_orientation
        )

        return cls.from_xyz(vec_x, vec_y, vec_z, coordinates, time, time_ref=time_ref)

    @classmethod
    def from_yz_and_orientation(
        cls,
        vec_y,
        vec_z,
        positive_orientation=True,
        coordinates=None,
        time=None,
        time_ref=None,
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
        time_ref :
            Reference Timestamp to use if time is Timedelta or pint.Quantity.

        Returns
        -------
        LocalCoordinateSystem
            Local coordinate system

        """
        vec_x = cls._calculate_orthogonal_axis(vec_y, vec_z) * cls._sign_orientation(
            positive_orientation
        )

        return cls.from_xyz(vec_x, vec_y, vec_z, coordinates, time, time_ref=time_ref)

    @classmethod
    def from_xz_and_orientation(
        cls,
        vec_x,
        vec_z,
        positive_orientation=True,
        coordinates=None,
        time=None,
        time_ref=None,
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
        time_ref :
            Reference Timestamp to use if time is Timedelta or pint.Quantity.

        Returns
        -------
        LocalCoordinateSystem
            Local coordinate system

        """
        vec_y = cls._calculate_orthogonal_axis(vec_z, vec_x) * cls._sign_orientation(
            positive_orientation
        )

        return cls.from_xyz(vec_x, vec_y, vec_z, coordinates, time, time_ref=time_ref)

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
        numpy.ndarray
            Orthogonal axis

        """
        return np.cross(a_0, a_1)

    @property
    def orientation(self) -> xr.DataArray:
        """Get the coordinate systems orientation matrix.

        Returns
        -------
        xarray.DataArray
            Orientation matrix

        """
        return self.dataset.orientation

    @property
    def coordinates(self) -> xr.DataArray:
        """Get the coordinate systems coordinates.

        Returns
        -------
        xarray.DataArray
            Coordinates of the coordinate system

        """
        return self.dataset.coordinates

    @property
    def is_time_dependent(self) -> bool:
        """Return `True` if the coordinate system is time dependent.

        Returns
        -------
        bool :
            `True` if the coordinate system is time dependent, `False` otherwise.

        """
        return self.time is not None

    @property
    def has_reference_time(self) -> bool:
        """Return `True` if the coordinate system has a reference time.

        Returns
        -------
        bool :
            `True` if the coordinate system has a reference time, `False` otherwise.

        """
        return self.reference_time is not None

    @property
    def reference_time(self) -> Union[pd.Timestamp, None]:
        """Get the coordinate systems reference time.

        Returns
        -------
        pandas.Timestamp:
            The coordinate systems reference time

        """
        return self._dataset.weldx.time_ref

    @property
    def datetimeindex(self) -> Union[pd.DatetimeIndex, None]:
        """Get the time as 'pandas.DatetimeIndex'.

        If the coordinate system has no reference time, 'None' is returned.

        Returns
        -------
        Union[pandas.DatetimeIndex, None]:
            The coordinate systems time as 'pandas.DatetimeIndex'

        """
        if not self.has_reference_time:
            return None
        return self.time + self.reference_time

    @property
    def time(self) -> Union[pd.TimedeltaIndex, None]:
        """Get the time union of the local coordinate system (None if system is static).

        Returns
        -------
        pandas.TimedeltaIndex
            DateTimeIndex-like time union

        """
        if "time" in self._dataset.coords:
            return self._dataset.time
        return None

    @property
    def time_quantity(self) -> pint.Quantity:
        """Get the time as 'pint.Quantity'.

        Returns
        -------
        pint.Quantity:
            The coordinate systems time as 'pint.Quantity'

        """
        return ut.pandas_time_delta_to_quantity(self.time)

    @property
    def dataset(self) -> xr.Dataset:
        """Get the underlying xarray.Dataset with ordered dimensions.

        Returns
        -------
        xarray.Dataset
            xarray Dataset with coordinates and orientation as DataVariables.

        """
        return self._dataset.transpose(..., "c", "v")

    @property
    def is_unity_translation(self) -> bool:
        """Return true if the LCS has a zero translation/coordinates value."""
        if self.coordinates.shape[-1] == 3 and np.allclose(
            self.coordinates, np.zeros(3)
        ):
            return True
        return False

    @property
    def is_unity_rotation(self) -> bool:
        """Return true if the LCS represents a unity rotation/orientations value."""
        if self.orientation.shape[-2:] == (3, 3) and np.allclose(
            self.orientation, np.eye(3)
        ):
            return True
        return False

    def interp_time(
        self,
        time: Union[
            pd.DatetimeIndex,
            pd.TimedeltaIndex,
            List[pd.Timestamp],
            "LocalCoordinateSystem",
            None,
        ],
        time_ref: Union[pd.Timestamp, None] = None,
    ) -> "LocalCoordinateSystem":
        """Interpolates the data in time.

        Parameters
        ----------
        time :
            Series of times.
            If passing "None" no interpolation will be performed.
        time_ref:
            The reference timestamp

        Returns
        -------
        LocalCoordinateSystem
            Coordinate system with interpolated data

        """
        if (not self.is_time_dependent) or (time is None):
            return self

        # use LCS reference time if none provided
        if isinstance(time, LocalCoordinateSystem) and time_ref is None:
            time_ref = time.reference_time
        time = ut.to_pandas_time_index(time)

        if self.has_reference_time != (
            time_ref is not None or isinstance(time, pd.DatetimeIndex)
        ):
            raise TypeError(
                "Only 1 reference time provided for time dependent coordinate "
                "system. Either the reference time of the coordinate system or the "
                "one passed to the function is 'None'. Only cases where the "
                "reference times are both 'None' or both contain a timestamp are "
                "allowed. Also check that the reference time has the correct type."
            )

        if self.has_reference_time:
            if not isinstance(time, pd.DatetimeIndex):
                time = time + time_ref

        orientation = ut.xr_interp_orientation_in_time(self.orientation, time)
        coordinates = ut.xr_interp_coordinates_in_time(self.coordinates, time)

        return LocalCoordinateSystem(orientation, coordinates, time_ref=time_ref)

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
        return LocalCoordinateSystem(
            orientation, coordinates, self.time, self.reference_time
        )

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
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial
            .transform.Rotation.as_euler.html
        degrees :
            Returned angles are in degrees if True, else they are in radians.
            Default is False.

        Returns
        -------
        numpy.ndarray
            Array of euler angles.

        """
        return self.as_rotation().as_euler(seq=seq, degrees=degrees)

    def reset_reference_time(self, time_ref_new: pd.Timestamp):
        """Reset the reference time of the coordinate system.

        The time values of the coordinate system are adjusted to the new reference time.
        If no reference time has been set before, the time values will remain
        unmodified. This assumes that the current time delta values are already
        referring to the new reference time.

        Parameters
        ----------
        time_ref_new: pandas.Timestamp
            The new reference time

        """
        self._dataset.weldx.time_ref = time_ref_new


# CoordinateSystemManager --------------------------------------------------------------


class CoordinateSystemManager:
    """Handles hierarchical dependencies between multiple coordinate systems.

    Notes
    -----
    Learn how to use this class by reading the
    :doc:`Tutorial <../tutorials/transformations_02_coordinate_system_manager>`.

    """

    _id_gen = itertools.count()

    @dataclass
    class CoordinateSystemData:
        """Class that stores data and the coordinate system, the data is assigned to."""

        coordinate_system_name: str
        data: xr.DataArray

    def __init__(
        self,
        root_coordinate_system_name: str,
        coordinate_system_manager_name: Union[str, None] = None,
        time_ref: pd.Timestamp = None,
        _graph: Union[nx.DiGraph, None] = None,
        _subsystems=None,
    ):
        """Construct a coordinate system manager.

        Parameters
        ----------
        root_coordinate_system_name : str
            Name of the root coordinate system.
        coordinate_system_manager_name : str
            Name of the coordinate system manager. If 'None' is passed, a default name
            is chosen.
        time_ref : pandas.Timestamp
            A reference timestamp. If it is defined, all time dependent information
            returned by the CoordinateSystemManager will refer to it by default.
        _graph:
            A graph that should be used internally. Do not set this parameter. It is
            only meant for class internal usage.
        _subsystems:
            A dictionary containing data about the CSMs attached subsystems. This
            parameter should never be set manually. It is for internal usage only.


        Returns
        -------
        CoordinateSystemManager

        """
        if coordinate_system_manager_name is None:
            coordinate_system_manager_name = self._generate_default_name()
        self._name = coordinate_system_manager_name
        if time_ref is not None and not isinstance(time_ref, pd.Timestamp):
            time_ref = pd.Timestamp(time_ref)
        self._reference_time = time_ref

        self._data = {}
        self._root_system_name = root_coordinate_system_name

        self._sub_system_data_dict = _subsystems
        if self._sub_system_data_dict is None:
            self._sub_system_data_dict = {}

        self._graph = _graph
        if self._graph is None:
            self._graph = nx.DiGraph()
            self._add_coordinate_system_node(root_coordinate_system_name)

    def __repr__(self):
        """Output representation of a CoordinateSystemManager class."""
        return (
            f"<CoordinateSystemManager>\nname:\n\t{self._name}\n"
            f"reference time:\n\t {self.reference_time}\n"
            f"coordinate systems:\n\t {self.coordinate_system_names}\n"
            f"data:\n\t {self._data!r}\n"
            f"sub systems:\n\t {self._sub_system_data_dict.keys()}\n"
            f")"
        )

    def __eq__(self: "CoordinateSystemManager", other: "CoordinateSystemManager"):
        """Test equality of CSM instances."""
        # todo: also check data  -> add tests
        if not isinstance(other, self.__class__):
            return False

        graph_0 = self.graph
        graph_1 = other.graph

        if self.name != other.name:
            return False

        if self.reference_time != other.reference_time:
            return False

        if len(graph_0.nodes) != len(graph_1.nodes):
            return False

        # if self.sub_system_data != other.sub_system_data:
        if not self._compare_subsystems_equal(
            self.sub_system_data, other.sub_system_data
        ):
            return False

        # check nodes
        for node in graph_0.nodes:
            if node not in graph_1.nodes:
                return False

        # check edges
        for edge in graph_0.edges:
            if edge not in graph_1.edges:
                return False

        # check coordinate systems
        for edge in graph_0.edges:
            lcs_0 = self.graph.edges[(edge[0], edge[1])]["lcs"]
            lcs_1 = other.graph.edges[(edge[0], edge[1])]["lcs"]
            if lcs_0 != lcs_1:
                return False

        return True

    @property
    def lcs(self) -> List["LocalCoordinateSystem"]:
        """Get a list of all attached `LocalCoordinateSystem` instances.

        Only the defined systems and not the automatically generated inverse systems
        are included.

        Returns
        -------
        List[LocalCoordinateSystem] :
           List of all attached `LocalCoordinateSystem` instances.

        """
        return [
            self.graph.edges[edge]["lcs"]
            for edge in self.graph.edges
            if self.graph.edges[edge]["defined"]
        ]

    @property
    def lcs_time_dependent(self) -> List["LocalCoordinateSystem"]:
        """Get a list of all attached time dependent `LocalCoordinateSystem` instances.

        Returns
        -------
        List[LocalCoordinateSystem] :
            List of all attached time dependent `LocalCoordinateSystem` instances

        """
        return [lcs for lcs in self.lcs if lcs.is_time_dependent]

    @property
    def uses_absolute_times(self) -> bool:
        """Return `True` if the CSM or one of its coord. systems has a reference time.

        Returns
        -------
        bool :[
            `True` if the `CoordinateSystemManager` or one of its attached coordinate
            systems possess a reference time. `False` otherwise

        """
        return self._has_lcs_with_time_ref or (self.has_reference_time)

    @property
    def has_reference_time(self) -> bool:
        """Return `True` if the coordinate system manager has a reference time.

        Returns
        -------
        bool :
            `True` if the coordinate system manager has a reference time, `False`
            otherwise.

        """
        return self.reference_time is not None

    def _add_coordinate_system_node(self, coordinate_system_name):
        self._check_new_coordinate_system_name(coordinate_system_name)
        self._graph.add_node(coordinate_system_name, data=[])

    def _add_edges(self, node_from: str, node_to: str, lcs: LocalCoordinateSystem):
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
        self._graph.add_edge(node_from, node_to, lcs=lcs, defined=True)
        self._graph.add_edge(node_to, node_from, lcs=lcs.invert(), defined=False)

    def _check_coordinate_system_exists(self, coordinate_system_name: str):
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

    def _check_new_coordinate_system_name(self, coordinate_system_name: str):
        """Raise an exception if the new coordinate systems' name is invalid.

        Parameters
        ----------
        coordinate_system_name :
            Name of the new coordinate system, that should be checked.

        """
        if not isinstance(coordinate_system_name, str):
            raise TypeError("The coordinate system name must be a string.")
        if self.has_coordinate_system(coordinate_system_name):
            raise ValueError(
                "There already is a coordinate system with name "
                + str(coordinate_system_name)
            )

    @classmethod
    def _compare_subsystems_equal(cls, data: Dict, other: Dict) -> bool:
        """Compare if two subsystem data dictionaries are equal.

        Parameters
        ----------
        data:
            First subsystem data dictionary.
        other
            Second subsystem data dictionary.

        Returns
        -------
        bool:
            'True' if both dictionaries are identical, 'False' otherwise.

        """
        if len(data) != len(other):
            return False
        for subsystem_name, subsystem_data in data.items():
            if subsystem_name not in other:
                return False
            other_data = other[subsystem_name]
            if subsystem_data["common node"] != other_data["common node"]:
                return False
            if subsystem_data["root"] != other_data["root"]:
                return False
            if subsystem_data["time_ref"] != other_data["time_ref"]:
                return False
            if set(subsystem_data["neighbors"]) != set(other_data["neighbors"]):
                return False
            if set(subsystem_data["original members"]) != set(
                other_data["original members"]
            ):
                return False
            if not cls._compare_subsystems_equal(
                subsystem_data["sub system data"], other_data["sub system data"]
            ):
                return False
        return True

    @staticmethod
    def _generate_default_name() -> str:
        """Get a default name for the current coordinate system manager instance.

        Returns
        -------
        str:
            Default name.

        """
        return f"Coordinate system manager {next(CoordinateSystemManager._id_gen)}"

    @property
    def _extended_sub_system_data(self) -> Dict:
        """Get an extended copy of the internal sub system data.

        The function adds a list of potential child coordinate systems to each
        sub system. Coordinate systems in this list might belong to other sub systems
        that share a common coordinate system with the current sub system.

        Returns
        -------
        Dict:
            Extended copy of the internal sub system data.

        """
        sub_system_data_dict = deepcopy(self._sub_system_data_dict)
        for _, sub_system_data in sub_system_data_dict.items():
            potential_members = []
            for cs_name in sub_system_data["neighbors"]:
                potential_members += self.get_child_system_names(cs_name, False)

            sub_system_data["nodes"] = potential_members + sub_system_data["neighbors"]

        return sub_system_data_dict

    @staticmethod
    def _get_sub_system_members(
        ext_sub_system_data, ext_sub_system_data_dict
    ) -> List[str]:
        """Get a list with all coordinate system names, that belong to the sub system.

        Parameters
        ----------
        ext_sub_system_data:
            The extended sub system data of a single sub system.
        ext_sub_system_data_dict:
            Dictionary containing the extended sub system data of all sub systems.

        Returns
        -------
        List[str]:
            List of all the sub systems coordinate systems.

        """
        all_members = ext_sub_system_data["nodes"]
        for _, other_sub_system_data in ext_sub_system_data_dict.items():
            if other_sub_system_data["common node"] in all_members:
                all_members = [
                    cs_name
                    for cs_name in all_members
                    if cs_name not in other_sub_system_data["nodes"]
                ]

        all_members += [ext_sub_system_data["common node"]]
        return all_members

    @property
    def _has_lcs_with_time_ref(self):
        """Return `True` if one of the attached coordinate systems has a reference time.

        Returns
        -------
        bool :
            `True` if one of the attached coordinate systems has a reference time.
            `False` otherwise

        """
        return any(lcs.has_reference_time for lcs in self.lcs_time_dependent)

    def _ipython_display_(self):
        """Display the coordinate system manager as plot in jupyter notebooks."""
        self.plot()

    @property
    def _number_of_time_dependent_lcs(self):
        """Get the number of time dependent coordinate systems.

        Note that the automatically added inverse systems have no effect on the returned
        val

        Returns
        -------
        int :
            Number of time dependent coordinate systems

        """
        return len(self.lcs_time_dependent)

    def _update_local_coordinate_system(
        self, node_from: str, node_to: str, lcs: LocalCoordinateSystem
    ):
        """Update the local coordinate systems on the edges between two nodes.

        Parameters
        ----------
        node_from :
            Start node of the edge
        node_to :
            End node of the edge
        lcs :
            Local coordinate system

        """
        edge_from_to = self.graph.edges[(node_from, node_to)]
        edge_from_to["lcs"] = lcs
        edge_from_to["defined"] = True

        edge_to_from = self.graph.edges[(node_to, node_from)]
        edge_to_from["lcs"] = lcs.invert()
        edge_to_from["defined"] = False

    @property
    def graph(self) -> nx.DiGraph:
        """Get the internal graph.

        Returns
        -------
        networkx.DiGraph

        """
        return self._graph

    @property
    def name(self) -> str:
        """Get the name of the coordinate system manager instance.

        Returns
        -------
        str:
            Name of the coordinate system manager instance.

        """
        return self._name

    @property
    def number_of_coordinate_systems(self) -> int:
        """Get the number of coordinate systems inside the coordinate system manager.

        Returns
        -------
        int
            Number of coordinate systems

        """
        return self._graph.number_of_nodes()

    @property
    def number_of_subsystems(self) -> int:
        """Get the number of attached subsystems.

        Returns
        -------
        int:
            Number of attached subsystems.

        """
        return len(self._sub_system_data_dict)

    @property
    def reference_time(self):
        """Get the reference time of the `CoordinateSystemManager`.

        Returns
        -------
        pandas.Timestamp :
            Reference time of the `CoordinateSystemManager`

        """
        return self._reference_time

    @property
    def root_system_name(self) -> str:
        """Get the name of the root system.

        Returns
        -------
        str:
            Name of the root system

        """
        return self._root_system_name

    @property
    def sub_system_data(self) -> Dict:
        """Get a dictionary containing data about the attached subsystems."""
        return self._sub_system_data_dict

    @property
    def subsystem_names(self) -> List[str]:
        """Get the names of all subsystems.

        Returns
        -------
        List[str]:
            List with subsystem names.

        """
        return self._sub_system_data_dict.keys()

    def add_cs(
        self,
        coordinate_system_name: str,
        reference_system_name: str,
        lcs: LocalCoordinateSystem,
        lsc_child_in_parent: bool = True,
    ):
        """Add a coordinate system to the coordinate system manager.

        If the specified system already exists with the same parent system it will be
        updated. If the parent systems does not match, an exception is raised.

        Notes
        -----
        The time component of coordinate systems without defined reference time is
        assumed to refer to the same reference time as the `CoordinateSystemManager`.
        In case that the `CoordinateSystemManager` does not possess a reference time,
        you have to assure that either all or none of the added coordinate systems have
        a reference time.
        Violation of this rule will cause an exception.
        If neither the `CoordinateSystemManager` nor the attached coordinate systems
        have a reference time, all time deltas are expected to have common but undefined
        reference time.

        Parameters
        ----------
        coordinate_system_name : str
            Name of the new coordinate system.
        reference_system_name : str
            Name of the parent system. This must have been already added.
        lcs : LocalCoordinateSystem
            An instance of
            `~weldx.transformations.LocalCoordinateSystem` that describes how the new
            coordinate system is oriented in its parent system.
        lsc_child_in_parent: bool
            If set to `True`, the passed `LocalCoordinateSystem` instance describes
            the new system orientation towards is parent. If `False`, it describes
            how the parent system is positioned in its new child system.

        """
        if not isinstance(lcs, LocalCoordinateSystem):
            raise TypeError(
                "'local_coordinate_system' must be an instance of "
                + "weldx.transformations.LocalCoordinateSystem"
            )

        if (
            lcs.is_time_dependent  # always add static lcs
            and self._number_of_time_dependent_lcs > 0  # CSM is not static
            and (
                (lcs.has_reference_time and not self.uses_absolute_times)
                or (
                    (not lcs.has_reference_time and not self.has_reference_time)
                    and self.uses_absolute_times
                )
            )
        ):
            raise Exception(
                "Inconsistent usage of reference times! If you didn't specify a "
                "reference time for the CoordinateSystemManager, either all or "
                "none of the added coordinate systems must have a reference time."
            )

        if self.has_coordinate_system(coordinate_system_name):
            # todo:
            #  discuss: update and add functionality should be separated
            #  why?   : to prevent errors. Misspelling of the system name might cause
            #           unwanted updates or unwanted additions. Separate function can
            #           catch that by knowing about the users intention.
            if not self.is_neighbor_of(coordinate_system_name, reference_system_name):
                raise ValueError(
                    f'Can not update coordinate system. "{reference_system_name}" is '
                    f"not a neighbor of {coordinate_system_name}"
                )
            if lsc_child_in_parent:
                self._update_local_coordinate_system(
                    coordinate_system_name, reference_system_name, lcs,
                )
            else:
                self._update_local_coordinate_system(
                    reference_system_name, coordinate_system_name, lcs,
                )
        else:
            self._check_coordinate_system_exists(reference_system_name)
            self._add_coordinate_system_node(coordinate_system_name)
            if lsc_child_in_parent:
                self._add_edges(
                    coordinate_system_name, reference_system_name, lcs,
                )
            else:
                self._add_edges(
                    reference_system_name, coordinate_system_name, lcs,
                )

    def assign_data(
        self, data: xr.DataArray, data_name: str, coordinate_system_name: str
    ):
        """Assign spatial data to a coordinate system.

        Parameters
        ----------
        data :
            Spatial data
        data_name :
            Name of the data.
        coordinate_system_name :
            Name of the coordinate system the data should be
            assigned to.

        """
        # TODO: How to handle time dependent data? some things to think about:
        # - times of coordinate system and data are not equal
        # - which time is taken as reference? (probably the one of the data)
        # - what happens during cal of time interpolation functions with data? Also
        #   interpolated or not?
        if not isinstance(data_name, str):
            raise TypeError("The data name must be a string.")
        self._check_coordinate_system_exists(coordinate_system_name)

        self._data[data_name] = self.CoordinateSystemData(coordinate_system_name, data)
        self._graph.nodes[coordinate_system_name]["data"].append(data_name)

    def create_cs(
        self,
        coordinate_system_name: str,
        reference_system_name: str,
        orientation: Union[xr.DataArray, np.ndarray, List[List], Rot] = None,
        coordinates: Union[xr.DataArray, np.ndarray, List] = None,
        time: Union[pd.TimedeltaIndex, pd.DatetimeIndex] = None,
        time_ref: pd.Timestamp = None,
        lsc_child_in_parent: bool = True,
    ):
        """Create a coordinate system and add it to the coordinate system manager.

        This function uses the '__init__' method of the 'LocalCoordinateSystem' class.

        Parameters
        ----------
        coordinate_system_name :
            Name of the new coordinate system.
        reference_system_name :
            Name of the parent system. This must have been already added.
        orientation :
            Matrix of 3 orthogonal column vectors which represent
            the coordinate systems orientation. Keep in mind, that the columns of the
            corresponding orientation matrix is equal to the normalized orientation
            vectors. So each orthogonal transformation matrix can also be
            provided as orientation.
            Passing a scipy.spatial.transform.Rotation object is also supported.
        coordinates :
            Coordinates of the origin.
        time :
            Time data for time dependent coordinate systems.
        time_ref :
            Reference time for time dependent coordinate systems
        lsc_child_in_parent:
            If set to 'True', the passed 'LocalCoordinateSystem' instance describes
            the new system orientation towards is parent. If 'False', it describes
            how the parent system is positioned in its new child system.

        """
        lcs = LocalCoordinateSystem(orientation, coordinates, time, time_ref)
        self.add_cs(
            coordinate_system_name, reference_system_name, lcs, lsc_child_in_parent
        )

    def create_cs_from_euler(
        self,
        coordinate_system_name: str,
        reference_system_name: str,
        sequence,
        angles,
        degrees=False,
        coordinates: Union[xr.DataArray, np.ndarray, List] = None,
        time: pd.DatetimeIndex = None,
        lsc_child_in_parent: bool = True,
    ):
        """Create a coordinate system and add it to the coordinate system manager.

        This function uses the 'from_euler' method of the
        'LocalCoordinateSystem' class.

        Parameters
        ----------
        coordinate_system_name :
            Name of the new coordinate system.
        reference_system_name :
            Name of the parent system. This must have been already added.
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
            Coordinates of the origin.
        time :
            Time data for time dependent coordinate systems.
        lsc_child_in_parent:
            If set to 'True', the passed 'LocalCoordinateSystem' instance describes
            the new system orientation towards is parent. If 'False', it describes
            how the parent system is positioned in its new child system.


        """
        lcs = LocalCoordinateSystem.from_euler(
            sequence, angles, degrees, coordinates, time
        )
        self.add_cs(
            coordinate_system_name, reference_system_name, lcs, lsc_child_in_parent
        )

    def create_cs_from_xyz(
        self,
        coordinate_system_name: str,
        reference_system_name: str,
        vec_x,
        vec_y,
        vec_z,
        coordinates: Union[xr.DataArray, np.ndarray, List] = None,
        time: pd.DatetimeIndex = None,
        lsc_child_in_parent: bool = True,
    ):
        """Create a coordinate system and add it to the coordinate system manager.

        This function uses the 'from_xyz' method of the
        'LocalCoordinateSystem' class.

        Parameters
        ----------
        coordinate_system_name :
            Name of the new coordinate system.
        reference_system_name :
            Name of the parent system. This must have been already added.
        vec_x :
            Vector defining the x-axis
        vec_y :
            Vector defining the y-axis
        vec_z :
            Vector defining the z-axis
        coordinates :
            Coordinates of the origin.
        time :
            Time data for time dependent coordinate systems.
        lsc_child_in_parent:
            If set to 'True', the passed 'LocalCoordinateSystem' instance describes
            the new system orientation towards is parent. If 'False', it describes
            how the parent system is positioned in its new child system.

        """
        lcs = LocalCoordinateSystem.from_xyz(vec_x, vec_y, vec_z, coordinates, time)
        self.add_cs(
            coordinate_system_name, reference_system_name, lcs, lsc_child_in_parent
        )

    def create_cs_from_xy_and_orientation(
        self,
        coordinate_system_name: str,
        reference_system_name: str,
        vec_x,
        vec_y,
        positive_orientation=True,
        coordinates: Union[xr.DataArray, np.ndarray, List] = None,
        time: pd.DatetimeIndex = None,
        lsc_child_in_parent: bool = True,
    ):
        """Create a coordinate system and add it to the coordinate system manager.

        This function uses the 'from_xy_and_orientation' method of the
        'LocalCoordinateSystem' class.

        Parameters
        ----------
        coordinate_system_name :
            Name of the new coordinate system.
        reference_system_name :
            Name of the parent system. This must have been already added.
        vec_x :
            Vector defining the x-axis
        vec_y :
            Vector defining the y-axis
        positive_orientation :
            Set to True if the orientation should
            be positive and to False if not (Default value = True)
        coordinates :
            Coordinates of the origin.
        time :
            Time data for time dependent coordinate systems.
        lsc_child_in_parent:
            If set to 'True', the passed 'LocalCoordinateSystem' instance describes
            the new system orientation towards is parent. If 'False', it describes
            how the parent system is positioned in its new child system.

        """
        lcs = LocalCoordinateSystem.from_xy_and_orientation(
            vec_x, vec_y, positive_orientation, coordinates, time
        )
        self.add_cs(
            coordinate_system_name, reference_system_name, lcs, lsc_child_in_parent
        )

    def create_cs_from_xz_and_orientation(
        self,
        coordinate_system_name: str,
        reference_system_name: str,
        vec_x,
        vec_z,
        positive_orientation=True,
        coordinates: Union[xr.DataArray, np.ndarray, List] = None,
        time: pd.DatetimeIndex = None,
        lsc_child_in_parent: bool = True,
    ):
        """Create a coordinate system and add it to the coordinate system manager.

        This function uses the 'from_xz_and_orientation' method of the
        'LocalCoordinateSystem' class.

        Parameters
        ----------
        coordinate_system_name :
            Name of the new coordinate system.
        reference_system_name :
            Name of the parent system. This must have been already added.
        vec_x :
            Vector defining the x-axis
        vec_z :
            Vector defining the z-axis
        positive_orientation :
            Set to True if the orientation should
            be positive and to False if not (Default value = True)
        coordinates :
            Coordinates of the origin.
        time :
            Time data for time dependent coordinate systems.
        lsc_child_in_parent:
            If set to 'True', the passed 'LocalCoordinateSystem' instance describes
            the new system orientation towards is parent. If 'False', it describes
            how the parent system is positioned in its new child system.

        """
        lcs = LocalCoordinateSystem.from_xz_and_orientation(
            vec_x, vec_z, positive_orientation, coordinates, time
        )
        self.add_cs(
            coordinate_system_name, reference_system_name, lcs, lsc_child_in_parent
        )

    def create_cs_from_yz_and_orientation(
        self,
        coordinate_system_name: str,
        reference_system_name: str,
        vec_y,
        vec_z,
        positive_orientation=True,
        coordinates: Union[xr.DataArray, np.ndarray, List] = None,
        time: pd.DatetimeIndex = None,
        lsc_child_in_parent: bool = True,
    ):
        """Create a coordinate system and add it to the coordinate system manager.

        This function uses the 'from_yz_and_orientation' method of the
        'LocalCoordinateSystem' class.

        Parameters
        ----------
        coordinate_system_name :
            Name of the new coordinate system.
        reference_system_name :
            Name of the parent system. This must have been already added.
        vec_y :
            Vector defining the y-axis
        vec_z :
            Vector defining the z-axis
        positive_orientation :
            Set to True if the orientation should
            be positive and to False if not (Default value = True)
        coordinates :
            Coordinates of the origin.
        time :
            Time data for time dependent coordinate systems.
        lsc_child_in_parent:
            If set to 'True', the passed 'LocalCoordinateSystem' instance describes
            the new system orientation towards is parent. If 'False', it describes
            how the parent system is positioned in its new child system.

        """
        lcs = LocalCoordinateSystem.from_yz_and_orientation(
            vec_y, vec_z, positive_orientation, coordinates, time
        )
        self.add_cs(
            coordinate_system_name, reference_system_name, lcs, lsc_child_in_parent
        )

    def delete_cs(self, coordinate_system_name: str, delete_children: bool = False):
        """Delete a coordinate system from the coordinate system manager.

        If the Coordinate system manager has attached sub system, there are multiple
        possible  consequences.

        - All subsystems attached to the deleted coordinate system or one of
          its child systems are removed from the coordinate system manager
        - If the coordinate system is part of a subsystem and belongs to the systems
          that were present when the subsystem was merged, the subsystem is removed and
          can not be restored using `subsystems` or `unmerge`. Coordinate systems
          of the subsystem that aren't a child of the deleted coordinate system will
          remain in the coordinate system manager
        - If the coordinate system is part of a subsystem but was added after merging,
          only the systems and its children are removed. The subsystem remains in the
          coordinate system manager.

        Parameters
        ----------
        coordinate_system_name:
            Name of the coordinate system that should be deleted.
        delete_children:
            If 'False', an exception is raised if the coordinate system has one or more
            children since deletion would cause them to be disconnected to the root.
            If 'True', all children are deleted as well.

        """
        if not self.has_coordinate_system(coordinate_system_name):
            return

        if coordinate_system_name == self._root_system_name:
            raise ValueError("The root system can't be deleted.")

        children = self.get_child_system_names(coordinate_system_name, False)

        if not delete_children and len(children) > 0:
            raise Exception(
                f'Can not delete coordinate system "{coordinate_system_name}". It '
                "has one or more children that would be disconnected to the root "
                f'after deletion. Set the delete_children option to "True" to '
                f"delete the coordinate system and all its children. "
                f"The attached child systems are: {children}"
            )

        # update subsystems
        remove_systems = []
        for sub_system_name, sub_system_data in self._sub_system_data_dict.items():
            if (
                coordinate_system_name in sub_system_data["original members"]
            ) or coordinate_system_name in nx.shortest_path(
                self.graph, sub_system_data["root"], self._root_system_name
            ):
                remove_systems += [sub_system_name]

        for sub_system_name in remove_systems:
            del self._sub_system_data_dict[sub_system_name]

        # delete nodes and edges
        if delete_children:
            for child in children:
                self._graph.remove_node(child)
        self._graph.remove_node(coordinate_system_name)

    def get_child_system_names(
        self, coordinate_system_name: str, neighbors_only: bool = True
    ) -> List[str]:
        """Get a list with the passed coordinate systems children.

        Parameters
        ----------
        coordinate_system_name:
            Name of the coordinate system
        neighbors_only:
            If 'True', only child coordinate systems that are directly connected to the
            specified coordinate system are included in the returned list. If 'False',
            child systems of arbitrary hierarchical depth are included.

        Returns
        -------
        List[str]:
            List of child systems.

        """
        if neighbors_only:
            return [
                cs
                for cs in self.neighbors(coordinate_system_name)
                if cs != self.get_parent_system_name(coordinate_system_name)
            ]

        current_children = self.get_child_system_names(coordinate_system_name, True)
        all_children = deepcopy(current_children)
        while current_children:
            new_children = []
            for child in current_children:
                new_children += self.get_child_system_names(child, True)
            all_children += new_children
            current_children = new_children

        return all_children

    @property
    def coordinate_system_names(self) -> List:
        """Get the names of all contained coordinate systems.

        Returns
        -------
        List:
            List of coordinate system names.

        """
        return list(self.graph.nodes)

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
        numpy.ndarray
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

    def get_cs(
        self,
        coordinate_system_name: str,
        reference_system_name: Union[str, None] = None,
        time: Union[pd.TimedeltaIndex, pd.DatetimeIndex, pint.Quantity, str] = None,
        time_ref: pd.Timestamp = None,
    ) -> LocalCoordinateSystem:
        """Get a coordinate system in relation to another reference system.

        If no reference system is specified, the parent system will be used as
        reference.

        If any coordinate system that is involved in the coordinate transformation has
        a time dependency, the returned coordinate system will also be time dependent.

        The timestamps of the returned system depend on the functions time parameter.
        By default, the time union of all involved coordinate systems is taken.

        Notes
        -----
        **Reference time of the returned system**

        The reference time of the returned coordinate system depends on multiple
        factors like the one passed to the function and the internally stored reference
        times. Generally, the following rules apply:

        - if a reference time was passed to the function, it will be used as reference
          time of the returned coordinate system as long as a time was also passed to
          the function.
        - else the reference time of the `CoordinateSystemManager` will be used if it
          has one
        - if only the coordinate systems have a reference time, the lowest (earliest)
          will be used
        - if there is no reference time at all, the resulting coordinate system won't
          have one either
        - if no time was passed to the function, a passed reference time will be
          ignored
        - a `pandas.DatetimeIndex` always has its lowest date as implicit reference time
          which will be used if the `CoordinateSystemManager` doesn't possess one and
          the functions reference time isn't set.


        A overview of all possible combinations using a `pandas.TimedeltaIndex` or
        a `pint.Quantity` as ``time`` parameter is given in the table below.

        +------------+--------------+-----------+----------------+-----------------+
        | function   | function has | CSM has   | CS have        | Returned system |
        | has        | reference    | reference | reference      | uses reference  |
        | time       | time         | time      | times          | time of         |
        +============+==============+===========+================+=================+
        | Yes        | Yes          | Yes       | all/mixed/none | function        |
        +------------+--------------+-----------+----------------+-----------------+
        | No         | Yes          | Yes       | all/mixed/none | CSM             |
        +------------+--------------+-----------+----------------+-----------------+
        | Yes / No   | No           | Yes       | all/mixed/none | CSM             |
        +------------+--------------+-----------+----------------+-----------------+
        | Yes        | Yes          | No        | all            | function        |
        +------------+--------------+-----------+----------------+-----------------+
        | No         | Yes          | No        | all            | CS (lowest)     |
        +------------+--------------+-----------+----------------+-----------------+
        | Yes / No   | Yes / No     | No        | mixed          | impossible -> 1.|
        +------------+--------------+-----------+----------------+-----------------+
        | Yes        | Yes          | No        | none           | error -> 2.     |
        +------------+--------------+-----------+----------------+-----------------+
        | No         | Yes          | No        | none           | `None`          |
        +------------+--------------+-----------+----------------+-----------------+
        | Yes / No   | No           | No        | all            | CS (lowest)     |
        +------------+--------------+-----------+----------------+-----------------+
        | Yes / No   | No           | No        | none           | `None`          |
        +------------+--------------+-----------+----------------+-----------------+

        1. This case can not occur since it is not allowed to add a combination of
           coordinate systems with and without reference time to a
           `CoordinateSystemManager` without own reference time. See `add_cs`
           documentation for further details
        2. If neither the `CoordinateSystemManager` nor its attached coordinate systems
           have a reference time, the intention of passing a time and a reference time
           to the function is unclear. The caller might be unaware of the missing
           reference times. Therefore an exception is raised. If your intention is to
           add a reference time to the resulting coordinate system, you should call this
           function without a specified reference time and add it explicitly to the
           returned `LocalCoordinateSystem`.


        **Information regarding the implementation:**

        It is important to mention that all coordinate systems that are involved in the
        transformation should be interpolated to a common time line before they are
        combined using the 'LocalCoordinateSystem's __add__ and __sub__ functions.
        If this is not done before, serious interpolation errors for rotations can
        occur. The reason is, that those operators also perform time interpolations
        if the timestamps of 2 systems do not match. When chaining multiple
        transformations already interpolated values might be used to perform another
        interpolation.

        To see why this is problematic, consider a coordinate system which is statically
        attached to a not moving but rotating parent coordinate system. If it gets
        transformed to the reference systems of its parent, it will follow a circular
        trajectory around the parent system. For discrete timestamps, the trajectory is
        described by a set of corresponding coordinates. If we now interpolate again,
        the positions between those coordinates will be interpolated linearly, ignoring
        the originally circular trajectory. The dependency on the rotating parent system
        is not considered in further transformations.

        Additionally, if the transformed system is rotating itself, the transformation
        to the parent's reference system might cause the rotation angle between to
        time steps to exceed 180 degrees. Since the SLERP always takes the shortest
        angle between 2 ''keyframes'', further interpolations wrongly change the
        rotation order.

        Parameters
        ----------
        coordinate_system_name : str
            Name of the coordinate system
        reference_system_name : str
            Name of the reference coordinate system
        time : pandas.TimedeltaIndex, pandas.DatetimeIndex, pint.Quantity or str
            Specifies the desired time of the returned coordinate system. You can also
            pass the name of another coordinate system to use its time attribute as
            reference
        time_ref : pandas.Timestamp
            The desired reference time of the returned coordinate system

        Returns
        -------
        LocalCoordinateSystem
            Local coordinate system

        """
        if reference_system_name is None:
            reference_system_name = self.get_parent_system_name(coordinate_system_name)
            if reference_system_name is None:
                raise ValueError(
                    f"The system {coordinate_system_name} has no parent system. "
                    f"You need to explicitly specify a reference system"
                )
        self._check_coordinate_system_exists(coordinate_system_name)
        self._check_coordinate_system_exists(reference_system_name)

        if coordinate_system_name == reference_system_name:
            return LocalCoordinateSystem()

        path = nx.shortest_path(
            self.graph, coordinate_system_name, reference_system_name
        )
        path_edges = list(zip(path[:-1], path[1:]))

        if time is None:
            time_ref = None  # ignore passed reference time if no time was passed
            time = self.time_union(path_edges)

        elif isinstance(time, str):
            parent_name = self.get_parent_system_name(time)
            if parent_name is None:
                raise ValueError("The root system has no time dependency.")

            time = self.get_cs(time, parent_name).time
            if time is None:
                raise ValueError(f'The system "{time}" is not time dependent')
        elif not isinstance(time, (pd.DatetimeIndex, pint.Quantity)):
            time = pd.TimedeltaIndex(time)

        if time_ref is None:
            time_ref = self.reference_time
        else:
            time_ref = pd.Timestamp(time_ref)

        time_interp, time_ref_interp = _build_time_index(time, time_ref)

        lcs_result = LocalCoordinateSystem()
        for edge in path_edges:
            lcs = self.graph.edges[edge]["lcs"]
            if lcs.is_time_dependent:
                if not lcs.has_reference_time and self.has_reference_time:
                    time_lcs = time_interp + (time_ref_interp - self.reference_time)
                    lcs = lcs.interp_time(time_lcs)
                    lcs.reset_reference_time(self.reference_time)
                    lcs.reset_reference_time(time_ref_interp)
                else:
                    lcs = lcs.interp_time(time_interp, time_ref_interp)
            lcs_result += lcs
        return lcs_result

    def get_parent_system_name(self, coordinate_system_name) -> Union[str, None]:
        """Get the name of a coordinate systems parent system.

        The parent is the next system on the path towards the root node.

        Parameters
        ----------
        coordinate_system_name :
            Name of the coordinate system

        Returns
        -------
        str
            Name of the parent system
        None
            If the coordinate system has no parent (root system)

        """
        if coordinate_system_name == self._root_system_name:
            return None

        self._check_coordinate_system_exists(coordinate_system_name)
        path = nx.shortest_path(
            self.graph, coordinate_system_name, self._root_system_name
        )

        return path[1]

    @property
    def subsystems(self) -> List["CoordinateSystemManager"]:
        """Extract all subsystems from the CoordinateSystemManager.

        Returns
        -------
        List:
            List containing all the subsystems.

        """
        ext_sub_system_data_dict = self._extended_sub_system_data

        sub_system_list = []
        for sub_system_name, ext_sub_system_data in ext_sub_system_data_dict.items():
            members = self._get_sub_system_members(
                ext_sub_system_data, ext_sub_system_data_dict
            )

            csm_sub = CoordinateSystemManager(
                ext_sub_system_data["root"],
                sub_system_name,
                time_ref=ext_sub_system_data["time_ref"],
                _graph=self._graph.subgraph(members).copy(),
                _subsystems=ext_sub_system_data["sub system data"],
            )
            sub_system_list.append(csm_sub)

        return sub_system_list

    def has_coordinate_system(self, coordinate_system_name: str) -> bool:
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

    def has_data(self, coordinate_system_name: str, data_name: str) -> bool:
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

    def interp_time(
        self,
        time: Union[
            pd.DatetimeIndex,
            pd.TimedeltaIndex,
            List[pd.Timestamp],
            "LocalCoordinateSystem",
        ],
        time_ref: pd.Timestamp = None,
        affected_coordinate_systems: Union[str, List[str], None] = None,
        in_place: bool = False,
    ) -> "CoordinateSystemManager":
        """Interpolates the coordinate systems in time.

        If no list of affected coordinate systems is provided, all systems will be
        interpolated to the same timeline.

        Parameters
        ----------
        time : pandas.DatetimeIndex, pandas.TimedeltaIndex, List[pandas.Timestamp], or \
               LocalCoordinateSystem
            The target time for the interpolation. In addition to the supported
            time formats, the function also accepts a `LocalCoordinateSystem` as
            ``time`` source object
        time_ref : pandas.Timestamp
            A reference timestamp that can be provided if the ``time`` parameter is a
            `pandas.TimedeltaIndex`
        affected_coordinate_systems : str or List[str]
            A single coordinate system name or a list of coordinate system names that
            should be interpolated in time. Only transformations towards the systems
            root node are affected.
        in_place : bool
            If 'True' the interpolation is performed in place, otherwise a
            new instance is returned. (Default value = False)

        Returns
        -------
        CoordinateSystemManager
            Coordinate system manager with interpolated data

        """
        if in_place:
            if affected_coordinate_systems is not None:
                if isinstance(affected_coordinate_systems, str):
                    affected_coordinate_systems = [affected_coordinate_systems]

                affected_edges = []
                for cs in affected_coordinate_systems:
                    ps = self.get_parent_system_name(cs)
                    affected_edges.append((cs, ps))
                    affected_edges.append((ps, cs))
            else:
                affected_edges = self._graph.edges

            for edge in affected_edges:
                if self._graph.edges[edge]["defined"]:
                    self._graph.edges[edge]["lcs"] = self._graph.edges[edge][
                        "lcs"
                    ].interp_time(time, time_ref)
            for edge in affected_edges:
                if not self._graph.edges[edge]["defined"]:
                    self._graph.edges[edge]["lcs"] = self._graph.edges[
                        (edge[1], edge[0])
                    ]["lcs"].invert()
            return self

        return deepcopy(self).interp_time(
            time, time_ref, affected_coordinate_systems, in_place=True
        )

    def is_neighbor_of(
        self, coordinate_system_name_0: str, coordinate_system_name_1: str
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

    def merge(self, other: "CoordinateSystemManager"):
        """Merge another coordinate system managers into the current instance.

        Both 'CoordinateSystemManager' need to have exactly one common coordinate
        system. They are merged at this node. Internally, information is kept
        to undo the merge process.

        Parameters
        ----------
        other:
            CoordinateSystemManager instance that should be merged into the current
            instance.

        """
        if (
            other._number_of_time_dependent_lcs > 0
            and self.reference_time != other.reference_time
        ):
            raise Exception(
                "You can only merge subsystems with time dependent coordinate systems"
                "if the reference times of both 'CoordinateSystemManager' instances"
                "are identical."
            )

        intersection = list(
            set(self.coordinate_system_names) & set(other.coordinate_system_names)
        )

        if len(intersection) != 1:
            raise ValueError(
                "Both instances must have exactly one common coordinate system. "
                f"Found the following common systems: {intersection}"
            )

        self._graph = nx.compose(self._graph, other.graph)

        subsystem_data = {
            "common node": intersection[0],
            "root": other.root_system_name,
            "time_ref": other.reference_time,
            "neighbors": other.neighbors(intersection[0]),
            "original members": other.coordinate_system_names,
            "sub system data": other.sub_system_data,
        }
        self._sub_system_data_dict[other.name] = subsystem_data

    def neighbors(self, coordinate_system_name: str) -> List:
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

    def _get_tree_positions_for_plot(self):
        """Create the position data for the plot function."""
        pos = {}
        lcs_names = [self.root_system_name]
        meta_data = {self._root_system_name: {"position": (1, 0), "boundaries": [0, 2]}}
        level = 1
        while lcs_names:
            lcs_names_next = []
            for lcs_name in lcs_names:
                children_names = self.get_child_system_names(lcs_name)
                num_children = len(children_names)
                if num_children == 0:
                    continue

                bound = meta_data[lcs_name]["boundaries"]
                delta = (bound[1] - bound[0]) / num_children

                for i, child_name in enumerate(children_names):
                    pos_child = [bound[0] + (i + 0.5) * delta, -level]
                    bound_child = [bound[0] + i * delta, bound[0] + (i + 1) * delta]
                    meta_data[child_name] = {
                        "position": pos_child,
                        "boundaries": bound_child,
                    }
                lcs_names_next += children_names

            level += 1
            lcs_names = lcs_names_next

        for child, data in meta_data.items():
            pos[child] = data["position"]
        return pos

    def plot(self):
        """Plot the graph of the coordinate system manager."""
        plt.figure()
        color_map = []
        pos = self._get_tree_positions_for_plot()

        graph = deepcopy(self._graph)  # TODO: Check if deepcopy is necessary
        # only plot inverted directional arrows
        remove_edges = [edge for edge in graph.edges if graph.edges[edge]["defined"]]
        graph.remove_edges_from(remove_edges)

        nx.draw(graph, pos, with_labels=True, font_weight="bold", node_color=color_map)

    def remove_subsystems(self):
        """Remove all subsystems from the coordinate system manager."""
        cs_delete = []
        for _, sub_system_data in self._sub_system_data_dict.items():
            for lcs in sub_system_data["neighbors"]:
                cs_delete += [lcs]

        self._sub_system_data_dict = {}
        for lcs in cs_delete:
            self.delete_cs(lcs, True)

    def time_union(
        self, list_of_edges: List = None,
    ) -> Union[None, pd.DatetimeIndex, pd.TimedeltaIndex]:
        """Get the time union of all or selected local coordinate systems.

         If neither the `CoordinateSystemManager` nor its attached
         `LocalCoordinateSystem` instances possess a reference time, the function
         returns a `pandas.TimedeltaIndex`. Otherwise, a `pandas.DatetimeIndex` is
         returned. The following table gives an overview of all possible reference time
         combinations and the corresponding return type:


        +------------+------------------+-------------------------+
        | CSM        | LCS              | Return type             |
        | reference  | reference        |                         |
        | time       | times            |                         |
        +============+==================+=========================+
        | True       | all/mixed/none   | `pandas.DatetimeIndex`  |
        +------------+------------------+-------------------------+
        | False      | all              | `pandas.DatetimeIndex`  |
        +------------+------------------+-------------------------+
        | False      | none             | `pandas.TimedeltaIndex` |
        +------------+------------------+-------------------------+



        Parameters
        ----------
        list_of_edges :
            If not `None`, the union is only calculated from the specified edges.

        Returns
        -------
        pandas.DatetimeIndex or pandas.TimedeltaIndex
            Time union

        """
        if list_of_edges is None:
            lcs_list = self.lcs_time_dependent
        else:
            lcs_list = [self.graph.edges[edge]["lcs"] for edge in list_of_edges]
            lcs_list = [lcs for lcs in lcs_list if lcs.is_time_dependent]

        if not lcs_list:
            return None

        time_list = [ut.to_pandas_time_index(lcs) for lcs in lcs_list]
        if self.has_reference_time:
            time_list = [
                t + self.reference_time if isinstance(t, pd.TimedeltaIndex) else t
                for t in time_list
            ]

        return ut.get_time_union(time_list)

    def transform_data(
        self,
        data: Union[xr.DataArray, np.ndarray, List],
        source_coordinate_system_name: str,
        target_coordinate_system_name: str,
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
        numpy.ndarray
            Transformed data

        """
        lcs = self.get_cs(source_coordinate_system_name, target_coordinate_system_name)
        if isinstance(data, xr.DataArray):
            mul = ut.xr_matmul(
                lcs.orientation, data, dims_a=["c", "v"], dims_b=["c"], dims_out=["c"]
            )
            return mul + lcs.coordinates

        data = ut.to_float_array(data)
        rotation = lcs.orientation.data
        translation = lcs.coordinates.data
        return ut.mat_vec_mul(rotation, data) + translation

    def unmerge(self) -> List["CoordinateSystemManager"]:
        """Undo previous merges and return a list of all previously merged instances.

        If additional coordinate systems were added after merging two instances, they
        won't be lost. Depending on their parent system, they will be kept in one of the
        returned sub-instances or the current instance. All new systems with the
        parent system being the shared node of two merged systems are kept in the
        current instance and won't be passed to the sub-instances.

        Returns
        -------
        List[CoordinateSystemManager]:
            A list containing previously merged 'CoordinateSystemManager' instances.

        """
        subsystems = self.subsystems
        self.remove_subsystems()

        return subsystems
