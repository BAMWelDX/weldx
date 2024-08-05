from __future__ import annotations

from typing import Any

import numpy as np
from xarray import DataArray

import weldx.transformations as tf
from weldx.time import Time
from weldx.transformations import WXRotation


def check_coordinate_system_orientation(
    orientation: DataArray,
    orientation_expected: np.ndarray,
    positive_orientation_expected: bool,
):
    """Check if the orientation of a local coordinate system is as expected.

    Parameters
    ----------
    orientation :
        Orientation
    orientation_expected :
        Expected orientation
    positive_orientation_expected :
        True, if the orientation is expected to be
        positive. False otherwise.

    """
    # test expected positive orientation
    det = np.linalg.det(orientation.sel(v=[2, 0, 1]))
    assert np.all((det > 0) == positive_orientation_expected)

    assert tf.is_orthogonal_matrix(orientation.values)

    orientation_expected = tf.normalize(orientation_expected)

    assert np.allclose(orientation, orientation_expected)


def check_coordinate_system(
    lcs: tf.LocalCoordinateSystem,
    orientation_expected: np.ndarray | list[list[Any]] | DataArray,
    coordinates_expected: np.ndarray | list[Any] | DataArray,
    positive_orientation_expected: bool = True,
    time=None,
    time_ref=None,
):
    """Check the values of a coordinate system.

    Parameters
    ----------
    lcs :
        Coordinate system that should be checked
    orientation_expected :
        Expected orientation
    coordinates_expected :
        Expected coordinates
    positive_orientation_expected :
        Expected orientation
    time :
        A pandas.DatetimeIndex object, if the coordinate system is expected to
        be time dependent. None otherwise.
    time_ref:
        The expected reference time

    """
    orientation_expected = np.array(orientation_expected)

    if time is not None:
        assert orientation_expected.ndim == 3 or coordinates_expected.ndim == 2
        assert np.all(lcs.time == Time(time, time_ref))
        assert lcs.reference_time == time_ref

    check_coordinate_system_orientation(
        lcs.orientation, orientation_expected, positive_orientation_expected
    )

    atol_unit = coordinates_expected.u

    assert np.allclose(
        lcs.coordinates.data, coordinates_expected, atol=1e-9 * atol_unit
    )


def check_cs_close(lcs_0, lcs_1):
    """Check if 2 coordinate systems are nearly identical.

    Parameters
    ----------
    lcs_0:
        First coordinate system.
    lcs_1
        Second coordinate system.

    """
    check_coordinate_system(
        lcs_0,
        lcs_1.orientation.data,
        lcs_1.coordinates.data,
        True,
        lcs_1.time,
        lcs_1.reference_time,
    )


def r_mat_x(factors) -> np.ndarray:
    """Get an array of rotation matrices that represent a rotation around the x-axis.

    The rotation angles are the provided factors times pi.

    Parameters
    ----------
    factors:
        List of factors that are multiplied with pi to get the rotation angles.

    Returns
    -------
    numpy.ndarray:
        An array of rotation matrices

    """
    return WXRotation.from_euler("x", np.array(factors) * np.pi).as_matrix()


def r_mat_y(factors) -> np.ndarray:
    """Get an array of rotation matrices that represent a rotation around the y-axis.

    The rotation angles are the provided factors times pi.

    Parameters
    ----------
    factors:
        List of factors that are multiplied with pi to get the rotation angles.

    Returns
    -------
    numpy.ndarray:
        An array of rotation matrices

    """
    return WXRotation.from_euler("y", np.array(factors) * np.pi).as_matrix()


def r_mat_z(factors) -> np.ndarray:
    """Get an array of rotation matrices that represent a rotation around the z-axis.

    The rotation angles are the provided factors times pi.

    Parameters
    ----------
    factors:
        List of factors that are multiplied with pi to get the rotation angles.

    Returns
    -------
    numpy.ndarray:
        An array of rotation matrices

    """
    return WXRotation.from_euler("z", np.array(factors) * np.pi).as_matrix()
