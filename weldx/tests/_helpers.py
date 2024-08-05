"""Provides some utility functions for tests."""

from typing import Any

import numpy as np
import pint

from weldx.constants import Q_
from weldx.geometry import _vector_is_close as vector_is_close
from weldx.transformations import LocalCoordinateSystem, WXRotation

__all__ = [
    "rotated_coordinate_system",
    "are_all_columns_unique",
    "get_test_name",
    "vector_is_close",
    "matrix_is_close",
]


def rotated_coordinate_system(
    angle_x=np.pi / 3,
    angle_y=np.pi / 4,
    angle_z=np.pi / 5,
    coordinates=np.array([0, 0, 0]),  # noqa B008
) -> LocalCoordinateSystem:
    """Get a coordinate system with rotated orientation.

    The transformation order is x-y-z

    Parameters
    ----------
    angle_x :
        Rotation angle around the x axis (Default value = np.pi / 3)
    angle_y :
        Rotation angle around the y axis (Default value = np.pi / 4)
    angle_z :
        Rotation angle around the z axis (Default value = np.pi / 5)
    coordinates :
        Coordinates of the coordinate system (Default value = np.array([0, 0, 0]))


    Returns
    -------
    weldx.transformations.LocalCoordinateSystem
        Coordinate system with rotated orientation

    """
    orientation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # rotate axes to produce a more general test case
    r_x = WXRotation.from_euler("x", angle_x).as_matrix()
    r_y = WXRotation.from_euler("y", angle_y).as_matrix()
    r_z = WXRotation.from_euler("z", angle_z).as_matrix()

    r_tot = np.matmul(r_z, np.matmul(r_y, r_x))

    rotated_orientation = np.matmul(r_tot, orientation)

    if not isinstance(coordinates, Q_):
        coordinates = np.array(coordinates)
    return LocalCoordinateSystem(rotated_orientation, coordinates)


def are_all_columns_unique(matrix, decimals=3):
    """Check if all columns in a matrix are unique.

    Parameters
    ----------
    matrix :
        Matrix
    decimals :
        Number of decimals that should be considered during
        comparison (Default value = 3)

    Returns
    -------
    bool
        True or False

    """
    unique = np.unique(np.round(matrix, decimals=decimals), axis=1)
    return unique.shape[0] == matrix.shape[0] and unique.shape[1] == matrix.shape[1]


def get_test_name(param: Any) -> str:
    """Get the test name from the parameter list of a parametrized test.

    Parameters
    ----------
    param : Any
        A parameter of the test

    Returns
    -------
    str :
        The name of the test or an empty string.

    """
    if isinstance(param, str) and param[0] == "#":
        return param[1:]
    return ""


def matrix_is_close(mat_a, mat_b, abs_tol=1e-9) -> bool:
    """Check if a matrix is close or equal to another matrix.

    Parameters
    ----------
    mat_a :
        First matrix
    mat_b :
        Second matrix
    abs_tol :
        Absolute tolerance (Default value = 1e-9)

    Returns
    -------
    bool
        True or False

    """
    if not isinstance(mat_a, pint.Quantity):
        mat_a = np.array(mat_a, dtype=float)
    if not isinstance(mat_b, pint.Quantity):
        mat_b = np.array(mat_b, dtype=float)

    if mat_a.shape != mat_b.shape:
        return False

    atol_unit = 1.0
    if isinstance(mat_b, pint.Quantity):
        atol_unit = mat_b.u

    return np.all(np.isclose(mat_a, mat_b, atol=abs_tol * atol_unit)).__bool__()
