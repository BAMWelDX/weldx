"""Provides some utility functions for tests."""

from typing import Any

import numpy as np

from weldx.transformations import LocalCoordinateSystem, WXRotation


def rotated_coordinate_system(
    angle_x=np.pi / 3,
    angle_y=np.pi / 4,
    angle_z=np.pi / 5,
    coordinates=np.array([0, 0, 0]),
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

    return LocalCoordinateSystem(rotated_orientation, np.array(coordinates))


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
