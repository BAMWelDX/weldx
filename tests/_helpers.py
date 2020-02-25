"""Provides some utility functions for tests."""

import weldx.transformations as tf
import numpy as np


def rotated_coordinate_system(
    angle_x=np.pi / 3, angle_y=np.pi / 4, angle_z=np.pi / 5, origin=np.array([0, 0, 0])
):
    """
    Get a coordinate system with rotated basis.

    The transformation order is x-y-z

    :param angle_x: Rotation angle around the x axis
    :param angle_y: Rotation angle around the y axis
    :param angle_z: Rotation angle around the z axis
    :param origin: Origin of the coordinate system
    :return: Coordinate system with rotated basis
    """
    basis = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # rotate axes to produce a more general test case
    r_x = tf.rotation_matrix_x(angle_x)
    r_y = tf.rotation_matrix_y(angle_y)
    r_z = tf.rotation_matrix_z(angle_z)

    r_tot = np.matmul(r_z, np.matmul(r_y, r_x))

    rotated_basis = np.matmul(r_tot, basis)

    return tf.LocalCoordinateSystem(rotated_basis, np.array(origin))


def are_all_columns_unique(matrix, decimals=3):
    """
    Check if all columns in a matrix are unique.

    :param matrix: Matrix
    :param decimals: Number of decimals that should be considered during
    comparison
    :return: True or False
    """
    unique = np.unique(np.round(matrix, decimals=decimals), axis=1)
    return unique.shape[0] == matrix.shape[0] and unique.shape[1] == matrix.shape[1]
