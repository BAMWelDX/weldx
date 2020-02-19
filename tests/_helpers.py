import weldx.transformations as tf
import numpy as np
import math


def check_vectors_identical(a, b, tolerance=1E-9):
    a = np.array(a)
    b = np.array(b)
    assert a.size == b.size
    for i in range(a.size):
        assert math.isclose(a[i], b[i], abs_tol=tolerance)


def check_matrices_identical(a, b):
    assert a.shape == b.shape
    for i in range(3):
        for j in range(3):
            assert math.isclose(a[i, j], b[i, j], abs_tol=1E-9)


def rotated_coordinate_system(angle_x=np.pi / 3, angle_y=np.pi / 4,
                              angle_z=np.pi / 5, origin=np.array([0, 0, 0])):
    basis = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # rotate axes to produce a more general test case
    r_x = tf.rotation_matrix_x(angle_x)
    r_y = tf.rotation_matrix_y(angle_y)
    r_z = tf.rotation_matrix_z(angle_z)

    r_tot = np.matmul(r_z, np.matmul(r_y, r_x))

    rotated_basis = np.matmul(r_tot, basis)

    return tf.LocalCoordinateSystem(rotated_basis, np.array(origin))


def are_all_points_unique(data, decimals=3):
    unique = np.unique(np.round(data, decimals=decimals), axis=1)
    return (unique.shape[0] == data.shape[0] and
            unique.shape[1] == data.shape[1])
