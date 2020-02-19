"""Test the internal utility functions."""

import numpy as np
import weldx.utility as ut


def test_is_column_in_matrix():
    """
    Test the is_column_in_matrix function.

    Test should be self explanatory.

    :return: ---
    """
    c_0 = [1, 5, 2]
    c_1 = [3, 2, 2]
    c_2 = [1, 6, 1]
    c_3 = [1, 6, 0]
    matrix = np.array([c_0, c_1, c_2, c_3]).transpose()

    assert ut.is_column_in_matrix(c_0, matrix)
    assert ut.is_column_in_matrix(c_1, matrix)
    assert ut.is_column_in_matrix(c_2, matrix)
    assert ut.is_column_in_matrix(c_3, matrix)

    assert not ut.is_column_in_matrix([1, 6], matrix)
    assert not ut.is_column_in_matrix([1, 6, 2], matrix)
    assert not ut.is_column_in_matrix([1, 1, 3, 1], matrix)


def test_is_row_in_matrix():
    """
    Test the is_row_in_matrix function.

    Test should be self explanatory.

    :return: ---
    """
    c_0 = [1, 5, 2]
    c_1 = [3, 2, 2]
    c_2 = [1, 6, 1]
    c_3 = [1, 6, 0]
    matrix = np.array([c_0, c_1, c_2, c_3])

    assert ut.is_row_in_matrix(c_0, matrix)
    assert ut.is_row_in_matrix(c_1, matrix)
    assert ut.is_row_in_matrix(c_2, matrix)
    assert ut.is_row_in_matrix(c_3, matrix)

    assert not ut.is_row_in_matrix([1, 6], matrix)
    assert not ut.is_row_in_matrix([1, 6, 2], matrix)
    assert not ut.is_row_in_matrix([1, 1, 3, 1], matrix)


def test_matrix_is_close():
    """
    Test the matrix_is_close function.

    Test should be self explanatory.

    :return: ---
    """
    mat_a = np.array([[0, 1, 2], [3, 4, 5]])
    mat_b = np.array([[3, 5, 1], [7, 1, 9]])

    assert ut.matrix_is_close(mat_a, mat_a)
    assert ut.matrix_is_close(mat_b, mat_b)
    assert not ut.matrix_is_close(mat_a, mat_b)
    assert not ut.matrix_is_close(mat_b, mat_a)

    # check tolerance
    mat_c = mat_a + 0.0001
    assert ut.matrix_is_close(mat_a, mat_c, abs_tol=0.00011)
    assert not ut.matrix_is_close(mat_a, mat_c, abs_tol=0.00009)

    # vectors have different size
    assert not ut.matrix_is_close(mat_a, mat_a[0:2, 0:2])


def test_vector_is_close():
    """
    Test the vector_is_close function.

    Test should be self explanatory.

    :return: ---
    """
    vec_a = np.array([0, 1, 2])
    vec_b = np.array([3, 5, 1])

    assert ut.vector_is_close(vec_a, vec_a)
    assert ut.vector_is_close(vec_b, vec_b)
    assert not ut.vector_is_close(vec_a, vec_b)
    assert not ut.vector_is_close(vec_b, vec_a)

    # check tolerance
    vec_c = vec_a + 0.0001
    assert ut.vector_is_close(vec_a, vec_c, abs_tol=0.00011)
    assert not ut.vector_is_close(vec_a, vec_c, abs_tol=0.00009)

    # vectors have different size
    assert not ut.vector_is_close(vec_a, vec_a[0:2])
