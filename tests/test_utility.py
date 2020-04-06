"""Test the internal utility functions."""

import numpy as np
import pandas as pd
import xarray as xr
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


def test_xr_interp_like():
    """
    Test behaviour of custom interpolation method for xarray Objects.

    :return: ---
    """
    # basic interpolation behavior on a single coordinate
    n_a = 10  # range of "a" coordinate in da1
    s_a = 0.5  # default steps in "a" coordinate in da1
    da1 = xr.DataArray(
        np.arange(0, n_a + s_a, s_a),
        dims=["a"],
        coords={"a": np.arange(0, n_a + s_a, s_a)},
    )

    # interp from subset inside original data
    test = ut.xr_interp_like(da1.loc[2:7:2], da1)
    assert test.a[0] == da1.a[0]
    assert test.a[-1] == da1.a[-1]
    assert test[0] == da1.loc[2]
    assert test[-1] == da1.loc[7]

    # interp with overlap
    test = ut.xr_interp_like(da1.loc[1:6], da1.loc[4:8])
    assert test.a[0] == da1.a.loc[4]
    assert test.a[-1] == da1.a.loc[8]
    assert test[0] == da1.loc[4]
    assert test[-1] == da1.loc[6]
    assert np.all(test.loc[6:8] == da1.loc[6])

    # overlap without fill (expecting nan values for out of range indexes)
    test = ut.xr_interp_like(da1.loc[1:6], da1.loc[4:8], fillna=False)
    assert test.a[0] == da1.a.loc[4]
    assert test.a[-1] == da1.a.loc[8]
    assert test[0] == da1.loc[4]
    assert np.isnan(test[-1])
    assert np.all(np.isnan(test.where(test.a > 6, drop=True)))

    # outside interpolation without overlap
    test = ut.xr_interp_like(da1.loc[2:4:2], da1.loc[7:9])
    assert test.a[0] == da1.a.loc[7]
    assert test.a[-1] == da1.a.loc[9]
    assert np.all(test.loc[7:9] == da1.loc[4])

    # single point broadcasting
    # important: da1.loc[5] for indexing would drop coordinates (unsure why)
    test = ut.xr_interp_like(da1.loc[5:5], da1)
    assert np.all(test.a == da1.a)
    assert np.all(test == da1.loc[5:5])

    # TODO: complex tests with multiple dimensions
    # basic interpolation behavior with different coordinates (broadcasting)
    n_b = 7  # range of "b" coordinate in da2
    s_b = 1  # default steps in "b" coordinate in da2
    da2 = xr.DataArray(
        np.arange(0, n_b + s_b, s_b) ** 2,
        dims=["b"],
        coords={"b": np.arange(0, n_b + s_b, s_b)},
    )

    assert da1.broadcast_equals(ut.xr_interp_like(da1, da2))
    assert da1.broadcast_like(da2).broadcast_equals(
        ut.xr_interp_like(da1, da2, broadcast_missing=True)
    )

    # tests with time dtypes
    t = pd.timedelta_range(start="0s", end="10s", freq="1s")
    da3 = xr.DataArray(np.arange(0, 11, 1), dims=["t"], coords={"t": t})
