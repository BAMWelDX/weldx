"""Test functions of the visualization package."""

import weldx.visualization as vs
import weldx.transformations as tf

# pylint: disable=W0611
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

# pylint: enable=W0611

import matplotlib.pyplot as plt
import pandas as pd
import pytest


def test_plot_coordinate_system():
    """Test executing all possible code paths."""
    lcs_constant = tf.LocalCoordinateSystem()

    time = pd.DatetimeIndex(["2016-01-10", "2016-01-11", "2016-01-12"])
    orientation_tdp = [
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
        [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
    ]
    coordinates_tdp = [[0, 0, 1], [0, 0, 2], [0, -1, 0]]
    lcs_tdp = tf.LocalCoordinateSystem(
        orientation=orientation_tdp, coordinates=coordinates_tdp, time=time
    )

    fig = plt.figure()
    ax = fig.gca(projection="3d")

    vs.plot_coordinate_system(lcs_constant, ax, "g")
    vs.plot_coordinate_system(lcs_tdp, ax, "r", "2016-01-10")
    vs.plot_coordinate_system(lcs_tdp, ax, "b", "2016-01-11", time_idx=1)
    vs.plot_coordinate_system(lcs_tdp, ax, "y", "2016-01-12", "2016-01-12")

    # exceptions ------------------------------------------

    # label without color
    with pytest.raises(Exception):
        vs.plot_coordinate_system(lcs_constant, ax, label="label")


def test_set_axes_equal():
    """Test executing all possible code paths."""
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    vs.set_axes_equal(ax)
