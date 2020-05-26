"""Test functions of the visualization package."""

import weldx.visualization as vs
import weldx.transformations as tf

# pylint: disable=W0611
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

# pylint: enable=W0611

import matplotlib.pyplot as plt
import pytest


def test_plot_coordinate_system():
    """Test executing all possible code paths."""
    cs = tf.LocalCoordinateSystem()
    fig = plt.figure()
    ax = fig.gca(projection="3d")

    vs.plot_coordinate_system(cs, ax, "g")
    vs.plot_coordinate_system(cs, ax, "r", "test")

    # exceptions ------------------------------------------

    # label without color
    with pytest.raises(Exception):
        vs.plot_coordinate_system(cs, ax, label="label")


def test_set_axes_equal():
    """Test executing all possible code paths."""
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    vs.set_axes_equal(ax)
