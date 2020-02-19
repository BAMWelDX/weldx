"""Contains some functions to help with visualization."""

import numpy as np


def _check_color_key_valid(color_key):
    """
    Check if a color key is valid.

    :param color_key: Matplotlib color key letter (e.g. 'r' for red etc).
    :return: ---
    """
    valid_colors = ["r", "g", "b", "y", "c", "m", "k", "w"]
    if color_key not in valid_colors:
        raise Exception("Invalid color key.")


def plot_coordinate_system(coordinate_system, axes, color=None, label=None):
    """
    Plot a coordinate system in a matplotlib 3d plot.

    :param coordinate_system: Coordinate system
    :param axes: Matplotlib axes object (output from plt.gca())
    :param color: Matplotlib color key letter (e.g. 'r' for red etc). The
    origin of the coordinate system will be marked with this color.
    :param label: Name that appears in the legend. Only viable if a color
    was specified.
    :return: ---
    """
    p_0 = coordinate_system.origin
    p_x = p_0 + coordinate_system.orientation[:, 0]
    p_y = p_0 + coordinate_system.orientation[:, 1]
    p_z = p_0 + coordinate_system.orientation[:, 2]

    axes.plot([p_0[0], p_x[0]], [p_0[1], p_x[1]], [p_0[2], p_x[2]], "r")
    axes.plot([p_0[0], p_y[0]], [p_0[1], p_y[1]], [p_0[2], p_y[2]], "g")
    axes.plot([p_0[0], p_z[0]], [p_0[1], p_z[1]], [p_0[2], p_z[2]], "b")
    if color is not None:
        _check_color_key_valid(color)
        axes.plot([p_0[0]], [p_0[1]], [p_0[2]], color + "o", label=label)
    elif label is not None:
        raise Exception("Labels can only be assigned if a color was specified")


def set_axes_equal(axes):
    """
    Adjust axis in a 3d plot to be equally scaled.

    Source code taken from the stackoverflow answer of 'karlo' in the
    following question:
    https://stackoverflow.com/questions/13685386/matplotlib-equal-unit
    -length-with-equal-aspect-ratio-z-axis-is-not-equal-to

    :param axes: Matplotlib axes object (output from plt.gca())
    """
    x_limits = axes.get_xlim3d()
    y_limits = axes.get_ylim3d()
    z_limits = axes.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    axes.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    axes.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    axes.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
