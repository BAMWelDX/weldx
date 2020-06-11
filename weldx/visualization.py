"""Contains some functions to help with visualization."""

import numpy as np
import pandas as pd


def plot_coordinate_system(
    coordinate_system, axes, color=None, label=None, time_idx=None
):
    """Plot a coordinate system in a matplotlib 3d plot.

    Parameters
    ----------
    coordinate_system :
        Coordinate system
    axes :
        Matplotlib axes object (output from plt.gca())
    color :
        Valid matplotlib color selection. The origin of the coordinate system
        will be marked with this color. (Default value = None)
    label :
        Name that appears in the legend. Only viable if a color
        was specified. (Default value = None)
    time_idx :
        Selects time dependent data by index if the coordinate system has
        a time dependency.

    """
    if "time" in coordinate_system.dataset.coords:
        if time_idx is None:
            time_idx = 0
        if isinstance(time_idx, int):
            dsx = coordinate_system.dataset.isel(time=time_idx)
        else:
            dsx = coordinate_system.dataset.sel(time=pd.DatetimeIndex([time_idx])).isel(
                time=0
            )
    else:
        dsx = coordinate_system.dataset

    p_0 = dsx.coordinates

    orientation = dsx.orientation
    p_x = p_0 + orientation[:, 0]
    p_y = p_0 + orientation[:, 1]
    p_z = p_0 + orientation[:, 2]

    axes.plot([p_0[0], p_x[0]], [p_0[1], p_x[1]], [p_0[2], p_x[2]], "r")
    axes.plot([p_0[0], p_y[0]], [p_0[1], p_y[1]], [p_0[2], p_y[2]], "g")
    axes.plot([p_0[0], p_z[0]], [p_0[1], p_z[1]], [p_0[2], p_z[2]], "b")
    if color is not None:
        axes.plot([p_0[0]], [p_0[1]], [p_0[2]], "o", color=color, label=label)
    elif label is not None:
        raise Exception("Labels can only be assigned if a color was specified")


def set_axes_equal(axes):
    """Adjust axis in a 3d plot to be equally scaled.

    Source code taken from the stackoverflow answer of 'karlo' in the
    following question:
    https://stackoverflow.com/questions/13685386/matplotlib-equal-unit
    -length-with-equal-aspect-ratio-z-axis-is-not-equal-to

    Parameters
    ----------
    axes :
        Matplotlib axes object (output from plt.gca())

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
