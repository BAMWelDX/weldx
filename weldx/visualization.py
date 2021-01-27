"""Contains some functions to help with visualization."""

import k3d
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from ipywidgets import Dropdown, FloatSlider
from pandas import TimedeltaIndex

import weldx.utility as ut
from weldx.constants import WELDX_QUANTITY as Q_


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
            dsx = coordinate_system.dataset.sel(time=time_idx).isel(time=0)
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


def plot_coordinate_system_manager_matplotlib(
    csm,
    axes=None,
    reference_system=None,
    time=None,
    time_ref=None,
    show_trace=True,
    show_axes=True,
):
    if time is not None:
        plot_coordinate_system_manager_matplotlib(
            csm.interp_time(time=time, time_ref=time_ref),
            axes=axes,
            reference_system=reference_system,
            show_trace=show_trace,
            show_axes=show_axes,
        )
    else:
        if axes is None:
            _, axes = plt.subplots(
                subplot_kw={"projection": "3d", "proj_type": "ortho"}
            )
            axes.set_xlabel("x")
            axes.set_ylabel("y")
            axes.set_zlabel("z")

        if reference_system is None:
            reference_system = csm._root_system_name

        for lcs_name in csm.coordinate_system_names:
            # https://stackoverflow.com/questions/13831549/
            # get-matplotlib-color-cycle-state
            color = next(axes._get_lines.prop_cycler)["color"]
            lcs = csm.get_cs(lcs_name, reference_system)
            lcs.plot(
                axes=axes,
                color=color,
                label=lcs_name,
                show_trace=show_trace,
                show_axes=show_axes,
            )
        axes.legend()


# k3d ----------------------------------------------------------------------------------


class CoordinateSystemVisualizerK3D:
    def __init__(self, lcs, plot=None):
        self._lcs = lcs

        coordinates = np.array(lcs.coordinates.values, dtype="float32")
        orientation = np.array(lcs.orientation.values, dtype="float32")
        if lcs.is_time_dependent:
            coordinates = coordinates[0]
            orientation = orientation[0]
        orientation = orientation.transpose()

        self._vectors = k3d.vectors(
            origins=[coordinates for _ in range(3)],
            vectors=orientation,
            colors=[[0xFF0000, 0xFF0000], [0x00FF00, 0x00FF00], [0x0000FF, 0x0000FF]],
            labels=[],
            label_size=1.5,
        )

        if plot is not None:
            plot += self._vectors

    def update_time(self, time, time_ref=None):

        lcs = self._lcs.interp_time(time, time_ref)

        coordinates = np.array(lcs.coordinates.values, dtype="float32")
        orientation = np.array(lcs.orientation.values, dtype="float32")
        if lcs.is_time_dependent:
            coordinates = coordinates[0]
            orientation = orientation[0]
        orientation = orientation.transpose()

        self._vectors.origins = [coordinates for _ in range(3)]
        self._vectors.vectors = orientation

    def update_lcs(self, lcs):
        self._lcs = lcs


class CoordinateSystemManagerVisualizerK3D:
    def __init__(self, csm):
        self._csm = csm
        self._current_time = None
        plot = k3d.plot(camera_auto_fit=False)

        time_union = csm.time_union()

        if time_union is not None:
            if isinstance(time_union, TimedeltaIndex):
                time_union = ut.pandas_time_delta_to_quantity(time_union, "s")
                self._current_time = time_union[0]
            else:
                raise Exception("Only TimedeltaIndex supported at the moment.")

        self._time_slider = FloatSlider(
            min=time_union[0].magnitude,
            max=time_union[-1].magnitude,
            value=0,
            description="Time:",
        )

        root_name = csm._root_system_name
        self._lcs_vis = [
            CoordinateSystemVisualizerK3D(csm.get_cs(lcs_name, root_name), plot)
            for lcs_name in csm.coordinate_system_names
        ]

        def on_time_change(change):
            time = Q_(change["new"], "s")
            self._current_time = time
            self.update_time(time)

        self._time_slider.observe(on_time_change, names="value")

        def on_reference_change(change):
            if change["type"] == "change" and change["name"] == "value":
                self.update_reference_system(change["new"])
                print(change["new"])

        self._reference_dropdown = Dropdown(
            options=csm.coordinate_system_names,
            value=root_name,
            description="Reference system:",
            disabled=False,
        )
        self._reference_dropdown.observe(on_reference_change, names="value")

        plot.display()
        display(self._time_slider)
        display(self._reference_dropdown)

    def update_time(self, time, time_ref=None):
        for lcs_vis in self._lcs_vis:
            lcs_vis.update_time(time, time_ref)

    def update_reference_system(self, reference_system):
        for i, lcs_vis in enumerate(self._lcs_vis):

            lcs_vis.update_lcs(
                self._csm.get_cs(self._csm.coordinate_system_names[i], reference_system)
            )
            lcs_vis.update_time(self._current_time)


def plot_coordinate_cross_k3d(plot):
    origins = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    colors = [[0xFF0000, 0xFF0000], [0x00FF00, 0x00FF00], [0x0000FF, 0x0000FF]]
    cc = k3d.vectors(origins, vectors, colors=colors, labels=[], label_size=1.5)
    x_axis = k3d.line([[0, 0, 0], [1, 0, 0]], shader="mesh", width=0.05)
    plot += cc
    return cc


def plot_coordinate_system_manager_k3d(csm):
    pass
