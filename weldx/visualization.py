"""Contains some functions to help with visualization."""

import k3d
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from ipywidgets import Dropdown, HBox, IntSlider, Play, jslink

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
    """Plot the coordinate systems of a `CoordinateSystemManager` using matplotlib.

    Parameters
    ----------
    csm : weldx.CoordinateSystemManager
        The `CoordinateSystemManager` that should be plotted
    axes :
        The target axes object that should be drawn to. If `None` is provided, a new
        one will be created.
    reference_system : str
        The name of the reference system for the plotted coordinate systems
    time : pandas.DatetimeIndex, pandas.TimedeltaIndex, List[pandas.Timestamp], or \
           LocalCoordinateSystem
        The time steps that should be plotted
    time_ref : pandas.Timestamp
        A reference timestamp that can be provided if the ``time`` parameter is a
        `pandas.TimedeltaIndex`
    show_trace : bool
        If `True`, the trace of time dependent coordinate systems is plotted.
    show_axes :
        If `True`, the coordinate cross of time dependent coordinate systems is plotted.

    Returns
    -------

    """
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
    """Visualizes a `weldx.LocalCoordinateSystem` using k3d."""

    def __init__(self, lcs, plot=None, name=None, color=0x000000):
        """Create a `CoordinateSystemVisualizerK3D`

        Parameters
        ----------
        lcs : weldx.LocalCoordinateSystem
            Coordinate system that should be visualized
        plot :
            A k3d plotting widget.
        name : str
            Name of the coordinate system
        color :
            The color of the coordinate system (affects trace and label)

        """
        coordinates, orientation = self._get_coordinates_and_orientation(lcs)
        self._lcs = lcs
        self._color = color

        self._vectors = k3d.vectors(
            origins=[coordinates for _ in range(3)],
            vectors=orientation,
            colors=[[0xFF0000, 0xFF0000], [0x00FF00, 0x00FF00], [0x0000FF, 0x0000FF]],
            labels=[],
            label_size=1.5,
        )

        self._label = None
        if name is not None:
            self._label = k3d.text(
                text=name,
                position=coordinates + 0.1,
                color=self._color,
                size=1,
                label_box=False,
            )

        self._trace = k3d.line(
            np.array(lcs.coordinates.values, dtype="float32"),
            shader="simple",
            width=0.05,
            color=color,
        )

        if plot is not None:
            plot += self._vectors
            plot += self._trace
            if self._label is not None:
                plot += self._label

    def _update_positions(self, coordinates, orientation):
        """Update the positions of the coordinate cross and label.

        Parameters
        ----------
        coordinates : numpy.ndarray
            The new coordinates
        orientation : numpy.ndarray
            The new orientation

        """
        self._vectors.origins = [coordinates for _ in range(3)]
        self._vectors.vectors = orientation
        if self._label is not None:
            self._label.position = coordinates + 0.1

    @staticmethod
    def _get_coordinates_and_orientation(lcs, index=0):
        """Get the coordinates and orientation of a coordinate system

        Parameters
        ----------
        lcs : weldx.LocalCoordinateSystem
            The coordinate system
        index :
            If the coordinate system is time dependent, the passed value is the index
            of the values that should be returned

        Returns
        -------
        coordinates : numpy.ndarray
            The coordinates
        orientation : numpy.ndarray
            The orientation

        """
        coordinates = np.array(lcs.coordinates.values, dtype="float32")
        orientation = np.array(lcs.orientation.values, dtype="float32")
        if lcs.is_time_dependent:
            coordinates = coordinates[index]
            orientation = orientation[index]
        orientation = orientation.transpose()

        return coordinates, orientation

    def update_lcs(self, lcs, index: int = 0):
        """Pass a new coordinate system to the visualizer.

        Parameters
        ----------
        lcs : weldx.LocalCoordinateSystem
            The new coordinate system
        index : int
            The time index of the new coordinate system that should be visualized.

        """
        self._lcs = lcs
        self._trace.vertices = np.array(lcs.coordinates.values, dtype="float32")
        self.update_time_index(index)

    def update_time(self, time, time_ref=None):
        """Update the plotted time step.

        Parameters
        ----------
        time : pandas.DatetimeIndex, pandas.TimedeltaIndex, List[pandas.Timestamp], or \
               LocalCoordinateSystem
            The time steps that should be plotted
        time_ref : pandas.Timestamp
            A reference timestamp that can be provided if the ``time`` parameter is a
            `pandas.TimedeltaIndex`

        """
        lcs = self._lcs.interp_time(time, time_ref)
        coordinates, orientation = self._get_coordinates_and_orientation(lcs)

        self._update_positions(coordinates, orientation)

    def update_time_index(self, index):
        """Update the plotted time step.

        Parameters
        ----------
        index : int
            The array index of the time step

        """
        coordinates, orientation = self._get_coordinates_and_orientation(
            self._lcs, index
        )
        self._update_positions(coordinates, orientation)


class CoordinateSystemManagerVisualizerK3D:
    """Visualizes a `weldx.CoordinateSystemManager` using k3d."""

    color_table = [0xFF0000, 0x00FF00, 0x0000FF, 0xFFFF00, 0xFF00FF, 0x00FFFF]

    def __init__(self, csm):
        """Create a `CoordinateSystemManagerVisualizerK3D`.

        Parameters
        ----------
        csm : weldx.CoordinateSystemManager
            The `CoordinateSystemManager` that should be visualized

        """
        time = csm.time_union()
        num_times = len(time)

        self._csm = csm.interp_time(time)
        self._current_time_index = 0
        root_name = csm._root_system_name

        # create controls
        play = Play(min=0, max=num_times - 1, value=self._current_time_index, step=1)
        time_slider = IntSlider(
            min=0,
            max=num_times - 1,
            value=self._current_time_index,
            description="Time:",
        )
        reference_dropdown = Dropdown(
            options=csm.coordinate_system_names,
            value=root_name,
            description="Reference:",
            disabled=False,
        )
        jslink((play, "value"), (time_slider, "value"))
        self._controls = HBox([time_slider, play, reference_dropdown])

        # callback functions
        def on_reference_change(change):
            """Handle events of the reference system drop down.

            Parameters
            ----------
            change : Dict
                A dictionary containing the event data

            """
            self.update_reference_system(change["new"])

        def on_time_change(change):
            """Handle events of the time slider.

            Parameters
            ----------
            change : Dict
                A dictionary containing the event data

            """
            self._current_time_index = change["new"]
            self.update_time_index(self._current_time_index)

        # register callbacks
        time_slider.observe(on_time_change, names="value")
        reference_dropdown.observe(on_reference_change, names="value")

        # create plot
        plot = k3d.plot(grid_auto_fit=False, camera_auto_fit=False)
        self._lcs_vis = {
            lcs_name: CoordinateSystemVisualizerK3D(
                csm.get_cs(lcs_name, root_name),
                plot,
                lcs_name,
                color=self.color_table[i % len(self.color_table)],
            )
            for i, lcs_name in enumerate(csm.coordinate_system_names)
        }

        # display everything
        plot.display()
        display(self._controls)

    def update_time(self, time, time_ref=None):
        """Update the plotted time.

        Parameters
        ----------
        time : pandas.DatetimeIndex, pandas.TimedeltaIndex, List[pandas.Timestamp], or \
            LocalCoordinateSystem
            The time steps that should be plotted
        time_ref : pandas.Timestamp
            A reference timestamp that can be provided if the ``time`` parameter is a
            `pandas.TimedeltaIndex`

        """
        for _, lcs_vis in self._lcs_vis.items():
            lcs_vis.update_time(time, time_ref)

    def update_time_index(self, index):
        """Update the plotted time by index.

        Parameters
        ----------
        index : int
            The new index

        """
        for _, lcs_vis in self._lcs_vis.items():
            lcs_vis.update_time_index(index)

    def update_reference_system(self, reference_system):
        """Update the reference system of the plot.

        Parameters
        ----------
        reference_system : str
            Name of the new reference system

        """
        for lcs_name in self._csm.coordinate_system_names:

            self._lcs_vis[lcs_name].update_lcs(
                self._csm.get_cs(lcs_name, reference_system), self._current_time_index
            )
