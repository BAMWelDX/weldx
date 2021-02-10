"""Contains some functions to help with visualization."""

import k3d
import k3d.platonic as platonic
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from ipywidgets import Checkbox, Dropdown, HBox, IntSlider, Layout, Play, VBox, jslink

import weldx.geometry as geo


def random_color_rgb():
    return np.random.choice(range(256), size=3)


def color_rgb_to_int(rgb_color_tuple):
    return int("0x{:02x}{:02x}{:02x}".format(*rgb_color_tuple), 0)


def color_int_to_rgb(integer):
    return ((integer >> 16) & 255, (integer >> 8) & 255, integer & 255)


def color_rgb_to_rgb_normalized(rgb):
    return tuple([val / 255 for val in rgb])


def color_int_to_rgb_normalized(integer):
    rgb = color_int_to_rgb(integer)
    return color_rgb_to_rgb_normalized(rgb)


_color_list = [
    0xFF0000,
    0x00AA00,
    0x0000FF,
    0xAAAA00,
    0xFF00FF,
    0x00FFFF,
    # todo: add colors manually or find better solution since random colors are not
    #       guaranteed to have good visibility and to differ significantly from already
    #       existing colors
    *[color_rgb_to_int(random_color_rgb()) for _ in range(100)],
]


def color_generator_function():
    for color in _color_list:
        yield color
    raise Exception("No more colors available.")


def _get_color(lcs_name, color_dict, color_generator):
    if color_dict is not None and lcs_name in color_dict:
        return color_rgb_to_int(color_dict[lcs_name])
    return next(color_generator)


def new_3d_figure_and_axes(
    num_subplots: int = 1, height: int = 500, width: int = 500, pixel_per_inch: int = 50
):
    """Get a matplotlib figure and axes for 3d plots.

    Parameters
    ----------
    num_subplots : int
        Number of subplots (horizontal)
    height : int
        Height in pixels
    width : int
        Width in pixels
    pixel_per_inch :
        Defines how many pixels an inch covers. This is only relevant fon the fallback
        method.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object
    ax : matplotlib..axes.Axes
        The matplotlib axes object

    """
    fig, ax = plt.subplots(
        ncols=num_subplots, subplot_kw={"projection": "3d", "proj_type": "ortho"}
    )
    try:
        fig.canvas.layout.height = f"{height}px"
        fig.canvas.layout.width = f"{width}px"
    except:
        fig.set_size_inches(w=width / pixel_per_inch, h=height / pixel_per_inch)
    return fig, ax


def plot_coordinate_system(
    coordinate_system,
    axes,
    color=None,
    label=None,
    time_idx=None,
    show_origin=True,
    show_vectors=True,
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
    if not (show_vectors or show_origin):
        return
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

    if show_vectors:
        orientation = dsx.orientation
        p_x = p_0 + orientation[:, 0]
        p_y = p_0 + orientation[:, 1]
        p_z = p_0 + orientation[:, 2]

        axes.plot([p_0[0], p_x[0]], [p_0[1], p_x[1]], [p_0[2], p_x[2]], "r")
        axes.plot([p_0[0], p_y[0]], [p_0[1], p_y[1]], [p_0[2], p_y[2]], "g")
        axes.plot([p_0[0], p_z[0]], [p_0[1], p_z[1]], [p_0[2], p_z[2]], "b")
    if color is not None:
        if show_origin:
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


def _get_color_matplotlib(axes):
    color_table = ["r", "g", "b", "y"]
    for color in color_table:
        yield color
    yield next(axes._get_lines.prop_cycler)["color"]


def plot_local_coordinate_system_matplotlib(
    lcs,
    axes=None,
    color=None,
    label=None,
    time=None,
    time_ref=None,
    time_index=None,
    show_origin=True,
    show_trace=True,
    show_vectors=True,
):
    if axes is None:
        _, axes = plt.subplots(subplot_kw={"projection": "3d", "proj_type": "ortho"})

    if lcs.is_time_dependent and time is not None:
        lcs = lcs.interp_time(time, time_ref)

    if lcs.is_time_dependent and time_index is None:
        for i, _ in enumerate(lcs.time):
            plot_coordinate_system(
                lcs,
                axes,
                color=color,
                label=label,
                time_idx=i,
                show_origin=show_origin,
                show_vectors=show_vectors,
            )
            label = None
    else:
        plot_coordinate_system(
            lcs,
            axes,
            color=color,
            label=label,
            time_idx=time_index,
            show_origin=show_origin,
            show_vectors=show_vectors,
        )

    if show_trace and lcs.coordinates.values.ndim > 1:
        coords = lcs.coordinates.values
        if color is None:
            color = "k"
        axes.plot(coords[:, 0], coords[:, 1], coords[:, 2], ":", color=color)

    return axes


def plot_coordinate_system_manager_matplotlib(
    csm,
    axes=None,
    reference_system=None,
    coordinate_systems=None,
    data_sets=None,
    colors=None,
    time=None,
    time_ref=None,
    title=None,
    limits=None,
    show_origins=True,
    show_trace=True,
    show_vectors=True,
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
    show_vectors :
        If `True`, the coordinate cross of time dependent coordinate systems is plotted.

    Returns
    -------

    """
    if time is not None:
        return plot_coordinate_system_manager_matplotlib(
            csm.interp_time(time=time, time_ref=time_ref),
            axes=axes,
            reference_system=reference_system,
            coordinate_systems=coordinate_systems,
            title=title,
            show_origins=show_origins,
            show_trace=show_trace,
            show_vectors=show_vectors,
        )
    else:
        if axes is None:
            _, axes = new_3d_figure_and_axes()
            axes.set_xlabel("x")
            axes.set_ylabel("y")
            axes.set_zlabel("z")

        if reference_system is None:
            reference_system = csm._root_system_name
        if coordinate_systems is None:
            coordinate_systems = csm.coordinate_system_names
        if data_sets is None:
            data_sets = csm.data_names
        if title is not None:
            axes.set_title(title)

        # plot coordinate systems
        color_gen = color_generator_function()
        for lcs_name in coordinate_systems:
            # https://stackoverflow.com/questions/13831549/
            # get-matplotlib-color-cycle-state
            color = color_int_to_rgb_normalized(_get_color(lcs_name, colors, color_gen))
            lcs = csm.get_cs(lcs_name, reference_system)
            lcs.plot(
                axes=axes,
                color=color,
                label=lcs_name,
                show_origin=show_origins,
                show_trace=show_trace,
                show_vectors=show_vectors,
            )
        # plot data
        for data_name in data_sets:
            color = color_int_to_rgb_normalized(_get_color(lcs_name, colors, color_gen))
            data = csm.get_data(data_name, reference_system)
            triangles = None
            if isinstance(data, geo.PointCloud):
                triangles = data.triangles
                data = data.coordinates

            data = data.data
            while data.ndim > 2:
                data = data[0]

            axes.plot(
                data[:, 0], data[:, 1], data[:, 2], "x", color=color, label=data_name
            )
            if triangles is not None:
                for triangle in triangles:
                    triangle_data = data[[*triangle, triangle[0]], :]
                    axes.plot(
                        triangle_data[:, 0],
                        triangle_data[:, 1],
                        triangle_data[:, 2],
                        color=color,
                    )

        if limits is None:
            set_axes_equal(axes)
        else:
            axes.set_xlim(limits)
            axes.set_ylim(limits)
            axes.set_zlim(limits)
        try:
            axes.legend()
        except:
            pass
        return axes


def plot_coordinate_systems(
    cs_data,
    axes=None,
    title: str = None,
    limits=None,
    time_index=None,
    legend_pos="lower left",
):
    # todo Use kwargs dict instead of tuple
    for i, data in enumerate(cs_data):
        if len(data) == 3:
            cs_data[i] = (*data, None)

    if axes is None:
        _, axes = new_3d_figure_and_axes()

    for lcs, color, label, time_index_instance in cs_data:
        if time_index_instance is None:
            time_index_instance = time_index
        lcs.plot(axes, color=color, label=label, time_index=time_index_instance)

    if limits is None:
        set_axes_equal(axes)
    else:
        axes.set_xlim(limits)
        axes.set_ylim(limits)
        axes.set_zlim(limits)

    if title is not None:
        axes.set_title(title)
    axes.legend(loc=legend_pos)


# k3d ----------------------------------------------------------------------------------


class CoordinateSystemVisualizerK3D:
    """Visualizes a `weldx.LocalCoordinateSystem` using k3d."""

    def __init__(
        self,
        lcs,
        plot=None,
        name=None,
        color=0x000000,
        show_origin=True,
        show_trace=True,
        show_vectors=True,
    ):
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
            vectors=orientation.transpose(),
            colors=[[0xFF0000, 0xFF0000], [0x00FF00, 0x00FF00], [0x0000FF, 0x0000FF]],
            labels=[],
            label_size=1.5,
        )
        self._vectors.visible = show_vectors

        self._label = None
        if name is not None:
            self._label = k3d.text(
                text=name,
                position=coordinates + 0.05,
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
        self._trace.visible = show_trace

        self.origin = platonic.Octahedron(size=0.1).mesh
        self.origin.color = color
        self.origin.model_matrix = self._get_model_matrix(coordinates, orientation)
        self.origin.visible = show_origin

        if plot is not None:
            plot += self._vectors
            plot += self._trace
            plot += self.origin
            if self._label is not None:
                plot += self._label

    @staticmethod
    def _get_model_matrix(coordinates, orientation):
        model_matrix = np.eye(4, dtype="float32")
        model_matrix[:3, :3] = orientation
        model_matrix[:3, 3] = coordinates
        return model_matrix

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
        self._vectors.vectors = orientation.transpose()
        self.origin.model_matrix = self._get_model_matrix(coordinates, orientation)
        if self._label is not None:
            self._label.position = coordinates + 0.05

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
        if coordinates.ndim > 1:
            coordinates = coordinates[index]

        if orientation.ndim > 2:
            orientation = orientation[index]

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

    color_table = [0xFF0000, 0x00AA00, 0x0000FF, 0xAAAA00, 0xFF00FF, 0x00FFFF]

    def __init__(
        self,
        csm,
        coordinate_systems=None,
        colors=None,
        reference_system=None,
        title=None,
        limits=None,
        time=None,
        time_ref=None,
        show_labels=True,
        show_origins=True,
        show_traces=True,
        show_vectors=True,
    ):
        """Create a `CoordinateSystemManagerVisualizerK3D`.

        Parameters
        ----------
        csm : weldx.CoordinateSystemManager
            The `CoordinateSystemManager` that should be visualized
        time : pandas.DatetimeIndex, pandas.TimedeltaIndex, List[pandas.Timestamp], or \
               LocalCoordinateSystem
            The time steps that should be plotted
        time_ref : pandas.Timestamp
            A reference timestamp that can be provided if the ``time`` parameter is a
            `pandas.TimedeltaIndex`

        """
        if time is None:
            time = csm.time_union()
        if time is not None:
            csm = csm.interp_time(time=time, time_ref=time_ref)
        self._csm = csm.interp_time(time=time, time_ref=time_ref)
        self._current_time_index = 0

        if coordinate_systems is None:
            coordinate_systems = csm.coordinate_system_names
        if reference_system is None:
            reference_system = self._csm._root_system_name
        if limits is None:
            limits = [-1, 1]

        # create controls
        self._controls = self._create_controls(
            time,
            reference_system,
            show_labels,
            show_origins,
            show_traces,
            show_vectors,
        )

        # create plot using dict comprehension
        self._color_generator = color_generator_function()
        plot = k3d.plot(
            grid_auto_fit=False,
            camera_auto_fit=False,
            grid=(limits[0], limits[0], limits[0], limits[1], limits[1], limits[1]),
        )
        self._lcs_vis = {
            lcs_name: CoordinateSystemVisualizerK3D(
                self._csm.get_cs(lcs_name, reference_system),
                plot,
                lcs_name,
                color=self._get_color(lcs_name, colors),
                show_origin=show_origins,
                show_trace=show_traces,
                show_vectors=show_vectors,
            )
            for lcs_name in coordinate_systems
        }

        # add title
        self.title = None
        if title is not None:
            self.title = k3d.text2d(
                f"<b>{title}</b>",
                position=(0.5, 0),
                color=0x000000,
                is_html=True,
                size=1.5,
                reference_point="ct",
            )
            plot += self.title

        # add time info
        self._time = time
        self._time_ref = time_ref
        self._time_info = None
        if time is not None:
            self._time_info = k3d.text2d(
                f"<b>time:</b> {time[0]}",
                position=(0, 1),
                color=0x000000,
                is_html=True,
                size=1.0,
                reference_point="lb",
            )
            plot += self._time_info

        # display everything
        plot.display()
        display(self._controls)

        # workaround since using it inside the init method of the coordinate system
        # visualizer somehow causes the labels to be created twice with one version
        # being always visible
        self.show_labels(show_labels)

    def _create_controls(
        self,
        time,
        reference_system,
        show_labels,
        show_origins,
        show_traces,
        show_vectors,
    ):
        num_times = 1
        disable_time_widgets = True

        # create widgets
        if time is not None:
            num_times = len(time)
            disable_time_widgets = False

        play = Play(
            min=0,
            max=num_times - 1,
            value=self._current_time_index,
            step=1,
        )
        time_slider = IntSlider(
            min=0,
            max=num_times - 1,
            value=self._current_time_index,
            description="Time:",
        )
        reference_dropdown = Dropdown(
            options=self._csm.coordinate_system_names,
            value=reference_system,
            description="Reference:",
            disabled=False,
        )
        lo = Layout(width="200px")
        vectors_cb = Checkbox(value=show_vectors, description="show vectors", layout=lo)
        origin_cb = Checkbox(value=show_origins, description="show origins", layout=lo)
        traces_cb = Checkbox(value=show_traces, description="show traces", layout=lo)
        labels_cb = Checkbox(value=show_labels, description="show labels", layout=lo)

        jslink((play, "value"), (time_slider, "value"))
        play.disabled = disable_time_widgets
        time_slider.disabled = disable_time_widgets

        # callback functions
        def on_reference_change(change):
            """Handle events of the reference system drop down."""
            self.update_reference_system(change["new"])

        def on_time_change(change):
            """Handle events of the time slider."""
            self.update_time_index(change["new"])

        def on_vectors_change(change):
            """Handle events of the vectors checkbox."""
            self.show_vectors(change["new"])

        def on_origins_change(change):
            """Handle events of the origins checkbox."""
            self.show_origins(change["new"])

        def on_traces_change(change):
            """Handle events of the traces checkbox."""
            self.show_traces(change["new"])

        def on_labels_change(change):
            """Handle events of the labels checkbox."""
            self.show_labels(change["new"])

        # register callbacks
        time_slider.observe(on_time_change, names="value")
        reference_dropdown.observe(on_reference_change, names="value")
        vectors_cb.observe(on_vectors_change, names="value")
        origin_cb.observe(on_origins_change, names="value")
        traces_cb.observe(on_traces_change, names="value")
        labels_cb.observe(on_labels_change, names="value")

        # create control panel
        row_1 = HBox([time_slider, play, reference_dropdown])
        row_2 = HBox([vectors_cb, origin_cb, traces_cb, labels_cb])
        return VBox([row_1, row_2])

    def _get_color(self, lcs_name, color_dict):
        if color_dict is not None and lcs_name in color_dict:
            return color_rgb_to_int(color_dict[lcs_name])
        return next(self._color_generator)

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
        f"<b>time:</b> {time[0]}"

    def update_time_index(self, index):
        """Update the plotted time by index.

        Parameters
        ----------
        index : int
            The new index

        """
        self._current_time_index = index
        for _, lcs_vis in self._lcs_vis.items():
            lcs_vis.update_time_index(index)
        self._time_info.text = f"<b>time:</b> {self._time[index]}"

    def update_reference_system(self, reference_system):
        """Update the reference system of the plot.

        Parameters
        ----------
        reference_system : str
            Name of the new reference system

        """
        for lcs_name, lcs_vis in self._lcs_vis.items():
            lcs_vis.update_lcs(
                self._csm.get_cs(lcs_name, reference_system), self._current_time_index
            )

    def show_vectors(self, show_vectors):
        for _, lcs_vis in self._lcs_vis.items():
            lcs_vis._vectors.visible = show_vectors

    def show_origins(self, show_origins):
        for _, lcs_vis in self._lcs_vis.items():
            lcs_vis.origin.visible = show_origins

    def show_traces(self, show_traces):
        for _, lcs_vis in self._lcs_vis.items():
            lcs_vis._trace.visible = show_traces

    def show_labels(self, show_labels):
        for _, lcs_vis in self._lcs_vis.items():
            lcs_vis._label.visible = show_labels
