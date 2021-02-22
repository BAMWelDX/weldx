"""Contains some functions to help with visualization."""

from typing import Any, Dict, Generator, List, Tuple, Union

import k3d
import k3d.platonic as platonic
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from ipywidgets import Checkbox, Dropdown, HBox, IntSlider, Layout, Play, VBox, jslink

import weldx.geometry as geo


def color_rgb_to_int(rgb_color_tuple: Tuple[int, int, int]) -> int:
    """Convert an RGB color tuple to an 24 bit integer.

    Parameters
    ----------
    rgb_color_tuple : Tuple[int, int, int]
        The color as RGB tuple. Values must be in the range 0-255.

    Returns
    -------
    int :
        Color as 24 bit integer

    """
    return int("0x{:02x}{:02x}{:02x}".format(*rgb_color_tuple), 0)


def color_int_to_rgb(integer: int) -> Tuple[int, int, int]:
    """Convert an 24 bit integer into a RGB color tuple with the value range (0-255).

    Parameters
    ----------
    integer : int
        The value that should be converted

    Returns
    -------
    Tuple[int, int, int]:
        The resulting RGB tuple.

    """
    return ((integer >> 16) & 255, (integer >> 8) & 255, integer & 255)


def color_rgb_to_rgb_normalized(
    rgb: Tuple[int, int, int]
) -> Tuple[float, float, float]:
    """Normalize an RGB color tuple with the range (0-255) to the range (0.0-1.0).

    Parameters
    ----------
    rgb : Tuple[int, int, int]
        Color tuple with values in the range (0-255)

    Returns
    -------
    Tuple[float, float, float] :
        Color tuple with values in the range (0.0-1.0)

    """
    return tuple([val / 255 for val in rgb])


def color_rgb_normalized_to_rgb(
    rgb: Tuple[float, float, float]
) -> Tuple[int, int, int]:
    """Normalize an RGB color tuple with the range (0.0-1.0) to the range (0-255).

    Parameters
    ----------
    rgb : Tuple[float, float, float]
        Color tuple with values in the range (0.0-1.0)

    Returns
    -------
    Tuple[int, int, int] :
        Color tuple with values in the range (0-255)

    """
    return tuple([int(np.round(val * 255)) for val in rgb])


def color_int_to_rgb_normalized(integer):
    """Convert an 24 bit integer into a RGB color tuple with the value range (0.0-1.0).

    Parameters
    ----------
    integer : int
        The value that should be converted

    Returns
    -------
    Tuple[float, float, float]:
        The resulting RGB tuple.

    """
    rgb = color_int_to_rgb(integer)
    return color_rgb_to_rgb_normalized(rgb)


def color_rgb_normalized_to_int(rgb: Tuple[float, float, float]) -> int:
    """Convert a normalized RGB color tuple to an 24 bit integer.

    Parameters
    ----------
    rgb : Tuple[float, float, float]
        The color as RGB tuple. Values must be in the range 0.0-1.0.

    Returns
    -------
    int :
        Color as 24 bit integer

    """
    return color_rgb_to_int(color_rgb_normalized_to_rgb(rgb))


def shuffled_tab20_colors() -> List[int]:
    """Get a shuffled list of matplotlib 'tab20' colors.

    Returns
    -------
    List[int] :
        List of colors

    """
    num_colors = 20
    colormap = plt.cm.get_cmap("tab20", num_colors)
    colors = [colormap(i)[:3] for i in range(num_colors)]

    # randomize colors
    np.random.seed(42)
    np.random.shuffle(colors)

    return [color_rgb_normalized_to_int(color) for color in colors]


_color_list = [
    0xFF0000,
    0x00AA00,
    0x0000FF,
    0xAAAA00,
    0xFF00FF,
    0x00FFFF,
    *shuffled_tab20_colors(),
]


def color_generator_function() -> int:
    """Yield a 24 bit RGB color integer.

    The returned value is taken from a predefined list.

    Yields
    ------
    int:
        24 bit RGB color integer

    """
    while True:
        for color in _color_list:
            yield color


def _get_color(key: str, color_dict: Dict[str, int], color_generator: Generator) -> int:
    """Get a 24 bit RGB color from a dictionary or generator function.

    If the provided key is found in the dictionary, the corresponding color is returned.
    Otherwise, the generator is used to provide a color.

    Parameters
    ----------
    key : str
        The key that should be searched for in the dictionary
    color_dict : Dict[str, int]
        A dictionary containing name to color mappings
    color_generator : Generator
        A generator that returns a color integer

    Returns
    -------
    int :
        RGB color as 24 bit integer

    """
    if color_dict is not None and key in color_dict:
        return color_rgb_to_int(color_dict[key])
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
    except Exception:  # skipcq: PYL-W0703
        fig.set_size_inches(w=width / pixel_per_inch, h=height / pixel_per_inch)
    return fig, ax


# todo rename to something like render_cs
def plot_coordinate_system(
    coordinate_system,
    axes: plt.Axes.axes,
    color: Any = None,
    label: str = None,
    time_idx: int = None,
    show_origin: bool = True,
    show_vectors: bool = True,
):
    """Plot a coordinate system in a matplotlib 3d plot.

    Parameters
    ----------
    coordinate_system : weldx.transformations.LocalCoordinateSystem
        Coordinate system
    axes : matplotlib.pyplot.Axes.axes
        Target matplotlib axes object
    color : Any
        Valid matplotlib color selection. The origin of the coordinate system
        will be marked with this color.
    label : str
        Name that appears in the legend. Only viable if a color
        was specified.
    time_idx : int
        Selects time dependent data by index if the coordinate system has
        a time dependency.
    show_origin : bool
        If `True`, the origin of the coordinate system will be highlighted in the
        color passed as another parameter
    show_vectors : bool
        If `True`, the the coordinate axes of the coordinate system are visualized

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


# todo remove
# def _get_color_matplotlib(axes):
#    color_table = ["r", "g", "b", "y"]
#    for color in color_table:
#        yield color
#    yield next(axes._get_lines.prop_cycler)["color"]


def plot_local_coordinate_system_matplotlib(
    lcs,
    axes: plt.Axes.axes = None,
    color: Any = None,
    label: str = None,
    time: Union[pd.DatetimeIndex, pd.TimedeltaIndex, List[pd.Timestamp]] = None,
    time_ref: pd.Timestamp = None,
    time_index: int = None,
    show_origin: bool = True,
    show_trace: bool = True,
    show_vectors: bool = True,
) -> plt.Axes.axes:
    """Visualize a `weldx.transformations.LocalCoordinateSystem` using matplotlib.

    Parameters
    ----------
    lcs : weldx.transformations.LocalCoordinateSystem
        The coordinate system that should be visualized
    axes : matplotlib.pyplot.Axes.axes
        The target matplotlib axes. If `None` is provided, a new one will be created
    color : Any
        An arbitrary color. The data type must be compatible with matplotlib.
    label : str
        Name of the coordinate system
    time : pandas.DatetimeIndex, pandas.TimedeltaIndex, List[pandas.Timestamp], or \
           LocalCoordinateSystem
        The time steps that should be plotted
    time_ref : pandas.Timestamp
        A reference timestamp that can be provided if the ``time`` parameter is a
        `pandas.TimedeltaIndex`
    time_index : int
        Index of a specific time step that should be plotted
    show_origin : bool
        If `True`, the origin of the coordinate system will be highlighted in the
        color passed as another parameter
    show_trace :
        If `True`, the trace of a time dependent coordinate system will be visualized in
        the color passed as another parameter
    show_vectors : bool
        If `True`, the the coordinate axes of the coordinate system are visualized

    Returns
    -------
    matplotlib.pyplot.Axes.axes :
        The axes object that was used as canvas for the plot

    """
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
    axes: plt.Axes.axes = None,
    reference_system: str = None,
    coordinate_systems: List[str] = None,
    data_sets: List[str] = None,
    colors: Dict[str, int] = None,
    time: Union[pd.DatetimeIndex, pd.TimedeltaIndex, List[pd.Timestamp]] = None,
    time_ref: pd.Timestamp = None,
    title: str = None,
    limits: List[Tuple[float, float]] = None,
    show_origins: bool = True,
    show_trace: bool = True,
    show_vectors: bool = True,
) -> plt.Axes.axes:
    """Plot the coordinate systems of a `CoordinateSystemManager` using matplotlib.

    Parameters
    ----------
    csm : weldx.CoordinateSystemManager
        The `CoordinateSystemManager` that should be plotted
    axes : plt.Axes.axes
        The target axes object that should be drawn to. If `None` is provided, a new
        one will be created.
    reference_system : str
        The name of the reference system for the plotted coordinate systems
    coordinate_systems : List[str]
        Names of the coordinate systems that should be drawn. If `None` is provided,
        all systems are plotted.
    data_sets : List[str]
        Names of the data sets that should be drawn. If `None` is provided, all data
        is plotted.
    colors: Dict[str, int]
        A mapping between a coordinate system name or a data set name and a color.
        The colors must be provided as 24 bit integer values that are divided into
        three 8 bit sections for the rgb values. For example `0xFF0000` for pure
        red.
        Each coordinate system or data set that does not have a mapping in this
        dictionary will get a default color assigned to it.
    time : pandas.DatetimeIndex, pandas.TimedeltaIndex, List[pandas.Timestamp], or \
           LocalCoordinateSystem
        The time steps that should be plotted
    time_ref : pandas.Timestamp
        A reference timestamp that can be provided if the ``time`` parameter is a
        `pandas.TimedeltaIndex`
    title : str
        The title of the plot
    limits : List[Tuple[float, float]]
        The limits of the plotted volume
    show_origins : bool
        If `True`, the origins of the coordinate system are visualized in the color
        assigned to the coordinate system.
    show_trace : bool
        If `True`, the trace of time dependent coordinate systems is plotted.
    show_vectors : bool
        If `True`, the coordinate cross of time dependent coordinate systems is plotted.

    Returns
    -------
    matplotlib.pyplot.Axes.axes :
        The axes object that was used as canvas for the plot

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
    if axes is None:
        _, axes = new_3d_figure_and_axes()
        axes.set_xlabel("x")
        axes.set_ylabel("y")
        axes.set_zlabel("z")

    if reference_system is None:
        reference_system = csm.root_system_name
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
        color = color_int_to_rgb_normalized(_get_color(data_name, colors, color_gen))
        data = csm.get_data(data_name, reference_system)
        triangles = None
        if isinstance(data, geo.SpatialData):
            triangles = data.triangles
            data = data.coordinates

        data = data.data
        while data.ndim > 2:
            data = data[0]

        axes.plot(data[:, 0], data[:, 1], data[:, 2], "x", color=color, label=data_name)
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
        if len(limits) == 1:
            limits = [limits[0] for i in range(3)]
        axes.set_xlim(limits[0])
        axes.set_ylim(limits[1])
        axes.set_zlim(limits[2])
    axes.legend()

    return axes


def plot_coordinate_systems(
    cs_data: Tuple[str, Dict],
    axes: plt.Axes.axes = None,
    title: str = None,
    limits: List[Tuple[float, float]] = None,
    time_index: int = None,
    legend_pos: str = "lower left",
) -> plt.Axes.axes:
    """Plot multiple coordinate systems.

    Parameters
    ----------
    cs_data : Tuple[str, Dict]
        A tuple containing the coordinate system that should be plotted and a dictionary
        with the key word arguments that should be passed to its plot function.
    axes : plt.Axes.axes
        The target axes object that should be drawn to. If `None` is provided, a new
        one will be created.
    title : str
        The title of the plot
    limits : List[Tuple[float, float]]
        The limits of the plotted volume
    time_index : int
        Index of a specific time step that should be plotted if the corresponding
        coordinate system is time dependent
    legend_pos : str
        A string that specifies the position of the legend. See the matplotlib
        documentation for further details

    Returns
    -------
    matplotlib.pyplot.Axes.axes :
        The axes object that was used as canvas for the plot

    """
    if axes is None:
        _, axes = new_3d_figure_and_axes()

    for lcs, kwargs in cs_data:
        if "time_index" not in kwargs:
            kwargs["time_index"] = time_index
        lcs.plot(axes, **kwargs)

    if limits is None:
        set_axes_equal(axes)
    else:
        axes.set_xlim(limits)
        axes.set_ylim(limits)
        axes.set_zlim(limits)

    if title is not None:
        axes.set_title(title)
    axes.legend(loc=legend_pos)

    return axes


# k3d ----------------------------------------------------------------------------------


class CoordinateSystemVisualizerK3D:
    """Visualizes a `weldx.LocalCoordinateSystem` using k3d."""

    def __init__(
        self,
        lcs,
        plot: k3d.Plot = None,
        name: str = None,
        color: int = 0x000000,
        show_origin=True,
        show_trace=True,
        show_vectors=True,
    ):
        """Create a `CoordinateSystemVisualizerK3D`.

        Parameters
        ----------
        lcs : weldx.LocalCoordinateSystem
            Coordinate system that should be visualized
        plot : k3d.Plot
            A k3d plotting widget.
        name : str
            Name of the coordinate system
        color : int
            The RGB color of the coordinate system (affects trace and label) as a 24 bit
            integer value.
        show_origin : bool
            If `True`, the origin of the coordinate system will be highlighted in the
            color passed as another parameter
        show_trace :
            If `True`, the trace of a time dependent coordinate system will be
            visualized in the color passed as another parameter
        show_vectors : bool
            If `True`, the the coordinate axes of the coordinate system are visualized

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
        self.origin.model_matrix = self._create_model_matrix(coordinates, orientation)
        self.origin.visible = show_origin

        if plot is not None:
            plot += self._vectors
            plot += self._trace
            plot += self.origin
            if self._label is not None:
                plot += self._label

    @staticmethod
    def _create_model_matrix(
        coordinates: np.ndarray, orientation: np.ndarray
    ) -> np.ndarray:
        """Create the model matrix from an orientation and coordinates.

        Parameters
        ----------
        coordinates : numpy.ndarray
            The coordinates of the origin
        orientation : numpy.ndarray
            The orientation of the coordinate system

        Returns
        -------
        numpy.ndarray :
            The model matrix

        """
        model_matrix = np.eye(4, dtype="float32")
        model_matrix[:3, :3] = orientation
        model_matrix[:3, 3] = coordinates
        return model_matrix

    def _update_positions(self, coordinates: np.ndarray, orientation: np.ndarray):
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
        self.origin.model_matrix = self._create_model_matrix(coordinates, orientation)
        if self._label is not None:
            self._label.position = coordinates + 0.05

    @staticmethod
    def _get_coordinates_and_orientation(lcs, index: int = 0):
        """Get the coordinates and orientation of a coordinate system.

        Parameters
        ----------
        lcs : weldx.LocalCoordinateSystem
            The coordinate system
        index : int
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

    def show_label(self, show_label: bool):
        """Set the visibility of the label.

        Parameters
        ----------
        show_label : bool
            If `True`, the label will be shown

        """
        self._label.visible = show_label

    def show_origin(self, show_origin: bool):
        """Set the visibility of the coordinate systems' origin.

        Parameters
        ----------
        show_origin : bool
            If `True`, the coordinate systems origin is shown.

        """
        self.origin.visible = show_origin

    def show_trace(self, show_trace: bool):
        """Set the visibility of coordinate systems' trace.

        Parameters
        ----------
        show_trace : bool
            If `True`, the coordinate systems' trace is shown.

        """
        self._trace.visible = show_trace

    def show_vectors(self, show_vectors: bool):
        """Set the visibility of the coordinate axis vectors.

        Parameters
        ----------
        show_vectors : bool
            If `True`, the coordinate axis vectors are shown.

        """
        self._vectors.visible = show_vectors

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

    def update_time(
        self,
        time: Union[pd.DatetimeIndex, pd.TimedeltaIndex, List[pd.Timestamp]],
        time_ref: pd.Timestamp = None,
    ):
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

    def update_time_index(self, index: int):
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


class SpatialDataVisualizer:
    """Visualizes spatial data."""

    visualization_methods = ["auto", "point", "mesh", "both"]

    def __init__(
        self,
        data,
        name: str,
        cs_vis: CoordinateSystemVisualizerK3D,
        plot: k3d.Plot = None,
        color: int = 0x000000,
        visualization_method: str = "auto",
        show_wireframe: bool = False,
    ):
        """Create a 'SpatialDataVisualizer' instance.

        Parameters
        ----------
        data : numpy.ndarray or weldx.geometry.SpatialData
            The data that should be visualized
        name : str
            Name of the data
        cs_vis : CoordinateSystemVisualizerK3D
            An instance of the 'CoordinateSystemVisualizerK3D'. This serves as reference
            coordinate system for the data and is needed to calculate the correct
            position of the data
        plot : k3d.Plot
            A k3d plotting widget.
        color : int
            The RGB color of the coordinate system (affects trace and label) as a 24 bit
            integer value.
        visualization_method : str
            The initial data visualization method. Options are 'point', 'mesh', 'both'
            and 'auto'. If 'auto' is selected, a mesh will be drawn if triangle data is
            available and points if not.
        show_wireframe : bool
            If 'True', meshes will be drawn as wireframes

        """
        triangles = None
        if isinstance(data, geo.SpatialData):
            triangles = data.triangles
            data = data.coordinates.data

        self._cs_vis = cs_vis

        self._label_pos = np.mean(data, axis=0)
        self._label = None
        if name is not None:
            self._label = k3d.text(
                text=name,
                position=self._label_pos,
                reference_point="cc",
                color=color,
                size=0.5,
                label_box=True,
            )
        self._points = k3d.points(data, point_size=0.05, color=color)
        self._mesh = None
        if triangles is not None:
            self._mesh = k3d.mesh(
                data, triangles, side="double", color=color, wireframe=show_wireframe
            )

        self.update_model_matrix()
        self.set_visualization_method(visualization_method)

        if plot is not None:
            plot += self._points
            if self._mesh is not None:
                plot += self._mesh
            if self._label is not None:
                plot += self._label

    def set_visualization_method(self, method: str):
        """Set the visualization method.

        Parameters
        ----------
        method : str
            The data visualization method. Options are 'point', 'mesh', 'both' and
            'auto'. If 'auto' is selected, a mesh will be drawn if triangle data is
            available and points if not.

        """
        if method not in SpatialDataVisualizer.visualization_methods:
            raise ValueError(f"Unknown visualization method: '{method}'")

        if method == "auto":
            if self._mesh is not None:
                method = "mesh"
            else:
                method = "point"

        self._points.visible = method in ["point", "both"]
        if self._mesh is not None:
            self._mesh.visible = method in ["mesh", "both"]

    def show_label(self, show_label: bool):
        """Set the visibility of the label.

        Parameters
        ----------
        show_label : bool
            If `True`, the label will be shown

        """
        self._label.visible = show_label

    def show_wireframe(self, show_wireframe: bool):
        """Set wireframe rendering.

        Parameters
        ----------
        show_wireframe : bool
            If `True`, the mesh will be rendered as wireframe

        """
        if self._mesh is not None:
            self._mesh.wireframe = show_wireframe

    def update_model_matrix(self):
        """Update the model matrices of the k3d objects."""
        model_mat = self._cs_vis.origin.model_matrix
        self._points.model_matrix = model_mat
        if self._mesh is not None:
            self._mesh.model_matrix = model_mat
        if self._label is not None:
            self._label.position = (
                np.matmul(model_mat[0:3, 0:3], self._label_pos) + model_mat[0:3, 3]
            )


class CoordinateSystemManagerVisualizerK3D:
    """Visualizes a `weldx.CoordinateSystemManager` using k3d."""

    def __init__(
        self,
        csm,
        coordinate_systems: List[str] = None,
        data_sets: List[str] = None,
        colors: Dict[str, int] = None,
        reference_system: str = None,
        title: str = None,
        limits: List[Tuple[float, float]] = None,
        time: Union[pd.DatetimeIndex, pd.TimedeltaIndex, List[pd.Timestamp]] = None,
        time_ref: pd.Timestamp = None,
        show_data_labels: bool = True,
        show_labels: bool = True,
        show_origins: bool = True,
        show_traces: bool = True,
        show_vectors: bool = True,
        show_wireframe: bool = True,
    ):
        """Create a `CoordinateSystemManagerVisualizerK3D`.

        Parameters
        ----------
        csm : weldx.CoordinateSystemManager
            The `CoordinateSystemManager` that should be visualized
        coordinate_systems : List[str]
            The names of the coordinate systems that should be visualized. If ´None´ is
            provided, all systems are plotted
        data_sets : List[str]
            The names of data sets that should be visualized. If ´None´ is provided, all
            data is plotted
        colors : Dict[str, int]
            A mapping between a coordinate system name or a data set name and a color.
            The colors must be provided as 24 bit integer values that are divided into
            three 8 bit sections for the rgb values. For example `0xFF0000` for pure
            red.
            Each coordinate system or data set that does not have a mapping in this
            dictionary will get a default color assigned to it.
        reference_system : str
            Name of the initial reference system. If `None` is provided, the root system
            of the `CoordinateSystemManager` instance will be used
        title : str
            The title of the plot
        limits : List[Tuple[float, float]]
            The limits of the plotted volume
        time : pandas.DatetimeIndex, pandas.TimedeltaIndex, List[pandas.Timestamp], or \
               LocalCoordinateSystem
            The time steps that should be plotted initially
        time_ref : pandas.Timestamp
            A reference timestamp that can be provided if the ``time`` parameter is a
            `pandas.TimedeltaIndex`
        show_data_labels : bool
            If `True`, the data labels will be shown initially
        show_labels  : bool
            If `True`, the coordinate system labels will be shown initially
        show_origins : bool
            If `True`, the coordinate systems' origins will be shown initially
        show_traces : bool
            If `True`, the coordinate systems' traces will be shown initially
        show_vectors : bool
            If `True`, the coordinate systems' axis vectors will be shown initially
        show_wireframe : bool
            If `True`, spatial data containing mesh data will be drawn as wireframe

        """
        if time is None:
            time = csm.time_union()
        if time is not None:
            csm = csm.interp_time(time=time, time_ref=time_ref)
        self._csm = csm.interp_time(time=time, time_ref=time_ref)
        self._current_time_index = 0

        if coordinate_systems is None:
            coordinate_systems = csm.coordinate_system_names
        if data_sets is None:
            data_sets = self._csm.data_names
        if reference_system is None:
            reference_system = self._csm._root_system_name

        grid_auto_fit = True
        grid = (-1, -1, -1, 1, 1, 1)
        if limits is not None:
            grid_auto_fit = False
            if len(limits) == 1:
                grid = [limits[0][int(i / 3)] for i in range(6)]
                print(grid)
            else:
                grid = [limits[i % 3][int(i / 3)] for i in range(6)]
                print(grid)
        # create plot
        self._color_generator = color_generator_function()
        plot = k3d.plot(
            grid_auto_fit=grid_auto_fit,
            grid=grid,
        )
        self._lcs_vis = {
            lcs_name: CoordinateSystemVisualizerK3D(
                self._csm.get_cs(lcs_name, reference_system),
                plot,
                lcs_name,
                color=_get_color(lcs_name, colors, self._color_generator),
                show_origin=show_origins,
                show_trace=show_traces,
                show_vectors=show_vectors,
            )
            for lcs_name in coordinate_systems
        }
        self._data_vis = {
            data_name: SpatialDataVisualizer(
                self._csm.get_data(data_name=data_name),
                data_name,
                self._lcs_vis[self._csm.get_data_system_name(data_name=data_name)],
                plot,
                color=_get_color(data_name, colors, self._color_generator),
                show_wireframe=show_wireframe,
            )
            for data_name in data_sets
        }

        # create controls
        self._controls = self._create_controls(
            time,
            reference_system,
            show_data_labels,
            show_labels,
            show_origins,
            show_traces,
            show_vectors,
            show_wireframe,
        )

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
        time: Union[pd.DatetimeIndex, pd.TimedeltaIndex, List[pd.Timestamp]],
        reference_system: str,
        show_data_labels: bool,
        show_labels: bool,
        show_origins: bool,
        show_traces: bool,
        show_vectors: bool,
        show_wireframe: bool,
    ):
        """Create the control panel.

        Parameters
        ----------
        time : pandas.DatetimeIndex, pandas.TimedeltaIndex, List[pandas.Timestamp], or \
               LocalCoordinateSystem
            The time steps that should be plotted initially
        reference_system : str
            Name of the initial reference system. If `None` is provided, the root system
            of the `CoordinateSystemManager` instance will be used
        show_data_labels : bool
            If `True`, the data labels will be shown initially
        show_labels  : bool
            If `True`, the coordinate system labels will be shown initially
        show_origins : bool
            If `True`, the coordinate systems' origins will be shown initially
        show_traces : bool
            If `True`, the coordinate systems' traces will be shown initially
        show_vectors : bool
            If `True`, the coordinate systems' axis vectors will be shown initially
        show_wireframe : bool
            If `True`, spatial data containing mesh data will be drawn as wireframe

        """
        num_times = 1
        disable_time_widgets = True
        lo = Layout(width="200px")

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
        data_dropdown = Dropdown(
            options=SpatialDataVisualizer.visualization_methods,
            value="auto",
            description="data repr.:",
            disabled=False,
            layout=lo,
        )

        lo = Layout(width="200px")
        vectors_cb = Checkbox(value=show_vectors, description="show vectors", layout=lo)
        origin_cb = Checkbox(value=show_origins, description="show origins", layout=lo)
        traces_cb = Checkbox(value=show_traces, description="show traces", layout=lo)
        labels_cb = Checkbox(value=show_labels, description="show labels", layout=lo)
        wf_cb = Checkbox(value=show_wireframe, description="show wireframe", layout=lo)
        data_labels_cb = Checkbox(
            value=show_data_labels, description="show data labels", layout=lo
        )

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

        def _on_data_change(change):
            self.set_data_visualization_method(change["new"])

        def _on_data_label_cb_change(change):
            self.show_data_labels(change["new"])

        def _on_show_wireframe(change):
            self.show_wireframes(change["new"])

        # register callbacks
        time_slider.observe(on_time_change, names="value")
        reference_dropdown.observe(on_reference_change, names="value")
        vectors_cb.observe(on_vectors_change, names="value")
        origin_cb.observe(on_origins_change, names="value")
        traces_cb.observe(on_traces_change, names="value")
        labels_cb.observe(on_labels_change, names="value")
        data_dropdown.observe(_on_data_change, names="value")
        data_labels_cb.observe(_on_data_label_cb_change, names="value")
        wf_cb.observe(_on_show_wireframe, names="value")

        # create control panel
        row_1 = HBox([time_slider, play, reference_dropdown])
        row_2 = HBox([vectors_cb, origin_cb, traces_cb, labels_cb])
        if len(self._data_vis) > 0:
            row_3 = HBox([data_dropdown, wf_cb, data_labels_cb])
            return VBox([row_1, row_2, row_3])
        return VBox([row_1, row_2])

    def update_time(
        self,
        time: Union[pd.DatetimeIndex, pd.TimedeltaIndex, List[pd.Timestamp]],
        time_ref: pd.Timestamp = None,
    ):
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
        for _, data_vis in self._data_vis.items():
            data_vis.update_model_matrix()

    def update_time_index(self, index: int):
        """Update the plotted time by index.

        Parameters
        ----------
        index : int
            The new index

        """
        self._current_time_index = index
        for _, lcs_vis in self._lcs_vis.items():
            lcs_vis.update_time_index(index)
        for _, data_vis in self._data_vis.items():
            data_vis.update_model_matrix()
        self._time_info.text = f"<b>time:</b> {self._time[index]}"

    def show_data_labels(self, show_data_labels: bool):
        """Set the visibility of data labels.

        Parameters
        ----------
        show_data_labels: bool
            If `True`, labels are shown.

        """
        for _, data_vis in self._data_vis.items():
            data_vis.show_label(show_data_labels)

    def show_vectors(self, show_vectors: bool):
        """Set the visibility of the coordinate axis vectors.

        Parameters
        ----------
        show_vectors : bool
            If `True`, the coordinate axis vectors are shown.

        """
        for _, lcs_vis in self._lcs_vis.items():
            lcs_vis.show_vectors(show_vectors)

    def show_origins(self, show_origins: bool):
        """Set the visibility of the coordinate systems' origins.

        Parameters
        ----------
        show_origins : bool
            If `True`, the coordinate systems origins are shown.

        """
        for _, lcs_vis in self._lcs_vis.items():
            lcs_vis.show_origin(show_origins)

    def show_traces(self, show_traces: bool):
        """Set the visibility of coordinate systems' traces.

        Parameters
        ----------
        show_traces : bool
            If `True`, the coordinate systems' traces are shown.

        """
        for _, lcs_vis in self._lcs_vis.items():
            lcs_vis.show_trace(show_traces)

    def show_labels(self, show_labels: bool):
        """Set the visibility of the coordinate systems' labels.

        Parameters
        ----------
        show_labels : bool
            If `True`, the coordinate systems' labels are shown.

        """
        for _, lcs_vis in self._lcs_vis.items():
            lcs_vis.show_label(show_labels)

    def show_wireframes(self, show_wireframes: bool):
        """Set if meshes should be drawn in wireframe mode.

        Parameters
        ----------
        show_wireframes : bool
            If `True`, meshes are rendered as wireframes

        """
        for _, data_vis in self._data_vis.items():
            data_vis.show_wireframe(show_wireframes)

    def set_data_visualization_method(self, representation: str):
        """Set the data visualization method.

        Parameters
        ----------
        representation : str
            The data visualization method. Options are 'point', 'mesh', 'both' and
            'auto'. If 'auto' is selected, a mesh will be drawn if triangle data is
            available and points if not.

        """
        for _, data_vis in self._data_vis.items():
            data_vis.set_visualization_method(representation)

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
        for _, data_vis in self._data_vis.items():
            data_vis.update_model_matrix()
