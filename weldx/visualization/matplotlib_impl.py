"""Contains some functions written in matplotlib to help with visualization."""

from typing import Dict

from weldx.visualization.colors import (
    color_generator_function,
    color_int_to_rgb_normalized,
    get_color,
)

from typing import Any, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import weldx.geometry as geo


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
        Defines how many pixels an inch covers. This is only relevant for the fallback
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


def draw_coordinate_system_matplotlib(
    coordinate_system,
    axes: plt.Axes.axes,
    color: Any = None,
    label: str = None,
    time_idx: int = None,
    show_origin: bool = True,
    show_vectors: bool = True,
):
    """Draw a coordinate system in a matplotlib 3d plot.

    Parameters
    ----------
    coordinate_system : weldx.transformations.LocalCoordinateSystem
        Coordinate system
    axes : matplotlib.axes.Axes
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
    axes : matplotlib.axes.Axes
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
    matplotlib.axes.Axes :
        The axes object that was used as canvas for the plot

    """
    if axes is None:
        _, axes = plt.subplots(subplot_kw={"projection": "3d", "proj_type": "ortho"})

    if lcs.is_time_dependent and time is not None:
        lcs = lcs.interp_time(time, time_ref)

    if lcs.is_time_dependent and time_index is None:
        for i, _ in enumerate(lcs.time):
            draw_coordinate_system_matplotlib(
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
        draw_coordinate_system_matplotlib(
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


def _set_limits_matplotlib(
    axes: plt.Axes.axes, limits: Union[List[Tuple[float, float]], Tuple[float, float]]
):
    """Set the limits of an axes object.

    Parameters
    ----------
    axes : matplotlib.axes.Axes
        The axes object
    limits :  Tuple[float, float] or List[Tuple[float, float]]
        Each tuple marks lower and upper boundary of the x, y and z axis. If only a
        single tuple is passed, the boundaries are used for all axis. If `None`
        is provided, the axis are adjusted to be of equal length.

    """
    if limits is None:
        set_axes_equal(axes)
    else:
        if isinstance(limits, Tuple):
            limits = [limits]
        if len(limits) == 1:
            limits = [limits[0] for _ in range(3)]
        axes.set_xlim(limits[0])
        axes.set_ylim(limits[1])
        axes.set_zlim(limits[2])


def plot_coordinate_systems(
    cs_data: Tuple[str, Dict],
    axes: plt.Axes.axes = None,
    title: str = None,
    limits: Union[List[Tuple[float, float]], Tuple[float, float]] = None,
    time_index: int = None,
    legend_pos: str = "lower left",
) -> plt.Axes.axes:
    """Plot multiple coordinate systems.

    Parameters
    ----------
    cs_data : Tuple[str, Dict]
        A tuple containing the coordinate system that should be plotted and a dictionary
        with the key word arguments that should be passed to its plot function.
    axes : matplotlib.axes.Axes
        The target axes object that should be drawn to. If `None` is provided, a new
        one will be created.
    title : str
        The title of the plot
    limits :  Tuple[float, float] or List[Tuple[float, float]]
        Each tuple marks lower and upper boundary of the x, y and z axis. If only a
        single tuple is passed, the boundaries are used for all axis. If `None`
        is provided, the axis are adjusted to be of equal length.
    time_index : int
        Index of a specific time step that should be plotted if the corresponding
        coordinate system is time dependent
    legend_pos : str
        A string that specifies the position of the legend. See the matplotlib
        documentation for further details

    Returns
    -------
    matplotlib.axes.Axes :
        The axes object that was used as canvas for the plot

    """
    if axes is None:
        _, axes = new_3d_figure_and_axes()

    for lcs, kwargs in cs_data:
        if "time_index" not in kwargs:
            kwargs["time_index"] = time_index
        lcs.plot(axes, **kwargs)

    _set_limits_matplotlib(axes, limits)

    if title is not None:
        axes.set_title(title)
    axes.legend(loc=legend_pos)

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
    limits: Union[List[Tuple[float, float]], Tuple[float, float]] = None,
    show_origins: bool = True,
    show_trace: bool = True,
    show_vectors: bool = True,
) -> plt.Axes.axes:
    """Plot the coordinate systems of a `weldx.transformations.CoordinateSystemManager`.

    Parameters
    ----------
    csm : weldx.transformations.CoordinateSystemManager
        The `weldx.transformations.CoordinateSystemManager` that should be plotted
    axes : matplotlib.axes.Axes
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
           weldx.transformations.LocalCoordinateSystem
        The time steps that should be plotted
    time_ref : pandas.Timestamp
        A reference timestamp that can be provided if the ``time`` parameter is a
        `pandas.TimedeltaIndex`
    title : str
        The title of the plot
    limits :  Tuple[float, float] or List[Tuple[float, float]]
        Each tuple marks lower and upper boundary of the x, y and z axis. If only a
        single tuple is passed, the boundaries are used for all axis. If `None`
        is provided, the axis are adjusted to be of equal length.
    show_origins : bool
        If `True`, the origins of the coordinate system are visualized in the color
        assigned to the coordinate system.
    show_trace : bool
        If `True`, the trace of time dependent coordinate systems is plotted.
    show_vectors : bool
        If `True`, the coordinate cross of time dependent coordinate systems is plotted.

    Returns
    -------
    matplotlib.axes.Axes :
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
        color = color_int_to_rgb_normalized(get_color(lcs_name, colors, color_gen))
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
        color = color_int_to_rgb_normalized(get_color(data_name, colors, color_gen))
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

    _set_limits_matplotlib(axes, limits)
    axes.legend()

    return axes
