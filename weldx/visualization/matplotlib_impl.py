"""Contains some functions written in matplotlib to help with visualization."""

from typing import Any, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure

import weldx.geometry as geo
from weldx import CoordinateSystemManager, LocalCoordinateSystem
from weldx.visualization.colors import (
    color_generator_function,
    color_int_to_rgb_normalized,
    color_to_rgb_normalized,
    get_color,
)
from weldx.visualization.types import types_limits, types_timeindex


def new_3d_figure_and_axes(
    num_subplots: int = 1, height: int = 500, width: int = 500, pixel_per_inch: int = 50
) -> Tuple[Figure, Axes]:
    """Get a matplotlib figure and axes for 3d plots.

    Parameters
    ----------
    num_subplots :
        Number of subplots (horizontal)
    height :
        Height in pixels
    width :
        Width in pixels
    pixel_per_inch :
        Defines how many pixels an inch covers. This is only relevant for the fallback
        method.

    Returns
    -------
    fig :
        The matplotlib figure object
    matplotlib.axes.Axes :
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


def axes_equal(axes: Axes):
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


def draw_coordinate_system_matplotlib(
    coordinate_system: LocalCoordinateSystem,
    axes: Axes,
    color: Any = None,
    label: str = None,
    time_idx: int = None,
    scale_vectors: Union[float, List, np.ndarray] = None,
    show_origin: bool = True,
    show_vectors: bool = True,
):
    """Draw a coordinate system in a matplotlib 3d plot.

    Parameters
    ----------
    coordinate_system :
        Coordinate system
    axes :
        Target matplotlib axes object
    color :
        Valid matplotlib color selection. The origin of the coordinate system
        will be marked with this color.
    label :
        Name that appears in the legend. Only viable if a color
        was specified.
    time_idx :
        Selects time dependent data by index if the coordinate system has
        a time dependency.
    scale_vectors :
        A scaling factor or array to adjust the vector length
    show_origin :
        If `True`, the origin of the coordinate system will be highlighted in the
        color passed as another parameter
    show_vectors :
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
        if scale_vectors is None:
            tips = dsx.orientation
        else:
            if not isinstance(scale_vectors, np.ndarray):
                if isinstance(scale_vectors, List):
                    scale_vectors = np.array(scale_vectors)
                else:
                    scale_vectors = np.array([scale_vectors for _ in range(3)])

            scale_mat = np.eye(3, 3)
            for i in range(3):
                scale_mat[i, i] = scale_vectors[i]
            tips = np.matmul(scale_mat, dsx.orientation.data)

        p_x = p_0 + tips[:, 0]
        p_y = p_0 + tips[:, 1]
        p_z = p_0 + tips[:, 2]

        axes.plot([p_0[0], p_x[0]], [p_0[1], p_x[1]], [p_0[2], p_x[2]], "r")
        axes.plot([p_0[0], p_y[0]], [p_0[1], p_y[1]], [p_0[2], p_y[2]], "g")
        axes.plot([p_0[0], p_z[0]], [p_0[1], p_z[1]], [p_0[2], p_z[2]], "b")
    if color is not None:
        if show_origin:
            axes.plot([p_0[0]], [p_0[1]], [p_0[2]], "o", color=color, label=label)
    elif label is not None:
        raise Exception("Labels can only be assigned if a color was specified")


def plot_local_coordinate_system_matplotlib(
    lcs: LocalCoordinateSystem,
    axes: Axes = None,
    color: Any = None,
    label: str = None,
    time: types_timeindex = None,
    time_ref: pd.Timestamp = None,
    time_index: int = None,
    scale_vectors: Union[float, List, np.ndarray] = None,
    show_origin: bool = True,
    show_trace: bool = True,
    show_vectors: bool = True,
) -> Axes:
    """Visualize a `weldx.transformations.LocalCoordinateSystem` using matplotlib.

    Parameters
    ----------
    lcs :
        The coordinate system that should be visualized
    axes :
        The target matplotlib axes. If `None` is provided, a new one will be created
    color :
        An arbitrary color. The data type must be compatible with matplotlib.
    label :
        Name of the coordinate system
    time :
        The time steps that should be plotted
    time_ref :
        A reference timestamp that can be provided if the ``time`` parameter is a
        `pandas.TimedeltaIndex`
    time_index :
        Index of a specific time step that should be plotted
    scale_vectors :
        A scaling factor or array to adjust the vector length
    show_origin :
        If `True`, the origin of the coordinate system will be highlighted in the
        color passed as another parameter
    show_trace :
        If `True`, the trace of a time dependent coordinate system will be visualized in
        the color passed as another parameter
    show_vectors :
        If `True`, the the coordinate axes of the coordinate system are visualized

    Returns
    -------
    matplotlib.axes.Axes :
        The axes object that was used as canvas for the plot.

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
                scale_vectors=scale_vectors,
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
            scale_vectors=scale_vectors,
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
    axes: Axes,
    limits: types_limits,
    set_axes_equal: bool = False,
):
    """Set the limits of an axes object.

    Parameters
    ----------
    axes :
        The axes object
    limits :
        Each tuple marks lower and upper boundary of the x, y and z axis. If only a
        single tuple is passed, the boundaries are used for all axis. If `None`
        is provided, the axis are adjusted to be of equal length.
    set_axes_equal :
        (matplotlib only) If `True`, all axes are adjusted to cover an equally large
         range of value. That doesn't mean, that the limits are identical

    """
    if limits is not None:
        if isinstance(limits, Tuple):
            limits = [limits]
        if len(limits) == 1:
            limits = [limits[0] for _ in range(3)]
        axes.set_xlim(limits[0])
        axes.set_ylim(limits[1])
        axes.set_zlim(limits[2])
    elif set_axes_equal:
        axes_equal(axes)


def plot_coordinate_systems(
    cs_data: Tuple[str, Dict],
    axes: Axes = None,
    title: str = None,
    limits: types_limits = None,
    time_index: int = None,
    legend_pos: str = "lower left",
) -> Axes:
    """Plot multiple coordinate systems.

    Parameters
    ----------
    cs_data :
        A tuple containing the coordinate system that should be plotted and a dictionary
        with the key word arguments that should be passed to its plot function.
    axes :
        The target axes object that should be drawn to. If `None` is provided, a new
        one will be created.
    title :
        The title of the plot
    limits :
        Each tuple marks lower and upper boundary of the x, y and z axis. If only a
        single tuple is passed, the boundaries are used for all axis. If `None`
        is provided, the axis are adjusted to be of equal length.
    time_index :
        Index of a specific time step that should be plotted if the corresponding
        coordinate system is time dependent
    legend_pos :
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
    csm: CoordinateSystemManager,
    axes: Axes = None,
    reference_system: str = None,
    coordinate_systems: List[str] = None,
    data_sets: List[str] = None,
    colors: Dict[str, int] = None,
    time: types_timeindex = None,
    time_ref: pd.Timestamp = None,
    title: str = None,
    limits: types_limits = None,
    scale_vectors: Union[float, List, np.ndarray] = None,
    set_axes_equal: bool = False,
    show_origins: bool = True,
    show_trace: bool = True,
    show_vectors: bool = True,
    show_wireframe: bool = True,
) -> Axes:
    """Plot the coordinate systems of a `weldx.transformations.CoordinateSystemManager`.

    Parameters
    ----------
    csm :
        The coordinate system manager instance that should be plotted.
    axes :
        The target axes object that should be drawn to. If `None` is provided, a new
        one will be created.
    reference_system :
        The name of the reference system for the plotted coordinate systems
    coordinate_systems :
        Names of the coordinate systems that should be drawn. If `None` is provided,
        all systems are plotted.
    data_sets :
        Names of the data sets that should be drawn. If `None` is provided, all data
        is plotted.
    colors :
        A mapping between a coordinate system name or a data set name and a color.
        The colors must be provided as 24 bit integer values that are divided into
        three 8 bit sections for the rgb values. For example `0xFF0000` for pure
        red.
        Each coordinate system or data set that does not have a mapping in this
        dictionary will get a default color assigned to it.
    time :
        The time steps that should be plotted
    time_ref :
        A reference timestamp that can be provided if the ``time`` parameter is a
        `pandas.TimedeltaIndex`
    title :
        The title of the plot
    limits :
        Each tuple marks lower and upper boundary of the x, y and z axis. If only a
        single tuple is passed, the boundaries are used for all axis. If `None`
        is provided, the axis are adjusted to be of equal length.
    scale_vectors :
        A scaling factor or array to adjust the length of the coordinate system vectors
    set_axes_equal :
        (matplotlib only) If `True`, all axes are adjusted to cover an equally large
         range of value. That doesn't mean, that the limits are identical
    show_origins :
        If `True`, the origins of the coordinate system are visualized in the color
        assigned to the coordinate system.
    show_trace :
        If `True`, the trace of time dependent coordinate systems is plotted.
    show_vectors :
        If `True`, the coordinate cross of time dependent coordinate systems is plotted.
    show_wireframe :
        If `True`, the mesh is visualized as wireframe. Otherwise, it is not shown.

    Returns
    -------
    matplotlib.axes.Axes :
        The axes object that was used as canvas for the plot.

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
            scale_vectors=scale_vectors,
            show_origin=show_origins,
            show_trace=show_trace,
            show_vectors=show_vectors,
        )
    # plot data
    for data_name in data_sets:
        color = color_int_to_rgb_normalized(get_color(data_name, colors, color_gen))
        data = csm.get_data(data_name, reference_system)
        plot_spatial_data_matplotlib(
            data=data,
            axes=axes,
            color=color,
            label=data_name,
            show_wireframe=show_wireframe,
        )

    _set_limits_matplotlib(axes, limits, set_axes_equal)
    axes.legend()

    return axes


def plot_spatial_data_matplotlib(
    data: geo.SpatialData,
    axes: Axes = None,
    color: Union[int, Tuple[int, int, int], Tuple[float, float, float]] = None,
    label: str = None,
    show_wireframe: bool = True,
) -> Axes:
    """Visualize a `weldx.geometry.SpatialData` instance.

    Parameters
    ----------
    data :
        The data that should be visualized
    axes :
        The target `matplotlib.axes.Axes` object of the plot. If 'None' is passed, a
        new figure will be created
    color :
        A 24 bit integer, a triplet of integers with a value range of 0-255
        or a triplet of floats with a value range of 0.0-1.0 that represent an RGB
        color
    label :
        Label of the plotted geometry
    show_wireframe :
        If `True`, the mesh is plotted as wireframe. Otherwise only the raster
        points are visualized. Currently, the wireframe can't be visualized if a
        `weldx.geometry.VariableProfile` is used.

    Returns
    -------
    matplotlib.axes.Axes :
        The `matplotlib.axes.Axes` instance that was used for the plot.

    """
    if axes is None:
        _, axes = new_3d_figure_and_axes()

    if not isinstance(data, geo.SpatialData):
        data = geo.SpatialData(data)

    if color is None:
        color = (0.0, 0.0, 0.0)
    else:
        color = color_to_rgb_normalized(color)

    coordinates = data.coordinates.data
    triangles = data.triangles

    # if data is time dependent or has other extra dimensions, just take the first value
    while coordinates.ndim > 2:
        coordinates = coordinates[0]

    axes.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        coordinates[:, 2],
        marker=".",
        color=color,
        label=label,
        zorder=2,
    )
    if triangles is not None and show_wireframe:
        for triangle in triangles:
            triangle_data = coordinates[[*triangle, triangle[0]], :]
            axes.plot(
                triangle_data[:, 0],
                triangle_data[:, 1],
                triangle_data[:, 2],
                color=color,
                zorder=1,
            )

    return axes
