"""Visualization of spatial data, coordinate systems (cs), and cs manager."""
try:
    from warnings import warn

    from weldx_widgets.visualization.csm_k3d import (
        CoordinateSystemManagerVisualizerK3D,
        SpatialDataVisualizer,
    )

    warn("Using experimental 'weldx-widgets' implementation.", UserWarning)
except ImportError:
    from .k3d_impl import CoordinateSystemManagerVisualizerK3D, SpatialDataVisualizer

from .matplotlib_impl import (
    axes_equal,
    draw_coordinate_system_matplotlib,
    new_3d_figure_and_axes,
    plot_coordinate_system_manager_matplotlib,
    plot_coordinate_systems,
    plot_local_coordinate_system_matplotlib,
    plot_spatial_data_matplotlib,
)

__all__ = (
    "CoordinateSystemManagerVisualizerK3D",
    "SpatialDataVisualizer",
    "axes_equal",
    "draw_coordinate_system_matplotlib",
    "new_3d_figure_and_axes",
    "plot_coordinate_system_manager_matplotlib",
    "plot_coordinate_systems",
    "plot_local_coordinate_system_matplotlib",
    "plot_spatial_data_matplotlib",
)
