"""Visualization of spatial data, coordinate systems (cs), and cs manager.

The implementation details are now served by the "weldx-widgets" package (since 0.6.1).
"""

try:
    from weldx_widgets.visualization.csm_k3d import (
        CoordinateSystemManagerVisualizerK3D,
        SpatialDataVisualizer,
    )
    from weldx_widgets.visualization.csm_mpl import (
        axes_equal,
        draw_coordinate_system_matplotlib,
        new_3d_figure_and_axes,
        plot_coordinate_system_manager_matplotlib,
        plot_coordinate_systems,
        plot_local_coordinate_system_matplotlib,
        plot_spatial_data_matplotlib,
    )
except ModuleNotFoundError as mnf:
    msg = mnf.args[0]
    if "k3d" in msg:
        from unittest.mock import MagicMock as _MagickMock

        class _Hint(_MagickMock):
            def __init__(self, *args, **kwargs):
                print(
                    "Visualization requires k3d module, but could not be found! "
                    "Please install it first (or weldx-widgets)."
                )

        CoordinateSystemManagerVisualizerK3D = _Hint()
        SpatialDataVisualizer = _Hint()
    elif "matplotlib" in msg:
        from weldx.util import check_matplotlib_available as _mpl_avail

        @_mpl_avail
        def _dummy(*args, **kwargs):
            pass

        axes_equal = (
            draw_coordinate_system_matplotlib
        ) = (
            new_3d_figure_and_axes
        ) = (
            plot_coordinate_system_manager_matplotlib
        ) = (
            plot_coordinate_systems
        ) = (
            plot_local_coordinate_system_matplotlib
        ) = plot_spatial_data_matplotlib = _mpl_avail
    else:
        # something else is missing, pass the exception.
        raise

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
