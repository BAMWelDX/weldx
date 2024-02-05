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
    if "weldx_widgets" in str(mnf):
        import sys as _sys
        from unittest.mock import MagicMock as _MagickMock

        from weldx.util import check_matplotlib_available as _mpl_avail

        def _warn(stacklevel=2):
            import warnings as _warnings

            _warnings.warn(
                "'weldx_widgets' unavailable! Cannot plot. "
                "Please install weldx_widgets prior plotting.",
                stacklevel=stacklevel,
            )

        # warn now, that weldx_widgets is not available.
        _warn(stacklevel=2)

        class _Hint(_MagickMock):  # warn again, if actual features are requested.
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                _warn(stacklevel=3)

        CoordinateSystemManagerVisualizerK3D = _Hint
        SpatialDataVisualizer = _Hint

        axes_equal = draw_coordinate_system_matplotlib = new_3d_figure_and_axes = (
            plot_coordinate_system_manager_matplotlib
        ) = plot_coordinate_systems = plot_local_coordinate_system_matplotlib = (
            plot_spatial_data_matplotlib
        ) = _warn
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
