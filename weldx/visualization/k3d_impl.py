"""Contains a k3d based visualization of coordinate systems and spatial data."""

from typing import Union, List, Dict, Tuple

import k3d
import numpy as np
import pandas as pd
from IPython.core.display import display
from ipywidgets import (
    Checkbox,
    Dropdown,
    HBox,
    IntSlider,
    Layout,
    Play,
    VBox,
    jslink,
)
from k3d.platonic import Octahedron

from weldx import geometry as geo
from .colors import (
    RGB_BLACK,
    RGB_RED,
    RGB_GREEN,
    RGB_BLUE,
    color_generator_function,
    get_color,
)


class CoordinateSystemVisualizerK3D:
    """Visualizes a `weldx.transformations.LocalCoordinateSystem` using k3d."""

    def __init__(
        self,
        lcs,
        plot: k3d.Plot = None,
        name: str = None,
        color: int = RGB_BLACK,
        show_origin=True,
        show_trace=True,
        show_vectors=True,
    ):
        """Create a `CoordinateSystemVisualizerK3D`.

        Parameters
        ----------
        lcs : weldx.transformations.LocalCoordinateSystem
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
            colors=[[RGB_RED, RGB_RED], [RGB_GREEN, RGB_GREEN], [RGB_BLUE, RGB_BLUE]],
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

        self.origin = Octahedron(size=0.1).mesh
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
        coordinates = lcs.coordinates.isel(
            time=index, missing_dims="ignore"
        ).values.astype("float32")

        orientation = lcs.orientation.isel(
            time=index, missing_dims="ignore"
        ).values.astype("float32")

        return coordinates, orientation

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
        lcs : weldx.transformations.LocalCoordinateSystem
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


class CoordinateSystemManagerVisualizerK3D:
    """Visualizes a `weldx.transformations.CoordinateSystemManager` using k3d."""

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
        csm : weldx.transformations.CoordinateSystemManager
            The `weldx.transformations.CoordinateSystemManager` instance that should be
            visualized
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
            of the `weldx.transformations.CoordinateSystemManager` instance will be used
        title : str
            The title of the plot
        limits : List[Tuple[float, float]]
            The limits of the plotted volume
        time : pandas.DatetimeIndex, pandas.TimedeltaIndex, List[pandas.Timestamp], or \
               weldx.transformations.LocalCoordinateSystem
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
            reference_system = self._csm.root_system_name

        grid_auto_fit = True
        grid = (-1, -1, -1, 1, 1, 1)
        if limits is not None:
            grid_auto_fit = False
            if len(limits) == 1:
                grid = [limits[0][int(i / 3)] for i in range(6)]
            else:
                grid = [limits[i % 3][int(i / 3)] for i in range(6)]

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
                color=get_color(lcs_name, colors, self._color_generator),
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
                color=get_color(data_name, colors, self._color_generator),
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
        self._title = None
        if title is not None:
            self._title = k3d.text2d(
                f"<b>{title}</b>",
                position=(0.5, 0),
                color=RGB_BLACK,
                is_html=True,
                size=1.5,
                reference_point="ct",
            )
            plot += self._title

        # add time info
        self._time = time
        self._time_ref = time_ref
        self._time_info = None
        if time is not None:
            self._time_info = k3d.text2d(
                f"<b>time:</b> {time[0]}",
                position=(0, 1),
                color=RGB_BLACK,
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
        def _reference_callback(change):
            self.update_reference_system(change["new"])

        def _time_callback(change):
            self.update_time_index(change["new"])

        def _vectors_callback(change):
            self.show_vectors(change["new"])

        def _origins_callback(change):
            self.show_origins(change["new"])

        def _traces_callback(change):
            self.show_traces(change["new"])

        def _labels_callback(change):
            self.show_labels(change["new"])

        def _data_callback(change):
            self.set_data_visualization_method(change["new"])

        def _data_labels_callback(change):
            self.show_data_labels(change["new"])

        def _wireframe_callback(change):
            self.show_wireframes(change["new"])

        # register callbacks
        time_slider.observe(_time_callback, names="value")
        reference_dropdown.observe(_reference_callback, names="value")
        vectors_cb.observe(_vectors_callback, names="value")
        origin_cb.observe(_origins_callback, names="value")
        traces_cb.observe(_traces_callback, names="value")
        labels_cb.observe(_labels_callback, names="value")
        data_dropdown.observe(_data_callback, names="value")
        data_labels_cb.observe(_data_labels_callback, names="value")
        wf_cb.observe(_wireframe_callback, names="value")

        # create control panel
        row_1 = HBox([time_slider, play, reference_dropdown])
        row_2 = HBox([vectors_cb, origin_cb, traces_cb, labels_cb])
        if len(self._data_vis) > 0:
            row_3 = HBox([data_dropdown, wf_cb, data_labels_cb])
            return VBox([row_1, row_2, row_3])
        return VBox([row_1, row_2])

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

    def show_data_labels(self, show_data_labels: bool):
        """Set the visibility of data labels.

        Parameters
        ----------
        show_data_labels: bool
            If `True`, labels are shown.

        """
        for _, data_vis in self._data_vis.items():
            data_vis.show_label(show_data_labels)

    def show_labels(self, show_labels: bool):
        """Set the visibility of the coordinate systems' labels.

        Parameters
        ----------
        show_labels : bool
            If `True`, the coordinate systems' labels are shown.

        """
        for _, lcs_vis in self._lcs_vis.items():
            lcs_vis.show_label(show_labels)

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

    def show_vectors(self, show_vectors: bool):
        """Set the visibility of the coordinate axis vectors.

        Parameters
        ----------
        show_vectors : bool
            If `True`, the coordinate axis vectors are shown.

        """
        for _, lcs_vis in self._lcs_vis.items():
            lcs_vis.show_vectors(show_vectors)

    def show_wireframes(self, show_wireframes: bool):
        """Set if meshes should be drawn in wireframe mode.

        Parameters
        ----------
        show_wireframes : bool
            If `True`, meshes are rendered as wireframes

        """
        for _, data_vis in self._data_vis.items():
            data_vis.show_wireframe(show_wireframes)

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


class SpatialDataVisualizer:
    """Visualizes spatial data."""

    visualization_methods = ["auto", "point", "mesh", "both"]

    def __init__(
        self,
        data,
        name: str,
        cs_vis: CoordinateSystemVisualizerK3D,
        plot: k3d.Plot = None,
        color: int = RGB_BLACK,
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
