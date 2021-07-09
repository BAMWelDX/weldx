"""Contains methods and classes for coordinate transformations."""
from __future__ import annotations

import itertools
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import pint
import xarray as xr

from weldx import util
from weldx.constants import WELDX_UNIT_REGISTRY as UREG
from weldx.core import TimeSeries
from weldx.geometry import SpatialData

from .local_cs import LocalCoordinateSystem
from .types import (
    types_coordinates,
    types_orientation,
    types_time_and_lcs,
    types_timeindex,
)

# shared type aliases
from .util import build_time_index

# only import heavy-weight packages on type checking
if TYPE_CHECKING:  # pragma: no cover
    import matplotlib.axes
    import networkx as nx

    import weldx  # noqa


_DEFAULT_LEN_UNIT = UREG.millimeters
_DEFAULT_ANG_UNIT = UREG.rad

__all__ = ["CoordinateSystemManager"]


class CoordinateSystemManager:
    """Handles hierarchical dependencies between multiple coordinate systems.

    Notes
    -----
    Learn how to use this class by reading the
    :doc:`Tutorial <../tutorials/transformations_02_coordinate_system_manager>`.

    """

    _id_gen = itertools.count()

    @dataclass
    class CoordinateSystemData:
        """Class that stores data and the coordinate system, the data is assigned to."""

        coordinate_system_name: str
        data: xr.DataArray

    def __init__(
        self,
        root_coordinate_system_name: str,
        coordinate_system_manager_name: Union[str, None] = None,
        time_ref: pd.Timestamp = None,
    ):
        """Construct a coordinate system manager.

        Parameters
        ----------
        root_coordinate_system_name
            Name of the root coordinate system.
        coordinate_system_manager_name
            Name of the coordinate system manager. If `None` is passed, a default name
            is chosen.
        time_ref
            A reference timestamp. If it is defined, all time dependent information
            returned by the CoordinateSystemManager will refer to it by default.

        Returns
        -------
        CoordinateSystemManager

        """
        from networkx import DiGraph

        if coordinate_system_manager_name is None:
            coordinate_system_manager_name = self._generate_default_name()
        self._name = coordinate_system_manager_name
        if time_ref is not None and not isinstance(time_ref, pd.Timestamp):
            time_ref = pd.Timestamp(time_ref)
        self._reference_time = time_ref

        self._data = {}
        self._root_system_name = root_coordinate_system_name

        self._sub_system_data_dict = {}

        self._graph = DiGraph()
        self._add_coordinate_system_node(root_coordinate_system_name)

    @classmethod
    def _from_subsystem_graph(
        cls,
        root_coordinate_system_name: str,
        coordinate_system_manager_name: Union[str, None] = None,
        time_ref: pd.Timestamp = None,
        graph: Union[nx.DiGraph, None] = None,
        subsystems=None,
    ):
        """Construct a coordinate system manager from existing graph and subsystems.

        This function is used internally to recreate subsystem structures.

        Parameters
        ----------
        root_coordinate_system_name
            Name of the root coordinate system.
        coordinate_system_manager_name
            Name of the coordinate system manager. If `None` is passed, a default name
            is chosen.
        time_ref
            A reference timestamp. If it is defined, all time dependent information
            returned by the CoordinateSystemManager will refer to it by default.
        graph:
            Pass on an existing graph.
        subsystems:
            A dictionary containing data about the CSMs attached subsystems.

        Returns
        -------
        CoordinateSystemManager

        """
        csm = cls(root_coordinate_system_name, coordinate_system_manager_name, time_ref)

        if subsystems is not None:
            csm._sub_system_data_dict = subsystems

        if graph is not None:
            csm._graph = graph

        return csm

    def __repr__(self):
        """Output representation of a CoordinateSystemManager class."""
        return (
            f"<CoordinateSystemManager>\nname:\n\t{self._name}\n"
            f"reference time:\n\t {self.reference_time}\n"
            f"coordinate systems:\n\t {self.coordinate_system_names}\n"
            f"data:\n\t {self._data!r}\n"
            f"sub systems:\n\t {self._sub_system_data_dict.keys()}\n"
            f")"
        )

    def __eq__(self: "CoordinateSystemManager", other: "CoordinateSystemManager"):
        """Test equality of CSM instances."""
        # todo: also check data  -> add tests
        if not isinstance(other, self.__class__):
            return False

        graph_0 = self.graph
        graph_1 = other.graph

        if self.name != other.name:
            return False

        if self.reference_time != other.reference_time:
            return False

        if len(graph_0.nodes) != len(graph_1.nodes):
            return False

        # if self.sub_system_data != other.sub_system_data:
        if not self._compare_subsystems_equal(
            self.sub_system_data, other.sub_system_data
        ):
            return False

        # check nodes
        for node in graph_0.nodes:
            if node not in graph_1.nodes:
                return False

        # check edges
        for edge in graph_0.edges:
            if edge not in graph_1.edges:
                return False

        # check coordinate systems
        for edge in graph_0.edges:
            lcs_0 = self.graph.edges[(edge[0], edge[1])]["lcs"]
            lcs_1 = other.graph.edges[(edge[0], edge[1])]["lcs"]
            if lcs_0 != lcs_1:
                return False

        return True

    @property
    def lcs(self) -> List[LocalCoordinateSystem]:
        """Get a list of all attached `~weldx.transformations.LocalCoordinateSystem` \
        instances.

        Only the defined systems and not the automatically generated inverse systems
        are included.

        Returns
        -------
        List[~weldx.transformations.LocalCoordinateSystem] :
           List of all attached `~weldx.transformations.LocalCoordinateSystem`
           instances.

        """
        return [
            self.graph.edges[edge]["lcs"]
            for edge in self.graph.edges
            if self.graph.edges[edge]["defined"]
        ]

    @property
    def lcs_time_dependent(self) -> List[LocalCoordinateSystem]:
        """Get a list of all attached time dependent \
        `~weldx.transformations.LocalCoordinateSystem` instances.

        Returns
        -------
        List[~weldx.transformations.LocalCoordinateSystem] :
            List of all attached time dependent
            `~weldx.transformations.LocalCoordinateSystem` instances

        """
        return [lcs for lcs in self.lcs if lcs.is_time_dependent]

    @property
    def uses_absolute_times(self) -> bool:
        """Return `True` if the CSM or one of its coord. systems has a reference time.

        Returns
        -------
        bool :
            `True` if the `CoordinateSystemManager` or one of its attached coordinate
            systems possess a reference time. `False` otherwise

        """
        return self._has_lcs_with_time_ref or self.has_reference_time

    @property
    def has_reference_time(self) -> bool:
        """Return `True` if the coordinate system manager has a reference time.

        Returns
        -------
        bool :
            `True` if the coordinate system manager has a reference time, `False`
            otherwise.

        """
        return self.reference_time is not None

    def _add_coordinate_system_node(self, coordinate_system_name):
        self._check_new_coordinate_system_name(coordinate_system_name)
        self._graph.add_node(coordinate_system_name, data=[])

    def _add_edges(self, node_from: str, node_to: str, lcs: LocalCoordinateSystem):
        """Add an edge to the internal graph.

        Parameters
        ----------
        node_from :
            Start node of the edge
        node_to :
            End node of the edge
        lcs :
            Local coordinate system

        """
        self._graph.add_edge(node_from, node_to, lcs=lcs, defined=True)

        # only store inverted lcs if coordinates and orientations are discrete values
        lcs_invert = None
        if not isinstance(lcs.coordinates, TimeSeries):
            lcs_invert = lcs.invert()
        self._graph.add_edge(node_to, node_from, lcs=lcs_invert, defined=False)

    def _check_coordinate_system_exists(self, coordinate_system_name: str):
        """Raise an exception if the specified coordinate system does not exist.

        Parameters
        ----------
        coordinate_system_name :
            Name of the coordinate system, that should be
            checked.

        """
        if not self.has_coordinate_system(coordinate_system_name):
            raise ValueError(
                "There is no coordinate system with name " + str(coordinate_system_name)
            )

    def _check_new_coordinate_system_name(self, coordinate_system_name: str):
        """Raise an exception if the new coordinate systems' name is invalid.

        Parameters
        ----------
        coordinate_system_name :
            Name of the new coordinate system, that should be checked.

        """
        if not isinstance(coordinate_system_name, str):
            raise TypeError("The coordinate system name must be a string.")
        if self.has_coordinate_system(coordinate_system_name):
            raise ValueError(
                "There already is a coordinate system with name "
                + str(coordinate_system_name)
            )

    @classmethod
    def _compare_subsystems_equal(cls, data: Dict, other: Dict) -> bool:
        """Compare if two subsystem data dictionaries are equal.

        Parameters
        ----------
        data:
            First subsystem data dictionary.
        other
            Second subsystem data dictionary.

        Returns
        -------
        bool:
            `True` if both dictionaries are identical, `False` otherwise.

        """
        if len(data) != len(other):
            return False
        for subsystem_name, subsystem_data in data.items():
            if subsystem_name not in other:
                return False
            other_data = other[subsystem_name]
            if subsystem_data["common node"] != other_data["common node"]:
                return False
            if subsystem_data["root"] != other_data["root"]:
                return False
            if subsystem_data["time_ref"] != other_data["time_ref"]:
                return False
            if set(subsystem_data["neighbors"]) != set(other_data["neighbors"]):
                return False
            if set(subsystem_data["original members"]) != set(
                other_data["original members"]
            ):
                return False
            if not cls._compare_subsystems_equal(
                subsystem_data["sub system data"], other_data["sub system data"]
            ):
                return False
        return True

    @staticmethod
    def _generate_default_name() -> str:
        """Get a default name for the current coordinate system manager instance.

        Returns
        -------
        str:
            Default name.

        """
        id_ = next(CoordinateSystemManager._id_gen)  # skipcq: PTC-W0063
        return f"Coordinate system manager {id_}"

    @property
    def _extended_sub_system_data(self) -> Dict:
        """Get an extended copy of the internal sub system data.

        The function adds a list of potential child coordinate systems to each
        sub system. Coordinate systems in this list might belong to other sub systems
        that share a common coordinate system with the current sub system.

        Returns
        -------
        Dict:
            Extended copy of the internal sub system data.

        """
        sub_system_data_dict = deepcopy(self._sub_system_data_dict)
        for _, sub_system_data in sub_system_data_dict.items():
            potential_members = []
            for cs_name in sub_system_data["neighbors"]:
                potential_members += self.get_child_system_names(cs_name, False)

            sub_system_data["nodes"] = potential_members + sub_system_data["neighbors"]

        return sub_system_data_dict

    @staticmethod
    def _get_sub_system_members(
        ext_sub_system_data, ext_sub_system_data_dict
    ) -> List[str]:
        """Get a list with all coordinate system names, that belong to the sub system.

        Parameters
        ----------
        ext_sub_system_data:
            The extended sub system data of a single sub system.
        ext_sub_system_data_dict:
            Dictionary containing the extended sub system data of all sub systems.

        Returns
        -------
        List[str]:
            List of all the sub systems coordinate systems.

        """
        all_members = ext_sub_system_data["nodes"]
        for _, other_sub_system_data in ext_sub_system_data_dict.items():
            if other_sub_system_data["common node"] in all_members:
                all_members = [
                    cs_name
                    for cs_name in all_members
                    if cs_name not in other_sub_system_data["nodes"]
                ]

        all_members += [ext_sub_system_data["common node"]]
        return all_members

    @property
    def _has_lcs_with_time_ref(self):
        """Return `True` if one of the attached coordinate systems has a reference time.

        Returns
        -------
        bool :
            `True` if one of the attached coordinate systems has a reference time.
            `False` otherwise

        """
        return any(lcs.has_reference_time for lcs in self.lcs_time_dependent)

    def _ipython_display_(self):
        """Display the coordinate system manager as plot in jupyter notebooks."""
        self.plot_graph()

    @property
    def _number_of_time_dependent_lcs(self):
        """Get the number of time dependent coordinate systems.

        Note that the automatically added inverse systems have no effect on the returned
        val

        Returns
        -------
        int :
            Number of time dependent coordinate systems

        """
        return len(self.lcs_time_dependent)

    def _update_local_coordinate_system(
        self, node_from: str, node_to: str, lcs: LocalCoordinateSystem
    ):
        """Update the local coordinate systems on the edges between two nodes.

        Parameters
        ----------
        node_from :
            Start node of the edge
        node_to :
            End node of the edge
        lcs :
            Local coordinate system

        """
        edge_from_to = self.graph.edges[(node_from, node_to)]
        edge_from_to["lcs"] = lcs
        edge_from_to["defined"] = True

        edge_to_from = self.graph.edges[(node_to, node_from)]
        if isinstance(lcs.coordinates, TimeSeries):
            edge_to_from["lcs"] = None
        else:
            edge_to_from["lcs"] = lcs.invert()
        edge_to_from["defined"] = False

    @property
    def graph(self) -> nx.DiGraph:
        """Get the internal graph.

        Returns
        -------
        networkx.DiGraph

        """
        return self._graph

    @property
    def name(self) -> str:
        """Get the name of the coordinate system manager instance.

        Returns
        -------
        str:
            Name of the coordinate system manager instance.

        """
        return self._name

    @property
    def number_of_coordinate_systems(self) -> int:
        """Get the number of coordinate systems inside the coordinate system manager.

        Returns
        -------
        int
            Number of coordinate systems

        """
        return self._graph.number_of_nodes()

    @property
    def number_of_subsystems(self) -> int:
        """Get the number of attached subsystems.

        Returns
        -------
        int:
            Number of attached subsystems.

        """
        return len(self._sub_system_data_dict)

    @property
    def reference_time(self) -> pd.Timestamp:
        """Get the reference time of the `CoordinateSystemManager`.

        Returns
        -------
        pandas.Timestamp :
            Reference time of the `CoordinateSystemManager`

        """
        return self._reference_time

    @property
    def root_system_name(self) -> str:
        """Get the name of the root system.

        Returns
        -------
        str:
            Name of the root system

        """
        return self._root_system_name

    @property
    def sub_system_data(self) -> Dict:
        """Get a dictionary containing data about the attached subsystems."""
        return self._sub_system_data_dict

    @property
    def subsystem_names(self) -> List[str]:
        """Get the names of all subsystems.

        Returns
        -------
        List[str]:
            List with subsystem names.

        """
        return list(self._sub_system_data_dict.keys())

    def add_cs(
        self,
        coordinate_system_name: str,
        reference_system_name: str,
        lcs: LocalCoordinateSystem,
        lsc_child_in_parent: bool = True,
    ):
        """Add a coordinate system to the coordinate system manager.

        If the specified system already exists with the same parent system it will be
        updated. If the parent systems does not match, an exception is raised.

        Notes
        -----
        The time component of coordinate systems without defined reference time is
        assumed to refer to the same reference time as the `CoordinateSystemManager`.
        In case that the `CoordinateSystemManager` does not possess a reference time,
        you have to assure that either all or none of the added coordinate systems have
        a reference time.
        Violation of this rule will cause an exception.
        If neither the `CoordinateSystemManager` nor the attached coordinate systems
        have a reference time, all time deltas are expected to have common but undefined
        reference time.

        Parameters
        ----------
        coordinate_system_name
            Name of the new coordinate system.
        reference_system_name
            Name of the parent system. This must have been already added.
        lcs
            An instance of
            `~weldx.transformations.LocalCoordinateSystem` that describes how the new
            coordinate system is oriented in its parent system.
        lsc_child_in_parent
            If set to `True`, the passed
            `~weldx.transformations.LocalCoordinateSystem` instance describes
            the new system orientation towards is parent. If `False`, it describes
            how the parent system is positioned in its new child system.

        """
        if not isinstance(lcs, LocalCoordinateSystem):
            raise TypeError(
                "'local_coordinate_system' must be an instance of "
                + "weldx.transformations.LocalCoordinateSystem"
            )

        if (
            lcs.is_time_dependent  # always add static lcs
            and self._number_of_time_dependent_lcs > 0  # CSM is not static
            and (
                (lcs.has_reference_time and not self.uses_absolute_times)
                or (
                    (not lcs.has_reference_time and not self.has_reference_time)
                    and self.uses_absolute_times
                )
            )
        ):
            raise Exception(
                "Inconsistent usage of reference times! If you didn't specify a "
                "reference time for the CoordinateSystemManager, either all or "
                "none of the added coordinate systems must have a reference time."
            )

        if self.has_coordinate_system(coordinate_system_name):
            # todo:
            #  discuss: update and add functionality should be separated
            #  why?   : to prevent errors. Misspelling of the system name might cause
            #           unwanted updates or unwanted additions. Separate function can
            #           catch that by knowing about the users intention.
            if not self.is_neighbor_of(coordinate_system_name, reference_system_name):
                raise ValueError(
                    f'Can not update coordinate system. "{reference_system_name}" is '
                    f"not a neighbor of {coordinate_system_name}"
                )
            if lsc_child_in_parent:
                self._update_local_coordinate_system(
                    coordinate_system_name,
                    reference_system_name,
                    lcs,
                )
            else:
                self._update_local_coordinate_system(
                    reference_system_name,
                    coordinate_system_name,
                    lcs,
                )
        else:
            self._check_coordinate_system_exists(reference_system_name)
            self._add_coordinate_system_node(coordinate_system_name)
            if lsc_child_in_parent:
                self._add_edges(
                    coordinate_system_name,
                    reference_system_name,
                    lcs,
                )
            else:
                self._add_edges(
                    reference_system_name,
                    coordinate_system_name,
                    lcs,
                )

    def relabel(self, mapping: Dict[str, str]):
        """Rename one or more nodes of the graph.

        See `networkx.relabel.relabel_nodes` for details.
        CSM will always be changed inplace.

        Parameters
        ----------
        mapping
             A dictionary mapping with the old node names as keys and new node names
             labels as values.

        """
        if self.subsystems:
            raise NotImplementedError("Cannot relabel nodes on merged systems.")

        if self.root_system_name in mapping:
            self._root_system_name = mapping[self._root_system_name]

        from networkx import relabel_nodes

        relabel_nodes(self.graph, mapping, copy=False)

    def assign_data(
        self,
        data: Union[xr.DataArray, SpatialData],
        data_name: str,
        coordinate_system_name: str,
    ):
        """Assign spatial data to a coordinate system.

        Parameters
        ----------
        data :
            Spatial data
        data_name :
            Name of the data.
        coordinate_system_name :
            Name of the coordinate system the data should be
            assigned to.

        """
        # TODO: How to handle time dependent data? some things to think about:
        # - times of coordinate system and data are not equal
        # - which time is taken as reference? (probably the one of the data)
        # - what happens during cal of time interpolation functions with data? Also
        #   interpolated or not?
        if not isinstance(data_name, str):
            raise TypeError("The data name must be a string.")
        if data_name in self._data:
            raise ValueError(f"There already is a dataset with the name '{data_name}'.")
        self._check_coordinate_system_exists(coordinate_system_name)

        if not isinstance(data, (xr.DataArray, SpatialData)):
            data = xr.DataArray(data, dims=["n", "c"], coords={"c": ["x", "y", "z"]})

        self._data[data_name] = self.CoordinateSystemData(coordinate_system_name, data)
        self._graph.nodes[coordinate_system_name]["data"].append(data_name)

    def create_cs(
        self,
        coordinate_system_name: str,
        reference_system_name: str,
        orientation: types_orientation = None,
        coordinates: types_coordinates = None,
        time: Union[pd.TimedeltaIndex, pd.DatetimeIndex] = None,
        time_ref: pd.Timestamp = None,
        lsc_child_in_parent: bool = True,
    ):
        """Create a coordinate system and add it to the coordinate system manager.

        This function uses the ``__init__`` method of the
        `~weldx.transformations.LocalCoordinateSystem` class.

        Parameters
        ----------
        coordinate_system_name :
            Name of the new coordinate system.
        reference_system_name :
            Name of the parent system. This must have been already added.
        orientation :
            Matrix of 3 orthogonal column vectors which represent
            the coordinate systems orientation. Keep in mind, that the columns of the
            corresponding orientation matrix is equal to the normalized orientation
            vectors. So each orthogonal transformation matrix can also be
            provided as orientation.
            Passing a scipy.spatial.transform.Rotation object is also supported.
        coordinates :
            Coordinates of the origin.
        time :
            Time data for time dependent coordinate systems.
        time_ref :
            Reference time for time dependent coordinate systems
        lsc_child_in_parent :
            If set to `True`, the passed
            `~weldx.transformations.LocalCoordinateSystem` instance describes
            the new system orientation towards is parent. If `False`, it describes
            how the parent system is positioned in its new child system.

        """
        lcs = LocalCoordinateSystem(orientation, coordinates, time, time_ref)
        self.add_cs(
            coordinate_system_name, reference_system_name, lcs, lsc_child_in_parent
        )

    def create_cs_from_euler(
        self,
        coordinate_system_name: str,
        reference_system_name: str,
        sequence,
        angles,
        degrees: bool = False,
        coordinates: types_coordinates = None,
        time: pd.DatetimeIndex = None,
        lsc_child_in_parent: bool = True,
    ):
        """Create a coordinate system and add it to the coordinate system manager.

        This function uses the `~weldx.transformations.LocalCoordinateSystem.from_euler`
        method of the `~weldx.transformations.LocalCoordinateSystem` class.

        Parameters
        ----------
        coordinate_system_name :
            Name of the new coordinate system.
        reference_system_name :
            Name of the parent system. This must have been already added.
        sequence :
            Specifies sequence of axes for rotations. Up to 3 characters
            belonging to the set {``X``, ``Y``, ``Z``} for intrinsic rotations,
            or {``x``, ``y``, ``z``} for extrinsic rotations.
            Extrinsic and intrinsic rotations cannot be mixed in one function call.
        angles :
            Euler angles specified in radians (degrees is False) or degrees
            (degrees is True). For a single character seq, angles can be:
            - a single value
            - array_like with shape (N,), where each angle[i] corresponds to a single
            rotation
            - array_like with shape (N, 1), where each angle[i, 0] corresponds to a
            single rotation
            For 2- and 3-character wide seq, angles can be:
            - array_like with shape (W,) where W is the width of seq, which corresponds
            to a single rotation with W axes
            - array_like with shape (N, W) where each angle[i] corresponds to a sequence
            of Euler angles describing a single rotation
        degrees :
            If True, then the given angles are assumed to be in degrees.
            Default is False.
        coordinates :
            Coordinates of the origin.
        time :
            Time data for time dependent coordinate systems.
        lsc_child_in_parent :
            If set to `True`, the passed
            `~weldx.transformations.LocalCoordinateSystem` instance describes
            the new system orientation towards is parent. If `False`, it describes
            how the parent system is positioned in its new child system.


        """
        lcs = LocalCoordinateSystem.from_euler(
            sequence, angles, degrees, coordinates, time
        )
        self.add_cs(
            coordinate_system_name, reference_system_name, lcs, lsc_child_in_parent
        )

    def create_cs_from_xyz(
        self,
        coordinate_system_name: str,
        reference_system_name: str,
        vec_x,
        vec_y,
        vec_z,
        coordinates: types_coordinates = None,
        time: pd.DatetimeIndex = None,
        lsc_child_in_parent: bool = True,
    ):
        """Create a coordinate system and add it to the coordinate system manager.

        This function uses the `~weldx.transformations.LocalCoordinateSystem.from_xyz`
        method of the `~weldx.transformations.LocalCoordinateSystem` class.

        Parameters
        ----------
        coordinate_system_name :
            Name of the new coordinate system.
        reference_system_name :
            Name of the parent system. This must have been already added.
        vec_x :
            Vector defining the x-axis
        vec_y :
            Vector defining the y-axis
        vec_z :
            Vector defining the z-axis
        coordinates :
            Coordinates of the origin.
        time :
            Time data for time dependent coordinate systems.
        lsc_child_in_parent :
            If set to `True`, the passed
            `~weldx.transformations.LocalCoordinateSystem` instance describes
            the new system orientation towards is parent. If `False`, it describes
            how the parent system is positioned in its new child system.

        """
        lcs = LocalCoordinateSystem.from_xyz(vec_x, vec_y, vec_z, coordinates, time)
        self.add_cs(
            coordinate_system_name, reference_system_name, lcs, lsc_child_in_parent
        )

    def create_cs_from_xy_and_orientation(
        self,
        coordinate_system_name: str,
        reference_system_name: str,
        vec_x,
        vec_y,
        positive_orientation: bool = True,
        coordinates: types_coordinates = None,
        time: pd.DatetimeIndex = None,
        lsc_child_in_parent: bool = True,
    ):
        """Create a coordinate system and add it to the coordinate system manager.

        This function uses the
        `~weldx.transformations.LocalCoordinateSystem.from_xy_and_orientation` method
        of the `~weldx.transformations.LocalCoordinateSystem` class.

        Parameters
        ----------
        coordinate_system_name :
            Name of the new coordinate system.
        reference_system_name :
            Name of the parent system. This must have been already added.
        vec_x :
            Vector defining the x-axis
        vec_y :
            Vector defining the y-axis
        positive_orientation :
            Set to True if the orientation should
            be positive and to False if not (Default value = True)
        coordinates :
            Coordinates of the origin.
        time :
            Time data for time dependent coordinate systems.
        lsc_child_in_parent :
            If set to `True`, the passed
            `~weldx.transformations.LocalCoordinateSystem` instance describes
            the new system orientation towards is parent. If `False`, it describes
            how the parent system is positioned in its new child system.

        """
        lcs = LocalCoordinateSystem.from_xy_and_orientation(
            vec_x, vec_y, positive_orientation, coordinates, time
        )
        self.add_cs(
            coordinate_system_name, reference_system_name, lcs, lsc_child_in_parent
        )

    def create_cs_from_xz_and_orientation(
        self,
        coordinate_system_name: str,
        reference_system_name: str,
        vec_x,
        vec_z,
        positive_orientation=True,
        coordinates: types_coordinates = None,
        time: pd.DatetimeIndex = None,
        lsc_child_in_parent: bool = True,
    ):
        """Create a coordinate system and add it to the coordinate system manager.

        This function uses the
        `~weldx.transformations.LocalCoordinateSystem.from_xz_and_orientation` method
        of the `~weldx.transformations.LocalCoordinateSystem` class.

        Parameters
        ----------
        coordinate_system_name :
            Name of the new coordinate system.
        reference_system_name :
            Name of the parent system. This must have been already added.
        vec_x :
            Vector defining the x-axis
        vec_z :
            Vector defining the z-axis
        positive_orientation :
            Set to True if the orientation should
            be positive and to False if not (Default value = True)
        coordinates :
            Coordinates of the origin.
        time :
            Time data for time dependent coordinate systems.
        lsc_child_in_parent :
            If set to `True`, the passed
            `~weldx.transformations.LocalCoordinateSystem` instance describes
            the new system orientation towards is parent. If `False`, it describes
            how the parent system is positioned in its new child system.

        """
        lcs = LocalCoordinateSystem.from_xz_and_orientation(
            vec_x, vec_z, positive_orientation, coordinates, time
        )
        self.add_cs(
            coordinate_system_name, reference_system_name, lcs, lsc_child_in_parent
        )

    def create_cs_from_yz_and_orientation(
        self,
        coordinate_system_name: str,
        reference_system_name: str,
        vec_y,
        vec_z,
        positive_orientation: bool = True,
        coordinates: types_coordinates = None,
        time: pd.DatetimeIndex = None,
        lsc_child_in_parent: bool = True,
    ):
        """Create a coordinate system and add it to the coordinate system manager.

        This function uses the
        `~weldx.transformations.LocalCoordinateSystem.from_yz_and_orientation` method
        of the `~weldx.transformations.LocalCoordinateSystem` class.

        Parameters
        ----------
        coordinate_system_name :
            Name of the new coordinate system.
        reference_system_name :
            Name of the parent system. This must have been already added.
        vec_y :
            Vector defining the y-axis
        vec_z :
            Vector defining the z-axis
        positive_orientation :
            Set to True if the orientation should
            be positive and to False if not (Default value = True)
        coordinates :
            Coordinates of the origin.
        time :
            Time data for time dependent coordinate systems.
        lsc_child_in_parent :
            If set to `True`, the passed
            `~weldx.transformations.LocalCoordinateSystem` instance describes
            the new system orientation towards is parent. If `False`, it describes
            how the parent system is positioned in its new child system.

        """
        lcs = LocalCoordinateSystem.from_yz_and_orientation(
            vec_y, vec_z, positive_orientation, coordinates, time
        )
        self.add_cs(
            coordinate_system_name, reference_system_name, lcs, lsc_child_in_parent
        )

    def delete_cs(self, coordinate_system_name: str, delete_children: bool = False):
        """Delete a coordinate system from the coordinate system manager.

        If the Coordinate system manager has attached sub system, there are multiple
        possible  consequences.

        - All subsystems attached to the deleted coordinate system or one of
          its child systems are removed from the coordinate system manager
        - If the coordinate system is part of a subsystem and belongs to the systems
          that were present when the subsystem was merged, the subsystem is removed and
          can not be restored using `subsystems` or `unmerge`. Coordinate systems
          of the subsystem that aren't a child of the deleted coordinate system will
          remain in the coordinate system manager
        - If the coordinate system is part of a subsystem but was added after merging,
          only the systems and its children are removed. The subsystem remains in the
          coordinate system manager.

        Parameters
        ----------
        coordinate_system_name :
            Name of the coordinate system that should be deleted.
        delete_children :
            If `False`, an exception is raised if the coordinate system has one or more
            children since deletion would cause them to be disconnected to the root.
            If `True`, all children are deleted as well.

        """
        if not self.has_coordinate_system(coordinate_system_name):
            return

        if coordinate_system_name == self._root_system_name:
            raise ValueError("The root system can't be deleted.")

        children = self.get_child_system_names(coordinate_system_name, False)

        if not delete_children and len(children) > 0:
            raise Exception(
                f'Can not delete coordinate system "{coordinate_system_name}". It '
                "has one or more children that would be disconnected to the root "
                f'after deletion. Set the delete_children option to "True" to '
                f"delete the coordinate system and all its children. "
                f"The attached child systems are: {children}"
            )

        # update subsystems
        from networkx import shortest_path

        remove_systems = []
        for sub_system_name, sub_system_data in self._sub_system_data_dict.items():
            if (
                coordinate_system_name in sub_system_data["original members"]
            ) or coordinate_system_name in shortest_path(
                self.graph, sub_system_data["root"], self._root_system_name
            ):
                remove_systems += [sub_system_name]

        for sub_system_name in remove_systems:
            del self._sub_system_data_dict[sub_system_name]

        # delete nodes and edges
        if delete_children:
            for child in children:
                self._graph.remove_node(child)
        self._graph.remove_node(coordinate_system_name)

    def get_child_system_names(
        self, coordinate_system_name: str, neighbors_only: bool = True
    ) -> List[str]:
        """Get a list with the passed coordinate systems children.

        Parameters
        ----------
        coordinate_system_name :
            Name of the coordinate system
        neighbors_only :
            If `True`, only child coordinate systems that are directly connected to the
            specified coordinate system are included in the returned list. If `False`,
            child systems of arbitrary hierarchical depth are included.

        Returns
        -------
        List[str]:
            List of child systems.

        """
        if neighbors_only:
            return [
                cs
                for cs in self.neighbors(coordinate_system_name)
                if cs != self.get_parent_system_name(coordinate_system_name)
            ]

        current_children = self.get_child_system_names(coordinate_system_name, True)
        all_children = deepcopy(current_children)
        while current_children:
            new_children = []
            for child in current_children:
                new_children += self.get_child_system_names(child, True)
            all_children += new_children
            current_children = new_children

        return all_children

    @property
    def coordinate_system_names(self) -> List:
        """Get the names of all contained coordinate systems.

        Returns
        -------
        List :
            List of coordinate system names.

        """
        return list(self.graph.nodes)

    @property
    def data_names(self) -> List[str]:
        """Get the names of the attached data sets.

        Returns
        -------
        List[str] :
            Names of the attached data sets

        """
        return list(self._data.keys())

    def get_data(
        self, data_name, target_coordinate_system_name=None
    ) -> Union[np.ndarray, SpatialData]:
        """Get the specified data, optionally transformed into any coordinate system.

        Parameters
        ----------
        data_name :
            Name of the data
        target_coordinate_system_name :
            Name of the target coordinate system. If it is not None or not identical to
            the owning coordinate system name, the data will be transformed to the
            desired system. (Default value = None)

        Returns
        -------
        numpy.ndarray
            Transformed data

        """
        data_struct = self._data[data_name]
        if (
            target_coordinate_system_name is None
            or target_coordinate_system_name == data_struct.coordinate_system_name
        ):
            return data_struct.data

        return self.transform_data(
            data_struct.data,
            data_struct.coordinate_system_name,
            target_coordinate_system_name,
        )

    def get_data_system_name(self, data_name: str) -> str:
        """Get the name of the data's reference coordinate system.

        Parameters
        ----------
        data_name :
            Name of the data

        Returns
        -------
        str :
            Name of the reference coordinate system

        """
        return self._data[data_name].coordinate_system_name

    def get_cs(
        self,
        coordinate_system_name: str,
        reference_system_name: str = None,
        time: Union[types_timeindex, str] = None,
        time_ref: pd.Timestamp = None,
    ) -> LocalCoordinateSystem:
        """Get a coordinate system in relation to another reference system.

        If no reference system is specified, the parent system will be used as
        reference.

        If any coordinate system that is involved in the coordinate transformation has
        a time dependency, the returned coordinate system will also be time dependent.

        The timestamps of the returned system depend on the functions time parameter.
        By default, the time union of all involved coordinate systems is taken.

        Parameters
        ----------
        coordinate_system_name :
            Name of the coordinate system.
        reference_system_name :
            Name of the reference coordinate system.
        time :
            Specifies the desired time of the returned coordinate system. You can also
            pass the name of another coordinate system to use its time attribute as
            reference.
        time_ref :
            The desired reference time of the returned coordinate system.

        Returns
        -------
        `~weldx.transformations.LocalCoordinateSystem` :
            The requested coordinate system.

        Notes
        -----
        **Reference time of the returned system**

        The reference time of the returned coordinate system depends on multiple
        factors like the one passed to the function and the internally stored reference
        times. Generally, the following rules apply:

        - if a reference time was passed to the function, it will be used as reference
          time of the returned coordinate system as long as a time was also passed to
          the function.
        - else the reference time of the `CoordinateSystemManager` will be used if it
          has one
        - if only the coordinate systems have a reference time, the lowest (earliest)
          will be used
        - if there is no reference time at all, the resulting coordinate system won't
          have one either
        - if no time was passed to the function, a passed reference time will be
          ignored
        - a `pandas.DatetimeIndex` always has its lowest date as implicit reference time
          which will be used if the `CoordinateSystemManager` doesn't possess one and
          the functions reference time isn't set.


        A overview of all possible combinations using a `pandas.TimedeltaIndex` or
        a `pint.Quantity` as ``time`` parameter is given in the table below.

        +------------+--------------+-----------+----------------+-----------------+
        | function   | function has | CSM has   | CS have        | Returned system |
        | has        | reference    | reference | reference      | uses reference  |
        | time       | time         | time      | times          | time of         |
        +============+==============+===========+================+=================+
        | Yes        | Yes          | Yes       | all/mixed/none | function        |
        +------------+--------------+-----------+----------------+-----------------+
        | No         | Yes          | Yes       | all/mixed/none | CSM             |
        +------------+--------------+-----------+----------------+-----------------+
        | Yes / No   | No           | Yes       | all/mixed/none | CSM             |
        +------------+--------------+-----------+----------------+-----------------+
        | Yes        | Yes          | No        | all            | function        |
        +------------+--------------+-----------+----------------+-----------------+
        | No         | Yes          | No        | all            | CS (lowest)     |
        +------------+--------------+-----------+----------------+-----------------+
        | Yes / No   | Yes / No     | No        | mixed          | impossible -> 1.|
        +------------+--------------+-----------+----------------+-----------------+
        | Yes        | Yes          | No        | none           | error -> 2.     |
        +------------+--------------+-----------+----------------+-----------------+
        | No         | Yes          | No        | none           | `None`          |
        +------------+--------------+-----------+----------------+-----------------+
        | Yes / No   | No           | No        | all            | CS (lowest)     |
        +------------+--------------+-----------+----------------+-----------------+
        | Yes / No   | No           | No        | none           | `None`          |
        +------------+--------------+-----------+----------------+-----------------+

        1. This case can not occur since it is not allowed to add a combination of
           coordinate systems with and without reference time to a
           `CoordinateSystemManager` without own reference time. See `add_cs`
           documentation for further details
        2. If neither the `CoordinateSystemManager` nor its attached coordinate systems
           have a reference time, the intention of passing a time and a reference time
           to the function is unclear. The caller might be unaware of the missing
           reference times. Therefore an exception is raised. If your intention is to
           add a reference time to the resulting coordinate system, you should call this
           function without a specified reference time and add it explicitly to the
           returned `~weldx.transformations.LocalCoordinateSystem`.


        **Information regarding the implementation:**

        It is important to mention that all coordinate systems that are involved in the
        transformation should be interpolated to a common time line before they are
        combined using the `~weldx.transformations.LocalCoordinateSystem` 's __add__
        and __sub__ functions.
        If this is not done before, serious interpolation errors for rotations can
        occur. The reason is, that those operators also perform time interpolations
        if the timestamps of 2 systems do not match. When chaining multiple
        transformations already interpolated values might be used to perform another
        interpolation.

        To see why this is problematic, consider a coordinate system which is statically
        attached to a not moving but rotating parent coordinate system. If it gets
        transformed to the reference systems of its parent, it will follow a circular
        trajectory around the parent system. For discrete timestamps, the trajectory is
        described by a set of corresponding coordinates. If we now interpolate again,
        the positions between those coordinates will be interpolated linearly, ignoring
        the originally circular trajectory. The dependency on the rotating parent system
        is not considered in further transformations.

        Additionally, if the transformed system is rotating itself, the transformation
        to the parent's reference system might cause the rotation angle between to
        time steps to exceed 180 degrees. Since the SLERP always takes the shortest
        angle between 2 ``keyframes``, further interpolations wrongly change the
        rotation order.

        """
        if reference_system_name is None:
            reference_system_name = self.get_parent_system_name(coordinate_system_name)
            if reference_system_name is None:
                raise ValueError(
                    f"The system {coordinate_system_name} has no parent system. "
                    f"You need to explicitly specify a reference system"
                )
        self._check_coordinate_system_exists(coordinate_system_name)
        self._check_coordinate_system_exists(reference_system_name)

        if coordinate_system_name == reference_system_name:
            return LocalCoordinateSystem()

        from networkx import shortest_path

        path = shortest_path(self.graph, coordinate_system_name, reference_system_name)
        path_edges = list(zip(path[:-1], path[1:]))

        if time is None:
            time_ref = None  # ignore passed reference time if no time was passed
            time = self.time_union(path_edges)

        elif isinstance(time, str):
            parent_name = self.get_parent_system_name(time)
            if parent_name is None:
                raise ValueError("The root system has no time dependency.")

            time = self.get_cs(time, parent_name).time
            if time is None:
                raise ValueError(f'The system "{time}" is not time dependent')
        elif not isinstance(time, (pd.DatetimeIndex, pint.Quantity)):
            time = pd.TimedeltaIndex(time)

        if time_ref is None:
            time_ref = self.reference_time
        else:
            time_ref = pd.Timestamp(time_ref)

        time_interp, time_ref_interp = build_time_index(time, time_ref)

        lcs_result = LocalCoordinateSystem()
        for edge in path_edges:
            invert = False
            lcs = self.graph.edges[edge]["lcs"]

            # lcs has an expression as coordinates
            if lcs is None:
                lcs = self.graph.edges[(edge[1], edge[0])]["lcs"]
                invert = True

            if lcs.is_time_dependent:
                if not lcs.has_reference_time and self.has_reference_time:
                    time_lcs = time_interp + (time_ref_interp - self.reference_time)
                    lcs = lcs.interp_time(time_lcs)
                    lcs.reset_reference_time(self.reference_time)
                    lcs.reset_reference_time(time_ref_interp)
                else:
                    lcs = lcs.interp_time(time_interp, time_ref_interp)

            if invert:
                if isinstance(lcs.coordinates, TimeSeries):
                    raise Exception(
                        "The chosen transformation is time dependent, but no time is "
                        "given. This is usually the case if the time dependencies are "
                        "only described by mathematical expressions. Provide the "
                        "desired time using the corresponding parameter to solve this "
                        "issue."
                    )
                lcs = lcs.invert()
            if len(path_edges) == 1:
                return lcs
            lcs_result += lcs
        return lcs_result

    def get_parent_system_name(self, coordinate_system_name) -> Union[str, None]:
        """Get the name of a coordinate systems parent system.

        The parent is the next system on the path towards the root node.

        Parameters
        ----------
        coordinate_system_name :
            Name of the coordinate system

        Returns
        -------
        str
            Name of the parent system
        None
            If the coordinate system has no parent (root system)

        """
        if coordinate_system_name == self._root_system_name:
            return None

        from networkx import shortest_path

        self._check_coordinate_system_exists(coordinate_system_name)
        path = shortest_path(self.graph, coordinate_system_name, self._root_system_name)

        return path[1]

    @property
    def subsystems(self) -> List["CoordinateSystemManager"]:
        """Extract all subsystems from the CoordinateSystemManager.

        Returns
        -------
        List :
            List containing all the subsystems.

        """
        ext_sub_system_data_dict = self._extended_sub_system_data

        sub_system_list = []
        for sub_system_name, ext_sub_system_data in ext_sub_system_data_dict.items():
            members = self._get_sub_system_members(
                ext_sub_system_data, ext_sub_system_data_dict
            )

            csm_sub = CoordinateSystemManager._from_subsystem_graph(
                ext_sub_system_data["root"],
                sub_system_name,
                time_ref=ext_sub_system_data["time_ref"],
                graph=self._graph.subgraph(members).copy(),
                subsystems=ext_sub_system_data["sub system data"],
            )
            sub_system_list.append(csm_sub)

        return sub_system_list

    def has_coordinate_system(self, coordinate_system_name: str) -> bool:
        """Return `True` if a coordinate system with specified name already exists.

        Parameters
        ----------
        coordinate_system_name :
            Name of the coordinate system, that should be checked.

        Returns
        -------
        bool
            `True` or `False`

        """
        return coordinate_system_name in self._graph.nodes

    def has_data(self, coordinate_system_name: str, data_name: str) -> bool:
        """Return `True` if the desired coordinate system owns the specified data.

        Parameters
        ----------
        coordinate_system_name :
            Name of the coordinate system
        data_name :
            Name of the data

        Returns
        -------
        bool
            `True` or `False`

        """
        return data_name in self._graph.nodes[coordinate_system_name]["data"]

    def interp_time(
        self,
        time: types_time_and_lcs,
        time_ref: pd.Timestamp = None,
        affected_coordinate_systems: Union[str, List[str], None] = None,
        in_place: bool = False,
    ) -> "CoordinateSystemManager":
        """Interpolates the coordinate systems in time.

        If no list of affected coordinate systems is provided, all systems will be
        interpolated to the same timeline.

        Parameters
        ----------
        time :
            The target time for the interpolation. In addition to the supported
            time formats, the function also accepts a LocalCoordinateSystem as
            ``time`` source object.
        time_ref :
            A reference timestamp that can be provided if the ``time`` parameter is a
            `~pandas.TimedeltaIndex`.
        affected_coordinate_systems :
            A single coordinate system name or a list of coordinate system names that
            should be interpolated in time. Only transformations towards the systems
            root node are affected.
        in_place :
            If `True` the interpolation is performed in place, otherwise a
            new instance is returned. (Default value = False)

        Returns
        -------
        CoordinateSystemManager
            Coordinate system manager with interpolated data

        """
        if in_place:
            if affected_coordinate_systems is not None:
                if isinstance(affected_coordinate_systems, str):
                    affected_coordinate_systems = [affected_coordinate_systems]

                affected_edges = []
                for cs in affected_coordinate_systems:
                    ps = self.get_parent_system_name(cs)
                    affected_edges.append((cs, ps))
                    affected_edges.append((ps, cs))
            else:
                affected_edges = self._graph.edges

            for edge in affected_edges:
                if self._graph.edges[edge]["defined"]:
                    lcs = self._graph.edges[edge]["lcs"]
                    # this prevents failures when calling lcs.interp_time with reference
                    # times or DatetimeIndex.
                    if lcs.reference_time is None and self._reference_time is not None:
                        lcs.reset_reference_time(self._reference_time)
                    self._graph.edges[edge]["lcs"] = lcs.interp_time(time, time_ref)

            for edge in affected_edges:
                if not self._graph.edges[edge]["defined"]:
                    self._graph.edges[edge]["lcs"] = self._graph.edges[
                        (edge[1], edge[0])
                    ]["lcs"].invert()
            return self

        return deepcopy(self).interp_time(
            time, time_ref, affected_coordinate_systems, in_place=True
        )

    def is_neighbor_of(
        self, coordinate_system_name_0: str, coordinate_system_name_1: str
    ) -> bool:
        """Get a boolean result, specifying if 2 coordinate systems are neighbors.

        Parameters
        ----------
        coordinate_system_name_0 :
            Name of the first coordinate system
        coordinate_system_name_1 :
            Name of the second coordinate system

        """
        self._check_coordinate_system_exists(coordinate_system_name_0)
        self._check_coordinate_system_exists(coordinate_system_name_1)

        return coordinate_system_name_1 in self.neighbors(coordinate_system_name_0)

    def merge(self, other: "CoordinateSystemManager"):
        """Merge another coordinate system managers into the current instance.

        Both `CoordinateSystemManager` need to have exactly one common coordinate
        system. They are merged at this node. Internally, information is kept
        to undo the merge process.

        Parameters
        ----------
        other:
            `CoordinateSystemManager` instance that should be merged into the current
            instance.

        """
        if other._number_of_time_dependent_lcs > 0 and (
            (not self.uses_absolute_times and other.uses_absolute_times)
            or (
                (self.uses_absolute_times and not self.has_reference_time)
                and not other.uses_absolute_times
            )
            or (
                (self.has_reference_time and other.uses_absolute_times)
                and (self.reference_time != other.reference_time)
            )
        ):
            raise Exception(
                "You can only merge subsystems with time dependent coordinate systems "
                "if the reference times of both `CoordinateSystemManager` instances "
                "are identical."
            )

        intersection = list(
            set(self.coordinate_system_names) & set(other.coordinate_system_names)
        )

        if len(intersection) != 1:
            raise ValueError(
                "Both instances must have exactly one common coordinate system. "
                f"Found the following common systems: {intersection}"
            )

        from networkx import compose

        self._graph = compose(self._graph, other.graph)

        subsystem_data = {
            "common node": intersection[0],
            "root": other.root_system_name,
            "time_ref": other.reference_time,
            "neighbors": other.neighbors(intersection[0]),
            "original members": other.coordinate_system_names,
            "sub system data": other.sub_system_data,
        }
        self._sub_system_data_dict[other.name] = subsystem_data

    def neighbors(self, coordinate_system_name: str) -> List:
        """Get a list of neighbors of a certain coordinate system.

        Parameters
        ----------
        coordinate_system_name :
            Name of the coordinate system

        Returns
        -------
        list
            List of neighbors

        """
        self._check_coordinate_system_exists(coordinate_system_name)
        return list(self._graph.neighbors(coordinate_system_name))

    def number_of_neighbors(self, coordinate_system_name) -> int:
        """Get the number of neighbors  of a certain coordinate system.

        Parameters
        ----------
        coordinate_system_name :
            Name of the coordinate system

        Returns
        -------
        int
            Number of neighbors

        """
        return len(self.neighbors(coordinate_system_name))

    def _get_tree_positions_for_plot(self):
        """Create the position data for the plot function."""
        pos = {}
        lcs_names = [self.root_system_name]
        meta_data = {self._root_system_name: {"position": (1, 0), "boundaries": [0, 2]}}
        level = 1
        while lcs_names:
            lcs_names_next = []
            for lcs_name in lcs_names:
                children_names = self.get_child_system_names(lcs_name)
                num_children = len(children_names)
                if num_children == 0:
                    continue

                bound = meta_data[lcs_name]["boundaries"]
                delta = (bound[1] - bound[0]) / num_children

                for i, child_name in enumerate(children_names):
                    pos_child = [bound[0] + (i + 0.5) * delta, -level]
                    bound_child = [bound[0] + i * delta, bound[0] + (i + 1) * delta]
                    meta_data[child_name] = {
                        "position": pos_child,
                        "boundaries": bound_child,
                    }
                lcs_names_next += children_names

            level += 1
            lcs_names = lcs_names_next

        for child, data in meta_data.items():
            pos[child] = data["position"]
        return pos

    def plot_graph(self, ax=None):
        """Plot the graph of the coordinate system manager.

        Time dependent (orange) and static (black) edges will be rendered differently.

        Parameters
        ----------
        ax :
            Matplotlib axes object that should be drawn to. If None is provided, this
            function will create one.

        Returns
        -------
        matplotlib.axes.Axes :
            The matplotlib axes object the graph has been drawn to

        """
        if ax is None:
            from matplotlib import pylab as plt

            _, ax = plt.subplots()
        color_map = []
        pos = self._get_tree_positions_for_plot()

        # only plot inverted directional arrows
        all_edges = [
            edge for edge in self._graph.edges if not self._graph.edges[edge]["defined"]
        ]

        def _is_edge_time_dependent(edge):
            lcs = self._graph.edges[edge]["lcs"]
            # inverse lcs contains a TimeSeries
            if lcs is None:
                return True
            return lcs.is_time_dependent

        # separate time dependent and static edges
        tdp_edges = [edge for edge in all_edges if _is_edge_time_dependent(edge)]
        stc_edges = [edge for edge in all_edges if edge not in tdp_edges]

        from networkx import draw, draw_networkx_edges

        draw(
            self._graph,
            pos,
            ax,
            with_labels=True,
            font_weight="bold",
            node_color=color_map,
            edgelist=stc_edges,
        )
        draw_networkx_edges(
            self._graph, pos, edgelist=tdp_edges, ax=ax, edge_color=(0.9, 0.6, 0)
        )

        return ax

    def plot(
        self,
        backend: str = "mpl",
        axes: matplotlib.axes.Axes = None,
        reference_system: str = None,
        coordinate_systems: List[str] = None,
        data_sets: List[str] = None,
        colors: Dict[str, int] = None,
        title: str = None,
        limits: List[Tuple[float, float]] = None,
        time: types_time_and_lcs = None,
        time_ref: pd.Timestamp = None,
        axes_equal: bool = False,
        scale_vectors: Union[float, List, np.ndarray] = None,
        show_data_labels: bool = True,
        show_labels: bool = True,
        show_origins: bool = True,
        show_traces: bool = True,
        show_vectors: bool = True,
        show_wireframe: bool = False,
    ):
        """Plot the coordinate systems of the coordinate system manager.

        Parameters
        ----------
        backend :
            Select the rendering backend of the plot. The options are:

            - ``k3d`` to get an interactive plot using `k3d <https://k3d-jupyter.org/>`_
            - ``mpl`` for static plots using `matplotlib <https://matplotlib.org/>`_

            Note that k3d only works inside jupyter notebooks
        axes : matplotlib.axes.Axes
            (matplotlib only) The target axes object that should be drawn to. If `None`
            is provided, a new one will be created.
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
        title :
            The title of the plot
        limits :
            The coordinate limits of the plot.
        time :
            The time steps that should be plotted
        time_ref :
            A reference timestamp that can be provided if the ``time`` parameter is a
            `pandas.TimedeltaIndex`
        axes_equal :
            (matplotlib only) If `True`, all axes are adjusted to cover an equally large
             range of value. That doesn't mean, that the limits are identical
        scale_vectors :
            (matplotlib only) A scaling factor or array to adjust the length of the
            coordinate system vectors
        show_data_labels :
            (k3d only) If `True`, plotted data sets get labels with their names attached
            to them
        show_labels :
            (k3d only) If `True`, plotted coordinate systems get labels with their names
            attached to them
        show_origins :
            If `True`, the origins of the coordinate system are visualized in the color
            assigned to the coordinate system.
        show_traces :
            If `True`, the trace of time dependent coordinate systems is plotted in the
            coordinate systems color.
        show_vectors :
            (matplotlib only) If `True`, the coordinate cross of time dependent
            coordinate systems is plotted.
        show_wireframe :
            (k3d only) If `True`, data sets that contain mesh data are rendered in
            wireframe mode. If `False`, the data

        """
        if backend not in ("mpl", "k3d"):
            raise ValueError(
                f"backend has to be one of ('mpl', 'k3d'), but was {backend}"
            )
        vis = None
        if backend == "k3d":
            from weldx.visualization import CoordinateSystemManagerVisualizerK3D

            vis = CoordinateSystemManagerVisualizerK3D(
                csm=self,
                reference_system=reference_system,
                coordinate_systems=coordinate_systems,
                data_sets=data_sets,
                colors=colors,
                limits=limits,
                show_data_labels=show_data_labels,
                show_labels=show_labels,
                show_origins=show_origins,
                show_traces=show_traces,
                show_vectors=show_vectors,
                show_wireframe=show_wireframe,
            )
        if backend == "mpl":
            from weldx.visualization import plot_coordinate_system_manager_matplotlib

            vis = plot_coordinate_system_manager_matplotlib(
                csm=self,
                axes=axes,
                reference_system=reference_system,
                coordinate_systems=coordinate_systems,
                data_sets=data_sets,
                colors=colors,
                time=time,
                time_ref=time_ref,
                title=title,
                limits=limits,
                set_axes_equal=axes_equal,
                scale_vectors=scale_vectors,
                show_origins=show_origins,
                show_trace=show_traces,
                show_vectors=show_vectors,
                show_wireframe=show_wireframe,
            )
        return vis

    def remove_subsystems(self):
        """Remove all subsystems from the coordinate system manager."""
        cs_delete = []
        for _, sub_system_data in self._sub_system_data_dict.items():
            for lcs in sub_system_data["neighbors"]:
                cs_delete += [lcs]

        self._sub_system_data_dict = {}
        for lcs in cs_delete:
            self.delete_cs(lcs, True)

    def time_union(
        self,
        list_of_edges: List = None,
    ) -> Union[None, pd.DatetimeIndex, pd.TimedeltaIndex]:
        """Get the time union of all or selected local coordinate systems.

         If neither the `CoordinateSystemManager` nor its attached
         `~weldx.transformations.LocalCoordinateSystem` instances possess a reference
         time, the function
         returns a `pandas.TimedeltaIndex`. Otherwise, a `pandas.DatetimeIndex` is
         returned. The following table gives an overview of all possible reference time
         combinations and the corresponding return type:


        +------------+------------------+-------------------------+
        | CSM        | LCS              | Return type             |
        | reference  | reference        |                         |
        | time       | times            |                         |
        +============+==================+=========================+
        | True       | all/mixed/none   | `pandas.DatetimeIndex`  |
        +------------+------------------+-------------------------+
        | False      | all              | `pandas.DatetimeIndex`  |
        +------------+------------------+-------------------------+
        | False      | none             | `pandas.TimedeltaIndex` |
        +------------+------------------+-------------------------+



        Parameters
        ----------
        list_of_edges :
            If not `None`, the union is only calculated from the specified edges.

        Returns
        -------
        pandas.DatetimeIndex or pandas.TimedeltaIndex
            Time union

        """
        if list_of_edges is None:
            lcs_list = self.lcs_time_dependent
        else:

            def _get_lcs(edge):
                lcs = self.graph.edges[edge]["lcs"]
                if lcs is not None:
                    return lcs
                return self.graph.edges[(edge[1], edge[0])]["lcs"]

            lcs_list = [_get_lcs(edge) for edge in list_of_edges]
        lcs_list = [lcs for lcs in lcs_list if lcs.time is not None]

        if not lcs_list:
            return None

        time_list = [util.to_pandas_time_index(lcs) for lcs in lcs_list]
        reference_time = self.reference_time
        if self.uses_absolute_times and not reference_time:
            reference_time = min(
                [
                    lcs.reference_time
                    for lcs in self.lcs_time_dependent
                    if lcs.reference_time
                ]
            )

        if reference_time:
            time_list = [
                t + reference_time if isinstance(t, pd.TimedeltaIndex) else t
                for t in time_list
            ]

        return util.get_time_union(time_list)

    def transform_data(
        self,
        data: types_coordinates,
        source_coordinate_system_name: str,
        target_coordinate_system_name: str,
    ):
        """Transform spatial data from one coordinate system to another.

        Parameters
        ----------
        data :
            Point cloud input as array-like with cartesian x,y,z-data stored in
            the last dimension. When using xarray objects, the vector dimension is
            expected to be named "c" and have coordinates "x","y","z"
        source_coordinate_system_name :
            Name of the coordinate system the data is
            defined in
        target_coordinate_system_name :
            Name of the coordinate system the data
            should be transformed to

        Returns
        -------
        numpy.ndarray
            Transformed data

        """
        if isinstance(data, SpatialData):
            return SpatialData(
                coordinates=self.transform_data(
                    data.coordinates,
                    source_coordinate_system_name,
                    target_coordinate_system_name,
                ),
                attributes=data.attributes,
                triangles=data.triangles,
            )
        if not isinstance(data, xr.DataArray):
            data = xr.DataArray(data, dims=["n", "c"], coords={"c": ["x", "y", "z"]})

        lcs = self.get_cs(source_coordinate_system_name, target_coordinate_system_name)
        mul = util.xr_matmul(
            lcs.orientation, data, dims_a=["c", "v"], dims_b=["c"], dims_out=["c"]
        )
        return mul + lcs.coordinates

    def unmerge(self) -> List["CoordinateSystemManager"]:
        """Undo previous merges and return a list of all previously merged instances.

        If additional coordinate systems were added after merging two instances, they
        won't be lost. Depending on their parent system, they will be kept in one of the
        returned sub-instances or the current instance. All new systems with the
        parent system being the shared node of two merged systems are kept in the
        current instance and won't be passed to the sub-instances.

        Returns
        -------
        List[`~weldx.transformations.CoordinateSystemManager`] :
            A list containing previously merged `CoordinateSystemManager` instances.

        """
        subsystems = self.subsystems
        self.remove_subsystems()

        return subsystems
