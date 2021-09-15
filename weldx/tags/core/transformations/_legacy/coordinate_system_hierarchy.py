from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Tuple

from pandas import Timestamp

from weldx.asdf.types import WeldxConverter
from weldx.transformations import CoordinateSystemManager, LocalCoordinateSystem


@dataclass
class CoordinateTransformation:
    """Stores data of a coordinate transformation."""

    name: str
    reference_system: str
    transformation: LocalCoordinateSystem


class CoordinateTransformationConverter(WeldxConverter):
    """Legacy serialization class for CoordinateTransformation"""

    tags = [
        "tag:weldx.bam.de:weldx/core/transformations/coordinate_transformation-1.0.0"
    ]
    types = [CoordinateTransformation]

    def to_yaml_tree(self, obj: CoordinateTransformation, tag: str, ctx) -> dict:
        """Convert to python dict."""
        tree = {
            "name": obj.name,
            "reference_system": obj.reference_system,
            "transformation": obj.transformation,
        }
        return tree

    def from_yaml_tree(self, node: dict, tag: str, ctx):
        """Construct from tree."""
        return CoordinateTransformation(
            name=node["name"],
            reference_system=node["reference_system"],
            transformation=node["transformation"],
        )


@dataclass
class CoordinateSystemManagerSubsystem:
    """Helper class to collect all relevant data of a CSM subsystem."""

    name: str
    parent_system: str
    reference_time: Timestamp
    root_cs: str
    subsystems: List[str]
    members: List[str]

    def __post_init__(self):
        """Sort the string lists."""
        self.subsystems = sorted(self.subsystems)
        self.members = sorted(self.members)


class CoordinateSystemManagerSubsystemConverter(WeldxConverter):
    """Legacy serialization class for a CoordinateSystemManagerSubsystem instance"""

    tags = [
        "tag:weldx.bam.de:weldx/core/transformations/"
        "coordinate_system_hierarchy_subsystem-1.0.0"
    ]
    types = [CoordinateSystemManagerSubsystem]

    def to_yaml_tree(
        self, obj: CoordinateSystemManagerSubsystem, tag: str, ctx
    ) -> dict:
        """Convert to python dict."""
        tree = {
            "name": obj.name,
            "root_cs": obj.root_cs,
            "reference_time": obj.reference_time,
            "parent_system": obj.parent_system,
            "subsystem_names": obj.subsystems,
            "members": obj.members,
        }
        return tree

    def from_yaml_tree(self, node: dict, tag: str, ctx):
        """Construct from tree."""
        return node


class _CoordinateSystemManager:
    pass


class CoordinateSystemManagerConverter(WeldxConverter):
    """Legacy serialization class for weldx.transformations.CoordinateSystemManager"""

    tags = [
        "tag:weldx.bam.de:weldx/core/transformations/coordinate_system_hierarchy-1.0.0"
    ]
    types = [_CoordinateSystemManager]

    @classmethod
    def _extract_all_subsystems(
        cls, csm: CoordinateSystemManager, _recursive_call: bool = False
    ) -> List[Tuple[CoordinateSystemManager, str]]:
        """Return a list of all subsystems and their corresponding parent names.

        This function extracts all subsystems and nested subsystems of the passed
        coordinate system manager instance. Each subsystem is stored inside a tuple
        together with the parent systems' name.
        Parameters
        ----------
        csm:
            Coordinate system manager instance
        _recursive_call:
            Specifies if the current function call happened recursively

        Returns
        -------
        List[Tuple[CoordinateSystemManager, str]]:
            List of subsystems and their parent names

        """
        if not _recursive_call:
            csm = deepcopy(csm)

        subsystems = csm.unmerge()

        subsystems = [(subsystem, csm.name) for subsystem in subsystems]
        for subsystem, _ in subsystems:
            if subsystem.number_of_subsystems > 0:
                subsystems += cls._extract_all_subsystems(subsystem, True)

        return subsystems

    @classmethod
    def _extract_subsystem_data(
        cls, csm: CoordinateSystemManager
    ) -> List[CoordinateSystemManagerSubsystem]:
        """Get the subsystem data of a CoordinateSystemManager instance.

        Parameters
        ----------
        csm:
            CoordinateSystemManager instance.

        Returns
        -------
        List[CoordinateSystemManagerSubsystem]:
            Data of all subsystems

        """
        subsystems = cls._extract_all_subsystems(csm)
        subsystem_data = []
        for subsystem, parent in subsystems:
            child_systems = [
                child.name for child, parent in subsystems if parent == subsystem.name
            ]
            subsystem_data += [
                CoordinateSystemManagerSubsystem(
                    subsystem.name,
                    parent,
                    subsystem.reference_time,
                    subsystem.root_system_name,
                    child_systems,
                    subsystem.coordinate_system_names,
                )
            ]
        return subsystem_data

    @classmethod
    def _merge_subsystems(cls, tree, csm, subsystem_data_list):
        """Merge all subsystems into the CoordinateSystemManager instance.

        Parameters
        ----------
        tree:
            The dictionary representing the ASDF files YAML tree
        csm:
            CoordinateSystemManager instance
        subsystem_data_list:
            List containing all relevant subsystem data from the CoordinateSystemManager
            instance

        """
        subsystem_data_dict = {
            subsystem_data["name"]: subsystem_data
            for subsystem_data in subsystem_data_list
        }

        if subsystem_data_list:
            cls._recursively_merge_subsystems(
                csm, tree["subsystem_names"], subsystem_data_dict
            )

    @classmethod
    def _recursively_merge_subsystems(
        cls, csm, subsystem_names, subsystem_data_dict: Dict
    ):
        """Merge a list of subsystems into a CoordinateSystemManager instance.

        This function also considers nested subsystems using recursive function calls.

        Parameters
        ----------
        csm:
            CoordinateSystemManager instance
        subsystem_names:
            Names of all subsystems that should be added.
        subsystem_data_dict
            Dictionary containing the data of all subsystems from the
            CoordinateSystemManager instance

        """
        for subsystem_name in subsystem_names:
            subsystem_data = subsystem_data_dict[subsystem_name]
            if subsystem_data["subsystem_names"]:
                cls._recursively_merge_subsystems(
                    subsystem_data["csm"],
                    subsystem_data["subsystem_names"],
                    subsystem_data_dict,
                )
            csm.merge(subsystem_data["csm"])

    @classmethod
    def _add_coordinate_systems_to_subsystems(
        cls, tree, csm, subsystem_data_list: List[Dict]
    ):
        """Add all coordinate systems to the owning subsystem.

        Parameters
        ----------
        tree:
            The dictionary representing the ASDF files YAML tree
        csm:
            CoordinateSystemManager instance
        subsystem_data_list:
            A list containing all relevant subsystem data

        """
        main_system_lcs = []

        for lcs_data in tree["coordinate_systems"]:
            edge = [lcs_data.name, lcs_data.reference_system]
            is_subsystem_lcs = False
            for subsystem_data in subsystem_data_list:
                if set(edge).issubset(subsystem_data["members"]):
                    subsystem_data["lcs"] += [(*edge, lcs_data.transformation)]
                    is_subsystem_lcs = True
                    break
            if not is_subsystem_lcs:
                main_system_lcs += [(*edge, lcs_data.transformation)]

        # add coordinate systems to corresponding csm
        cls._add_coordinate_systems_to_manager(csm, main_system_lcs)
        for subsystem_data in subsystem_data_list:
            cls._add_coordinate_systems_to_manager(
                subsystem_data["csm"], subsystem_data["lcs"]
            )

    @classmethod
    def _add_coordinate_systems_to_manager(
        cls,
        csm: CoordinateSystemManager,
        lcs_data_list: List[Tuple[str, str, LocalCoordinateSystem]],
    ):
        """Add all coordinate systems to a CSM instance.

        Parameters
        ----------
        csm:
            CoordinateSystemManager instance.
        lcs_data_list:
            List containing all the necessary data of all coordinate systems that should
            be added.

        """
        # todo: ugly but does the job. check if this can be enhanced
        leaf_nodes = [csm.root_system_name]
        while lcs_data_list:
            leaf_nodes_next = []
            lcs_data_list_next = []
            for lcs_data in lcs_data_list:
                lcs_added = False
                for leaf_node in leaf_nodes:
                    if leaf_node in lcs_data[0:2]:
                        if leaf_node == lcs_data[0]:
                            csm.add_cs(lcs_data[1], leaf_node, lcs_data[2], False)
                            leaf_nodes_next += [lcs_data[1]]
                        else:
                            csm.add_cs(lcs_data[0], leaf_node, lcs_data[2], True)
                            leaf_nodes_next += [lcs_data[0]]
                        lcs_added = True
                        break
                if not lcs_added:
                    lcs_data_list_next += [lcs_data]

            leaf_nodes = leaf_nodes_next
            lcs_data_list = lcs_data_list_next

    def to_yaml_tree(self, obj: CoordinateSystemManager, tag: str, ctx) -> dict:
        """Convert to python dict."""

        raise Exception("This should not call 'to_yaml_tree'")

        # work on subgraph view containing only original defined edges
        defined_edges = [e for e in obj.graph.edges if obj.graph.edges[e]["defined"]]
        graph = obj.graph.edge_subgraph(defined_edges)

        coordinate_system_data = []
        for name, reference_system in graph.edges:
            coordinate_system_data += [
                CoordinateTransformation(
                    name,
                    reference_system,
                    graph.edges[(name, reference_system)]["lcs"],
                )
            ]

        subsystem_data = self._extract_subsystem_data(obj)
        subsystems = [
            subsystem.name
            for subsystem in subsystem_data
            if subsystem.parent_system == obj.name
        ]

        spatial_data = None

        if len(obj.data_names) > 0:
            spatial_data = []
            for cs in obj.graph.nodes:
                spatial_data += [
                    dict(name=k, coordinate_system=cs, data=v)
                    for k, v in obj.graph.nodes[cs]["data"].items()
                ]

        tree = {
            "name": obj.name,
            "reference_time": obj.reference_time,
            "subsystem_names": subsystems,
            "subsystems": subsystem_data,
            "root_system_name": obj.root_system_name,
            "coordinate_systems": coordinate_system_data,
            "spatial_data": spatial_data,
        }
        return tree

    def from_yaml_tree(self, node: dict, tag: str, ctx):
        """Construct from tree."""
        reference_time = None
        if "reference_time" in node:
            reference_time = node["reference_time"]
        csm = CoordinateSystemManager(
            node["root_system_name"], node["name"], time_ref=reference_time
        )

        subsystem_data_list = node["subsystems"]

        for subsystem_data in subsystem_data_list:
            subsystem_reference_time = None
            if "reference_time" in subsystem_data:
                subsystem_reference_time = subsystem_data["reference_time"]
            subsystem_data["csm"] = CoordinateSystemManager(
                subsystem_data["root_cs"],
                subsystem_data["name"],
                subsystem_reference_time,
            )
            subsystem_data["lcs"] = []

        self._add_coordinate_systems_to_subsystems(node, csm, subsystem_data_list)
        self._merge_subsystems(node, csm, subsystem_data_list)

        if (spatial_data := node.get("spatial_data")) is not None:
            for item in spatial_data:
                csm.assign_data(item["data"], item["name"], item["coordinate_system"])

        return csm
