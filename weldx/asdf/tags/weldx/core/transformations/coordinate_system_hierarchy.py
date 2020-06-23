from copy import deepcopy
from dataclasses import dataclass, field
from weldx.asdf.types import WeldxType
from weldx.transformations import CoordinateSystemManager, LocalCoordinateSystem


@dataclass
class CoordinateSystemData:
    """<TODO CLASS DOCSTRING>"""

    name: str
    reference_system: str
    data: LocalCoordinateSystem


class CoordinateSystemDataASDF(WeldxType):
    """Serialization class for weldx.transformations.CoordinateSystemData"""

    name = "core/transformations/coordinate_system_data"
    version = "1.0.0"
    types = [CoordinateSystemData]
    requires = ["weldx"]
    handle_dynamic_subclasses = True
    validators = {}

    @classmethod
    def to_tree(cls, node: CoordinateSystemData, ctx):
        """
        Convert a 'CoordinateSystemData' instance into YAML representations.

        Parameters
        ----------
        node :
            Instance of the 'CoordinateSystemData' type to be serialized.

        ctx :
            An instance of the 'AsdfFile' object that is being written out.

        Returns
        -------
            A basic YAML type ('dict', 'list', 'str', 'int', 'float', or
            'complex') representing the properties of the 'CoordinateSystemData'
            type to be serialized.

        """
        tree = {"name": node.name, "reference_system": node.reference_system,
                "data": node.data}
        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        """
        Converts basic types representing YAML trees into a 'CoordinateSystemData'.

        Parameters
        ----------
        tree :
            An instance of a basic Python type (possibly nested) that
            corresponds to a YAML subtree.
        ctx :
            An instance of the 'AsdfFile' object that is being constructed.

        Returns
        -------
        CoordinateSystemData :
            An instance of the 'CoordinateSystemData' type.

        """
        return CoordinateSystemData(name=tree["name"],
                                    reference_system=tree["reference_system"],
                                    data=tree["data"])


class LocalCoordinateSystemASDF(WeldxType):
    """Serialization class for weldx.transformations.LocalCoordinateSystem"""

    name = "core/transformations/coordinate_system_hierarchy"
    version = "1.0.0"
    types = [CoordinateSystemManager]
    requires = ["weldx"]
    handle_dynamic_subclasses = True
    validators = {}

    @classmethod
    def to_tree(cls, node: CoordinateSystemManager, ctx):
        """
        Convert a 'CoordinateSystemManager' instance into YAML representations.

        Parameters
        ----------
        node :
            Instance of the 'CoordinateSystemManager' type to be serialized.

        ctx :
            An instance of the 'AsdfFile' object that is being written out.

        Returns
        -------
            A basic YAML type ('dict', 'list', 'str', 'int', 'float', or
            'complex') representing the properties of the 'CoordinateSystemManager'
            type to be serialized.

        """
        graph = deepcopy(node.graph)  # TODO: Check if deepcopy is necessary

        # remove automatically computed edges (inverted directions)
        remove_edges = []
        for edge in graph.edges:
            if not graph.edges[edge]["defined"]:
                remove_edges.append(edge)
        graph.remove_edges_from(remove_edges)

        graph_edges = [*graph.edges]

        # find root coordinate system
        root_system_name = None
        for graph_node in graph.nodes:
            if graph.out_degree(graph_node) == 0:
                root_system_name = graph_node
                break

        coordinate_system_data = []

        for name, reference_system in graph.edges:
            cs_data = CoordinateSystemData(name, reference_system,
                                           node.get_local_coordinate_system(
                                               name,
                                               reference_system))
            coordinate_system_data.append(cs_data)

        tree = {"root_system_name": root_system_name,
                "coordinate_systems": coordinate_system_data}
        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        """
        Converts basic types representing YAML trees into a 'CoordinateSystemManager'.

        Parameters
        ----------
        tree :
            An instance of a basic Python type (possibly nested) that
            corresponds to a YAML subtree.
        ctx :
            An instance of the 'AsdfFile' object that is being constructed.

        Returns
        -------
        CoordinateSystemManager :
            An instance of the 'CoordinateSystemManager' type.

        """
        csm = CoordinateSystemManager(
            root_coordinate_system_name=tree["root_system_name"])

        coordinate_systems = tree["coordinate_systems"]

        all_systems_included = False
        while not all_systems_included:
            all_systems_included = True
            for cs_data in coordinate_systems:
                if not csm.has_coordinate_system(cs_data.name):
                    if csm.has_coordinate_system(cs_data.reference_system):
                        csm.add_coordinate_system(cs_data.name,
                                                  cs_data.reference_system,
                                                  cs_data.data)
                    else:
                        all_systems_included = False
        return csm
