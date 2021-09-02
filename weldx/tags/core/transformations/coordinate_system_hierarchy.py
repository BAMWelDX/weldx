from weldx.asdf.types import WeldxConverter
from weldx.transformations import CoordinateSystemManager


class CoordinateSystemManagerConverter(WeldxConverter):
    """Serialization class for weldx.transformations.LocalCoordinateSystem"""

    name = "core/transformations/coordinate_system_hierarchy"
    version = "1.0.0"
    types = [CoordinateSystemManager]

    # todo
    #  - check possible bug: WeldxFile does not clear old yaml tree when writing. It
    #    just overwrites the bytes it needs. Write large tree first, then remove stuff
    #    and rewrite

    def to_yaml_tree(self, obj: CoordinateSystemManager, tag: str, ctx) -> dict:
        """Convert to python dict."""
        # work on subgraph view containing only original defined edges
        defined_edges = [e for e in obj.graph.edges if obj.graph.edges[e]["defined"]]
        graph = obj.graph.edge_subgraph(defined_edges)

        subsystem_info = [sub.to_yaml_tree() for sub in obj.subsystem_info]

        tree = dict(
            name=obj.name,
            reference_time=obj.reference_time,
            graph=graph,
            subsystems=subsystem_info,
        )
        return tree

    def from_yaml_tree(self, node: dict, tag: str, ctx):
        """Construct from tree."""
        return CoordinateSystemManager._from_yaml_tree(node)
