from weldx.asdf.types import WeldxConverter
from weldx.transformations import CoordinateSystemManager


class CoordinateSystemManagerConverter(WeldxConverter):
    """Serialization class for weldx.transformations.CoordinateSystemManager"""

    tags = [
        "asdf://weldx.bam.de/weldx/tags/core/transformations/"
        "coordinate_system_hierarchy-0.1.*"
    ]
    types = [CoordinateSystemManager]

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
