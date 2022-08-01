from weldx.asdf.types import WeldxConverter
from weldx.measurement import MeasurementChain

__all__ = [
    "MeasurementChain",
    "MeasurementChainConverter",
]


class MeasurementChainConverter(WeldxConverter):
    """Serialization class for measurement chains"""

    tags = ["asdf://weldx.bam.de/weldx/tags/measurement/measurement_chain-0.1.*"]
    types = [MeasurementChain]

    def to_yaml_tree(self, obj: MeasurementChain, tag: str, ctx) -> dict:
        """Convert to python dict."""
        return obj.to_dict()

    def from_yaml_tree(self, node: dict, tag: str, ctx):
        """Reconstruct from tree."""
        return MeasurementChain.from_dict(node)
