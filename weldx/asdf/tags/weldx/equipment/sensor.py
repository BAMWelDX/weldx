from weldx.asdf.types import WeldxType
from weldx.asdf.utils import drop_none_attr
from weldx.measurement import Sensor

__all__ = ["Sensor", "SensorType"]


class SensorType(WeldxType):
    """<TODO ASDF TYPE DOCSTRING>"""

    name = "equipment/sensor"
    version = "1.0.0"
    types = [Sensor]
    requires = ["weldx"]
    handle_dynamic_subclasses = True

    @classmethod
    def to_tree(cls, node: Sensor, ctx):
        """convert to tagged tree and remove all None entries from node dictionary"""
        tree = drop_none_attr(node)
        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        obj = Sensor(**tree)
        return obj
