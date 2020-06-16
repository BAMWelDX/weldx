from dataclasses import dataclass
from typing import List  # noqa: F401

from weldx.asdf.types import WeldxType
from weldx.asdf.utils import drop_none_attr

__all__ = ["Sensor", "SensorType"]


@dataclass
class Sensor:
    """<TODO CLASS DOCSTRING>"""

    data: str


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
