from copy import deepcopy

import numpy as np

from weldx.asdf.util import dataclass_serialization_class
from weldx.geometry import SpatialData


def _to_tree_mod(tree):
    tree = deepcopy(tree)
    tree["coordinates"] = tree["coordinates"].data
    return tree


def from_tree_mod(tree):
    if "coordinates" in tree:
        tree["coordinates"] = np.asarray(tree["coordinates"])
    return tree


SpatialDataTypeASDF = dataclass_serialization_class(
    class_type=SpatialData,
    class_name="core/geometry/spatial_data",
    version="1.0.0",
    to_tree_mod=_to_tree_mod,
    from_tree_mod=from_tree_mod,
)


# @classmethod
# def to_tree(cls, node: SpatialData, ctx):
#    tree = deepcopy(node.__dict__)
#    tree["coordinates"] = tree["coordinates"].data
#    return tree
