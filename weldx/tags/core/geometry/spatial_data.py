from copy import deepcopy

import numpy as np

from weldx.asdf.util import dataclass_serialization_class
from weldx.geometry import SpatialData


def _to_yaml_tree_mod(tree):
    tree = deepcopy(tree)
    tree["coordinates"] = tree["coordinates"].data
    return tree


def _from_yaml_tree_mod(tree):
    if "coordinates" in tree:
        tree["coordinates"] = np.asarray(tree["coordinates"])
    return tree


SpatialDataConverter = dataclass_serialization_class(
    class_type=SpatialData,
    class_name="core/geometry/spatial_data",
    version="0.1.0",
    to_yaml_tree_mod=_to_yaml_tree_mod,
    from_yaml_tree_mod=_from_yaml_tree_mod,
)
