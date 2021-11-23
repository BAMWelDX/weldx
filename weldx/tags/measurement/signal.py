from weldx.asdf.util import dataclass_serialization_class
from weldx.measurement import Signal

__all__ = ["Signal", "SignalConverter"]


def _from_yaml_tree_mod(tree: dict):
    if "data" not in tree:
        tree["data"] = None
    return tree


SignalConverter = dataclass_serialization_class(
    class_type=Signal,
    class_name="measurement/signal",
    version="0.1.0",
    from_yaml_tree_mod=_from_yaml_tree_mod,
)
