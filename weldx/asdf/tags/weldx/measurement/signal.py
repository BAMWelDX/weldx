from weldx.asdf.util import dataclass_serialization_class
from weldx.measurement import Signal

__all__ = ["Signal", "SignalType"]


def _from_tree_mod(tree):
    if "data" not in tree:
        tree["data"] = None
    return tree


SignalType = dataclass_serialization_class(
    class_type=Signal,
    class_name="measurement/signal",
    version="1.0.0",
    from_tree_mod=_from_tree_mod,
)
