from dataclasses import dataclass

from asdf import ValidationError

from weldx.asdf.types import WeldxType
from weldx.asdf.utils import drop_none_attr

__all__ = ["AnyOfClass", "AnyOfClassType"]


@dataclass
class AnyOfClass:
    name: str
    data: dict


class AnyOfClassType(WeldxType):
    name = "debug/anyof_class"
    version = "1.0.0"
    types = [AnyOfClass]

    @classmethod
    def to_tree(cls, node: AnyOfClass, ctx):
        """convert to tagged tree and remove all None entries from node dictionary"""
        tree = drop_none_attr(node)
        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        obj = AnyOfClass(**tree)
        return obj


def debug_validator(validator, debug_validator, instance, schema):
    """Yield Validation error if true."""
    if debug_validator:
        print(f"triggered validation on schema {schema['properties']}")
        # yield ValidationError(f"triggered validation on {instance}")


AnyOfClassType.validators = {
    "debug_validator": debug_validator,
}
