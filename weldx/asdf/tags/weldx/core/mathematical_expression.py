import sympy

from weldx.asdf.types import WeldxType
from weldx.core import MathematicalExpression

__all__ = ["MathematicalExpression", "MathematicalExpressionType"]


class MathematicalExpressionType(WeldxType):
    """Serialization class for sympy style math expressions."""

    name = "core/mathematical_expression"
    version = "1.0.0"
    types = [MathematicalExpression]
    requires = ["weldx"]
    handle_dynamic_subclasses = True

    @classmethod
    def to_tree(cls, node: MathematicalExpression, ctx):
        """convert to tagged tree and remove all None entries from node dictionary"""
        tree = {"expression": node.expression.__str__(), "parameters": node.parameters}
        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        obj = MathematicalExpression(
            sympy.sympify(tree["expression"]), parameters=tree["parameters"]
        )
        return obj
