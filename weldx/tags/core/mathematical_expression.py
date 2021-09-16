import sympy

from weldx.asdf.types import WeldxConverter
from weldx.core import MathematicalExpression

__all__ = ["MathematicalExpression", "MathematicalExpressionConverter"]


class MathematicalExpressionConverter(WeldxConverter):
    """Serialization class for sympy style math expressions."""

    name = "core/mathematical_expression"
    version = "0.1.0"
    types = [MathematicalExpression]

    def to_yaml_tree(self, obj: MathematicalExpression, tag: str, ctx) -> dict:
        """Convert to python dict."""
        tree = {"expression": obj.expression.__str__(), "parameters": obj.parameters}
        return tree

    def from_yaml_tree(self, node: dict, tag: str, ctx):
        """Construct from tree."""
        obj = MathematicalExpression(
            sympy.sympify(node["expression"]), parameters=node["parameters"]
        )
        return obj
