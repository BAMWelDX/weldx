import warnings

import sympy
from xarray import DataArray

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
        parameters = {}
        for k, v in obj.parameters.items():
            if isinstance(v, DataArray):
                if len(v.coords) > 0:
                    warnings.warn("Coordinates are dropped during serialization.")
                dims = v.dims
                v = v.data
                v.wx_metadata = dict(dims=dims)
            parameters[k] = v

        return {"expression": obj.expression.__str__(), "parameters": parameters}

    def from_yaml_tree(self, node: dict, tag: str, ctx):
        """Construct from tree."""

        parameters = {}
        for k, v in node["parameters"].items():
            if hasattr(v, "wx_metadata"):
                dims = v.wx_metadata["dims"]
                delattr(v, "wx_metadata")
                v = (v, dims)
            parameters[k] = v

        return MathematicalExpression(
            sympy.sympify(node["expression"]), parameters=parameters
        )
