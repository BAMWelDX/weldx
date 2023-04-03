import warnings

import sympy
from xarray import DataArray

from weldx.asdf.types import WeldxConverter
from weldx.constants import META_ATTR
from weldx.core import MathematicalExpression

__all__ = ["MathematicalExpression", "MathematicalExpressionConverter"]


class MathematicalExpressionConverter(WeldxConverter):
    """Serialization class for sympy style math expressions."""

    tags = ["asdf://weldx.bam.de/weldx/tags/core/mathematical_expression-0.1.*"]
    types = [MathematicalExpression]

    def to_yaml_tree(self, obj: MathematicalExpression, tag: str, ctx) -> dict:
        """Convert to python dict."""
        parameters = {}
        for k, v in obj.parameters.items():
            if isinstance(v, DataArray):
                if len(v.coords) > 0:
                    warnings.warn(
                        "Coordinates are dropped during serialization.", stacklevel=0
                    )
                dims = v.dims
                v = v.data
                setattr(v, META_ATTR, dict(dims=dims))
            parameters[k] = v

        return {"expression": obj.expression.__str__(), "parameters": parameters}

    def from_yaml_tree(self, node: dict, tag: str, ctx):
        """Construct from tree."""

        parameters = {}
        for k, v in node["parameters"].items():
            if hasattr(v, META_ATTR):
                dims = getattr(v, META_ATTR)["dims"]
                delattr(v, META_ATTR)
                v = (v, dims)
            parameters[k] = v

        return MathematicalExpression(
            sympy.sympify(node["expression"]), parameters=parameters
        )
