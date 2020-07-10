import sympy

from weldx.asdf.types import WeldxType
from weldx.asdf.utils import drop_none_attr

__all__ = ["MathematicalExpression", "MathematicalExpressionType"]


class MathematicalExpression:
    def __init__(self, expression):
        self.expression = expression
        self.function = sympy.lambdify(
            self.expression.free_symbols, self.expression, "numpy"
        )
        self.parameters = {}

    def set_parameter(self, name, value):
        self.parameters[name] = value

    def get_variable_names(self):
        variable_names = []
        for var in self.expression.free_symbols:
            if var.__str__() not in self.parameters:
                variable_names.append(var.__str__())
        return variable_names

    def evaluate(self, **kwargs):
        inputs = {**kwargs, **self.parameters}
        return self.function(**inputs)


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
        obj = MathematicalExpression(sympy.sympify(tree["expression"]))
        obj.parameters = tree["parameters"]
        return obj
