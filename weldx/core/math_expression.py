"""Contains the MathematicalExpression class."""

from __future__ import annotations

from typing import Any, Union

import pint
import sympy
import xarray as xr

from weldx import Q_

ExpressionParameterTypes = Union[
    pint.Quantity, str, tuple[pint.Quantity, str], xr.DataArray
]

__all__ = ["MathematicalExpression", "ExpressionParameterTypes"]


class MathematicalExpression:
    """Mathematical expression using sympy syntax."""

    def __init__(
        self,
        expression: sympy.Expr | str,
        parameters: ExpressionParameterTypes = None,
    ):
        """Construct a MathematicalExpression.

        Parameters
        ----------
        expression
            A sympy expression or a string that can be evaluated as one.
        parameters :
            A dictionary containing constant values for variables of the
            expression.

        """
        if not isinstance(expression, sympy.Expr):
            expression = sympy.sympify(expression)
        if not isinstance(expression, sympy.Expr):
            raise TypeError("'expression' can't be converted to a sympy expression")
        self._expression: sympy.Expr = expression

        self.function = sympy.lambdify(
            tuple(self._expression.free_symbols), self._expression, ("numpy", "scipy")
        )

        self._parameters: dict[str, pint.Quantity | xr.DataArray] = {}
        if parameters is not None:
            self.set_parameters(parameters)

    def __repr__(self):
        """Give __repr__ output."""
        # todo: Add covered dimensions to parameters - coordinates as well?
        representation = (
            f"<MathematicalExpression>\n"
            f"Expression:\n\t{self._expression.__repr__()}\n"
            f"Parameters:\n"
        )
        if len(self._parameters) > 0:
            for parameter, value in self._parameters.items():
                if isinstance(value, xr.DataArray):
                    value = value.data
                representation += f"\t{parameter} = {value}\n"
        else:
            representation += "\tNone"
        return representation

    def __eq__(self, other):
        """Return the result of a structural equality comparison with another object.

        If the other object is not a 'MathematicalExpression' this function always
        returns 'False'.

        Parameters
        ----------
        other:
            Other object.

        Returns
        -------
        bool:
            'True' if the compared object is also a 'MathematicalExpression' and equal
            to this instance, 'False' otherwise

        """
        return self.equals(other, check_parameters=True, check_structural_equality=True)

    __hash__ = None

    def equals(
        self,
        other: Any,
        check_parameters: bool = True,
        check_structural_equality: bool = False,
    ):
        """Compare the instance with another object for equality and return the result.

        If the other object is not a MathematicalExpression this function always returns
        'False'. The function offers the choice to compare for structural or
        mathematical equality by setting the 'structural_expression_equality' parameter
        accordingly. Additionally, the comparison can be limited to the expression only,
        if 'check_parameters' is set to 'False'.

        Parameters
        ----------
        other:
            Arbitrary other object.
        check_parameters
            Set to 'True' if the parameters should be included during the comparison. If
            'False', only the expression is checked for equality.
        check_structural_equality:
            Set to 'True' if the expression should be checked for structural equality.
            Set to 'False' if mathematical equality is sufficient.

        Returns
        -------
        bool:
            'True' if both objects are equal, 'False' otherwise

        """
        if isinstance(other, MathematicalExpression):
            if check_structural_equality:
                equality = self.expression == other.expression
            else:
                from sympy import simplify

                equality = simplify(self.expression - other.expression) == 0

            if check_parameters:
                from weldx.util import compare_nested

                equality = equality and compare_nested(
                    self._parameters, other.parameters
                )
            return equality
        return False

    def set_parameter(self, name, value):
        """Define an expression parameter as constant value.

        Parameters
        ----------
        name
            Name of the parameter used in the expression.
        value
            Parameter value. This can be number, array or pint.Quantity

        """
        self.set_parameters({name: value})

    # todo: Use kwargs here???
    def set_parameters(self, params: ExpressionParameterTypes):
        """Set the expressions parameters.

        Parameters
        ----------
        params:
            Dictionary that contains the values for the specified parameters.

        """
        if not isinstance(params, dict):
            raise ValueError(f'"parameters" must be dictionary, got {type(params)}')

        variable_names = [str(v) for v in self._expression.free_symbols]

        for k, v in params.items():
            if k not in variable_names:
                raise ValueError(f'The expression does not have a parameter "{k}"')
            if isinstance(v, tuple):
                v = xr.DataArray(v[0], dims=v[1])
            if not isinstance(v, xr.DataArray):
                v = Q_(v)
            else:  # quantify as dimensionless if no unit provided
                if v.weldx.units is None:
                    v = v.pint.quantify("")
                v = v.pint.quantify()
            self._parameters[k] = v

    @property
    def num_parameters(self):
        """Get the expressions number of parameters.

        Returns
        -------
        int:
            Number of parameters.

        """
        return len(self._parameters)

    @property
    def num_variables(self):
        """Get the expressions number of free variables.

        Returns
        -------
        int:
            Number of free variables.

        """
        return len(self.expression.free_symbols) - len(self._parameters)

    @property
    def expression(self):
        """Return the internal sympy expression.

        Returns
        -------
        sympy.core.expr.Expr:
            Internal sympy expression

        """
        return self._expression

    @property
    def parameters(self) -> dict[str, pint.Quantity | xr.DataArray]:
        """Return the internal parameters dictionary.

        Returns
        -------
        Dict
            Internal parameters dictionary

        """
        return self._parameters

    def get_variable_names(self) -> list[str]:
        """Get a list of all expression variables.

        Returns
        -------
        List:
            List of all expression variables

        """
        return [
            str(var)
            for var in self._expression.free_symbols
            if str(var) not in self._parameters
        ]

    def evaluate(self, **kwargs) -> Any:
        """Evaluate the expression for specific variable values.

        Parameters
        ----------
        kwargs
            additional keyword arguments (variable assignment) to pass.

        Returns
        -------
        Any:
            Result of the evaluated function

        """
        intersection = set(kwargs).intersection(self._parameters)
        if len(intersection) > 0:
            raise ValueError(
                f"The variables {intersection} are already defined as parameters."
            )

        variables = {
            k: v if isinstance(v, xr.DataArray) else xr.DataArray(Q_(v))
            for k, v in kwargs.items()
        }

        parameters = {
            k: v if isinstance(v, xr.DataArray) else xr.DataArray(v)
            for k, v in self._parameters.items()
        }
        return self.function(**variables, **parameters)
