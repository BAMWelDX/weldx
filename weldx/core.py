"""Collection of common classes and functions."""


import sympy


class MathematicalExpression:
    """Mathematical expression using sympy syntax."""

    def __init__(self, expression: sympy.core.basic.Basic, parameters=None):
        """Initialize a MathematicalExpression from sympy objects.

        Parameters
        ----------
        expression
            sympy object that can be turned into an expression.
            E.g. any valid combination of previously defined sympy.symbols.
        """
        if isinstance(expression, str):
            expression = sympy.sympify(expression)
        self.expression = expression
        self.function = sympy.lambdify(
            self.expression.free_symbols, self.expression, "numpy"
        )
        self.parameters = {}
        if parameters is not None:
            if not isinstance(parameters, dict):
                raise ValueError('"parameters" must be dictionary')
            variable_names = self.get_variable_names()
            for key in parameters:
                if key not in variable_names:
                    raise ValueError(
                        f'The expression does not have a parameter "{key}"'
                    )
            self.parameters = parameters

    def set_parameter(self, name, value):
        """Define an expression parameter as constant value.

        Parameters
        ----------
        name
            Name of the parameter used in the expression.
        value
            Parameter value. This can be number, array or pint.Quantity

        """
        self.parameters[name] = value

    def num_parameters(self):
        """
        Get the expressions number of parameters.

        Returns
        -------
        int:
            Number of parameters.
        """
        return len(self.parameters)

    def num_variables(self):
        """
        Get the expressions number of free variables.

        Returns
        -------
        int:
            Number of free variables.
        """
        return len(self.expression.free_symbols) - len(self.parameters)

    def get_variable_names(self):
        """Get a list of all expression variables.

        Returns
        -------
        List

        """
        variable_names = []
        for var in self.expression.free_symbols:
            if var.__str__() not in self.parameters:
                variable_names.append(var.__str__())
        return variable_names

    def evaluate(self, **kwargs):
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
        inputs = {**kwargs, **self.parameters}
        return self.function(**inputs)
