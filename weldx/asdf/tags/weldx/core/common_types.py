from dataclasses import dataclass
from typing import List

import numpy as np
import pint
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_timedelta64_dtype as is_timedelta

from weldx.asdf.types import WeldxType
from weldx.constants import WELDX_QUANTITY as Q_


@dataclass
class Dimension:
    """
    Stores data of a dimension.
    """

    name: str
    length: int


class DimensionTypeASDF(WeldxType):
    """Serialization class for the 'Dimension' type"""

    name = "core/dimension"
    version = "1.0.0"
    types = [Dimension]
    requires = ["weldx"]
    handle_dynamic_subclasses = True

    @classmethod
    def to_tree(cls, node: Dimension, ctx):
        """
        Convert an instance of the 'Dimension' type into YAML representations.

        Parameters
        ----------
        node :
            Instance of the 'Dimension' type to be serialized.

        ctx :
            An instance of the 'AsdfFile' object that is being written out.

        Returns
        -------
            A basic YAML type ('dict', 'list', 'str', 'int', 'float', or
            'complex') representing the properties of the 'Dimension' type to be
            serialized.

        """
        tree = {"name": node.name, "length": node.length}

        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        """
        Converts basic types representing YAML trees into custom types.

        Parameters
        ----------
        tree :
            An instance of a basic Python type (possibly nested) that
            corresponds to a YAML subtree.
        ctx :
            An instance of the 'AsdfFile' object that is being constructed.

        Returns
        -------
        Dimension :
            An instance of the 'Dimension' type.

        """
        return Dimension(tree["name"], tree["length"])


@dataclass
class Variable:
    """Represents an n-dimensional piece of data."""

    name: str
    dimensions: List
    data: np.ndarray


class VariableTypeASDF(WeldxType):
    """Serialization class for a Variable"""

    name = "core/variable"
    version = "1.0.0"
    types = [Variable]
    requires = ["weldx"]
    handle_dynamic_subclasses = True

    @staticmethod
    def convert_time_dtypes(data: np.ndarray):
        """
        Convert time format data types to a corresponding numeric data type.

        If the data's type isn't a time format, the function returns the unmodified
        data.

        Parameters
        ----------
        data :
            Data that should be converted.

        Returns
        -------
        np.ndarray :
            Unmodified or converted data.

        """
        if is_datetime(data.dtype) or is_timedelta(data.dtype):
            return data.astype(np.int64)
        return data

    @classmethod
    def to_tree(cls, node: Variable, ctx):
        """
        Convert an instance of the 'Variable' type into YAML representations.

        Parameters
        ----------
        node :
            Instance of the 'Variable' type to be serialized.

        ctx :
            An instance of the 'AsdfFile' object that is being written out.

        Returns
        -------
            A basic YAML type ('dict', 'list', 'str', 'int', 'float', or
            'complex') representing the properties of the 'Variable' type to be
            serialized.

        """

        if isinstance(node.data, pint.Quantity):
            unit = str(node.data.units)
            data = node.data.magnitude
        else:
            unit = None
            data = node.data
        dtype = node.data.dtype.str
        data = cls.convert_time_dtypes(data=data)
        tree = {
            "name": node.name,
            "dimensions": node.dimensions,
            "dtype": dtype,
            "data": data,
        }
        if unit:
            tree["unit"] = unit

        return tree

    @classmethod
    def from_tree(cls, tree, ctx):
        """
        Converts basic types representing YAML trees into custom types.

        Parameters
        ----------
        tree :
            An instance of a basic Python type (possibly nested) that
            corresponds to a YAML subtree.
        ctx :
            An instance of the 'AsdfFile' object that is being constructed.

        Returns
        -------
        Variable :
            An instance of the 'Variable' type.

        """
        dtype = np.dtype(tree["dtype"])
        if "unit" in tree:  # convert to pint.Quantity
            data = Q_(tree["data"].astype(dtype), tree["unit"])
        else:
            data = tree["data"].astype(dtype)

        return Variable(tree["name"], tree["dimensions"], data)
