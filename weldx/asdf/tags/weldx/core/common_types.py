import dataclasses
from dataclasses import dataclass
from typing import Any, Hashable, List, Mapping

import numpy as np
import pint
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_timedelta64_dtype as is_timedelta

from weldx.asdf.types import WeldxType
from weldx.asdf.util import dataclass_serialization_class
from weldx.constants import WELDX_QUANTITY as Q_


# Dimension ----------------------------------------------------------------------------
@dataclass
class Dimension:
    """
    Stores data of a dimension.
    """

    name: str
    length: int


DimensionTypeASDF = dataclass_serialization_class(
    class_type=Dimension, class_name="core/dimension", version="1.0.0"
)


# Variable -----------------------------------------------------------------------------
@dataclass
class Variable:
    """Represents an n-dimensional piece of data."""

    name: str
    dimensions: List
    data: np.ndarray
    attrs: Mapping[Hashable, Any] = dataclasses.field(default_factory=dict)


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
        if not data.shape:  # scalar
            data = data.item()
        tree = {
            "name": node.name,
            "dimensions": node.dimensions,
            "dtype": dtype,
            "data": data,
            "attrs": node.attrs if node.attrs else None,
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
        data = np.array(tree["data"], dtype=dtype)
        if "unit" in tree:  # convert to pint.Quantity
            data = Q_(data, tree["unit"])

        attrs = tree.get("attrs", None)

        return Variable(tree["name"], tree["dimensions"], data, attrs)
