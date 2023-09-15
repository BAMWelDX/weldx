from __future__ import annotations

import dataclasses
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pint
from asdf.tagged import TaggedDict
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_timedelta64_dtype as is_timedelta

from weldx.asdf.types import WeldxConverter
from weldx.asdf.util import _get_instance_shape, dataclass_serialization_class
from weldx.constants import Q_


# Dimension ----------------------------------------------------------------------------
@dataclass
class Dimension:
    """
    Stores data of a dimension.
    """

    name: str
    length: int


DimensionConverter = dataclass_serialization_class(
    class_type=Dimension, class_name="core/dimension", version="0.1.0"
)


# Variable -----------------------------------------------------------------------------
@dataclass
class Variable:
    """Represents an n-dimensional piece of data."""

    name: str
    dimensions: list
    data: np.ndarray
    attrs: Mapping[Hashable, Any] = dataclasses.field(default_factory=dict)


class VariableConverter(WeldxConverter):
    """Serialization class for a Variable"""

    tags = [
        "asdf://weldx.bam.de/weldx/tags/core/variable-0.1.*",
    ]
    types = [Variable]

    def select_tag(self, obj, tags, ctx):
        """Set highest available weldx tag for deserialization."""
        return sorted(tags)[-1]

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

    def to_yaml_tree(self, obj: Variable, tag: str, ctx) -> dict:
        """Convert to python dict."""
        if isinstance(obj.data, pint.Quantity):
            unit = obj.data.units
            data = obj.data.magnitude
        else:
            unit = None
            data = obj.data
        dtype = obj.data.dtype.str
        data = self.convert_time_dtypes(data=data)
        if not data.shape:  # scalar
            data = data.item()
        tree = {
            "name": obj.name,
            "dimensions": obj.dimensions,
            "dtype": dtype,
            "data": data,
            "attrs": obj.attrs if obj.attrs else None,
        }
        if unit:
            tree["units"] = unit

        return tree

    def from_yaml_tree(self, node: dict, tag: str, ctx):
        """Construct from tree."""
        dtype = np.dtype(node["dtype"])
        # TODO: it would be ideal, if asdf would handle time types natively.
        if dtype.char in ("M", "m"):  # handle np.timedelta64 and np.datetime64
            data = np.array(node["data"], dtype=dtype)
            # assert data.base is tree["data"]
        else:
            data = node["data"]  # let asdf handle np arrays with its own wrapper.

        if "units" in node:  # convert to pint.Quantity
            data = Q_(data, node["units"])

        attrs = node.get("attrs", None)

        return Variable(node["name"], node["dimensions"], data, attrs)

    @staticmethod
    def shape_from_tagged(node: TaggedDict) -> list[int]:
        """Calculate the shape from static tagged tree instance."""
        return _get_instance_shape(node["data"])
