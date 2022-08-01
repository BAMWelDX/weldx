from dataclasses import dataclass

import pint

from weldx.asdf.types import WeldxConverter
from weldx.asdf.util import dataclass_serialization_class
from weldx.core import GenericSeries
from weldx.util.xarray import _get_coordinate_quantities


@dataclass
class GenericSeriesFreeDimension:
    units: pint.Quantity
    symbol: "str" = None


GenericSeriesFreeDimensionConverter = dataclass_serialization_class(
    class_type=GenericSeriesFreeDimension,
    class_name="core/generic_series_free_dimension",
    version="0.1.0",
)


class GenericSeriesConverter(WeldxConverter):
    """Serialization class for weldx.core.GenericSeries"""

    tags = ["asdf://weldx.bam.de/weldx/tags/core/generic_series-0.1.*"]
    types = [GenericSeries]

    def to_yaml_tree(self, obj: GenericSeries, tag: str, ctx) -> dict:
        """Convert to python dict."""
        if obj.is_expression:
            dims = {}
            for var in obj.variable_names:
                dim_data = dict(units=obj.variable_units[var])
                dim = obj._symbol_dims.get(var)

                # check if name of expression variable and dimension are identical
                if dim is not None and dim != var:
                    dim_data["symbol"] = var
                else:
                    dim = var
                dims[dim] = GenericSeriesFreeDimension(**dim_data)

            return dict(
                expression=str(obj.data.expression),
                parameters=obj.data.parameters,
                free_dimensions=dims,
                units=obj.units,
            )
        return dict(
            data=obj.data,
            dimensions=obj.data_array.dims,
            coordinates=_get_coordinate_quantities(obj.data_array),
            interpolation=obj.interpolation,
        )

    def from_yaml_tree(self, node: dict, tag: str, ctx):
        """Construct from tree."""
        if "expression" in node:
            dims_asdf = node.get("free_dimensions")
            units = {}
            dims = {}
            for k, v in dims_asdf.items():
                sym = v.symbol
                if sym is None:
                    sym = k
                else:
                    dims[sym] = k
                units[sym] = v.units

            if len(dims) == 0:
                dims = None

            return GenericSeries(
                node.get("expression"),
                parameters=node.get("parameters"),
                dims=dims,
                units=units,
            )

        return GenericSeries(
            node.get("data"),
            dims=node.get("dimensions"),
            coords=node.get("coordinates"),
            interpolation=node.get("interpolation"),
        )
