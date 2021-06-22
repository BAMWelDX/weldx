from copy import deepcopy

from weldx.asdf.types import WeldxType
from weldx.asdf.util import asdf_dataclass_serialization
from weldx.geometry import SpatialData


@asdf_dataclass_serialization
class SpatialDataTypeASDF(WeldxType):
    """ASDF serialization class for `SpatialData`."""

    name = "core/geometry/spatial_data"
    version = "1.0.0"
    types = [SpatialData]
    requires = ["weldx"]
    handle_dynamic_subclasses = True
