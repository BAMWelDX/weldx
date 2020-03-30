import warnings

# versioneer
from ._version import get_versions

__all__ = ["geometry", "transformations", "utility", "asdf"]

__version__ = get_versions()["version"]
del get_versions

# geometry packages
import weldx.geometry
import weldx.transformations
import weldx.utility

# asdf extensions and tags
import weldx.asdf

# from .constants import WELDX_UNIT_REGISTRY as ureg
from .constants import WELDX_QUANTITY as Q_

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    Q_([])
