import warnings

from ._version import get_versions

__all__ = ["geometry", "transformations", "utility", "asdf"]

# geometry packages
from weldx import geometry
from weldx import transformations
from weldx import utility

# asdf extensions and tags
from weldx import asdf

# from .constants import WELDX_UNIT_REGISTRY as ureg
from .constants import WELDX_QUANTITY as Q_

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    Q_([])

__version__ = get_versions()["version"]
del get_versions
