import warnings

__all__ = []

# from .constants import WELDX_UNIT_REGISTRY as ureg
from .constants import WELDX_QUANTITY as Q_

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    Q_([])

# asdf extensions and tags
import weldx.asdf

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
