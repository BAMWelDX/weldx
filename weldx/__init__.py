import warnings

# versioneer
from ._version import get_versions

__all__ = ["geometry", "transformations", "utility", "asdf", "Q_"]

from .constants import WELDX_QUANTITY as Q_

# geometry packages
import weldx.geometry
import weldx.transformations
import weldx.utility

# asdf extensions and tags
import weldx.asdf

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    Q_([])

__version__ = get_versions()["version"]
del get_versions
