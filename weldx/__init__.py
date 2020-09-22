import warnings

# asdf extensions and tags
import weldx.asdf

# main packages
import weldx.core
import weldx.geometry
import weldx.transformations
import weldx.utility
import weldx.welding
from weldx.constants import WELDX_QUANTITY as Q_

# versioneer
from ._version import get_versions

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    Q_([])

__version__ = get_versions()["version"]
del get_versions

__all__ = [
    "core",
    "geometry",
    "measurement",
    "transformations",
    "utility",
    "asdf",
    "Q_",
    "welding",
]
