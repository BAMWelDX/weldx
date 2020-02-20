import warnings

# setup unit registry
from . import constants
from .constants import WELDX_UNIT_REGISTRY as ureg
from .constants import WELDX_QUANTITY as Q_

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    Q_([])

# asdf extensions and tags
import weldx.asdf
