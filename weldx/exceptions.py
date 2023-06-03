__all__ = [
    "WeldxDeprecationWarning",
    "WeldxException",
    "WeldxWarning",
]


class WeldxWarning(Warning):
    """
    The base warning class to inherit for all weldx warnings.
    """


class WeldxException(Exception):
    """
    The base exception class to inherit for all weldx exceptions.
    """


class WeldxDeprecationWarning(WeldxWarning, DeprecationWarning):
    """
    A warning class to indicate a deprecated feature.
    """
