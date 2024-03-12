"""Contains general (mostly internal) utility functions."""

from __future__ import annotations

import functools
import sys
import warnings
from collections.abc import Callable, Collection, Hashable, Mapping, Sequence, Set
from functools import wraps
from importlib.util import find_spec
from inspect import getmembers, isfunction
from typing import ClassVar, Union

import numpy as np
import pandas as pd
import pint
import xarray as xr
from asdf.tags.core import NDArrayType
from boltons import iterutils
from decorator import decorator

from weldx.constants import WELDX_UNIT_REGISTRY as ureg
from weldx.exceptions import WeldxDeprecationWarning
from weldx.time import Time

__all__ = [
    "deprecated",
    "ureg_check_class",
    "inherit_docstrings",
    "dataclass_nested_eq",
    "compare_nested",
    "is_jupyterlab_session",
    "is_interactive_session",
    "_patch_mod_all",
    "apply_func_by_mapping",
    "check_matplotlib_available",
]


def deprecated(since: str = None, removed: str = None, message: str = None) -> Callable:
    """Mark a functions as deprecated.

    This decorator emits a warning when the function is used.

    Parameters
    ----------
    since :
        The version that marked the function as deprecated
    removed :
        The version that will remove the function
    message :
        Additional information that should be added to the warning

    Returns
    -------
    Callable :
        Wrapped function

    Notes
    -----
    Original source: https://stackoverflow.com/a/30253848/6700329

    """

    def _decorator(func):
        @wraps(func)
        def _new_func(*args, **kwargs):
            wm = f"Call to deprecated function {func.__name__}.\n"
            if since is not None:
                wm += f"Deprecated since: {since}\n"
            if removed is not None:
                wm += f"Removed in: {removed}\n"
            if message is not None:
                wm += message

            warnings.warn(wm, category=WeldxDeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return _new_func

    return _decorator


def ureg_check_class(*args):
    """Decorate class :code:`__init__` function with `pint.UnitRegistry.check`.

    Useful for adding unit checks to classes created with :code:`@dataclass` decorator.

    Parameters
    ----------
    args: str or pint.util.UnitsContainer or None
        Dimensions of each of the input arguments.
        Use :code:`None` to skip argument conversion.

    Returns
    -------
    type
        The class with unit checks added to its :code:`__init__` function.

    Raises
    ------
    TypeError
        If number of given dimensions does not match the number of function parameters.
    ValueError
        If the any of the provided dimensions cannot be parsed as a dimension.

    Examples
    --------
    A simple dataclass could look like this::

        @ureg_check_class("[length]","[time]")
        @dataclass
        class A:
            a: pint.Quantity
            b: pint.Quantity

        A(Q_(3,"mm"),Q_(3,"s"))

    """

    def _inner_decorator(
        original_class,
    ):
        # Make copy of original __init__, so we can call it without recursion
        orig_init = original_class.__init__

        # apply pint check decorator
        new_init = ureg.check(None, *args)(orig_init)

        # set new init
        original_class.__init__ = new_init  # Set the class' __init__ to the new one
        return original_class

    return _inner_decorator


def inherit_docstrings(cls):
    """Inherits (public) docstrings from parent classes.

    Traverses the MRO until it finds a docstring to use, or leave it blank,
    in case no parent has a docstring available.

    Parameters
    ----------
    cls: type
        The class to decorate.

    Returns
    -------
    cls: type
        The class with updated doc strings.

    """
    for name, func in getmembers(
        cls, predicate=lambda x: isfunction(x) or isinstance(x, property)
    ):
        if func.__doc__ or name.startswith("_"):
            continue
        for parent in cls.__mro__[1:]:
            if hasattr(parent, name):
                func.__doc__ = getattr(parent, name).__doc__
                if not func.__doc__:
                    warnings.warn(
                        f"could not derive docstring for {cls}.{name}",
                        stacklevel=2,
                        category=ImportWarning,
                    )
    return cls


def _array_equal(a, b):
    if a.shape != b.shape:
        return False
    return np.all(a == b)


_eq_compare_nested_input_types = Union[
    Sequence,
    Mapping,
    Collection,
    Set,
]


class _EqCompareNested:
    """Compares nested data structures like lists, sets, tuples, arrays, etc."""

    # some types need special comparison handling.
    compare_funcs: ClassVar = {
        (np.ndarray, NDArrayType, pint.Quantity, pd.Index): _array_equal,
        (xr.DataArray, xr.Dataset): lambda x, y: x.identical(y),
        (set): lambda x, y: x == y,  # list here to prevent entering nested for sets
        (Time): lambda x, y: x.equals(y),
    }
    # these types will be treated as equivalent.
    _type_equalities: ClassVar = [
        (np.ndarray, NDArrayType),
    ]

    @staticmethod
    def _compare(x, y) -> bool:
        # 1. strict type comparison (exceptions defined in _type_equalities).
        # 2. handle special comparison cases
        if not any(
            (type(x) in e and type(y) in e) for e in _EqCompareNested._type_equalities
        ) and type(x) is not type(y):
            return False

        for types, func in _EqCompareNested.compare_funcs.items():
            if isinstance(x, types):
                return func(x, y)

        return x == y

    @staticmethod
    def _enter(path, key, value):
        # Do not traverse types defined in compare_funcs. All other types are handled
        # like in boltons.iterutils.default_enter (e.g. descend into nested structures).
        # See `boltons.iterutils.remap` for details.
        if any(isinstance(value, t) for t in _EqCompareNested.compare_funcs):
            return value, False

        return iterutils.default_enter(path, key, value)

    @staticmethod
    def _visit(path, key, value, a, b) -> bool:
        """Traverses all elements in `compare_nested` argument a and b...

        and tries to obtain the path `p` in `b` using boltons.iterutils.get_path.
        The following cases can occur:
        1. If the path does not exist in `b` a KeyError will be raised.
        2. If the index `k` does not exist an IndexError is raised.
        3. If the other path exists, a comparison will be made using `_compare`.
           When the elements are not equal traversing `a` will be stopped
           by raising a RuntimeError.
        """
        data_structure = iterutils.get_path(a, path)
        other_data_structure = iterutils.get_path(b, path)

        other_value = other_data_structure[key]

        if not _EqCompareNested._enter(None, key, value)[1]:
            other_value = other_data_structure[key]
            # check lengths of Sequence types first and raise
            # prior starting a more expensive comparison!
            if isinstance(other_data_structure, (Sequence, Set)) and len(
                other_data_structure
            ) != len(data_structure):
                raise RuntimeError("len does not match")
            if isinstance(other_data_structure, Mapping) and any(
                other_data_structure.keys() ^ data_structure.keys()
            ):
                raise RuntimeError("keys do not match")
            if not _EqCompareNested._compare(value, other_value):
                raise RuntimeError("not equal")
        return True

    @staticmethod
    def compare_nested(
        a: _eq_compare_nested_input_types, b: _eq_compare_nested_input_types
    ) -> bool:
        """Deeply compares [nested] data structures combined of tuples, lists, dicts...

        Also compares non-nested data-structures.
        Arrays are compared using np.all and xr.DataArray.identical.
        Sets are compared by calling ``==`` and not traversed, hence the order of all
        set items is important.

        Parameters
        ----------
        a :
            a [nested] data structure to compare to `b`.
        b :
            a [nested] data structure to compare to `a`.

        Returns
        -------
        bool :
            True, if all elements (including dict keys) of a and b are equal.

        Raises
        ------
        TypeError
            When a or b is not a nested structure.

        """
        # we bind the input structures a, b to the visit function.
        visit = functools.partial(_EqCompareNested._visit, a=a, b=b)

        try:
            iterutils.remap(
                a, visit=visit, reraise_visit=True, enter=_EqCompareNested._enter
            )
        # Key not found in b, values not equal, more elements in a than in b
        except (KeyError, RuntimeError, IndexError):
            return False
        except TypeError as e:
            raise TypeError(
                "One of a or b is not a nested data structure (or a set)."
            ) from e

        return True


compare_nested = _EqCompareNested.compare_nested


def dataclass_nested_eq(original_class):
    """Set class :code:`__eq__` using :code:`util.compare_nested` on :code:`__dict__`.

    Useful for implementing :code:`__eq__` on classes
    created with :code:`@dataclass` decorator.

    Parameters
    ----------
    original_class:
        original class to decorate

    Returns
    -------
    type
        The class with overridden :code:`__eq__` function.

    Examples
    --------
    A simple dataclass could look like this::

        @dataclass_nested_eq
        @dataclass
        class A:
            a: np.ndarray

        a = A(np.arange(3))
        b = A(np.arange(3))
        assert a==b

    """

    def _new_eq(self, other):
        if not isinstance(other, type(self)):
            return False

        return compare_nested(self.__dict__, other.__dict__)

    # set new eq function
    original_class.__eq__ = _new_eq  # Set the class' __eq__ to the new one
    return original_class


def is_interactive_session() -> bool:
    """Check whether this Python session is interactive, e.g. Jupyter/IPython."""
    try:
        get_ipython = sys.modules["IPython"].get_ipython  # type: ignore[attr-defined]
        if not get_ipython():
            return False
        if "IPKernelApp" not in get_ipython().config:  # pragma: no cover
            return False
    except KeyError:
        return False
    else:
        return True


def is_jupyterlab_session() -> bool:
    """Check whether we are in a Jupyter-Lab session.

    Notes
    -----
    This is a heuristic based process inspection based on the current Jupyter lab
    (major 3) version. So it could fail in the future.
    It will also report false positive in case a classic notebook frontend is started
    via Jupyter lab.
    """
    import psutil

    # inspect parent process for any signs of being a jupyter lab server

    parent = psutil.Process().parent()
    if parent.name() == "jupyter-lab":
        return True
    keys = (
        "JUPYTERHUB_API_KEY",
        "JPY_API_TOKEN",
        "JUPYTERHUB_API_TOKEN",
    )
    env = parent.environ()
    if any(k in env for k in keys):
        return True

    return False


def _patch_mod_all(module_name: str):
    """Hack the __module__ attribute of __all__ members to the given module.

    Parameters
    ----------
    module_name :
        the fully qualified module name.

    This is needed as Sphinx currently does not respect the all variable and ignores
    the contents. By simulating that the "all" attributes are belonging here, we work
    around this situation. Can be removed up this is fixed:
    https://github.com/sphinx-doc/sphinx/issues/2021
    """
    this_mod = sys.modules[module_name]
    for name in getattr(this_mod, "__all__", ()):
        obj = getattr(this_mod, name)
        obj.__module__ = module_name


def apply_func_by_mapping(func_map: dict[Hashable, Callable], inputs):
    """Transform a dict by running functions mapped by keys over its values."""
    return {k: (func_map[k](v) if k in func_map else v) for k, v in inputs.items()}


@decorator
def check_matplotlib_available(func, *args, **kwargs):
    """Emit a warning if matplotlib is not available."""

    def _warn():
        warnings.warn(
            "Matplotlib unavailable! Cannot plot. "
            "Please install matplotlib or weldx_widgets.",
            stacklevel=3,
        )

    try:
        if find_spec("matplotlib.pyplot") is None:
            _warn()
            return
    except ModuleNotFoundError:
        _warn()
        return
    except ValueError:
        warnings.warn("Matplotlib is unavailable (module set to None).", stacklevel=2)
        return

    return func(*args, **kwargs)
