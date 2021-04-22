"""Test the measurement package."""

from typing import Dict

import pytest
import xarray as xr

from weldx import Q_
from weldx.core import MathematicalExpression
from weldx.measurement import (
    Error,
    MeasurementChain,
    Signal,
    SignalSource,
    SignalTransformation,
)

from ._helpers import get_test_name


class TestMeasurementChain:
    """Test the `MeasurementChain` class."""

    # helper functions -----------------------------------------------------------------

    @staticmethod
    def _default_source_kwargs(kwargs: Dict = None) -> Dict:
        """Return a dictionary of keyword arguments required to create a `SignalSource`.

        Parameters
        ----------
        kwargs :
            A dictionary containing some key word arguments that should replace the
            default ones.

        Returns
        -------
        Dict :
            Dictionary with keyword arguments required to create a `SignalSource`

        """
        default_kwargs = dict(
            name="source", output_signal=Signal("analog", "V"), error=Error(0.01)
        )

        if kwargs is not None:
            default_kwargs.update(kwargs)

        return default_kwargs

    @classmethod
    def _default_init_kwargs(
        cls, kwargs: Dict = None, source_kwargs: Dict = None
    ) -> Dict:
        """Return a dictionary of keyword arguments required by the `__init__` method.

        Parameters
        ----------
        kwargs :
            A dictionary containing some key word arguments that should replace the
            default ones.

        Returns
        -------
        Dict :
            Dictionary with keyword arguments for the `__init__` method

        """
        source_kwargs = cls._default_source_kwargs(source_kwargs)

        default_kwargs = dict(
            name="name",
            source=SignalSource(**source_kwargs),
            signal_data=[1, 3, 5],
        )
        if kwargs is not None:
            default_kwargs.update(kwargs)

        return default_kwargs

    @staticmethod
    def _default_transformation(kwargs: Dict = None) -> SignalTransformation:
        """Return a default `SignalTransformation`.

        Parameters
        ----------
        kwargs :
            A dictionary containing some key word arguments that should replace the
            default ones.

        Returns
        -------
        Dict :
            Default `SignalTransformation`

        """
        default_kwargs = dict(
            name="transformation",
            error=Error(0.1),
            func=MathematicalExpression("a*x", parameters={"a": Q_(1, "1/V")}),
            type_transformation="AD",
        )
        if kwargs is not None:
            default_kwargs.update(kwargs)

        return SignalTransformation(**default_kwargs)

    @classmethod
    def _default_add_transformation_kwargs(cls, kwargs: Dict = None) -> Dict:
        """Return a dictionary of keyword arguments required by `add_transformation`.

        Parameters
        ----------
        kwargs :
            A dictionary containing some keyword arguments that should replace the
            default ones.

        Returns
        -------
        Dict :
            Dictionary with keyword arguments for the `add_transformation` method

        """

        default_kwargs = dict(
            transformation=cls._default_transformation(),
            error=Error(0.02),
            output_signal_type="digital",
            output_signal_unit="",
        )
        if kwargs is not None:
            default_kwargs.update(kwargs)

        return default_kwargs

    # test_init ------------------------------------------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        "kwargs, source_kwargs",
        [
            ({}, {}),
            (dict(signal_data=None), dict(output_signal=Signal("analog", "V", [1]))),
        ],
    )
    def test_init(kwargs: Dict, source_kwargs: Dict):
        """Test the `__init__` method of the `MeasurementChain`.

        Parameters
        ----------
        kwargs:
            A dictionary with keyword arguments that are passed to the `__init__`
            method. Missing arguments are added.
        source_kwargs :
            A dictionary with keyword arguments that are used to construct the
            `SignalSource` that is passed to the `__init__` method. Missing arguments
            are added.

        """
        kwargs = TestMeasurementChain._default_init_kwargs(kwargs, source_kwargs)
        MeasurementChain(**kwargs)

    # test_init_exceptions -------------------------------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        "kwargs, source_kwargs,  exception_type, test_name",
        [({}, {"output_signal": Signal("analog", "V", [1])}, KeyError, "# 2x data")],
        ids=get_test_name,
    )
    def test_init_exceptions(
        kwargs: Dict, source_kwargs: Dict, exception_type, test_name: str
    ):
        """Test the exceptions of the `__init__` method.

        Parameters
        ----------
        kwargs :
            A dictionary with keyword arguments that are passed to the `__init__`
            method. Missing arguments are added.
        source_kwargs :
            A dictionary with keyword arguments that are used to construct the
            `SignalSource` that is passed to the `__init__` method. Missing arguments
            are added.
        exception_type :
            The expected exception type
        test_name :
            Name of the test

        """
        kwargs = TestMeasurementChain._default_init_kwargs(kwargs, source_kwargs)
        with pytest.raises(exception_type):
            MeasurementChain(**kwargs)

    # test_add_transformations ---------------------------------------------------------

    @pytest.mark.parametrize(
        "tf_kwargs",
        [
            {},
            dict(type_transformation="AA"),
            dict(type_transformation=None),
            dict(func=None),
        ],
    )
    def test_add_transformation(self, tf_kwargs):
        """Test the `add_transformation` method of the `MeasurementChain`.

        Parameters
        ----------
        tf_kwargs:
            A dictionary with keyword arguments that are used to construct the
            `SignalTransformation` that is passed to the `add_transformation` method.
            Missing arguments are added.

        """
        mc = MeasurementChain(**self._default_init_kwargs())

        mc.add_transformation(self._default_transformation(tf_kwargs))

        # todo: add assertions (check returned signal)

    # test_add_transformation_exceptions -----------------------------------------------

    @pytest.mark.parametrize(
        "tf_kwargs, input_signal_source, exception_type, test_name",
        [
            (dict(type_transformation="DA"), None, ValueError, "# inv. signal type #1"),
            (dict(type_transformation="DD"), None, ValueError, "# inv. signal type #2"),
            ({}, "not found", KeyError, "# invalid input signal source"),
        ],
        ids=get_test_name,
    )
    def test_add_transformation_exceptions(
        self, tf_kwargs: Dict, input_signal_source: str, exception_type, test_name: str
    ):
        """Test the exceptions of the `add_transformation` method.

        Parameters
        ----------
        tf_kwargs:
            A dictionary with keyword arguments that are used to construct the
            `SignalTransformation` that is passed to the `add_transformation` method.
            Missing arguments are added.
        input_signal_source :
            The value of the corresponding parameter of 'add_transformation'
        exception_type :
            The expected exception type
        test_name :
            Name of the test

        """
        mc = MeasurementChain(**self._default_init_kwargs())

        tf = self._default_transformation(tf_kwargs)

        with pytest.raises(exception_type):
            mc.add_transformation(tf, input_signal_source=input_signal_source)

    # test_add_signal_data -------------------------------------------------------------

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(data=xr.DataArray([2, 3])),
            dict(signal_source="source"),
        ],
    )
    def test_add_signal_data(self, kwargs):
        """Test the `add_signal_data` method of the `MeasurementChain`.

        Parameters
        ----------
        kwargs:
            A dictionary with keyword arguments that are passed to the
            `add_signal_data` method. If no name is in the kwargs, a default one is
            added.

        """
        mc = MeasurementChain(**self._default_init_kwargs({"signal_data": None}))
        mc.add_transformation(self._default_transformation())

        full_kwargs = dict(data=xr.DataArray([1, 2]))
        full_kwargs.update(kwargs)

        mc.add_signal_data(**full_kwargs)

    # test_add_signal_data_exceptions --------------------------------------------------

    @pytest.mark.parametrize(
        "kwargs,  exception_type, test_name",
        [
            (dict(signal_source="what"), KeyError, "# invalid signal source"),
            (dict(signal_source="source"), KeyError, "# already has data #1"),
            (dict(signal_source="transformation"), KeyError, "# already has data #2"),
        ],
        ids=get_test_name,
    )
    def test_add_signal_data_exceptions(
        self, kwargs: Dict, exception_type, test_name: str
    ):
        """Test the exceptions of the `add_signal_data` method.

        Parameters
        ----------
        kwargs :
            A dictionary with keyword arguments that are passed to the `add_signal_data`
            method. Missing arguments are added.
        exception_type :
            The expected exception type
        test_name :
            Name of the test

        """
        mc = MeasurementChain(**self._default_init_kwargs())
        mc.add_transformation(self._default_transformation(), data=[1, 2, 3])
        mc.add_transformation(
            self._default_transformation(
                dict(name="transformation 2", type_transformation="DA")
            )
        )

        full_kwargs = dict(data=xr.DataArray([1, 2]))
        full_kwargs.update(kwargs)

        with pytest.raises(exception_type):
            mc.add_signal_data(**full_kwargs)

    # test_get_signal_data -------------------------------------------------------------

    def test_get_signal_data(self):
        """Test the `get_signal_data` method.

        This test assures that the returned data is identical to the one passed
        to the
        measurement chain and that a key error is raised if the requested data is
        not
        present.

        """
        data = xr.DataArray([1, 2, 3])

        mc = MeasurementChain(**self._default_init_kwargs())
        mc.add_transformation(self._default_transformation(), data=data)

        assert mc.get_signal_data("transformation").identical(data)

        with pytest.raises(KeyError):
            mc.get_signal_data("not found")

    # test_get_transformation ----------------------------------------------------------

    def test_get_transformation(self):
        """Test the `get_transformation` method."""

        mc = MeasurementChain(**self._default_init_kwargs())
        mc.add_transformation(self._default_transformation())

        transformation = mc.get_transformation("transformation")

        assert transformation == self._default_transformation()

    # test_get_transformation_exception ------------------------------------------------

    def test_get_transformation_exception(self):
        """Test that a `KeyError` is raised if the transformation does not exist."""
        mc = MeasurementChain(**self._default_init_kwargs())
        mc.add_transformation(self._default_transformation())

        with pytest.raises(KeyError):
            mc.get_transformation("not found")
