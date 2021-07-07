"""Test the measurement package."""

from typing import Dict

import pytest
import xarray as xr

from weldx import Q_
from weldx.core import MathematicalExpression
from weldx.measurement import (
    Error,
    MeasurementChain,
    MeasurementEquipment,
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
        """Update a dict with default keyword arguments to create a `SignalSource`."""
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
        source_kwargs :
            A dictionary of key word arguments that should replace the arguments used
            for the default source.

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

        Use the kwargs parameter to modify the default values.

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
        """Update a dict with default keyword arguments to call `add_transformation`."""
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

    # test_from_equipment --------------------------------------------------------------

    @pytest.mark.parametrize(
        "num_sources, source_name, exception",
        [
            (0, None, ValueError),
            (1, None, None),
            (2, "source_1", None),
            (2, "wrong name", KeyError),
            (2, None, ValueError),
        ],
    )
    def test_from_equipment(
        self, num_sources: int, source_name: str, exception: Exception
    ):
        """Test the `from_equipment` factory and its exceptions.

        Parameters
        ----------
        num_sources :
            Number of sources of the generated equipment
        source_name :
            Corresponding parameter of `from_equipment`
        exception :
            Expected exception

        """
        sources = [
            SignalSource(**self._default_source_kwargs({"name": f"source_{i}"}))
            for i in range(num_sources)
        ]
        equipment = MeasurementEquipment("Equipment", sources=sources)

        if exception is not None:
            with pytest.raises(exception):
                MeasurementChain.from_equipment(
                    name="name", equipment=equipment, source_name=source_name
                )
        else:
            MeasurementChain.from_equipment(
                name="name", equipment=equipment, source_name=source_name
            )

    # test_add_transformations ---------------------------------------------------------

    @pytest.mark.parametrize(
        "tf_kwargs, exp_signal_type, exp_signal_unit",
        [
            ({}, "digital", "dimensionless"),
            (dict(type_transformation="AA"), "analog", "dimensionless"),
            (dict(type_transformation=None), "analog", "dimensionless"),
            (dict(func=None), "digital", "V"),
        ],
    )
    def test_add_transformation(self, tf_kwargs, exp_signal_type, exp_signal_unit):
        """Test the `add_transformation` method of the `MeasurementChain`.

        Parameters
        ----------
        tf_kwargs:
            A dictionary with keyword arguments that are used to construct the
            `SignalTransformation` that is passed to the `add_transformation` method.
            Missing arguments are added.
        exp_signal_type :
            The expected signal type after the transformation
        exp_signal_unit :
            The expected unit after the transformation

        """
        mc = MeasurementChain(**self._default_init_kwargs())

        mc.add_transformation(self._default_transformation(tf_kwargs))

        signal = mc.output_signal
        assert signal.signal_type == exp_signal_type
        assert signal.unit == exp_signal_unit

    # test_add_transformation_exceptions -----------------------------------------------

    @pytest.mark.parametrize(
        "tf_kwargs, input_signal_source, exception_type, test_name",
        [
            (dict(type_transformation="DA"), None, ValueError, "# inv. signal type #1"),
            (dict(type_transformation="DD"), None, ValueError, "# inv. signal type #2"),
            ({}, "not found", KeyError, "# invalid input signal source"),
            (dict(name="source"), None, KeyError, "# Name does already exist"),
            (
                dict(func=MathematicalExpression("x+a", parameters={"a": Q_(1, "A")})),
                None,
                ValueError,
                "# incompatible function",
            ),
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

    # test_add_transformation_from_equipment -------------------------------------------

    @pytest.mark.parametrize(
        "num_transformations, transformation_name, exception",
        [
            (0, None, ValueError),
            (1, None, None),
            (2, "transformation_1", None),
            (2, "wrong name", KeyError),
            (2, None, ValueError),
        ],
    )
    def test_add_transformation_from_equipment(
        self, num_transformations: int, transformation_name: str, exception
    ):
        """Test `add_transformation_from_equipment` and its exceptions.

        Parameters
        ----------
        num_transformations :
            Number of transformations of the generated equipment
        transformation_name :
            Corresponding parameter of `add_transformation_from_equipment`
        exception :
            Expected exception

        """
        mc = MeasurementChain(**self._default_init_kwargs())
        transformations = [
            self._default_transformation({"name": f"transformation_{i}"})
            for i in range(num_transformations)
        ]
        equipment = MeasurementEquipment(name="name", transformations=transformations)

        if exception is not None:
            with pytest.raises(exception):
                mc.add_transformation_from_equipment(
                    equipment=equipment, transformation_name=transformation_name
                )
        else:
            mc.add_transformation_from_equipment(
                equipment=equipment, transformation_name=transformation_name
            )

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

    # test_get_equipment ---------------------------------------------------------------

    @pytest.mark.parametrize(
        "signal_source, exception",
        [
            ("source", None),
            ("transformation_1", None),
            ("transformation_2", None),
            ("transformation_3", KeyError),
        ],
    )
    def test_get_equipment(self, signal_source, exception):
        """Test the `get_equipment` function and their exceptions.

        Parameters
        ----------
        signal_source :
            Corresponding function parameter
        exception :
            Expected exception

        """
        src_eq = MeasurementEquipment(
            "Source Eq", sources=[SignalSource(**self._default_source_kwargs())]
        )
        tf_eq = MeasurementEquipment(
            "Transformation_eq",
            transformations=[
                self._default_transformation({"name": "transformation_1"})
            ],
        )

        mc = MeasurementChain.from_equipment("Chain", src_eq)
        mc.add_transformation_from_equipment(tf_eq)
        mc.create_transformation("transformation_2", None, output_signal_unit="A")

        if exception is not None:
            with pytest.raises(exception):
                mc.get_equipment(signal_source=signal_source)
        else:
            mc.get_equipment(signal_source=signal_source)

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
        mc.create_transformation("transformation_2", None, output_signal_unit="A")

        assert mc.get_signal_data("transformation").identical(data)

        # no data
        with pytest.raises(KeyError):
            mc.get_signal_data("transformation_2")

        # source not present
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
