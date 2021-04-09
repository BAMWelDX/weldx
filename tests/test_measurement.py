"""Test the measurement package."""

from typing import Dict

import pytest
import xarray as xr

from tests._helpers import get_test_name
from weldx.measurement import Error, MeasurementChainGraph, Signal, SignalTransformation


class TestMeasurementChain:
    """Test the `MeasurementChain` class."""

    # helper functions -----------------------------------------------------------------

    @staticmethod
    def _default_init_kwargs(kwargs: Dict = None) -> Dict:
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
        default_kwargs = dict(
            name="name",
            source_name="source",
            source_error=Error(0.01),
            output_signal_type="analog",
            output_signal_unit="V",
        )
        if kwargs is not None:
            default_kwargs.update(kwargs)

        return default_kwargs

    @staticmethod
    def _default_transformation_kwargs(kwargs: Dict = None) -> Dict:
        """Return a dictionary of keyword arguments required by `add_transformation`.

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
        default_kwargs = dict(
            name="transformation",
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
        "kwargs",
        [
            dict(output_signal_type="analog"),
            dict(output_signal_type="digital"),
            dict(output_signal_unit="V"),
            dict(output_signal_unit="m"),
        ],
    )
    def test_init(kwargs: Dict):
        """Test the `__init__` method of the `MeasurementChain`.

        Parameters
        ----------
        kwargs:
            A dictionary with keyword arguments that are passed to the `__init__`
            method. Missing arguments are added.

        """

        kwargs = TestMeasurementChain._default_init_kwargs(kwargs)
        MeasurementChainGraph(**kwargs)

    # test_init_exceptions -------------------------------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        "kwargs,  exception_type, test_name",
        [(dict(output_signal_type="some type"), ValueError, "# invalid signal type")],
        ids=get_test_name,
    )
    def test_init_exceptions(kwargs: Dict, exception_type, test_name: str):
        """Test the exceptions of the `__init__` method.

        Parameters
        ----------
        kwargs :
            A dictionary with keyword arguments that are passed to the `__init__`
            method. Missing arguments are added.
        exception_type :
            The expected exception type
        test_name :
            Name of the test

        """
        kwargs = TestMeasurementChain._default_init_kwargs(kwargs)
        with pytest.raises(exception_type):
            MeasurementChainGraph(**kwargs)

    # test_add_transformations ---------------------------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(output_signal_type="analog"),
            dict(output_signal_type="digital"),
            dict(output_signal_unit="V"),
            dict(output_signal_unit="m"),
        ],
    )
    def test_add_transformation(kwargs):
        """Test the `add_transformation` method of the `MeasurementChain`.

        Parameters
        ----------
        kwargs:
            A dictionary with keyword arguments that are passed to the
            `add_transformation` method. Missing arguments are added.

        """
        mc = MeasurementChainGraph(**TestMeasurementChain._default_init_kwargs())

        kwargs = TestMeasurementChain._default_transformation_kwargs(kwargs)

        mc.add_transformation(**kwargs)

        # todo: add assertions

    # test_add_transformation_exceptions -----------------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        "kwargs,  exception_type, test_name",
        [
            (dict(output_signal_type="some type"), ValueError, "# invalid signal type"),
            (dict(input_signal_source="what"), KeyError, "# invalid signal source"),
        ],
        ids=get_test_name,
    )
    def test_add_transformation_exceptions(
        kwargs: Dict, exception_type, test_name: str
    ):
        """Test the exceptions of the `add_transformation` method.

        Parameters
        ----------
        kwargs :
            A dictionary with keyword arguments that are passed to the
            `add_transformation` method. Missing arguments are added.
        exception_type :
            The expected exception type
        test_name :
            Name of the test

        """
        mc = MeasurementChainGraph(**TestMeasurementChain._default_init_kwargs())

        kwargs = TestMeasurementChain._default_transformation_kwargs(kwargs)

        with pytest.raises(exception_type):
            mc.add_transformation(**kwargs)

    # test_get_signal ------------------------------------------------------------------

    @staticmethod
    def test_get_signal():
        """Test the `get_signal` method.

        This test assures that the returned signal is identical to the one passed
        to the measurement chain and that a key error is raised if the specified signal
        source is not present.

        """
        init_kwargs = TestMeasurementChain._default_init_kwargs()
        mc = MeasurementChainGraph(**init_kwargs)

        transformation_kwargs = TestMeasurementChain._default_transformation_kwargs()
        mc.add_transformation(**transformation_kwargs)

        source_signal = mc.get_signal(init_kwargs["source_name"])
        assert source_signal.signal_type == init_kwargs["output_signal_type"]
        assert source_signal.unit == init_kwargs["output_signal_unit"]

        tf_signal = mc.get_signal(transformation_kwargs["name"])
        assert tf_signal.signal_type == transformation_kwargs["output_signal_type"]
        assert tf_signal.unit == transformation_kwargs["output_signal_unit"]

        with pytest.raises(KeyError):
            mc.get_signal("not found")

    # test_add_signal_data -------------------------------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(data=xr.DataArray([2, 3])),
            dict(signal_source="source"),
        ],
    )
    def test_add_signal_data(kwargs):
        """Test the `add_signal_data` method of the `MeasurementChain`.

        Parameters
        ----------
        kwargs:
            A dictionary with keyword arguments that are passed to the
            `add_signal_data` method. If no name is in the kwargs, a default one is
            added.

        """
        mc = MeasurementChainGraph(**TestMeasurementChain._default_init_kwargs())
        mc.add_transformation(**TestMeasurementChain._default_transformation_kwargs())

        full_kwargs = dict(name="my_data", data=xr.DataArray([1, 2]))
        full_kwargs.update(kwargs)

        mc.add_signal_data(**full_kwargs)

    # test_add_signal_data_exceptions --------------------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        "kwargs,  exception_type, test_name",
        [
            (dict(signal_source="what"), KeyError, "# invalid signal source"),
        ],
        ids=get_test_name,
    )
    def test_add_signal_data_exceptions(kwargs: Dict, exception_type, test_name: str):
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
        mc = MeasurementChainGraph(**TestMeasurementChain._default_init_kwargs())
        mc.add_transformation(**TestMeasurementChain._default_transformation_kwargs())

        full_kwargs = dict(name="my_data", data=xr.DataArray([1, 2]))
        full_kwargs.update(kwargs)

        with pytest.raises(exception_type):
            mc.add_signal_data(**full_kwargs)

    # test_get_signal_data
    # -------------------------------------------------------------

    @staticmethod
    def test_get_signal_data():
        """Test the `get_signal_data` method.

        This test assures that the returned data is identical to the one passed
        to the
        measurement chain and that a key error is raised if the requested data is
        not
        present.

        """
        mc = MeasurementChainGraph(**TestMeasurementChain._default_init_kwargs())
        mc.add_transformation(**TestMeasurementChain._default_transformation_kwargs())

        data_name = "data"
        data = xr.DataArray([1, 2, 3])
        mc.add_signal_data(data_name, data)

        assert mc.get_signal_data(data_name).identical(data)

        with pytest.raises(KeyError):
            mc.get_signal_data("not found")

    # test_get_and_attach_transformation -----------------------------------------------

    @staticmethod
    def test_get_and_attach_transformation():
        """Test the `get_transformation` and `attach_transformation` methods."""
        mc = MeasurementChainGraph(**TestMeasurementChain._default_init_kwargs())
        mc.add_transformation(**TestMeasurementChain._default_transformation_kwargs())

        transformation = mc.get_transformation("transformation")

        mc_2 = MeasurementChainGraph(**TestMeasurementChain._default_init_kwargs())
        mc_2.attach_transformation(transformation)

        transformation_2 = mc_2.get_transformation("transformation")

        assert transformation == transformation_2

    # test_get_transformation_exception ------------------------------------------------

    @staticmethod
    def test_get_transformation_exception():
        """Test that a `KeyError` is raised if the transformation does not exist."""
        mc = MeasurementChainGraph(**TestMeasurementChain._default_init_kwargs())
        mc.add_transformation(**TestMeasurementChain._default_transformation_kwargs())

        with pytest.raises(KeyError):
            mc.get_transformation("not found")

    # test_attach_transformation_exceptions --------------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        "tf_kwargs, signal_source, exception_type, test_name",
        [
            ({}, "not present", KeyError, "# invalid signal source"),
            (
                dict(input_signal=Signal("analog", "")),
                None,
                ValueError,
                "# invalid input signal #1",
            ),
            (
                dict(input_signal=Signal("digital", "2")),
                None,
                ValueError,
                "# invalid input signal #2",
            ),
        ],
        ids=get_test_name,
    )
    def test_attach_transformation_exceptions(
        tf_kwargs: Dict, signal_source: str, exception_type, test_name: str
    ):
        """Test the exceptions of the `attach_transformation` method.

        Parameters
        ----------
        tf_kwargs :
            A dictionary with keyword arguments that are passed to the `__init__`
            method of the `SignalTransformation` class. Missing arguments are added.
        signal_source :
            The source of the signal that should be transformed
        exception_type :
            The expected exception type
        test_name :
            Name of the test

        """
        mc = MeasurementChainGraph(**TestMeasurementChain._default_init_kwargs())
        mc.add_transformation(**TestMeasurementChain._default_transformation_kwargs())

        kwargs = dict(
            name="transformation 2",
            error=Error(0.12),
            input_signal=Signal("digital", ""),
            output_signal=Signal("digital", "A"),
        )
        kwargs.update(tf_kwargs)

        transformation = SignalTransformation(**kwargs)

        with pytest.raises(exception_type):
            mc.attach_transformation(transformation, signal_source)
