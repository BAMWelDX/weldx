from dataclasses import dataclass

import pandas as pd

from weldx.asdf.util import dataclass_serialization_class

__all__ = ["PropertyTagTestClass", "PropertyTagTestClassConverter"]


@dataclass
class PropertyTagTestClass:
    """Helper class to test the shape validator"""

    prop1: pd.Timestamp = pd.Timestamp("2020-01-01")
    prop2: pd.Timestamp = pd.Timestamp("2020-01-02")
    prop3: pd.Timestamp = pd.Timestamp("2020-01-03")


PropertyTagTestClassConverter = dataclass_serialization_class(
    class_type=PropertyTagTestClass,
    class_name="debug/test_property_tag",
    version="0.1.0",
)
