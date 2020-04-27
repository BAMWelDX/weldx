"""Debug asdf save of groove implementation (run after save_script.py)."""

import asdf

from weldx.asdf.extension import WeldxExtension, WeldxAsdfExtension
from weldx.asdf.tags.weldx.core.groove import (
    VGroove,
    UGroove,
    IGroove,
    UVGroove,
    VVGroove,
    HVGroove,
    HUGroove,
    DVGroove,
    DUGroove,
    DHVGroove,
    DHUGroove,
    FFGroove,
)


opened = asdf.open("testfile.yml", extensions=[WeldxAsdfExtension(), WeldxExtension()])

if isinstance(opened["test001"], VGroove):
    print(opened["test001"])
    opened["test001"].plot()
else:
    print("false")

if isinstance(opened["test002"], UGroove):
    print(opened["test002"])
    opened["test002"].plot()
else:
    print("false")

if isinstance(opened["test003"], IGroove):
    print(opened["test003"])
    opened["test003"].plot()
else:
    print("false")

if isinstance(opened["test004"], UVGroove):
    print(opened["test004"])
    opened["test004"].plot()
else:
    print("false")

if isinstance(opened["test005"], VVGroove):
    print(opened["test005"])
    opened["test005"].plot()
else:
    print("false")

if isinstance(opened["test006"], HVGroove):
    print(opened["test006"])
    opened["test006"].plot()
else:
    print("false")

if isinstance(opened["test007"], HUGroove):
    print(opened["test007"])
    opened["test007"].plot()
else:
    print("false")

if isinstance(opened["test008"], DVGroove):
    print(opened["test008"])
    opened["test008"].plot()
else:
    print("false")

if isinstance(opened["test009"], DUGroove):
    print(opened["test009"])
    opened["test009"].plot()
else:
    print("false")

if isinstance(opened["test010"], DHVGroove):
    print(opened["test010"])
    opened["test010"].plot()
else:
    print("false")

if isinstance(opened["test011"], DHUGroove):
    print(opened["test011"])
    opened["test011"].plot()
else:
    print("false")

if isinstance(opened["test012"], FFGroove):
    print(opened["test012"])
    opened["test012"].plot()
else:
    print("false")

if isinstance(opened["test013"], FFGroove):
    print(opened["test013"])
    opened["test013"].plot()
else:
    print("false")

if isinstance(opened["test014"], FFGroove):
    print(opened["test014"])
    opened["test014"].plot()
else:
    print("false")

if isinstance(opened["test015"], FFGroove):
    print(opened["test015"])
    opened["test015"].plot()
else:
    print("false")

if isinstance(opened["test016"], FFGroove):
    print(opened["test016"])
    opened["test016"].plot()
else:
    print("false")

if isinstance(opened["test017"], FFGroove):
    print(opened["test017"])
    opened["test017"].plot()
else:
    print("false")
