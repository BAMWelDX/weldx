"""Test Script. Use save_script.py before launching this script"""

import asdf

from weldx.asdf.extension import WeldxExtension, WeldxAsdfExtension
from weldx.asdf.tags.weldx.core.groove import (
    VGroove,
    UGroove,
    IGroove,
    UVGroove,
    VVGroove,
)
from weldx.asdf.tags.weldx.core.groove import HVGroove, HUGroove, DVGroove, DUGroove
from weldx.asdf.tags.weldx.core.groove import DHVGroove, DHUGroove, FFGroove
from weldx.all_groove import plot_groove


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
    plot_groove(opened["test004"])
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
    plot_groove(opened["test007"])
else:
    print("false")

if isinstance(opened["test008"], DVGroove):
    print(opened["test008"])
    plot_groove(opened["test008"])
else:
    print("false")

if isinstance(opened["test009"], DUGroove):
    print(opened["test009"])
    plot_groove(opened["test009"])
else:
    print("false")

if isinstance(opened["test010"], DHVGroove):
    print(opened["test010"])
    plot_groove(opened["test010"])
else:
    print("false")

if isinstance(opened["test011"], DHUGroove):
    print(opened["test011"])
    plot_groove(opened["test011"])
else:
    print("false")

if isinstance(opened["test012"], FFGroove):
    print(opened["test012"])
    plot_groove(opened["test012"])
else:
    print("false")

if isinstance(opened["test013"], FFGroove):
    print(opened["test013"])
    plot_groove(opened["test013"])
else:
    print("false")

if isinstance(opened["test014"], FFGroove):
    print(opened["test014"])
    plot_groove(opened["test014"])
else:
    print("false")

if isinstance(opened["test015"], FFGroove):
    print(opened["test015"])
    plot_groove(opened["test015"])
else:
    print("false")

if isinstance(opened["test016"], FFGroove):
    print(opened["test016"])
    plot_groove(opened["test016"])
else:
    print("false")

if isinstance(opened["test017"], FFGroove):
    print(opened["test017"])
    plot_groove(opened["test017"])
else:
    print("false")
