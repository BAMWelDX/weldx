"""Test Script. Use save_script.py before launching this script"""

import asdf
import matplotlib.pyplot as plt

from weldx.asdf.extension import WeldxExtension, WeldxAsdfExtension
from weldx.asdf.tags.weldx.core.groove import VGroove, UGroove, IGroove, UVGroove, VVGroove
from weldx.all_groove import plot_groove


opened = asdf.open("testfile.yml", extensions=[WeldxAsdfExtension(), WeldxExtension()])

if isinstance(opened["test001"], VGroove):
    print(opened["test001"])
    plot_groove(opened["test001"])
    # title(f"single V Groove Butt Weld\n {alpha}° groove angle")
else:
    print("false")

if isinstance(opened["test002"], UGroove):
    print(opened["test002"])
    plot_groove(opened["test002"])
    # title(f"single U Groove Butt Weld\n {beta}° groove angle")
else:
    print("false")

if isinstance(opened["test003"], IGroove):
    print(opened["test003"])
    plot_groove(opened["test003"])
    # title(f"I Groove")
else:
    print("false")

if isinstance(opened["test004"], UVGroove):
    print(opened["test004"])
    plot_groove(opened["test004"])
else:
    print("false")

if isinstance(opened["test005"], VVGroove):
    print(opened["test005"])
    plot_groove(opened["test005"])
else:
    print("false")