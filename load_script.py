"""Test Script. Use save_script.py before launching this script"""

import asdf
import matplotlib.pyplot as plt

from weldx.asdf.extension import WeldxExtension, WeldxAsdfExtension
from weldx.asdf.tags.weldx.core.groove import VGroove, UGroove, IGroove, UVGroove
from weldx.all_groove import groove_to_profile


opened = asdf.open("testfile.yml", extensions=[WeldxAsdfExtension(), WeldxExtension()])

fig, ax = plt.subplots(1, 3)
ax[0].axis("equal")
ax[0].grid(True)
ax[1].axis("equal")
ax[1].grid(True)
ax[2].axis("equal")
ax[2].grid(True)

if isinstance(opened["test001"], VGroove):
    print(opened["test001"])
    profile01 = groove_to_profile(opened["test001"])
    profile01_data = profile01.rasterize(0.2)
    ax[0].plot(profile01_data[0], profile01_data[1], ".")
    alpha = opened["test001"].alpha.to("deg").magnitude
    ax[0].set_title(f"single V Groove Butt Weld\n {alpha}° groove angle")
else:
    print("false")

if isinstance(opened["test002"], UGroove):
    print(opened["test002"])
    profile02 = groove_to_profile(opened["test002"])
    profile02_data = profile02.rasterize(0.3)
    ax[1].plot(profile02_data[0], profile02_data[1], ".")
    beta = opened["test002"].beta.to("deg").magnitude
    ax[1].set_title(f"single U Groove Butt Weld\n {beta}° groove angle")
else:
    print("false")

if isinstance(opened["test003"], IGroove):
    print(opened["test003"])
    profile03 = groove_to_profile(opened["test003"])
    profile03_data = profile03.rasterize(0.3)
    ax[2].plot(profile03_data[0], profile03_data[1], ".")
    ax[2].set_title(f"I Groove")
else:
    print("false")

plt.show()
