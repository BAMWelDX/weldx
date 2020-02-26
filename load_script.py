import asdf

from weldx.asdf.extension import WeldxExtension, WeldxAsdfExtension
from weldx.asdf.tags.weldx.core.groove import VGroove, UGroove


opened = asdf.open("testfile.yml", extensions=[WeldxAsdfExtension(), WeldxExtension()])

if isinstance(opened["test001"], VGroove):
    print(opened["test001"])
else:
    print("false")

if isinstance(opened["test002"], UGroove):
    print(opened["test002"])
else:
    print("false")
