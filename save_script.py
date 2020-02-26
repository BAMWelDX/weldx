import asdf

from weldx import Q_
from weldx.asdf.extension import WeldxExtension, WeldxAsdfExtension
from weldx.asdf.tags.weldx.core.groove import get_groove

v_groove = get_groove(
    groove_type="VGroove",
    **dict(t=Q_(9, "mm"), alpha=Q_(50, "deg"), c=Q_(4, "mm"), b=Q_(2, "mm"))
)
u_groove = get_groove(
    groove_type="UGroove",
    **dict(
        t=Q_(15, "mm"), beta=Q_(9, "deg"), R=Q_(6, "mm"), c=Q_(3, "mm"), b=Q_(1, "mm")
    )
)

target = asdf.AsdfFile(
    dict(test001=v_groove, test002=u_groove),
    extensions=[WeldxAsdfExtension(), WeldxExtension()],
)
target.write_to("testfile.yml", all_array_storage="inline")
