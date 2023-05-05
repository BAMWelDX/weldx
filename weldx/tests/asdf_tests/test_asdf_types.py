import pandas as pd

from weldx.asdf.util import write_read_buffer_context
from weldx.constants import META_ATTR, USER_ATTR
from weldx.measurement import Error
from weldx.util import compare_nested


def test_meta_attr():
    e = Error(3.0)

    ts = pd.DatetimeIndex(["2020", "2021"])
    setattr(ts, META_ATTR, {"name": "reference years"})

    setattr(e, META_ATTR, {"ts": ts})
    setattr(e, USER_ATTR, {"description": "user info"})

    tree = {"Error": e}

    with write_read_buffer_context(tree) as data:
        e2 = data["Error"]

        assert e2 == e
        assert compare_nested(getattr(e2, META_ATTR), getattr(e, META_ATTR))
        assert compare_nested(getattr(e2, USER_ATTR), getattr(e, USER_ATTR))
        assert compare_nested(
            getattr(getattr(e2, META_ATTR)["ts"], META_ATTR), getattr(ts, META_ATTR)
        )
