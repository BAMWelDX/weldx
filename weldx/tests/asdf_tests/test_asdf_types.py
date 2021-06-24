import pandas as pd

from weldx.asdf.types import META_ATTR, USER_ATTR
from weldx.asdf.util import write_read_buffer
from weldx.measurement import Error


def test_meta_attr():
    e = Error(3.0)

    ts = pd.Timestamp("2020-01-01")
    setattr(ts, META_ATTR, {"name": "Timestamp"})

    setattr(e, META_ATTR, {"ts": ts})
    setattr(e, USER_ATTR, {"description": "user info"})

    tree = {"Error": e}

    data = write_read_buffer(tree)

    e2 = data["Error"]

    assert e2 == e
    assert getattr(e2, META_ATTR) == getattr(e, META_ATTR)
    assert getattr(e2, USER_ATTR) == getattr(e, USER_ATTR)
    assert getattr(getattr(e2, META_ATTR)["ts"], META_ATTR) == getattr(ts, META_ATTR)
