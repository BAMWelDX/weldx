"""Test the `CoordinateSystemManager` class."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pandas import Timestamp as TS

import weldx.transformations as tf
from weldx.constants import Q_
from weldx.core import MathematicalExpression, TimeSeries
from weldx.exceptions import WeldxException
from weldx.geometry import SpatialData
from weldx.tests._helpers import get_test_name, matrix_is_close
from weldx.time import Time, types_time_like, types_timestamp_like
from weldx.transformations import CoordinateSystemManager as CSM
from weldx.transformations import LocalCoordinateSystem as LCS
from weldx.transformations import WXRotation

from ._util import check_coordinate_system, check_cs_close, r_mat_x, r_mat_y, r_mat_z


@pytest.fixture
def csm_fix():
    """Create default coordinate system fixture."""
    csm_default = CSM("root")
    lcs_1 = LCS(coordinates=Q_([0, 1, 2], "mm"))
    lcs_2 = LCS(coordinates=Q_([0, -1, -2], "mm"))
    lcs_3 = LCS(coordinates=Q_([-1, -2, -3], "mm"))
    lcs_4 = LCS(r_mat_y(1 / 2), Q_([1, 2, 3], "mm"))
    lcs_5 = LCS(r_mat_y(3 / 2), Q_([2, 3, 1], "mm"))
    csm_default.add_cs("lcs1", "root", lcs_1)
    csm_default.add_cs("lcs2", "root", lcs_2)
    csm_default.add_cs("lcs3", "lcs1", lcs_3)
    csm_default.add_cs("lcs4", "lcs1", lcs_4)
    csm_default.add_cs("lcs5", "lcs2", lcs_5)

    return csm_default


@pytest.fixture()
def list_of_csm_and_lcs_instances():
    """Get a list of LCS and CSM instances."""
    lcs = [LCS(coordinates=Q_([i, 0, 0], "mm")) for i in range(11)]

    csm_0 = CSM("lcs0", "csm0")
    csm_0.add_cs("lcs1", "lcs0", lcs[1])
    csm_0.add_cs("lcs2", "lcs0", lcs[2])
    csm_0.add_cs("lcs3", "lcs2", lcs[3])

    csm_1 = CSM("lcs0", "csm1")
    csm_1.add_cs("lcs4", "lcs0", lcs[4])

    csm_2 = CSM("lcs5", "csm2")
    csm_2.add_cs("lcs3", "lcs5", lcs[5], lcs_child_in_parent=False)
    csm_2.add_cs("lcs6", "lcs5", lcs[6])

    csm_3 = CSM("lcs6", "csm3")
    csm_3.add_cs("lcs7", "lcs6", lcs[7])
    csm_3.add_cs("lcs8", "lcs6", lcs[8])

    csm_4 = CSM("lcs9", "csm4")
    csm_4.add_cs("lcs3", "lcs9", lcs[9], lcs_child_in_parent=False)

    csm_5 = CSM("lcs7", "csm5")
    csm_5.add_cs("lcs10", "lcs7", lcs[10])

    csm = [csm_0, csm_1, csm_2, csm_3, csm_4, csm_5]
    return [csm, lcs]


# test_init ----------------------------------------------------------------------------


def test_init():
    """Test the init method of the coordinate system manager."""
    # default construction ----------------------
    csm = CSM(root_coordinate_system_name="root")
    assert csm.number_of_coordinate_systems == 1
    assert csm.number_of_neighbors("root") == 0

    # Exceptions---------------------------------
    # Invalid root system name
    with pytest.raises(TypeError):
        CSM({})


# test_add_cs --------------------------------------------------------------------------


def test_add_cs():
    """Test the 'add_cs' function."""
    csm = CSM("r")
    ts = TimeSeries(MathematicalExpression("a*t", dict(a=Q_("m/s"))))

    lcs_data = [
        ("a", "r", LCS(coordinates=Q_([0, 1, 2], "mm")), True),
        ("b", "r", LCS(coordinates=Q_([0, -1, -2], "mm")), False),
        (
            "b",
            "r",
            LCS(coordinates=Q_([[0, -1, -2], [8, 2, 7]], "mm"), time=["1s", "2s"]),
            False,
        ),
        ("c", "b", LCS(r_mat_y(1 / 2), Q_([1, 2, 3], "mm")), True),
        ("c", "b", LCS(coordinates=Q_([-1, -2, -3], "mm")), True),
        ("b", "c", LCS(coordinates=Q_([-1, -2, -3], "mm")), False),
        ("b", "c", LCS(coordinates=Q_([-1, -2, -3], "mm")), True),
        ("d", "b", LCS(coordinates=Q_([0, 1, 2], "mm")), True),
        ("d", "b", LCS(r_mat_y(1 / 2), Q_([1, 2, 3], "mm")), True),
        ("e", "a", LCS(r_mat_y(3 / 2), Q_([2, 3, 1], "mm")), True),
        ("e", "a", LCS(coordinates=ts), True),
    ]
    exp_num_cs = 1
    assert csm.number_of_coordinate_systems == exp_num_cs

    for i, d in enumerate(lcs_data):
        name = d[0]
        parent = d[1]
        lcs = d[2]
        child_in_parent = d[3]

        if name not in csm.coordinate_system_names:
            exp_num_cs += 1

        csm.add_cs(name, parent, lcs, child_in_parent)

        assert csm.number_of_coordinate_systems == exp_num_cs, f"Testcase {i} failed"
        if child_in_parent:
            assert csm.get_cs(name, parent) == lcs, f"Testcase {i} failed"
            if not isinstance(lcs.coordinates, TimeSeries):
                assert csm.get_cs(parent, name) == lcs.invert(), f"Testcase {i} failed"
        else:
            if not isinstance(lcs.coordinates, TimeSeries):
                assert csm.get_cs(name, parent) == lcs.invert(), f"Testcase {i} failed"
            assert csm.get_cs(parent, name) == lcs, f"Testcase {i} failed"


# test_add_cs_reference_time -----------------------------------------------------------


@pytest.mark.parametrize(
    "has_timestamp_csm, has_timestamp_lcs_1, has_timestamp_lcs_2, exp_exception",
    [
        (True, False, False, None),
        (True, True, False, None),
        (True, False, True, None),
        (True, True, True, None),
        (False, False, False, None),
        (False, True, False, Exception),
        (False, False, True, Exception),
        (False, True, True, None),
    ],
)
def test_add_cs_reference_time(
    has_timestamp_csm, has_timestamp_lcs_1, has_timestamp_lcs_2, exp_exception
):
    """Test if reference time issues are caught while adding new coordinate systems.

    See 'Notes' section of the add_cs method documentation.

    Parameters
    ----------
    has_timestamp_csm : bool
        Set to `True` if the CoordinateSystemManager should have a reference time.
    has_timestamp_lcs_1 : bool
        Set to `True` if the first added coordinate system should have a reference
        time.
    has_timestamp_lcs_2 : bool
        Set to `True` if the second added coordinate system should have a reference
        time.
    exp_exception : Any
        Pass the expected exception type if the test should raise. Otherwise set to
        `None`

    """
    timestamp_csm = None
    timestamp_lcs_1 = None
    timestamp_lcs_2 = None

    if has_timestamp_csm:
        timestamp_csm = pd.Timestamp("2000-01-01")
    if has_timestamp_lcs_1:
        timestamp_lcs_1 = pd.Timestamp("2000-01-02")
    if has_timestamp_lcs_2:
        timestamp_lcs_2 = pd.Timestamp("2000-01-03")
    csm = tf.CoordinateSystemManager("root", time_ref=timestamp_csm)
    lcs_1 = tf.LocalCoordinateSystem(
        coordinates=Q_([[1, 2, 3], [3, 2, 1]], "mm"),
        time=pd.TimedeltaIndex([1, 2]),
        time_ref=timestamp_lcs_1,
    )
    lcs_2 = tf.LocalCoordinateSystem(
        coordinates=Q_([[1, 5, 3], [3, 5, 1]], "mm"),
        time=pd.TimedeltaIndex([0, 2]),
        time_ref=timestamp_lcs_2,
    )

    csm.add_cs("lcs_1", "root", lcs_1)

    if exp_exception is not None:
        with pytest.raises(exp_exception):
            csm.add_cs("lcs_2", "root", lcs_2)
    else:
        csm.add_cs("lcs_2", "root", lcs_2)


# test_add_coordinate_system_timeseries ------------------------------------------------


def test_add_coordinate_system_timeseries():
    """Test if adding an LCS with a `TimeSeries` as coordinates is possible."""
    csm = CSM("r")
    me = MathematicalExpression("a*t", dict(a=Q_([[1, 0, 0]], "m/s")))
    ts = TimeSeries(me)
    lcs = LCS(coordinates=ts)

    csm.add_cs("cs1", "r", lcs)


# test_add_coordinate_system_exceptions ------------------------------------------------


@pytest.mark.parametrize(
    "name, parent_name, lcs, exception_type, test_name",
    [
        ("lcs", "r00t", LCS(), ValueError, "# invalid parent system"),
        ("lcs4", "root", LCS(), ValueError, "# can't update - no neighbors"),
        ("lcs", LCS(), LCS(), TypeError, "# invalid parent system name type"),
        (LCS(), "root", LCS(), TypeError, "# invalid system name type"),
        ("new_lcs", "root", "a string", TypeError, "# invalid system type"),
    ],
    ids=get_test_name,
)
def test_add_coordinate_system_exceptions(
    csm_fix, name, parent_name, lcs, exception_type, test_name
):
    """Test the exceptions of the 'add_cs' method."""
    with pytest.raises(exception_type):
        csm_fix.add_cs(name, parent_name, lcs)


# test_create_cs_from_axis_vectors -----------------------------------------------------


@pytest.mark.parametrize("time_dep_orient", [True, False])
@pytest.mark.parametrize("time_dep_coord", [True, False])
@pytest.mark.parametrize("has_time_ref", [True, False])
def test_create_cs_from_axis_vectors(
    time_dep_orient: bool, time_dep_coord: bool, has_time_ref: bool
):
    """Test the ``create_cs_from_axis_vectors`` method."""
    if has_time_ref and not (time_dep_coord or time_dep_orient):
        return

    t = ["1s", "2s", "3s", "4s"] if time_dep_orient or time_dep_coord else None
    time_ref = "2011-07-22" if has_time_ref else None
    angles = [[30, 45, 60], [40, 35, 80], [1, 33, 7], [90, 180, 270]]

    o = WXRotation.from_euler("xyz", angles, degrees=True).as_matrix()
    c = Q_([[-1, 3, 2], [4, 2, 4], [5, 1, 2], [3, 3, 3]], "mm")

    if not time_dep_orient:
        o = o[0]
    if not time_dep_coord:
        c = c[0]

    x = o[..., 0] * 2
    y = o[..., 1] * 5
    z = o[..., 2] * 3
    kwargs = dict(coordinates=c, time=t, time_ref=time_ref)

    ref = LCS(o, c, t, time_ref)

    csm = CSM("r")
    csm.create_cs_from_axis_vectors("xyz", "r", x, y, z, **kwargs)
    csm.create_cs_from_axis_vectors("xy", "r", x=x, y=y, **kwargs)
    csm.create_cs_from_axis_vectors("yz", "r", y=y, z=z, **kwargs)
    csm.create_cs_from_axis_vectors("xz", "r", x=x, z=z, **kwargs)

    check_cs_close(csm.get_cs("xyz"), ref)
    check_cs_close(csm.get_cs("xy"), ref)
    check_cs_close(csm.get_cs("yz"), ref)
    check_cs_close(csm.get_cs("xz"), ref)


# test num_neighbors -------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, exp_num_neighbors",
    [("root", 2), ("lcs1", 3), ("lcs2", 2), ("lcs3", 1), ("lcs4", 1), ("lcs5", 1)],
)
def test_num_neighbors(csm_fix, name, exp_num_neighbors):
    """Test the num_neighbors function."""
    assert csm_fix.number_of_neighbors(name) == exp_num_neighbors


# test is_neighbor_of ------------------------------------------------------------------


@pytest.mark.parametrize(
    "name1, exp_result",
    [
        ("root", [False, True, True, False, False, False]),
        ("lcs1", [True, False, False, True, True, False]),
        ("lcs2", [True, False, False, False, False, True]),
        ("lcs3", [False, True, False, False, False, False]),
        ("lcs4", [False, True, False, False, False, False]),
        ("lcs5", [False, False, True, False, False, False]),
    ],
)
@pytest.mark.parametrize(
    "name2, result_idx",
    [("root", 0), ("lcs1", 1), ("lcs2", 2), ("lcs3", 3), ("lcs4", 4), ("lcs5", 5)],
)
def test_is_neighbor_of(csm_fix, name1, name2, result_idx, exp_result):
    """Test the is_neighbor_of function."""
    assert csm_fix.is_neighbor_of(name1, name2) is exp_result[result_idx]


# test_get_child_system_names ----------------------------------------------------------


@pytest.mark.parametrize(
    "cs_name, neighbors_only, result_exp",
    [
        ("root", True, ["lcs1", "lcs2"]),
        ("lcs1", True, ["lcs3", "lcs4"]),
        ("lcs2", True, ["lcs5"]),
        ("lcs3", True, []),
        ("lcs4", True, []),
        ("lcs5", True, []),
        ("root", False, ["lcs1", "lcs2", "lcs3", "lcs4", "lcs5"]),
        ("lcs1", False, ["lcs3", "lcs4"]),
        ("lcs2", False, ["lcs5"]),
        ("lcs3", False, []),
        ("lcs4", False, []),
        ("lcs5", False, []),
    ],
)
def test_get_child_system_names(csm_fix, cs_name, neighbors_only, result_exp):
    """Test the get_child_system_names function."""
    result = csm_fix.get_child_system_names(cs_name, neighbors_only)

    # check -------------------------------------------
    assert len(result) == len(result_exp)
    for name in result_exp:
        assert name in result


# test_delete_coordinate_system --------------------------------------------------------


@pytest.mark.parametrize(
    "lcs_del, delete_children, num_cs_exp, exp_children_deleted",
    [
        ("lcs1", True, 3, ["lcs3", "lcs4"]),
        ("lcs2", True, 4, ["lcs5"]),
        ("lcs3", True, 5, []),
        ("lcs4", True, 5, []),
        ("lcs5", True, 5, []),
        ("lcs3", False, 5, []),
        ("lcs4", False, 5, []),
        ("lcs5", False, 5, []),
        ("not included", False, 6, []),
        ("not included", True, 6, []),
    ],
)
def test_delete_coordinate_system(
    csm_fix, lcs_del, delete_children, exp_children_deleted, num_cs_exp
):
    """Test the delete function of the CSM."""
    # setup
    removed_lcs_exp = [lcs_del] + exp_children_deleted

    # delete coordinate system
    csm_fix.delete_cs(lcs_del, delete_children)

    # check
    edges = csm_fix.graph.edges

    assert csm_fix.number_of_coordinate_systems == num_cs_exp
    for lcs in removed_lcs_exp:
        assert not csm_fix.has_coordinate_system(lcs)
        for edge in edges:
            assert lcs not in edge


# test_delete_coordinate_system_exceptions ---------------------------------------------


@pytest.mark.parametrize(
    "name, delete_children, exception_type, test_name",
    [
        ("root", True, ValueError, "# root system can't be deleted #1"),
        ("root", False, ValueError, "# root system can't be deleted #2"),
        ("lcs1", False, Exception, "# system has children"),
    ],
    ids=get_test_name,
)
def test_delete_coordinate_system_exceptions(
    csm_fix, name, delete_children, exception_type, test_name
):
    """Test the exceptions of the 'add_cs' method."""
    with pytest.raises(exception_type):
        csm_fix.delete_cs(name, delete_children)


# test_comparison ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "csm_data, cs_data, merge_data, csm_diffs, cs_diffs, merge_diffs, exp_results",
    [
        (  # No diff in CSM
            [("root", "csm_root", "2000-05-26")],
            [],
            [],
            [],
            [],
            [],
            [True],
        ),
        (  # Diff in CSM root system
            [("root", "csm_root", "2000-05-26")],
            [],
            [],
            [(0, ("diff", "csm_root", "2000-05-26"))],
            [],
            [],
            [False],
        ),
        (  # Diff in CSM name
            [("root", "csm_root", "2000-05-26")],
            [],
            [],
            [(0, ("root", "csm_diff", "2000-05-26"))],
            [],
            [],
            [False],
        ),
        (  # Diff in CSM reference time
            [("root", "csm_root", "2000-05-26")],
            [],
            [],
            [(0, ("root", "csm_root", "2000-01-11"))],
            [],
            [],
            [False],
        ),
        (  # No diffs in CSM with coordinate systems
            [("root", "csm_root")],
            [(0, ("cs_1", "root")), (0, ("cs_2", "root"))],
            [],
            [],
            [],
            [],
            [True],
        ),
        (  # Different number of coordinate systems
            [("root", "csm_root"), ("root", "csm_root_2")],
            [(0, ("cs_1", "root")), (0, ("cs_2", "root"))],
            [],
            [],
            [(1, (1, ("cs_2", "root")))],
            [],
            [False, False],
        ),
        (  # different coordinate systems names
            [("root", "csm_root")],
            [(0, ("cs_1", "root")), (0, ("cs_2", "root"))],
            [],
            [],
            [(1, (0, ("cs_3", "root")))],
            [],
            [False],
        ),
        (  # different coordinate system references
            [("root", "csm_root")],
            [(0, ("cs_1", "root")), (0, ("cs_2", "root"))],
            [],
            [],
            [(1, (0, ("cs_2", "cs_1")))],
            [],
            [False],
        ),
        (  # no diffs in CSM with multiple subsystems
            [("root", "csm_root"), ("cs_1", "csm_1"), ("cs_2", "csm_2")],
            [(0, ("cs_1", "root")), (1, ("cs_3", "cs_1")), (2, ("cs_1", "cs_2"))],
            [(1, 0), (2, 0)],
            [],
            [],
            [],
            [True, True, True],
        ),
        (  # different merge order
            [("root", "csm_root"), ("cs_1", "csm_1"), ("cs_2", "csm_2")],
            [(0, ("cs_1", "root")), (1, ("cs_3", "cs_1")), (2, ("cs_1", "cs_2"))],
            [(1, 0), (2, 0)],
            [],
            [],
            [(0, (2, 0)), (1, (1, 0))],
            [True, True, True],
        ),
        (  # different number of subsystems
            [("root", "csm_root"), ("cs_1", "csm_1"), ("cs_2", "csm_2")],
            [(0, ("cs_1", "root")), (1, ("cs_3", "cs_1")), (2, ("cs_1", "cs_2"))],
            [(1, 0), (2, 0)],
            [],
            [],
            [(1, None)],
            [False, True, True],
        ),
        (  # different root system name of subsystem
            [("root", "csm_root"), ("cs_1", "csm_1"), ("cs_2", "csm_2")],
            [(0, ("cs_1", "root")), (1, ("cs_3", "cs_1")), (2, ("cs_1", "cs_2"))],
            [(1, 0), (2, 0)],
            [(2, ("diff", "csm_2"))],
            [(2, (2, ("cs_1", "diff")))],
            [],
            [False, True, False],
        ),
        (  # different subsystem name
            [("root", "csm_root"), ("cs_1", "csm_1"), ("cs_2", "csm_2")],
            [(0, ("cs_1", "root")), (1, ("cs_3", "cs_1")), (2, ("cs_1", "cs_2"))],
            [(1, 0), (2, 0)],
            [(1, ("cs_1", "diff"))],
            [],
            [],
            [False, False, True],
        ),
        (  # different subsystem reference time
            [("root", "csm_root"), ("cs_1", "csm_1"), ("cs_2", "csm_2")],
            [(0, ("cs_1", "root")), (1, ("cs_3", "cs_1")), (2, ("cs_1", "cs_2"))],
            [(1, 0), (2, 0)],
            [(1, ("cs_1", "csm_1", "2000-01-01"))],
            [],
            [],
            [False, False, True],
        ),
        (  # subsystem merged at different nodes
            [("root", "csm_root"), ("cs_1", "csm_1"), ("cs_2", "csm_2")],
            [(0, ("cs_1", "root")), (1, ("cs_3", "cs_1")), (2, ("cs_1", "cs_2"))],
            [(1, 0), (2, 0)],
            [],
            [(2, (2, ("root", "cs_2")))],
            [],
            [False, True, False],
        ),
        (  # subsystem lcs name different
            [("root", "csm_root"), ("cs_1", "csm_1"), ("cs_2", "csm_2")],
            [(0, ("cs_1", "root")), (1, ("cs_3", "cs_1")), (2, ("cs_1", "cs_2"))],
            [(1, 0), (2, 0)],
            [],
            [(1, (1, ("diff", "cs_1")))],
            [],
            [False, False, True],
        ),
        (  # subsystem lcs different
            [("root", "csm_root"), ("cs_1", "csm_1"), ("cs_2", "csm_2")],
            [(0, ("cs_1", "root")), (1, ("cs_3", "cs_1")), (2, ("cs_1", "cs_2"))],
            [(1, 0), (2, 0)],
            [],
            [(1, (1, ("cs_3", "cs_1", None, Q_([1, 0, 0], "mm"))))],
            [],
            [False, False, True],
        ),
        (  # no diffs in CSM with nested subsystems
            [("root", "csm_root"), ("cs_1", "csm_1"), ("cs_2", "csm_2")],
            [(0, ("cs_1", "root")), (1, ("cs_3", "cs_1")), (2, ("cs_1", "cs_2"))],
            [(2, 1), (1, 0)],
            [],
            [],
            [],
            [True, True, True],
        ),
        (  # nested vs. not nested
            [("root", "csm_root"), ("cs_1", "csm_1"), ("cs_2", "csm_2")],
            [(0, ("cs_1", "root")), (1, ("cs_3", "cs_1")), (2, ("cs_1", "cs_2"))],
            [(2, 1), (1, 0)],
            [],
            [],
            [(0, (1, 0)), (0, (2, 0))],
            [False, False, True],
        ),
        (  # nested system has different root system
            [("root", "csm_root"), ("cs_1", "csm_1"), ("cs_2", "csm_2")],
            [(0, ("cs_1", "root")), (1, ("cs_3", "cs_1")), (2, ("cs_1", "cs_2"))],
            [(2, 1), (1, 0)],
            [(2, ("cs_4", "csm_2"))],
            [(2, (2, ("cs_1", "cs_4")))],
            [],
            [False, False, False],
        ),
        (  # nested system has different name
            [("root", "csm_root"), ("cs_1", "csm_1"), ("cs_2", "csm_2")],
            [(0, ("cs_1", "root")), (1, ("cs_3", "cs_1")), (2, ("cs_1", "cs_2"))],
            [(2, 1), (1, 0)],
            [(2, ("cs_2", "diff"))],
            [],
            [],
            [False, False, False],
        ),
        (  # nested system has different reference time
            [("root", "csm_root"), ("cs_1", "csm_1"), ("cs_2", "csm_2")],
            [(0, ("cs_1", "root")), (1, ("cs_3", "cs_1")), (2, ("cs_1", "cs_2"))],
            [(2, 1), (1, 0)],
            [(2, ("cs_2", "csm_2", "2000-04-01"))],
            [],
            [],
            [False, False, False],
        ),
        (  # nested system has lcs with different name
            [("root", "csm_root"), ("cs_1", "csm_1"), ("cs_2", "csm_2")],
            [
                (0, ("cs_1", "root")),
                (1, ("cs_3", "cs_1")),
                (2, ("cs_1", "cs_2")),
                (2, ("cs_4", "cs_2")),
            ],
            [(2, 1), (1, 0)],
            [],
            [(3, (2, ("diff", "cs_2")))],
            [],
            [False, False, False],
        ),
        (  # nested system has lcs with different reference system
            [("root", "csm_root"), ("cs_1", "csm_1"), ("cs_2", "csm_2")],
            [
                (0, ("cs_1", "root")),
                (1, ("cs_3", "cs_1")),
                (2, ("cs_1", "cs_2")),
                (2, ("cs_4", "cs_2")),
            ],
            [(2, 1), (1, 0)],
            [],
            [(3, (2, ("cs_4", "cs_1")))],
            [],
            [False, False, False],
        ),
        (  # nested system has different lcs
            [("root", "csm_root"), ("cs_1", "csm_1"), ("cs_2", "csm_2")],
            [(0, ("cs_1", "root")), (1, ("cs_3", "cs_1")), (2, ("cs_1", "cs_2"))],
            [(2, 1), (1, 0)],
            [],
            [(2, (2, ("cs_1", "cs_2", None, Q_([1, 0, 0], "mm"))))],
            [],
            [False, False, False],
        ),
    ],
)
def test_comparison(
    csm_data, cs_data, merge_data, csm_diffs, cs_diffs, merge_diffs, exp_results
):
    """Test the `__eq__` function.

    The test creates one or more CSM instances, adds coordinate systems and merges
    them. Then a second set of instances is created using a modified copy of the
    data used to create the first set of CSM instances. Afterwards, all instances
    are compared using the `==` operator and the results are checked to match the
    expectation.

    Parameters
    ----------
    csm_data :
        A list containing the arguments that should be passed to the CSM
        constructor. For each list entry a CSM instance is generated
    cs_data :
        A list containing the data to create coordinate systems. Each entry is a
        tuple containing the list index of the target CSM instance and the
        arguments that should be passed to the ``create_cs`` method
    merge_data :
        A list of tuples. Each tuple consists of two indices. The first one is the
        index of the source CSM and the second one of the target CSM. If an entry
        is `None`, it is skipped and no merge operation is performed
    csm_diffs :
        A list of modifications that should be applied to the ``csm_data`` before
        creating the second set of CSM instances. Each entry is a tuple containing
        the index and new value of the data that should be modified.
    cs_diffs :
        A list of modifications that should be applied to the ``cs_data`` before
        creating the coordinate systems of the second set of CSM instances. Each
        entry is a tuple containing the index and new value of the data that should
        be modified.
    merge_diffs :
        A list of modifications that should be applied to the ``merge_data`` before
        merging the second set of CSM instances. Each entry is a tuple containing
        the index and new value of the data that should be modified.
    exp_results :
        A list containing the expected results of each instance comparison

    """

    # define support function
    def create_csm_list(csm_data_list, cs_data_list, merge_data_list):
        """Create a list of CSM instances."""
        csm_list = []
        csm_list = [tf.CoordinateSystemManager(*args) for args in csm_data_list]

        for data in cs_data_list:
            csm_list[data[0]].create_cs(*data[1])

        for merge in merge_data_list:
            if merge is not None:
                csm_list[merge[1]].merge(csm_list[merge[0]])

        return csm_list

    # create diff inputs
    csm_data_diff = deepcopy(csm_data)
    for diff in csm_diffs:
        csm_data_diff[diff[0]] = diff[1]

    cs_data_diff = deepcopy(cs_data)
    for diff in cs_diffs:
        cs_data_diff[diff[0]] = diff[1]

    merge_data_diff = deepcopy(merge_data)
    for diff in merge_diffs:
        merge_data_diff[diff[0]] = diff[1]

    # create CSM instances
    csm_list_1 = create_csm_list(csm_data, cs_data, merge_data)
    csm_list_2 = create_csm_list(csm_data_diff, cs_data_diff, merge_data_diff)

    # test
    for i, _ in enumerate(csm_list_1):
        assert (csm_list_1[i] == csm_list_2[i]) is exp_results[i]
        assert (csm_list_1[i] != csm_list_2[i]) is not exp_results[i]


# test_comparison_wrong_type -----------------------------------------------------------


def test_comparison_wrong_type():
    """Test the comparison with other types."""
    csm = tf.CoordinateSystemManager("root", "csm")
    assert (csm == 4) is False
    assert (csm != 4) is True


def test_comparison_data():
    csm1 = tf.CoordinateSystemManager("root", "csm")
    csm2 = tf.CoordinateSystemManager("root", "csm")
    data = np.arange(12).reshape((4, 3))
    csm1.assign_data(data, data_name="foo", reference_system="root")
    csm2.assign_data(data, data_name="foo", reference_system="root")

    assert csm1 == csm2
    csm2.assign_data(data, data_name="bar", reference_system="root")
    assert csm1 != csm2


# test_time_union ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "csm_ref_time_day, lcs_times, lcs_ref_time_days, edges,exp_time, exp_ref_time_day",
    [
        # all systems are time dependent
        ("21", [[1, 5, 6], [3, 6, 9]], ["22", "21"], None, [2, 3, 6, 7, 9], "21"),
        ("21", [[1, 5, 6], [3, 6, 9]], ["22", None], None, [2, 3, 6, 7, 9], "21"),
        ("21", [[2, 6, 7], [3, 6, 9]], [None, None], None, [2, 3, 6, 7, 9], "21"),
        (None, [[1, 5, 6], [3, 6, 9]], ["22", "21"], None, [2, 3, 6, 7, 9], "21"),
        (None, [[1, 5, 6], [3, 6, 9]], [None, None], None, [1, 3, 5, 6, 9], None),
        ("21", [[3, 4], [6, 9], [4, 8]], ["22", "20", "21"], None, [4, 5, 8], "21"),
        ("21", [[3, 4], [5, 8], [4, 8]], ["22", None, None], None, [4, 5, 8], "21"),
        ("21", [[3, 4], [3, 8], [4, 8]], [None, None, None], None, [3, 4, 8], "21"),
        (None, [[3, 4], [6, 9], [4, 8]], ["22", "20", "21"], None, [4, 5, 8], "21"),
        (None, [[3, 4], [3, 8], [4, 8]], [None, None, None], None, [3, 4, 8], None),
        # Contains static systems
        ("21", [[3, 4], None, [4, 8]], ["22", None, "21"], None, [4, 5, 8], "21"),
        ("21", [[3, 4], None, [4, 8]], ["22", None, None], None, [4, 5, 8], "21"),
        ("21", [[3, 4], None, [4, 8]], [None, None, None], None, [3, 4, 8], "21"),
        (None, [[3, 4], None, [4, 8]], ["22", None, "21"], None, [4, 5, 8], "21"),
        (None, [[3, 4], None, [4, 8]], [None, None, None], None, [3, 4, 8], None),
        # include only specific edges
        ("21", [[3, 4], None, [4, 8]], ["22", None, "21"], [0, 1], [4, 5], "21"),
        ("21", [[3, 4], None, [4, 8]], ["22", None, None], [0, 1], [4, 5], "21"),
        ("21", [[3, 4], None, [4, 8]], [None, None, None], [0, 1], [3, 4], "21"),
        (None, [[3, 4], None, [4, 8]], ["22", None, "21"], [0, 1], [4, 5], "21"),
        (None, [[3, 4], None, [4, 8]], [None, None, None], [0, 1], [3, 4], None),
        ("21", [[3, 4], None, [4, 8]], ["22", None, "21"], [0, 2], [4, 5, 8], "21"),
        ("21", [[3, 4], None, [4, 8]], ["22", None, None], [0, 2], [4, 5, 8], "21"),
        ("21", [[3, 4], None, [4, 8]], [None, None, None], [0, 2], [3, 4, 8], "21"),
        (None, [[3, 4], None, [4, 8]], ["22", None, "21"], [0, 2], [4, 5, 8], "21"),
        (None, [[3, 4], None, [4, 8]], [None, None, None], [0, 2], [3, 4, 8], None),
        ("21", [[3, 4], None, [4, 8]], ["22", None, "21"], [1, 2], [4, 8], "21"),
        ("21", [[3, 4], None, [4, 8]], ["22", None, None], [1, 2], [4, 8], "21"),
        ("21", [[3, 4], None, [4, 8]], [None, None, None], [1, 2], [4, 8], "21"),
        (None, [[3, 4], None, [4, 8]], ["22", None, "21"], [1, 2], [4, 8], "21"),
        (None, [[3, 4], None, [4, 8]], [None, None, None], [1, 2], [4, 8], None),
    ],
)
def test_time_union(
    csm_ref_time_day,
    lcs_times,
    lcs_ref_time_days,
    edges,
    exp_time,
    exp_ref_time_day,
):
    """Test the time_union function of the CSM.

    Parameters
    ----------
    csm_ref_time_day : str
        An arbitrary day number string in the range [1, 31] or `None`. The value is
        used to create the reference timestamp of the CSM
    lcs_times : List
        A list containing an arbitrary number of time delta values (days) that are
        used to create a corresponding number of `LocalCoordinateSystem` instances
        which are added to the CSM. If a value is `None`, the generated coordinate
        system will be static
    lcs_ref_time_days : List
        A list where the values are either arbitrary day number strings in the range
        [1, 31] or `None`. Those values are used to create the reference timestamps
        for the coordinate systems of the CSM. The list must have the same length as
        the one passed to the ``lcs_times`` parameter
    edges : List
        A list that specifies the indices of the ``lcs_times`` parameter that should
        be considered in the time union. If `None` is passed, all are used. Note
        that the information is used to create the correct inputs to the
        ``time_union`` function and isn't passed directly.
    exp_time : List
        A list containing time delta values (days) that are used to generate the
        expected result data
    exp_ref_time_day : str
        An arbitrary day number string in the range [1, 31] or `None`. The value is
        used as reference time to create the expected result data. If it is set to
        `None`, the expected result data type is a `pandas.TimedeltaIndex` and a
        `pandas.DatetimeIndex` otherwise

    """
    # create full time data
    csm_time_ref = None
    if csm_ref_time_day is not None:
        csm_time_ref = f"2010-03-{csm_ref_time_day}"

    lcs_time_ref = [None for _ in range(len(lcs_times))]
    for i, _ in enumerate(lcs_times):
        if lcs_times[i] is not None:
            lcs_times[i] = pd.to_timedelta(lcs_times[i], "D")
        if lcs_ref_time_days[i] is not None:
            lcs_time_ref[i] = pd.Timestamp(f"2010-03-{lcs_ref_time_days[i]}")

    # create coordinate systems
    lcs = []
    for i, _ in enumerate(lcs_times):
        if isinstance(lcs_times[i], pd.TimedeltaIndex):
            coordinates = Q_([[j, j, j] for j in range(len(lcs_times[i]))], "mm")
        else:
            coordinates = Q_([1, 2, 3], "mm")
        lcs += [
            tf.LocalCoordinateSystem(
                None,
                coordinates,
                lcs_times[i],
                lcs_time_ref[i],
            )
        ]

    # create CSM and add coordinate systems
    csm = tf.CoordinateSystemManager("root", "base", csm_time_ref)
    for i, lcs_ in enumerate(lcs):
        csm.add_cs(f"lcs_{i}", "root", lcs_)

    # create expected data type
    exp_time = pd.to_timedelta(exp_time, "D")
    if exp_ref_time_day is not None:
        exp_time = pd.Timestamp(f"2010-03-{exp_ref_time_day}") + exp_time

    # create correct list of edges
    if edges is not None:
        for i, _ in enumerate(edges):
            edges[i] = ("root", f"lcs_{edges[i]}")

    # check time_union result
    assert np.all(csm.time_union(list_of_edges=edges) == exp_time)


# test_time_union_time_series_coords ---------------------------------------------------


@pytest.mark.parametrize(
    " tdp_orientation, add_discrete_lcs, list_of_edges, exp_time",
    [
        (False, False, None, None),
        (True, False, None, [1, 2]),
        (False, True, None, [2, 3]),
        (True, True, None, [1, 2, 3]),
        (False, True, [("tdp", "base"), ("ts", "base")], [2, 3]),
        (False, True, [("tdp", "base"), ("base", "ts")], [2, 3]),
        (False, True, [("st", "base"), ("ts", "base")], None),
        (False, True, [("st", "base"), ("base", "ts")], None),
        (False, True, [("ts", "base")], None),
        (False, True, [("base", "ts")], None),
        (False, True, [("tdp", "base"), ("st", "base")], [2, 3]),
        (False, True, [("tdp", "base"), ("st", "base"), ("ts", "base")], [2, 3]),
        (False, True, [("tdp", "base"), ("st", "base"), ("base", "ts")], [2, 3]),
        (True, True, [("tdp", "base"), ("ts", "base")], [1, 2, 3]),
        (True, True, [("tdp", "base"), ("base", "ts")], [1, 2, 3]),
        (True, True, [("st", "base"), ("ts", "base")], [1, 2]),
        (True, True, [("st", "base"), ("base", "ts")], [1, 2]),
        (True, True, [("ts", "base")], [1, 2]),
        (True, True, [("base", "ts")], [1, 2]),
        (True, True, [("tdp", "base"), ("st", "base")], [2, 3]),
        (True, True, [("tdp", "base"), ("st", "base"), ("ts", "base")], [1, 2, 3]),
        (True, True, [("tdp", "base"), ("st", "base"), ("base", "ts")], [1, 2, 3]),
    ],
)
def test_time_union_time_series_coords(
    tdp_orientation, add_discrete_lcs, list_of_edges, exp_time
):
    """Test time_union with an lcs that has a `TimeSeries` as coordinates.

    Parameters
    ----------
    tdp_orientation :
        If `True`, the LCS with the `TimeSeries` also has discrete time dependent
        orientations
    add_discrete_lcs :
        If `True`, another time dependent system with discrete values is added to
        the CSM
    list_of_edges :
        A list of edges that should be passed to `time_union`
    exp_time :
        The expected time values (in seconds)

    """
    ts = TimeSeries(MathematicalExpression("a*t", dict(a=Q_([[1, 2, 3]], "mm/s"))))
    lcs_ts_orientation = None
    lcs_ts_time = None
    if tdp_orientation:
        lcs_ts_orientation = WXRotation.from_euler("x", [0, 2]).as_matrix()
        lcs_ts_time = Q_([1, 2], "s")

    csm = CSM("base")
    csm.create_cs("st", "base", coordinates=Q_([2, 2, 2], "mm"))
    csm.create_cs("ts", "base", lcs_ts_orientation, ts, lcs_ts_time)
    if add_discrete_lcs:
        csm.create_cs(
            "tdp",
            "base",
            coordinates=Q_([[2, 4, 5], [2, 2, 2]], "mm"),
            time=Q_([2, 3], "s"),
        )

    if exp_time is not None:
        exp_time = pd.to_timedelta(exp_time, unit="s")
    assert np.all(exp_time == csm.time_union(list_of_edges))


# test_get_local_coordinate_system_no_time_dep -----------------------------------------


@pytest.mark.parametrize(
    "system_name, reference_name, exp_orientation, exp_coordinates",
    [
        ("lcs_1", None, r_mat_z(0.5), Q_([1, 2, 3], "m")),
        ("lcs_2", None, r_mat_y(0.5), Q_([3, -3, 1], "m")),
        ("lcs_3", None, r_mat_x(0.5), Q_([1, -1, 3], "m")),
        ("lcs_3", "root", [[0, 1, 0], [0, 0, -1], [-1, 0, 0]], Q_([6, -4, 0], "m")),
        ("root", "lcs_3", [[0, 0, -1], [1, 0, 0], [0, -1, 0]], Q_([0, -6, -4], "m")),
        ("lcs_3", "lcs_1", [[0, 0, -1], [0, -1, 0], [-1, 0, 0]], Q_([-6, -5, -3], "m")),
        ("lcs_1", "lcs_3", [[0, 0, -1], [0, -1, 0], [-1, 0, 0]], Q_([-3, -5, -6], "m")),
    ],
)
def test_get_local_coordinate_system_no_time_dep(
    system_name, reference_name, exp_orientation, exp_coordinates
):
    """Test the ``get_cs`` function without time dependencies.

    Have a look into the tests setup section to see which coordinate systems are
    defined in the CSM.

    Parameters
    ----------
    system_name : str
        Name of the system that should be returned
    reference_name : str
        Name of the reference system
    exp_orientation : List or numpy.ndarray
        The expected orientation of the returned system
    exp_coordinates
        The expected coordinates of the returned system

    """
    # setup
    csm = tf.CoordinateSystemManager(root_coordinate_system_name="root")
    csm.create_cs("lcs_1", "root", r_mat_z(0.5), Q_([1, 2, 3], "m"))
    csm.create_cs("lcs_2", "root", r_mat_y(0.5), Q_([3, -3, 1], "m"))
    csm.create_cs("lcs_3", "lcs_2", r_mat_x(0.5), Q_([1, -1, 3], "m"))

    check_coordinate_system(
        csm.get_cs(system_name, reference_name),
        exp_orientation,
        exp_coordinates,
        True,
    )


# test_get_local_coordinate_system_time_dep --------------------------------------------


@pytest.mark.parametrize(
    "function_arguments, time_refs, exp_orientation, exp_coordinates,"
    "exp_time_data, exp_failure",
    [
        # get cs in its parent system - no reference times
        (
            ("cs_1",),
            [None, None, None, None],
            [np.eye(3) for _ in range(3)],
            [[i, 0, 0] for i in [0, 0.25, 1]],
            ([0, 3, 12], None),
            False,
        ),
        # get cs in its parent system - only CSM has reference time
        (
            ("cs_1",),
            ["2000-03-03", None, None, None],
            [np.eye(3) for _ in range(3)],
            [[i, 0, 0] for i in [0, 0.25, 1]],
            ([0, 3, 12], "2000-03-03"),
            False,
        ),
        # get cs in its parent system - only system has reference time
        (
            ("cs_1",),
            [None, "2000-03-03", "2000-03-03", "2000-03-03"],
            [np.eye(3) for _ in range(3)],
            [[i, 0, 0] for i in [0, 0.25, 1]],
            ([0, 3, 12], "2000-03-03"),
            False,
        ),
        # get cs in its parent system - function and CSM have reference times
        (
            ("cs_1", None, pd.to_timedelta([6, 9, 18], "D"), "2000-03-10"),
            ["2000-03-16", None, None, None],
            [np.eye(3) for _ in range(3)],
            [[i, 0, 0] for i in [0, 0.25, 1]],
            ([6, 9, 18], "2000-03-10"),
            False,
        ),
        # get cs in its parent system - system and CSM have diff. reference times
        (
            ("cs_1",),
            ["2000-03-10", "2000-03-16", None, None],
            [np.eye(3) for _ in range(3)],
            [[i, 0, 0] for i in [0, 0.25, 1]],
            ([6, 9, 18], "2000-03-10"),
            False,
        ),
        # get transformed cs - no reference times
        (
            ("cs_3", "root"),
            [None, None, None, None],
            [np.eye(3) for _ in range(7)],
            [[1, 0, 0] for _ in range(7)],
            ([0, 3, 4, 6, 8, 9, 12], None),
            False,
        ),
        # get transformed cs - only CSM has reference time
        (
            ("cs_3", "root"),
            ["2000-03-10", None, None, None],
            [np.eye(3) for _ in range(7)],
            [[1, 0, 0] for _ in range(7)],
            ([0, 3, 4, 6, 8, 9, 12], "2000-03-10"),
            False,
        ),
        # get transformed cs - CSM and two systems have a reference time
        (
            ("cs_3", "root"),
            ["2000-03-10", "2000-03-04", None, "2000-03-16"],
            r_mat_x([0, 0, 0, 2 / 3, 1, 1, 1, 1, 0.5, 0]),
            [[i, 0, 0] for i in [1, 1.25, 1.5, 1.5, 1.5, 4 / 3, 1.25, 1, 1, 1]],
            ([-6, -3, 0, 4, 6, 8, 9, 12, 15, 18], "2000-03-10"),
            False,
        ),
        # get transformed cs - CSM and all systems have a reference time
        (
            ("cs_3", "root"),
            ["2000-03-08", "2000-03-04", "2000-03-10", "2000-03-16"],
            r_mat_x([0, 0, 0, 2 / 3, 1, 1, 1, 1, 0.5, 0]),
            [[i, 0, 0] for i in [1, 1.25, 1.5, 1.5, 1.5, 4 / 3, 1.25, 1, 1, 1]],
            ([-4, -1, 2, 6, 8, 10, 11, 14, 17, 20], "2000-03-08"),
            False,
        ),
        # get transformed cs - all systems have a reference time
        (
            ("cs_3", "root"),
            [None, "2000-03-04", "2000-03-10", "2000-03-16"],
            r_mat_x([0, 0, 0, 2 / 3, 1, 1, 1, 1, 0.5, 0]),
            [[i, 0, 0] for i in [1, 1.25, 1.5, 1.5, 1.5, 4 / 3, 1.25, 1, 1, 1]],
            ([0, 3, 6, 10, 12, 14, 15, 18, 21, 24], "2000-03-04"),
            False,
        ),
        # get transformed cs at specific times - all systems and CSM have a
        # reference time
        (
            ("cs_3", "root", pd.to_timedelta([-4, 8, 20], "D")),
            ["2000-03-08", "2000-03-04", "2000-03-10", "2000-03-16"],
            r_mat_x([0, 1, 0]),
            [[i, 0, 0] for i in [1, 1.5, 1]],
            ([-4, 8, 20], "2000-03-08"),
            False,
        ),
        # get transformed cs at specific times - some systems, CSM and function
        # have a reference time
        (
            ("cs_3", "root", pd.to_timedelta([-4, 8, 20], "D"), "2000-03-08"),
            ["2000-03-10", "2000-03-04", None, "2000-03-16"],
            r_mat_x([0, 1, 0]),
            [[i, 0, 0] for i in [1, 1.5, 1]],
            ([-4, 8, 20], "2000-03-08"),
            False,
        ),
        # get transformed cs at specific times - all systems, CSM and function
        # have a reference time
        (
            ("cs_3", "root", pd.to_timedelta([-4, 8, 20], "D"), "2000-03-08"),
            ["2000-03-02", "2000-03-04", "2000-03-10", "2000-03-16"],
            r_mat_x([0, 1, 0]),
            [[i, 0, 0] for i in [1, 1.5, 1]],
            ([-4, 8, 20], "2000-03-08"),
            False,
        ),
        # get transformed cs at specific times - all systems, and the function
        # have a reference time
        (
            ("cs_3", "root", pd.to_timedelta([-4, 8, 20], "D"), "2000-03-08"),
            [None, "2000-03-04", "2000-03-10", "2000-03-16"],
            r_mat_x([0, 1, 0]),
            [[i, 0, 0] for i in [1, 1.5, 1]],
            ([-4, 8, 20], "2000-03-08"),
            False,
        ),
        # get transformed cs at specific times - the function and the CSM have a
        # reference time
        (
            ("cs_4", "root", pd.to_timedelta([0, 6, 12, 18], "D"), "2000-03-08"),
            ["2000-03-14", None, None, None],
            r_mat_x([0, 0, 1, 2]),
            [[0, 1, 0], [0, 1, 0], [0, -1, 0], [0, 1, 0]],
            ([0, 6, 12, 18], "2000-03-08"),
            False,
        ),
        # get transformed cs at times of another system - no reference times
        (
            ("cs_3", "root", "cs_1"),
            [None, None, None, None],
            [np.eye(3) for _ in range(3)],
            [[1, 0, 0] for _ in range(3)],
            ([0, 3, 12], None),
            False,
        ),
        # get transformed cs at specific times - no reference times
        (
            ("cs_4", "root", pd.to_timedelta([0, 3, 6, 9, 12], "D")),
            [None, None, None, None],
            r_mat_x([0, 0.5, 1, 1.5, 2]),
            [[0, 1, 0], [0, 0, 1], [0, -1, 0], [0, 0, -1], [0, 1, 0]],
            ([0, 3, 6, 9, 12], None),
            False,
        ),
        # self referencing
        (
            ("cs_3", "cs_3"),
            [None, None, None, None],
            np.eye(3),
            [0, 0, 0],
            (None, None),
            False,
        ),
        # get transformed cs at specific times using a quantity - all systems,
        # CSM and function have a reference time
        (
            ("cs_3", "root", Q_([-4, 8, 20], "day"), "2000-03-08"),
            ["2000-03-02", "2000-03-04", "2000-03-10", "2000-03-16"],
            r_mat_x([0, 1, 0]),
            [[i, 0, 0] for i in [1, 1.5, 1]],
            ([-4, 8, 20], "2000-03-08"),
            False,
        ),
        # get transformed cs at specific times using a quantity - all systems and
        # CSM have a reference time
        (
            ("cs_3", "root", Q_([-4, 8, 20], "day")),
            ["2000-03-08", "2000-03-04", "2000-03-10", "2000-03-16"],
            r_mat_x([0, 1, 0]),
            [[i, 0, 0] for i in [1, 1.5, 1]],
            ([-4, 8, 20], "2000-03-08"),
            False,
        ),
        # get transformed cs at specific times using a list of timedelta strings
        # - all systems and CSM have a reference time
        (
            ("cs_3", "root", ["-4day", "8day", "20day"]),
            ["2000-03-08", "2000-03-04", "2000-03-10", "2000-03-16"],
            r_mat_x([0, 1, 0]),
            [[i, 0, 0] for i in [1, 1.5, 1]],
            ([-4, 8, 20], "2000-03-08"),
            False,
        ),
        # get transformed cs at a specific time using a timedelta string
        # - all systems and CSM have a reference time
        (
            ("cs_3", "root", "20day"),
            ["2000-03-08", "2000-03-04", "2000-03-10", "2000-03-16"],
            r_mat_x([0]),
            [[1, 0, 0]],
            (None, None),
            False,
        ),
        # get transformed cs at specific times using a DatetimeIndex - all systems,
        # CSM and function have a reference time
        (
            (
                "cs_3",
                "root",
                pd.DatetimeIndex(["2000-03-04", "2000-03-16", "2000-03-28"]),
                "2000-03-08",
            ),
            ["2000-03-02", "2000-03-04", "2000-03-10", "2000-03-16"],
            r_mat_x([0, 1, 0]),
            [[i, 0, 0] for i in [1, 1.5, 1]],
            ([-4, 8, 20], "2000-03-08"),
            False,
        ),
        # get transformed cs at specific times using a DatetimeIndex - all systems,
        # and the CSM have a reference time
        (
            (
                "cs_3",
                "root",
                pd.DatetimeIndex(["2000-03-04", "2000-03-16", "2000-03-28"]),
            ),
            ["2000-03-08", "2000-03-04", "2000-03-10", "2000-03-16"],
            r_mat_x([0, 1, 0]),
            [[i, 0, 0] for i in [1, 1.5, 1]],
            ([-4, 8, 20], "2000-03-08"),
            False,
        ),
        # get transformed cs at specific times using a list of date strings - all
        # systems and the CSM have a reference time
        (
            (
                "cs_3",
                "root",
                ["2000-03-04", "2000-03-16", "2000-03-28"],
            ),
            ["2000-03-08", "2000-03-04", "2000-03-10", "2000-03-16"],
            r_mat_x([0, 1, 0]),
            [[i, 0, 0] for i in [1, 1.5, 1]],
            ([-4, 8, 20], "2000-03-08"),
            False,
        ),
        # get transformed cs at a specific time using a date string - all
        # systems and the CSM have a reference time
        (
            ("cs_3", "root", "2000-03-04"),
            ["2000-03-08", "2000-03-04", "2000-03-10", "2000-03-16"],
            r_mat_x([0]),
            [[1, 0, 0]],
            (None, None),
            False,
        ),
        # get transformed cs at specific times using a DatetimeIndex - all systems
        # have a reference time - this is a special case since the internally used
        # mechanism will use the first value of the DatetimeIndex as reference
        # value. Using Quantities or a TimedeltaIndex will cause an exception since
        # the reference time of the time delta is undefined.
        (
            (
                "cs_3",
                "root",
                pd.DatetimeIndex(["2000-03-16", "2000-03-28", "2000-03-30"]),
            ),
            [None, "2000-03-04", "2000-03-10", "2000-03-16"],
            r_mat_x([1, 0, 0]),
            [[i, 0, 0] for i in [1.5, 1, 1]],
            ([0, 12, 14], "2000-03-16"),
            False,
        ),
        # should fail - if only the coordinate systems have a reference time,
        # passing just a time delta results in an undefined reference timestamp of
        # the resulting coordinate system
        (
            ("cs_3", "root", pd.to_timedelta([0, 8, 20], "D")),
            [None, "2000-03-04", "2000-03-10", "2000-03-16"],
            None,
            None,
            (None, None),
            TypeError,
        ),
        # should fail - if neither the CSM nor its attached coordinate systems have
        # a reference time, passing one to the function results in undefined
        # behavior
        (
            ("cs_3", "root", pd.to_timedelta([0, 8, 20], "D"), "2000-03-16"),
            [None, None, None, None],
            None,
            None,
            (None, None),
            TypeError,
        ),
    ],
)
def test_get_local_coordinate_system_time_dep(
    function_arguments,
    time_refs,
    exp_orientation,
    exp_coordinates,
    exp_time_data,
    exp_failure,
):
    """Test the ``get_cs`` function with time dependencies.

    The test setup is as follows:

    - 'cs_1' moves in 12 days 1 unit along the x-axis in positive direction.
      It starts at the origin and refers to the root system
    - 'cs_2' moves in 12 days 1 unit along the x-axis in negative direction.
      In the same time it positively rotates 360 degrees around the x-axis.
      It starts at the origin and refers to 'cs_1'
    - 'cs_3' rotates in 12 days 360 degrees negatively around the x-axis.
      It remains static at the coordinate [1, 0, 0] and refers to 'cs_2'
    - 'cs_4' remains static at the coordinates [0, 1, 0] of its reference system
      'cs_2'
    - initially and after their movements, all systems have the same orientation as
      their reference system

    In case all systems have the same reference time, the following behavior can be
    observed in the root system:

    - 'cs_1' moves as described before
    - 'cs_2' remains at the origin and rotates around the x-axis
    - 'cs_3' remains completely static at the coordinates [1, 0, 0]
    - 'cs_4' rotates around the x-axis with a fixed distance of 1 unit to the origin

    Have a look into the tests setup for further details.

    Parameters
    ----------
    function_arguments : Tuple
        A tuple of values that should be passed to the function
    time_refs : List(str)
        A list of date strings. The first entry is used as reference time of the
        CSM. The others are passed as reference times to the coordinate systems that
        have the same number as the list index in their name. For example: The
        second list value with index 1 belongs to 'cs_1'.
    exp_orientation : List or numpy.ndarray
        The expected orientation of the returned system
    exp_coordinates
        The expected coordinates of the returned system
    exp_time_data : Tuple(List(int), str)
        A tuple containing the expected time data of the returned coordinate system.
        The first value is a list of the expected time deltas and the second value
        is the expected reference time as date string.
    exp_failure : bool
        Set to `True` if the function call with the given parameters should raise an
        error

    """
    if exp_coordinates is not None:
        exp_coordinates = Q_(exp_coordinates, "mm")

    # setup -------------------------------------------
    # set reference times
    time_refs = [t if t is None else pd.Timestamp(t) for t in time_refs]

    # moves in positive x-direction
    time_1 = pd.to_timedelta([0, 3, 12], "D")
    time_ref_1 = time_refs[1]
    orientation_1 = None
    coordinates_1 = Q_([[i, 0, 0] for i in [0, 0.25, 1]], "mm")

    # moves in negative x-direction and rotates positively around the x-axis
    time_2 = pd.to_timedelta([0, 4, 8, 12], "D")
    time_ref_2 = time_refs[2]
    coordinates_2 = Q_([[-i, 0, 0] for i in [0, 1 / 3, 2 / 3, 1]], "mm")
    orientation_2 = r_mat_x([0, 2 / 3, 4 / 3, 2])

    # rotates negatively around the x-axis
    time_3 = pd.to_timedelta([0, 3, 6, 9, 12], "D")
    time_ref_3 = time_refs[3]
    coordinates_3 = Q_([1, 0, 0], "mm")
    orientation_3 = r_mat_x([0, -0.5, -1, -1.5, -2])

    # static
    time_4 = None
    time_ref_4 = None
    orientation_4 = None
    coordinates_4 = Q_([0, 1, 0], "mm")

    csm = tf.CoordinateSystemManager("root", "CSM", time_refs[0])
    csm.create_cs("cs_1", "root", orientation_1, coordinates_1, time_1, time_ref_1)
    csm.create_cs("cs_2", "cs_1", orientation_2, coordinates_2, time_2, time_ref_2)
    csm.create_cs("cs_3", "cs_2", orientation_3, coordinates_3, time_3, time_ref_3)
    csm.create_cs("cs_4", "cs_2", orientation_4, coordinates_4, time_4, time_ref_4)

    if not exp_failure:
        # create expected time data
        exp_time = exp_time_data[0]
        if exp_time is not None:
            exp_time = pd.to_timedelta(exp_time, "D")
        exp_time_ref = exp_time_data[1]
        if exp_time_ref is not None:
            exp_time_ref = pd.Timestamp(exp_time_ref)

        check_coordinate_system(
            csm.get_cs(*function_arguments),
            exp_orientation,
            exp_coordinates,
            True,
            exp_time,
            exp_time_ref,
        )
    else:
        with pytest.raises(exp_failure):
            csm.get_cs(*function_arguments)


# test_get_local_coordinate_system_timeseries ------------------------------------------


@pytest.mark.parametrize(
    "lcs, in_lcs, exp_coords, exp_time, exp_angles",
    [
        ("r", "ts", [[0, -1, -1], [0, -2, -1]], [1, 2], [0, -90]),
        ("ts", "r", [[0, 1, 1], [-2, 0, 1]], [1, 2], [0, 90]),
        ("s", "trl", [[0, 0, 0], [-1, 0, 0]], [2, 3], [0, 0]),
        ("trl", "s", [[0, 0, 0], [1, 0, 0]], [2, 3], [0, 0]),
        ("s", "r", [[1, 1, 1], [-2, 1, 1]], [1, 2], [0, 90]),
        ("r", "s", [[-1, -1, -1], [-1, -2, -1]], [1, 2], [0, -90]),
        ("trl", "r", [[1, 1, 1], [-2, 1, 1], [-3, 2, 1]], [1, 2, 3], [0, 90, 90]),
    ],
)
@pytest.mark.parametrize("units", ["inch", "mm"])
def test_get_local_coordinate_system_timeseries(
    lcs, in_lcs, exp_coords, exp_time, exp_angles, units
):
    """Test the get_cs method with one lcs having a `TimeSeries` as coordinates.

    Parameters
    ----------
    lcs :
        The lcs that should be transformed
    in_lcs :
        The target lcs
    exp_coords :
        Expected coordinates
    exp_time :
        The expected time (in seconds)
    exp_angles :
        The expected rotation angles around the z-axis

    """
    me_units = f"{units}/s" if units else "1/s"
    me = MathematicalExpression("a*t", {"a": Q_([0, 1, 0], me_units)})
    ts = TimeSeries(me)
    rotation = WXRotation.from_euler("z", [0, 90], degrees=True).as_matrix()
    translation = Q_([[1, 0, 0], [2, 0, 0]], units)
    exp_orient = WXRotation.from_euler("z", exp_angles, degrees=True).as_matrix()
    coords_1 = [0, 0, 1]
    coords_2 = [1, 0, 0]

    translation = Q_(translation, units)
    exp_coords = Q_(exp_coords, units)
    coords_1 = Q_(coords_1, units)
    coords_2 = Q_(coords_2, units)

    csm = CSM("r")
    csm.create_cs("rot", "r", rotation, coords_1, time=Q_([1, 2], "s"))
    csm.create_cs("ts", "rot", coordinates=ts)
    csm.create_cs("s", "ts", coordinates=coords_2)
    csm.create_cs("trl", "ts", coordinates=translation, time=Q_([2, 3], "s"))

    result = csm.get_cs(lcs, in_lcs)
    assert np.allclose(result.orientation, exp_orient)
    assert np.allclose(result.coordinates.data, exp_coords)
    assert np.allclose(result.time.as_quantity().m, exp_time)


# test_get_local_coordinate_system_exceptions ------------------------------------------


@pytest.mark.parametrize(
    "function_arguments, exception_type, test_name",
    [
        (("not there", "root"), ValueError, "# system does not exist"),
        (("root", "not there"), ValueError, "# reference system does not exist"),
        (("not there", "not there"), ValueError, "# both systems do not exist"),
        (("not there", None), ValueError, "# root system has no reference"),
        (("cs_4", "root", "not there"), ValueError, "# ref. system does not exist"),
        (("cs_4", "root", "cs_1"), ValueError, "# ref. system is not time dep."),
        (("cs_4", "root", 1), TypeError, "# Invalid time type #1"),
        (("cs_4", "root", ["grr", "4", "af"]), Exception, "# Invalid time type #2"),
    ],
    ids=get_test_name,
)
def test_get_local_coordinate_system_exceptions(
    function_arguments, exception_type, test_name
):
    """Test the exceptions of the ``get_cs`` function.

    Parameters
    ----------
    function_arguments : Tuple
        A tuple of values that should be passed to the function
    exception_type :
        Expected exception type
    test_name : str
        Name of the testcase

    """
    # setup
    time_1 = pd.to_timedelta([0, 3], "D")
    time_2 = pd.to_timedelta([4, 7], "D")

    csm = tf.CoordinateSystemManager(root_coordinate_system_name="root")
    csm.create_cs("cs_1", "root", r_mat_z(0.5), Q_([1, 2, 3], "mm"))
    csm.create_cs("cs_2", "root", r_mat_y(0.5), Q_([3, -3, 1], "mm"))
    csm.create_cs("cs_3", "cs_2", r_mat_x(0.5), Q_([1, -1, 3], "mm"))
    csm.create_cs("cs_4", "cs_2", r_mat_x([0, 1]), Q_([2, -1, 2], "mm"), time=time_1)
    csm.create_cs("cs_5", "root", r_mat_y([0, 1]), Q_([1, -7, 3], "mm"), time=time_2)

    # test
    with pytest.raises(exception_type):
        csm.get_cs(*function_arguments)


# test_get_cs_exception_timeseries -----------------------------------------------------


@pytest.mark.parametrize(
    "lcs, in_lcs, exp_exception",
    [
        ("trl1", "ts", WeldxException),
        ("ts", "trl1", False),
        ("s", "trl1", WeldxException),
        ("trl1", "s", WeldxException),
        ("trl1", "trl2", False),
        ("trl2", "trl1", False),
        ("r", "trl2", False),
        ("trl2", "r", False),
        ("s", "r", False),
        ("r", "s", False),
    ],
)
@pytest.mark.parametrize("units", ["inch", "mm"])
def test_get_cs_exception_timeseries(lcs, in_lcs, exp_exception, units):
    """Test exceptions of get_cs method if 1 lcs has a `TimeSeries` as coordinates.

    Parameters
    ----------
    lcs :
        The lcs that should be transformed
    in_lcs :
        The target lcs
    exp_exception :
        Set to `True` if the transformation should raise
    units :
        The length unit that should be used

    """
    me_units = f"{units}/s" if units else "1/s"
    me = MathematicalExpression("a*t", {"a": Q_([0, 1, 0], me_units)})
    ts = TimeSeries(me)
    translation = Q_([[1, 0, 0], [2, 0, 0]], units)
    coords_1 = Q_([1, 0, 0], units)

    csm = CSM("r")
    csm.create_cs("trl1", "r", coordinates=translation, time=Q_([1, 2], "s"))
    csm.create_cs("ts", "trl1", coordinates=ts)
    csm.create_cs("s", "ts", coordinates=coords_1)
    csm.create_cs("trl2", "ts", coordinates=translation, time=Q_([2, 3], "s"))
    if exp_exception:
        with pytest.raises(exp_exception):
            csm.get_cs(lcs, in_lcs)
    else:
        csm.get_cs(lcs, in_lcs)


# test_merge ---------------------------------------------------------------------------


@pytest.mark.parametrize("nested", [(True,), (False,)])
def test_merge(list_of_csm_and_lcs_instances, nested):
    """Test the merge function."""
    # setup -------------------------------------------
    csm = list_of_csm_and_lcs_instances[0]
    lcs = list_of_csm_and_lcs_instances[1]

    # merge -------------------------------------------
    csm_mg = deepcopy(csm[0])

    if nested:
        csm_n3 = deepcopy(csm[3])
        csm_n3.merge(csm[5])
        csm_n2 = deepcopy(csm[2])
        csm_n2.merge(csm_n3)
        csm_mg.merge(csm[1])
        csm_mg.merge(csm[4])
        csm_mg.merge(csm_n2)
    else:
        csm_mg.merge(csm[1])
        csm_mg.merge(csm[2])
        csm_mg.merge(csm[3])
        csm_mg.merge(csm[4])
        csm_mg.merge(csm[5])

    # check merge results -----------------------------
    csm_0_systems = csm_mg.coordinate_system_names
    assert np.all([f"lcs{i}" in csm_0_systems for i in range(len(lcs))])

    for i, cur_lcs in enumerate(lcs):
        child = f"lcs{i}"
        parent = csm_mg.get_parent_system_name(child)
        if i == 0:
            assert parent is None
            continue
        assert csm_mg.get_cs(child, parent) == cur_lcs
        assert csm_mg.get_cs(parent, child) == cur_lcs.invert()


# test_merge_reference_times -----------------------------------------------------------


@pytest.mark.parametrize(
    "time_ref_day_parent, time_ref_day_sub, is_static_parent, is_static_sub,"
    "should_fail",
    [
        # both static
        (None, None, True, True, False),
        ("01", None, True, True, False),
        ("01", "01", True, True, False),
        ("01", "03", True, True, False),
        (None, "01", True, True, False),
        # sub static
        (None, None, False, True, False),
        ("01", None, False, True, False),
        ("01", "01", False, True, False),
        ("01", "03", False, True, False),
        (None, "01", False, True, False),
        # parent static
        (None, None, True, False, False),
        ("01", None, True, False, False),
        ("01", "01", True, False, False),
        ("01", "03", True, False, True),
        (None, "01", True, False, True),
        # both dynamic
        (None, None, False, False, False),
        ("01", None, False, False, False),
        ("01", "01", False, False, False),
        ("01", "03", False, False, True),
        (None, "01", False, False, True),
    ],
)
def test_merge_reference_times(
    time_ref_day_parent,
    time_ref_day_sub,
    is_static_parent,
    is_static_sub,
    should_fail,
):
    """Test if ``merge`` raises an error for invalid reference time combinations.

    Parameters
    ----------
    time_ref_day_parent : str
        `None` or day number of the parent systems reference timestamp
    time_ref_day_sub : str
        `None` or day number of the merged systems reference timestamp
    is_static_parent : bool
        `True` if the parent system should be static, `False` otherwise
    is_static_sub : bool
        `True` if the merged system should be static, `False` otherwise
    should_fail : bool
        `True` if the merge operation should fail. `False` otherwise

    """
    # setup
    lcs_static = tf.LocalCoordinateSystem(coordinates=Q_([1, 1, 1], "mm"))
    lcs_dynamic = tf.LocalCoordinateSystem(
        coordinates=Q_([[0, 4, 2], [7, 2, 4]], "mm"), time=pd.to_timedelta([4, 8], "D")
    )
    time_ref_parent = None
    if time_ref_day_parent is not None:
        time_ref_parent = f"2000-01-{time_ref_day_parent}"
    csm_parent = tf.CoordinateSystemManager(
        "root", "csm_parent", time_ref=time_ref_parent
    )
    if is_static_parent:
        csm_parent.add_cs("cs_1", "root", lcs_static)
    else:
        csm_parent.add_cs("cs_1", "root", lcs_dynamic)

    time_ref_sub = None
    if time_ref_day_sub is not None:
        time_ref_sub = f"2000-01-{time_ref_day_sub}"
    csm_sub = tf.CoordinateSystemManager("base", "csm_sub", time_ref=time_ref_sub)
    if is_static_sub:
        csm_sub.add_cs("cs_1", "base", lcs_static)
    else:
        csm_sub.add_cs("cs_1", "base", lcs_dynamic)

    # test
    if should_fail:
        with pytest.raises(ValueError):
            csm_parent.merge(csm_sub)
    else:
        csm_parent.merge(csm_sub)


# test get_subsystems_merged_serially --------------------------------------------------


def test_get_subsystems_merged_serially(list_of_csm_and_lcs_instances):
    """Test the get_subsystem method.

    In this test case, all sub systems are merged into the same target system.
    """
    # setup -------------------------------------------
    csm = list_of_csm_and_lcs_instances[0]

    csm[0].merge(csm[1])
    csm[0].merge(csm[2])
    csm[0].merge(csm[3])
    csm[0].merge(csm[4])
    csm[0].merge(csm[5])

    # get subsystems ----------------------------------
    subs = csm[0].subsystems

    # checks ------------------------------------------
    assert len(subs) == 5

    assert subs[0] == csm[1]
    assert subs[1] == csm[2]
    assert subs[2] == csm[3]
    assert subs[3] == csm[4]
    assert subs[4] == csm[5]


# test get_subsystems_merged_nested ----------------------------------------------------


def test_get_subsystems_merged_nested(list_of_csm_and_lcs_instances):
    """Test the get_subsystem method.

    In this test case, several systems are merged together before they are merged
    to the target system. This creates a nested subsystem structure.
    """
    # setup -------------------------------------------
    csm = list_of_csm_and_lcs_instances[0]

    csm_n3 = deepcopy(csm[3])
    csm_n3.merge(csm[5])

    csm_n2 = deepcopy(csm[2])
    csm_n2.merge(csm_n3)

    csm_mg = deepcopy(csm[0])
    csm_mg.merge(csm[1])
    csm_mg.merge(csm[4])
    csm_mg.merge(csm_n2)

    # get sub systems ---------------------------------
    subs = csm_mg.subsystems

    # checks ------------------------------------------
    assert len(subs) == 3

    assert subs[0] == csm[1]
    assert subs[1] == csm[4]
    assert subs[2] == csm_n2

    # get sub sub system ------------------------------
    sub_subs = subs[2].subsystems

    # check -------------------------------------------
    assert len(sub_subs) == 1

    assert sub_subs[0] == csm_n3

    # get sub sub sub systems -------------------------
    sub_sub_subs = sub_subs[0].subsystems

    # check -------------------------------------------
    assert len(sub_sub_subs) == 1

    assert sub_sub_subs[0] == csm[5]


# test_remove_subsystems ---------------------------------------------------------------


@pytest.mark.parametrize("nested", [(True,), (False,)])
def test_remove_subsystems(list_of_csm_and_lcs_instances, nested):
    """Test the remove_subsystem function."""
    # setup -------------------------------------------
    csm = list_of_csm_and_lcs_instances[0]

    csm_mg = deepcopy(csm[0])

    if nested:
        csm_n3 = deepcopy(csm[3])
        csm_n3.merge(csm[5])
        csm_n2 = deepcopy(csm[2])
        csm_n2.merge(csm_n3)
        csm_mg.merge(csm[1])
        csm_mg.merge(csm[4])
        csm_mg.merge(csm_n2)
    else:
        csm_mg.merge(csm[1])
        csm_mg.merge(csm[2])
        csm_mg.merge(csm[3])
        csm_mg.merge(csm[4])
        csm_mg.merge(csm[5])

    # remove subsystems -------------------------------
    csm_mg.remove_subsystems()

    # check -------------------------------------------
    assert csm_mg == csm[0]


# test_unmerge_merged_serially ---------------------------------------------------------


@pytest.mark.parametrize(
    "additional_cs",
    [
        ({}),
        ({"lcs0": 0}),
        ({"lcs1": 0}),
        ({"lcs2": 0}),
        ({"lcs3": 0}),
        ({"lcs4": 1}),
        ({"lcs5": 2}),
        ({"lcs6": 2}),
        ({"lcs7": 3}),
        ({"lcs8": 3}),
        ({"lcs9": 4}),
        ({"lcs10": 5}),
        ({"lcs2": 0, "lcs5": 2, "lcs7": 3, "lcs8": 3}),
        ({"lcs0": 0, "lcs3": 0, "lcs4": 1, "lcs6": 2, "lcs10": 5}),
    ],
)
@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:The following coordinate systems.*:UserWarning")
def test_unmerge_merged_serially(list_of_csm_and_lcs_instances, additional_cs):
    """Test the CSM unmerge function.

    In this test case, all sub systems are merged into the same target system.
    """
    # setup -------------------------------------------
    csm = deepcopy(list_of_csm_and_lcs_instances[0])

    csm_mg = deepcopy(csm[0])

    csm_mg.merge(csm[1])
    csm_mg.merge(csm[2])
    csm_mg.merge(csm[3])
    csm_mg.merge(csm[4])
    csm_mg.merge(csm[5])

    count = 0
    for parent_cs, target_csm in additional_cs.items():
        lcs = LCS(coordinates=Q_([count, count + 1, count + 2], "mm"))
        csm_mg.add_cs(f"additional_{count}", parent_cs, lcs)
        csm[target_csm].add_cs(f"additional_{count}", parent_cs, lcs)
        count += 1

    # unmerge -----------------------------------------
    subs = csm_mg.unmerge()

    # checks ------------------------------------------
    csm_res = [csm_mg] + subs
    assert len(csm_res) == 6

    for i, current_lcs in enumerate(csm_res):
        assert csm_res[i] == current_lcs


# test_unmerge_merged_nested -----------------------------------------------------------


@pytest.mark.parametrize(
    "additional_cs",
    [
        ({}),
        ({"lcs0": 0}),
        ({"lcs1": 0}),
        ({"lcs2": 0}),
        ({"lcs3": 0}),
        ({"lcs4": 1}),
        ({"lcs5": 2}),
        ({"lcs6": 2}),
        ({"lcs7": 3}),
        ({"lcs8": 3}),
        ({"lcs9": 4}),
        ({"lcs10": 5}),
        ({"lcs2": 0, "lcs5": 2, "lcs7": 3, "lcs8": 3}),
        ({"lcs0": 0, "lcs3": 0, "lcs4": 1, "lcs6": 2, "lcs10": 5}),
    ],
)
@pytest.mark.slow
def test_unmerge_merged_nested(list_of_csm_and_lcs_instances, additional_cs):
    """Test the CSM unmerge function.

    In this test case, several systems are merged together before they are merged
    to the target system. This creates a nested subsystem structure.
    """
    # setup -------------------------------------------
    csm = deepcopy(list_of_csm_and_lcs_instances[0])

    csm_mg = deepcopy(csm[0])

    csm_n3 = deepcopy(csm[3])
    csm_n3.merge(csm[5])
    csm_n2 = deepcopy(csm[2])
    csm_n2.merge(csm_n3)
    csm_mg.merge(csm[1])
    csm_mg.merge(csm[4])
    csm_mg.merge(csm_n2)

    count = 0
    exp_orphan_node_warning = False
    for parent_cs, target_csm in additional_cs.items():
        lcs = LCS(coordinates=Q_([count, count + 1, count + 2], "mm"))
        csm_mg.add_cs(f"additional_{count}", parent_cs, lcs)
        if target_csm == 0:
            csm[target_csm].add_cs(f"additional_{count}", parent_cs, lcs)
        else:
            exp_orphan_node_warning = True
        count += 1

    # unmerge -----------------------------------------
    if exp_orphan_node_warning:
        with pytest.warns(UserWarning):
            subs = csm_mg.unmerge()
    else:
        subs = csm_mg.unmerge()

    # checks ------------------------------------------
    assert len(subs) == 3

    assert csm_mg == csm[0]
    assert subs[0] == csm[1]
    assert subs[1] == csm[4]
    assert subs[2] == csm_n2

    # unmerge sub -------------------------------------
    sub_subs = subs[2].unmerge()

    # checks ------------------------------------------
    assert len(sub_subs) == 1

    assert subs[2] == csm[2]
    assert sub_subs[0] == csm_n3

    # unmerge sub sub ---------------------------------
    sub_sub_subs = sub_subs[0].unmerge()

    # checks ------------------------------------------
    assert len(sub_sub_subs) == 1

    assert sub_subs[0] == csm[3]
    assert sub_sub_subs[0] == csm[5]


# test_delete_cs_with_serially_merged_subsystems ---------------------------------------


@pytest.mark.parametrize(
    "name, subsystems_exp, num_cs_exp",
    [
        ("lcs1", ["csm1", "csm2", "csm3", "csm4", "csm5"], 15),
        ("lcs2", ["csm1"], 4),
        ("lcs3", ["csm1"], 6),
        ("lcs4", ["csm2", "csm3", "csm4", "csm5"], 15),
        ("lcs5", ["csm1", "csm4"], 8),
        ("lcs6", ["csm1", "csm4"], 10),
        ("lcs7", ["csm1", "csm2", "csm4"], 12),
        ("lcs8", ["csm1", "csm2", "csm4", "csm5"], 15),
        ("lcs9", ["csm1", "csm2", "csm3", "csm5"], 15),
        ("lcs10", ["csm1", "csm2", "csm3", "csm4"], 14),
        ("add0", ["csm1", "csm2", "csm3", "csm4", "csm5"], 15),
        ("add1", ["csm1", "csm2", "csm3", "csm4", "csm5"], 15),
        ("add2", ["csm1", "csm2", "csm3", "csm4", "csm5"], 15),
        ("add3", ["csm1", "csm2", "csm3", "csm4", "csm5"], 15),
        ("add4", ["csm1", "csm2", "csm3", "csm4", "csm5"], 15),
    ],
)
def test_delete_cs_with_serially_merged_subsystems(
    list_of_csm_and_lcs_instances, name, subsystems_exp, num_cs_exp
):
    """Test the delete_cs function with subsystems that were merged serially."""
    # setup -------------------------------------------
    csm = deepcopy(list_of_csm_and_lcs_instances[0])

    csm_mg = deepcopy(csm[0])

    csm_mg.merge(csm[1])
    csm_mg.merge(csm[2])
    csm_mg.merge(csm[3])
    csm_mg.merge(csm[4])
    csm_mg.merge(csm[5])

    # add some additional coordinate systems
    target_system_index = [0, 2, 5, 7, 10]
    for i, _ in enumerate(target_system_index):
        lcs = LCS(coordinates=Q_([i, 2 * i, -i], "mm"))
        csm_mg.add_cs(f"add{i}", f"lcs{target_system_index[i]}", lcs)

    # just to avoid useless tests (delete does nothing if the lcs doesn't exist)
    assert name in csm_mg.coordinate_system_names

    # delete coordinate system ------------------------
    csm_mg.delete_cs(name, True)

    # check -------------------------------------------
    assert csm_mg.number_of_subsystems == len(subsystems_exp)
    assert csm_mg.number_of_coordinate_systems == num_cs_exp

    for sub_exp in subsystems_exp:
        assert sub_exp in csm_mg.subsystem_names


# test_delete_cs_with_nested_subsystems ------------------------------------------------


@pytest.mark.parametrize(
    "name, subsystems_exp, num_cs_exp",
    [
        ("lcs1", ["csm1", "csm2", "csm4"], 17),
        ("lcs2", ["csm1"], 4),
        ("lcs3", ["csm1"], 6),
        ("lcs4", ["csm2", "csm4"], 17),
        ("lcs5", ["csm1", "csm4"], 8),
        ("lcs6", ["csm1", "csm4"], 11),
        ("lcs7", ["csm1", "csm4"], 14),
        ("lcs8", ["csm1", "csm4"], 16),
        ("lcs9", ["csm1", "csm2"], 17),
        ("lcs10", ["csm1", "csm4"], 16),
        ("add0", ["csm1", "csm2", "csm4"], 17),
        ("add1", ["csm1", "csm2", "csm4"], 17),
        ("add2", ["csm1", "csm2", "csm4"], 17),
        ("add3", ["csm1", "csm2", "csm4"], 17),
        ("add4", ["csm1", "csm2", "csm4"], 17),
        ("nes0", ["csm1", "csm4"], 17),
        ("nes1", ["csm1", "csm4"], 17),
    ],
)
def test_delete_cs_with_nested_subsystems(
    list_of_csm_and_lcs_instances, name, subsystems_exp, num_cs_exp
):
    """Test the delete_cs function with nested subsystems."""
    # setup -------------------------------------------
    csm = deepcopy(list_of_csm_and_lcs_instances[0])

    csm_mg = deepcopy(csm[0])

    csm_n3 = deepcopy(csm[3])
    csm_n3.add_cs("nes0", "lcs8", LCS(coordinates=Q_([1, 2, 3], "mm")))
    csm_n3.merge(csm[5])
    csm_n2 = deepcopy(csm[2])
    csm_n2.add_cs("nes1", "lcs5", LCS(coordinates=Q_([-1, -2, -3], "mm")))
    csm_n2.merge(csm_n3)
    csm_mg.merge(csm[1])
    csm_mg.merge(csm[4])
    csm_mg.merge(csm_n2)

    # add some additional coordinate systems
    target_system_indices = [0, 2, 5, 7, 10]
    for i, target_system_index in enumerate(target_system_indices):
        lcs = LCS(coordinates=Q_([i, 2 * i, -i], "mm"))
        csm_mg.add_cs(f"add{i}", f"lcs{target_system_index}", lcs)

    # just to avoid useless tests (delete does nothing if the lcs doesn't exist)
    assert name in csm_mg.coordinate_system_names

    # delete coordinate system ------------------------
    csm_mg.delete_cs(name, True)

    # check -------------------------------------------
    assert csm_mg.number_of_subsystems == len(subsystems_exp)
    assert csm_mg.number_of_coordinate_systems == num_cs_exp

    for sub_exp in subsystems_exp:
        assert sub_exp in csm_mg.subsystem_names


# test_plot ----------------------------------------------------------------------------


def test_plot():
    """Test if the plot function runs without problems. Output is not checked."""
    csm_global = CSM("root", "global coordinate systems")
    csm_global.create_cs("specimen", "root", coordinates=Q_([1, 2, 3], "m"))
    csm_global.create_cs("robot head", "root", coordinates=Q_([4, 5, 6], "m"))

    csm_specimen = CSM("specimen", "specimen coordinate systems")
    csm_specimen.create_cs("thermocouple 1", "specimen", coordinates=Q_([1, 1, 0], "m"))
    csm_specimen.create_cs("thermocouple 2", "specimen", coordinates=Q_([1, 4, 0], "m"))

    csm_robot = CSM("robot head", "robot coordinate systems")
    csm_robot.create_cs("torch", "robot head", coordinates=Q_([0, 0, -2], "m"))
    csm_robot.create_cs("mount point 1", "robot head", coordinates=Q_([0, 1, -1], "m"))
    csm_robot.create_cs("mount point 2", "robot head", coordinates=Q_([0, -1, -1], "m"))

    csm_scanner = CSM("scanner", "scanner coordinate systems")
    csm_scanner.create_cs("mount point 1", "scanner", coordinates=Q_([0, 0, 2], "m"))

    csm_robot.merge(csm_scanner)
    csm_global.merge(csm_robot)
    csm_global.merge(csm_specimen)

    csm_global.plot_graph()


# test_assign_and_get_data -------------------------------------------------------------


def setup_csm_test_assign_data() -> tf.CoordinateSystemManager:
    """Get a predefined CSM instance.

    Returns
    -------
    weldx.transformations.CoordinateSystemManager :
        Predefined CSM instance.

    """
    # test setup
    lcs1_in_root = tf.LocalCoordinateSystem(
        tf.WXRotation.from_euler("z", np.pi / 2).as_matrix(), Q_([1, 2, 3], "mm")
    )
    lcs2_in_root = tf.LocalCoordinateSystem(r_mat_y(0.5), Q_([3, -3, 1], "mm"))
    lcs3_in_lcs2 = tf.LocalCoordinateSystem(r_mat_x(0.5), Q_([1, -1, 3], "mm"))

    csm = tf.CoordinateSystemManager(root_coordinate_system_name="root")
    csm.add_cs("lcs_1", "root", lcs1_in_root)
    csm.add_cs("lcs_2", "root", lcs2_in_root)
    csm.add_cs("lcs_3", "lcs_2", lcs3_in_lcs2)

    return csm


@pytest.mark.parametrize(
    "lcs_ref, data_name, data, lcs_out, exp",
    [
        (
            "lcs_3",
            "my_data",
            [[1, -3, -1], [2, 4, -1], [-1, 2, 3], [3, -4, 2]],
            None,
            [[1, -3, -1], [2, 4, -1], [-1, 2, 3], [3, -4, 2]],
        ),
        (
            "lcs_3",
            "my_data",
            [[1, -3, -1], [2, 4, -1], [-1, 2, 3], [3, -4, 2]],
            "lcs_3",
            [[1, -3, -1], [2, 4, -1], [-1, 2, 3], [3, -4, 2]],
        ),
        (
            "lcs_3",
            "my_data",
            Q_([[1, -3, -1], [2, 4, -1], [-1, 2, 3], [3, -4, 2]], "mm"),
            "lcs_1",
            Q_([[-5, -2, -4], [-5, -9, -5], [-9, -7, -2], [-8, -1, -6]], "mm"),
        ),
        (
            "lcs_3",
            "my_data",
            SpatialData(Q_([[1, -3, -1], [2, 4, -1], [-1, 2, 3], [3, -4, 2]], "mm")),
            "lcs_1",
            Q_([[-5, -2, -4], [-5, -9, -5], [-9, -7, -2], [-8, -1, -6]], "mm"),
        ),
        (
            "lcs_3",
            "my_data",
            xr.DataArray(
                data=Q_([[1, -3, -1], [2, 4, -1], [-1, 2, 3], [3, -4, 2]], "mm"),
                dims=["n", "c"],
                coords={"c": ["x", "y", "z"]},
            ),
            "lcs_1",
            Q_([[-5, -2, -4], [-5, -9, -5], [-9, -7, -2], [-8, -1, -6]], "mm"),
        ),
    ],
)
def test_data_functions(lcs_ref, data_name, data, lcs_out, exp):
    """Test the `assign_data`, `has_data` and `get_data` functions.

    Parameters
    ----------
    lcs_ref : str
        Name of the data's reference system
    data_name : str
        Name of the data
    data :
        The data that should be assigned
    lcs_out : str
        Name of the target coordinate system
    exp: List[List[float]]
        Expected return values

    """
    csm = setup_csm_test_assign_data()

    assert csm.has_data(lcs_ref, data_name) is False
    assert data_name not in csm.data_names

    csm.assign_data(data, data_name, lcs_ref)

    assert data_name in csm.data_names
    assert csm.get_data_system_name(data_name) == lcs_ref
    for lcs in csm.coordinate_system_names:
        assert csm.has_data(lcs, data_name) == (lcs == lcs_ref)

    transformed_data = csm.get_data(data_name, lcs_out)
    if isinstance(transformed_data, SpatialData):
        transformed_data = transformed_data.coordinates.data
    else:
        transformed_data = transformed_data.data

    assert matrix_is_close(transformed_data, exp)


# test_assign_data_exceptions ----------------------------------------------------------


@pytest.mark.parametrize(
    "arguments, exception_type, test_name",
    [
        (([[1, 2, 3]], {"wrong"}, "root"), TypeError, "# invalid data name"),
        (([[1, 2, 3]], "data", "not there"), ValueError, "# system does not exist"),
        (([[1, 2, 3]], "some data", "root"), ValueError, "# name already taken 1"),
        (([[1, 2, 3]], "some data", "lcs_1"), ValueError, "# name already taken 2"),
    ],
)
def test_assign_data_exceptions(arguments, exception_type, test_name):
    """Test exceptions of the `assign_data` method.

    Parameters
    ----------
    arguments : Tuple
        A tuple of arguments that are passed to the function
    exception_type :
        The expected exception type
    test_name : str
        A string starting with an `#` that describes the test.

    """
    csm = setup_csm_test_assign_data()
    csm.assign_data([[1, 2, 3], [3, 2, 1]], "some data", "root")
    with pytest.raises(exception_type):
        csm.assign_data(*arguments)


def test_delete_data(list_of_csm_and_lcs_instances):
    """Test delete data (with subsystems)."""
    csm = list_of_csm_and_lcs_instances[0]

    data_name = "foo"
    csm[0].assign_data([[1, 2, 3], [3, 2, 1]], data_name, "lcs0")
    assert data_name in csm[0].data_names

    csm_n3 = deepcopy(csm[3])
    csm_n3.merge(csm[5])

    csm_n2 = deepcopy(csm[2])
    csm_n2.merge(csm_n3)

    csm_mg = deepcopy(csm[0])
    csm_mg.merge(csm[1])
    csm_mg.merge(csm[4])
    csm_mg.merge(csm_n2)

    csm[0].delete_data(data_name)
    csm_mg.delete_data(data_name)
    assert data_name not in csm[0].data_names
    for sub_sys in csm_mg.subsystems:
        assert data_name not in sub_sys.data_names


def test_delete_non_existent_data():
    """Ensure we receive an exception upon deleting non-existent data."""
    csm = tf.CoordinateSystemManager("root")
    with pytest.raises(ValueError):
        csm.delete_data("no")


# test_has_data_exceptions -------------------------------------------------------------


@pytest.mark.parametrize(
    "arguments, exception_type, test_name",
    [
        (("wrong", "not there"), KeyError, "# system does not exist"),
    ],
)
def test_has_data_exceptions(arguments, exception_type, test_name):
    """Test exceptions of the `has_data` method.

    Parameters
    ----------
    arguments : Tuple
        A tuple of arguments that are passed to the function
    exception_type :
        The expected exception type
    test_name : str
        A string starting with an `#` that describes the test.

    """
    csm = setup_csm_test_assign_data()
    csm.assign_data([[1, 2, 3], [3, 2, 1]], "some_data", "root")
    with pytest.raises(exception_type):
        csm.has_data(*arguments)


# test_get_data_exceptions -------------------------------------------------------------


@pytest.mark.parametrize(
    "arguments, exception_type, test_name",
    [
        (("some data", "not there"), ValueError, "# system does not exist"),
        (("not there", "root"), KeyError, "# data does not exist"),
    ],
)
def test_get_data_exceptions(arguments, exception_type, test_name):
    """Test exceptions of the `get_data` method.

    Parameters
    ----------
    arguments : Tuple
        A tuple of arguments that are passed to the function
    exception_type :
        The expected exception type
    test_name : str
        A string starting with an `#` that describes the test.

    """
    csm = setup_csm_test_assign_data()
    csm.assign_data([[1, 2, 3], [3, 2, 1]], "some data", "root")
    with pytest.raises(exception_type):
        csm.get_data(*arguments)


# test_merge_unmerge_with_data ---------------------------------------------------------


@pytest.mark.parametrize("node_parent", ["a", "b"])
@pytest.mark.parametrize("node_child", ["x", "y"])
@pytest.mark.parametrize("data_parent", [True, False])
@pytest.mark.parametrize("data_index", [0, 1])
def test_merge_unmerge_with_data(node_parent, node_child, data_parent, data_index):
    """Test if assigned data is treated correctly during merging and unmerging.

    Parameters
    ----------
    node_parent :
        The node of the parent system that is merged
    node_child :
        The node of the child system that is merged (will be renamed to match parent)
    data_parent :
        If `True`, the parent CSM will have the data assigned and the child CSM
        otherwise
    data_index :
        Index of the LCS that will get the data. 0 is the root LCS and 1 and 2 are the
        other child systems

    Returns
    -------

    """

    def _create_csm(nodes, name):
        csm = CSM(nodes[0], name)
        csm.create_cs(nodes[1], nodes[0])
        csm.create_cs(nodes[2], nodes[0])
        return csm

    parent_nodes = ["a", "b", "c"]
    csm_parent = _create_csm(parent_nodes, "csm_parent")

    child_nodes = ["x", "y", "z"]
    child_nodes = [node if node != node_child else node_parent for node in child_nodes]
    csm_child = _create_csm(child_nodes, "csm_child")

    data = [[0, 1, 2], [3, 4, 5]]
    data_name = "data"
    if data_parent:
        csm_parent.assign_data(data, data_name, parent_nodes[data_index])
    else:
        csm_child.assign_data(data, data_name, child_nodes[data_index])

    csm_parent.merge(csm_child)

    # check data after merge
    assert data_name in csm_parent.data_names
    assert np.all(csm_parent.get_data(data_name) == data)

    csm_child_unmerged = csm_parent.unmerge()[0]

    # check data after unmerging
    if data_parent:
        assert data_name in csm_parent.data_names
        assert data_name not in csm_child_unmerged.data_names
        assert np.all(csm_parent.get_data(data_name) == data)
    else:
        assert data_name not in csm_parent.data_names
        assert data_name in csm_child_unmerged.data_names
        assert np.all(csm_child_unmerged.get_data(data_name) == data)


# test_unmerge_multi_data --------------------------------------------------------------


def test_unmerge_multi_data():
    """Test if unmerge restores multiple data on a common cs correctly.

    The test creates multiple CSM instances with data on the common cs. After unmerging,
    all data must be assigned to the original CSM and must also be removed from the
    others.

    """
    # create CSMs
    csm_p = CSM("m", "parent")
    csm_p.create_cs("a", "m")
    data_p = [[1, 2, 3]]
    csm_p.assign_data(data_p, "parent_data", "m")

    csm_c1 = CSM("m", "child1")
    csm_c1.create_cs("b", "m")
    data_c1 = [[4, 5, 6]]
    csm_c1.assign_data(data_c1, "child1_data", "m")

    csm_c2 = CSM("c", "child2")
    csm_c2.create_cs("m", "c")
    data_c2 = [[7, 8, 9]]
    csm_c2.assign_data(data_c2, "child2_data", "m")

    csm_c3 = CSM("m", "child3")
    csm_c3.create_cs("d", "m")

    # merge
    csm_p.merge(csm_c1)
    csm_p.merge(csm_c2)
    csm_p.merge(csm_c3)

    # check after merge
    assert all(
        data in ["parent_data", "child1_data", "child2_data"]
        for data in csm_p.data_names
    )

    # unmerge
    unmerged = csm_p.unmerge()

    # check if data is restored correctly
    assert len(csm_p.data_names) == 1
    assert "parent_data" in csm_p.data_names
    assert np.all(csm_p.get_data("parent_data") == data_p)

    for csm in unmerged:
        if csm.name == "child3":
            assert len(csm.data_names) == 0
        else:
            assert len(csm.data_names) == 1
            assert f"{csm.name}_data" in csm.data_names
            if csm.name == "child1":
                assert np.all(csm.get_data("child1_data") == data_c1)
            else:
                assert np.all(csm.get_data("child2_data") == data_c2)


# test_merge_data_name_collision -------------------------------------------------------


@pytest.mark.parametrize("data_cs_parent", ["rp", "a", "m"])
@pytest.mark.parametrize("data_cs_child", ["rc", "b", "m"])
def test_merge_data_name_collision(data_cs_parent, data_cs_child):
    """Check name collisions are handled appropriately."""
    csm_parent = CSM("rp", "parent")
    csm_parent.create_cs("a", "rp")
    csm_parent.create_cs("m", "rp")
    csm_parent.assign_data([[1, 2, 3]], "conflict", data_cs_parent)

    csm_child = CSM("rc", "child")
    csm_child.create_cs("b", "rc")
    csm_child.create_cs("m", "rc")
    csm_child.assign_data([[4, 5, 6]], "conflict", data_cs_child)

    with pytest.raises(NameError):
        csm_parent.merge(csm_child)


# test_interp_time ---------------------------------------------------------------------


def _orientation_from_value(val, clip_min=None, clip_max=None):
    if clip_min is not None and clip_max is not None:
        angles = np.clip(val, clip_min, clip_max)
    else:
        angles = val
    if len(angles) == 1:
        angles = angles[0]
    return WXRotation.from_euler("z", angles, degrees=True).as_matrix()


def _coordinates_from_value(val, clip_min=None, clip_max=None):
    if clip_min is not None and clip_max is not None:
        val = np.clip(val, clip_min, clip_max)
    if len(val) > 1:
        return Q_([[v, 2 * v, -v] for v in val], "mm")
    return Q_([val[0], 2 * val[0], -val[0]], "mm")


@pytest.mark.parametrize(
    "time, time_ref, systems, csm_has_time_ref, num_abs_systems",
    [
        (pd.to_timedelta([1, 7, 11, 20], "D"), None, None, False, 0),
        (pd.to_timedelta([3], "D"), None, None, False, 0),
        (pd.Timedelta(3, "D"), None, None, False, 0),
        (Q_([1, 7, 11, 20], "days"), None, None, False, 0),
        (["5days", "8days"], None, None, False, 0),
        (pd.to_timedelta([1, 7, 11, 20], "D"), None, ["lcs_1"], False, 0),
        (pd.to_timedelta([1, 7, 11, 20], "D"), None, ["lcs_1", "lcs_2"], False, 0),
        (pd.to_timedelta([1, 7, 11, 20], "D"), "2000-01-10", None, True, 0),
        (pd.to_timedelta([1, 7, 11, 20], "D"), "2000-01-13", None, True, 0),
        (pd.to_timedelta([1, 7, 11, 20], "D"), "2000-01-13", None, False, 3),
        (pd.to_timedelta([1, 7, 11, 20], "D"), "2000-01-13", None, True, 3),
        (pd.to_timedelta([1, 7, 11, 20], "D"), "2000-01-13", None, True, 2),
        (["2000-01-13", "2000-01-17"], None, None, True, 2),
    ],
)
def test_interp_time(
    time: types_time_like,
    time_ref: types_timestamp_like,
    systems: list[str],
    csm_has_time_ref: bool,
    num_abs_systems: int,
):
    """Test the ``interp_time`` method.

    Parameters
    ----------
    time :
        The value passed to the functions as ``time`` parameter
    time_ref :
        The value passed to the functions as ``time_ref`` parameter
    systems :
        The value passed to the functions as ``affected_coordinate_systems``
        parameter
    csm_has_time_ref :
        If `True`, a reference time is added to the CSM
    num_abs_systems :
        The number of time dependent systems that get a reference time assigned to
        them.

    """
    # csm data
    csm_time_ref = "2000-01-10" if csm_has_time_ref else None
    abs_systems = [f"lcs_{i}" for i in range(num_abs_systems)]
    lcs_data = dict(
        lcs_0=("root", [1, 4, 7], TS("2000-01-09")),
        lcs_1=("lcs_0", [1, 5, 9], TS("2000-01-14")),
        lcs_2=("root", [1, 6, 11], TS("2000-01-11")),
    )

    # time data
    time_class = Time(time, time_ref)
    days_interp = time_class.as_quantity().to("days").m
    if len(days_interp.shape) == 0:
        days_interp = days_interp.reshape(1)

    # create csm
    csm = tf.CoordinateSystemManager("root", time_ref=csm_time_ref)
    for k, v in lcs_data.items():
        csm.create_cs(
            k,
            v[0],
            _orientation_from_value(v[1]),
            _coordinates_from_value(v[1]),
            Q_(v[1], "day"),
            v[2] if k in abs_systems else None,
        )

    lcs_3 = tf.LocalCoordinateSystem(
        WXRotation.from_euler("y", 1).as_matrix(), Q_([4, 2, 0], "mm")
    )
    csm.add_cs("lcs_3", "lcs_2", lcs_3)

    # interpolate
    csm_interp = csm.interp_time(time, time_ref, systems)

    # evaluate results
    time_exp = time_class if len(time_class) > 1 else None
    time_ref_exp = time_class.reference_time
    for k, v in lcs_data.items():
        # create expected lcs
        if systems is None or k in systems:
            diff = 0
            if time_ref_exp is not None:
                if k in abs_systems:
                    diff = Time(time_class.reference_time - v[2])
                else:
                    diff = Time(time_class.reference_time - csm.reference_time)
                diff = diff.as_quantity().to("days").m

            lcs_exp = tf.LocalCoordinateSystem(
                _orientation_from_value(days_interp + diff, v[1][0], v[1][-1]),
                _coordinates_from_value(days_interp + diff, v[1][0], v[1][-1]),
                time_exp,
                csm.reference_time if csm.has_reference_time else time_ref_exp,
            )
        else:
            lcs_exp = csm.get_cs(k)

        # check results
        check_cs_close(csm_interp.get_cs(k), lcs_exp)
        check_cs_close(csm_interp.get_cs(v[0], k), lcs_exp.invert())

    # check static lcs unmodified
    check_cs_close(csm_interp.get_cs("lcs_3"), lcs_3)

    # check time union
    if systems is None or len(systems) == 3:
        assert np.all(csm_interp.time_union() == time_exp)


# issue 289 ----------------------------------------------------------------------------


@pytest.mark.parametrize("time_dep_coords", [True, False])
@pytest.mark.parametrize("time_dep_orient", [True, False])
@pytest.mark.parametrize("all_less", [True, False])
def test_issue_289_interp_outside_time_range(
    time_dep_orient: bool, time_dep_coords: bool, all_less: bool
):
    """Test if ``get_cs`` behaves as described in pull request #289.

    The requirement is that a static system is returned when all time values of the
    interpolation are outside of the value range of the involved coordinate systems.

    Parameters
    ----------
    time_dep_orient :
        If `True`, the orientation is time dependent
    time_dep_coords :
        If `True`, the coordinates are time dependent
    all_less :
        If `True`, all interpolation values are less than the time values of the
        LCS. Otherwise, all values are greater.

    """
    angles = [45, 135] if time_dep_orient else 135
    orientation = WXRotation.from_euler("x", angles, degrees=True).as_matrix()
    coordinates = Q_([[0, 0, 0], [1, 1, 1]] if time_dep_coords else [1, 1, 1], "mm")
    if time_dep_coords or time_dep_orient:
        time = ["5s", "6s"] if all_less else ["0s", "1s"]
    else:
        time = None

    csm = CSM("R")
    # add A as time dependent in base
    csm.create_cs("A", "R", orientation, coordinates, time)
    # add B as static in A
    csm.create_cs("B", "A")

    cs_br = csm.get_cs("B", "R", time=["2s", "3s", "4s"])

    exp_angle = 45 if time_dep_orient and all_less else 135
    exp_orient = WXRotation.from_euler("x", exp_angle, degrees=True).as_matrix()
    exp_coords = Q_([0, 0, 0] if time_dep_coords and all_less else [1, 1, 1], "mm")

    assert cs_br.is_time_dependent is False
    assert cs_br.time is None
    assert cs_br.coordinates.data.shape == (3,)
    assert cs_br.orientation.data.shape == (3, 3)
    assert np.all(cs_br.coordinates.data == exp_coords)
    assert np.all(cs_br.orientation.data == exp_orient)


# --------------------------------------------------------------------------------------
# old tests --- should be rewritten or merged into the existing ones above
# --------------------------------------------------------------------------------------


def test_relabel():
    """Test relabeling unmerged and merged CSM nodes.

    Test covers: relabeling of child system, relabeling root system, merge
    two systems after relabeling, make sure cannot relabel after merge.
    """
    csm1 = tf.CoordinateSystemManager("A")
    csm1.add_cs("B", "A", tf.LocalCoordinateSystem())

    csm2 = tf.CoordinateSystemManager("C")
    csm2.add_cs("D", "C", tf.LocalCoordinateSystem())

    csm1.relabel({"B": "X"})
    csm2.relabel({"C": "X"})

    assert "B" not in csm1.graph.nodes
    assert "X" in csm1.graph.nodes

    assert "C" not in csm2.graph.nodes
    assert "X" in csm2.graph.nodes
    assert csm2.root_system_name == "X"

    csm1.merge(csm2)
    for n in ["A", "D", "X"]:
        assert n in csm1.graph.nodes

    with pytest.raises(NotImplementedError):
        csm1.relabel({"A": "Z"})


def test_coordinate_system_manager_create_coordinate_system():
    """Test direct construction of coordinate systems in the coordinate system manager.

    Create multiple coordinate systems with all provided methods and check
    if they are constructed correctly.

    """
    angles_x = np.array([0.5, 1, 2, 2.5]) * np.pi / 2
    angles_y = np.array([1.5, 0, 1, 0.5]) * np.pi / 2
    angles = np.array([[*angles_y], [*angles_x]]).transpose()
    angles_deg = 180 / np.pi * angles

    rot_mat_x = WXRotation.from_euler("x", angles_x).as_matrix()
    rot_mat_y = WXRotation.from_euler("y", angles_y).as_matrix()

    time = pd.to_timedelta([0, 6, 12, 18], "h")
    orientations = np.matmul(rot_mat_x, rot_mat_y)
    coords = Q_([[1, 0, 0], [-1, 0, 2], [3, 5, 7], [-4, -5, -6]], "mm")

    transformation_matrix = np.resize(np.identity(4), (4, 4, 4))
    transformation_matrix[:, :3, :3] = orientations
    transformation_matrix[:, :3, 3] = coords.m

    csm = tf.CoordinateSystemManager("root")
    lcs_default = tf.LocalCoordinateSystem()

    # orientation and coordinates -------------------------
    csm.create_cs("lcs_init_default", "root")
    check_coordinate_system(
        csm.get_cs("lcs_init_default"),
        lcs_default.orientation,
        lcs_default.coordinates.data,
        True,
    )

    csm.create_cs("lcs_init_tdp", "root", orientations, coords, time)
    check_coordinate_system(
        csm.get_cs("lcs_init_tdp"),
        orientations,
        coords,
        True,
        time=time,
    )

    # from euler ------------------------------------------
    csm.create_cs_from_euler("lcs_euler_default", "root", "yx", angles[0])
    check_coordinate_system(
        csm.get_cs("lcs_euler_default"),
        orientations[0],
        lcs_default.coordinates.data,
        True,
    )

    csm.create_cs_from_euler(
        "lcs_euler_tdp", "root", "yx", angles_deg, True, coords, time
    )
    check_coordinate_system(
        csm.get_cs("lcs_euler_tdp"),
        orientations,
        coords,
        True,
        time=time,
    )

    # from homogeneous transformation ---------------------
    csm.create_cs_from_homogeneous_transformation(
        "lcs_homogeneous_default", "root", transformation_matrix, coords.u, time
    )
    check_coordinate_system(
        csm.get_cs("lcs_homogeneous_default"),
        orientations,
        coords,
        True,
        time=time,
    )


def test_coordinate_system_manager_transform_data():
    """Test the coordinate system managers transform_data function."""
    # define some coordinate systems
    # TODO: test more unique rotations - not 90°
    lcs1_in_root = tf.LocalCoordinateSystem(r_mat_z(0.5), Q_([1, 2, 3], "mm"))
    lcs2_in_root = tf.LocalCoordinateSystem(r_mat_y(0.5), Q_([3, -3, 1], "mm"))
    lcs3_in_lcs2 = tf.LocalCoordinateSystem(r_mat_x(0.5), Q_([1, -1, 3], "mm"))

    csm = tf.CoordinateSystemManager(root_coordinate_system_name="root")
    csm.add_cs("lcs_1", "root", lcs1_in_root)
    csm.add_cs("lcs_2", "root", lcs2_in_root)
    csm.add_cs("lcs_3", "lcs_2", lcs3_in_lcs2)

    data_list = Q_([[1, -3, -1], [2, 4, -1], [-1, 2, 3], [3, -4, 2]], "mm")
    data_exp = Q_([[-5, -2, -4], [-5, -9, -5], [-9, -7, -2], [-8, -1, -6]], "mm")

    # input list
    data_list_transformed = csm.transform_data(data_list, "lcs_3", "lcs_1").data
    assert matrix_is_close(data_list_transformed, data_exp)

    # input numpy array
    data_numpy_transformed = csm.transform_data(data_list, "lcs_3", "lcs_1").data
    assert matrix_is_close(data_numpy_transformed, data_exp)

    # input single numpy vector
    # data_numpy_transformed = csm.transform_data(data_np[0, :], "lcs_3", "lcs_1")
    # assert ut.vector_is_close(data_numpy_transformed, data_exp[0, :])

    # input xarray
    data_xr = xr.DataArray(
        data=data_list, dims=["n", "c"], coords={"c": ["x", "y", "z"]}
    )
    data_xr_transformed = csm.transform_data(data_xr, "lcs_3", "lcs_1")
    assert matrix_is_close(data_xr_transformed.data, data_exp)

    # TODO: Test time dependency

    # exceptions --------------------------------
    # names not in csm
    with pytest.raises(ValueError):
        csm.transform_data(data_xr, "not present", "lcs_1")
    with pytest.raises(ValueError):
        csm.transform_data(data_xr, "lcs_3", "not present")

    # data is not compatible
    with pytest.raises(np.core._exceptions._UFuncNoLoopError):
        csm.transform_data("wrong", "lcs_3", "lcs_1")
