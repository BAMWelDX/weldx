"""Tests asdf implementations of core module."""
from pathlib import Path
from tempfile import TemporaryDirectory

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from asdf import ValidationError
from fs.memoryfs import MemoryFS
from fs.osfs import OSFS
from scipy.spatial.transform import Rotation

import weldx.transformations as tf
from weldx.asdf.tags.weldx.core.file import ExternalFile
from weldx.asdf.util import write_buffer, write_read_buffer
from weldx.constants import WELDX_QUANTITY as Q_
from weldx.core import MathematicalExpression as ME  # nopep8
from weldx.core import TimeSeries
from weldx.geometry import SpatialData
from weldx.tests._helpers import get_test_name
from weldx.transformations import WXRotation

# WXRotation ---------------------------------------------------------------------
_base_rotation = Rotation.from_euler(
    seq="xyz", angles=[[10, 20, 60], [25, 50, 175]], degrees=True
)


@pytest.mark.parametrize(
    "inputs",
    [
        _base_rotation,
        WXRotation.from_quat(_base_rotation.as_quat()),
        WXRotation.from_matrix(_base_rotation.as_matrix()),
        WXRotation.from_rotvec(_base_rotation.as_rotvec()),
        WXRotation.from_euler(seq="xyz", angles=[10, 20, 60], degrees=True),
        WXRotation.from_euler(seq="xyz", angles=[0.2, 1.3, 3.14], degrees=False),
        WXRotation.from_euler(seq="xyz", angles=Q_([10, 20, 60], "degree")),
        WXRotation.from_euler(seq="xyz", angles=Q_([0.2, 1.3, 3.14], "rad")),
        WXRotation.from_euler(seq="xyz", angles=Q_([0.2, 1.3, 3.14], "")),
        WXRotation.from_euler(seq="XYZ", angles=[10, 20, 60], degrees=True),
        WXRotation.from_euler(seq="y", angles=[10, 60, 40, 90], degrees=True),
        WXRotation.from_euler(seq="Z", angles=[10, 60, 40, 90], degrees=True),
        WXRotation.from_euler(
            seq="xy", angles=[[10, 10], [60, 60], [40, 40], [70, 75]], degrees=True
        ),
        WXRotation.from_euler(
            seq="xy", angles=Q_([[10, 10], [60, 60], [40, 40], [70, 75]], "degree")
        ),
    ],
)
def test_rotation(inputs):
    data = write_read_buffer({"rot": inputs})
    r = data["rot"]
    assert np.allclose(r.as_quat(), inputs.as_quat())
    if hasattr(inputs, "wx_meta"):
        assert r.wx_meta == inputs.wx_meta


def test_rotation_euler_exception():
    with pytest.raises(ValueError):
        WXRotation.from_euler(seq="XyZ", angles=[10, 20, 60], degrees=True)


@pytest.mark.parametrize(
    "inputs",
    [
        Q_(10, "degree"),
        Q_(0.01, "kdegree"),
        Q_(10, "mdegree"),
        Q_(2, "rad"),
        Q_(0.002, "krad"),
        Q_(2, "mrad"),
    ],
)
def test_rotation_euler_prefix(inputs):
    """Test unit prefix handling."""
    degrees = "degree" in str(inputs.u)
    rot = WXRotation.from_euler(seq="x", angles=inputs)
    data = write_read_buffer({"rot": rot})
    r = data["rot"].as_euler("xyz", degrees=degrees)[0]
    r = Q_(r, "degree") if degrees else Q_(r, "rad")
    assert np.allclose(inputs, r)


# xarray.DataArray ---------------------------------------------------------------------
def get_xarray_example_data_array():
    """
    Get an xarray.DataArray for test purposes.

    Returns
    -------
    xarray.DataArray
        DataArray for test purposes

    """
    data = np.array([[0, 1], [2, 3]])
    data = np.repeat(data[:, :, np.newaxis], 3, axis=-1)

    time_labels = ["2020-05-01", "2020-05-03"]
    d1 = np.array([-1, 1])
    d2 = pd.DatetimeIndex(time_labels)
    d3 = pd.timedelta_range("0s", "2s", freq="1s")
    coords = {"d1": d1, "d2": d2, "d3": d3, "time_labels": (["d2"], time_labels)}

    dax = xr.DataArray(data=data, dims=["d1", "d2", "d3"], coords=coords)

    dax.attrs = {"answer": 42}

    return dax


@pytest.mark.parametrize("copy_arrays", [True, False])
@pytest.mark.parametrize("lazy_load", [True, False])
def test_xarray_data_array(copy_arrays, lazy_load):
    """Test ASDF read/write of xarray.DataArray."""
    dax = get_xarray_example_data_array()
    tree = {"dax": dax}
    dax_file = write_read_buffer(
        tree, open_kwargs={"copy_arrays": copy_arrays, "lazy_load": lazy_load}
    )["dax"]
    assert dax.identical(dax_file)


# xarray.Dataset ---------------------------------------------------------------------
def get_xarray_example_dataset():
    """
    Get an xarray.Dataset for test purposes.

    Returns
    -------
        Dataset for test purposes
    """

    temp_data = [
        [[15.0, 16.0, 17.0], [18.0, 19.0, 20.0]],
        [[21.0, 22.0, 23.0], [24.0, 25.0, 26.0]],
    ]
    temp = Q_(temp_data, "°C")
    precipitation = [
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
    ]
    lon = [[-99.83, -99.32], [-99.79, -99.23]]
    lat = [[42.25, 42.21], [42.63, 42.59]]

    dsx = xr.Dataset(
        {
            "temperature": (["x", "y", "time"], temp),
            "precipitation": (["x", "y", "time"], precipitation),
        },
        coords={
            "lon": (["x", "y"], lon),
            "lat": (["x", "y"], lat),
            "time": pd.date_range("2014-09-06", periods=3),
            "time_labels": (["time"], ["2014-09-06", "2014-09-09", "2014-09-12"]),
            "time_ref": pd.Timestamp("2014-09-05"),
        },
    )
    dsx.attrs = {"temperature": "Celsius"}
    return dsx


@pytest.mark.parametrize("copy_arrays", [True, False])
@pytest.mark.parametrize("lazy_load", [True, False])
def test_xarray_dataset(copy_arrays, lazy_load):
    dsx = get_xarray_example_dataset()
    tree = {"dsx": dsx}
    dsx_file = write_read_buffer(
        tree, open_kwargs={"copy_arrays": copy_arrays, "lazy_load": lazy_load}
    )["dsx"]
    assert dsx.identical(dsx_file)


# weldx.transformations.LocalCoordinateSystem ------------------------------------------
def get_local_coordinate_system(time_dep_orientation: bool, time_dep_coordinates: bool):
    """
    Get a local coordinate system.

    Parameters
    ----------
    time_dep_orientation :
        If True, the coordinate system has a time dependent orientation.
    time_dep_coordinates :
        If True, the coordinate system has a time dependent coordinates.

    Returns
    -------
    weldx.transformations.LocalCoordinateSystem:
        A local coordinate system

    """
    if not time_dep_coordinates:
        coords = Q_(np.asarray([2.0, 5.0, 1.0]), "mm")
    else:
        coords = Q_(
            np.asarray(
                [[2.0, 5.0, 1.0], [1.0, -4.0, 1.2], [0.3, 4.4, 4.2], [1.1, 2.3, 0.2]]
            ),
            "mm",
        )

    if not time_dep_orientation:
        orientation = WXRotation.from_euler("z", np.pi / 3).as_matrix()
    else:
        orientation = WXRotation.from_euler(
            "z", np.pi / 2 * np.array([1, 2, 3, 4])
        ).as_matrix()

    if not time_dep_orientation and not time_dep_coordinates:
        return tf.LocalCoordinateSystem(orientation=orientation, coordinates=coords)

    time = pd.DatetimeIndex(["2000-01-01", "2000-01-02", "2000-01-03", "2000-01-04"])
    return tf.LocalCoordinateSystem(
        orientation=orientation, coordinates=coords, time=time
    )


@pytest.mark.parametrize("time_dep_orientation", [False, True])
@pytest.mark.parametrize("time_dep_coordinates", [False, True])
@pytest.mark.parametrize("copy_arrays", [True, False])
@pytest.mark.parametrize("lazy_load", [True, False])
def test_local_coordinate_system(
    time_dep_orientation, time_dep_coordinates, copy_arrays, lazy_load
):
    """Test (de)serialization of LocalCoordinateSystem in ASDF."""
    lcs = get_local_coordinate_system(time_dep_orientation, time_dep_coordinates)
    data = write_read_buffer(
        {"lcs": lcs}, open_kwargs={"copy_arrays": copy_arrays, "lazy_load": lazy_load}
    )
    assert data["lcs"] == lcs


@pytest.mark.parametrize("copy_arrays", [True, False])
@pytest.mark.parametrize("lazy_load", [True, False])
@pytest.mark.parametrize("has_ref_time", [True, False])
@pytest.mark.parametrize("has_tdp_orientation", [True, False])
def test_local_coordinate_system_coords_timeseries(
    copy_arrays, lazy_load, has_ref_time, has_tdp_orientation
):
    """Test reading and writing a LCS with a `TimeSeries` as coordinates to asdf."""
    # create inputs to lcs __init__
    me = ME("a*t", dict(a=Q_([[1, 0, 0]], "1/s")))
    ts = TimeSeries(data=me)

    ref_time = None
    if has_ref_time:
        ref_time = pd.Timestamp("13:37")

    time = None
    orientation = None
    if has_tdp_orientation:
        time = Q_([1, 2], "s")
        orientation = WXRotation.from_euler("x", [0, 90], degrees=True).as_matrix()

    # create lcs
    lcs = tf.LocalCoordinateSystem(
        orientation=orientation, coordinates=ts, time=time, time_ref=ref_time
    )

    # round trip and compare
    lcs_buffer = write_read_buffer(
        {"lcs": lcs}, open_kwargs={"copy_arrays": copy_arrays, "lazy_load": lazy_load}
    )["lcs"]
    assert lcs_buffer == lcs


def test_local_coordinate_system_shape_violation():
    """Test if the shape validators work as expected."""
    # coordinates have wrong shape ------------------------
    orientation = xr.DataArray(
        data=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        dims=["u", "v"],
        coords={"u": ["x", "y", "z"], "v": [0, 1, 2]},
    )
    coordinates = xr.DataArray(
        data=[1, 2],
        dims=["c"],
        coords={"c": ["x", "y"]},
    )
    lcs = tf.LocalCoordinateSystem(
        orientation=orientation, coordinates=coordinates, construction_checks=False
    )

    with pytest.raises(ValidationError):
        write_buffer({"lcs": lcs})

    # orientations have wrong shape -----------------------
    orientation = xr.DataArray(
        data=[[1, 2], [3, 4]],
        dims=["c", "v"],
        coords={"c": ["x", "y"], "v": [0, 1]},
    )
    coordinates = xr.DataArray(
        data=[1, 2, 3],
        dims=["u"],
        coords={"u": ["x", "y", "z"]},
    )
    lcs = tf.LocalCoordinateSystem(
        orientation=orientation, coordinates=coordinates, construction_checks=False
    )

    with pytest.raises(ValidationError):
        write_buffer({"lcs": lcs})


# weldx.transformations.CoordinateSystemManager ----------------------------------------


def get_example_coordinate_system_manager():
    """Get a consistent CoordinateSystemManager instance for test purposes."""
    csm = tf.CoordinateSystemManager("root")
    csm.create_cs("lcs_01", "root", coordinates=[1, 2, 3])
    csm.create_cs(
        "lcs_02",
        "root",
        orientation=WXRotation.from_euler("z", np.pi / 3).as_matrix(),
        coordinates=[4, -7, 8],
    )
    csm.create_cs(
        "lcs_03",
        "lcs_02",
        orientation=WXRotation.from_euler("y", np.pi / 11),
        coordinates=[4, -7, 8],
    )
    return csm


@pytest.mark.parametrize("copy_arrays", [True, False])
@pytest.mark.parametrize("lazy_load", [True, False])
def test_coordinate_system_manager(copy_arrays, lazy_load):
    csm = get_example_coordinate_system_manager()
    tree = {"cs_hierarchy": csm}
    data = write_read_buffer(
        tree, open_kwargs={"copy_arrays": copy_arrays, "lazy_load": lazy_load}
    )
    csm_file = data["cs_hierarchy"]
    assert csm == csm_file


def get_coordinate_system_manager_with_subsystems(nested: bool):
    lcs = [tf.LocalCoordinateSystem(coordinates=[i, -i, -i]) for i in range(12)]

    # global system
    csm_global = tf.CoordinateSystemManager("base", "Global System", "2000-06-08")
    csm_global.add_cs("robot", "base", lcs[0])
    csm_global.add_cs("specimen", "base", lcs[1])

    # robot system
    csm_robot = tf.CoordinateSystemManager("robot", "Robot system")
    csm_robot.add_cs("head", "robot", lcs[2])

    # robot head system
    csm_head = tf.CoordinateSystemManager("head", "Head system")
    csm_head.add_cs("torch tcp", "head", lcs[3])
    csm_head.add_cs("camera tcp", "head", lcs[4], lsc_child_in_parent=False)
    csm_head.add_cs("scanner 1 tcp", "head", lcs[5])
    csm_head.add_cs("scanner 2 tcp", "head", lcs[6])

    # scanner system 1
    csm_scanner_1 = tf.CoordinateSystemManager("scanner 1", "Scanner 1 system")
    csm_scanner_1.add_cs("scanner 1 tcp", "scanner 1", lcs[7])

    # scanner system 2
    csm_scanner_2 = tf.CoordinateSystemManager("scanner 2", "Scanner 2 system")
    csm_scanner_2.add_cs("scanner 2 tcp", "scanner 2", lcs[8])

    # specimen system
    csm_specimen = tf.CoordinateSystemManager("specimen", "Specimen system")
    csm_specimen.add_cs("thermocouple 1", "specimen", lcs[9])
    csm_specimen.add_cs("thermocouple 2", "specimen", lcs[10])
    csm_specimen.add_cs("thermocouple 3", "thermocouple 2", lcs[11])

    if nested:
        csm_head.merge(csm_scanner_1)
        csm_head.merge(csm_scanner_2)
        csm_robot.merge(csm_head)
        csm_global.merge(csm_robot)
        csm_global.merge(csm_specimen)
    else:
        csm_global.merge(csm_specimen)
        csm_global.merge(csm_robot)
        csm_global.merge(csm_head)
        csm_global.merge(csm_scanner_1)
        csm_global.merge(csm_scanner_2)

    return csm_global


@pytest.mark.parametrize("copy_arrays", [True, False])
@pytest.mark.parametrize("lazy_load", [True, False])
@pytest.mark.parametrize("nested", [True, False])
def test_coordinate_system_manager_with_subsystems(copy_arrays, lazy_load, nested):
    csm = get_coordinate_system_manager_with_subsystems(nested)
    tree = {"cs_hierarchy": csm}
    data = write_read_buffer(
        tree, open_kwargs={"copy_arrays": copy_arrays, "lazy_load": lazy_load}
    )
    csm_file = data["cs_hierarchy"]
    assert csm == csm_file


@pytest.mark.parametrize("copy_arrays", [True, False])
@pytest.mark.parametrize("lazy_load", [True, False])
@pytest.mark.parametrize("csm_time_ref", [None, "2000-03-16"])
def test_coordinate_system_manager_time_dependencies(
    copy_arrays, lazy_load, csm_time_ref
):
    """Test serialization of time components from CSM and its attached LCS."""
    lcs_tdp_1_time_ref = None
    if csm_time_ref is None:
        lcs_tdp_1_time_ref = pd.Timestamp("2000-03-17")
    lcs_tdp_1 = tf.LocalCoordinateSystem(
        coordinates=[[1, 2, 3], [4, 5, 6]],
        time=pd.TimedeltaIndex([1, 2], "D"),
        time_ref=lcs_tdp_1_time_ref,
    )
    lcs_tdp_2 = tf.LocalCoordinateSystem(
        coordinates=[[3, 7, 3], [9, 5, 8]],
        time=pd.TimedeltaIndex([1, 2], "D"),
        time_ref=pd.Timestamp("2000-03-21"),
    )

    csm_root = tf.CoordinateSystemManager("root", "csm_root", csm_time_ref)
    csm_root.add_cs("cs_1", "root", lcs_tdp_2)

    csm_sub_1 = tf.CoordinateSystemManager("cs_2", "csm_sub_1", csm_time_ref)
    csm_sub_1.add_cs("cs_1", "cs_2", lcs_tdp_2)
    csm_sub_1.add_cs("cs_3", "cs_2", lcs_tdp_1)

    csm_sub_2 = tf.CoordinateSystemManager("cs_4", "csm_sub_2")
    csm_sub_2.create_cs("cs_1", "cs_4")

    csm_root.merge(csm_sub_1)
    csm_root.merge(csm_sub_2)

    tree = {"cs_hierarchy": csm_root}
    data = write_read_buffer(
        tree, open_kwargs={"copy_arrays": copy_arrays, "lazy_load": lazy_load}
    )
    csm_file = data["cs_hierarchy"]
    assert csm_root == csm_file


@pytest.mark.parametrize("copy_arrays", [True, False])
@pytest.mark.parametrize("lazy_load", [True, False])
def test_coordinate_system_manager_with_data(copy_arrays, lazy_load):
    """Test if data attached to a CSM is stored and read correctly."""
    csm = tf.CoordinateSystemManager("root", "csm")
    csm.create_cs("cs_1", "root", coordinates=[1, 1, 1])
    csm.create_cs("cs_2", "root", coordinates=[-1, -1, -1])
    csm.create_cs("cs_11", "cs_1", coordinates=[1, 1, 1])

    data_11 = SpatialData(coordinates=np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]))
    data_2 = SpatialData(
        coordinates=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        triangles=np.array([[0, 1, 2], [0, 2, 3]], dtype="uint32"),
    )

    csm.assign_data(data_11, "data_11", "cs_11")
    csm.assign_data(data_2, "data_2", "cs_2")

    tree = {"csm": csm}
    buffer = write_read_buffer(
        tree, open_kwargs={"copy_arrays": copy_arrays, "lazy_load": lazy_load}
    )
    csm_buffer = buffer["csm"]

    for data_name in csm.data_names:
        sd = csm.get_data(data_name)
        sd_buffer = csm_buffer.get_data(data_name)
        assert sd == sd_buffer


# --------------------------------------------------------------------------------------
# TimeSeries
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("copy_arrays", [True, False])
@pytest.mark.parametrize("lazy_load", [True, False])
@pytest.mark.parametrize(
    "ts",
    [
        TimeSeries(Q_(42, "m")),
        TimeSeries(Q_(42.0, "m")),
        TimeSeries(Q_([42, 23, 12], "m"), time=pd.TimedeltaIndex([0, 2, 4])),
        TimeSeries(Q_([42, 23, 12], "m"), time=pd.TimedeltaIndex([0, 2, 5])),
        TimeSeries(ME("a*t+b", parameters={"a": Q_(2, "1/s"), "b": Q_(5, "")})),
    ],
)
def test_time_series_discrete(ts, copy_arrays, lazy_load):
    ts_file = write_read_buffer(
        {"ts": ts}, open_kwargs={"copy_arrays": copy_arrays, "lazy_load": lazy_load}
    )["ts"]
    if isinstance(ts.data, ME):
        assert ts.data == ts_file.data
    else:
        assert np.all(ts_file.data == ts.data)
    assert np.all(ts_file.time == ts.time)
    assert ts_file.interpolation == ts.interpolation


# --------------------------------------------------------------------------------------
# ExternalFile
# --------------------------------------------------------------------------------------

weldx_root_dir = Path(__file__).parent.parent.absolute().as_posix()


class TestExternalFile:
    """Collects all tests related to the `ExternalFile` class."""

    # test_init ------------------------------------------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        "file_path, save_content, hostname",
        [
            (f"{weldx_root_dir}/data/WelDX_notext.svg", True, "a host"),
            (f"{weldx_root_dir}/data/WelDX_notext.svg", False, "a host"),
            (Path(f"{weldx_root_dir}/data/WelDX_notext.svg"), False, "a host"),
            (f"{weldx_root_dir}/data/WelDX_notext.svg", False, None),
        ],
    )
    def test_init(file_path, save_content, hostname):
        """Test the `__init__` method.

        Parameters
        ----------
        file_path: Union[str, Path]
            Path of the file
        save_content : bool
            If `True`, the file should be stored in the asdf file
        hostname:
            The files hostname

        """
        ef = ExternalFile(file_path, asdf_save_content=save_content, hostname=hostname)
        assert save_content == ef.asdf_save_content
        assert ef.filename == "WelDX_notext.svg"
        assert ef.suffix == "svg"

        if hostname is not None:
            assert hostname == ef.hostname
        else:
            print(hostname)
            assert isinstance(ef.hostname, str)

    # test_init_exceptions -------------------------------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        "kwargs, exception_type, test_name",
        [
            ({"path": "does_not.exist"}, ValueError, "# File does not exist"),
            ({"hashing_algorithm": "fancy"}, ValueError, "# Invalid hashing algorithm"),
        ],
        ids=get_test_name,
    )
    def test_init_exceptions(kwargs, exception_type, test_name):
        """Test the `__init__` methods exceptions.

        Parameters
        ----------
        kwargs : Dict
            Key word arguments that should be passed to the `__init__` method
        exception_type :
            The expected exception type
        test_name : str
            Name of the test

        """
        if "path" not in kwargs:
            kwargs["path"] = f"{weldx_root_dir}/data/WelDX_notext.svg"

        with pytest.raises(exception_type):
            ExternalFile(**kwargs)

    # test_write_to --------------------------------------------------------------------
    @staticmethod
    @pytest.mark.parametrize(
        "dir_read, file_name",
        [
            ("data", "WelDX_notext.svg"),
            ("", "__init__.py"),
        ],
    )
    def test_write_to(dir_read, file_name):
        """Test the `write_to` method by writing a read file to a virtual file system.

        Parameters
        ----------
        dir_read : str
            Directory that contains the source file
        file_name : str
            Name of the source file

        """
        path_read = f"{dir_read}/{file_name}"
        ef = ExternalFile(f"{weldx_root_dir}/{path_read}")

        with OSFS(weldx_root_dir) as file_system:
            original_hash = file_system.hash(path_read, "md5")

            # check writing to hard drive
            with TemporaryDirectory(dir=weldx_root_dir) as td:
                ef.write_to(td)
                new_file_path = Path(f"{Path(td).name}/{file_name}").as_posix()
                assert file_system.isfile(new_file_path)
                assert file_system.hash(new_file_path, "md5") == original_hash

        # check writing to a memory file system
        with MemoryFS() as file_system:
            file_system.makedir("some_directory")
            ef.write_to("some_directory", file_system)

            new_file_path = f"some_directory/{file_name}"
            assert file_system.isfile(new_file_path)
            assert file_system.hash(new_file_path, "md5") == original_hash

    # test_hashing ---------------------------------------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        "algorithm, buffer_size",
        [
            ("SHA-256", 1024),
            ("MD5", 2048),
        ],
    )
    def test_hashing(algorithm: str, buffer_size: int):
        """Test the hashing functions.

        Two things should be tested here:
        - All available algorithms can be selected and work properly
        - Both available hashing functions deliver the same hash for the same algorithm

        Parameters
        ----------
        algorithm : str
            The hashing algorithm
        buffer_size : int
            The size of the buffer that is used by the `calculate_hash` method.

        """
        file_path = f"{weldx_root_dir}/data/WelDX_notext.svg"
        ef = ExternalFile(file_path, hashing_algorithm=algorithm)
        buffer = ef.get_file_content()

        hash_buffer = ExternalFile.calculate_hash(buffer, algorithm)
        hash_file = ExternalFile.calculate_hash(file_path, algorithm, buffer_size)

        assert hash_buffer == hash_file

    # test_asdf_serialization ----------------------------------------------------------

    @staticmethod
    @pytest.mark.parametrize("copy_arrays", [True, False])
    @pytest.mark.parametrize("lazy_load", [True, False])
    @pytest.mark.parametrize("store_content", [True, False])
    def test_asdf_serialization(copy_arrays, lazy_load, store_content):
        """Test the asdf serialization of the `ExternalFile` class.

        Parameters
        ----------
        copy_arrays : bool
            If `False`, arrays are accessed via memory mapping whenever possible while
            reading them.
        lazy_load : bool
            If `True`, items from the asdf file are not loaded until accessed.
        store_content : bool
            If `True`, the file content is stored in the asdf file.

        """
        ef = ExternalFile(
            f"{weldx_root_dir}/data/WelDX_notext.svg",
            asdf_save_content=store_content,
        )
        tree = {"file": ef}
        ef_file = write_read_buffer(
            tree, open_kwargs={"copy_arrays": copy_arrays, "lazy_load": lazy_load}
        )["file"]

        assert ef.filename == ef_file.filename
        assert ef.suffix == ef_file.suffix
        assert ef.directory == ef_file.directory
        assert ef.hostname == ef_file.hostname

        assert ef.created == ef_file.created
        assert ef.modified == ef_file.modified
        assert ef.size == ef_file.size

        assert ef.hashing_algorithm == ef_file.hashing_algorithm

        if store_content:
            with OSFS(weldx_root_dir) as file_system:
                original_hash = file_system.hash("data/WelDX_notext.svg", "md5")

            with MemoryFS() as file_system:
                ef_file.write_to("", file_system)
                assert file_system.hash("WelDX_notext.svg", "md5") == original_hash


# --------------------------------------------------------------------------------------
# SpatialData
# --------------------------------------------------------------------------------------


class TestPointCloud:
    @staticmethod
    @pytest.mark.parametrize("copy_arrays", [True, False])
    @pytest.mark.parametrize("lazy_load", [True, False])
    def test_asdf_serialization(copy_arrays, lazy_load):
        coordinates = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]

        triangles = [[0, 1, 2], [2, 3, 0]]

        pc = SpatialData(coordinates=coordinates, triangles=triangles)
        tree = {"point_cloud": pc}
        pc_file = write_read_buffer(
            tree, open_kwargs={"copy_arrays": copy_arrays, "lazy_load": lazy_load}
        )["point_cloud"]

        assert np.all(pc_file.coordinates == pc.coordinates)


# --------------------------------------------------------------------------------------
# Graph
# --------------------------------------------------------------------------------------
class TestGraph:
    @staticmethod
    def test_graph_serialization():
        g = nx.DiGraph()
        g.add_edges_from(
            [("A", "B"), ("A", "C"), ("A", "F"), ("D", "C"), ("B", "H"), ("X", "A")]
        )
        g2 = write_read_buffer({"graph": g})["graph"]

        assert all(e in g.edges for e in g2.edges)
        assert all(n in g.nodes for n in g2.nodes)
