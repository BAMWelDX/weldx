"""Tests for MediaFile."""
import numpy as np
import pytest
import xarray as xr

from weldx import Q_, U_, WeldxFile
from weldx.util.media_file import MediaFile


def write_rgb_rotate(output, width, height, n_frames, fps):
    import PIL.Image as Image
    from av import VideoFrame

    stream = output.add_stream("mpeg4", fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    image = Image.new("RGB", (width, height), (0, 0, 255))

    for _ in range(n_frames):
        frame = VideoFrame(width, height, "rgb24")
        frame.planes[0].update(image.tobytes())

        for packet in stream.encode(frame):
            output.mux(packet)

    for packet in stream.encode(None):
        output.mux(packet)

    result = [np.array(image)] * n_frames
    return result


@pytest.fixture(scope="module")
def create_video():
    """Creates a test video (once per session/module initialization.)"""

    def impl(n_frames, tmp_path, width, height, fps, external=False):
        import av

        fn_out = str(tmp_path / "data.avi")
        with av.open(fn_out, "w") as output:
            frames = write_rgb_rotate(output, width, height, n_frames, fps)

        if external:
            return fn_out

        return frames

    return impl


@pytest.mark.parametrize("external", [True, False])
def test_media_file_external(external, tmp_path, create_video):
    """Test MediaFile encapsulation with external avi file and in-memory frames."""
    # create a video with given specs
    n_frames = 10
    fps = 3
    height = 200
    width = 320
    data = create_video(
        n_frames, tmp_path, height=height, width=width, external=external, fps=fps
    )
    # encapsulate in MediaFile
    args = dict(fps=fps, reference_time=None) if not external else {}
    mf = MediaFile(data, **args)

    wf = WeldxFile(tree=dict(mf=mf), mode="rw")
    restored = wf["mf"]
    assert restored.fps.u == U_("1/s")
    assert mf.fps.m == restored.fps.m == fps

    assert mf.duration == restored.duration == Q_(n_frames / fps, "s")
    assert restored.duration.u == U_("s")

    assert mf.resolution == restored.resolution == (width, height)

    if external:
        assert str(mf.file().path) == data

    # compare content
    first_frame_input = data[0]
    first_frame_restored = restored[0].compute()
    if not external:  # data are frames!
        np.testing.assert_equal(first_frame_input, first_frame_restored)
    # compare first frames...
    xr.testing.assert_equal(mf[0].compute(), first_frame_restored)
