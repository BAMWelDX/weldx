import sys

# from unittest.mock import patch


def test_redirection_matplotlib_not_found():
    """Check we receive a warning/print about matplotlib not being available."""
    # FIXME: this mock does not work as intended

    from pip._internal.utils.misc import captured_stdout

    mpl = sys.modules.pop("matplotlib", None)
    # delete weldx visualization module (if already loaded)
    vs = sys.modules.pop("weldx.visualization", None)

    try:
        import weldx.visualization as vs

        # now check matplotlib functions trigger a warning
        with captured_stdout() as stdout:
            vs.axes_equal()
        assert "not found" in stdout
    except BaseException:
        if mpl:
            sys.modules["matplotlib"] = mpl
