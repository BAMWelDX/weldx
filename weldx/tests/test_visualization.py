"""Perform some checks regarding the import redirection if weldx_widgets is missing."""

from unittest.mock import patch

import pytest


def test_redirection_weldx_widgets_not_found():
    """Check we receive a warning about weldx_widgets not being available."""
    orig_import = __import__  # Store original __import__

    def import_mock(name, *args, **kwargs):
        if "weldx_widgets" in name:
            raise ModuleNotFoundError("weldx_widgets not found")
        if "matplotlib" in name:
            raise ModuleNotFoundError("matplotlib not found")
        return orig_import(name, *args, **kwargs)

    pattern = ".*weldx_widget.*unavailable"

    with patch("builtins.__import__", side_effect=import_mock):
        with pytest.warns(match=pattern):
            import weldx.visualization as vs

        # ensure that using declared features emits the warning again.
        for name in vs.__all__:
            with pytest.warns(match=pattern):
                obj = getattr(vs, name)
                obj()
