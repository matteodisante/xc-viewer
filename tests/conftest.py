"""Shared fixtures for the Qt-widget tests under tests/viewer.

A ``QApplication`` is required before any ``QWidget`` can be constructed, and only one
may exist per process; ``qapp`` creates it once (offscreen -- no display needed) and
every widget test depends on it directly or by constructing a widget.

PyQt6 sits in the opt-in ``viewer`` dependency group (pyproject.toml), so it is absent
wherever the interactive viewer is not wanted, CI included. The three modules listed
below import a widget at module scope, which pytest reports as a collection error --
not a skip -- and one collection error fails the whole run. Drop them from collection
when the binding is missing, so an install without the viewer gives a smaller suite
rather than a red one. They are a real part of the suite: run them with
``uv sync --group viewer``, which is what a local checkout already has.
"""

from __future__ import annotations

import gc
import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if sys.platform == "win32" and os.environ["QT_QPA_PLATFORM"] == "offscreen":
    # Qt's Windows offscreen backend uses FreeType, not native font discovery.
    # Without a font directory, dummy glyph metrics distort widget size hints.
    os.environ.setdefault(
        "QT_QPA_FONTDIR", str(Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts")
    )

import pytest

_WIDGET_MODULES = ["test_flight_picker.py", "test_main_window.py", "test_map_view.py"]
collect_ignore = (
    [] if importlib.util.find_spec("PyQt6") is not None else list(_WIDGET_MODULES)
)


@pytest.fixture(scope="session")
def qapp():
    qt_widgets = pytest.importorskip("PyQt6.QtWidgets")
    from PyQt6.QtGui import QFontDatabase

    from xc_viewer.app import configure_graphics
    from xc_viewer.widgets.theme import apply_theme

    app = qt_widgets.QApplication.instance()
    if app is None:
        configure_graphics()
        app = qt_widgets.QApplication([])
        apply_theme(app)
    assert QFontDatabase.families(), (
        "Qt found no fonts; set QT_QPA_FONTDIR to a directory containing system fonts "
        "before running offscreen widget tests."
    )
    yield app


@pytest.fixture(autouse=True)
def _collect_widgets_between_tests(request):
    """Free widgets left in reference cycles here, never during event delivery.

    A closed widget kept alive only by a cycle (a signal connected to a lambda or a
    bound method) still has its creation-time PolishRequest queued. When a later
    test processes events, Python's cyclic collector can free that dialog in the
    middle of polishing it, and Qt then segfaults in ``QWidget::ensurePolished``.
    Only tests using ``qapp`` can create widgets, so only they pay for a collection.
    """
    yield
    if "qapp" in request.fixturenames:
        gc.collect()
