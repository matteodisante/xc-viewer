"""Window resizing and camera changes must preserve interactive navigation."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from xc_viewer.main_window import MainWindow
from xc_viewer.widgets.flight_picker import FlightPicker


@pytest.fixture
def window(qapp, monkeypatch):
    monkeypatch.setattr(FlightPicker, "_repopulate_filter_combos", lambda self: None)
    win = MainWindow()
    win._igc_path = Path("test.igc")
    win._raw = SimpleNamespace(
        fixes=pd.DataFrame(
            {
                "lon": np.linspace(3, 4, 20),
                "lat": np.linspace(43, 44, 20),
                "alt": np.linspace(1000, 1500, 20),
            }
        )
    )
    win._redraw()
    yield win
    win.close()


def _width_report(window):
    """Which widgets set the window's minimum width, for a failure on another OS."""
    from PyQt6.QtWidgets import QWidget

    def widest(root):
        rows = [
            (child.minimumSizeHint().width(), type(child).__name__, child.objectName())
            for child in root.findChildren(QWidget)
            if child.isVisibleTo(root)
        ]
        return sorted(rows, reverse=True)[:3]

    page = window._tabs.currentWidget()
    return (
        f"tab {window._tabs.tabText(window._tabs.currentIndex())!r}: "
        f"window min {window.minimumSizeHint().width()}, "
        f"picker min {window._picker.minimumSizeHint().width()} "
        f"{widest(window._picker)}, "
        f"page min {page.minimumSizeHint().width()} {widest(page)}"
    )


def _click_map_toggle(button, qapp):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest

    # Test the actual mouse target, not just the signal of a hidden button.
    assert button.isVisible()
    center = button.rect().center()
    assert button.window().childAt(button.mapTo(button.window(), center)) is button
    QTest.mouseClick(button, Qt.MouseButton.LeftButton)
    qapp.processEvents()


def _focused_map_height(window):
    # Account for the native menu bar on platforms that put it inside the window.
    # The tab strip keeps the return toggle and map navigation accessible.
    return window.centralWidget().height() - window._tabs.tabBar().height() - 8


def test_colour_and_visibility_preserve_2d_zoom_and_toolbar_history(window):
    ax = window._figure.axes[0]
    window._toolbar.push_current()
    ax.set_xlim(3.2, 3.3)
    ax.set_ylim(43.4, 43.5)
    window._toolbar.push_current()
    for change in (
        lambda: window._controls._color_combo.setCurrentIndex(1),
        lambda: window._controls._chk_raw.setChecked(False),
        lambda: window._controls._chk_raw.setChecked(True),
        lambda: window._controls._chk_dms.setChecked(True),
    ):
        change()
        assert window._figure.axes[0] is ax
        assert ax.get_xlim() == pytest.approx((3.2, 3.3))
        assert ax.get_ylim() == pytest.approx((43.4, 43.5))
    window._toolbar.back()
    assert ax.get_xlim()[0] < 3.1
    window._toolbar.forward()
    assert ax.get_xlim() == pytest.approx((3.2, 3.3))


def test_3d_zoom_preserves_mouse_rotation_and_orientation_preserves_pan(window):
    window._controls._chk_3d.setChecked(True)
    ax = window._figure.axes[0]
    ax.set_xlim(3.2, 3.3)
    ax.set_ylim(43.4, 43.5)
    ax.set_zlim(1100, 1200)
    ax.view_init(elev=12, azim=78, roll=9)
    window._controls._color_combo.setCurrentIndex(2)
    assert window._figure.axes[0] is ax
    assert (ax.elev, ax.azim, ax.roll) == (12, 78, 9)
    assert ax.get_xlim() == pytest.approx((3.2, 3.3))
    window._controls._slider_zoom.setValue(200)
    assert (ax.elev, ax.azim, ax.roll) == (12, 78, 9)
    assert ax.get_xlim() == pytest.approx((3.225, 3.275))
    assert ax.get_zlim() == pytest.approx((1125, 1175))
    window._controls._slider_azim.setValue(40)
    assert (ax.elev, ax.azim, ax.roll) == (12, 40, 9)
    assert ax.get_xlim() == pytest.approx((3.225, 3.275))
    window._controls._chk_3d.setChecked(False)
    window._controls._chk_3d.setChecked(True)
    ax = window._figure.axes[0]
    assert (ax.elev, ax.azim, ax.roll) == (12, 40, 9)
    assert ax.get_xlim() == pytest.approx((3.225, 3.275))
    window._controls._btn_reset_view.click()
    assert window._figure.axes[0].get_xlim()[0] < 3.1


def test_fullscreen_hides_picker_and_restores_layout_without_redraw(
    window, qapp, monkeypatch
):
    window.show()
    qapp.processEvents()
    geometry = window.geometry()
    sizes = window._splitter.sizes()
    axes = window._figure.axes[0]
    monkeypatch.setattr(window, "_redraw", lambda: pytest.fail("fullscreen redraw"))
    _click_map_toggle(window._fullscreen_button, qapp)
    assert window.isFullScreen()
    assert window._picker.isHidden()
    assert window._controls.isHidden()
    assert window._toolbar.isHidden()
    assert window._tabs.tabBar().isVisible()
    assert window._canvas.width() >= window.width() - 8
    assert window._canvas.height() >= _focused_map_height(window)
    assert "Exit" in window._fullscreen_button.text()
    _click_map_toggle(window._fullscreen_button, qapp)
    assert not window.isFullScreen()
    assert not window._picker.isHidden()
    assert window.geometry() == geometry
    assert window._splitter.sizes() == sizes
    assert window._figure.axes[0] is axes
    assert window._fullscreen_button.text() == "Map full screen"
    assert not window._controls.isHidden()
    assert not window._toolbar.isHidden()
    assert not window._tabs.tabBar().isHidden()


def test_fullscreen_exit_after_resize_and_single_map_focus(window, qapp, monkeypatch):
    view = window._thermal_plane
    monkeypatch.setattr(view, "ensure_loaded", lambda: None)
    window._tabs.setCurrentWidget(view)
    view._mode.setCurrentIndex(1)
    window.show()
    qapp.processEvents()
    _click_map_toggle(window._fullscreen_button, qapp)
    view._view.setCurrentIndex(view._view.findData("planes"))
    window.resize(1920, 1080)
    qapp.processEvents()
    # Qt can clear its fullscreen flag on resize without a WindowStateChange event.
    _click_map_toggle(window._fullscreen_button, qapp)
    assert not window.isFullScreen()
    assert not window._picker.isHidden()
    assert window._fullscreen_state is None
    assert view._view.currentData() == "planes"
    assert len(view._plane_axes) == 1 and view._map_ax is None


@pytest.mark.parametrize("width,height", [(1000, 700), (800, 600)])
def test_all_tabs_resize_and_keep_controls_inside_window(
    window, qapp, monkeypatch, width, height
):
    from PyQt6.QtCore import QPoint, QRect, QSize
    from PyQt6.QtWidgets import QLayout

    from xc_viewer.widgets.flow_layout import FlowLayout

    for view in (
        window._map_view,
        window._thermal_plane,
        window._thermal_density,
        window._route_comparison,
        window._group_flights,
    ):
        monkeypatch.setattr(view, "ensure_loaded", lambda: None)
    # Prepared cell descriptions must not grow the minimum window width either.
    window._thermal_plane._cells.blockSignals(True)
    window._thermal_plane._cells.addItem("High mountains · many flights · " * 6)
    window._controls._chk_3d.setChecked(True)
    window.show()
    window.resize(width, height)
    for index in range(window._tabs.count()):
        window._tabs.setCurrentIndex(index)
        qapp.processEvents()
        assert window.size() == QSize(width, height), _width_report(window)
        tab = window._tabs.currentWidget()
        canvas = window._canvas if index == 0 else getattr(tab, "_canvas", None)
        if canvas is not None:
            label = window._tabs.tabText(index)
            assert canvas.width() > 200 and canvas.height() > 100, label
        for layout in tab.findChildren(QLayout):
            if not isinstance(layout, FlowLayout):
                continue
            for item_index in range(layout.count()):
                widget = layout.itemAt(item_index).widget()
                if widget is not None and widget.isVisible():
                    bounds = QRect(widget.mapTo(window, QPoint()), widget.size())
                    assert window.rect().contains(bounds), (index, bounds)


def test_data_folder_button_switches_and_remembers_the_folder(
    window, tmp_path, monkeypatch
):
    from xc_viewer import app, datafolder
    from xc_viewer.core.disciplines import DATA_FOLDER_ENV, DISCIPLINES

    folder = tmp_path / "xc-viewer-data"
    (folder / "paragliders").mkdir(parents=True)
    # datafolder.use() writes the process environment; let monkeypatch restore it.
    monkeypatch.setenv(DATA_FOLDER_ENV, str(tmp_path))
    for discipline in DISCIPLINES.values():
        monkeypatch.delenv(discipline.env, raising=False)
    monkeypatch.setattr(app, "choose_data_folder", lambda parent, start="": folder)
    saved = {}

    class Settings:
        def setValue(self, key, value):  # noqa: N802
            saved[key] = value

    monkeypatch.setattr("PyQt6.QtCore.QSettings", Settings)
    window._picker._btn_data_folder.click()
    assert saved == {datafolder.SETTINGS_KEY: str(folder.resolve())}
    assert str(folder.resolve()) in window.windowTitle()
    assert window._picker._data_folder_label.text().endswith(folder.name)


def test_initial_window_fits_available_screen(window, qapp):
    window.show()
    qapp.processEvents()
    available = window.screen().availableGeometry()
    assert window.width() <= available.width()
    assert window.height() <= available.height()


@pytest.mark.parametrize("index", [1, 2, 3, 4, 5])
def test_map_only_fullscreen_preserves_canvas_zoom_and_optional_panels(
    window, qapp, monkeypatch, index
):
    tab = window._tabs.widget(index)
    monkeypatch.setattr(tab, "ensure_loaded", lambda: None)
    window._tabs.setCurrentIndex(index)
    window.show()
    qapp.processEvents()
    if index in (4, 5):
        viewer = tab if index == 4 else tab._viewer
        plot = viewer._scene_panel
        hidden = viewer._table.isHidden()
    else:
        plot = tab._canvas
        if not tab._figure.axes:
            tab._figure.add_subplot(111)
        axes = tuple(tab._figure.axes)
        axes[-1].set_xlim(1, 3)
        axes[-1].set_ylim(2, 4)
    _click_map_toggle(window._fullscreen_button, qapp)
    assert plot.width() >= window.width() - 8
    assert plot.height() >= _focused_map_height(window)
    assert window._tabs.tabBar().isVisible()
    if index in (4, 5):
        assert viewer._locator.isHidden() and viewer._table.isHidden()
    if index == 2:
        assert not tab._map_ax.get_visible()
        assert tab._plane_ax.get_visible()
    _click_map_toggle(window._fullscreen_button, qapp)
    if index in (4, 5):
        assert not viewer._locator.isHidden()
        assert viewer._table.isHidden() == hidden
    else:
        assert tuple(tab._figure.axes) == axes
        assert axes[-1].get_xlim() == (1, 3)
        assert axes[-1].get_ylim() == (2, 4)
    if index == 2:
        assert tab._map_ax.get_visible()


def test_pending_route_results_stay_hidden_until_fullscreen_exit(
    window, qapp, monkeypatch
):
    tab = window._route_comparison
    monkeypatch.setattr(tab, "ensure_loaded", lambda: None)
    window._tabs.setCurrentWidget(tab)
    window.show()
    _click_map_toggle(window._fullscreen_button, qapp)
    tab._table.show()  # A scene finishes loading while only the map is visible.
    qapp.processEvents()
    assert tab._table.isHidden()
    window._exit_full_screen()
    qapp.processEvents()
    assert not tab._table.isHidden()


def test_3d_dialog_fullscreen_shows_only_terrain_and_restores_controls(qapp):
    from xc_viewer.widgets.thermal_3d import Thermal3D

    dialog = Thermal3D()
    dialog.show()
    qapp.processEvents()
    camera = dialog._view.cameraParams()
    _click_map_toggle(dialog._fullscreen, qapp)
    assert dialog._heading.isHidden() and dialog._point_size.isHidden()
    assert dialog._view.height() >= dialog.height() - dialog._fullscreen.height() - 2
    assert dialog._view.width() >= dialog.width() - 2
    _click_map_toggle(dialog._fullscreen, qapp)
    assert not dialog.isFullScreen()
    assert not dialog._heading.isHidden() and not dialog._point_size.isHidden()
    assert dialog._view.cameraParams() == camera
    dialog.close()


def test_switching_tabs_while_focused_keeps_the_new_map_visible(
    window, qapp, monkeypatch
):
    window.show()
    window._fullscreen_button.click()
    for tab in (window._map_view, window._thermal_plane, window._thermal_density):
        monkeypatch.setattr(tab, "ensure_loaded", lambda: None)
        window._tabs.setCurrentWidget(tab)
        qapp.processEvents()
        assert tab._canvas.isVisible()
        assert tab._canvas.width() >= window.width() - 8
        assert tab._canvas.height() >= _focused_map_height(window)
        assert not tab._toolbar.isVisible()
    window._exit_full_screen()
    qapp.processEvents()
    assert not window._thermal_density._toolbar.isHidden()


@pytest.mark.parametrize("index", [0, 1, 2, 3, 4])
def test_native_fullscreen_preserves_complete_viewer(window, qapp, monkeypatch, index):
    tab = window._tabs.widget(index)
    if index:
        monkeypatch.setattr(tab, "ensure_loaded", lambda: None)
    window._tabs.setCurrentIndex(index)
    window.show()
    qapp.processEvents()
    toolbar = window._toolbar if index == 0 else getattr(tab, "_toolbar", None)
    for native_transition in (window.showFullScreen, window.showNormal):
        native_transition()
        qapp.processEvents()
        assert window._map_focus is None
        assert window._fullscreen_state is None
        assert window._picker.isVisible()
        assert window._tabs.tabBar().isVisible()
        assert window._fullscreen_button.isVisible()
        assert not window._fullscreen_button.isChecked()
        if toolbar is not None:
            assert toolbar.isVisible()
        if index == 0:
            assert window._controls.isVisible()
        elif index == 2:
            assert tab._map_ax.get_visible()
        elif index == 4:
            assert tab._locator.isVisible()


def test_map_toggle_restores_an_already_fullscreen_viewer(window, qapp):
    window.showFullScreen()
    qapp.processEvents()
    sizes = window._splitter.sizes()
    _click_map_toggle(window._fullscreen_button, qapp)
    assert window._map_focus is not None
    assert window._fullscreen_button.isChecked()
    assert window._picker.isHidden()
    _click_map_toggle(window._fullscreen_button, qapp)
    assert window.isFullScreen()
    assert window._map_focus is None
    assert window._picker.isVisible()
    assert window._controls.isVisible()
    assert window._splitter.sizes() == sizes
    assert not window._fullscreen_button.isChecked()


def test_map_focus_changes_only_with_its_toggle(window, qapp):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest

    window.show()
    qapp.processEvents()
    for key in (Qt.Key.Key_F11, Qt.Key.Key_Escape):
        QTest.keyClick(window, key)
        qapp.processEvents()
        assert window._map_focus is None
    _click_map_toggle(window._fullscreen_button, qapp)
    focus = window._map_focus
    for key in (Qt.Key.Key_F11, Qt.Key.Key_Escape):
        QTest.keyClick(window, key)
        qapp.processEvents()
        assert window._map_focus is focus
    # A native transition must not implicitly toggle the map layout either.
    window.showNormal()
    qapp.processEvents()
    assert window._map_focus is focus
    assert window._fullscreen_button.isChecked()
    _click_map_toggle(window._fullscreen_button, qapp)
    assert window._map_focus is None
    assert window._controls.isVisible()


def test_3d_native_fullscreen_and_map_toggle_are_independent(qapp):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest

    from xc_viewer.widgets.thermal_3d import Thermal3D

    dialog = Thermal3D()
    try:
        dialog.showFullScreen()
        qapp.processEvents()
        assert dialog._map_focus is None
        assert dialog._heading.isVisible() and dialog._point_size.isVisible()
        _click_map_toggle(dialog._fullscreen, qapp)
        assert dialog._map_focus is not None
        QTest.keyClick(dialog, Qt.Key.Key_Escape)
        qapp.processEvents()
        assert dialog.isVisible() and dialog._map_focus is not None
        _click_map_toggle(dialog._fullscreen, qapp)
        assert dialog.isFullScreen()
        assert dialog._map_focus is None
        assert dialog._heading.isVisible() and dialog._point_size.isVisible()
        dialog.showNormal()
        qapp.processEvents()
        assert dialog._heading.isVisible() and dialog._point_size.isVisible()
    finally:
        dialog.close()
