"""The terrain window preserves geometry and camera while appearance changes."""

import os
import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("pyqtgraph.opengl")

from xc_viewer.thermal_3d import TerrainScene
from xc_viewer.thermal_geometry import ThermalCell
from xc_viewer.widgets.thermal_3d import Thermal3D
from xc_viewer.widgets.thermal_plane import ThermalPlane


@pytest.fixture
def scene():
    return TerrainScene(
        cell=ThermalCell(193, 1309, "High mountains", 10, 2, 1500, 2500),
        x=np.array([-2500, 2500]),
        y=np.array([-2500, 2500]),
        terrain=np.array([[1000, 1500], [2000, 2500]], dtype=np.float32),
        points=np.array([[0, 0, 2020], [100, 100, 2040]], dtype=np.float32),
        reference={"grid_m": [25, 25]},
        climb_runs=20,
        contributing_flights=2,
        selected_flights=3,
        unavailable_flights=0,
        unclassified_flights=1,
        unknown_clock_flights=0,
        start=100,
        end=200,
    )


def test_opacity_size_and_terrain_toggle_keep_positions_and_camera(qapp, scene):
    view = Thermal3D()
    try:
        view.set_scene(scene)
        cloud, surface = view._cloud, view._surface
        view._view.setCameraPosition(azimuth=12, elevation=55, distance=7000)
        for strength in (0, 35, 100):
            view._point_strength.setValue(strength)
            assert cloud.color[-1] == strength / 100
        view._point_size.setValue(5)
        view._terrain.setChecked(False)
        assert cloud.size == 5 and not surface.visible()
        assert view._cloud is cloud and view._surface is surface
        np.testing.assert_array_equal(cloud.pos, scene.points)
        assert [view._view.opts[k] for k in ("azimuth", "elevation", "distance")] == [
            12,
            55,
            7000,
        ]
        view._terrain.setChecked(True)
        assert surface.visible()
        view._reset_view(top=True)
        assert view._view.opts["elevation"] == 90
    finally:
        view.close()


def test_orthophoto_switch_preserves_mesh_points_camera_and_hidden_terrain(qapp, scene):
    scene = replace(
        scene,
        aerial=np.zeros((32, 32, 3), dtype=np.uint8),
        aerial_reference={"acquisition_dates": ["2025-06-25"]},
    )
    view = Thermal3D()
    try:
        view.set_scene(scene)
        cloud, surface = view._cloud, view._surface
        mesh = surface.opts["meshdata"]
        view._view.setCameraPosition(azimuth=12, elevation=55, distance=7000)
        for mode in (1, 0, 1):
            view._surface_mode.setCurrentIndex(mode)
            assert surface._aerial == bool(mode)
            assert ("2025-06-25" in view._source.text()) == bool(mode)
            assert view._surface is surface and surface.opts["meshdata"] is mesh
            assert view._cloud is cloud
            np.testing.assert_array_equal(cloud.pos, scene.points)
            assert [
                view._view.opts[k] for k in ("azimuth", "elevation", "distance")
            ] == [
                12,
                55,
                7000,
            ]
        view._terrain.setChecked(False)
        view._surface_mode.setCurrentIndex(0)
        assert not surface.visible()
        view.set_scene(replace(scene, aerial=None))
        assert not view._surface_mode.isEnabled()
        assert view._surface_mode.currentIndex() == 0
        assert "Aerial photo unavailable" in view._status.text()
    finally:
        view.close()


def test_area_selector_reloads_same_dates_and_frames_the_complete_area(
    qapp, scene, monkeypatch
):
    from PyQt6.QtTest import QTest

    calls = []
    store = SimpleNamespace(
        has_climb_ranking=True,
        cells=lambda: [scene.cell],
        activity_counts={(scene.cell.ix, scene.cell.iy): {"climb_runs": 20}},
    )

    def load(selected_store, cell, start, end, *, area_km, **kwargs):
        calls.append((selected_store, cell, start, end, area_km))
        half = area_km * 500
        return replace(
            scene, x=np.array([-half, half]), y=np.array([-half, half]), area_km=area_km
        )

    monkeypatch.setattr("xc_viewer.widgets.thermal_3d.load_scene", load)
    view = Thermal3D()
    try:
        assert not view._area.isEnabled()
        view._point_strength.setValue(65)
        view._point_size.setValue(4)
        view.load(store, scene.cell, 100, 200)
        distances = []
        for index in (0, 1, 0):
            view._area.setCurrentIndex(index)
            deadline = time.monotonic() + 5
            while view._worker is not None and time.monotonic() < deadline:
                QTest.qWait(10)
            assert view._worker is None
            assert view._scene is not None
            assert view._area.isEnabled()
            area = (5, 10)[index]
            assert view._scene.area_km == area
            assert f"{area} x {area} km" in view._summary.text()
            assert view._cloud.color[-1] == 0.65 and view._cloud.size == 4
            grid = next(i for i in view._view.items if type(i).__name__ == "GLGridItem")
            assert grid.size()[:2] == [area * 1000, area * 1000]
            distances.append(view._view.opts["distance"])
        assert distances[1] > distances[0]
        assert distances[2] == distances[0]
        assert calls == [(store, scene.cell, 100, 200, area) for area in (5, 10, 5)]
    finally:
        view.close()


def test_button_captures_selected_cell_and_dates_and_reuses_window(
    qapp, scene, monkeypatch
):
    calls = []

    class Window:
        def __init__(self, parent):
            self.parent = parent

        def load(self, store, cell, start, end, *, windows, selection):
            calls.append((store, cell, start, end, windows, selection))

        def show(self):
            pass

        def raise_(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr("xc_viewer.widgets.thermal_3d.Thermal3D", Window)
    view = ThermalPlane()
    try:
        assert not view._terrain_3d.isEnabled()
        store = view._index = SimpleNamespace(has_climb_ranking=True, has_points=True)
        view._set_busy(False)
        assert not view._terrain_3d.isEnabled()
        cells = [scene.cell, replace(scene.cell, ix=89, iy=1375, terrain="Plains")]
        view._cells.blockSignals(True)
        for cell in cells:
            view._cells.addItem(cell.terrain, cell)
        view._set_busy(False)
        assert view._terrain_3d.isEnabled()
        monkeypatch.setattr(view, "_read_bounds", lambda: (100, 200))
        view._terrain_3d.click()
        panel = view._terrain_3d_panel
        view._cells.setCurrentIndex(1)
        monkeypatch.setattr(view, "_read_bounds", lambda: (300, 400))
        view._terrain_3d.click()
        assert view._terrain_3d_panel is panel
        assert calls == [
            (store, cells[0], 100, 200, None, None),
            (store, cells[1], 300, 400, None, None),
        ]
    finally:
        view.shutdown()
        view.close()


@pytest.mark.skipif(
    os.environ.get("XC_VIEWER_NATIVE_OPENGL") != "1",
    reason="Requires a native display and GPU; offscreen Qt cannot test compositing",
)
def test_visible_dialog_shares_context_and_keeps_rendering(qapp, scene, monkeypatch):
    from PyQt6.QtCore import QPoint
    from PyQt6.QtGui import QImage, QOpenGLContext
    from PyQt6.QtTest import QTest

    from xc_viewer.main_window import MainWindow
    from xc_viewer.widgets.flight_picker import FlightPicker

    monkeypatch.setattr(FlightPicker, "_repopulate_filter_combos", lambda _: None)
    window = MainWindow()
    panel = Thermal3D(window._thermal_plane)
    try:
        window.show()
        assert QTest.qWaitForWindowExposed(window)
        panel.show()
        assert QTest.qWaitForWindowExposed(panel)
        panel.set_scene(scene)
        qapp.processEvents()
        context = panel._view.context()
        shared = QOpenGLContext.globalShareContext()
        assert context is not None and context.isValid()
        assert shared is not None and QOpenGLContext.areSharing(context, shared)
        for terrain in (True, False, True):
            panel._terrain.setChecked(terrain)
            panel._point_size.setValue(1)
            panel._point_strength.setValue(75)
            panel.resize(1000 if terrain else 900, 750)
            panel._view.orbit(20, 10)
            qapp.processEvents()
            # Inspect Qt's composed dialog, not just the independent GL buffer.
            image = panel.grab().toImage()
            origin = panel._view.mapTo(panel, QPoint(0, 0))
            ratio = image.devicePixelRatio()
            image = image.copy(
                int(origin.x() * ratio),
                int(origin.y() * ratio),
                int(panel._view.width() * ratio),
                int(panel._view.height() * ratio),
            ).convertToFormat(QImage.Format.Format_RGBA8888)
            pixels = np.frombuffer(
                image.bits().asstring(image.sizeInBytes()), dtype=np.uint8
            ).reshape(image.height(), image.width(), 4)
            assert np.mean(pixels[:, :, :3].max(axis=2) < 10) < 0.05
    finally:
        panel.close()
        window.close()


@pytest.mark.skipif(
    os.environ.get("XC_VIEWER_NATIVE_OPENGL") != "1",
    reason="Requires a native display and GPU",
)
def test_native_orthophoto_orientation_and_texture_lifecycle(qapp, scene):
    from PyQt6.QtGui import QImage
    from PyQt6.QtTest import QTest

    image = np.empty((64, 64, 3), dtype=np.uint8)
    image[:32, :32] = (255, 0, 0)  # NW
    image[:32, 32:] = (0, 255, 0)  # NE
    image[32:, :32] = (0, 0, 255)  # SW
    image[32:, 32:] = (255, 255, 0)  # SE
    scene = replace(
        scene,
        aerial=image,
        terrain=np.full((2, 2), 1000, dtype=np.float32),
        points=np.empty((0, 3), dtype=np.float32),
    )
    panel = Thermal3D()
    try:
        panel.show()
        assert QTest.qWaitForWindowExposed(panel)
        panel.set_scene(scene)
        panel._surface_mode.setCurrentIndex(1)
        panel._reset_view(top=True)
        qapp.processEvents()
        frame = panel._view.grabFramebuffer().convertToFormat(
            QImage.Format.Format_RGBA8888
        )
        pixels = np.frombuffer(
            frame.bits().asstring(frame.sizeInBytes()), dtype=np.uint8
        )
        pixels = pixels.reshape(frame.height(), frame.width(), 4)
        cy, cx = frame.height() // 2, frame.width() // 2
        offset = min(cy, cx) // 4
        for dy, dx, expected in (
            (-offset, -offset, (255, 0, 0)),
            (-offset, offset, (0, 255, 0)),
            (offset, -offset, (0, 0, 255)),
            (offset, offset, (255, 255, 0)),
        ):
            np.testing.assert_allclose(pixels[cy + dy, cx + dx, :3], expected, atol=3)
        surface = panel._surface
        assert surface._texture is not None and surface._texture.isCreated()
        panel.set_scene(scene)
        assert surface._texture is None
        qapp.processEvents()
        panel._view.grabFramebuffer()
        assert panel._surface._texture.isCreated()
        panel.close()
        assert panel._surface._texture is None
    finally:
        panel.close()


@pytest.mark.skipif(
    os.environ.get("XC_VIEWER_NATIVE_OPENGL") != "1",
    reason="Requires a native display and GPU",
)
def test_native_orthophoto_keeps_xy_coordinates_on_slopes(qapp, scene):
    """Projected landmarks must keep their photo colours on an inclined surface."""
    from PyQt6.QtCore import QRect
    from PyQt6.QtGui import QImage, QVector3D
    from PyQt6.QtTest import QTest

    # Encode east/north position in the photo, independently of the DEM heights.
    samples = np.rint((np.arange(256) + 0.5) / 256 * 255).astype(np.uint8)
    image = np.empty((256, 256, 3), dtype=np.uint8)
    image[:, :, 0] = samples[None, :]
    image[:, :, 1] = samples[:, None]
    image[:, :, 2] = 128
    terrain = 3000 + 0.5 * scene.x[None, :] + 0.2 * scene.y[:, None]
    scene = replace(
        scene,
        aerial=image,
        terrain=terrain.astype(np.float32),
        points=np.empty((0, 3), dtype=np.float32),
    )
    panel = Thermal3D()
    try:
        panel.show()
        assert QTest.qWaitForWindowExposed(panel)
        panel.set_scene(scene)
        panel._surface_mode.setCurrentIndex(1)
        panel._view.setCameraPosition(azimuth=-60, elevation=45)
        qapp.processEvents()
        frame = panel._view.grabFramebuffer().convertToFormat(
            QImage.Format.Format_RGBA8888
        )
        pixels = np.frombuffer(
            frame.bits().asstring(frame.sizeInBytes()), dtype=np.uint8
        ).reshape(frame.height(), frame.width(), 4)
        surface = panel._surface
        viewport = QRect(0, 0, frame.width(), frame.height())
        for x, y in [
            (-1700, -900),
            (300, -1600),
            (1200, 800),
            (-800, 1400),
            (500, 400),
        ]:
            point = QVector3D(x, y, 3000 + 0.5 * x + 0.2 * y)
            screen = point.project(
                surface.modelViewMatrix(), surface.projectionMatrix(), viewport
            )
            row, col = frame.height() - 1 - int(screen.y()), int(screen.x())
            expected = [255 * (x + 2500) / 5000, 255 * (2500 - y) / 5000, 128]
            np.testing.assert_allclose(pixels[row, col, :3], expected, atol=3)
    finally:
        panel.close()
