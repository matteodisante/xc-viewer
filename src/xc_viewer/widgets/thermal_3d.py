"""Navigable GPU terrain and the complete 20 m intersection cloud of one cell."""

from __future__ import annotations

from datetime import datetime
from html import escape

import numpy as np
import pyqtgraph.opengl as gl
from matplotlib import colormaps
from OpenGL import GL
from PyQt6.QtCore import QElapsedTimer, QEvent, Qt
from PyQt6.QtGui import QMatrix4x4, QVector3D
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from ..thermal_3d import cell_label, load_scene
from ..thermal_daily import PARIS
from .flow_layout import FlowLayout
from .help import help_buttons
from .map_focus import MapFocus
from .terrain_surface import TerrainSurface
from .thermal_plane import _Worker


class TerrainView(gl.GLViewWidget):
    """Navigate with trackpad gestures, mouse drags and camera-relative movement."""

    def __init__(self, parent=None):
        """Inherit the application's shared OpenGL format and depth buffer."""
        super().__init__(parent)
        self.setBackgroundColor("#eef1f4")
        self.setMinimumSize(320, 240)
        self.scene_span = 5000
        self.noRepeatKeys += [
            Qt.Key.Key_W,
            Qt.Key.Key_A,
            Qt.Key.Key_S,
            Qt.Key.Key_D,
            Qt.Key.Key_Q,
            Qt.Key.Key_E,
        ]
        self._movement_clock = QElapsedTimer()

    def mousePressEvent(self, event):  # noqa: N802
        """Give keyboard movement to the scene when a drag starts."""
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        super().mousePressEvent(event)

    def paintGL(self, *args, **kwargs):  # noqa: N802
        """Clear depth correctly after a frame with translucent point markers."""
        GL.glDepthMask(True)
        try:
            super().paintGL(*args, **kwargs)
        finally:
            GL.glDepthMask(True)

    def mouseMoveEvent(self, event):  # noqa: N802
        """Offer one-finger pan and fixed-camera look-around on Mac trackpads."""
        pos = event.position()
        diff = pos - getattr(self, "mousePos", pos)
        left = event.buttons() == Qt.MouseButton.LeftButton
        if event.buttons() == Qt.MouseButton.RightButton or (
            left and event.modifiers() & Qt.KeyboardModifier.ShiftModifier
        ):
            self.mousePos = pos
            self.pan(diff.x(), diff.y(), 0, relative="view")
        elif left and event.modifiers() & Qt.KeyboardModifier.AltModifier:
            self.mousePos = pos
            position = self.cameraPosition()
            self.orbit(-diff.x() * 0.3, diff.y() * 0.3)
            self.opts["center"] += position - self.cameraPosition()
            self.update()
        else:
            super().mouseMoveEvent(event)

    def wheelEvent(self, event):  # noqa: N802
        """Zoom with two-finger scrolling at the original wheel sensitivity."""
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        angles = event.angleDelta()
        delta = angles.x() or angles.y()
        if not delta:
            pixels = event.pixelDelta()
            delta = pixels.y() or pixels.x()
        self._zoom(-delta * np.log(0.999))
        event.accept()

    def event(self, event):
        """Use macOS native pinch events without changing the field of view."""
        if (
            event.type() == QEvent.Type.NativeGesture
            and event.gestureType() == Qt.NativeGestureType.ZoomNativeGesture
        ):
            self.setFocus(Qt.FocusReason.MouseFocusReason)
            self._zoom(event.value())
            event.accept()
            return True
        return super().event(event)

    def _zoom(self, amount):
        """Allow close inspection while keeping a finite positive camera distance."""
        self.opts["distance"] = float(
            np.clip(
                self.opts["distance"] * np.exp(-np.clip(amount, -5, 5)),
                1,
                max(100000, self.scene_span * 4),
            )
        )
        self.update()

    def projectionMatrix(self, region, viewport):  # noqa: N802
        """Keep the terrain visible when inspecting it from inside the scene."""
        x0, y0, width, height = viewport
        distance = self.opts["distance"]
        near = max(0.01, min(distance * 0.001, 1))
        far = max(distance * 4, self.cameraPosition().length() + self.scene_span * 4)
        right = near * np.tan(np.deg2rad(self.opts["fov"] / 2))
        top = right * height / width
        matrix = QMatrix4x4()
        matrix.frustum(
            right * ((region[0] - x0) * 2 / width - 1),
            right * ((region[0] + region[2] - x0) * 2 / width - 1),
            top * ((region[1] - y0) * 2 / height - 1),
            top * ((region[1] + region[3] - y0) * 2 / height - 1),
            near,
            far,
        )
        return matrix

    def keyPressEvent(self, event):  # noqa: N802
        """Move only while the scene has focus; let Escape reach the dialog."""
        if event.key() in self.noRepeatKeys:
            super().keyPressEvent(event)
        else:
            event.ignore()

    def keyReleaseEvent(self, event):  # noqa: N802
        """Stop a released movement key without swallowing other shortcuts."""
        if event.key() in self.noRepeatKeys:
            super().keyReleaseEvent(event)
        else:
            event.ignore()

    def evalKeyState(self):  # noqa: N802
        """Translate camera and target together, including past the orbit centre."""
        if not self.keysPressed:
            self.stop_movement()
            return
        dt = (
            min(self._movement_clock.restart() / 1000, 0.1)
            if self.keyTimer.isActive()
            else 0.016
        )
        self._movement_clock.start()
        keys = self.keysPressed
        right = any(k in keys for k in (Qt.Key.Key_D, Qt.Key.Key_Right)) - any(
            k in keys for k in (Qt.Key.Key_A, Qt.Key.Key_Left)
        )
        forward = any(k in keys for k in (Qt.Key.Key_W, Qt.Key.Key_Up)) - any(
            k in keys for k in (Qt.Key.Key_S, Qt.Key.Key_Down)
        )
        up = any(k in keys for k in (Qt.Key.Key_E, Qt.Key.Key_PageUp)) - any(
            k in keys for k in (Qt.Key.Key_Q, Qt.Key.Key_PageDown)
        )
        camera_to_world, _ = self.viewMatrix().inverted()
        direction = camera_to_world.mapVector(QVector3D(right, 0, -forward))
        direction += QVector3D(0, 0, up)
        speed = max(5, self.opts["distance"] * 0.05)
        self.opts["center"] += direction.normalized() * (speed * dt)
        self.update()
        self.keyTimer.start(16)

    def stop_movement(self):
        """Release held keys when focus leaves the scene or the window closes."""
        self.keysPressed.clear()
        self.keyTimer.stop()

    def focusOutEvent(self, event):  # noqa: N802
        """Prevent a lost key release from leaving the camera moving."""
        self.stop_movement()
        super().focusOutEvent(event)

    def hideEvent(self, event):  # noqa: N802
        """Stop navigation while the scene is hidden."""
        self.stop_movement()
        super().hideEvent(event)


class Thermal3D(QDialog):
    """A modeless, cancellable snapshot of Thermal planes' selected interval."""

    def __init__(self, parent=None):
        """Create controls and defer all saved-data reads to an explicit request."""
        super().__init__(parent)
        self.setWindowTitle("3D terrain · Vilpellet")
        self.resize(1200, 850)
        self._worker = None
        self._scene = None
        self._cloud = self._surface = None
        self._request = None
        self._map_focus = None
        self._heading = QLabel("Vilpellet · planes every 20 m")
        self._heading.setWordWrap(True)
        self._summary = QLabel()
        self._summary.setWordWrap(True)
        self._status = QLabel()
        self._status.setWordWrap(True)
        self._area = QComboBox()
        self._area.addItem("5 x 5 km", 5)
        self._area.addItem("10 x 10 km", 10)
        self._area.setToolTip("Area centred on the selected cell, including neighbours")
        self._area.setEnabled(False)
        self._point_strength = QDoubleSpinBox()
        self._point_strength.setRange(0, 100)
        self._point_strength.setDecimals(0)
        self._point_strength.setSingleStep(5)
        self._point_strength.setValue(30)
        self._point_strength.setPrefix("Point opacity: ")
        self._point_strength.setSuffix("%")
        self._point_strength.setToolTip(
            "Opacity of each point: 0% invisible, 100% opaque. "
            "All intersection points remain included."
        )
        self._point_size = QDoubleSpinBox()
        self._point_size.setRange(1, 12)
        self._point_size.setSingleStep(0.5)
        self._point_size.setDecimals(1)
        self._point_size.setValue(2)
        self._point_size.setPrefix("Point size: ")
        self._point_size.setSuffix(" px")
        self._point_size.setToolTip("Point diameter in screen pixels")
        self._terrain = QCheckBox("Terrain")
        self._terrain.setChecked(True)
        self._terrain.setToolTip("Hide the surface to inspect every recorded point")
        self._surface_mode = QComboBox()
        self._surface_mode.addItems(["Terrain colours", "Aerial photo · IGN"])
        self._surface_mode.setToolTip("Drape the saved aerial photo over the same DEM")
        self._surface_mode.setEnabled(False)
        reset = QPushButton("Reset view")
        top = QPushButton("Top view")
        self._fullscreen = QPushButton("Map full screen")
        self._fullscreen.setCheckable(True)
        self._fullscreen.setAutoDefault(False)
        self._fullscreen.setToolTip(
            "Enlarge the terrain. Click again to restore the viewer controls."
        )
        self._cancel = QPushButton("Cancel loading")
        self._cancel.setEnabled(False)
        self._info, self._howto = help_buttons("terrain", self)
        controls = FlowLayout()
        for widget in (
            self._area,
            self._point_strength,
            self._point_size,
            self._terrain,
            self._surface_mode,
            reset,
            top,
            self._cancel,
            self._info,
            self._howto,
        ):
            controls.addWidget(widget)
        self._view = TerrainView(self)
        navigation = QLabel(
            "Trackpad: two-finger scroll or pinch to zoom · "
            "Shift + drag: move · Drag: orbit · Alt + drag: look around\n"
            "Click scene for keyboard: W/S forward/back · A/D left/right · "
            "Q/E down/up · Mouse wheel: zoom · "
            "Blue: Vilpellet intersections · Scale 1:1 (metres on all axes)"
        )
        navigation.setWordWrap(True)
        self._source = QLabel()
        self._source.setWordWrap(True)
        self._source.setOpenExternalLinks(True)
        layout = QVBoxLayout(self)
        layout.addWidget(self._fullscreen, alignment=Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self._heading)
        layout.addWidget(self._summary)
        layout.addLayout(controls)
        layout.addWidget(navigation)
        layout.addWidget(self._view, 1)
        layout.addWidget(self._source)
        layout.addWidget(self._status)
        self._point_strength.valueChanged.connect(self._style_changed)
        self._area.currentIndexChanged.connect(self._area_changed)
        self._point_size.valueChanged.connect(self._style_changed)
        self._terrain.toggled.connect(self._style_changed)
        self._surface_mode.currentIndexChanged.connect(self._surface_changed)
        reset.clicked.connect(self._reset_view)
        top.clicked.connect(lambda: self._reset_view(top=True))
        self._fullscreen.clicked.connect(self._toggle_fullscreen)
        self._cancel.clicked.connect(self._cancel_loading)

    def _set_heading(self, label, start, end, selection=None):
        """Identify the displayed cell and dates in the window and scene titles."""
        self.setWindowTitle(f"3D terrain · {label} · Vilpellet")
        if selection is None:
            dates = [
                datetime.fromtimestamp(t, PARIS).strftime("%Y-%m-%d %H:%M:%S")
                for t in (start, end)
            ]
            selection = f"{dates[0]} → {dates[1]} · Europe/Paris"
        self._heading.setText(
            f"{label} · {self._area.currentText()} · Vilpellet · planes every 20 m\n"
            f"{selection}"
        )

    def load(self, store, cell, start, end, *, windows=None, selection=None):
        """Replace the displayed snapshot with an explicit cell and date selection.

        ``windows`` keeps only the daily hour windows of Thermal planes, and
        ``selection`` describes them in the heading.
        """
        self.shutdown()
        self._view.clear()
        self._cloud = self._surface = self._scene = None
        self._surface_mode.setEnabled(False)
        self._request = (store, cell, start, end, windows, selection)
        self._area.setEnabled(True)
        area_km = self._area.currentData()
        self._set_heading(cell_label(store, cell), start, end, selection)
        self._summary.setText("Loading the selected interval from the data folder…")
        self._source.clear()
        self._status.clear()
        self._cancel.setEnabled(True)
        worker = self._worker = _Worker(
            lambda **kw: load_scene(
                store, cell, start, end, area_km=area_km, windows=windows, **kw
            ),
            self,
        )
        worker.progress.connect(self._progress)
        worker.succeeded.connect(self._loaded)
        worker.failed.connect(self._failed)
        worker.finished.connect(self._finished)
        worker.start()

    def _area_changed(self, *_):
        """Reload the captured cell and dates at the chosen geographic extent."""
        if self._request is not None:
            store, cell, start, end, windows, selection = self._request
            self.load(store, cell, start, end, windows=windows, selection=selection)

    def _progress(self, message):
        """Ignore progress queued by a superseded request."""
        if self.sender() is self._worker:
            self._status.setText(message)

    def _failed(self, message):
        """Show unavailable data without leaving a previous scene visible."""
        if self.sender() is self._worker:
            self._summary.setText("No 3D scene loaded")
            self._status.setText(message)

    def _loaded(self, scene):
        """Upload the mesh and all points only on the GUI thread."""
        if self.sender() is self._worker and not self._worker.cancel.is_set():
            self.set_scene(scene)

    def _finished(self):
        """Release the completed worker after all its queued results."""
        worker = self.sender()
        if worker is self._worker:
            self._worker = None
            self._cancel.setEnabled(False)
            if worker.cancel.is_set() and self._scene is None:
                self._summary.setText("Loading cancelled")
                self._status.setText("No partial intersection cloud is displayed.")
        worker.deleteLater()

    def set_scene(self, scene):
        """Render exact metric positions without vertical exaggeration or thinning."""
        self._release_texture()
        self._scene = scene
        self._view.scene_span = scene.area_km * 1000
        self._area.blockSignals(True)
        self._area.setCurrentIndex(self._area.findData(scene.area_km))
        self._area.blockSignals(False)
        self._set_heading(
            scene.label
            or f"{scene.cell.terrain} · cell {scene.cell.ix}/{scene.cell.iy}",
            scene.start,
            scene.end,
        )
        self._view.clear()
        terrain = scene.terrain
        lo, hi = float(terrain.min()), float(terrain.max())
        colours = colormaps["terrain"](
            0.25 + 0.7 * (terrain.T - lo) / max(hi - lo, 1)
        ).astype(np.float32)
        self._surface = TerrainSurface(
            image=scene.aerial,
            x=scene.x,
            y=scene.y,
            z=terrain.T,
            colors=colours,
            shader="shaded",
            smooth=True,
            glOptions="opaque",
        )
        self._view.addItem(self._surface)
        self._surface_mode.setEnabled(scene.aerial is not None)
        if scene.aerial is None:
            self._surface_mode.setCurrentIndex(0)
        self._surface_changed()
        self._cloud = gl.GLScatterPlotItem(
            pos=scene.points,
            size=self._point_size.value(),
            pxMode=True,
            color=(0.08, 0.35, 0.72, self._point_strength.value() / 100),
            glOptions="translucent",
        )
        # Keep terrain occlusion; transparent dots must not hide later dots.
        self._cloud.updateGLOptions({"glDepthMask": (False,)})
        self._cloud.setDepthValue(10)
        self._view.addItem(self._cloud)
        self._add_axes(lo)
        self._style_changed()
        self._reset_view()
        self._summary.setText(
            f"Area {scene.area_km} x {scene.area_km} km · "
            f"Central cell {scene.cell.ix}/{scene.cell.iy}: "
            f"{scene.climb_runs:,} Vilpellet climbs in the all-date ranking\n"
            f"{len(scene.points):,} intersections · "
            f"{scene.contributing_flights:,} contributing flights · "
            f"H = {scene.cell.ground_m:.2f} m + 0, 20, 40, … m"
        )
        message = "All saved intersections at 20 m levels are included; no subsampling."
        if not len(scene.points):
            message = (
                "No intersections at 20 m levels in this interval. "
                "Terrain remains visible."
            )
        if scene.unavailable_flights or scene.unclassified_flights:
            message += (
                f" Central cell: {scene.unavailable_flights:,} unavailable flights; "
                f"{scene.unclassified_flights:,} unclassified flights."
            )
        if scene.unknown_clock_flights:
            message += (
                f" {scene.unknown_clock_flights:,} central-cell flights "
                "have no usable UTC."
            )
        if scene.aerial is None:
            message += " Aerial photo unavailable."
            if scene.aerial_error:
                message += f" {scene.aerial_error}"
        self._status.setText(message)

    def _surface_changed(self, *_):
        """Change the drape and its attribution without moving the camera."""
        if self._scene is None or self._surface is None:
            return
        scene = self._scene
        aerial = self._surface_mode.currentIndex() == 1 and scene.aerial is not None
        self._surface.set_aerial(aerial)
        lo, hi = float(scene.terrain.min()), float(scene.terrain.max())
        source = (
            '<a href="https://www.data.gouv.fr/datasets/rge-alti-r">'
            "© IGN RGE ALTI · Licence Ouverte 2.0</a> · "
            f"DEM sampling {scene.reference['grid_m'][0]:g} m · "
            f"terrain {lo:.0f}-{hi:.0f} m ASL. "
            "Recorder GNSS heights are not harmonised with IGN terrain heights."
        )
        if aerial:
            reference = scene.aerial_reference or {}
            dates = ", ".join(reference.get("acquisition_dates", [])) or "unavailable"
            source += (
                '<br><a href="https://www.data.gouv.fr/datasets/bd-ortho-r">'
                "© IGN BD ORTHO · Licence Ouverte 2.0</a> · "
                f"Photo acquisition dates: {escape(dates)}. "
                "Imagery dates differ from the selected flight dates."
            )
        self._source.setText(source)

    def _release_texture(self):
        """Free the previous photo's GPU allocation in its rendering context."""
        if self._surface is not None and self._view.isValid():
            self._view.makeCurrent()
            try:
                self._surface.release_texture()
            finally:
                self._view.doneCurrent()

    def _add_axes(self, minimum):
        """Add a kilometre grid and labelled east, north and absolute altitude."""
        floor = 500 * np.floor(minimum / 500)
        width = self._scene.area_km * 1000
        half = width / 2
        grid = gl.GLGridItem(color=(90, 105, 120, 85), glOptions="translucent")
        grid.setSize(width, width)
        grid.setSpacing(1000, 1000)
        grid.translate(0, 0, floor)
        grid.setDepthValue(20)
        self._view.addItem(grid)
        labels = [
            ((half + 200, -half, floor), "E · 1 km grid"),
            ((-half, half + 200, floor), "N"),
        ]
        ceiling = max(self._scene.cell.max_alt_m, self._scene.terrain.max())
        for height in np.arange(floor, ceiling + 1, 500):
            labels.append(((-half - 100, -half - 100, height), f"{height:.0f} m ASL"))
        for pos, text in labels:
            item = gl.GLTextItem(pos=pos, text=text, color="#334155")
            item.setDepthValue(30)
            self._view.addItem(item)

    def _style_changed(self, *_):
        """Update appearance without reloading arrays or resetting the camera."""
        if self._surface is not None:
            self._surface.setVisible(self._terrain.isChecked())
        if self._cloud is not None:
            self._cloud.setData(
                color=(0.08, 0.35, 0.72, self._point_strength.value() / 100),
                size=self._point_size.value(),
            )

    def _reset_view(self, *_, top=False):
        """Frame the full cell and all displayed altitudes."""
        if self._scene is None:
            return
        scene = self._scene
        low = float(scene.terrain.min())
        high = max(
            float(scene.terrain.max()),
            float(scene.points[:, 2].max()) if len(scene.points) else low,
        )
        aspect = max(1, self._view.width() / max(self._view.height(), 1))
        half_angle = np.arctan(np.tan(np.deg2rad(30)) / aspect)
        radius = 0.5 * np.sqrt(2 * (scene.area_km * 1000) ** 2 + (high - low) ** 2)
        distance = 1.1 * radius / np.sin(half_angle)
        self._view.opts["fov"] = 60
        self._view.setCameraPosition(
            pos=QVector3D(0, 0, (low + high) / 2),
            distance=float(distance),
            elevation=90 if top else 35,
            azimuth=-90 if top else -55,
        )

    def _toggle_fullscreen(self):
        """Give just the terrain the complete screen, preserving its GL context."""
        if self._map_focus is not None:
            self._map_focus.restore()
            self._map_focus = None
            state, geometry = self._normal_state
            self.setWindowState(state)
            if not state & (
                Qt.WindowState.WindowMaximized | Qt.WindowState.WindowFullScreen
            ):
                self.setGeometry(geometry)
            self._fullscreen.setChecked(False)
            self._fullscreen.setText("Map full screen")
        else:
            self._normal_state = self.windowState(), self.geometry()
            self._fullscreen.setChecked(True)
            self._fullscreen.setText("Exit map full screen")
            self._map_focus = MapFocus(self, self._view, keep=(self._fullscreen,))
            self.showFullScreen()
            self._view.setFocus()

    def keyPressEvent(self, event):  # noqa: N802
        """Leave map focus to its explicit toggle; Escape closes ordinary dialogs."""
        if event.key() == Qt.Key.Key_Escape and self._map_focus is not None:
            event.accept()
        else:
            super().keyPressEvent(event)

    def _cancel_loading(self):
        """Cancel at the next saved flight boundary."""
        if self._worker is not None:
            self._worker.cancel.set()
            self._status.setText("Cancelling the saved-data read…")

    def shutdown(self):
        """Join before closing or replacing a request and discard queued results."""
        worker, self._worker = self._worker, None
        self._view.stop_movement()
        if worker is not None:
            worker.cancel.set()
            worker.wait()
        self._release_texture()
        self._cancel.setEnabled(False)

    def reject(self):
        """Stop loading when Escape closes the dialog."""
        self.shutdown()
        super().reject()

    def closeEvent(self, event):  # noqa: N802
        """Stop loading before the window is hidden."""
        self.shutdown()
        super().closeEvent(event)
