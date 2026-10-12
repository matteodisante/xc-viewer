"""Controls, legend and transparent all-flight thermal time on the route DEM."""

from __future__ import annotations

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import LogNorm
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .flow_layout import FlowLayout


class RouteDensity(QWidget):
    """Show a fixed archive population independently of selected route flights."""

    def __init__(self, parent=None):
        """Offer an overlay toggle, opacity and a quantitative logarithmic key."""
        super().__init__(parent)
        self.mesh = self.view = self._scene = None
        self.enabled = QCheckBox("Thermal hours · all flights")
        self.enabled.setChecked(True)
        self.enabled.setToolTip(
            "All available Vilpellet-classified flights crossing this area, both "
            "disciplines, all dates and heights; independent of the route filters"
        )
        self.opacity = QSpinBox()
        self.opacity.setRange(0, 100)
        self.opacity.setValue(75)
        self.opacity.setSuffix("% opacity")
        self.resolution = QComboBox()
        self.resolution.addItem("Thermal pixels: 250 m", 250)
        self.resolution.setToolTip(
            "Sum recorded seconds into larger ground pixels, then divide by their "
            "area. This grid is independent of the 10 km departure/arrival cells."
        )
        self.legend = QLabel()
        self.legend.setFixedWidth(160)
        self.range = QLabel()
        self.details = QLabel("Thermal density: waiting for a route")
        self.details.setWordWrap(True)
        flow = FlowLayout()
        for widget in (
            self.enabled,
            self.resolution,
            self.opacity,
            self.legend,
            self.range,
        ):
            flow.addWidget(widget)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addLayout(flow)
        layout.addWidget(self.details)
        self.enabled.toggled.connect(self.update_style)
        self.opacity.valueChanged.connect(self.update_style)
        self.resolution.currentIndexChanged.connect(self._draw_grid)
        self.clear()

    def clear(self):
        """Release the old texture before its view clears OpenGL items."""
        self._release_mesh()
        self.mesh = self.view = self._scene = None
        self.enabled.setEnabled(False)
        self.opacity.setEnabled(False)
        self.resolution.setEnabled(False)
        self.legend.clear()
        self.range.clear()
        self.details.setText("Thermal density: waiting for a route")

    def _release_mesh(self):
        """Free one resolution's texture while keeping the scene and controls."""
        if self.mesh is not None and self.view is not None:
            if self.view.isValid():
                self.view.makeCurrent()
                self.mesh.release_density()
                self.view.doneCurrent()
            self.mesh.set_density(None, self.mesh._bounds)

    def attach(self, scene, view, surface):
        """Drape a conservative density raster on the same valid DEM triangles."""
        self.clear()
        if scene.density is None:
            self.details.setText(
                "Thermal density unavailable: "
                f"{scene.density_error or 'no prepared product'}. "
                "Use Prepare / refresh index."
            )
            return
        self._scene, self.view, self.mesh = scene, view, surface
        selected = self.resolution.currentData() or 250
        self.resolution.blockSignals(True)
        self.resolution.clear()
        for step in scene.density.rasters:
            self.resolution.addItem(f"Thermal pixels: {step:g} m", step)
        self.resolution.setCurrentIndex(max(0, self.resolution.findData(selected)))
        self.resolution.blockSignals(False)
        self.resolution.setEnabled(True)
        self._draw_grid()

    def _draw_grid(self, *_):
        """Replace only the density texture, preserving the shared terrain mesh."""
        if self._scene is None:
            return
        scene = self._scene
        data = scene.density.rasters[self.resolution.currentData()]
        self._release_mesh()
        norm = LogNorm(0.01, data.maximum, clip=True)
        cmap = colormaps["magma_r"]
        key = np.ascontiguousarray(cmap(np.linspace(0, 1, 160), bytes=True)[None, :, :])
        self.legend.setPixmap(
            QPixmap.fromImage(
                QImage(
                    key.data, 160, 1, key.strides[0], QImage.Format.Format_RGBA8888
                ).copy()
            ).scaled(160, 12)
        )
        self.range.setText(f"0.01 — {data.maximum:.0f} h/km² · log")
        self.details.setText(
            f"{data.hours:,.1f} recorded climb hours in the density frame · "
            f"{data.step:g} m pixels · Vilpellet, all dates / heights / disciplines"
        )
        self.details.setToolTip(
            "Cumulative recorded climb time divided by ground area. "
            "This is not thermal probability or thermal count. "
            "Transparent pixels have no recorded climb time; they may be unvisited. "
            "Overview pixels sum the original 50 m grid. The aligned density frame "
            "can extend less than one pixel beyond the terrain."
        )
        if scene.terrain is None:
            self.details.setText(self.details.text() + " · DEM required for overlay")
            return
        rgba = cmap(norm(np.maximum(data.values, 0.01)), bytes=True)
        rgba[data.values <= 0, 3] = 0
        self.mesh.set_density(
            np.ascontiguousarray(rgba[::-1]),
            data.bounds - np.tile(scene.origin, 2),
        )
        self.enabled.setEnabled(True)
        self.opacity.setEnabled(True)
        self.update_style()

    def update_style(self, *_):
        """Change visibility and opacity without rebuilding or filtering seconds."""
        if self.mesh is not None:
            self.mesh.density_opacity = self.opacity.value() / 100
            self.mesh.set_density_visible(self.enabled.isChecked())
            self.mesh.update()
