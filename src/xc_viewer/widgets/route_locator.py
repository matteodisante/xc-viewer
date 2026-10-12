"""Offline geographic locator for the displayed route and its two endpoint cells."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..geography import FRANCE_EXTENT, WORLD_EXTENT, draw_land, load_basemap
from ..thermal_geometry import unproject
from .help import help_buttons


def geographic_outline(bounds):
    """Densify a Lambert rectangle before drawing its geographic footprint."""
    west, south, east, north = bounds
    corners = np.array(
        [[west, south], [east, south], [east, north], [west, north], [west, south]]
    )
    xy = np.concatenate([np.linspace(a, b, 22) for a, b in pairwise(corners)])
    return unproject(xy[:, 0], xy[:, 1])


class RouteLocator(QWidget):
    """France/world context plus an expandable map with normal zoom and pan."""

    def __init__(self, parent=None, *, expanded=False):
        """Reuse committed Natural Earth geometry, requiring no network tiles."""
        super().__init__(parent)
        self._scene = self._dialog = None
        self._basemap = load_basemap() or {}
        self.scope = QComboBox()
        self.scope.addItems(["France", "World"])
        self.figure = Figure(figsize=(3, 3))
        self.figure.subplots_adjust(left=0.18, right=0.98, bottom=0.17, top=0.9)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumSize(100, 80)
        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.ax = self.figure.add_subplot(111)
        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.addWidget(self.scope)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(controls)
        layout.addWidget(self.canvas, 1)
        if expanded:
            self._info, self._howto = help_buttons("locator", self)
            controls.addWidget(self._info)
            controls.addWidget(self._howto)
            layout.addWidget(NavigationToolbar2QT(self.canvas, self))
        else:
            button = QPushButton("Expand map")
            button.clicked.connect(self.expand)
            layout.addWidget(button)
            self.setMinimumWidth(150)
            self.setMaximumWidth(270)
        self.scope.currentIndexChanged.connect(self.redraw)
        self.redraw()

    def set_scene(self, scene):
        """Track exactly the current scene, including an already expanded map."""
        self._scene = scene
        self.redraw()
        if self._dialog is not None:
            self._dialog.locator.set_scene(scene)

    def redraw(self, *_):
        """Locate the footprint in France or the world; A and B are 10 km cells."""
        self.ax.clear()
        world = self.scope.currentIndex() == 1
        key = "world" if world else "france"
        extent = WORLD_EXTENT if world else FRANCE_EXTENT
        draw_land(self.ax, self._basemap.get(key, {}).get("rings", []), extent)
        if self._scene is not None:
            scene = self._scene
            lon, lat = geographic_outline(scene.bounds)
            self.ax.fill(lon, lat, facecolor="#ee5522", alpha=0.18, zorder=2)
            self.ax.plot(lon, lat, color="#c34218", linewidth=1.5, zorder=3)
            cells = np.asarray(scene.pair).reshape(-1, 2)
            for label, colour, (ix, iy) in zip(
                ("A", "B")[: len(cells)],
                ("#16874a", "#b02783")[: len(cells)],
                cells,
                strict=True,
            ):
                x, y = ix * scene.cell_m, iy * scene.cell_m
                cl, ca = geographic_outline((x, y, x + scene.cell_m, y + scene.cell_m))
                self.ax.fill(cl, ca, color=colour, alpha=0.8, zorder=4)
                if not world:
                    self.ax.annotate(
                        label,
                        (float(np.mean(cl)), float(np.mean(ca))),
                        xytext=(4, 4),
                        textcoords="offset points",
                        fontsize=9,
                        color=colour,
                        weight="bold",
                        zorder=5,
                    )
            if world:
                self.ax.plot(
                    float(np.mean(lon)),
                    float(np.mean(lat)),
                    "o",
                    color="#c34218",
                    markersize=5,
                    zorder=5,
                )
        if not world:
            for name, lon, lat in (
                ("Paris", 2.35, 48.86),
                ("Lyon", 4.84, 45.76),
                ("Toulouse", 1.44, 43.6),
                ("Nice", 7.26, 43.7),
            ):
                self.ax.plot(lon, lat, ".", color="#394455", markersize=2)
                self.ax.annotate(
                    name,
                    (lon, lat),
                    xytext=(3, 2),
                    textcoords="offset points",
                    fontsize=7,
                    color="#394455",
                )
        self.ax.set_title("Route location", fontsize=10)
        self.ax.tick_params(labelsize=7)
        self.ax.set_xlabel("Longitude · Natural Earth", fontsize=7)
        self.ax.set_ylabel("Latitude", fontsize=7)
        self.canvas.draw_idle()

    def expand(self):
        """Open a larger context map without interrupting the route comparison."""
        if self._dialog is None:
            self._dialog = QDialog(self)
            self._dialog.setWindowTitle("Route location · France / world")
            self._dialog.resize(950, 700)
            layout = QVBoxLayout(self._dialog)
            self._dialog.locator = RouteLocator(self._dialog, expanded=True)
            layout.addWidget(self._dialog.locator)
        self._dialog.locator.scope.setCurrentIndex(self.scope.currentIndex())
        self._dialog.locator.set_scene(self._scene)
        self._dialog.show()
        self._dialog.raise_()

    def shutdown(self):
        """Close the larger map with its owning viewer."""
        if self._dialog is not None:
            self._dialog.close()
