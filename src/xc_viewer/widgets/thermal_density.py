"""Cumulative thermal residence time per square kilometre over regional maps."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .. import geography, thermal_regions
from ..density_maps import view_request
from ..density_view import AdaptiveDensity
from ..thermal_store import load_store
from ..thermal_time import cache_path, load_grids
from ..thermal_time_prepare import SOURCE
from .density_background import DensityBackgrounds
from .flow_layout import FlowLayout
from .help import help_buttons
from .thermal_plane import _Worker


class ThermalDensity(QWidget):
    """Prepared time grids with adaptive density and geographic backgrounds."""

    def __init__(self, parent=None):
        """Build responsive controls without reading the archive at startup."""
        super().__init__(parent)
        self._grids, self._metadata = {}, {}
        self._cells = []
        self._worker = None
        self._tried = False
        self._densities = []
        self._backdrops, self._axes_by_key, self._limits = {}, {}, {}
        self._maps = None
        self._map_note = ""
        self._map_errors = {}
        self._norm = LogNorm(0.01, 1, clip=True)
        self._area = QComboBox()
        self._area.addItem("Regions", "regions")
        self._area.addItem("Cells", "cells")
        self._item = QComboBox()
        self._item.setMinimumContentsLength(24)
        self._item.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self._background = QComboBox()
        for text, kind in (
            ("Topography + contours · IGN", "topography"),
            ("Colour map · IGN", "colour"),
            ("Aerial photo · IGN", "aerial"),
            ("Shaded relief · Esri", "relief"),
            ("None", "none"),
        ):
            self._background.addItem(text, kind)
        self._terrain = self._opacity(85, "% terrain")
        self._strength = self._opacity(80, "% density")
        self._info, self._howto = help_buttons("density", self)
        self._reload = QPushButton("Reload data")
        self._reload.setProperty("emphasis", "primary")
        self._status = QLabel("The climbing-time maps load when this tab opens.")
        self._status.setWordWrap(True)
        self._figure = Figure(figsize=(10, 6), layout="compressed")
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(350)
        self._debounce.timeout.connect(self._request_maps)
        self._canvas.mpl_connect("resize_event", lambda _: self._debounce.start())
        self._canvas.mpl_connect("scroll_event", self._scroll)
        top, look = FlowLayout(), FlowLayout()
        for widget in (self._area, self._item, self._reload, self._info, self._howto):
            top.addWidget(widget)
        look.addWidget(QLabel("Background"))
        for widget in (self._background, self._terrain, self._strength, self._toolbar):
            look.addWidget(widget)
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addLayout(look)
        layout.addWidget(self._canvas, 1)
        layout.addWidget(self._status)
        self._area.currentIndexChanged.connect(self._area_changed)
        self._item.currentIndexChanged.connect(self._draw)
        self._background.currentIndexChanged.connect(self._background_changed)
        self._terrain.valueChanged.connect(self._terrain_changed)
        self._strength.valueChanged.connect(self._strength_changed)
        self._reload.clicked.connect(self._load)
        self._area_changed()

    @staticmethod
    def _opacity(value, suffix):
        widget = QDoubleSpinBox()
        widget.setRange(0, 100)
        widget.setDecimals(0)
        widget.setSingleStep(5)
        widget.setValue(value)
        widget.setSuffix(suffix)
        return widget

    def ensure_loaded(self):
        """Load the compact prepared product when this tab is first opened."""
        if not self._tried:
            self._tried = True
            self._load()

    def invalidate(self):
        """Release old data after changing archive folders."""
        self.shutdown()
        self._grids, self._metadata, self._limits, self._cells = {}, {}, {}, []
        self._tried = False
        self._area_changed()

    def shutdown(self):
        """Finish a local read and detach pending network work from the widget."""
        self._debounce.stop()
        if self._maps:
            self._maps.close()
            self._maps.deleteLater()
            self._maps = None
        if self._worker:
            self._worker.cancel.set()
            self._worker.succeeded.disconnect()
            self._worker.wait()
            self._worker = None

    def set_compact(self, compact):
        """Hide the status line when plots fill a full-screen window."""
        self._status.setVisible(not compact)

    def _load(self):
        if self._worker:
            return
        self._status.setText("Reading prepared thermal hours from the data folder…")
        self._reload.setEnabled(False)

        def read(progress, cancel):
            grids, metadata = load_grids()
            store = load_store()
            return grids, metadata, store.cells() if store else [], cache_path().parent

        self._worker = _Worker(read, self)
        self._worker.failed.connect(self._status.setText)
        self._worker.succeeded.connect(self._received)
        self._worker.finished.connect(self._finished)
        self._worker.start()

    def _received(self, result):
        self._grids, self._metadata, self._cells, folder = result
        maximum = max(
            (
                float(g.seconds.max()) / 3600 / (g.step / 1000) ** 2
                for g in self._grids.values()
                if len(g.seconds)
            ),
            default=1,
        )
        self._norm = LogNorm(0.01, max(1, maximum), clip=True)
        if self._maps:
            self._maps.close()
            self._maps.deleteLater()
        self._maps = DensityBackgrounds(folder / "density-maps", self)
        self._maps.ready.connect(self._map_received)
        self._maps.failed.connect(self._map_failed)
        self._limits = {}
        self._area_changed()

    def _finished(self):
        worker = self.sender()
        if worker is self._worker:
            self._worker = None
            self._reload.setEnabled(True)
        worker.deleteLater()

    def _rank(self, cell):
        return [c for c in self._cells if c.terrain == cell.terrain].index(cell) + 1

    def _area_changed(self, *_):
        cells = self._area.currentData() == "cells"
        self._item.blockSignals(True)
        self._item.clear()
        if cells:
            for terrain in geography.TERRAIN_ORDER:
                group = [c for c in self._cells if c.terrain == terrain]
                if group:
                    self._item.addItem(
                        f"{terrain}: {len(group)} cells", ("group", terrain)
                    )
            for cell in self._cells:
                self._item.addItem(
                    f"{cell.terrain} #{self._rank(cell)} · ground {cell.ground_m:g} m",
                    ("cell", (cell.ix, cell.iy)),
                )
        else:
            self._item.addItem("All regions side by side", None)
            for name in thermal_regions.REGIONS:
                self._item.addItem(name, name)
        self._item.blockSignals(False)
        self._draw()

    def _panels(self):
        data = self._item.currentData()
        if self._area.currentData() == "regions":
            names = list(thermal_regions.REGIONS) if data is None else [data]
            return [(f"region/{name}/{SOURCE}", name) for name in names]
        if data is None:
            return []
        what, value = data
        cells = [
            c
            for c in self._cells
            if (c.terrain == value if what == "group" else (c.ix, c.iy) == value)
        ]
        return [
            (f"cell/{c.ix}/{c.iy}/{SOURCE}", f"{c.terrain} #{self._rank(c)}")
            for c in cells
        ]

    def _draw(self, *_):
        for key, ax in self._axes_by_key.items():
            self._limits[key.rsplit("/", 1)[0]] = ax.get_xlim(), ax.get_ylim()
        for density in self._densities:
            density.close()
        if self._maps:
            self._maps.reset()
        self._densities, self._backdrops, self._axes_by_key = [], {}, {}
        self._figure.clear()
        panels = [(key, title) for key, title in self._panels() if key in self._grids]
        if not panels:
            if self._grids and self._panels():
                only = self._area.currentData()
                self._status.setText(
                    f"This data folder has no prepared climbing-time maps for "
                    f"the {only}. Choose a complete data folder."
                )
            self._canvas.draw_idle()
            return
        columns = min(3, len(panels))
        axes = self._figure.subplots(-(-len(panels) // columns), columns, squeeze=False)
        for ax in axes.ravel()[len(panels) :]:
            ax.axis("off")
        for ax, (key, title) in zip(axes.ravel(), panels, strict=False):
            grid = self._grids[key]
            w, s, e, n = grid.bounds / 1000
            ax.set(
                xlim=(w, e),
                ylim=(s, n),
                xlabel="Lambert-93 east (km)",
                ylabel="North (km)",
                facecolor="#eef0eb",
            )
            ax.set_aspect("equal")
            ax.set_autoscale_on(False)
            frame = key.rsplit("/", 1)[0]
            if frame in self._limits:
                ax.set_xlim(self._limits[frame][0])
                ax.set_ylim(self._limits[frame][1])
            self._axes_by_key[key] = ax
            density = AdaptiveDensity(
                ax,
                grid.window,
                label="Thermal hours/km²",
                norm=self._norm,
                alpha=self._strength.value() / 100,
            )
            density.refresh()
            self._densities.append(density)

            def caption(_, ax=ax, title=title, grid=grid, density=density):
                ax.set_title(
                    f"{title} · {grid.hours:,.1f} thermal h\n"
                    f"Displayed pixels: {density._size}",
                    fontsize=10,
                )

            caption(None)
            for event in ("xlim_changed", "ylim_changed"):
                ax.callbacks.connect(event, caption)
                ax.callbacks.connect(event, lambda _: self._debounce.start())
        self._figure.colorbar(
            ScalarMappable(norm=self._norm, cmap="magma_r"),
            ax=list(self._axes_by_key.values()),
            orientation="horizontal",
            shrink=0.75,
            fraction=0.065,
            pad=0.07,
            label="Thermal hours/km² · fixed logarithmic scale",
        )
        self._toolbar.update()
        self._canvas.draw_idle()
        self._background_changed()

    def _terrain_changed(self, *_):
        for artist in self._backdrops.values():
            artist.set_alpha(self._terrain.value() / 100)
        self._canvas.draw_idle()

    def _strength_changed(self, *_):
        for density in self._densities:
            density.set_alpha(self._strength.value() / 100)
        self._canvas.draw_idle()

    def _background_changed(self, *_):
        self._map_errors.clear()
        for artist in self._backdrops.values():
            artist.remove()
        self._backdrops.clear()
        if self._maps:
            self._maps.reset()
        self._map_note = (
            "" if self._background.currentData() == "none" else "Loading map detail…"
        )
        self._update_status()
        self._debounce.start()
        self._canvas.draw_idle()

    def _request_maps(self):
        if not self._maps:
            return
        kind = self._background.currentData()
        wanted = (
            {}
            if kind == "none"
            else {
                key: view_request(kind, ax, self._canvas.devicePixelRatioF())
                for key, ax in self._axes_by_key.items()
            }
        )
        self._maps.request(wanted)

    def _map_received(self, key, result):
        if key not in self._axes_by_key:
            return
        info, pixels = result
        self._map_errors.pop(key, None)
        w, s, e, n = np.asarray(info["extent"]) / 1000
        if key in self._backdrops:
            self._backdrops[key].remove()
        ax = self._axes_by_key[key]
        self._backdrops[key] = ax.imshow(
            pixels,
            extent=(w, e, s, n),
            origin="upper",
            zorder=0.5,
            alpha=self._terrain.value() / 100,
            aspect="equal",
            interpolation="bilinear",
        )
        source = (
            "saved Thermal planes background"
            if info.get("saved_planes")
            else "new views require internet"
        )
        self._map_note = info["attribution"] + " · " + source
        self._update_status()
        self._canvas.draw_idle()

    def _map_failed(self, key, message):
        self._map_errors[key] = message
        self._update_status()

    def _update_status(self):
        if not self._grids:
            return
        self._status.setText(
            "Vilpellet · all archived dates and heights · 50 m grid · hours/km² "
            f"(not probability). {self._map_note} "
            + (
                f"{len(self._map_errors)} map(s) unavailable; "
                "previous detail retained. " + next(iter(self._map_errors.values()))
                if self._map_errors
                else ""
            )
        )

    def _scroll(self, event):
        if event.inaxes not in self._axes_by_key.values() or event.xdata is None:
            return
        ax = event.inaxes
        self._toolbar.push_current()
        factor = 0.7 if event.button == "up" else 1 / 0.7
        for limits, setter, centre in (
            (ax.get_xlim(), ax.set_xlim, event.xdata),
            (ax.get_ylim(), ax.set_ylim, event.ydata),
        ):
            setter(*(centre + (np.asarray(limits) - centre) * factor))
        self._toolbar.push_current()
        self._canvas.draw_idle()
