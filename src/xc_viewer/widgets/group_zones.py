"""Pick where and when to look for group flights: active zones on a map, active days."""

from __future__ import annotations

import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.collections import PolyCollection
from matplotlib.colors import ListedColormap, LogNorm
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from ..geography import FRANCE_EXTENT, draw_land, load_basemap
from ..group_episodes import METRICS, activity, rank_episodes, zone_label
from ..route_index import ROUTE_CELL_M
from ..thermal_geometry import project, unproject

#: Days listed, the most active first.
LISTED_DAYS = 500
#: Degrees of margin around the active zones when the map first opens on them.
MARGIN_DEG = 0.4
HEADERS = ["Day", "Episodes", "Flights", "Hours"]
#: Orange to dark red: the palest tones would vanish on the pale land of the basemap.
PALETTE = ListedColormap(colormaps["YlOrRd"](np.linspace(0.25, 1, 256)))


def cell_corners(zones):
    """Longitude and latitude of the four corners of each 10 km cell ``(ix, iy)``."""
    ix, iy = np.asarray(zones, float).reshape(-1, 2).T
    x = np.stack([ix, ix + 1, ix + 1, ix], axis=1) * ROUTE_CELL_M
    y = np.stack([iy, iy, iy + 1, iy + 1], axis=1) * ROUTE_CELL_M
    lon, lat = unproject(x.ravel(), y.ravel())
    return np.stack([lon.reshape(x.shape), lat.reshape(x.shape)], axis=2)


def zone_at(lon, lat):
    """The 10 km cell ``(ix, iy)`` that contains a longitude and latitude."""
    x, y = project(lon, lat)
    return int(np.floor(float(x) / ROUTE_CELL_M)), int(
        np.floor(float(y) / ROUTE_CELL_M)
    )


class ZoneDayDialog(QDialog):
    """A map of the zones and a list of the days of the episodes, to choose from.

    Clicking a zone narrows the days to that zone, clicking a day narrows the map to
    that day; both counts follow the filters the tab passes to `set_data`.
    """

    #: The zone ``(ix, iy)`` and the day ``YYYY-MM-DD``; each is None when not chosen.
    chosen = pyqtSignal(object, object)

    def __init__(self, parent=None):
        """Build the controls; nothing is drawn before `set_data`."""
        super().__init__(parent)
        self.setWindowTitle("Group flights · zones and days")
        self.resize(1150, 720)
        self._tables = None
        self._filters = {}
        self._zone = self._day = None
        self._zones = set()
        self._ax = self._view = None
        self._basemap = load_basemap() or {}
        self._metric = QComboBox()
        for key, label in METRICS.items():
            self._metric.addItem(label, key)
        self._where = QLabel()
        self._figure = Figure(figsize=(6, 6))
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        self._days = QTableWidget(0, len(HEADERS))
        self._days.setHorizontalHeaderLabels(HEADERS)
        self._days.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._days.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._days.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._days.verticalHeader().hide()
        self._clear_zone = QPushButton("Clear zone")
        self._clear_day = QPushButton("Clear day")
        self._show = QPushButton("Show these episodes")
        self._show.setProperty("emphasis", "primary")
        close = QPushButton("Close")
        top = QHBoxLayout()
        top.addWidget(QLabel("Count"))
        top.addWidget(self._metric)
        top.addWidget(self._where, 1)
        left = QVBoxLayout()
        left.addWidget(self._toolbar)
        left.addWidget(self._canvas, 1)
        right = QVBoxLayout()
        right.addWidget(QLabel("Days, the most active first; pick one"))
        right.addWidget(self._days, 1)
        body = QHBoxLayout()
        body.addLayout(left, 3)
        body.addLayout(right, 2)
        bottom = QHBoxLayout()
        for widget in (self._clear_zone, self._clear_day):
            bottom.addWidget(widget)
        bottom.addStretch(1)
        bottom.addWidget(self._show)
        bottom.addWidget(close)
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addLayout(body, 1)
        layout.addLayout(bottom)
        self._canvas.mpl_connect("button_press_event", self._clicked)
        self._metric.currentIndexChanged.connect(self._refresh)
        self._days.itemSelectionChanged.connect(self._day_picked)
        self._clear_zone.clicked.connect(lambda: self.choose(None, self._day))
        self._clear_day.clicked.connect(lambda: self.choose(self._zone, None))
        self._show.clicked.connect(self._accept)
        close.clicked.connect(self.hide)

    def set_data(self, tables, filters, zone, day):
        """Count the episodes of `tables` that pass `filters`, from the zone and day.

        `filters` are the keyword arguments of `rank_episodes` other than zone and day.
        """
        self._tables, self._filters = tables, filters
        self._zone, self._day = zone, day
        self._view = None
        self._refresh()

    def choose(self, zone, day):
        """Set the zone and the day and redraw both views."""
        self._zone, self._day = zone, day
        self._refresh()

    # -- drawing -------------------------------------------------------------------

    def _refresh(self, *_):
        self._show.setEnabled(self._tables is not None)
        self._set_label()
        self._draw_map()
        self._fill_days()

    def _set_label(self):
        """Say which zone and day are chosen, and enable what can be cleared."""
        self._clear_zone.setEnabled(self._zone is not None)
        self._clear_day.setEnabled(self._day is not None)
        zone = zone_label(self._zone) if self._zone is not None else "all"
        self._where.setText(f"  Zone: {zone} · Day: {self._day or 'all'}")

    def _draw_map(self):
        """The zones, coloured by the count of the episodes of the chosen day."""
        if self._view is not None:
            self._view = self._ax.get_xlim(), self._ax.get_ylim()
        self._figure.clear()
        self._ax = ax = self._figure.add_subplot(111)
        self._figure.subplots_adjust(left=0.1, right=0.84, bottom=0.08, top=0.93)
        draw_land(ax, self._basemap.get("france", {}).get("rings", []), FRANCE_EXTENT)
        self._zones = set()
        if self._tables is not None:
            episodes = rank_episodes(
                self._tables, "flights", **self._filters, day=self._day
            )
            zones = activity(episodes, ["zone_x", "zone_y"])
            metric = self._metric.currentData()
            self._zones = {
                (int(x), int(y))
                for x, y in zip(zones.zone_x, zones.zone_y, strict=True)
            }
            if len(zones):
                values = zones[metric].to_numpy(float)
                low = values.min()
                corners = cell_corners(zones[["zone_x", "zone_y"]].to_numpy())
                cells = PolyCollection(
                    corners,
                    cmap=PALETTE,
                    norm=LogNorm(vmin=low, vmax=max(values.max(), low * 1.01)),
                    edgecolors="none",
                    zorder=3,
                )
                cells.set_array(values)
                ax.add_collection(cells)
                self._figure.colorbar(cells, ax=ax, fraction=0.04, pad=0.02).set_label(
                    METRICS[metric], fontsize=8
                )
                if self._view is None:
                    lon, lat = corners[..., 0], corners[..., 1]
                    self._view = (
                        (lon.min() - MARGIN_DEG, lon.max() + MARGIN_DEG),
                        (lat.min() - MARGIN_DEG, lat.max() + MARGIN_DEG),
                    )
            ax.set_title(
                f"{len(episodes):,} episodes in {len(zones):,} zones"
                + (f" on {self._day}" if self._day else ""),
                fontsize=10,
            )
        if self._zone is not None:
            ax.add_patch(
                Polygon(
                    cell_corners([self._zone])[0],
                    fill=False,
                    edgecolor="#1565c0",
                    linewidth=2,
                    zorder=5,
                )
            )
        if self._view is not None:
            ax.set_xlim(*self._view[0])
            ax.set_ylim(*self._view[1])
        else:
            self._view = (ax.get_xlim(), ax.get_ylim())
        ax.tick_params(labelsize=7)
        self._canvas.draw_idle()

    def _fill_days(self):
        """The days of the chosen zone, the most active first for the chosen count."""
        self._days.blockSignals(True)
        self._days.setRowCount(0)
        if self._tables is not None:
            episodes = rank_episodes(
                self._tables, "flights", **self._filters, zone=self._zone
            )
            order = list(dict.fromkeys([self._metric.currentData(), *METRICS]))
            days = (
                activity(episodes, ["date"])
                .sort_values(order, ascending=False, kind="stable")
                .head(LISTED_DAYS)
            )
            self._days.setRowCount(len(days))
            for row, day in enumerate(days.itertuples()):
                texts = (
                    day.date,
                    f"{day.episodes}",
                    f"{day.flights}",
                    f"{day.hours:.1f}",
                )
                for column, text in enumerate(texts):
                    item = QTableWidgetItem(text)
                    if column:
                        item.setTextAlignment(Qt.AlignmentFlag.AlignRight)
                    self._days.setItem(row, column, item)
                if day.date == self._day:
                    self._days.selectRow(row)
            self._days.resizeColumnsToContents()
        self._days.blockSignals(False)

    # -- choosing ------------------------------------------------------------------

    def _clicked(self, event):
        """Pick the zone under a click on the map, when the toolbar is not zooming."""
        if (
            event.inaxes is not self._ax
            or self._toolbar.mode != ""
            or event.button != 1
        ):
            return
        zone = zone_at(event.xdata, event.ydata)
        if zone in self._zones:
            self.choose(None if zone == self._zone else zone, self._day)

    def _day_picked(self):
        rows = self._days.selectionModel().selectedRows()
        day = self._days.item(rows[0].row(), 0).text() if rows else None
        if day != self._day:
            self._day = day
            self._set_label()
            self._draw_map()

    def _accept(self):
        self.chosen.emit(self._zone, self._day)
        self.hide()
