"""An interactive take-off density map.

Pan/zoom it (toolbar, scroll wheel, or a "Zone" shortcut); the density mesh's cells
shrink as the view narrows, and once few enough flights are in view, they draw as
individual points instead -- hoverable (a tooltip with the flight's metadata) and
clickable (loads that flight into the trajectory view).

Points are every *retained* flight's launch position (``lat0``/``lon0`` from
``flights_meta.parquet``, via :func:`xc_viewer.catalog_index.takeoff_points`) --
so a flight is always identified by where it actually started free flight, the same
origin the trajectory view's local frame uses (:mod:`xc_viewer.data`), not by a
catalog site name (which "can be wrong", :mod:`xc_viewer.catalog_index`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt6.QtCore import QPoint, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QLabel,
    QPushButton,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from xc_viewer.core.style import ILLUSTRATION_COLORS

from .. import catalog_index, geography
from ..core.disciplines import DISCIPLINES
from .flow_layout import FlowLayout, labeled_control
from .help import help_buttons

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.backend_bases import MouseEvent
    from matplotlib.collections import PathCollection, QuadMesh

# A click/hover within this many *pixels* of a plotted point selects it -- pixels, not
# data units, so the tolerance means the same thing at any zoom level.
_PICK_RADIUS_PX = 12.0

# At or below this many points inside the current view, individual dots are drawn
# (and become clickable/hoverable) instead of an aggregated density mesh.
_SCATTER_MAX_POINTS = 300

# The mesh cell adapts to how much is visible: aim for this many cells across the
# view's longitude span, no finer than the floor (metres-scale at that point is
# meaningless -- individual points take over well before it binds in practice) and
# no coarser than the cap, which only binds on wide views such as the whole world.
_TARGET_CELLS_ACROSS = 70
_MIN_CELL_DEG = 0.01
_MAX_CELL_DEG = 1.0

# How long to wait, after the view stops changing (drag, scroll, zone jump), before
# recomputing the mesh/points -- so a drag or a fast scroll redraws once at the end,
# not on every intermediate frame.
_RECOMPUTE_DELAY_MS = 150

_ZONES = [
    ("World", geography.WORLD_EXTENT),
    ("France", geography.FRANCE_EXTENT),
    ("La Reunion", geography.REUNION_EXTENT),
]

_PLOTTED_COLUMNS = [
    "flight_id",
    "season_year",
    "date",
    "lat0",
    "lon0",
    "alt0",
    "discipline",
    "region",
    "terrain",
    *catalog_index.TOOLTIP_COLUMNS,
]


def _field(row: pd.Series, key: str) -> Any | None:
    """``row[key]``, or ``None`` for anything that reads as "not really there"."""
    value = row.get(key)
    if value is None:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return value


def _tooltip_text(row: pd.Series) -> str:
    """The hover tooltip for one flight: date, pilot, category, distance, duration."""
    pilot = _field(row, "pilot")
    if pilot is not None and catalog_index.is_anonymised_pilot(str(pilot)):
        pilot = "(anonymised)"
    distance = _field(row, "distance_km")
    distance_text = f"{float(distance):.0f} km" if distance is not None else "unknown"
    duration = _field(row, "duration_s")
    # 0 means "not recorded", the same convention the IGC format itself uses for an
    # absent altitude channel (xc_viewer.core.preproc.altchannel): the catalog
    # leaves duration_s blank-as-zero for ~6% of paraglider rows
    # (docs/guide/data-on-disk.md), and a 124 km flight lasting 0 h is that, not a
    # measurement.
    duration_h = float(duration) / 3600.0 if duration else None
    duration_text = f"{duration_h:.1f} h" if duration_h is not None else "unknown"
    region = _field(row, "region")
    terrain = _field(row, "terrain")
    return (
        f"Flight {row['flight_id']} — {_field(row, 'date') or 'unknown date'}\n"
        f"Pilot: {pilot or 'unknown'}\n"
        f"Type: {_field(row, 'flight_type') or 'unknown'}   "
        f"Wing class: {_field(row, 'wing_class') or 'unknown'}\n"
        f"Distance: {distance_text}   Duration: {duration_text}\n"
        f"Takeoff: {_field(row, 'takeoff') or 'unknown'} "
        f"(dept. {_field(row, 'dept') or 'unknown'})\n"
        f"Landing: {_field(row, 'landing') or 'unknown'}\n"
        f"Region: {region or '(none of the named boxes)'}   "
        f"Terrain: {terrain or 'unclassified'}"
    )


class MapView(QWidget):
    """Emits ``flight_chosen`` -- ``(path, discipline, flight_id)`` -- on a click."""

    flight_chosen = pyqtSignal(object, object, str)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the discipline/zone controls, the map canvas, and its toolbar."""
        super().__init__(parent)
        self._points: dict[str, pd.DataFrame] = {}
        self._plotted = pd.DataFrame(columns=_PLOTTED_COLUMNS)
        self._visible = self._plotted
        self._loaded = False
        self._mode = "mesh"
        self._mesh_artist: QuadMesh | None = None
        self._scatter_artist: PathCollection | None = None
        self._cax: Axes | None = None

        self._discipline_combo = QComboBox()
        self._discipline_combo.addItem("All disciplines", None)
        for disc in DISCIPLINES.values():
            self._discipline_combo.addItem(disc.name.capitalize(), disc)
        self._zone_combo = QComboBox()
        for name, extent in _ZONES:
            self._zone_combo.addItem(name, extent)
        self._region_combo = QComboBox()
        self._region_combo.addItem("All regions", None)
        for name in geography.REGIONS:
            self._region_combo.addItem(name, name)
        self._terrain_combo = QComboBox()
        self._terrain_combo.addItem("All terrain", None)
        for name in geography.TERRAIN_ORDER:
            self._terrain_combo.addItem(name, name)
        self._btn_reload = QPushButton("Reload")
        self._info, self._howto = help_buttons("map", self)
        self._status = QLabel("Not loaded yet -- switch to this tab, or press Reload.")
        self._status.setWordWrap(True)

        top = FlowLayout()
        top.addWidget(labeled_control("Show", self._discipline_combo))
        top.addWidget(labeled_control("Zone", self._zone_combo))
        top.addWidget(labeled_control("Region", self._region_combo))
        top.addWidget(labeled_control("Terrain", self._terrain_combo))
        top.addWidget(self._btn_reload)
        top.addWidget(self._info)
        top.addWidget(self._howto)

        self._figure = Figure(figsize=(8.0, 6.0))
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addLayout(top)
        layout.addWidget(self._status)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas, 1)

        self._recompute_timer = QTimer(self)
        self._recompute_timer.setSingleShot(True)
        self._recompute_timer.timeout.connect(self._recompute_view)

        self._full_redraw()

        self._discipline_combo.currentIndexChanged.connect(self._on_filter_changed)
        self._region_combo.currentIndexChanged.connect(self._on_filter_changed)
        self._terrain_combo.currentIndexChanged.connect(self._on_filter_changed)
        self._zone_combo.currentIndexChanged.connect(self._on_zone_selected)
        self._btn_reload.clicked.connect(self.reload_points)
        self._canvas.mpl_connect("button_press_event", self._on_click)
        self._canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._canvas.mpl_connect("scroll_event", self._on_scroll)
        self._canvas.mpl_connect("figure_leave_event", lambda _e: QToolTip.hideText())

    def ensure_loaded(self) -> None:
        """Load take-off points on first use (called when this tab is first shown)."""
        if not self._loaded:
            self.reload_points()

    def invalidate(self) -> None:
        """Forget the loaded points (e.g. a discipline's folder was just repointed).

        Does not reload immediately: if this tab is not the one currently shown,
        :meth:`ensure_loaded` picks it up next time it is; the caller reloads
        directly only when this tab is already visible.
        """
        self._loaded = False

    def reload_points(self) -> None:
        """(Re-)read every discipline's take-off points from disk."""
        self._status.setText("Loading take-off points…")
        QApplication.processEvents()
        points: dict[str, pd.DataFrame] = {}
        errors: list[str] = []
        for disc in DISCIPLINES.values():
            try:
                points[disc.name] = catalog_index.takeoff_points(disc)
            except FileNotFoundError as exc:
                errors.append(str(exc))
        self._points = points
        self._loaded = True
        total = sum(len(df) for df in points.values())
        message = f"{total:,} take-off point(s) loaded."
        if errors:
            message += " " + " ".join(errors)
        self._status.setText(message)
        self._full_redraw()

    def _current_points(self) -> pd.DataFrame:
        chosen = self._discipline_combo.currentData()
        names = [chosen.name] if chosen is not None else list(self._points)
        frames = [
            self._points[name].assign(discipline=name)
            for name in names
            if name in self._points and len(self._points[name])
        ]
        if not frames:
            return self._plotted.iloc[0:0]
        points = pd.concat(frames, ignore_index=True)
        points["region"] = geography.classify_region(points["lat0"], points["lon0"])
        points["terrain"] = geography.classify_terrain(points["alt0"])

        region = self._region_combo.currentData()
        if region is not None:
            points = points[points["region"] == region]
        terrain = self._terrain_combo.currentData()
        if terrain is not None:
            points = points[points["terrain"] == terrain]
        return points.reset_index(drop=True)

    # -- redrawing --------------------------------------------------------------
    def _full_redraw(self) -> None:
        """Rebuild the basemap from scratch and reset the view to the whole world."""
        self._plotted = self._current_points()
        self._figure.clf()
        ax = self._figure.add_subplot(111)
        panels = geography.load_basemap()
        extent = geography.WORLD_EXTENT
        if panels is not None:
            geography.draw_land(ax, panels["world"]["rings"], extent)
        else:
            ax.set_xlim(extent[0], extent[2])
            ax.set_ylim(extent[1], extent[3])
        ax.set_xlabel("longitude [deg]")
        ax.set_ylabel("latitude [deg]")
        # Off, not left at the default: pcolormesh/scatter would otherwise nudge the
        # view to fit whatever they draw, fighting the zoom level on every recompute.
        ax.set_autoscale_on(False)
        ax.callbacks.connect("xlim_changed", self._on_view_changed)
        ax.callbacks.connect("ylim_changed", self._on_view_changed)
        self._mesh_artist = None
        self._scatter_artist = None
        # A dedicated, persistent colorbar axes -- cleared and hidden between
        # recomputes, never destroyed: matplotlib's own Colorbar.remove() corrupts
        # its host axes' gridspec bookkeeping on a second create/remove cycle
        # (upstream fragility, not specific to this figure), so this sidesteps it
        # entirely rather than hitting it on every third pan/zoom.
        self._cax = self._figure.add_axes((0.92, 0.15, 0.02, 0.7))
        self._cax.set_visible(False)
        self._recompute_view()

    def _on_filter_changed(self) -> None:
        # Not a full redraw: switching discipline/region/terrain should recompute the
        # density/points for the view the user is already looking at, not reset it.
        self._plotted = self._current_points()
        self._recompute_view()

    def _on_zone_selected(self) -> None:
        if not self._figure.axes:
            return
        extent = self._zone_combo.currentData()
        ax = self._figure.axes[0]
        ax.set_xlim(extent[0], extent[2])
        ax.set_ylim(extent[1], extent[3])
        ax.set_aspect(1.0 / np.cos(np.deg2rad(0.5 * (extent[1] + extent[3]))))
        self._canvas.draw_idle()

    def _on_view_changed(self, _ax) -> None:
        # Debounced: a box-zoom drag or a burst of scroll-wheel notches would
        # otherwise trigger a full histogram2d recompute on every intermediate frame.
        self._recompute_timer.start(_RECOMPUTE_DELAY_MS)

    def _recompute_view(self) -> None:
        if not self._figure.axes:
            return
        ax = self._figure.axes[0]
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        extent = (xlim[0], ylim[0], xlim[1], ylim[1])
        # Not only on a Zone jump: a manual pan/zoom/scroll (anything that lands
        # here through _on_view_changed) can drift far enough in latitude that the
        # aspect fixed at the last full redraw (the whole world's mean latitude)
        # visibly distorts shapes -- set_aspect changes only how the box is
        # rendered, not the data limits, so this cannot itself retrigger
        # xlim_changed/ylim_changed (checked empirically, not just by the docs).
        ax.set_aspect(1.0 / np.cos(np.deg2rad(0.5 * (extent[1] + extent[3]))))

        # Mesh/points are recreated every time (cheap: at most a few hundred
        # thousand points), so removing them is safe -- unlike the colorbar
        # (self._cax), which is cleared/hidden instead, never destroyed.
        if self._mesh_artist is not None:
            self._mesh_artist.remove()
            self._mesh_artist = None
        if self._scatter_artist is not None:
            self._scatter_artist.remove()
            self._scatter_artist = None
        if self._cax is not None:
            self._cax.clear()
            self._cax.set_visible(False)

        points = self._plotted
        if points.empty:
            self._visible = points
            ax.set_title("No take-off points loaded", fontsize=10, loc="left")
            self._canvas.draw_idle()
            return

        lon = points["lon0"].to_numpy(dtype=float)
        lat = points["lat0"].to_numpy(dtype=float)
        inside = (
            (lon >= extent[0])
            & (lon <= extent[2])
            & (lat >= extent[1])
            & (lat <= extent[3])
        )
        visible = points[inside]
        self._visible = visible

        if len(visible) <= _SCATTER_MAX_POINTS:
            self._mode = "points"
            if len(visible):
                self._scatter_artist = ax.scatter(
                    visible["lon0"],
                    visible["lat0"],
                    s=24,
                    c=ILLUSTRATION_COLORS["secondary"],
                    edgecolors="white",
                    linewidths=0.6,
                    zorder=3,
                )
                title = f"{len(visible)} flight(s) — hover or click a point"
            else:
                title = "No flights in view"
        else:
            self._mode = "mesh"
            span = max(extent[2] - extent[0], 1e-6)
            cell = float(
                np.clip(
                    span / _TARGET_CELLS_ACROSS,
                    _MIN_CELL_DEG,
                    min(_MAX_CELL_DEG, span),
                )
            )
            mesh = geography.draw_density(
                ax, lon[inside], lat[inside], extent, cell=cell
            )
            self._mesh_artist = mesh
            if mesh is not None and self._cax is not None:
                self._cax.set_visible(True)
                self._figure.colorbar(mesh, cax=self._cax, label="flights per cell")
            title = f"{len(visible):,} flights in view — zoom in for individual points"

        ax.set_title(title, fontsize=10, loc="left")
        self._canvas.draw_idle()

    # -- mouse interaction --------------------------------------------------------
    def _nearest_visible(
        self, ax, x_px: float, y_px: float
    ) -> tuple[int, float] | None:
        if self._visible.empty:
            return None
        xy_px = ax.transData.transform(
            self._visible[["lon0", "lat0"]].to_numpy(dtype=float)
        )
        distances = np.hypot(xy_px[:, 0] - x_px, xy_px[:, 1] - y_px)
        nearest = int(np.argmin(distances))
        return nearest, float(distances[nearest])

    def _on_click(self, event: MouseEvent) -> None:
        if self._mode != "points" or event.inaxes is None or event.x is None:
            return
        found = self._nearest_visible(event.inaxes, event.x, event.y)
        if found is None or found[1] > _PICK_RADIUS_PX:
            return
        row = self._visible.iloc[found[0]]
        discipline = DISCIPLINES[row["discipline"]]
        path = catalog_index.resolve_igc_path(discipline, row)
        if path is None:
            self._status.setText(f"Flight {row['flight_id']}: .igc file not found.")
            return
        self.flight_chosen.emit(path, discipline, str(row["flight_id"]))

    def _on_motion(self, event: MouseEvent) -> None:
        if self._mode != "points" or event.inaxes is None or event.x is None:
            QToolTip.hideText()
            return
        found = self._nearest_visible(event.inaxes, event.x, event.y)
        if found is None or found[1] > _PICK_RADIUS_PX:
            QToolTip.hideText()
            return
        row = self._visible.iloc[found[0]]
        # event.x/event.y are canvas-local pixels with matplotlib's bottom-left
        # origin; Qt's is top-left, hence the flip, before mapping to a screen
        # position QToolTip understands. Deliberately not event.guiEvent (whether a
        # backend populates it for a plain mouse-move is not guaranteed).
        local = QPoint(int(event.x), int(self._canvas.height() - event.y))
        global_pos = self._canvas.mapToGlobal(local)
        QToolTip.showText(global_pos, _tooltip_text(row), self._canvas)

    def _on_scroll(self, event: MouseEvent) -> None:
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        ax = event.inaxes
        factor = 0.85 if event.button == "up" else 1.0 / 0.85
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        x, y = event.xdata, event.ydata
        ax.set_xlim(x - (x - xlim[0]) * factor, x + (xlim[1] - x) * factor)
        ax.set_ylim(y - (y - ylim[0]) * factor, y + (ylim[1] - y) * factor)
        self._canvas.draw_idle()
