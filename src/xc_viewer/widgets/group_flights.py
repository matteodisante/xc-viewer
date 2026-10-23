"""Rank the episodes of flying in a group and replay one on IGN terrain."""

import numpy as np
from PyQt6.QtWidgets import (
    QComboBox,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .. import datafolder
from ..group_episodes import (
    ORDERS,
    attach_zones,
    load_episode_scene,
    load_tables,
    rank_episodes,
    zone_label,
)
from ..route_index import build_index, load_saved_index
from .flow_layout import FlowLayout, labeled_control
from .group_scene import EpisodeScene
from .group_zones import ZoneDayDialog
from .help import help_buttons
from .route_comparison import _RouteWorker

#: Folder of group tables inside the data folder.
GROUPS = "groups"
NO_TABLES = (
    "The data folder has no group tables: export a run of xc-lab to its groups/ folder "
    "with export_viewer.py."
)
#: Episodes listed after ranking.
LISTED = 300


def groups_folder():
    """The groups/ folder of the data folder, or None when it has no tables."""
    root = datafolder.current()
    if root is not None and (root / GROUPS / "episodes.parquet").is_file():
        return root / GROUPS
    return None


class GroupFlights(QWidget):
    """The group tables of the data folder, their episodes ranked, one replayed."""

    def __init__(self, parent=None):
        """Create the controls; nothing is read before the tab is first shown."""
        super().__init__(parent)
        self._worker = self._index = self._tables = None
        self._sources = ""
        self._tried = False
        self._prepare = QPushButton("Prepare / refresh index")
        self._prepare.setToolTip("The census of cleaned tracks shared with Routes")
        self._cancel = QPushButton("Cancel")
        self._order = QComboBox()
        for key, label in ORDERS.items():
            self._order.addItem(label, key)
        self._discipline = QComboBox()
        self._discipline.addItem("Both disciplines", None)
        for name in ("paragliders", "hang gliders"):
            self._discipline.addItem(name, name)
        self._year = QComboBox()
        self._year.addItem("All years", None)
        self._minimum = QSpinBox()
        self._minimum.setRange(2, 100)
        self._minimum.setKeyboardTracking(False)
        self._zone = self._day = self._dialog = None
        self._browse = QPushButton("Zones and days…")
        self._browse.setToolTip("Find the zones and the days with most group flights")
        self._where = QLabel()
        self._clear_where = QPushButton("All zones and days")
        self._episodes = QComboBox()
        self._episodes.setMinimumContentsLength(40)
        self._episodes.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self._load = QPushButton("Load episode")
        self._load.setProperty("emphasis", "primary")
        self._info, self._howto = help_buttons("groups", self)
        self._status = QLabel(NO_TABLES)
        self._status.setWordWrap(True)
        self._viewer = EpisodeScene(self)
        top = FlowLayout()
        for widget in (
            self._prepare,
            self._cancel,
            self._info,
            self._howto,
            labeled_control("Rank by", self._order),
            self._discipline,
            self._year,
            labeled_control("Minimum flights", self._minimum),
            self._browse,
            self._where,
            self._clear_where,
            labeled_control("Episode", self._episodes),
            self._load,
        ):
            top.addWidget(widget)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(top)
        layout.addWidget(self._viewer, 1)
        layout.addWidget(self._status)
        self._prepare.clicked.connect(self._prepare_index)
        self._cancel.clicked.connect(self._cancel_work)
        self._load.clicked.connect(self._load_episode)
        self._browse.clicked.connect(self._browse_zones)
        self._clear_where.clicked.connect(lambda: self._choose_where(None, None))
        for combo in (self._order, self._discipline, self._year):
            combo.currentIndexChanged.connect(self._rank)
        self._minimum.valueChanged.connect(self._rank)
        self._episodes.currentIndexChanged.connect(
            lambda _: self._viewer._clear_scene()
        )
        self._update_where()
        self._set_busy(False)

    # -- sources -------------------------------------------------------------------

    def ensure_loaded(self):
        """Read the group tables and the route index when the tab first shows."""
        if self._tried:
            return
        self._tried = True
        self._read_sources(groups_folder(), prepare=False)

    def _prepare_index(self):
        """Build the census of cleaned tracks, then read the tables again."""
        self._read_sources(groups_folder(), prepare=True)

    def _read_sources(self, folder, *, prepare):
        def operation(**kw):
            index = build_index(**kw) if prepare else load_saved_index()
            tables = load_tables(folder) if folder is not None else None
            if tables is not None and index is not None:
                tables = attach_zones(tables, index.flights())
            return index, tables

        self._viewer._clear_scene()
        self._run(operation, "sources")

    # -- workers -------------------------------------------------------------------

    def _run(self, operation, kind):
        """One job at a time; a result is used only if its job is still current."""
        if self._worker is not None:
            return
        worker = self._worker = _RouteWorker(operation, self)
        worker.kind = kind
        worker.progress.connect(self._progress)
        worker.failed.connect(self._failed)
        worker.succeeded.connect(self._received)
        worker.finished.connect(self._finished)
        self._status.setText("Loading…")
        self._set_busy(True)
        worker.start()

    def _progress(self, message):
        if self.sender() is self._worker:
            self._status.setText(message)

    def _failed(self, message):
        if self.sender() is self._worker:
            self._status.setText(message)

    def _received(self, result):
        if self.sender() is not self._worker or self._worker.cancel.is_set():
            return
        if self._worker.kind == "sources":
            self._index, self._tables = result
            self._show_sources()
        else:
            self._viewer._mode.setCurrentIndex(0)
            self._viewer.set_episode(*result)
            self._status.setText(
                "Episode loaded. Play, or drag the time slider; select rows to "
                "highlight members."
            )

    def _finished(self):
        worker = self.sender()
        if worker is self._worker:
            self._worker = None
            self._set_busy(False)
        worker.deleteLater()

    def _set_busy(self, busy):
        ready = self._tables is not None and self._index is not None
        for widget in (self._order, self._discipline, self._year, self._minimum):
            widget.setEnabled(not busy and self._tables is not None)
        self._episodes.setEnabled(not busy and self._episodes.count() > 0)
        self._browse.setEnabled(not busy and ready)
        self._clear_where.setEnabled(
            not busy and (self._zone is not None or self._day is not None)
        )
        self._prepare.setEnabled(not busy)
        self._cancel.setEnabled(busy)
        self._load.setEnabled(not busy and ready and self._episodes.count() > 0)

    def _cancel_work(self):
        if self._worker is not None:
            self._worker.cancel.set()
            self._status.setText("Cancelling…")

    # -- ranking -------------------------------------------------------------------

    def _show_sources(self):
        """Describe the tables, offer their years and rank their episodes."""
        tables = self._tables
        if tables is None:
            self._sources = ""
            self._status.setText(NO_TABLES)
            return
        c = tables.calibration
        p = c.get("params", {})
        tau = c.get("tau_star")
        self._sources = (
            f"{len(tables.episodes):,} episodes of {len(tables.flights):,} flights"
            + (
                f" · contact within {p['eps_h']:.0f} m, {p['eps_z']:.0f} m in altitude "
                f"and {p['delta'] / 60:.0f} min, lasting at least {tau / 60:.1f} min"
                if tau is not None and p
                else ""
            )
        )
        distance = self._order.model().item(self._order.findData("distance"))
        distance.setEnabled(tables.has_distance)
        self._year.blockSignals(True)
        self._year.clear()
        self._year.addItem("All years", None)
        for year in sorted(tables.episodes.date.str[:4].unique(), reverse=True):
            self._year.addItem(year, int(year))
        self._year.blockSignals(False)
        self._zone = self._day = None
        self._update_where()
        if self._index is None:
            self._status.setText(
                "No census of cleaned tracks yet: click Prepare / refresh index."
            )
        self._rank()

    def _rank(self, *_):
        """List the best episodes for the chosen order and filters."""
        self._viewer._clear_scene()
        self._episodes.blockSignals(True)
        self._episodes.clear()
        if self._tables is not None:
            ranked = rank_episodes(
                self._tables,
                self._order.currentData(),
                **self._filters(),
                zone=self._zone,
                day=self._day,
            )
            for k, row in enumerate(ranked.head(LISTED).itertuples(), 1):
                distance = (
                    f" · {row.distance_m / 1000:.0f} km"
                    if np.isfinite(row.distance_m)
                    else ""
                )
                zone_x = getattr(row, "zone_x", np.nan)
                where = (
                    f" · {zone_label((zone_x, row.zone_y))}"
                    if self._zone is None and np.isfinite(zone_x)
                    else ""
                )
                self._episodes.addItem(
                    f"{k}. {row.date} · {row.n_flights} flights (largest "
                    f"{row.n_max}) · {row.duration_s / 3600:.1f} h{distance}{where}",
                    int(row.episode_id),
                )
            if self._index is not None:
                self._status.setText(
                    (
                        f"{len(ranked):,} episodes match; load one of them."
                        if len(ranked)
                        else "No episode matches these filters."
                    )
                    + f" Tables: {self._sources}."
                )
        self._episodes.blockSignals(False)
        self._set_busy(self._worker is not None)

    def _filters(self):
        """The discipline, year and minimum flights, as `rank_episodes` takes them."""
        return {
            "discipline": self._discipline.currentData(),
            "year": self._year.currentData(),
            "minimum": self._minimum.value(),
        }

    def _browse_zones(self):
        """Open the map of the zones and the list of the days, from the filters."""
        if self._tables is None:
            return
        if self._dialog is None:
            self._dialog = ZoneDayDialog(self)
            self._dialog.chosen.connect(self._choose_where)
        self._dialog.set_data(self._tables, self._filters(), self._zone, self._day)
        self._dialog.show()
        self._dialog.raise_()

    def _choose_where(self, zone, day):
        """List the episodes of a zone and a day; None means all of them."""
        self._zone, self._day = zone, day
        self._update_where()
        self._rank()

    def _update_where(self):
        """Say which zone and day the episodes are limited to."""
        zone = "all zones" if self._zone is None else f"zone {zone_label(self._zone)}"
        day = "all days" if self._day is None else f"day {self._day}"
        self._where.setText(f"{zone} · {day}")
        self._clear_where.setEnabled(self._zone is not None or self._day is not None)

    def _load_episode(self):
        """Read the members' tracks and terrain in the background."""
        episode_id = self._episodes.currentData()
        if episode_id is None or self._tables is None or self._index is None:
            return
        index, tables = self._index, self._tables
        self._viewer._clear_scene()
        self._run(
            lambda **kw: load_episode_scene(index, tables, episode_id, **kw), "scene"
        )

    # -- life cycle ----------------------------------------------------------------

    def shutdown(self):
        """Stop workers before Qt destroys their owner or the shared OpenGL view."""
        if self._worker is not None:
            worker, self._worker = self._worker, None
            worker.cancel.set()
            worker.wait()
            worker.deleteLater()
        if self._dialog is not None:
            self._dialog.close()
        self._viewer.shutdown()

    def invalidate(self):
        """Forget the index and the scene when the data folder changes."""
        self.shutdown()
        self._index = None
        self._zone = self._day = None
        self._update_where()
        self._tried = False
        self._viewer._clear_scene()
        self._set_busy(False)
