"""Rank launch cells and explore disjoint same-day groups on IGN terrain."""

from datetime import datetime

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..group_flights import load_catalog, load_group_scene, ordered_cells, rank_groups
from ..route_density import prepare_density
from ..route_index import build_index, load_saved_index
from ..thermal_daily import PARIS
from .flow_layout import FlowLayout, labeled_control
from .group_scene import GroupScene
from .help import help_buttons
from .route_comparison import _RouteWorker


class GroupFlights(QWidget):
    """Small asynchronous catalog and ranking jobs, followed by shared 3-D rendering."""

    def __init__(self, parent=None):
        """Create controls lazily; no archive scan runs during application startup."""
        super().__init__(parent)
        self._worker = self._index = self._catalog = self._ranking = None
        self._next = None
        self._tried = False
        self._prepare = QPushButton("Prepare / refresh groups")
        self._cancel = QPushButton("Cancel")
        self._cancel.setEnabled(False)
        self._cell_size = QComboBox()
        self._cell_size.addItem("5 x 5 km", 5000)
        self._cell_size.addItem("10 x 10 km", 10000)
        self._minutes = QSpinBox()
        self._minutes.setRange(1, 1440)
        self._minutes.setValue(30)
        self._minutes.setSuffix(" min")
        self._minutes.setKeyboardTracking(False)
        self._minimum = QSpinBox()
        self._minimum.setRange(2, 1000)
        self._minimum.setValue(2)
        self._minimum.setKeyboardTracking(False)
        self._discipline = QComboBox()
        self._discipline.addItem("Both disciplines", None)
        self._year = QComboBox()
        self._year.addItem("All years", None)
        self._order = QComboBox()
        self._order.addItem("Most grouped flights", "grouped_flights")
        self._order.addItem("Largest single group", "largest")
        self._order.addItem("Most groups", "groups")
        self._cells, self._groups = QComboBox(), QComboBox()
        for combo in (self._cells, self._groups):
            combo.setMinimumContentsLength(20)
            combo.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            )
        self._load = QPushButton("Load selected group")
        self._load.setProperty("emphasis", "primary")
        self._export = QPushButton("Export group CSV")
        self._info, self._howto = help_buttons("groups", self)
        self._status = QLabel(
            "Select a cell size and time window. Saved clocks load when this tab opens."
        )
        self._status.setWordWrap(True)
        self._census = QLabel()
        self._census.setWordWrap(True)
        self._viewer = GroupScene(self)
        top = FlowLayout()
        for widget in (
            labeled_control("Departure cell", self._cell_size),
            labeled_control("Launch window", self._minutes),
            labeled_control("Minimum flights", self._minimum),
            self._discipline,
            self._year,
            self._prepare,
            self._cancel,
            self._info,
            self._howto,
        ):
            top.addWidget(widget)
        selection = FlowLayout()
        for widget in (labeled_control("Rank cells by", self._order), self._cells):
            selection.addWidget(widget)
        groups = FlowLayout()
        for widget in (
            labeled_control("Launch group", self._groups),
            self._load,
            self._export,
        ):
            groups.addWidget(widget)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(top)
        layout.addLayout(selection)
        layout.addWidget(self._census)
        layout.addLayout(groups)
        layout.addWidget(self._viewer, 1)
        layout.addWidget(self._status)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self._rank)
        self._prepare.clicked.connect(self._prepare_catalog)
        self._cancel.clicked.connect(self._cancel_work)
        self._load.clicked.connect(self._load_group)
        self._export.clicked.connect(self._export_group)
        self._cells.currentIndexChanged.connect(self._cell_changed)
        self._groups.currentIndexChanged.connect(self._group_changed)
        self._order.currentIndexChanged.connect(self._fill_cells)
        for combo in (self._cell_size, self._discipline, self._year):
            combo.currentIndexChanged.connect(self._request_rank)
        self._minutes.valueChanged.connect(self._request_rank)
        self._minimum.valueChanged.connect(self._request_rank)
        self._set_busy(False)

    def ensure_loaded(self):
        """Reuse current indexes on first opening; never silently scan raw clocks."""
        if self._tried:
            return
        self._tried = True
        self._run(lambda **kw: self._read_catalog(False, **kw), "catalog")

    @staticmethod
    def _read_catalog(prepare, **kwargs):
        """Only explicit preparation can build missing endpoint or clock indexes."""
        index = build_index(**kwargs) if prepare else load_saved_index()
        if index is None:
            raise ValueError(
                "No current endpoint index. Click Prepare / refresh groups."
            )
        if prepare:
            prepare_density(index.disciplines, **kwargs)
        return index, load_catalog(index, prepare=prepare, **kwargs)

    def _prepare_catalog(self):
        """Refresh clocks and the shared all-flight thermal context."""
        self._timer.stop()
        self._viewer._clear_scene()
        self._run(lambda **kw: self._read_catalog(True, **kw), "catalog")

    def _run(self, operation, kind):
        """Serialize work and reject obsolete results."""
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
        """Accept status only from the active worker."""
        if self.sender() is self._worker:
            self._status.setText(message)

    def _failed(self, message):
        """Display unavailable sources and cancellation without inventing a group."""
        if self.sender() is self._worker:
            self._next = None
            self._status.setText(message)

    def _received(self, result):
        """Publish catalog, ranking or scene only after a complete operation."""
        if self.sender() is not self._worker or self._worker.cancel.is_set():
            return
        if self._worker.kind == "catalog":
            self._index, self._catalog = result
            for combo in (self._discipline, self._year):
                combo.blockSignals(True)
                combo.clear()
            self._discipline.addItem("Both disciplines", None)
            for disc in self._index.disciplines:
                self._discipline.addItem(disc.name, disc.name)
            self._year.addItem("All years", None)
            for year in sorted(
                self._catalog.day.dropna().str[:4].unique(), reverse=True
            ):
                self._year.addItem(year, int(year))
            for combo in (self._discipline, self._year):
                combo.blockSignals(False)
            self._next = "rank"
        elif self._worker.kind == "ranking":
            self._ranking = result
            self._fill_cells()
            self._next = "scene" if self._groups.count() else None
        else:
            self._viewer._mode.setCurrentIndex(0)
            self._viewer.set_scene(result)
            self._status.setText(
                "Group loaded. Select rows to compare paths; "
                "drag the divider to enlarge the flight table."
            )

    def _finished(self):
        """Continue from catalog to ranking to the busiest group's scene."""
        worker = self.sender()
        if worker is self._worker:
            self._worker = None
            if worker.cancel.is_set():
                self._next = None
            self._set_busy(False)
            action, self._next = self._next, None
            if action == "rank":
                self._rank()
            elif action == "scene":
                self._load_group()
        worker.deleteLater()

    def _set_busy(self, busy):
        """Keep a coherent definition while its result is being computed."""
        for widget in (
            self._prepare,
            self._cell_size,
            self._minutes,
            self._minimum,
            self._discipline,
            self._year,
            self._order,
            self._cells,
            self._groups,
        ):
            widget.setEnabled(not busy)
        self._cancel.setEnabled(busy)
        self._load.setEnabled(not busy and self._groups.count() > 0)
        self._export.setEnabled(not busy and self._viewer._scene is not None)

    def _cancel_work(self):
        """Cancel between bounded blocks without publishing a partial selection."""
        if self._worker is not None:
            self._worker.cancel.set()
            self._status.setText("Cancelling…")

    def _request_rank(self, *_):
        """Clear stale scenes and debounce consecutive edits of the group definition."""
        self._viewer._clear_scene()
        self._ranking = None
        self._cells.clear()
        self._groups.clear()
        self._census.clear()
        self._set_busy(self._worker is not None)
        if self._catalog is not None:
            self._timer.start()

    def _rank(self):
        """Rank the full metadata population, with no trajectory reads or sampling."""
        if self._catalog is None:
            return
        if self._worker is not None:
            self._next = "rank"
            return
        options = {
            "cell_m": self._cell_size.currentData(),
            "window_minutes": self._minutes.value(),
            "minimum": self._minimum.value(),
            "discipline": self._discipline.currentData(),
            "year": self._year.currentData(),
        }
        catalog = self._catalog
        self._run(lambda **kw: rank_groups(catalog, **options, **kw), "ranking")

    def _fill_cells(self, *_):
        """Order all qualifying cells and retain the chosen cell where possible."""
        if self._ranking is None:
            return
        previous = self._cells.currentData()
        self._cells.blockSignals(True)
        self._cells.clear()
        rows = ordered_cells(self._ranking, self._order.currentData())
        for number, row in enumerate(rows.itertuples(index=False), 1):
            self._cells.addItem(
                f"{number}. Cell [{row.ix}, {row.iy}] · "
                f"{row.grouped_flights:,} grouped flights · "
                f"{row.groups:,} groups · largest {row.largest}",
                (int(row.ix), int(row.iy)),
            )
        choice = self._cells.findData(previous)
        if choice >= 0:
            self._cells.setCurrentIndex(choice)
        self._cells.blockSignals(False)
        self._cell_changed()

    def _cell_changed(self, *_):
        """Offer the cell's disjoint groups, largest first then earliest departure."""
        self._groups.blockSignals(True)
        self._groups.clear()
        self._viewer._clear_scene()
        cell = self._cells.currentData()
        if self._ranking is not None:
            r = self._ranking
            self._census.setText(
                f"{len(r.cells):,} cells · {len(r.groups):,} disjoint groups · "
                f"{int(r.groups.flights.sum()):,} grouped flights / "
                f"{len(r.flights):,} dated departures · "
                f"{r.unknown_clocks:,} flights excluded: unavailable clock "
                "(before year filtering)."
            )
            if cell is not None:
                groups = r.groups.loc[r.groups.ix.eq(cell[0]) & r.groups.iy.eq(cell[1])]
                groups = groups.sort_values(
                    ["flights", "departure_utc"], ascending=[False, True]
                )
                for i, row in groups.iterrows():
                    first = datetime.fromtimestamp(row.departure_utc, PARIS).strftime(
                        "%d/%m/%Y %H:%M:%S %Z"
                    )
                    last = datetime.fromtimestamp(
                        row.last_departure_utc, PARIS
                    ).strftime("%H:%M:%S %Z")
                    self._groups.addItem(
                        f"{first} to {last} · {row.flights} flights", int(i)
                    )
        self._groups.blockSignals(False)
        self._group_changed()

    def _group_changed(self, *_):
        """Never leave a previous cohort under a newly selected group label."""
        self._viewer._clear_scene()
        self._status.setText(
            "Load the selected group."
            if self._groups.count()
            else "No groups match this definition."
        )
        self._set_busy(self._worker is not None)

    def _load_group(self):
        """Read all selected members, keeping their departure ordering."""
        choice = self._groups.currentData()
        if choice is None or self._ranking is None or self._index is None:
            return
        index, ranking = self._index, self._ranking
        group = ranking.groups.loc[choice].copy()
        self._viewer._clear_scene()
        self._run(lambda **kw: load_group_scene(index, ranking, group, **kw), "scene")

    def _export_group(self):
        """Save the displayed cohort and measured metrics with explicit UTC fields."""
        scene = self._viewer._scene
        if scene is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export launch group", "launch-group.csv", "CSV (*.csv)"
        )
        if path:
            try:
                scene.selected.to_csv(path, index=False)
                self._status.setText(
                    f"Saved {len(scene.selected)} group members to {path}"
                )
            except OSError as exc:
                self._status.setText(str(exc))

    def shutdown(self):
        """Stop workers before Qt destroys their owner or the shared OpenGL view."""
        self._timer.stop()
        self._next = None
        if self._worker is not None:
            worker, self._worker = self._worker, None
            worker.cancel.set()
            worker.wait()
            worker.deleteLater()
        self._viewer.shutdown()

    def invalidate(self):
        """Discard both ranking and scene when archive roots change."""
        self.shutdown()
        self._index = self._catalog = self._ranking = None
        self._tried = False
        self._cells.clear()
        self._groups.clear()
        self._viewer._clear_scene()
