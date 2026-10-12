"""Compare up to 300 cleaned flights sharing two distant 10 km endpoint cells."""

from __future__ import annotations

from datetime import datetime
from itertools import pairwise, product
from threading import Event

import numpy as np
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QVector3D
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..route_density import prepare_density
from ..route_index import (
    DISTANCE_PRESETS_KM,
    MAX_FLIGHTS,
    PAIR_COLUMNS,
    ROUTE_CELL_M,
    build_index,
    load_saved_index,
    route_pairs,
)
from ..route_scene import load_scene, terrain_mesh
from ..thermal_daily import PARIS
from ..thermal_geometry import unproject
from ..thermal_index import CancelledError
from .departure_filter import DepartureFilter
from .flow_layout import FlowLayout
from .help import help_buttons
from .route_density import RouteDensity
from .route_locator import RouteLocator

COLOURS = {"fast": "#00e5ff", "slow": "#7cff32", "other": "#edf6ff", "both": "#65aaff"}
SELECTED_COLOUR = "#1464ff"
MODES = [
    (f"All · max {MAX_FLIGHTS}", "all"),
    ("5 fastest", "fast"),
    ("5 slowest", "slow"),
    ("Fastest + slowest", "extremes"),
]


def _flight_datetime(timestamp):
    """Show the local date, seconds and DST designation, or an explicit absence."""
    if timestamp is None or not np.isfinite(timestamp):
        return "Unavailable"
    return datetime.fromtimestamp(timestamp, PARIS).strftime("%d/%m/%Y %H:%M:%S %Z")


class _RouteWorker(QThread):
    """Run bounded I/O away from Qt, with cooperative cancellation."""

    progress = pyqtSignal(str)
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, operation, parent):
        """Keep the selected operation and a thread-safe cancellation event."""
        super().__init__(parent)
        self.operation = operation
        self.cancel = Event()

    def run(self):
        """Publish only complete results, with errors shown in the route tab."""
        try:
            result = self.operation(progress=self.progress.emit, cancel=self.cancel)
            if not self.cancel.is_set():
                self.succeeded.emit(result)
        except CancelledError:
            self.failed.emit("Loading cancelled.")
        except Exception as exc:
            self.failed.emit(str(exc))


class RouteComparison(QWidget):
    """Lazy census, explicit directed pair selection and an IGN terrain view."""

    def __init__(self, parent=None, *, scene_only=False):
        """Build the controls without opening the archive or creating an OpenGL view."""
        super().__init__(parent)
        self._worker = self._index = self._flights = self._scene = None
        self._view = self._surface = None
        self._lines = []
        self._annotations = None
        self._tried = False
        self._pending_scene = False
        self._base_source = ""
        self._build = QPushButton("Prepare / refresh index")
        self._build.setToolTip(
            "One resumable scan of cleaned archive endpoints; saved in the data folder"
        )
        self._discipline = QComboBox()
        self._discipline.addItem("Both disciplines", None)
        self._discipline.addItem("Pairs shared by both", "shared")
        self._distance = QComboBox()
        for distance in DISTANCE_PRESETS_KM:
            self._distance.addItem(f"Around {distance} km", distance)
        self._distance.setCurrentIndex(self._distance.findData(100))
        self._distance.setToolTip("Load the busiest cell pair within +/-10 km")
        self._minimum, self._maximum = QSpinBox(), QSpinBox()
        for spin, value, prefix in (
            (self._minimum, 90, "Min: "),
            (self._maximum, 110, "Max: "),
        ):
            spin.setRange(5, 1000)
            spin.setValue(value)
            spin.setPrefix(prefix)
            spin.setSuffix(" km")
        self._pairs = QComboBox()
        self._pairs.setMinimumContentsLength(22)
        self._pairs.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self._pairs.setToolTip(
            "A and B are cells in the same 10 x 10 km Lambert-93 grid. "
            "Column and row are positions, not cell counts. "
            "Distance is measured between the cell centres."
        )
        self._load = QPushButton("Load selected pair")
        self._load.setProperty("emphasis", "primary")
        self._cancel = QPushButton("Cancel")
        self._cancel.setEnabled(False)
        self._info, self._howto = help_buttons("routes", self)
        self._departure = DepartureFilter(self)
        controls = FlowLayout()
        for widget in (
            self._build,
            self._discipline,
            self._distance,
            self._minimum,
            self._maximum,
            self._load,
            self._cancel,
            self._info,
            self._howto,
        ):
            controls.addWidget(widget)
        self._mode = QComboBox()
        for title, key in MODES:
            self._mode.addItem(title, key)
        self._show_all = QPushButton("Show all flights")
        self._hide_all = QPushButton("Hide all flights")
        self._show_all.setToolTip("Check every flight and return to the All view")
        self._hide_all.setToolTip("Uncheck every flight; keep the list to choose a few")
        self._visible_count = QLabel()
        self._update_visible_count([])
        self._line_width = QDoubleSpinBox()
        self._line_width.setRange(0.1, 12)
        self._line_width.setSingleStep(0.1)
        self._line_width.setDecimals(1)
        self._line_width.setValue(2.5)
        self._line_width.setPrefix("Track width: ")
        self._line_width.setSuffix(" px")
        self._line_width.setToolTip(
            "Actual screen-pixel stroke width, with a dark outline. "
            "Fastest/slowest flights are 25% wider; selected rows 75% wider."
        )
        self._terrain = QCheckBox("IGN terrain")
        self._terrain.setChecked(True)
        self._surface_mode = QComboBox()
        self._surface_mode.addItems(
            [
                "Elevation colours · DEM",
                "Aerial imagery · IGN",
                "Grayscale relief · DEM",
            ]
        )
        self._surface_mode.setEnabled(False)
        self._vertical = QDoubleSpinBox()
        self._vertical.setRange(1, 10)
        self._vertical.setValue(1)
        self._vertical.setDecimals(1)
        self._vertical.setPrefix("Vertical exaggeration: ")
        self._vertical.setSuffix("x")
        self._vertical.setToolTip(
            "1x preserves real vertical proportions. 5x draws terrain and flight "
            "heights five times taller relative to horizontal distances. "
            "Display only: recorded coordinates, durations and density do not change."
        )
        self._reset = QPushButton("Reset view")
        self._top = QPushButton("Top view")
        style = FlowLayout()
        for widget in (
            self._mode,
            self._line_width,
            self._terrain,
            self._surface_mode,
            self._vertical,
            self._reset,
            self._top,
        ):
            style.addWidget(widget)
        self._summary = QLabel(
            "Same departure cell and same arrival cell for every displayed flight "
            "· 10 x 10 km cells"
        )
        self._summary.setWordWrap(True)
        self._legend = QLabel(
            f'<span style="color:{COLOURS["fast"]}">● Fastest</span> · '
            f'<span style="color:{COLOURS["slow"]}">● Slowest</span> · '
            "White: other flights · Blue: selected row · "
            "○ First clean fix · ■ Last clean fix"
        )
        self._legend.setWordWrap(True)
        self._placeholder = QLabel(
            "Prepare the endpoint index once, then choose a pair of cells."
        )
        self._placeholder.setWordWrap(True)
        self._density = RouteDensity(self)
        self._locator = RouteLocator(self)
        self._scene_panel = QWidget(self)
        self._scene_layout = QHBoxLayout(self._scene_panel)
        self._scene_layout.setContentsMargins(0, 0, 0, 0)
        self._scene_layout.addWidget(self._placeholder, 3)
        self._scene_layout.addWidget(self._locator, 1)
        self._table = QTableWidget(0, 7)
        self._table.setAlternatingRowColors(True)
        self._table.setShowGrid(False)
        self._table.setHorizontalHeaderLabels(
            [
                "Show / rank",
                "Discipline",
                "Flight ID",
                "Departure (Paris)",
                "Arrival (Paris)",
                "Elapsed time",
                "Group",
            ]
        )
        self._table.horizontalHeaderItem(0).setToolTip(
            "Tick to show a flight; untick to hide it. The number is its duration rank."
        )
        for column, endpoint in ((3, "First"), (4, "Last")):
            self._table.horizontalHeaderItem(column).setToolTip(
                f"{endpoint} retained fix of the displayed trajectory. "
                "Date and time in Europe/Paris (CET/CEST)."
            )
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.verticalHeader().hide()
        self._table.setMinimumHeight(80)
        self._flight_splitter = QSplitter(Qt.Orientation.Vertical)
        self._flight_splitter.setChildrenCollapsible(False)
        self._flight_splitter.setHandleWidth(8)
        self._flight_splitter.addWidget(self._scene_panel)
        self._flight_splitter.addWidget(self._table)
        self._flight_splitter.setStretchFactor(0, 1)
        self._flight_splitter.setStretchFactor(1, 0)
        self._flight_splitter.setSizes([600, 180])
        self._flight_splitter.handle(1).setToolTip(
            "Drag up to show more flights; drag down to enlarge the map"
        )
        self._table.hide()
        self._source = QLabel()
        self._source.setWordWrap(True)
        self._source.setOpenExternalLinks(True)
        self._status = QLabel()
        self._status.setWordWrap(True)
        self._layout = QVBoxLayout(self)
        self._queries = QWidget(self)
        query_layout = QVBoxLayout(self._queries)
        query_layout.setContentsMargins(0, 0, 0, 0)
        query_layout.addLayout(controls)
        query_layout.addWidget(self._pairs)
        query_layout.addWidget(self._departure)
        self._layout.addWidget(self._queries)
        self._queries.setVisible(not scene_only)
        self._layout.addWidget(self._summary)
        self._layout.addLayout(style)
        visibility = FlowLayout()
        for widget in (
            self._show_all,
            self._hide_all,
            self._visible_count,
        ):
            visibility.addWidget(widget)
        self._layout.addLayout(visibility)
        self._layout.addWidget(self._legend)
        self._layout.addWidget(self._density)
        self._layout.addWidget(self._flight_splitter, 1)
        self._layout.addWidget(self._source)
        self._layout.addWidget(self._status)
        self._build.clicked.connect(self._prepare)
        self._load.clicked.connect(self._load_pair)
        self._cancel.clicked.connect(self._cancel_work)
        self._discipline.currentIndexChanged.connect(self._refresh_pairs)
        self._distance.currentIndexChanged.connect(self._distance_changed)
        self._minimum.valueChanged.connect(self._refresh_pairs)
        self._maximum.valueChanged.connect(self._refresh_pairs)
        self._pairs.currentIndexChanged.connect(self._pair_changed)
        self._mode.currentIndexChanged.connect(self._style_changed)
        self._line_width.valueChanged.connect(self._style_changed)
        self._terrain.toggled.connect(self._style_changed)
        self._surface_mode.currentIndexChanged.connect(self._surface_changed)
        self._vertical.valueChanged.connect(self._scale_changed)
        self._reset.clicked.connect(self._reset_view)
        self._top.clicked.connect(self._top_view)
        self._table.itemSelectionChanged.connect(self._style_changed)
        self._table.itemChanged.connect(self._flight_visibility_changed)
        self._show_all.clicked.connect(lambda: self._set_all_flights_visible(True))
        self._hide_all.clicked.connect(lambda: self._set_all_flights_visible(False))
        self._departure.changed.connect(self._departure_changed)
        self._departure.apply_requested.connect(self._apply_departure)
        self._set_busy(False)

    def ensure_loaded(self):
        """Look for a completed small index when this tab is first opened."""
        if self._tried:
            return
        self._tried = True
        self._start(lambda **_: load_saved_index(), "index")

    def _prepare(self):
        """Prepare endpoint and all-flight thermal caches outside the GUI thread."""
        self._clear_scene()

        def prepare(**kwargs):
            index = build_index(**kwargs)
            prepare_density(index.disciplines, **kwargs)
            return index

        self._start(prepare, "index")

    def _start(self, operation, kind):
        """Serialize jobs and tag their result type without sharing SQL handles."""
        if self._worker is not None:
            return
        self._status.setText("Loading…")
        worker = _RouteWorker(operation, self)
        worker.kind = kind
        self._worker = worker
        worker.progress.connect(self._progress)
        worker.succeeded.connect(self._loaded)
        worker.failed.connect(self._failed)
        worker.finished.connect(self._finished)
        self._set_busy(True)
        worker.start()

    def _progress(self, message):
        """Ignore late progress from an invalidated archive request."""
        if self.sender() is self._worker:
            self._status.setText(message)

    def _failed(self, message):
        """Expose the error while retaining the explicitly labelled previous scene."""
        if self.sender() is self._worker:
            if self._scene is not None:
                message += " Previous departure selection is still displayed."
            self._status.setText(message)

    def _loaded(self, result):
        """Install a completed index or a scene on the GUI thread."""
        worker = self.sender()
        if worker is not self._worker or worker.cancel.is_set():
            return
        if worker.kind == "index":
            self._index = result
            if result is None:
                self._status.setText(
                    "No current route index. Use Prepare / refresh index; "
                    "only the first scan reads the archive."
                )
                return
            self._flights = result.flights()
            self._discipline.blockSignals(True)
            self._discipline.clear()
            self._discipline.addItem("Both disciplines", None)
            self._discipline.addItem("Pairs shared by both", "shared")
            for disc in result.disciplines:
                self._discipline.addItem(disc.name, disc.name)
            self._discipline.blockSignals(False)
            self._refresh_pairs()
            self._pending_scene = self._pairs.count() > 0
        else:
            self.set_scene(result)

    def _finished(self):
        """Release the job before automatically loading the busiest pair."""
        worker = self.sender()
        if worker is self._worker:
            self._worker = None
            self._set_busy(False)
            if worker.cancel.is_set():
                self._pending_scene = False
                self._status.setText(
                    "Loading cancelled. "
                    + (
                        "Previous departure selection is still displayed."
                        if self._scene is not None
                        else "No partial scene is displayed."
                    )
                )
            if self._pending_scene:
                self._pending_scene = False
                self._load_pair()
        worker.deleteLater()

    def _set_busy(self, busy):
        """Keep source filters fixed until the current operation has completed."""
        for widget in (
            self._build,
            self._discipline,
            self._distance,
            self._minimum,
            self._maximum,
            self._pairs,
        ):
            widget.setEnabled(not busy)
        self._load.setEnabled(
            not busy
            and self._pairs.count() > 0
            and (self._departure.selection() is None or self._departure.has_matches())
        )
        self._cancel.setEnabled(busy)
        self._departure.set_busy(busy)

    def _cancel_work(self):
        """Request cancellation at the next bounded I/O boundary."""
        if self._worker is not None:
            self._worker.cancel.set()
            self._status.setText("Cancelling…")

    def shutdown(self):
        """Join a cancelled worker before Qt destroys it."""
        if self._worker is not None:
            worker, self._worker = self._worker, None
            worker.cancel.set()
            worker.wait()
            worker.deleteLater()
        self._pending_scene = False
        self._density.clear()
        self._locator.shutdown()
        self._release_texture()

    def invalidate(self):
        """Discard cached data after the archive directories change."""
        self.shutdown()
        self._index = self._flights = None
        self._tried = False
        self._departure.reset()
        self._pairs.clear()
        self._clear_scene()
        self._set_busy(False)

    def set_compact(self, compact):
        """Give the scene extra room while retaining scientific controls."""
        self._source.setVisible(not compact)

    def _refresh_pairs(self, *_):
        """Offer actual directed pairs, including sparse populations honestly."""
        self._departure.reset()
        self._clear_scene()
        self._pairs.blockSignals(True)
        self._pairs.clear()
        try:
            if self._flights is None:
                return
            pairs = route_pairs(
                self._flights,
                self._minimum.value(),
                self._maximum.value(),
                self._selected_discipline(),
                require_both=self._discipline.currentData() == "shared",
            )
            for row in pairs.itertuples(index=False):
                pair = tuple(int(getattr(row, c)) for c in PAIR_COLUMNS)
                self._pairs.addItem(
                    f"A [col {pair[0]}, row {pair[1]}] → "
                    f"B [col {pair[2]}, row {pair[3]}] · "
                    f"{row.distance_km:.1f} km · {row.flights:,} flights",
                    pair,
                )
            endpoint_cells = np.unique(
                np.concatenate(
                    [
                        self._flights[PAIR_COLUMNS[:2]].to_numpy(),
                        self._flights[PAIR_COLUMNS[2:]].to_numpy(),
                    ]
                ),
                axis=0,
            )
            self._pairs.setToolTip(
                "10 x 10 km cells; column and row are grid positions. "
                f"{len(endpoint_cells):,} occupied departure/arrival cells in "
                f"{len(self._flights):,} indexed flights. Empty cells are omitted. "
                "Distance is between cell centres."
            )
            self._status.setText(
                f"{len(pairs):,} directed pairs · ranked by number of cleaned flights. "
                "Choose a pair and load it."
                if len(pairs)
                else "No flights share a pair in this distance interval. "
                "Adjust the distance or discipline."
            )
        except ValueError as exc:
            self._status.setText(str(exc))
        finally:
            self._pairs.blockSignals(False)
            self._load.setEnabled(self._worker is None and self._pairs.count() > 0)

    def _distance_changed(self, *_):
        """Load a distance preset, keeping the exact same 10 km endpoint rule."""
        distance = self._distance.currentData()
        for spin, value in (
            (self._minimum, distance - 10),
            (self._maximum, distance + 10),
        ):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
        self._refresh_pairs()
        self._load_pair()

    def _pair_changed(self, *_):
        """Clear the old scene immediately when its endpoint pair changes."""
        self._departure.reset()
        self._clear_scene()
        self._status.setText(
            "Load the selected pair to display its cleaned trajectories."
        )

    def _departure_changed(self):
        """Keep the applied scene visible while the controls preview a new request."""
        if (
            self._scene is not None
            and self._scene.departure_window == self._departure.selection()
        ):
            message = "Departure selection matches the displayed flights."
        elif self._scene is not None:
            message = (
                "Departure changes not applied. Previous selection is still "
                "displayed. Click Apply to load the preview."
            )
        else:
            message = "Departure selection changed. Click Apply to load it."
        self._status.setText(message)
        self._set_busy(self._worker is not None)

    def _apply_departure(self):
        """Show the complete new departure cohort before optional speed filtering."""
        self._mode.setCurrentIndex(0)
        self._load_pair()

    def _selected_discipline(self):
        """A shared-pair requirement retains both disciplines in the same cohort."""
        value = self._discipline.currentData()
        return None if value == "shared" else value

    def _clear_scene(self):
        """Avoid retaining an old cohort under a newly selected route label."""
        self._density.clear()
        self._locator.set_scene(None)
        self._release_texture()
        self._scene = self._surface = None
        self._lines = []
        self._annotations = None
        self._scene_items = []
        self._base_source = ""
        self._surface_mode.setEnabled(False)
        if self._view is not None:
            self._view.clear()
        self._table.setRowCount(0)
        self._table.hide()
        self._update_visible_count([])
        self._source.clear()
        self._summary.setText("Same departure cell and same arrival cell · 10 x 10 km")

    def _load_pair(self):
        """Capture the selected source and both cells for one cancellable job."""
        pair = self._pairs.currentData()
        if self._index is None or pair is None or self._worker is not None:
            return
        if (
            self._departure.selection() is not None
            and not self._departure.has_matches()
        ):
            return
        index, flights, discipline = (
            self._index,
            self._flights,
            self._selected_discipline(),
        )
        departure_window = self._departure.selection()
        self._start(
            lambda **kwargs: load_scene(
                index,
                flights,
                pair,
                discipline,
                departure_window=departure_window,
                **kwargs,
            ),
            "scene",
        )

    def set_scene(self, scene):
        """Upload cleaned segments and a terrain mesh; never bridge data gaps."""
        from matplotlib import colormaps
        from matplotlib.colors import LightSource
        from scipy.interpolate import RegularGridInterpolator

        from .route_annotations import RouteAnnotations
        from .route_lines import RouteLine
        from .terrain_drape import DrapedTerrain
        from .thermal_3d import TerrainView

        self._clear_scene()
        self._scene = scene
        self._departure.set_cohort(scene.departure_cohort)
        if self._view is None:
            self._view = TerrainView(self)
            self._scene_layout.replaceWidget(self._placeholder, self._view)
            self._placeholder.hide()
        self._view.scene_span = max(
            scene.bounds[2] - scene.bounds[0], scene.bounds[3] - scene.bounds[1]
        )
        self._scene_items = []
        if scene.terrain is not None:
            vertices, faces = terrain_mesh(scene.terrain, scene.origin)
            z = scene.terrain[2]
            ref = scene.terrain[3]
            heights = np.nan_to_num(z, nan=float(np.nanmedian(z)))
            colours = colormaps["terrain"](0.25 + 0.65 * np.clip(heights / 4500, 0, 1))
            light = LightSource(azdeg=315, altdeg=45).hillshade(
                heights, dx=ref["grid_m"][0], dy=ref["grid_m"][1]
            )
            colours[:, :, :3] = (0.65 * colours[:, :, :3] + 0.35) * (
                0.55 + 0.45 * light[:, :, None]
            )
            colours = colours.reshape(-1, 4).astype(np.float32)
            gray = np.ones_like(colours)
            gray[:, :3] = (0.4 + 0.55 * light.ravel())[:, None]
            self._terrain_colours = (colours, gray)
            self._surface = DrapedTerrain(
                image=scene.aerial,
                bounds=np.asarray(scene.bounds) - np.tile(scene.origin, 2),
                vertexes=vertices,
                faces=faces,
                vertexColors=colours,
                smooth=True,
                shader=None,
                glOptions="opaque",
                computeNormals=False,
            )
            self._view.addItem(self._surface)
            self._scene_items.append(self._surface)
        self._surface_mode.setEnabled(self._surface is not None)
        self._surface_mode.model().item(1).setEnabled(scene.aerial is not None)
        self._density.attach(scene, self._view, self._surface)
        self._locator.set_scene(scene)
        if scene.aerial is None and self._surface_mode.currentIndex() == 1:
            self._surface_mode.setCurrentIndex(0)
        for track in scene.tracks:
            lines = []
            for segment in track:
                if len(segment) < 2:
                    continue
                line = RouteLine(
                    pos=segment,
                    antialias=True,
                    mode="line_strip",
                    glOptions="translucent",
                )
                line.updateGLOptions({"glDepthMask": (False,)})
                line.setDepthValue(5)
                self._view.addItem(line)
                lines.append(line)
                self._scene_items.append(line)
            self._lines.append(lines)
        anchors = []
        for ix, iy in np.asarray(scene.pair).reshape(-1, 2):
            corners = np.array(
                [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]
            ) * scene.cell_m + [
                ix * scene.cell_m,
                iy * scene.cell_m,
            ]
            xy = np.concatenate(
                [np.linspace(a, b, 25, endpoint=False) for a, b in pairwise(corners)]
                + [corners[-1:]]
            )
            height = np.zeros(len(xy))
            if scene.terrain is not None:
                x, y, z, _ = scene.terrain
                height = RegularGridInterpolator(
                    (y, x), z, bounds_error=False, fill_value=np.nan
                )(xy[:, ::-1])
            outline = np.column_stack((xy - scene.origin, height + 35)).astype(
                np.float32
            )
            edges = np.stack((outline[:-1], outline[1:]), axis=1)
            edges = edges[np.isfinite(edges).all(axis=(1, 2))].reshape(-1, 3)
            item = RouteLine(
                pos=edges,
                color=(1, 1, 1, 1),
                width=2.5,
                outline=1.5,
                mode="lines",
                glOptions="translucent",
            )
            # Cell boundaries are annotations, drawn over terrain, not flight geometry.
            from OpenGL import GL

            item.updateGLOptions({GL.GL_DEPTH_TEST: False})
            item.setDepthValue(20)
            self._view.addItem(item)
            centre = np.array([ix + 0.5, iy + 0.5]) * scene.cell_m
            ground = float(np.nanmedian(height)) if np.isfinite(height).any() else 0
            anchors.append([*(centre - scene.origin), ground])
            self._scene_items.append(item)
        self._annotations = RouteAnnotations(scene.tracks, anchors)
        self._view.addItem(self._annotations)
        self._scene_items.append(self._annotations)
        self._populate_table(scene)
        self._set_scene_summary(scene)
        source = self._flight_source(scene)
        if scene.terrain is not None:
            ref = scene.terrain[3]
            source += (
                f'<a href="{ref["dataset_url"]}">© IGN RGE ALTI · '
                "Licence Ouverte 2.0</a> · DEM overview sampling "
                f"{ref['grid_m'][0]:.0f} x {ref['grid_m'][1]:.0f} m · "
                f"coverage {ref['coverage']:.1%}. "
            )
        source += (
            "Recorder GNSS and IGN height datums are not harmonised; "
            "flight altitudes are unchanged."
        )
        self._base_source = source
        self._surface_changed()
        self._status.setText(
            f"Terrain unavailable: {scene.terrain_error}. Load again to retry."
            if scene.terrain_error
            else "Drag: orbit · Shift + drag: pan · Scroll/pinch: zoom · "
            "Click a table row to highlight a flight"
        )
        if scene.aerial_error:
            self._status.setText(
                f"Aerial imagery unavailable: {scene.aerial_error}. "
                "The DEM remains visible; load again to retry."
            )
        self._style_changed()
        self._scale_changed()
        self._layout.activate()
        self._reset_view()
        # Fit once the table and parent controls have their final dimensions.
        QTimer.singleShot(0, self._reset_view)

    def _populate_table(self, scene):
        """List the cohort with its selection and timing fields."""
        self._table.blockSignals(True)
        self._table.setRowCount(len(scene.selected))
        for i, row in enumerate(scene.selected.itertuples(index=False)):
            seconds = round(row.duration_s)
            values = [
                str(row.rank),
                row.discipline,
                row.flight_id,
                _flight_datetime(getattr(row, "departure_utc", None)),
                _flight_datetime(getattr(row, "arrival_utc", None)),
                f"{seconds // 3600}:{seconds % 3600 // 60:02}:{seconds % 60:02}",
                row.speed_group,
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col == 0:
                    self._make_flight_checkable(item)
                if self._table.palette().base().color().lightness() < 128:
                    item.setForeground(QColor(COLOURS[row.speed_group]))
                self._table.setItem(i, col, item)
        self._table.blockSignals(False)
        self._table.show()
        self._mode.setItemText(1, f"{min(5, len(scene.selected))} fastest")
        self._mode.setItemText(2, f"{min(5, len(scene.selected))} slowest")

    def _set_scene_summary(self, scene):
        """Describe the route's two shared endpoints and applied departure filter."""
        cells = np.asarray(scene.pair).reshape(2, 2)
        lon, lat = unproject(*(cells * ROUTE_CELL_M + ROUTE_CELL_M / 2).T)
        distance = np.linalg.norm(cells[1] - cells[0]) * ROUTE_CELL_M / 1000
        self._summary.setText(
            f"{len(scene.selected)} loaded / {scene.total:,} matching flights · "
            f"10 x 10 km cells · A → B: {distance:.1f} km between centres\n"
            f"A: {lat[0]:.3f}° N, {lon[0]:.3f}° E · "
            f"B: {lat[1]:.3f}° N, {lon[1]:.3f}° E · "
            + ", ".join(
                f"{name}: {count}"
                for name, count in scene.selected.discipline.value_counts().items()
            )
            + "\nDisplayed departures: "
            + (scene.departure_window.label if scene.departure_window else "All dates")
        )

    def _flight_source(self, scene):
        """Explain the displayed route population without changing terrain credits."""
        source = (
            "Cleaned archive endpoints; elapsed time = last - first retained fix, "
            "including gaps. Gaps are not connected. "
        )
        if len(scene.selected) < 10:
            source += "Fastest/slowest groups overlap for fewer than 10 flights. "
        if scene.total > MAX_FLIGHTS:
            source += (
                "Sample: both groups of 5 extremes plus "
                f"{MAX_FLIGHTS - 10} evenly spaced duration ranks. "
            )
        return source

    def _surface_changed(self, *_):
        """Drape aerial imagery on the same DEM without moving geometry or camera."""
        self._source.setText(self._base_source)
        if self._scene is None or self._surface is None:
            return
        aerial = (
            self._surface_mode.currentIndex() == 1 and self._scene.aerial is not None
        )
        self._surface.set_aerial(aerial)
        if not aerial:
            gray = self._surface_mode.currentIndex() == 2
            self._surface.opts["meshdata"].setVertexColors(
                self._terrain_colours[int(gray)]
            )
            self._surface.meshDataChanged()
        source = self._base_source
        if aerial:
            ref = self._scene.aerial_reference
            source += (
                f' <a href="{ref["dataset_url"]}">© IGN BD ORTHO · '
                "Licence Ouverte 2.0</a> · Aerial overview "
                f"{ref['grid_m'][0]:.0f} x {ref['grid_m'][1]:.0f} m/pixel. "
                "Mosaic acquisition dates vary and differ from flight dates."
            )
        self._source.setText(source)

    def _release_texture(self):
        """Free the previous orthophoto while its OpenGL context still exists."""
        if (
            self._surface is not None
            and self._view is not None
            and self._view.isValid()
        ):
            self._view.makeCurrent()
            try:
                self._surface.release_texture()
            finally:
                self._view.doneCurrent()

    @staticmethod
    def _make_flight_checkable(item):
        """Start each freshly loaded flight visible, independently of row selection."""
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Checked)
        item.setToolTip(
            "Tick to show this flight; untick to hide its path and endpoints"
        )

    def _flight_checked(self, row):
        """Read visibility from the persistent checkbox, not the highlighted rows."""
        return self._table.item(row, 0).checkState() == Qt.CheckState.Checked

    def _flight_visibility_changed(self, item):
        """Refresh paths and endpoint markers after a mouse or keyboard toggle."""
        if item.column() == 0:
            self._style_changed()

    def _set_all_flights_visible(self, visible):
        """Batch checkbox updates without rebuilding geometry or moving the camera."""
        if self._scene is None:
            return
        state = Qt.CheckState.Checked if visible else Qt.CheckState.Unchecked
        self._table.blockSignals(True)
        for row in range(self._table.rowCount()):
            self._table.item(row, 0).setCheckState(state)
        self._table.blockSignals(False)
        if visible:
            self._mode.blockSignals(True)
            self._mode.setCurrentIndex(self._mode.findData("all"))
            self._mode.blockSignals(False)
        self._style_changed()

    def _update_visible_count(self, mask):
        """Show the actual number of paths enabled by both checkboxes and mode."""
        self._visible_count.setText(f"{sum(mask)} / {len(mask)} flights visible")
        self._show_all.setEnabled(bool(len(mask)))
        self._hide_all.setEnabled(bool(len(mask)))

    def _style_changed(self, *_):
        """Select the requested speed groups while preserving the camera."""
        if self._scene is None:
            return
        chosen = {item.row() for item in self._table.selectedItems()}
        mode = self._mode.currentData()
        mask = []
        for i, (row, lines) in enumerate(
            zip(self._scene.selected.itertuples(), self._lines, strict=True)
        ):
            matches_mode = (
                mode == "all"
                or row.speed_group == mode
                or row.speed_group == "both"
                or (mode == "extremes" and row.speed_group != "other")
            )
            visible = matches_mode and self._flight_checked(i)
            selected = i in chosen
            mask.append(visible)
            group = mode if mode in ("fast", "slow") else row.speed_group
            colour = QColor(SELECTED_COLOUR if selected else COLOURS[group])
            rgba = (
                *colour.getRgbF()[:3],
                0.85 if row.speed_group == "other" and not selected else 1.0,
            )
            for line in lines:
                line.setVisible(visible)
                line.outline = min(1.0, self._line_width.value() * 0.4)
                line.setData(
                    color=rgba,
                    width=self._line_width.value()
                    * (1.75 if selected else 1.25 if row.speed_group != "other" else 1),
                )
                line.setDepthValue(
                    15 if selected else 10 if row.speed_group != "other" else 5
                )
            # Unchecked flights remain available to tick again in the current mode.
            self._table.setRowHidden(i, not matches_mode)
        self._update_visible_count(mask)
        if self._surface is not None:
            self._surface.set_terrain_visible(self._terrain.isChecked())
        if self._annotations is not None:
            self._annotations.set_flights(mask, chosen)

    def _scale_changed(self, *_):
        """Apply explicit vertical exaggeration equally to terrain and tracks."""
        if self._scene is None:
            return
        for item in self._scene_items:
            item.resetTransform()
            item.scale(1, 1, self._vertical.value())

    def _reset_view(self):
        """Fit the entire route with equal horizontal metric scales."""
        if self._scene is not None:
            heights = [
                (segment[:, 2].min(), segment[:, 2].max())
                for track in self._scene.tracks
                for segment in track
            ]
            if self._scene.terrain is not None:
                z = self._scene.terrain[2]
                heights.append((np.nanmin(z), np.nanmax(z)))
            low, high = np.min(heights), np.max(heights)
            vertical = self._vertical.value()
            self._view.opts["center"] = QVector3D(
                0, 0, float((low + high) / 2) * vertical
            )
            cells = np.asarray(self._scene.pair).reshape(-1, 2)
            if len(cells) == 2:
                dx, dy = cells[1] - cells[0]
                azimuth = np.degrees(np.arctan2(dy, dx)) - 90
            else:
                azimuth = -55
            az, el = np.radians([azimuth, 45])
            right = np.array([-np.sin(az), np.cos(az), 0])
            up = np.array(
                [-np.cos(az) * np.sin(el), -np.sin(az) * np.sin(el), np.cos(el)]
            )
            toward = np.array(
                [np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)]
            )
            west, south, east, north = self._scene.bounds
            corners = np.array(
                list(
                    product(
                        [-(east - west) / 2, (east - west) / 2],
                        [-(north - south) / 2, (north - south) / 2],
                        [-(high - low) * vertical / 2, (high - low) * vertical / 2],
                    )
                )
            )
            horizontal = np.tan(np.radians(self._view.opts["fov"] / 2))
            vertical_fov = horizontal * self._view.height() / max(self._view.width(), 1)
            distance = (
                np.max(
                    corners @ toward
                    + np.maximum(
                        abs(corners @ right) / horizontal,
                        abs(corners @ up) / vertical_fov,
                    )
                )
                * 1.12
            )
            self._view.setCameraPosition(
                distance=float(distance), elevation=45, azimuth=float(azimuth)
            )

    def _top_view(self):
        """Look vertically down without resetting the current pan or zoom."""
        if self._view is not None:
            self._view.setCameraPosition(elevation=90, azimuth=-90)
