"""Four metric cells and a movable horizontal cross-section through their climbs."""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from html import escape
from threading import Event

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from PyQt6.QtCore import QDate, QDateTime, Qt, QThread, QTime, QTimeZone, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDateEdit,
    QDateTimeEdit,
    QDoubleSpinBox,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QTimeEdit,
    QVBoxLayout,
    QWidget,
)

from .. import geography
from ..thermal_daily import (
    PARIS,
    height_levels,
    in_windows,
    local_bounds,
    overlaps_windows,
    season_days,
    wall_windows,
)
from ..thermal_geometry import plane_intersections, unproject
from ..thermal_store import (
    CancelledError,
    PlaneData,
    ThermalStore,
    load_store,
    neighbour_frames,
)
from .flow_layout import ControlPanel, FlowLayout, labeled_control
from .help import help_buttons

NEIGHBOURS_MISSING = (
    "This data folder has no prepared neighbouring cells, so zooming out shows "
    "only the selected cell."
)
# The only climb labels shown: the Vilpellet segmentation.
SOURCE = "vilpellet"


def _read_backdrop(index, cell, kind):
    """Saved background of one square, or of France for ``cell=None``.

    Missing contours fall back to the saved colour map, and say so.
    """
    if kind == "none" or index is None:
        return None
    saved = (
        index.background(cell, kind)
        if hasattr(index, "background")
        else index.relief(cell)
        if hasattr(index, "relief")
        else None
    )
    if saved is None and kind == "topography" and hasattr(index, "background"):
        colour = index.background(cell, "colour")
        if colour is not None:
            pixels, metadata = colour
            saved = (
                pixels,
                {
                    **metadata,
                    "attribution": metadata.get("attribution", "Plan IGN")
                    + " · Elevation contours unavailable",
                },
            )
    return saved


def _neighbour_backdrops(index, cell, kind, cancel):
    """Decode the eight neighbours' saved backgrounds off the GUI thread.

    Image decoding releases the GIL, so a few threads divide the wait.
    """
    if cancel.is_set():
        raise CancelledError("Cancelled")
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = {
            (kind, (f.ix, f.iy)): pool.submit(_read_backdrop, index, f, kind)
            for f in neighbour_frames(cell)
        }
        images = {key: future.result() for key, future in pending.items()}
    if cancel.is_set():
        raise CancelledError("Cancelled")
    return images


def _visible_part(pixels, extent, limits, columns):
    """Crop a backdrop to the view, plus half a view each side, at screen resolution.

    Matplotlib resamples every input pixel on each draw: nine full 4000 px squares
    took about 8 s per redraw after Zoom -. Returns the pixels and their extent
    (left, right, bottom, top), or None when the image lies outside the view.
    """
    left, right, bottom, top = extent
    (x0, x1), (y0, y1) = limits
    rows, cols = pixels.shape[:2]
    dx, dy = (right - left) / cols, (top - bottom) / rows
    mx, my = (x1 - x0) / 2, (y1 - y0) / 2
    c0 = max(0, int(np.floor((x0 - mx - left) / dx)))
    c1 = min(cols, int(np.ceil((x1 + mx - left) / dx)))
    r0 = max(0, int(np.floor((top - y1 - my) / dy)))
    r1 = min(rows, int(np.ceil((top - y0 + my) / dy)))
    if c0 >= c1 or r0 >= r1:
        return None
    step = max(1, int((x1 - x0) / dx / max(columns, 1)))
    # The last kept pixel stands for up to ``step`` source pixels (< 1 on screen).
    n, m = -(-(c1 - c0) // step), -(-(r1 - r0) // step)
    return pixels[r0:r1:step, c0:c1:step], (
        left + c0 * dx,
        left + (c0 + n * step) * dx,
        top - (r0 + m * step) * dy,
        top - r0 * dy,
    )


class _Worker(QThread):
    """Run one cancellable I/O/decoding operation outside the GUI thread."""

    progress = pyqtSignal(str)
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, operation, parent):
        """Keep the operation and a cooperative cancellation flag."""
        super().__init__(parent)
        self.operation = operation
        self.cancel = Event()

    def run(self):
        """Publish only complete results; leave cancelled/failed caches untouched."""
        try:
            result = self.operation(progress=self.progress.emit, cancel=self.cancel)
            if not self.cancel.is_set():
                self.succeeded.emit(result)
        except CancelledError:
            self.failed.emit("Cancelled. No partial result is displayed.")
        except Exception as exc:
            self.failed.emit(f"Could not load thermal cells: {exc}")


class ThermalPlane(QWidget):
    """An all-time crossing census followed by a selected UTC climb-plane view."""

    def __init__(self, parent=None):
        """Build controls without reading any archive at application startup."""
        super().__init__(parent)
        self._index: ThermalStore | None = None
        self._plane: PlaneData | None = None
        self._worker: _Worker | None = None
        self._reliefs = OrderedDict()
        self._map_labels = []
        self._tried_cache = False
        self._auto_load = False
        self._dates_initialized = False
        self._daily_info = ""
        self._daily_days = {}
        self._daily_ranges = {}
        self._terrain_info = None
        self._neighbours = None
        self._neighbour_levels = OrderedDict()
        self._plane_limits = ((0, 5), (0, 5))
        self._pending_limits = None
        self._focused_axes = None
        self._window_cache = (None, np.empty((0, 2)))
        self._build = QPushButton("Reload data")
        self._build.setToolTip(
            "Read the completed thermal-planes.sqlite3 file from the data folder."
        )
        self._cells = QComboBox()
        self._cells.setMinimumContentsLength(20)
        self._cells.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self._cells.setToolTip(
            "The twelve prepared cells: in each terrain category, the three with "
            "the most Vilpellet climb runs over all dates."
        )
        self._start, self._end = QDateTimeEdit(), QDateTimeEdit()
        for edit in (self._start, self._end):
            edit.setTimeZone(QTimeZone(b"Europe/Paris"))
            edit.setDisplayFormat("yyyy-MM-dd HH:mm:ss 'Paris'")
            edit.setCalendarPopup(True)
            edit.setKeyboardTracking(False)
        self._load = QPushButton("Load climb intersections")
        self._load.setProperty("emphasis", "primary")
        self._cancel = QPushButton("Cancel")
        self._cancel.setEnabled(False)
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 0)
        self._levels = np.array([0.0])
        self._step = QComboBox()
        for step in (10, 20, 50, 100, 200):
            self._step.addItem(f"{step} m", step)
        self._step.setToolTip(
            "Vertical distance between selectable planes. Each plane has zero "
            "thickness: dots are exact interpolated climb intersections."
        )
        self._height = QDoubleSpinBox()
        self._height.setDecimals(2)
        self._height.setSuffix(" m above lowest terrain")
        self._height.setToolTip(
            "Plane altitude = lowest IGN terrain elevation inside this 5 km cell + z"
        )
        self._height.setRange(0, 0)
        self._height.setSingleStep(10)
        self._background = QComboBox()
        self._background.addItem("Colour map · IGN", "colour")
        self._background.addItem("Aerial photo · IGN", "aerial")
        self._background.addItem("Shaded relief", "relief")
        self._background.addItem("None", "none")
        self._background.addItem("Topography + contours · IGN", "topography")
        self._background.setCurrentIndex(4)
        self._image_info = QLabel()
        self._image_info.setWordWrap(True)
        self._mode = QComboBox()
        self._mode.addItems(["Whole interval", "Daily hour window"])
        self._mode.setToolTip(
            "Whole interval: one continuous Paris time span. Daily hour window: "
            "the same Paris clock hours on every selected day."
        )
        self._day = QDateEdit(QDate.currentDate())
        self._daily_start = QDateEdit(QDate.currentDate())
        self._daily_end = QDateEdit(QDate.currentDate())
        for edit in (self._day, self._daily_start, self._daily_end):
            edit.setDisplayFormat("yyyy-MM-dd")
            edit.setCalendarPopup(True)
            edit.setKeyboardTracking(False)
        self._daily_date_mode = QComboBox()
        self._daily_date_mode.addItem("Start / end dates", "range")
        self._daily_date_mode.addItem("Days around a date", "around")
        self._daily_date_mode.addItem("Same dates every year", "yearly")
        self._daily_start.setToolTip("First included civil day in Europe/Paris")
        self._daily_end.setToolTip("Last included civil day in Europe/Paris")
        # Day and month only; the leap year 2000 keeps 29 February selectable.
        self._season_start = QDateEdit(QDate(2000, 6, 1))
        self._season_end = QDateEdit(QDate(2000, 8, 31))
        for edit, tooltip in (
            (self._season_start, "First included day of each year"),
            (
                self._season_end,
                "Last included day of each year; an earlier date than the first "
                "day continues into the next year",
            ),
        ):
            edit.setDateRange(QDate(2000, 1, 1), QDate(2000, 12, 31))
            edit.setDisplayFormat("dd MMM")
            edit.setKeyboardTracking(False)
            edit.setToolTip(tooltip)
        self._first_year, self._last_year = QSpinBox(), QSpinBox()
        for spin, tooltip in (
            (self._first_year, "First included year"),
            (self._last_year, "Last included year"),
        ):
            spin.setRange(1990, 2100)
            spin.setValue(QDate.currentDate().year())
            spin.setKeyboardTracking(False)
            spin.setToolTip(tooltip)
        self._best_day = QPushButton("Busiest summer day")
        self._best_day.setToolTip(
            "Select the busiest summer day; in Start / end dates, select that day only"
        )
        self._before, self._after = QSpinBox(), QSpinBox()
        for spin in (self._before, self._after):
            spin.setRange(0, 365)
        self._hours = [QTimeEdit(QTime(h, 0)) for h in (8, 18)]
        for edit, tooltip in zip(
            self._hours,
            (
                "First included Paris time on each day",
                "End of the window on each day (Paris time, excluded); "
                "00:00 is midnight at the end of the day",
            ),
            strict=True,
        ):
            edit.setDisplayFormat("HH:mm")
            edit.setToolTip(tooltip)
            edit.setKeyboardTracking(False)
        self._relief_strength = QDoubleSpinBox()
        self._relief_strength.setRange(0, 100)
        self._relief_strength.setDecimals(0)
        self._relief_strength.setSingleStep(5)
        self._relief_strength.setValue(85)
        self._relief_strength.setSuffix("% background")
        self._relief_strength.setToolTip(
            "Strength of the saved terrain backdrop on both maps"
        )
        self._point_strength = QDoubleSpinBox()
        self._point_strength.setRange(0, 100)
        self._point_strength.setDecimals(0)
        self._point_strength.setSingleStep(5)
        self._point_strength.setValue(95)
        self._point_strength.setSuffix("% points")
        self._point_strength.setToolTip(
            "Opacity of intersection points in every horizontal-plane panel: "
            "0% hides them; 100% makes them fully opaque"
        )
        self._summary = QLabel(
            "Metropolitan France · both available disciplines · "
            "5 x 5 km Lambert-93 cells.\n"
            "Ranked by all-time Vilpellet climb runs. Plane reference: "
            "lowest IGN terrain elevation inside the cell."
        )
        self._summary.setWordWrap(True)
        self._status = QLabel(
            "Choose a data folder to read the prepared cells and Vilpellet climbs."
        )
        self._status.setWordWrap(True)
        self._view = QComboBox()
        self._view.addItem("France + horizontal plane", "overview")
        self._view.addItem("France only", "france")
        self._view.addItem("Horizontal plane only", "planes")
        self._view.setToolTip(
            "Choose the maps to enlarge; use Full screen at the top right"
        )
        self._details = QPushButton("Cell details")
        self._details.setCheckable(True)
        self._details.toggled.connect(self._summary.setVisible)
        self._info, self._howto = help_buttons("planes", self)
        self._terrain_3d_panel = None
        self._terrain_3d = QPushButton("3D terrain · selected cell")
        self._terrain_3d.setToolTip(
            "Open the selected cell in 3D, with a 5 x 5 or 10 x 10 km area: "
            "IGN terrain and all intersections every 20 m, for the selected dates"
        )
        self._terrain_3d.clicked.connect(self._show_terrain_3d)
        self._summary.hide()
        top = FlowLayout()
        for widget in (self._mode, self._cells):
            top.addWidget(widget)
        self._time_settings = QWidget()
        times = FlowLayout(self._time_settings)
        times.setContentsMargins(0, 0, 0, 0)
        for label, widget in (("From", self._start), ("To", self._end)):
            times.addWidget(labeled_control(label, widget))
        heights = FlowLayout()
        heights.addWidget(QLabel("Horizontal plane"))
        heights.addWidget(self._slider)
        heights.addWidget(labeled_control("Height increment", self._step))
        heights.addWidget(self._height)
        heights.addWidget(self._background)
        heights.addWidget(self._relief_strength)
        heights.addWidget(self._point_strength)
        self._figure = Figure(figsize=(10, 5), layout="constrained")
        self._map_ax, self._plane_ax = self._figure.subplots(
            1, 2, width_ratios=[1, 1.5]
        )
        self._plane_axes = [self._plane_ax]
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._canvas.setMinimumHeight(150)
        controls = QVBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(4)
        controls.addLayout(top)
        controls.addWidget(self._summary)
        controls.addWidget(self._time_settings)
        self._daily_settings = QWidget()
        daily = FlowLayout(self._daily_settings)
        daily.setContentsMargins(0, 0, 0, 0)
        self._daily_range_fields = [
            labeled_control("From day", self._daily_start),
            labeled_control("To day", self._daily_end),
        ]
        self._daily_relative_fields = [
            labeled_control("Day", self._day),
            labeled_control("Days before", self._before),
            labeled_control("after", self._after),
        ]
        self._daily_yearly_fields = [
            labeled_control("From", self._season_start),
            labeled_control("To", self._season_end),
            labeled_control("Years", self._first_year),
            labeled_control("to", self._last_year),
        ]
        for field in (*self._daily_relative_fields, *self._daily_yearly_fields):
            field.hide()
        for item in (
            self._daily_date_mode,
            *self._daily_range_fields,
            *self._daily_relative_fields,
            *self._daily_yearly_fields,
            self._best_day,
            labeled_control("Paris hours", self._hours[0]),
            labeled_control("to", self._hours[1]),
        ):
            daily.addWidget(item)
        controls.addWidget(self._daily_settings)
        self._daily_settings.hide()
        controls.addLayout(heights)
        self._provenance = QLabel()
        self._provenance.setWordWrap(True)
        self._provenance.setOpenExternalLinks(True)
        controls.addWidget(self._provenance)
        controls.addWidget(self._image_info)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        navigation = FlowLayout()
        navigation.addWidget(self._toolbar)
        navigation.addWidget(labeled_control("View", self._view))
        self._zoom_in = QPushButton("Zoom +")
        self._zoom_out = QPushButton("Zoom -")
        self._reset_view = QPushButton("Reset cell")
        self._zoom_in.setToolTip("Zoom in, down to a 500 m wide view")
        self._zoom_out.setToolTip(
            "Zoom out up to 10 km, showing the eight neighbouring cells "
            "prepared in the data folder"
        )
        self._reset_view.setToolTip("Return to the selected 5 x 5 km cell")
        for button in (self._zoom_in, self._zoom_out, self._reset_view):
            navigation.addWidget(button)
        for button in (
            self._terrain_3d,
            self._info,
            self._howto,
            self._details,
            self._build,
            self._load,
            self._cancel,
        ):
            navigation.addWidget(button)
        controls.addLayout(navigation)
        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        layout.addWidget(ControlPanel(controls))
        layout.addWidget(self._canvas, 1)
        layout.addWidget(self._status)
        self._build.clicked.connect(self._start_index)
        self._cells.currentIndexChanged.connect(self._cell_changed)
        self._start.dateTimeChanged.connect(self._invalidate_plane)
        self._end.dateTimeChanged.connect(self._invalidate_plane)
        self._load.clicked.connect(self._start_plane)
        self._cancel.clicked.connect(self._cancel_work)
        self._step.currentIndexChanged.connect(self._step_changed)
        self._background.currentIndexChanged.connect(self._relief_changed)
        self._mode.currentIndexChanged.connect(self._mode_changed)
        self._view.currentIndexChanged.connect(self._layout_changed)
        self._best_day.clicked.connect(self._select_best_day)
        self._daily_date_mode.currentIndexChanged.connect(self._daily_date_mode_changed)
        self._daily_start.dateChanged.connect(self._daily_range_changed)
        self._daily_end.dateChanged.connect(self._daily_range_changed)
        self._day.dateChanged.connect(self._daily_changed)
        self._before.valueChanged.connect(self._daily_changed)
        self._after.valueChanged.connect(self._daily_changed)
        for edit in (self._season_start, self._season_end):
            edit.dateChanged.connect(self._season_changed)
        for spin in (self._first_year, self._last_year):
            spin.valueChanged.connect(self._season_changed)
        for edit in self._hours:
            edit.timeChanged.connect(self._draw_plane)
        self._slider.valueChanged.connect(self._slider_changed)
        self._height.valueChanged.connect(self._height_changed)
        self._relief_strength.valueChanged.connect(self._relief_changed)
        self._point_strength.valueChanged.connect(self._point_strength_changed)
        self._canvas.mpl_connect("button_press_event", self._map_clicked)
        self._canvas.mpl_connect("button_release_event", self._pan_finished)
        self._zoom_in.clicked.connect(lambda: self._zoom_plane(0.5))
        self._zoom_out.clicked.connect(lambda: self._zoom_plane(2))
        self._reset_view.clicked.connect(self._reset_plane_view)
        self._set_busy(False)
        self._draw_map()
        self._draw_plane()

    def set_compact(self, compact):
        """Hide status and provenance lines so the plots fill a full-screen window."""
        self._status.setVisible(not compact)
        self._image_info.setVisible(not compact)

    def set_map_focus(self, focused):
        """Enlarge horizontal planes without rebuilding axes or changing their zoom."""
        if focused and self._focused_axes is None:
            if self._map_ax is None or not self._plane_axes:
                return
            from matplotlib.gridspec import GridSpec

            self._focused_axes = (
                self._map_ax,
                self._map_ax.get_visible(),
                self._map_ax.get_in_layout(),
                [(ax, ax.get_subplotspec()) for ax in self._plane_axes],
            )
            self._map_ax.set_visible(False)
            self._map_ax.set_in_layout(False)
            grid = GridSpec(1, len(self._plane_axes), figure=self._figure)
            for i, ax in enumerate(self._plane_axes):
                ax.set_subplotspec(grid[i])
        elif not focused and self._focused_axes is not None:
            map_ax, visible, in_layout, specs = self._focused_axes
            map_ax.set_visible(visible)
            map_ax.set_in_layout(in_layout)
            for ax, spec in specs:
                ax.set_subplotspec(spec)
            self._focused_axes = None
        self._canvas.draw_idle()

    def ensure_loaded(self):
        """Read saved data only; opening a tab never starts expensive preparation."""
        if not self._tried_cache and self._worker is None:
            self._tried_cache = True
            self._run(lambda **_: load_store(), self._saved_index_ready)

    def _saved_index_ready(self, index):
        """Offer explicit preparation when there is no current saved census."""
        if index is None:
            self._index = self._plane = None
            self._cells.clear()
            self._draw_map()
            self._draw_plane()
            self._status.setText(
                "This data folder has no prepared Thermal planes data "
                "(thermal-planes.sqlite3). Choose a complete data folder."
            )
        else:
            self._index_ready(index)

    def invalidate(self):
        """Drop results after the application's archive folders change."""
        self.shutdown()
        self._index = self._plane = None
        self._drop_neighbours()
        self._plane_limits = ((0, 5), (0, 5))
        self._terrain_info = None
        self._provenance.clear()
        self._tried_cache = False
        self._cells.clear()
        self._set_busy(False)
        self._draw_map()
        self._draw_plane()
        self._status.setText("Data folder changed. Reload the prepared data.")

    def shutdown(self):
        """Join the cooperative worker before Qt destroys its owning widget."""
        if self._terrain_3d_panel is not None:
            self._terrain_3d_panel.close()
        self._auto_load = False
        if self._worker is not None:
            self._worker.cancel.set()
            # Disconnect queued results: a completed old archive must never replace
            # the state after a folder change or widget shutdown.
            self._worker.succeeded.disconnect()
            self._worker.wait()
            self._worker = None

    def _cancel_work(self):
        """Request cancellation at the next archive/flight boundary."""
        if self._worker is not None:
            self._worker.cancel.set()
            self._status.setText("Cancelling after the current flight/block…")

    def _set_busy(self, busy):
        """Prevent controls from changing the meaning of an in-flight request."""
        self._build.setEnabled(not busy)
        self._cancel.setEnabled(busy)
        self._terrain_3d.setEnabled(
            not busy
            and bool(getattr(self._index, "has_climb_ranking", False))
            and bool(getattr(self._index, "has_points", False))
            and self._cells.currentData() is not None
        )
        for widget in (
            self._mode,
            self._cells,
            self._start,
            self._end,
            self._load,
            self._day,
            self._daily_date_mode,
            self._daily_start,
            self._daily_end,
            self._season_start,
            self._season_end,
            self._first_year,
            self._last_year,
            self._best_day,
            self._before,
            self._after,
            self._background,
        ):
            widget.setEnabled(
                not busy and self._index is not None and self._cells.count() > 0
            )
        for widget in (self._slider, self._height, self._step):
            widget.setEnabled(not busy and self._plane is not None)
        for widget in (self._zoom_in, self._zoom_out, self._reset_view):
            widget.setEnabled(
                not busy and self._plane is not None and bool(self._plane_axes)
            )

    def _run(self, operation, on_success):
        """Connect a worker result to this widget's GUI-thread slot."""
        if self._worker is not None:
            return
        self._set_busy(True)
        self._worker = _Worker(operation, self)
        self._on_success = on_success
        self._worker.progress.connect(self._progress)
        self._worker.failed.connect(self._operation_failed)
        self._worker.succeeded.connect(self._receive_result)
        self._worker.finished.connect(self._finished)
        self._worker.start()

    def _progress(self, message):
        """Ignore queued messages from an invalidated archive request."""
        if self.sender() is self._worker:
            self._status.setText(message)

    def _operation_failed(self, message):
        """Restore the previous viewport when a neighbour load fails or is cancelled."""
        if self.sender() is self._worker:
            if self._pending_limits is not None:
                self._plane_limits = self._pending_limits
                self._pending_limits = None
                self._draw_plane()
            self._status.setText(message)

    def _receive_result(self, result):
        """Accept only the live, uncancelled request's result on the GUI thread."""
        if self.sender() is self._worker and not self._worker.cancel.is_set():
            self._on_success(result)

    def _finished(self):
        """Restore controls once the worker has fully exited."""
        worker = self.sender()
        if worker is self._worker:
            self._worker = None
            self._set_busy(False)
            if self._auto_load:
                self._auto_load = False
                self._start_plane()
        worker.deleteLater()

    def _start_index(self):
        """Reload only the standalone prepared file; never prepare inside the GUI."""
        self._plane = None
        self._draw_plane()
        self._run(lambda **_: load_store(), self._saved_index_ready)

    def _index_ready(self, index):
        """Expose saved cells with their DEM categories and crossing counts."""
        self._index = index
        self._reliefs.clear()
        previous = self._cells.currentData()
        ranks = {}
        self._cells.blockSignals(True)
        self._cells.clear()
        for cell in index.cells():
            ranks[cell.terrain] = ranks.get(cell.terrain, 0) + 1
            activity = getattr(index, "activity_counts", {}).get((cell.ix, cell.iy))
            population = (
                f"{activity['climb_runs']:,} Vilpellet climbs"
                if activity
                else f"{cell.flights:,} cell visitors"
            )
            self._cells.addItem(
                f"{cell.terrain} #{ranks[cell.terrain]} · "
                f"{population} · terrain {self._terrain_label(cell)}",
                cell,
            )
        if getattr(index, "has_climb_ranking", False):
            self._cells.setToolTip(
                "Ranked by continuous Vilpellet climb runs inside each cell, "
                "using all dates and both disciplines. Each run counts once per "
                "cell, regardless of duration, repeated entries or sample count. "
                "Changing the display method does not change this ranking."
            )
        if previous is not None:
            for i in range(self._cells.count()):
                c = self._cells.itemData(i)
                if (c.ix, c.iy) == (previous.ix, previous.iy):
                    self._cells.setCurrentIndex(i)
                    break
        self._cells.blockSignals(False)
        if not self._dates_initialized and self._cells.count():
            if hasattr(index, "time_extent"):
                start, end = index.time_extent()
            else:
                start, _ = index.defaults(self._cells.currentData())
                end = start + 86399
            for edit, stamp in (
                (self._start, np.floor(start)),
                (self._end, np.ceil(end)),
            ):
                edit.blockSignals(True)
                edit.setDateTime(
                    QDateTime.fromSecsSinceEpoch(int(stamp), QTimeZone(b"Europe/Paris"))
                )
                edit.blockSignals(False)
            for spin, edit in (
                (self._first_year, self._start),
                (self._last_year, self._end),
            ):
                spin.blockSignals(True)
                spin.setValue(edit.date().year())
                spin.blockSignals(False)
            self._dates_initialized = True
        self._cell_changed()
        bands = {self._cells.itemData(i).terrain for i in range(self._cells.count())}
        missing = [name for name in geography.TERRAIN_ORDER if name not in bands]
        self._status.setText(
            f"Indexed archives: {', '.join(index.disciplines)}. "
            + (f"No prepared cell in: {', '.join(missing)}. " if missing else "")
            + "Select an interval in Paris local time to read saved intersections."
            + (
                " Saved selection predates terrain-based ranking; ranks compare "
                "the prepared cells only."
                if not getattr(index, "has_terrain_ranking", True)
                else ""
            )
        )

    def _cell_changed(self, *_):
        """Set a useful one-day window and the cell's fixed all-time height range."""
        cell = self._cells.currentData()
        self._plane = None
        self._drop_neighbours()
        self._plane_limits = ((0, 5), (0, 5))
        self._pending_limits = None
        if cell is not None and self._index is not None:
            self._terrain_info = (
                self._index.terrain_reference(cell)
                if hasattr(self._index, "terrain_reference")
                else None
            )
            self._update_provenance()
            _, height = self._index.defaults(cell)
            self._daily_info = ""
            if hasattr(self._index, "summer_days"):
                days = self._index.summer_days(cell)
                if days:
                    self._day.blockSignals(True)
                    self._day.setDate(
                        self._daily_days.get(
                            (cell.ix, cell.iy),
                            QDate.fromString(days[0][0], "yyyy-MM-dd"),
                        )
                    )
                    self._day.blockSignals(False)
                    self._daily_info = (
                        f"Busiest summer day: {days[0][0]} · "
                        f"{days[0][1]} crossing flights"
                    )
            days = (
                self._day.date().addDays(-self._before.value()),
                self._day.date().addDays(self._after.value()),
            )
            if self._daily_date_mode.currentData() == "range":
                days = self._daily_ranges.get((cell.ix, cell.iy), days)
            self._set_daily_range(*days)
            self._levels = height_levels(cell.max_agl_m, self._step.currentData())
            height = self._levels[np.argmin(abs(self._levels - height))]
            self._height.blockSignals(True)
            self._height.setRange(0, cell.max_agl_m)
            self._height.setValue(height)
            self._height.blockSignals(False)
            launch = cell.launch_median_m
            activity = getattr(self._index, "activity_counts", {}).get(
                (cell.ix, cell.iy)
            )
            highest = (self._terrain_info or {}).get("maximum_m")
            self._summary.setText(
                f"{cell.terrain} #{self._rank(cell)} · 5 x 5 km (Lambert-93) · "
                + (
                    f"{activity['climb_runs']:,} Vilpellet climb runs from "
                    f"{activity['climb_flights']:,} flights (ranking, all dates). "
                    if activity
                    else ""
                )
                + f"{cell.flights:,} distinct "
                f"crossing flights, all dates. "
                f"Lowest terrain (plane reference): {cell.ground_m:.2f} m ASL. "
                + (
                    f"Highest terrain (category): {highest:.2f} m ASL. "
                    if highest is not None
                    else ""
                )
                + f"Maximum height above lowest terrain: {cell.max_agl_m:.1f} m.\n"
                "Category bands use highest terrain: <300 / 300-<800 / "
                "800-<1500 / ≥1500 m. "
                + (
                    f"Launch median (audit only): {launch:.1f} m "
                    f"({cell.launches:,} starts). "
                    if launch is not None
                    else ""
                )
                + (
                    " Few starts: launch median has limited support."
                    if launch is not None and cell.launches < 10
                    else ""
                )
            )
            if hasattr(self._index, "reference_audit"):
                audit = self._index.reference_audit(cell)
                if audit is not None:
                    median = (
                        f"median {audit[1]:g} m"
                        if audit[1] is not None
                        else "no usable launch altitude"
                    )
                    self._summary.setText(
                        self._summary.text() + f" Raw start audit: {audit[0]} starts, "
                        f"{median}; "
                        f"{audit[0] - cell.launches} excluded by origin screen."
                    )
        self._set_busy(self._worker is not None)
        self._height_changed()
        self._draw_map()
        if cell is not None:
            if self._worker is None:
                self._start_plane()
            else:
                self._auto_load = True

    def _update_provenance(self):
        """Keep flight and elevation sources visible independently of backgrounds."""
        terrain = self._terrain_info
        detail = (
            f" · {terrain['grid_m'][0]:g} m grid · {terrain['samples']:,} pixels/cell"
            f" · retrieved {terrain['retrieved_utc'][:10]}"
            if terrain
            else ""
        )
        api = (
            f' · <a href="{escape(terrain["source_url"], quote=True)}">'
            "Download cell elevations (TIFF)</a>"
            if terrain
            else ""
        )
        self._provenance.setText(
            'Terrain: <a href="https://www.data.gouv.fr/datasets/rge-alti-r">'
            "IGN RGE ALTI</a> · Licence Ouverte 2.0"
            f"{detail}{api}. Flights: FFVL CFD IGC · GNSS altitude · "
            "climb labels: Vilpellet et al. (2026)."
        )

    def _invalidate_plane(self, *_):
        """Hide stale results immediately when the selected dates change."""
        self._plane = None
        self._drop_neighbours()
        self._pending_limits = None
        self._update_provenance()
        self._set_busy(self._worker is not None)
        self._status.setText(
            self._date_error()
            or "Selection changed. Load climb intersections for this interval."
        )
        self._draw_plane()

    def _date_error(self):
        """Explain an unusable date selection, or return an empty string."""
        yearly = self._daily_date_mode.currentData() == "yearly"
        if self._mode.currentIndex() and yearly:
            if self._last_year.value() < self._first_year.value():
                return "The last year must be at or after the first year."
            return ""
        start, end = self._read_bounds()
        if end >= start:
            return ""
        if self._mode.currentIndex():
            return "The end date must be at or after the start date."
        return "The end time must be at or after the start time."

    def _start_plane(self):
        """Read the saved climb edges for the selected window."""
        cell = self._cells.currentData()
        if cell is None or self._index is None:
            return
        if self._date_error():
            self._status.setText(self._date_error())
            return
        start, end = self._read_bounds()
        if self._needs_neighbors():
            self._start_neighborhood()
            return
        index = self._index
        self._plane = None
        self._draw_plane()
        self._run(
            lambda **kwargs: index.read_plane(cell, start, end, SOURCE, **kwargs),
            self._plane_ready,
        )

    def _needs_neighbors(self):
        """Check whether the viewport reaches outside the selected square."""
        return any(lo < 0 or hi > 5 for lo, hi in self._plane_limits)

    def _drop_neighbours(self):
        """Forget the neighbourhood of a previous cell, source or interval."""
        self._neighbours = None
        self._neighbour_levels.clear()

    def _start_neighborhood(self):
        """Read the saved neighbourhood, and the cell's plane when not yet loaded."""
        cell, index = self._cells.currentData(), self._index
        if index is None or not hasattr(index, "neighbour_flights"):
            self._status.setText(NEIGHBOURS_MISSING)
            return
        bounds, kind = self._read_bounds(), self._background.currentData()
        reload = self._plane is None

        def load(progress, cancel):
            flights = index.neighbour_flights(cell, SOURCE)
            if flights is None:
                raise ValueError(NEIGHBOURS_MISSING)
            plane = (
                index.read_plane(
                    cell, *bounds, SOURCE, progress=progress, cancel=cancel
                )
                if reload
                else None
            )
            progress("Reading neighbouring backgrounds from the data folder")
            return plane, flights, _neighbour_backdrops(index, cell, kind, cancel)

        self._run(load, self._neighborhood_ready)

    def _neighborhood_ready(self, result):
        """Publish one complete neighbourhood; all tiles use one absolute altitude."""
        plane, flights, images = result
        self._drop_neighbours()
        self._neighbours = (
            np.array([f[0] for f in flights], dtype=object),
            np.array([f[1] for f in flights], dtype=object),
        )
        self._pending_limits = None
        self._backgrounds_ready(images, draw=plane is None)
        if plane is not None:
            self._plane_ready(plane)
        else:
            self._status.setText(
                f"Neighbouring cells read from saved data: {len(flights):,} flights "
                "with climb crossings there, all dates."
            )

    def _neighbour_level(self, cell, level):
        """Saved neighbour crossings on one plane, clipped to the selected interval."""
        if level not in self._neighbour_levels:
            start, end = self._read_bounds()
            frame = self._index.neighbour_points(cell, SOURCE, level)
            frame = frame.loc[frame.utc.between(start, end)].reset_index(drop=True)
            disciplines, flight_ids = self._neighbours
            number = frame.pop("flight").to_numpy(dtype=np.int64)
            frame["discipline"] = disciplines[number]
            frame["flight_id"] = flight_ids[number]
            frame["level"] = level
            self._neighbour_levels[level] = frame
        self._neighbour_levels.move_to_end(level)
        while len(self._neighbour_levels) > 64:
            self._neighbour_levels.popitem(last=False)
        return self._neighbour_levels[level]

    def _zoom_plane(self, factor):
        """Bound the display width to 0.5-10 km within the available 3 x 3 cells."""
        if self._worker is not None or self._plane is None:
            return
        previous = self._plane_limits
        limits = []
        for lo, hi in self._plane_limits:
            width = float(np.clip((hi - lo) * factor, 0.5, 10))
            centre = float(np.clip((lo + hi) / 2, -5 + width / 2, 10 - width / 2))
            limits.append((centre - width / 2, centre + width / 2))
        self._plane_limits = tuple(limits)
        if self._needs_neighbors() and self._neighbours is None:
            self._pending_limits = previous
            self._start_neighborhood()
        else:
            self._draw_plane()

    def _reset_plane_view(self):
        """Restore the central cell without changing height, dates or segmentation."""
        self._plane_limits = ((0, 5), (0, 5))
        self._draw_plane()

    def _pan_finished(self, event):
        """Keep pan and toolbar zoom within the supported neighbourhood extent."""
        if event.inaxes not in self._plane_axes or self._worker is not None:
            return
        previous = self._plane_limits
        limits = []
        for lo, hi in (event.inaxes.get_xlim(), event.inaxes.get_ylim()):
            width = float(np.clip(hi - lo, 0.5, 10))
            centre = float(np.clip((lo + hi) / 2, -5 + width / 2, 10 - width / 2))
            limits.append((centre - width / 2, centre + width / 2))
        if np.allclose(limits, self._plane_limits):
            return
        self._plane_limits = tuple(limits)
        if self._needs_neighbors() and self._neighbours is None:
            self._pending_limits = previous
            self._start_neighborhood()
        else:
            self._draw_plane()

    def _plane_ready(self, plane):
        """Report absent models/UTC explicitly, including an entirely empty slice."""
        self._plane = plane
        span = (
            "from the first to the last selected day"
            if self._mode.currentIndex()
            else "in the interval"
        )
        self._status.setText(
            f"{plane.selected:,} cell visitors {span}, at any altitude "
            f"and in any flight phase; {plane.cached:,} read from the data folder; "
            f"{plane.unclassified:,} entirely unclassified by this method; "
            f"{plane.unavailable:,} unavailable. "
            f"{plane.unknown_clock:,} all-time visitors lack a recoverable UTC origin. "
            "Only climb intersections with the selected plane are drawn; "
            "a visiting flight can contribute no points."
        )
        self._draw_plane()

    def _cell_flights(self, windows=None):
        """Distinct flights inside the cell in the selection, thermal or not.

        A flight counts when its time span in the cell overlaps the loaded
        interval, and with ``windows`` at least one of the daily hour windows.
        """
        visits = getattr(self._plane, "visits", None)
        if visits is None:
            return None
        if windows is None:
            return len(visits)
        return int(overlaps_windows(visits.start, visits.end, windows).sum())

    def _utc_bounds(self):
        """UTC epoch seconds, independent of the computer's local timezone."""
        return (
            self._start.dateTime().toSecsSinceEpoch(),
            self._end.dateTime().toSecsSinceEpoch(),
        )

    def _read_bounds(self):
        """Interval to load: the whole interval, or the span of the selected days."""
        if not self._mode.currentIndex():
            return self._utc_bounds()
        if self._daily_date_mode.currentData() == "yearly":
            days = self._selected_days()
            if not days:
                return (0.0, -1.0)
            first, last = days[0], days[-1]
        else:
            first = self._daily_start.date().toPyDate()
            last = self._daily_end.date().toPyDate()
        return local_bounds(first)[0], local_bounds(last)[1] - 1e-6

    def _selected_days(self):
        """Paris civil days pooled by the daily hour window, in calendar order."""
        if self._daily_date_mode.currentData() == "yearly":
            first, last = (
                (edit.date().month(), edit.date().day())
                for edit in (self._season_start, self._season_end)
            )
            years = range(self._first_year.value(), self._last_year.value() + 1)
            return season_days(first, last, years)
        first = self._daily_start.date().toPyDate()
        last = self._daily_end.date().toPyDate()
        return [
            date.fromordinal(n) for n in range(first.toordinal(), last.toordinal() + 1)
        ]

    def _hour_window(self):
        """Paris clock hours (fractional) of the daily window, or None if empty."""
        start, end = (e.time().hour() + e.time().minute() / 60 for e in self._hours)
        end = end or 24.0
        return (start, end) if start < end else None

    def _windows(self):
        """UTC [start, end) of the hour window on each selected day.

        None means no daily filter: the whole interval is shown.
        """
        if not self._mode.currentIndex():
            return None
        hours = self._hour_window()
        if hours is None:
            return np.empty((0, 2))
        days = self._selected_days()
        # Height redraws reuse the windows; only a new selection recomputes them.
        key = (tuple(days), hours)
        if self._window_cache[0] != key:
            self._window_cache = (key, wall_windows(days, hours))
        return self._window_cache[1]

    def _selection_text(self):
        """Describe the pooled days and hours for titles and the 3D window."""
        hours = " to ".join(e.time().toString("HH:mm") for e in self._hours)
        if self._daily_date_mode.currentData() == "yearly":
            dates = (
                f"{self._season_start.date().toString('dd MMM')} → "
                f"{self._season_end.date().toString('dd MMM')} of each year "
                f"{self._first_year.value()}-{self._last_year.value()}"
            )
        else:
            first, last = (
                datetime.fromtimestamp(t, PARIS).date() for t in self._read_bounds()
            )
            dates = f"{first} → {last}"
        return f"{dates} · {len(self._selected_days())} days pooled · {hours} Paris"

    def _select_best_day(self):
        """Restore the recommendation as one day or as the relative anchor."""
        cell = self._cells.currentData()
        if (
            self._index is not None
            and cell is not None
            and hasattr(self._index, "summer_days")
        ):
            days = self._index.summer_days(cell)
            if days:
                day = QDate.fromString(days[0][0], "yyyy-MM-dd")
                if self._daily_date_mode.currentData() == "range":
                    self._set_daily_range(day, day)
                    self._daily_range_changed()
                else:
                    self._day.setDate(day)

    def _set_daily_range(self, first, last):
        """Update both calendar fields without loading an intermediate range."""
        for edit, day in ((self._daily_start, first), (self._daily_end, last)):
            edit.blockSignals(True)
            edit.setDate(day)
            edit.blockSignals(False)

    def _daily_range_changed(self, *_):
        """Remember explicit inclusive dates and wait for the user's load request."""
        cell = self._cells.currentData()
        if cell is not None:
            self._daily_ranges[(cell.ix, cell.iy)] = (
                self._daily_start.date(),
                self._daily_end.date(),
            )
        if self._mode.currentIndex():
            self._invalidate_plane()

    def _season_changed(self, *_):
        """Yearly dates change the loaded span; wait for the user's load request."""
        if self._mode.currentIndex() and self._daily_date_mode.currentData() == (
            "yearly"
        ):
            self._invalidate_plane()

    def _daily_date_mode_changed(self, *_):
        """Show the controls of direct, relative or yearly date selection."""
        mode = self._daily_date_mode.currentData()
        for fields, shown in (
            (self._daily_range_fields, "range"),
            (self._daily_relative_fields, "around"),
            (self._daily_yearly_fields, "yearly"),
        ):
            for field in fields:
                field.setVisible(mode == shown)
        self._best_day.setVisible(mode != "yearly")
        if mode == "around":
            # Keep the relative controls' own anchor and offsets when returning.
            self._daily_changed()
        elif self._mode.currentIndex():
            self._invalidate_plane()

    def _daily_changed(self, *_):
        """Convert the relative window into dates without starting partial loads."""
        cell = self._cells.currentData()
        if cell is not None:
            self._daily_days[(cell.ix, cell.iy)] = self._day.date()
        self._set_daily_range(
            self._day.date().addDays(-self._before.value()),
            self._day.date().addDays(self._after.value()),
        )
        self._daily_range_changed()

    def _mode_changed(self, *_):
        """Change time selection independently from the visual layout."""
        daily = bool(self._mode.currentIndex())
        self._time_settings.setVisible(not daily)
        self._daily_settings.setVisible(daily)
        self._plane = None
        self._invalidate_plane()
        self._start_plane()

    def _layout_changed(self, *_):
        """Allocate the figure to the chosen maps without loading or changing data."""
        self._figure.clear()
        self._map_ax = None
        self._plane_axes = []
        view = self._view.currentData()
        if view == "france":
            self._map_ax = self._figure.subplots()
        elif view == "overview":
            self._map_ax, ax = self._figure.subplots(1, 2, width_ratios=[1, 1.5])
            self._plane_axes = [ax]
        else:
            self._plane_axes = [self._figure.subplots()]
        self._plane_ax = self._plane_axes[0] if self._plane_axes else None
        self._toolbar.update()
        self._draw_map()
        self._draw_plane()

    def _step_changed(self, *_):
        cell = self._cells.currentData()
        if cell is not None:
            self._levels = height_levels(cell.max_agl_m, self._step.currentData())
            self._height.setSingleStep(self._step.currentData())
            self._height_changed()

    def _slider_changed(self, value):
        """Choose a height without rereading or relabelling flights."""
        self._height.setValue(self._levels[value])

    def _height_changed(self, *_):
        index = int(np.argmin(abs(self._levels - self._height.value())))
        self._height.blockSignals(True)
        self._height.setValue(self._levels[index])
        self._height.blockSignals(False)
        self._slider.blockSignals(True)
        self._slider.setRange(0, len(self._levels) - 1)
        self._slider.setValue(index)
        self._slider.blockSignals(False)
        self._draw_plane()

    def _show_terrain_3d(self):
        """Open this tab's selected cell and capture its current time selection."""
        cell = self._cells.currentData()
        if self._index is None or cell is None:
            return
        if self._date_error():
            self._status.setText(self._date_error())
            return
        daily = bool(self._mode.currentIndex())
        if daily and self._hour_window() is None:
            self._status.setText("The daily hour window must start before it ends.")
            return
        start, end = self._read_bounds()
        try:
            from .thermal_3d import Thermal3D
        except ImportError:
            self._status.setText(
                "The 3D view could not start: its OpenGL libraries are missing."
            )
            return
        if self._terrain_3d_panel is None:
            self._terrain_3d_panel = Thermal3D(self)
        self._terrain_3d_panel.load(
            self._index,
            cell,
            start,
            end,
            windows=self._windows(),
            selection=self._selection_text() if daily else None,
        )
        self._terrain_3d_panel.show()
        self._terrain_3d_panel.raise_()

    def _rank(self, cell):
        """Population rank is local to an altitude category, never across categories."""
        peers = [
            self._cells.itemData(i)
            for i in range(self._cells.count())
            if self._cells.itemData(i).terrain == cell.terrain
        ]
        return peers.index(cell) + 1

    def _terrain_label(self, cell):
        """Show both terrain extrema, with an explicit unavailable maximum."""
        reference = (
            self._index.terrain_reference(cell)
            if hasattr(self._index, "terrain_reference")
            else None
        )
        highest = (reference or {}).get("maximum_m")
        maximum = f"{highest:.0f}" if highest is not None else "n/a"
        return f"min {cell.ground_m:.0f} · max {maximum} m ASL"

    def _saved_relief(self, cell=None):
        """Decode each saved image once; slider redraws use only in-memory arrays."""
        kind = self._background.currentData()
        if kind == "none":
            return None
        key = (kind, "france" if cell is None else (cell.ix, cell.iy))
        if key not in self._reliefs and self._index is not None:
            self._reliefs[key] = _read_backdrop(self._index, cell, kind)
        if key in self._reliefs:
            self._reliefs.move_to_end(key)
        while len(self._reliefs) > 12:
            self._reliefs.popitem(last=False)
        return self._reliefs.get(key)

    def _point_strength_changed(self, *_):
        """Update point opacity in place, preserving the loaded slice and viewport."""
        alpha = self._point_strength.value() / 100
        for ax in self._plane_axes:
            for artist in ax.collections:
                artist.set_alpha(alpha)
            legend = ax.get_legend()
            if legend is not None:
                for artist in legend.legend_handles:
                    artist.set_alpha(alpha)
        self._canvas.draw_idle()

    def _relief_changed(self, *_):
        """Adjust background strength without changing any scientific result."""
        cell, kind = self._cells.currentData(), self._background.currentData()
        if (
            self._neighbours is not None
            and self._worker is None
            and kind != "none"
            and any(
                (kind, (f.ix, f.iy)) not in self._reliefs
                for f in neighbour_frames(cell)
            )
        ):
            index = self._index
            self._run(
                lambda cancel, **_: _neighbour_backdrops(index, cell, kind, cancel),
                self._backgrounds_ready,
            )
            return
        self._draw_map()
        self._draw_plane()

    def _backgrounds_ready(self, images, draw=True):
        """Keep backgrounds decoded by the worker; the GUI thread only draws."""
        for key, image in images.items():
            self._reliefs[key] = image
            self._reliefs.move_to_end(key)
        if draw:
            self._draw_map()
            self._draw_plane()

    def _draw_map(self):
        """Show numbered, category-coloured cells and separated ranking callouts."""
        ax = self._map_ax
        self._map_labels = []
        if ax is None:
            return
        ax.clear()
        basemap = geography.load_basemap()
        if basemap:
            geography.draw_land(ax, basemap["france"]["rings"], geography.FRANCE_EXTENT)
        else:
            ax.set_xlim(-5.5, 10)
            ax.set_ylim(41, 51.5)
        saved = self._saved_relief()
        if saved is not None:
            pixels, info = saved
            w, s, e, n = info["extent"]
            ax.imshow(
                pixels,
                extent=(w, e, s, n),
                origin="upper",
                alpha=self._relief_strength.value() / 100,
                zorder=0.5,
                aspect=1 / np.cos(np.deg2rad(46.25)),
            )
        current = self._cells.currentData()
        colors = dict(
            zip(
                geography.TERRAIN_ORDER,
                ("#197548", "#ad6412", "#7444a6", "#1d609a"),
                strict=True,
            )
        )
        codes = dict(zip(geography.TERRAIN_ORDER, ("P", "H", "L", "M"), strict=True))
        for i in range(self._cells.count()):
            cell = self._cells.itemData(i)
            rank = self._rank(cell)
            activity = getattr(self._index, "activity_counts", {}).get(
                (cell.ix, cell.iy)
            )
            score = activity["climb_runs"] if activity else cell.flights
            west, south, east, north = cell.bounds
            lon, lat = unproject([west, east, east, west], [south, south, north, north])
            color = colors[cell.terrain]
            selected = cell == current
            ax.add_patch(
                Polygon(
                    np.column_stack([lon, lat]),
                    facecolor=color,
                    edgecolor="black" if selected else color,
                    linewidth=2 if selected else 0.8,
                    alpha=0.85,
                    zorder=3,
                )
            )
            cx, cy = unproject((west + east) / 2, (south + north) / 2)
            ax.plot(
                cx,
                cy,
                "o",
                color=color,
                markersize=7 if selected else 4,
                markeredgecolor="black" if selected else "white",
                markeredgewidth=0.7,
                zorder=4,
            )
            # Callouts have fixed distinct rows even when 5 km cells are neighbours.
            # P/H on the west, L/M on the east; each retains its category and rank.
            band = geography.TERRAIN_ORDER.index(cell.terrain)
            tx = 0.01 if band < 2 else 0.59
            ty = 0.93 - (band % 2) * 0.32 - (rank - 1) * 0.073
            annotation = ax.annotate(
                f"{codes[cell.terrain]}{rank} · {score:,}\n{self._terrain_label(cell)}",
                (cx, cy),
                xytext=(tx, ty),
                textcoords="axes fraction",
                fontsize=8,
                color=color,
                fontweight="bold" if selected else "normal",
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "fc": "white",
                    "ec": color if selected else "none",
                    "alpha": 0.94,
                },
                arrowprops={
                    "arrowstyle": "-",
                    "color": color,
                    "lw": 1.2 if selected else 0.55,
                    "alpha": 0.7,
                },
                zorder=6 if selected else 5,
            )
            self._map_labels.append((annotation, i))
        ax.text(
            0.02,
            0.02,
            "P: Plains   H: Hills\nL: Low mountains   M: High mountains\n"
            + (
                "Count = Vilpellet climb runs\n"
                if getattr(self._index, "has_climb_ranking", False)
                else "Count = cell visitors\n"
            )
            + "min / max = terrain elevations (m above sea level)\n"
            + "Number after letter = rank within category",
            transform=ax.transAxes,
            fontsize=7,
            va="bottom",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9},
            zorder=7,
        )
        ax.set_title(
            "Top Vilpellet climb cells per terrain category"
            if getattr(self._index, "has_climb_ranking", False)
            else (
                "Top visitor cells per highest-terrain category"
                if getattr(self._index, "has_terrain_ranking", True)
                else "Prepared cells by highest-terrain category"
            ),
            fontsize=11,
        )
        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")
        self._canvas.draw_idle()

    def _map_clicked(self, event):
        """Select a winning cell by clicking its map marker as well as the dropdown."""
        if (
            self._map_ax is None
            or event.inaxes is not self._map_ax
            or self._worker is not None
        ):
            return
        renderer = self._canvas.get_renderer()
        for annotation, i in self._map_labels:
            if (
                annotation.get_bbox_patch()
                .get_window_extent(renderer)
                .contains(event.x, event.y)
            ):
                self._cells.setCurrentIndex(i)
                return
        for i in range(self._cells.count()):
            cell = self._cells.itemData(i)
            west, south, east, north = cell.bounds
            center = unproject((west + east) / 2, (south + north) / 2)
            px, py = self._map_ax.transData.transform(center)
            if np.hypot(px - event.x, py - event.y) < 12:
                self._cells.setCurrentIndex(i)
                return

    def _draw_plane(self, *_):
        """Filter saved points by level and Paris hour; keep all panels aligned."""
        cell = self._cells.currentData()
        saved = (
            self._saved_relief(cell if self._plane_axes else None)
            if cell is not None
            else None
        )
        info = saved[1] if saved else {}
        if (
            self._background.currentData() == "aerial"
            and saved
            and not self._plane_axes
        ):
            self._image_info.setText(
                "France overview: IGN mosaic with multiple acquisition dates. "
                "Select a horizontal plane to see its cell's acquisition dates. "
                f"Downloaded: {info.get('retrieved', '')[:10]}."
            )
        elif self._background.currentData() == "aerial" and saved:
            dates = info.get("acquisition_dates", [])
            acquisition = ", ".join(dates) if dates else "not supplied by IGN"
            self._image_info.setText(
                f"IGN aerial acquisition dates in this cell: {acquisition}. "
                "Dates differ from flight dates; France is a multi-date mosaic. "
                f"Downloaded: {info.get('retrieved', '')[:10]} · 1.25 m/pixel."
            )
        elif saved:
            resolution = ""
            if cell is not None and self._plane_axes:
                width = saved[0].shape[1]
                metres = (info["extent"][2] - info["extent"][0]) / width
                resolution = f" · {width:,} px · {metres:.2f} m/pixel"
            self._image_info.setText(
                info.get("attribution", "Saved relief") + resolution
            )
        else:
            self._image_info.setText(
                "Background not prepared"
                if self._background.currentData() != "none"
                else ""
            )
        points = None
        # Preserve the exact terminal level, even when the spin box rounds it.
        height = self._levels[self._slider.value()]
        if self._plane is not None and cell is not None and self._plane_axes:
            level = int(np.argmin(abs(height_levels(cell.max_agl_m) - height)))
            if self._plane.points is not None:
                points = self._plane.points.loc[self._plane.points.level == level]
            else:
                points = plane_intersections(
                    self._plane.edges, cell, height, *self._read_bounds()
                )
            if self._neighbours is not None:
                # Same 10 m lattice index: the neighbours lie on this cell's planes.
                points = pd.concat(
                    [points, self._neighbour_level(cell, level)], ignore_index=True
                )
        daily = bool(self._mode.currentIndex())
        windows = self._windows()
        image_cells = (
            [cell, *neighbour_frames(cell)]
            if self._neighbours is not None and cell is not None
            else [cell]
        )
        images = [self._saved_relief(c) for c in image_cells if c is not None]
        for ax in self._plane_axes:
            ax.clear()
            ax.set(
                xlim=self._plane_limits[0],
                ylim=self._plane_limits[1],
                xlabel="East from selected cell (km)",
                ylabel="North from selected cell (km)",
            )
            ax.set_aspect("equal")
            ax.grid(alpha=0.15)
            for backdrop in images:
                if backdrop is None:
                    continue
                pixels, info = backdrop
                w, s, e, n = info["extent"]
                west, south, _, _ = cell.bounds
                visible = _visible_part(
                    pixels,
                    (
                        (w - west) / 1000,
                        (e - west) / 1000,
                        (s - south) / 1000,
                        (n - south) / 1000,
                    ),
                    self._plane_limits,
                    ax.get_window_extent().width,
                )
                if visible is None:
                    continue
                ax.imshow(
                    visible[0],
                    extent=visible[1],
                    origin="upper",
                    interpolation="nearest",
                    alpha=self._relief_strength.value() / 100,
                    zorder=0,
                )
            if saved is not None:
                ax.text(
                    0.01,
                    0.01,
                    saved[1].get("attribution", "Relief: Esri / contributors"),
                    transform=ax.transAxes,
                    fontsize=6,
                    zorder=4,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
                )
            selected = points
            if selected is not None and not selected.empty:
                west, south, _, _ = cell.bounds
                (xmin, xmax), (ymin, ymax) = self._plane_limits
                selected = selected.loc[
                    selected.x.between(
                        west + xmin * 1000, west + xmax * 1000, inclusive="left"
                    )
                    & selected.y.between(
                        south + ymin * 1000, south + ymax * 1000, inclusive="left"
                    )
                ]
            if self._neighbours is not None:
                from matplotlib.patches import Rectangle

                ax.add_patch(
                    Rectangle(
                        (0, 0),
                        5,
                        5,
                        fill=False,
                        edgecolor="black",
                        linewidth=1.2,
                        linestyle="--",
                        zorder=4,
                    )
                )
            if windows is not None and selected is not None and not selected.empty:
                selected = selected.loc[in_windows(selected.utc, windows)]
            count = len(selected) if selected is not None else 0
            flights = 0
            if count:
                flights = len(selected[["discipline", "flight_id"]].drop_duplicates())
                west, south, _, _ = cell.bounds
                for discipline, group in selected.groupby("discipline"):
                    from ..core.disciplines import DISCIPLINES

                    ax.scatter(
                        (group.x - west) / 1000,
                        (group.y - south) / 1000,
                        s=23,
                        alpha=self._point_strength.value() / 100,
                        edgecolors="white",
                        linewidths=0.6,
                        zorder=3,
                        color=DISCIPLINES[discipline].color,
                        label=f"{discipline}: {len(group):,}",
                    )
                ax.legend(
                    fontsize=7,
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.13),
                    ncol=3,
                    columnspacing=0.8,
                    handletextpad=0.3,
                    borderaxespad=0,
                )
            else:
                message = (
                    "No intersections at this height and time"
                    if points is not None
                    else "Load saved intersections"
                )
                if daily and self._hour_window() is None:
                    message = "The window must start before it ends"
                ax.text(
                    0.5,
                    0.5,
                    message,
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=9,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
                )
            prefix = (
                f"{self._hours[0].time().toString('HH:mm')}-"
                f"{self._hours[1].time().toString('HH:mm')} Paris on each day\n"
                if daily
                else ""
            )
            crossing = self._cell_flights(windows)
            if crossing is not None:
                prefix += f"{crossing:,} cell visitors · any altitude / flight phase\n"
            ax.set_title(
                f"{prefix}z = {height:.2f} m above lowest terrain\n"
                + (f"Plane: {cell.ground_m + height:.1f} m ASL · " if cell else "")
                + f"{count:,} climb intersections · {flights:,} contributing flights",
                fontsize=10,
            )
        if daily:
            error = self._date_error()
            self._figure.suptitle(
                (
                    f"Invalid date range: {error}"
                    if error
                    else self._selection_text()
                    + (
                        f" · {self._daily_info}"
                        if self._daily_info
                        and self._daily_date_mode.currentData() != "yearly"
                        else ""
                    )
                ),
                fontsize=10,
            )
        else:
            self._figure.suptitle("")
        self._canvas.draw_idle()
