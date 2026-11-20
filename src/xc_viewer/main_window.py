"""The trajectory viewer's main window: wires the flight picker to the plot.

The only module besides :mod:`xc_viewer.app` and the ``widgets`` package that
imports Qt (see the package docstring).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QAction, QPalette
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from xc_viewer.core.style import DISCIPLINE_COLORS

from . import data, datafolder, plotting, segmentation
from .widgets.about import AboutDialog
from .widgets.flight_picker import FlightPicker
from .widgets.group_flights import GroupFlights
from .widgets.info_browser import set_methods_opener
from .widgets.map_focus import MapFocus
from .widgets.map_view import MapView
from .widgets.plot_controls import PlotControls
from .widgets.route_comparison import RouteComparison
from .widgets.sources_methods import SourcesMethods
from .widgets.thermal_density import ThermalDensity
from .widgets.thermal_plane import ThermalPlane

if TYPE_CHECKING:
    from mpl_toolkits.mplot3d import Axes3D

    from .core.disciplines import Discipline
    from .core.preproc.enu import LocalFrame
    from .core.preproc.pipeline import FlightResult

# Why a flight is unlabelled, in words a status line can carry.  The keys are the
# `phase_reason` values of both segmentations.
_REASON_TEXT = {
    "flight_shorter_than_author_guard": (
        "shorter than the author's 3600-fix minimum flight"
    ),
    "cadence_above_gate": "logged slower than the 1.2 s cadence gate",
    "non_increasing_or_repeated_time": "repeated or non-increasing times",
    "segment_shorter_than_persistence_window": (
        "every segment shorter than the 30-fix persistence window"
    ),
    "dropped_flight_tail": "discarded as flight tail",
    "run_shorter_than_minimum": "every run shorter than the configured minimum",
    "search_not_followed_by_climb": "no search run confirmed by a climb",
    "outside_saved_runs": "outside every saved run",
    "not_in_saved_runs": "not in the saved runs",
}


class MainWindow(QMainWindow):
    """The whole application: a flight picker beside a redrawable plot."""

    def __init__(self) -> None:
        """Build the picker/controls/canvas layout and show an empty plot."""
        super().__init__()
        self._update_title()
        # Leave space for the native title bar and window frame on small screens.
        initial_size = QSize(1300, 820)
        screen = self.screen()
        if screen is not None:
            available = screen.availableGeometry().size() - QSize(40, 60)
            initial_size = initial_size.boundedTo(available)
        self.resize(initial_size)
        self._about_dialog: AboutDialog | None = None
        about_action = QAction("About XC Viewer", self)
        about_action.setMenuRole(QAction.MenuRole.AboutRole)
        about_action.triggered.connect(self.show_about)
        self.menuBar().addMenu("&Help").addAction(about_action)

        from .core.config import load_preproc_config

        self._cfg = load_preproc_config()
        self._igc_path: Path | None = None
        self._discipline: Discipline | None = None
        self._flight_id: str | None = None
        self._raw: data.RawTrack | None = None
        self._cleaned: FlightResult | None = None
        self._tracks: dict[str, data.PhaseTrack | None] = {}
        self._phases: data.PhaseTrack | None = None
        self._status_head = ""
        self._frame: LocalFrame | None = None
        self._view_key: tuple | None = None
        self._saved_views: dict[tuple, dict] = {}
        self._last_view_controls = (-60, 30, 100)
        self._has_3d_data = False

        self._picker = FlightPicker()
        self._picker.flight_chosen.connect(self._on_flight_chosen)
        self._picker.folders_changed.connect(self._on_folders_changed)
        self._picker.data_folder_requested.connect(self._on_choose_data_folder)
        self._picker.setMinimumWidth(260)
        self._picker.setMaximumWidth(420)

        self._controls = PlotControls()
        self._controls.changed.connect(self._redraw)
        self._controls.view_changed.connect(self._apply_view)
        self._controls.reset_view_requested.connect(self._reset_view)
        self._controls.save_pdf_requested.connect(self._on_save_pdf)

        self._figure = Figure(figsize=(7.5, 6.5))
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)

        trajectory_tab = QWidget()
        trajectory_layout = QVBoxLayout(trajectory_tab)
        trajectory_layout.setContentsMargins(0, 0, 0, 0)
        trajectory_layout.addWidget(self._controls)
        trajectory_layout.addWidget(self._toolbar)
        trajectory_layout.addWidget(self._canvas, 1)

        self._map_view = MapView()
        self._map_view.flight_chosen.connect(self._on_flight_chosen_from_map)

        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)
        self._tabs.addTab(trajectory_tab, "Trajectory")
        self._tabs.addTab(self._map_view, "Map")
        self._thermal_plane = ThermalPlane()
        self._tabs.addTab(self._thermal_plane, "Thermal planes")
        self._thermal_density = ThermalDensity()
        self._tabs.addTab(self._thermal_density, "Thermal density")
        self._route_comparison = RouteComparison()
        self._tabs.addTab(self._route_comparison, "Routes · 50-300 km")
        self._group_flights = GroupFlights()
        self._tabs.addTab(self._group_flights, "Group flights")
        self._sources_methods = SourcesMethods()
        # "&&": a single "&" would underline the next letter as a shortcut.
        self._tabs.addTab(self._sources_methods, "Sources && methods")
        for index in range(self._tabs.count()):
            page = self._tabs.widget(index)
            page.setAutoFillBackground(True)
            page.setBackgroundRole(QPalette.ColorRole.Base)
        # Links in any Info window open their section of Sources & methods here.
        set_methods_opener(self.show_methods)
        # The map's take-off points are only read from disk the first time this tab is
        # actually shown, not at startup: a full catalog + flights_meta read for both
        # disciplines is seconds of work the app should not pay before its window
        # even appears, for a tab the user may never open.
        self._tabs.currentChanged.connect(self._on_tab_changed)
        self._fullscreen_button = QPushButton("Map full screen")
        self._fullscreen_button.setCheckable(True)
        self._fullscreen_button.setToolTip(
            "Enlarge the current map. Click again to restore the viewer controls."
        )
        self._fullscreen_button.clicked.connect(self._toggle_full_screen)
        self._segmentation_combo = QComboBox()
        for seg in segmentation.SEGMENTATIONS.values():
            self._segmentation_combo.addItem(seg.label, seg.key)
        self._segmentation_combo.setCurrentIndex(
            self._segmentation_combo.findData(segmentation.current().key)
        )
        self._segmentation_combo.setToolTip(
            "Flight phases used by the Trajectory, Thermal planes, Thermal density and "
            "Routes tabs. Vilpellet: the published model. Refit: the same model "
            "estimated again on the FFVL flights."
        )
        self._segmentation_combo.currentIndexChanged.connect(
            self._on_segmentation_changed
        )
        self._corner = QWidget()
        corner_layout = QHBoxLayout(self._corner)
        corner_layout.setContentsMargins(0, 0, 0, 0)
        corner_layout.addWidget(QLabel("Phases:"))
        corner_layout.addWidget(self._segmentation_combo)
        corner_layout.addWidget(self._fullscreen_button)
        self._tabs.setCornerWidget(self._corner)
        self._fullscreen_state = None
        self._map_focus: MapFocus | None = None

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(5)
        splitter.addWidget(self._picker)
        splitter.addWidget(self._tabs)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([320, 980])
        self._splitter = splitter
        self.setCentralWidget(splitter)

        self._redraw()

    def show_about(self) -> None:
        """Show the project's author, internship and research supervision."""
        if self._about_dialog is None:
            self._about_dialog = AboutDialog(self)
        self._about_dialog.show()
        self._about_dialog.raise_()
        self._about_dialog.activateWindow()

    def _toggle_full_screen(self):
        """Toggle map focus explicitly, independently of native window fullscreen."""
        if self._fullscreen_state is not None:
            self._exit_full_screen()
        else:
            self._fullscreen_state = (
                self.windowState(),
                self.geometry(),
                self._splitter.sizes(),
                not self._picker.isHidden(),
            )
            self._fullscreen_button.setChecked(True)
            self._fullscreen_button.setText("Exit map full screen")
            self._focus_map()
            self.showFullScreen()

    def _exit_full_screen(self):
        """Restore the prior layout and native window state from the map toggle."""
        if self._fullscreen_state is not None:
            state = self._fullscreen_state
            self.setWindowState(state[0])
            if not state[0] & (
                Qt.WindowState.WindowMaximized | Qt.WindowState.WindowFullScreen
            ):
                self.setGeometry(state[1])
            self._restore_full_screen_layout(state)

    def _restore_full_screen_layout(self, state):
        """Restore the sidebar even if a resize already cleared the Qt window flag."""
        if self._map_focus is not None:
            self._map_focus.restore()
            self._map_focus = None
        self._thermal_plane.set_map_focus(False)
        if state:
            self._picker.setVisible(state[3])
            self._splitter.setSizes(state[2])
        self._fullscreen_state = None
        self._fullscreen_button.setChecked(False)
        self._fullscreen_button.setText("Map full screen")

    def _focus_map(self):
        """Fill the window with the active plot while keeping its canvas and camera."""
        if self._map_focus is not None:
            return
        tab = self._tabs.currentWidget()
        hide = ()
        if tab is self._route_comparison:
            target = tab._scene_panel
            hide = (tab._locator,)
        elif tab is self._group_flights:
            target = tab._viewer._scene_panel
            hide = (tab._viewer._locator,)
        elif self._tabs.currentIndex() == 0:
            target = self._canvas
        else:
            target = getattr(tab, "_canvas", tab)
        if tab is self._thermal_plane:
            tab.set_map_focus(True)
        self._map_focus = MapFocus(
            self._splitter,
            target,
            hide=hide,
            keep=(self._tabs.tabBar(), self._corner),
        )
        target.setFocus()

    def _on_tab_changed(self, index: int) -> None:
        """Load a tab's own data only the first time it is actually shown."""
        if self._map_focus is not None:
            self._map_focus.restore()
            self._map_focus = None
            self._thermal_plane.set_map_focus(False)
            self._focus_map()
        if self._tabs.widget(index) is self._map_view:
            self._map_view.ensure_loaded()
        elif self._tabs.widget(index) is self._thermal_plane:
            self._thermal_plane.ensure_loaded()
        elif self._tabs.widget(index) is self._thermal_density:
            self._thermal_density.ensure_loaded()
        elif self._tabs.widget(index) is self._route_comparison:
            self._route_comparison.ensure_loaded()
        elif self._tabs.widget(index) is self._group_flights:
            self._group_flights.ensure_loaded()
        elif self._tabs.widget(index) is self._sources_methods:
            self._sources_methods.ensure_loaded()

    def closeEvent(self, event) -> None:  # noqa: N802
        """Cancel archive work before Qt destroys the thermal-plane worker."""
        if self._about_dialog is not None:
            self._about_dialog.close()
        self._thermal_plane.shutdown()
        self._thermal_density.shutdown()
        self._route_comparison.shutdown()
        self._group_flights.shutdown()
        super().closeEvent(event)

    def show_methods(self, anchor: str) -> None:
        """Bring the window forward on one section of the Sources & methods tab."""
        if self._fullscreen_state is not None:
            self._exit_full_screen()
        self._tabs.setCurrentWidget(self._sources_methods)
        self._sources_methods.show_section(anchor)
        self.raise_()
        self.activateWindow()

    def _update_title(self) -> None:
        """Name the data folder in the window title, so it is never ambiguous."""
        folder = datafolder.current()
        suffix = f" — {folder}" if folder is not None else " — no data folder"
        self.setWindowTitle("XC Viewer" + suffix)

    def _on_choose_data_folder(self) -> None:
        """Switch every tab to another data folder, and remember it."""
        from PyQt6.QtCore import QSettings

        from .app import choose_data_folder

        folder = choose_data_folder(self, datafolder.current() or "")
        if folder is None:
            return
        folder = datafolder.use(folder)
        QSettings().setValue(datafolder.SETTINGS_KEY, str(folder))
        self._update_title()
        self._picker.refresh_folders()

    def _on_folders_changed(self) -> None:
        """Invalidate the map and thermal-plane caches when the archive root changes."""
        # The map cached takeoff_points() per discipline on its own (catalog_index's
        # cache was already cleared by the picker); without this it would keep
        # showing whichever root was current when it last loaded, silently stale.
        self._map_view.invalidate()
        self._thermal_plane.invalidate()
        self._thermal_density.invalidate()
        self._route_comparison.invalidate()
        self._group_flights.invalidate()
        if self._tabs.currentWidget() is self._map_view:
            self._map_view.ensure_loaded()

    def _on_flight_chosen_from_map(
        self, igc_path: Path, discipline: Discipline, flight_id: str
    ) -> None:
        """Switch to the Trajectory tab and load the flight picked on the map."""
        self._tabs.setCurrentIndex(0)
        self._on_flight_chosen(igc_path, discipline, flight_id)

    def _phase_status_text(self) -> str:
        """A short status clause for the chosen segmentation of the current flight.

        Returns:
            A sentence naming the classified fraction, or the reason nothing was
            classified when the flight failed the segmenter's own eligibility gate.
        """
        seg = segmentation.current()
        track = self._phases
        if track is None:
            needs = segmentation.needs_text()
            if seg.key == segmentation.VILPELLET.key:
                return f"{seg.label} configuration unavailable; using segment colours."
            return needs or f"{seg.label} segmentation unavailable."
        classified = track.fixes["phase"].ne("unclassified")
        if not classified.any():
            reasons = track.fixes["phase_reason"].value_counts()
            reason = str(reasons.index[0]) if len(reasons) else "unknown"
            text = _REASON_TEXT.get(reason, reason)
            return f"{seg.label} segmentation: none classified ({text})."
        percent = 100.0 * classified.mean()
        return f"{seg.label} segmentation: {percent:.1f}% of cleaned fixes classified."

    def _select_phases(self) -> None:
        """Use the track of the chosen segmentation for the Trajectory tab."""
        self._phases = self._tracks.get(segmentation.current().key)

    def _load_tracks(self) -> list[str]:
        """Label the kept flight with every saved segmentation; return the failures."""
        failures = []
        self._tracks = {}
        for seg in segmentation.SEGMENTATIONS.values():
            try:
                self._tracks[seg.key] = data.load_phases(
                    self._cleaned.fixes, self._discipline, self._flight_id, seg.key
                )
            except Exception as exc:
                self._tracks[seg.key] = None
                failures.append(
                    f"{seg.label} segmentation could not be decoded ({exc})."
                )
        self._select_phases()
        return failures

    def _on_segmentation_changed(self) -> None:
        """Switch the flight phases of every tab to the segmentation just chosen."""
        from PyQt6.QtCore import QSettings

        key = self._segmentation_combo.currentData()
        segmentation.use(key)
        QSettings().setValue(segmentation.SETTINGS_KEY, key)
        self._select_phases()
        if self._cleaned is not None and self._cleaned.kept:
            self._picker.set_status(f"{self._status_head} {self._phase_status_text()}")
        elif segmentation.needs_text():
            self._picker.set_status(segmentation.needs_text())
        self._saved_views.clear()
        self._thermal_plane.invalidate()
        self._thermal_density.invalidate()
        self._route_comparison.invalidate()
        self._redraw()
        self._on_tab_changed(self._tabs.currentIndex())

    # -- flight loading ------------------------------------------------------------
    def _on_flight_chosen(
        self, igc_path: Path, discipline: Discipline, flight_id: str
    ) -> None:
        """Load one flight's raw and cleaned tracks and its flight phases."""
        self._saved_views.clear()
        self._view_key = None
        self._igc_path = igc_path
        self._discipline = discipline
        self._flight_id = flight_id
        self._cleaned = None
        self._tracks = {}
        self._phases = None
        self._frame = None

        try:
            self._raw = data.load_raw(igc_path, self._cfg)
        except Exception as exc:
            self._raw = None
            self._picker.set_status(f"Could not parse {igc_path.name}: {exc}")
            self._redraw()
            return

        status = f"{igc_path.name} — {discipline.name}, flight {flight_id}."
        try:
            self._cleaned = data.load_cleaned(
                igc_path,
                self._cfg,
                source=discipline.source,
                flight_id=flight_id,
                discipline=discipline.name,
            )
            self._frame = data.frame_from_meta(self._cleaned.meta)
            if self._cleaned.kept:
                status += " Current preprocessing: kept."
            if not self._cleaned.kept:
                meta = self._cleaned.meta
                status = (
                    f"{igc_path.name}: the pipeline dropped this flight at "
                    f"{meta.drop_stage} ({meta.drop_reason}); showing raw only."
                )
        except Exception as exc:
            status = (
                f"{igc_path.name}: the pipeline raised while cleaning it ({exc}); "
                "showing raw only."
            )

        self._status_head = status
        if self._cleaned is not None and self._cleaned.kept:
            failures = self._load_tracks()
            status += " " + self._phase_status_text()
            if failures:
                status += " " + " ".join(failures)

        if self._frame is None:
            # No pipeline-produced frame -- either it never ran that far, or it
            # raised. Fall back to raw's own, so the ENU view still shows the full
            # raw track instead of nothing (there is never a cleaned trajectory to
            # align with in this case anyway: this early, kept is always False).
            self._frame = data.raw_only_frame(self._raw)

        self._picker.set_status(status)
        self._redraw()

    # -- drawing -----------------------------------------------------------------
    def _redraw(self) -> None:
        """Draw the current flight under the selected axes, phase filter and camera.

        ``climb_only`` filters the trajectory to the Vilpellet climb fixes before
        drawing, and never draws the raw track.
        """
        frame_kind = self._controls.frame_kind
        is_3d = self._controls.is_3d
        x, y = self._controls.x_column, self._controls.y_column
        z = self._controls.z_column if is_3d else None
        side = self._controls.color_mode == "side"
        climb_only = self._controls.climb_only and not side
        compare = self._controls.color_mode == "compare"
        # Extended with `climb_only`, `compare` and `side`, which change what is drawn:
        # the saved-view lookup below keys on the whole tuple, not just the axes.
        key = (frame_kind, x, y, z, climb_only, compare, side)

        if self._view_key is not None and self._figure.axes:
            old_primary = self._figure.axes[0]
            view = {"xlim": old_primary.get_xlim(), "ylim": old_primary.get_ylim()}
            if self._view_key[3] is not None:
                old_primary3d = cast("Axes3D", old_primary)
                view.update(
                    zlim=old_primary3d.get_zlim(),
                    elev=old_primary3d.elev,
                    azim=old_primary3d.azim,
                    roll=old_primary3d.roll,
                )
            self._saved_views[self._view_key] = view
        saved_view = self._saved_views.get(key)

        if (
            key == self._view_key
            and len(self._figure.axes) == 1
            and not compare
            and not side
        ):
            ax = self._figure.axes[0]
            ax.clear()
        else:
            ax = plotting.make_axes(
                self._figure, is_3d=is_3d, ribbons=compare, side=side
            )
            self._toolbar.update()
        self._view_key = key

        frame = self._frame
        self._has_3d_data = False

        if self._raw is None and self._cleaned is None:
            ax.set_axis_off()
            self._hide_second_panel()
            write = cast("Axes3D", ax).text2D if is_3d else ax.text
            write(
                0.5,
                0.54,
                "Choose a flight to begin",
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=16,
                fontweight="bold",
                color="#193f4a",
                wrap=True,
            )
            write(
                0.5,
                0.47,
                "Open an IGC file or search\nthe catalog on the left.",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=11,
                color="#627781",
                linespacing=1.6,
            )
            self._canvas.draw_idle()
            return

        if frame_kind == "enu" and frame is None:
            self._hide_second_panel()
            # Only reachable when even raw's own fallback frame failed: an IGC file
            # with no decodable position fix at all (data.raw_only_frame).
            plotting.center_message(
                ax,
                "No usable position fixes in this file — switch to the geographic "
                "frame, or pick another flight.",
                is_3d=is_3d,
            )
            self._canvas.draw_idle()
            return

        raw_table = None
        if self._controls.show_raw and self._raw is not None and not climb_only:
            if frame_kind == "geographic":
                raw_table = self._raw.fixes
            else:
                # The early return above already ruled out frame_kind == "enu" with
                # frame is None.
                assert frame is not None
                raw_table = data.raw_to_enu(self._raw, frame)

        color = (
            self._discipline.color
            if self._discipline is not None
            else DISCIPLINE_COLORS["paragliders"]
        )
        draw = self._draw_side_by_side if side else self._draw_panel
        drawn = draw(
            ax, raw_table=raw_table, x=x, y=y, z=z, climb_only=climb_only, color=color
        )
        self._has_3d_data = is_3d and drawn

        if saved_view is not None:
            ax.set_xlim(saved_view["xlim"])
            ax.set_ylim(saved_view["ylim"])
            if is_3d:
                primary3d = cast("Axes3D", ax)
                primary3d.set_zlim(saved_view["zlim"])
                primary3d.view_init(
                    elev=saved_view["elev"],
                    azim=saved_view["azim"],
                    roll=saved_view["roll"],
                    share=True,
                )
        elif is_3d:
            primary3d = cast("Axes3D", ax)
            primary3d.view_init(
                elev=self._controls.elev_deg,
                azim=self._controls.azim_deg,
                share=True,
            )
            factor = 100.0 / self._controls.zoom_percent
            ax.set_xlim(self._scaled(ax.get_xlim(), factor))
            ax.set_ylim(self._scaled(ax.get_ylim(), factor))
            primary3d.set_zlim(self._scaled(primary3d.get_zlim(), factor))
        self._last_view_controls = (
            self._controls.azim_deg,
            self._controls.elev_deg,
            self._controls.zoom_percent,
        )
        self._canvas.draw_idle()

    def _draw_panel(
        self,
        ax,
        *,
        raw_table,
        x: str,
        y: str,
        z: str | None,
        climb_only: bool,
        color: str,
    ) -> bool:
        """Draw the flight's trajectory, phase-coloured on request.

        Args:
            ax: The plot's axes.
            raw_table: The raw track to overlay, or ``None``.
            x: Column to plot on the x-axis.
            y: Column to plot on the y-axis.
            z: Column to plot on the z-axis, or ``None`` for 2D.
            climb_only: Whether to keep only the climb fixes.
            color: The cleaned trajectory's fallback colour.

        Returns:
            Whether a non-empty trajectory was actually plotted.
        """
        phase_track = self._phases
        compare = self._controls.color_mode == "compare"
        differs = self._differences() if compare else None
        self._controls.set_agreement(self._agreement_text() if compare else "")
        cleaned_table = None
        color_by: str | None = "segment_id"
        group_by: str | None = None
        color_map = None
        frame_kind = self._controls.frame_kind
        frame = self._frame
        show_cleaned = self._controls.show_cleaned
        if show_cleaned and self._cleaned is not None and self._cleaned.kept:
            if climb_only:
                cleaned_fixes = (
                    data.climb_only(phase_track.fixes)
                    if phase_track is not None
                    else self._cleaned.fixes.iloc[0:0]
                )
                phase_mode = phase_track is not None
            else:
                phase_mode = (
                    self._controls.color_mode in ("phase", "compare")
                    and phase_track is not None
                    and not phase_track.fixes.empty
                )
                cleaned_fixes = (
                    phase_track.fixes
                    if phase_mode and phase_track is not None
                    else self._cleaned.fixes
                )
                if phase_mode and differs is not None:
                    cleaned_fixes = cleaned_fixes.assign(differs=differs)
            if frame_kind == "geographic" and len(cleaned_fixes):
                # cleaned.kept implies frame is not None: both are set together, at
                # (and only past) pipeline stage (v) -- see data.frame_from_meta.
                assert frame is not None
                cleaned_table = data.cleaned_to_geographic(cleaned_fixes, frame)
            else:
                cleaned_table = cleaned_fixes
            if phase_mode:
                color_by = "phase"
                group_by = "phase_run"
                color_map = plotting.PHASE_COLORS
            elif self._controls.color_mode == "single":
                color_by = None
                group_by = "segment_id"

        has_data = bool(
            (raw_table is not None and len(raw_table))
            or (cleaned_table is not None and len(cleaned_table))
        )
        title = (
            f"{segmentation.current().label} segmentation"
            if color_by == "phase"
            else ""
        )
        if compare and title:
            title += "; x marks the fixes where the other segmentation differs"
        if climb_only and cleaned_table is not None and len(cleaned_table):
            n_thermals = (
                cleaned_table["phase_run"].nunique()
                if "phase_run" in cleaned_table
                else 0
            )
            total_s = 0.0
            if {"t", "phase_run"}.issubset(cleaned_table.columns):
                total_s = float(
                    cleaned_table.groupby("phase_run")["t"]
                    .apply(lambda s: s.max() - s.min())
                    .sum()
                )
            title += f" — {n_thermals} thermals, {total_s / 60:.0f} min"
        if title:
            ax.set_title(title)

        if climb_only and not has_data:
            plotting.center_message(
                ax,
                f"No {segmentation.current().label} climb fixes to show.",
                is_3d=z is not None,
            )
            return False
        if compare and differs is None:
            reason = segmentation.needs_text(segmentation.REFIT.key)
            plotting.center_message(
                ax,
                f"Comparing needs both segmentations of this flight. {reason}".strip(),
                is_3d=z is not None,
            )
            self._figure.axes[1].set_axis_off()
            return False

        plotting.plot_trajectory(
            ax,
            raw=raw_table,
            cleaned=cleaned_table,
            x=x,
            y=y,
            z=z,
            dms=self._controls.dms,
            cleaned_color=color,
            color_by=color_by,
            group_by=group_by,
            color_map=color_map,
        )
        if compare and differs is not None:
            if cleaned_table is not None and "differs" in cleaned_table:
                marked = cleaned_table.loc[cleaned_table["differs"]]
                plotting.mark_differences(ax, marked, x, y, z)
                ax.legend(fontsize=8)
            pair = self._track_pair()
            assert pair is not None
            plotting.draw_ribbons(
                self._figure.axes[1],
                [
                    (seg.label, track.fixes)
                    for seg, track in zip(
                        segmentation.SEGMENTATIONS.values(), pair, strict=True
                    )
                ],
                differs,
            )
        return has_data

    def _hide_second_panel(self) -> None:
        """Blank the right panel of the side-by-side view when nothing is drawn."""
        if len(self._figure.axes) > 1 and self._controls.color_mode == "side":
            self._figure.axes[1].set_axis_off()

    def _draw_side_by_side(
        self, ax, *, raw_table, x, y, z, climb_only: bool, color: str
    ) -> bool:
        """Draw the cleaned track once per segmentation, in two linked panels.

        The panels (:func:`plotting.make_axes` with ``side``) share zoom, pan and, in
        3D, the view angle; one legend serves both. ``climb_only`` is unused: the
        whole track is drawn.

        Returns:
            Whether both panels were drawn.
        """
        pair = self._track_pair()
        self._controls.set_agreement(self._agreement_text())
        if pair is None:
            reason = segmentation.needs_text(segmentation.REFIT.key)
            plotting.center_message(
                ax,
                f"Comparing needs both segmentations of this flight. {reason}".strip(),
                is_3d=z is not None,
            )
            self._hide_second_panel()
            return False
        panels = self._figure.axes[:2]
        for panel, seg, track in zip(
            panels, segmentation.SEGMENTATIONS.values(), pair, strict=True
        ):
            fixes = track.fixes
            if self._controls.frame_kind == "geographic":
                assert self._frame is not None
                fixes = data.cleaned_to_geographic(fixes, self._frame)
            plotting.plot_trajectory(
                panel,
                raw=raw_table,
                cleaned=fixes if self._controls.show_cleaned else None,
                x=x,
                y=y,
                z=z,
                dms=self._controls.dms,
                cleaned_color=color,
                color_by="phase",
                group_by="phase_run",
                color_map=plotting.PHASE_COLORS,
            )
            panel.set_title(seg.label)
        plotting.shared_legend(self._figure, panels)
        return True

    def _track_pair(self) -> tuple[data.PhaseTrack, data.PhaseTrack] | None:
        """The flight's tracks under both segmentations, ``None`` if one is absent."""
        tracks = [self._tracks.get(key) for key in segmentation.SEGMENTATIONS]
        if any(track is None or track.fixes.empty for track in tracks):
            return None
        return cast("tuple[data.PhaseTrack, data.PhaseTrack]", tuple(tracks))

    def _differences(self):
        """Per fix, whether both segmentations label it and give different phases."""
        pair = self._track_pair()
        if pair is None:
            return None
        a, b = (t.fixes["phase"].to_numpy(dtype=object) for t in pair)
        return (a != b) & (a != "unclassified") & (b != "unclassified")

    def _agreement_text(self) -> str:
        """Overall and per-phase agreement of the two segmentations, for the panel."""
        pair = self._track_pair()
        if pair is None:
            return ""
        result = data.agreement(*pair)
        first, second = (s.label for s in segmentation.SEGMENTATIONS.values())
        if result["share"] is None:
            return f"No fix is labelled by both {first} and {second}."
        phases = ", ".join(
            f"{name} {100 * v['share']:.0f}%"
            for name, v in result["by_phase"].items()
            if v["share"] is not None
        )
        return (
            f"{first} and {second} both label {result['both_classified']:,} of "
            f"{result['n_fixes']:,} fixes and give the same phase on "
            f"{100 * result['share']:.1f}% of them. Overlap by phase: {phases}. "
            f"Labelled by {first} only: {result['only_first']:,}; by {second} only: "
            f"{result['only_second']:,}."
        )

    def _reset_view(self) -> None:
        """Reset the camera only when the user explicitly requests it."""
        self._saved_views.clear()
        self._view_key = None
        self._redraw()

    def _apply_view(self) -> None:
        """Orient/zoom the current 3D axes to match the controls, without re-plotting.

        A no-op in 2D, and whenever there is no plotted trajectory to orient (both
        guarded by ``_has_3d_data`` -- see ``_redraw``).
        """
        if not self._has_3d_data:
            self._canvas.draw_idle()
            return
        ax = cast("Axes3D", self._figure.axes[0])
        old_azim, old_elev, old_zoom = self._last_view_controls
        azim, elev, zoom = (
            self._controls.azim_deg,
            self._controls.elev_deg,
            self._controls.zoom_percent,
        )
        # A zoom change must preserve mouse rotation; an orientation change must
        # preserve pan/zoom. Apply only the control that actually changed.
        if azim != old_azim or elev != old_elev:
            ax.view_init(
                elev=elev if elev != old_elev else ax.elev,
                azim=azim if azim != old_azim else ax.azim,
                roll=ax.roll,
                share=True,
            )
        if zoom != old_zoom:
            factor = old_zoom / zoom
            ax.set_xlim(self._scaled(ax.get_xlim(), factor))
            ax.set_ylim(self._scaled(ax.get_ylim(), factor))
            ax.set_zlim(self._scaled(ax.get_zlim(), factor))
        self._last_view_controls = (azim, elev, zoom)
        self._canvas.draw_idle()

    @staticmethod
    def _scaled(limits: tuple[float, float], factor: float) -> tuple[float, float]:
        """``limits`` scaled by ``factor`` around its own midpoint."""
        lo, hi = limits
        center = (lo + hi) / 2.0
        half = (hi - lo) / 2.0 * factor
        return center - half, center + half

    def _on_save_pdf(self) -> None:
        """Prompt for a path and save the current figure as a vector PDF."""
        default_name = f"{self._flight_id or 'trajectory'}.pdf"
        path_str, _ = QFileDialog.getSaveFileName(
            self, "Save trajectory as PDF", default_name, "PDF files (*.pdf)"
        )
        if not path_str:
            return
        try:
            plotting.save_pdf(self._figure, path_str)
        except OSError as exc:
            QMessageBox.warning(self, "Could not save PDF", str(exc))
