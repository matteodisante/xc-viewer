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
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from xc_viewer.core.style import DISCIPLINE_COLORS

from . import data, datafolder, plotting
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

# Display names of the segmentations, keyed as `PlotControls.segmentation`.
SEGMENTATION_NAMES = {"vilpellet": "Vilpellet", "gaussian": "Gaussian"}

# Why a segmenter left a flight unlabelled, in words a status line can carry.  The
# keys are the `phase_reason` values of both segmenters.
_REASON_TEXT = {
    "no_eligible_decisions": "no piece of track logged every 10 s or faster",
    "feature_edge": "every piece of track shorter than the 30-s feature window",
    "quality_masked": "every window touches reconstructed altitude",
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
        self._phases: dict[str, data.PhaseTrack | None] = {}
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
        self._tabs.setCornerWidget(self._fullscreen_button)
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
            keep=(self._tabs.tabBar(), self._fullscreen_button),
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

    def _phase_status_text(self, method: str) -> str:
        """A short status clause for one segmentation of the current flight.

        Returns:
            A sentence naming the classified fraction, or the reason nothing was
            classified when the flight failed the segmenter's own eligibility gate.
        """
        name = SEGMENTATION_NAMES[method]
        track = self._phases.get(method)
        if track is None:
            return f"{name} configuration unavailable."
        classified = track.fixes["phase"].ne("unclassified")
        if not classified.any():
            reasons = track.fixes["phase_reason"].value_counts()
            reason = str(reasons.index[0]) if len(reasons) else "unknown"
            text = _REASON_TEXT.get(reason, reason)
            return f"{name} segmentation: none classified ({text})."
        percent = 100.0 * classified.mean()
        return f"{name} segmentation: {percent:.1f}% of cleaned fixes classified."

    # -- flight loading ------------------------------------------------------------
    def _on_flight_chosen(
        self, igc_path: Path, discipline: Discipline, flight_id: str
    ) -> None:
        """Load one flight's raw and cleaned tracks and both its segmentations."""
        self._saved_views.clear()
        self._view_key = None
        self._igc_path = igc_path
        self._discipline = discipline
        self._flight_id = flight_id
        self._cleaned = None
        self._phases = {}
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

        if self._cleaned is not None and self._cleaned.kept:
            loaders = {
                "vilpellet": data.load_vilpellet_phases,
                "gaussian": data.load_gaussian_phases,
            }
            for method, load in loaders.items():
                try:
                    self._phases[method] = load(self._cleaned.fixes, discipline)
                    status += " " + self._phase_status_text(method)
                except Exception as exc:
                    name = SEGMENTATION_NAMES[method]
                    status += f" {name} segmentation could not be decoded ({exc})."

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

        A phase filter keeps only that phase's fixes and never draws the raw track.
        The saved views are keyed on the filter but not on the segmentation, so
        switching between segmentations keeps the zoom.
        """
        frame_kind = self._controls.frame_kind
        is_3d = self._controls.is_3d
        x, y = self._controls.x_column, self._controls.y_column
        z = self._controls.z_column if is_3d else None
        phase_filter = self._controls.phase_filter
        # Extended with `phase_filter`, which changes what is drawn: the saved-view
        # lookup below keys on the whole tuple, not just the axes.
        key = (frame_kind, x, y, z, phase_filter)

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

        if key == self._view_key and len(self._figure.axes) == 1:
            ax = self._figure.axes[0]
            ax.clear()
        else:
            ax = plotting.make_axes(self._figure, is_3d=is_3d)
            self._toolbar.update()
        self._view_key = key

        frame = self._frame
        self._has_3d_data = False

        if self._raw is None and self._cleaned is None:
            ax.set_axis_off()
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
        if self._controls.show_raw and self._raw is not None and phase_filter is None:
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
        drawn = self._draw_panel(
            ax,
            raw_table=raw_table,
            x=x,
            y=y,
            z=z,
            phase_filter=phase_filter,
            color=color,
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
                )
        elif is_3d:
            primary3d = cast("Axes3D", ax)
            primary3d.view_init(
                elev=self._controls.elev_deg, azim=self._controls.azim_deg
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
        phase_filter: str | None,
        color: str,
    ) -> bool:
        """Draw the flight's trajectory, phase-coloured by the chosen segmentation.

        With both segmentations, Vilpellet is a wide band and Gaussian a thin line
        over it, each coloured by its own phases: where they agree, the line runs
        inside a band of its colour.

        Args:
            ax: The plot's axes.
            raw_table: The raw track to overlay, or ``None``.
            x: Column to plot on the x-axis.
            y: Column to plot on the y-axis.
            z: Column to plot on the z-axis, or ``None`` for 2D.
            phase_filter: The only phase to keep, or ``None`` for all.
            color: The cleaned trajectory's fallback colour.

        Returns:
            Whether a non-empty trajectory was actually plotted.
        """
        frame = self._frame
        cleaned = self._cleaned
        show_cleaned = (
            self._controls.show_cleaned and cleaned is not None and cleaned.kept
        )
        both = self._controls.segmentation == "both"
        methods = list(SEGMENTATION_NAMES) if both else [self._controls.segmentation]

        def placed(fixes):
            if self._controls.frame_kind == "geographic" and len(fixes):
                # cleaned.kept implies frame is not None: both are set together, at
                # (and only past) pipeline stage (v) -- see data.frame_from_meta.
                assert frame is not None
                return data.cleaned_to_geographic(fixes, frame)
            return fixes

        # One phase-coloured layer per segmentation that labelled this flight.
        layers = []
        if show_cleaned and self._controls.color_mode == "phase":
            for method in methods:
                track = self._phases.get(method)
                if track is None or track.fixes.empty:
                    continue
                fixes = track.fixes
                if phase_filter is not None:
                    fixes = data.phase_only(fixes, phase_filter)
                layers.append((method, placed(fixes)))

        common = {"x": x, "y": y, "z": z, "dms": self._controls.dms}
        if not layers:
            cleaned_table = None
            color_by: str | None = "segment_id"
            group_by: str | None = None
            if show_cleaned and phase_filter is None:
                assert cleaned is not None
                cleaned_table = placed(cleaned.fixes)
                if self._controls.color_mode == "single":
                    color_by, group_by = None, "segment_id"
            has_data = bool(
                (raw_table is not None and len(raw_table))
                or (cleaned_table is not None and len(cleaned_table))
            )
            if phase_filter is not None and not has_data:
                plotting.center_message(
                    ax, f"No {phase_filter} fixes to show.", is_3d=z is not None
                )
                return False
            plotting.plot_trajectory(
                ax,
                raw=raw_table,
                cleaned=cleaned_table,
                cleaned_color=color,
                color_by=color_by,
                group_by=group_by,
                **common,
            )
            return has_data

        names = [SEGMENTATION_NAMES[method] for method, _ in layers]
        if phase_filter is None:
            title = (
                f"{names[0]} (band) and {names[1]} (line) segmentations"
                if len(layers) == 2
                else f"{names[0]} segmentation"
            )
        else:
            summaries = [
                f"{name}: {self._runs_summary(table, phase_filter)}"
                for name, (_, table) in zip(names, layers, strict=True)
            ]
            title = " · ".join(summaries)
        ax.set_title(title)

        has_data = bool(
            (raw_table is not None and len(raw_table))
            or any(len(table) for _, table in layers)
        )
        if not has_data:
            plotting.center_message(
                ax,
                f"No {' or '.join(names)} {phase_filter} fixes to show.",
                is_3d=z is not None,
            )
            return False
        for index, (method, table) in enumerate(layers):
            plotting.plot_trajectory(
                ax,
                raw=raw_table if index == 0 else None,
                cleaned=table,
                color_by="phase",
                group_by="phase_run",
                color_map=plotting.PHASE_COLORS,
                cleaned_band=both and method == "vilpellet",
                legend_prefix=f"{SEGMENTATION_NAMES[method]} " if both else "",
                **common,
            )
        return has_data

    @staticmethod
    def _runs_summary(table, phase: str) -> str:
        """``"12 climb runs, 85 min"`` for the runs of one phase in ``table``."""
        if table.empty:
            return f"no {phase}"
        durations = table.groupby("phase_run")["t"].agg(lambda t: t.max() - t.min())
        runs = "run" if len(durations) == 1 else "runs"
        return f"{len(durations)} {phase} {runs}, {durations.sum() / 60:.0f} min"

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
