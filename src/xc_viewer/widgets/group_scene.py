"""One group episode on IGN terrain, replayed in time."""

from datetime import datetime

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from matplotlib import colormaps
from OpenGL import GL
from PyQt6.QtCore import QRectF, Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QGuiApplication, QPainter, QVector3D
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QGraphicsRectItem,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem

from ..catalog_index import is_anonymised_pilot
from ..group_episodes import reeb_graph
from ..thermal_daily import PARIS
from .flow_layout import FlowLayout
from .route_annotations import screen_points
from .route_comparison import RouteComparison, _flight_datetime
from .route_lines import RouteLine

#: Colours of the configurations of an episode, in the order they start.
PALETTE = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#bcbd22",
    "#17becf",
)
#: Colours of the pilots: tab20, its strong colours first.
PILOTS = tuple(
    QColor.fromRgbF(*colormaps["tab20"](k)[:3]).name()
    for k in [*range(0, 20, 2), *range(1, 20, 2)]
)
SOLO = "#9a9893"
LOCAL = "#d9d7d2"
TRACK = "#edf6ff"
EPISODE = "#ffd23f"  # every track while its flight is in the episode
SELECTED = "#1464ff"
CONTACT = "#ffffff"  # contacts when the dots are coloured by pilot
OTHER = "#ffffff"  # members in a configuration of another episode
#: A click picks the member whose dot is at most this many pixels away.
PICK_PX = 18
FOCUS_HINT = "Click a pilot to follow its configuration"
SPEEDS = (10, 30, 60, 120, 300)
STEP_S = 5
TICK_MS = 50
#: Flight seconds between two rebuilds of the flown tracks; the trail covers the rest.
FLOWN_STEP_S = 30
#: Columns of the Who is who window: header and members column (None: the number).
WHO = (
    ("Member", None),
    ("Pilot", "pilot"),
    ("Flight ID", "flight_id"),
    ("Club", "club"),
    ("Wing", "wing"),
    ("Takeoff site", "takeoff_site"),
    ("Takeoff (Paris)", "takeoff"),
    ("Landing site", "landing_site"),
    ("Landing (Paris)", "landing"),
    ("CFD distance (km)", "distance_km"),
)


def _rgba(colour, alpha=1.0):
    return (*QColor(colour).getRgbF()[:3], alpha)


def _segments(pieces):
    """Polylines as the vertex pairs of a GL "lines" item."""
    pairs = [
        np.stack((p[:-1], p[1:]), axis=1).reshape(-1, 3) for p in pieces if len(p) > 1
    ]
    return np.concatenate(pairs).astype(np.float32) if pairs else None


def _conf_colours(configurations):
    """A colour of PALETTE per configuration, taken in turn.

    Each differs from the colours of the configurations in progress when it starts and
    of those it continues.
    """
    out, turn = {}, 0
    for c in configurations.sort_values("t_start", kind="stable").itertuples():
        near = configurations[
            (configurations.t_start <= c.t_start) & (configurations.t_end >= c.t_start)
        ]
        busy = {out[d] for d in near.conf_id if d in out}
        free = [k % len(PALETTE) for k in range(turn, turn + len(PALETTE))]
        pick = next((k for k in free if PALETTE[k] not in busy), turn % len(PALETTE))
        out[int(c.conf_id)] = PALETTE[pick]
        turn = pick + 1
    return out


def _who_text(column, value):
    """One cell of the Who is who window; empty where the catalog has no value."""
    if column in ("takeoff", "landing"):
        return _flight_datetime(value)
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return ""
    if column == "pilot" and is_anonymised_pilot(str(value)):
        return "(anonymised)"
    if column == "distance_km":
        return f"{float(value):.0f}"
    return str(value)


class _WhoIsWho(QDialog):
    """Which flight and which pilot each member number is."""

    def __init__(self, parent):
        """An empty table until `set_members`."""
        super().__init__(parent)
        self.setWindowTitle("Who is who")
        self.resize(1100, 260)
        self.table = QTableWidget(0, len(WHO))
        self.table.setHorizontalHeaderLabels([header for header, _ in WHO])
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        QVBoxLayout(self).addWidget(self.table)

    def set_members(self, members):
        """One row per member, its number in the colour of the pilot."""
        self.table.setRowCount(len(members))
        for i, row in enumerate(members.to_dict("records")):
            for col, (_, column) in enumerate(WHO):
                if column is None:
                    item = QTableWidgetItem(str(i + 1))
                    item.setBackground(QColor(PILOTS[i % len(PILOTS)]))
                    item.setForeground(QColor("#061725"))
                else:
                    item = QTableWidgetItem(_who_text(column, row.get(column)))
                self.table.setItem(i, col, item)


class _Numbers(GLGraphicsItem):
    """Member numbers in white on dark plates beside their dots, at screen size."""

    def __init__(self):
        """Nothing to write until `set_points`."""
        super().__init__()
        self.points, self.texts = np.zeros((0, 3)), []
        self.setGLOptions("translucent")
        self.updateGLOptions({GL.GL_DEPTH_TEST: False, "glDepthMask": (True,)})
        self.setDepthValue(110)

    def set_points(self, points, texts):
        """Write `texts` beside the scene `points`."""
        self.points, self.texts = np.asarray(points, float).reshape(-1, 3), texts
        self.update()

    def paint(self):
        """One plate per number, up and to the right of its dot."""
        if not self.texts:
            return
        self.setupGLState()
        painter = QPainter(self.view())
        painter.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing
        )
        try:
            painter.setFont(QFont("Helvetica", 11, QFont.Weight.Bold))
            metrics = painter.fontMetrics()
            points, visible = screen_points(self, self.points)
            for (x, y), show, text in zip(points, visible, self.texts, strict=True):
                if not show:
                    continue
                w, h = metrics.horizontalAdvance(text) + 8, metrics.height() + 2
                box = QRectF(float(x) + 7, float(y) - h - 3, w, h)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(16, 32, 46, 225))
                painter.drawRoundedRect(box, 4, 4)
                painter.setPen(QColor("#ffffff"))
                painter.drawText(box, Qt.AlignmentFlag.AlignCenter, text)
        finally:
            painter.end()


def nearest_member(points, visible, x, y, radius=PICK_PX):
    """Index of the visible pixel `points` nearest (x, y) within `radius`, or None."""
    if not len(points):
        return None
    gap = np.hypot(points[:, 0] - x, points[:, 1] - y)
    gap[~visible] = np.inf
    k = int(np.argmin(gap))
    return k if gap[k] <= radius else None


class _ClockAxis(pg.AxisItem):
    """UTC seconds shown as the time of day in Paris."""

    def tickStrings(self, values, scale, spacing):  # noqa: N802 (Qt name)
        """One "HH:MM" label per tick."""
        return [datetime.fromtimestamp(v, PARIS).strftime("%H:%M") for v in values]


class EpisodeScene(RouteComparison):
    """Terrain, imagery and tracks of Routes, with one episode replayed on them.

    At the chosen instant every member is a numbered dot in the colour of its
    configuration, joined to the members it is in contact with and followed by the trail
    of its last minutes; tracks are whole or drawn up to the instant. Beside the scene,
    the contact graph G(t). Under it, the membership of every member and the Reeb graph
    of the episode, each with a cursor that moves the instant.
    """

    def __init__(self, parent=None):
        """Add the timeline under the scene; nothing is read until an episode loads."""
        super().__init__(parent, scene_only=True)
        self._episode = None
        self._markers = self._numbers_item = self._flown_at = None
        self._edges, self._trails, self._flown = {}, [], []
        self._conf_colour = {}
        self._reeb_data = None
        self._now = 0.0
        self._focus, self._located, self._clicks = (), {}, None
        self._mode.clear()
        self._mode.addItem("All members", "all")
        self._mode.addItem("Selected rows only", "selected")
        self._line_width.setValue(1.0)
        self._legend.setText(
            "Dots: members at the instant, large and in the colour of their "
            "configuration, white in another episode, small and grey alone · Lines "
            "between dots: pairs in contact, in the same colour · Tracks: one colour "
            "per pilot, faint outside the episode · Bright trail: last minutes · "
            "○ ■ Start and end of the replay"
        )
        self._placeholder.setText("Pick an episode and click Load episode.")
        self._summary.setText("")
        self._table.setColumnCount(7)
        self._table.setHorizontalHeaderLabels(
            [
                "Show / member",
                "Discipline",
                "Flight ID",
                "Takeoff (Paris)",
                "Landing (Paris)",
                "In the episode (min)",
                "Configurations",
            ]
        )
        self._table.horizontalHeaderItem(0).setToolTip(
            "Tick to show a member; untick to hide it. The number labels it in the "
            "scene, the timeline and the list of configurations."
        )
        self._play = QPushButton("Play")
        self._play.setCheckable(True)
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._clock = QLabel("--")
        self._speed = QComboBox()
        for speed in SPEEDS:
            self._speed.addItem(f"x{speed}", speed)
        self._speed.setCurrentIndex(SPEEDS.index(60))
        self._speed.setToolTip("Seconds of flight per second of replay")
        self._trail = QSpinBox()
        self._trail.setRange(0, 60)
        self._trail.setValue(5)
        self._trail.setSuffix(" min trail")
        self._who = QPushButton("Who is who")
        self._who.setToolTip("Which flight, pilot, club and wing each member number is")
        self._who_window = None
        self._now_text = QLabel()
        self._now_text.setWordWrap(True)
        timeline = QHBoxLayout()
        for widget, stretch in (
            (self._play, 0),
            (self._slider, 1),
            (self._clock, 0),
            (self._speed, 0),
            (self._trail, 0),
            (self._who, 0),
        ):
            timeline.addWidget(widget, stretch)
        at = self._layout.indexOf(self._flight_splitter) + 1
        self._layout.insertLayout(at, timeline)
        self._layout.insertWidget(at + 1, self._now_text)
        self._build_timeline()
        self._build_reeb()
        self._build_graph()
        self._timer = QTimer(self)
        self._timer.setInterval(TICK_MS)
        self._timer.setSingleShot(True)  # re-armed after each frame, see _advance
        self._timer.timeout.connect(self._advance)
        self._play.toggled.connect(self._toggle_play)
        self._slider.valueChanged.connect(self._slid)
        self._trail.valueChanged.connect(lambda _: self._redraw())
        self._who.clicked.connect(self._show_who)
        self._set_timeline_enabled(False)

    def _build_timeline(self):
        """The tab of the membership timeline and display options, beside the table."""
        self._colour_by = QComboBox()
        self._colour_by.addItem("Dots by configuration", "configuration")
        self._colour_by.addItem("Dots by pilot", "pilot")
        self._colour_by.addItem("Dots yellow in a configuration", "episode")
        self._track_by = QComboBox()
        self._track_by.addItem("Tracks by pilot", "pilot")
        self._track_by.addItem("Tracks yellow in the episode", "episode")
        self._extent = QComboBox()
        self._extent.addItem("Whole tracks", "whole")
        self._extent.addItem("Tracks up to the instant", "flown")
        self._extent.addItem("Trail only", "trail")
        self._extent.setToolTip(
            "Each track whole, only up to the instant, or only its bright trail"
        )
        self._numbers = QCheckBox("Numbers")
        self._numbers.setChecked(True)
        self._numbers.setToolTip("Write the member number next to each dot")
        self._follow = QCheckBox("Follow the group")
        self._follow.setToolTip("Keep the camera centred on the members in flight")
        self._focus_only = QCheckBox("Only the focused configuration")
        self._focus_only.setToolTip(
            "Hide the members that are not in the configuration picked with a click"
        )
        self._focus_clear = QPushButton("Clear focus")
        self._focus_clear.setToolTip("Follow and show every member again")
        self._focus_text = QLabel(FOCUS_HINT)
        self._strip = pg.PlotWidget(axisItems={"bottom": _ClockAxis("bottom")})
        self._strip.setMenuEnabled(False)
        self._strip.setMouseEnabled(x=True, y=False)
        self._strip.setMinimumHeight(60)
        self._strip.getAxis("left").setLabel("member")
        self._cursor = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen("#ffffff"))
        self._cursor.sigPositionChanged.connect(lambda line: self.seek(line.value()))
        controls = (
            self._colour_by,
            self._track_by,
            self._extent,
            self._numbers,
            self._follow,
            self._focus_only,
        )
        options = FlowLayout()
        for widget in (*controls, self._focus_clear, self._focus_text):
            options.addWidget(widget)
        self._focus_clear.clicked.connect(lambda: self.focus_configuration(None))
        self._reset_focus()
        page = QWidget()
        column = QVBoxLayout(page)
        column.setContentsMargins(0, 0, 0, 0)
        column.addLayout(options)
        column.addWidget(self._strip, 1)
        members = QWidget()
        holder = QVBoxLayout(members)
        holder.setContentsMargins(0, 0, 0, 0)
        self._lower = QTabWidget()
        self._lower.setDocumentMode(True)
        self._flight_splitter.replaceWidget(1, self._lower)
        holder.addWidget(self._table)
        self._lower.addTab(page, "Timeline")
        self._lower.addTab(members, "Members")
        for control in controls:
            signal = getattr(control, "currentIndexChanged", None) or control.toggled
            signal.connect(lambda *_: self._style_changed())

    def _build_reeb(self):
        """The tab of the Reeb graph, live up to the instant or the whole episode."""
        self._reeb_mode = QComboBox()
        self._reeb_mode.addItem("Live: up to the instant", "live")
        self._reeb_mode.addItem("Whole episode", "whole")
        legend = QLabel(
            "Bars: configurations, thicker with more members (number above) · White "
            "lines: a configuration continuing into the next (split, merge, a member "
            "joining or leaving) · Dots: starts and ends"
        )
        legend.setWordWrap(True)
        self._reeb = pg.PlotWidget(axisItems={"bottom": _ClockAxis("bottom")})
        self._reeb.setMenuEnabled(False)
        self._reeb.setMouseEnabled(x=True, y=False)
        self._reeb.setMinimumHeight(60)
        self._reeb.invertY(True)
        self._reeb.hideAxis("left")
        self._reeb_cursor = pg.InfiniteLine(
            angle=90, movable=True, pen=pg.mkPen("#ffffff")
        )
        self._reeb_cursor.setZValue(20)
        self._reeb_cursor.sigPositionChanged.connect(
            lambda line: self.seek(line.value())
        )
        self._reeb_future = QGraphicsRectItem()
        self._reeb_future.setPen(pg.mkPen(None))
        self._reeb_future.setBrush(self._reeb.backgroundBrush())
        self._reeb_future.setZValue(10)
        options = FlowLayout()
        options.addWidget(self._reeb_mode)
        options.addWidget(legend)
        page = QWidget()
        column = QVBoxLayout(page)
        column.setContentsMargins(0, 0, 0, 0)
        column.addLayout(options)
        column.addWidget(self._reeb, 1)
        self._lower.insertTab(1, page, "Reeb graph")
        self._reeb_mode.currentIndexChanged.connect(lambda *_: self._redraw())
        self._colour_by.currentIndexChanged.connect(lambda *_: self._draw_reeb())

    def _build_graph(self):
        """The contact graph G(t) beside the 3D view, under the locator."""
        self._graph = pg.PlotWidget()
        self._graph.setMenuEnabled(False)
        self._graph.setMouseEnabled(x=False, y=False)
        self._graph.hideAxis("bottom")
        self._graph.hideAxis("left")
        self._graph.setAspectLocked(True)
        self._graph.setMinimumSize(140, 40)
        self._graph.setXRange(-1.35, 1.35)
        self._graph.setYRange(-1.35, 1.35)
        self._graph_edges = {
            colour: pg.PlotCurveItem(pen=pg.mkPen(colour, width=2.5))
            for colour in dict.fromkeys((*PALETTE, EPISODE, CONTACT, SOLO))
        }
        self._graph_nodes = pg.ScatterPlotItem(size=16, pen=pg.mkPen("#061725"))
        for item in (*self._graph_edges.values(), self._graph_nodes):
            self._graph.addItem(item)
        self._graph_labels = []
        self._node_xy = np.zeros((0, 2))
        self._scene_layout.removeWidget(self._locator)
        column = QWidget(self)
        box = QVBoxLayout(column)
        box.setContentsMargins(0, 0, 0, 0)
        box.addWidget(self._locator, 1)
        box.addWidget(self._graph, 2)
        self._scene_layout.addWidget(column, 1)

    def _place_graph(self):
        """Members on a circle, in the order they first enter the episode."""
        ep = self._episode
        for label in self._graph_labels:
            self._graph.removeItem(label)
        m = ep.membership[ep.membership.conf_id.isin(ep.configurations.conf_id)]
        first = m.groupby("flight").t_start.min()
        flights = ep.members.flight.to_list()
        order = sorted(range(len(flights)), key=lambda k: first.get(flights[k], np.inf))
        angle = np.empty(len(flights))
        angle[order] = np.pi / 2 - 2 * np.pi * np.arange(len(flights)) / max(
            len(flights), 1
        )
        self._node_xy = np.column_stack([np.cos(angle), np.sin(angle)])
        self._graph_labels = []
        for k, (x, y) in enumerate(self._node_xy):
            label = pg.TextItem(str(k + 1), anchor=(0.5, 0.5), color="#ffffff")
            label.setPos(1.2 * x, 1.2 * y)
            self._graph.addItem(label)
            self._graph_labels.append(label)

    # -- loading -------------------------------------------------------------------

    def set_episode(self, scene, episode):
        """Draw a scene of `group_episodes.load_episode_scene` at the replay start."""
        self._episode = episode
        self._conf_colour = _conf_colours(episode.configurations)
        self.set_scene(scene)
        if self._clicks is not self._view:
            self._view.clicked.connect(self._picked)
            self._clicks = self._view
        n = len(episode.members)
        self._markers = gl.GLScatterPlotItem(pos=np.zeros((n, 3)), size=11, pxMode=True)
        self._markers.setGLOptions("translucent")
        self._markers.setDepthValue(30)
        self._edges = {}
        for colour in dict.fromkeys((*PALETTE, EPISODE, CONTACT, SOLO)):
            self._edges[colour] = self._line(25, width=3.0, color=_rgba(colour))
        self._trails = [self._line(20, width=3.0) for _ in range(n)]
        self._flown = [self._line(10) for _ in range(n)]
        self._numbers_item = _Numbers()
        self._flown_at = None
        for item in (
            self._markers,
            *self._edges.values(),
            *self._trails,
            *self._flown,
            self._numbers_item,
        ):
            self._view.addItem(item)
            self._scene_items.append(item)
        self._scale_changed()
        self._draw_strip()
        self._reeb_data = reeb_graph(episode.configurations)
        self._draw_reeb()
        self._place_graph()
        low, high = episode.window
        self._slider.blockSignals(True)
        self._slider.setRange(0, int((high - low) // STEP_S))
        self._slider.blockSignals(False)
        self._set_timeline_enabled(True)
        if self._who_window is not None:
            self._who_window.set_members(episode.members)
        self.seek(low)

    @staticmethod
    def _line(depth, **kwargs):
        """An empty, hidden outlined line, drawn at `depth`."""
        line = RouteLine(
            pos=np.zeros((2, 3), np.float32),
            outline=1.0,
            mode="lines",
            glOptions="translucent",
            **kwargs,
        )
        line.setDepthValue(depth)
        line.setVisible(False)
        return line

    def _draw_strip(self):
        """One row per member: its local phase, solo time and configurations."""
        ep = self._episode
        self._strip.clear()
        low, high = ep.window
        n = len(ep.members)
        row = {f: n - k for k, f in enumerate(ep.members.flight)}
        m = ep.membership[(ep.membership.t_end > low) & (ep.membership.t_start < high)]
        colours = {"L": LOCAL, "S": SOLO}
        brushes = [
            self._conf_colour.get(int(c), OTHER) if s == "G" else colours[s]
            for s, c in zip(m.state, m.conf_id, strict=True)
        ]
        self._strip.addItem(
            pg.BarGraphItem(
                x0=np.clip(m.t_start.to_numpy(float), low, high),
                x1=np.clip(m.t_end.to_numpy(float), low, high),
                y=m.flight.map(row).to_numpy(float),
                height=0.7,
                brushes=brushes,
                pen=None,
            )
        )
        self._strip.getAxis("left").setTicks(
            [[(n - k, str(k + 1)) for k in range(n)]] if n <= 30 else None
        )
        self._strip.setXRange(low, high, padding=0)
        self._strip.setYRange(0.4, n + 0.6, padding=0)
        self._cursor.setBounds((low, high))
        self._strip.addItem(self._cursor)

    def _draw_reeb(self):
        """The configurations on their lanes, in the colours of the scene."""
        ep = self._episode
        self._reeb.clear()
        if ep is None or self._reeb_data is None:
            return
        edges, links = self._reeb_data
        by_conf = self._colour_by.currentData() == "configuration"
        thick = 0.15 + 0.6 * edges.n / edges.n.max()
        self._reeb.addItem(
            pg.BarGraphItem(
                x0=edges.t0.to_numpy(),
                x1=edges.t1.to_numpy(),
                y=edges.lane.to_numpy(float),
                height=thick.to_numpy(),
                brushes=[
                    self._conf_colour[c] if by_conf else EPISODE for c in edges.conf_id
                ],
                pen=None,
            )
        )
        self._reeb.addItem(
            pg.PlotCurveItem(
                np.repeat(links.t.to_numpy(), 2),
                links[["lane0", "lane1"]].to_numpy().ravel(),
                connect="pairs",
                pen=pg.mkPen("#ffffff", width=1.5),
            )
        )
        self._reeb.addItem(
            pg.ScatterPlotItem(
                x=np.r_[edges.t0, edges.t1],
                y=np.r_[edges.lane, edges.lane].astype(float),
                size=5,
                brush="#ffffff",
                pen=None,
            )
        )
        low, high = ep.window
        long = edges[edges.t1 - edges.t0 >= 0.04 * (high - low)]
        for e, half in zip(long.itertuples(), thick[long.index] / 2, strict=True):
            label = pg.TextItem(str(e.n), anchor=(0.5, 1.0), color="#ffffff")
            label.setPos((e.t0 + e.t1) / 2, e.lane - half)
            self._reeb.addItem(label)
        self._reeb.setXRange(low, high, padding=0)
        self._reeb.setYRange(-1.0, edges.lane.max() + 0.6, padding=0)
        self._reeb_cursor.setBounds((low, high))
        self._reeb.addItem(self._reeb_future)
        self._reeb.addItem(self._reeb_cursor)
        self._redraw()

    def _clear_scene(self):
        """Stop the replay and drop its items with the scene.

        The episode stays until the next one replaces it: set_scene clears the scene
        before it reads the episode.
        """
        if hasattr(self, "_timer"):
            self._timer.stop()
            self._play.setChecked(False)
            self._set_timeline_enabled(False)
            self._reset_focus()
            self._located = {}
            self._now_text.clear()
            self._clock.setText("--")
            self._strip.clear()
            self._reeb.clear()
            for item in self._graph_edges.values():
                item.setData([], [])
            self._graph_nodes.setData([])
            self._graph.setTitle("")
            self._lower.hide()
            if self._who_window is not None:
                self._who_window.table.setRowCount(0)
        self._markers = self._numbers_item = self._reeb_data = None
        self._edges, self._trails, self._flown = {}, [], []
        super()._clear_scene()
        self._summary.setText("")

    def _set_timeline_enabled(self, enabled):
        for widget in (
            self._play,
            self._slider,
            self._speed,
            self._trail,
            self._who,
            self._strip,
            self._reeb,
        ):
            widget.setEnabled(enabled)

    def _show_who(self):
        """Open the Who is who window on the members of the episode."""
        if self._who_window is None:
            self._who_window = _WhoIsWho(self)
        self._who_window.set_members(self._episode.members)
        self._who_window.show()
        self._who_window.raise_()

    # -- the instant ---------------------------------------------------------------

    def seek(self, t):
        """Show every member at UTC second `t`."""
        if self._episode is None:
            return
        low, high = self._episode.window
        self._now = float(np.clip(t, low, high))
        for control, value in (
            (self._slider, int((self._now - low) // STEP_S)),
            (self._cursor, self._now),
            (self._reeb_cursor, self._now),
        ):
            control.blockSignals(True)
            control.setValue(value)
            control.blockSignals(False)
        self._redraw()

    def _slid(self, value):
        self.seek(self._episode.window[0] + value * STEP_S)

    def _toggle_play(self, playing):
        self._play.setText("Pause" if playing else "Play")
        if playing and self._episode is not None:
            if self._now >= self._episode.window[1]:
                self.seek(self._episode.window[0])
            self._timer.start()
        else:
            self._timer.stop()

    def _advance(self):
        """One frame; the next one waits TICK_MS after it, leaving room for clicks.

        The replay holds while a mouse button is down, so that dragging the camera or
        the cursor is not drawn against the replay frames.
        """
        if QGuiApplication.mouseButtons() != Qt.MouseButton.NoButton:
            self._timer.start()
            return
        step = self._speed.currentData() * TICK_MS / 1000
        if self._episode is None or self._now + step > self._episode.window[1]:
            self._play.setChecked(False)
            return
        self.seek(self._now + step)
        if self._play.isChecked():
            self._timer.start()

    def _reset_focus(self):
        """No configuration is focused."""
        self._focus = ()
        self._focus_only.blockSignals(True)
        self._focus_only.setChecked(False)
        self._focus_only.blockSignals(False)
        self._focus_only.setEnabled(False)
        self._focus_clear.setEnabled(False)
        self._focus_text.setText(FOCUS_HINT)

    def focus_configuration(self, conf_id):
        """Follow the members of configuration `conf_id` with the camera; None clears.

        The focus is the set of members of the configuration and stays when it ends.
        With the box ticked, the other members are hidden.
        """
        ep = self._episode
        if conf_id is None or ep is None:
            self._reset_focus()
        else:
            row = ep.configurations[ep.configurations.conf_id == conf_id].iloc[0]
            self._focus = tuple(int(f) for f in row.members)
            number = {f: k + 1 for k, f in enumerate(ep.members.flight)}
            self._focus_only.setEnabled(True)
            self._focus_clear.setEnabled(True)
            self._focus_text.setText(
                "Following members "
                + ", ".join(str(n) for n in sorted(number[f] for f in self._focus))
            )
        self._style_changed()

    def _picked(self, x, y):
        """A click at pixel (x, y) of the scene: focus the configuration under it."""
        ep = self._episode
        if ep is None or not self._located:
            return
        flights = list(self._located)
        points, visible = screen_points(
            self._numbers_item, np.array(list(self._located.values()))
        )
        k = nearest_member(points, visible, x, y)
        if k is None:
            return
        confs = ep.configurations_at(self._now)
        mine = confs[confs.members.map(lambda members: flights[k] in members)]
        if mine.empty:
            number = ep.members.flight.to_list().index(flights[k]) + 1
            self._focus_text.setText(f"Member {number} is in no configuration now")
            return
        self.focus_configuration(int(mine.conf_id.iloc[0]))

    def _shown(self):
        """Members drawn: ticked, selected when the mode asks, and focused if asked."""
        selected = {item.row() for item in self._table.selectedItems()}
        flights = self._episode.members.flight.to_list()
        only = set(self._focus) if self._focus_only.isChecked() else None
        return [
            self._flight_checked(k)
            and (self._mode.currentData() == "all" or k in selected)
            and (only is None or flights[k] in only)
            for k in range(len(flights))
        ]

    def _colours(self, t):
        """Colour of every member at `t`, and whether it is in a configuration."""
        ep = self._episode
        conf = {}
        for c in ep.configurations_at(t).itertuples():
            for f in c.members:
                conf[int(f)] = self._conf_colour[int(c.conf_id)]
        other = set(self._elsewhere(t))
        mode = self._colour_by.currentData()
        out = []
        for k, f in enumerate(ep.members.flight):
            if mode == "pilot":
                colour = PILOTS[k % len(PILOTS)]
            elif f in other:
                colour = OTHER
            elif mode == "episode":
                colour = EPISODE if f in conf else SOLO
            else:
                colour = conf.get(f, SOLO)
            out.append((colour, f in conf or f in other))
        return out

    def _elsewhere(self, t):
        """Members in a configuration of another episode at `t`."""
        m = self._episode.membership
        at = m[(m.t_start <= t) & (t < m.t_end) & (m.state == "G")]
        return at.flight[at.episode_id != self._episode.row.episode_id].to_list()

    def _track_colour(self, k, grouped):
        """Colour of the track and trail of member `k`."""
        if self._track_by.currentData() == "pilot":
            return PILOTS[k % len(PILOTS)]
        return EPISODE if grouped else TRACK

    def _contacts(self, t, index, colours):
        """Pairs in contact at `t` among `index` (flight -> k), by line colour.

        A line takes the colour of the dots of its configuration, or white when the dots
        show pilots.
        """
        by_pilot = self._colour_by.currentData() == "pilot"
        groups = {}
        for a, b in self._episode.edges_at(t):
            if a in index and b in index:
                colour = CONTACT if by_pilot else colours[index[a]][0]
                groups.setdefault(colour, []).append((index[a], index[b]))
        return groups

    def _redraw(self):
        """Markers, numbers, contacts, trails, flown tracks, camera and text."""
        ep, t = self._episode, self._now
        if ep is None or self._markers is None:
            return
        flights = ep.members.flight.to_list()
        shown = self._shown()
        colours = self._colours(t)
        up_to_now = self._extent.currentData() == "flown"
        rebuild = up_to_now and (
            self._flown_at is None or not 0 <= t - self._flown_at < FLOWN_STEP_S
        )
        if rebuild:
            self._flown_at = t
        recent = 60 * self._trail.value()
        if up_to_now:
            recent = max(recent, t - self._flown_at)
        positions, dots, sizes, where = [], [], [], {}
        for k, (f, (colour, grouped)) in enumerate(zip(flights, colours, strict=True)):
            p = ep.tracks[f].at(t)
            visible = p is not None and shown[k]
            positions.append(p if p is not None else np.zeros(3))
            dots.append(_rgba(colour, 1.0 if visible else 0.0))
            sizes.append(13 if grouped else 7)
            if visible:
                where[f] = p
            trail = _segments(ep.tracks[f].trail(t, recent))
            self._trails[k].setVisible(visible and trail is not None)
            if visible and trail is not None:
                self._trails[k].setData(
                    pos=trail, color=_rgba(self._track_colour(k, grouped))
                )
            if rebuild:
                flown = _segments(ep.tracks[f].trail(t, t - ep.window[0]))
                self._flown[k].setVisible(shown[k] and flown is not None)
                if flown is not None:
                    self._flown[k].setData(
                        pos=flown,
                        color=_rgba(self._track_colour(k, True), 0.8),
                        width=self._line_width.value(),
                    )
        self._located = where
        self._markers.setData(
            pos=np.array(positions), color=np.array(dots), size=np.array(sizes)
        )
        numbered = [k for k, f in enumerate(flights) if f in where]
        if not self._numbers.isChecked():
            numbered = []
        self._numbers_item.set_points(
            [where[flights[k]] for k in numbered], [str(k + 1) for k in numbered]
        )
        index = {f: k for k, f in enumerate(flights) if f in where}
        groups = self._contacts(t, index, colours)
        for colour, line in self._edges.items():
            ends = [
                (where[flights[i]], where[flights[j]])
                for i, j in groups.get(colour, [])
            ]
            line.setVisible(bool(ends))
            if ends:
                line.setData(pos=np.array(ends, np.float32).reshape(-1, 3))
        followed = where
        if self._focus:
            followed = {f: p for f, p in where.items() if f in self._focus}
        elif not self._follow.isChecked():
            followed = {}
        if followed and self._view is not None:
            x, y, z = np.mean(list(followed.values()), axis=0)
            self._view.opts["center"] = QVector3D(x, y, z * self._vertical.value())
            self._view.update()
        self._clock.setText(_flight_datetime(t).split(" ")[1])
        self._now_text.setText(self._describe(t, flights))
        self._draw_graph(t, flights, shown, colours)
        live = self._reeb_mode.currentData() == "live"
        self._reeb_future.setVisible(live)
        if live and self._reeb_data is not None:
            high, lanes = ep.window[1], self._reeb_data[0].lane.max()
            self._reeb_future.setRect(QRectF(t, -1, high - t, lanes + 2))

    def _draw_graph(self, t, flights, shown, colours):
        """G(t) in the colours of the scene; members not in flight are faded."""
        ep = self._episode
        if len(self._node_xy) != len(flights):
            return
        groups = self._contacts(t, {f: k for k, f in enumerate(flights)}, colours)
        for colour, item in self._graph_edges.items():
            ends = self._node_xy[np.array(groups.get(colour, []), int).ravel()]
            item.setData(ends[:, 0], ends[:, 1], connect="pairs")
        brushes, sizes = [], []
        for k, (f, (colour, grouped)) in enumerate(zip(flights, colours, strict=True)):
            flying = ep.tracks[f].at(t) is not None
            fill = QColor(colour if flying else LOCAL)
            fill.setAlpha(255 if flying and shown[k] else 70)
            brushes.append(pg.mkBrush(fill))
            sizes.append(16 if grouped else 11)
        self._graph_nodes.setData(pos=self._node_xy, brush=brushes, size=sizes)
        count = sum(len(pairs) for pairs in groups.values())
        plural = "s" if count != 1 else ""
        self._graph.setTitle(f"Contact graph: {count} pair{plural} in contact")

    def _describe(self, t, flights):
        """The configurations of this episode and the episodes of the day at `t`."""
        ep = self._episode
        number = {f: k + 1 for k, f in enumerate(flights)}
        confs = ep.configurations_at(t)
        parts = []
        for c in confs.itertuples():
            members = ", ".join(
                str(number[int(f)])
                for f in sorted(c.members, key=lambda f: number[int(f)])
            )
            since = _flight_datetime(c.t_start).split(" ")[1]
            parts.append(
                f'<span style="color:{self._conf_colour[int(c.conf_id)]}">●</span> '
                f"[{members}] since {since}"
            )
        flying = sum(ep.state_of(f, t) in ("S", "G") for f in flights)
        plural = "s" if len(confs) != 1 else ""
        text = (
            f"<b>{len(confs)} configuration{plural}</b> of this episode · "
            f"{flying} of {len(flights)} members flying after departure"
        )
        if parts:
            text += ": " + " · ".join(parts)
        away = sorted(number[f] for f in self._elsewhere(t))
        if away:
            text += (
                "<br><b>In another episode:</b> "
                + ", ".join(map(str, away))
                + " (white: a configuration not linked to this one)"
            )
        others = ep.episodes_at(t)
        others = others[others.episode_id != ep.row.episode_id]
        text += f"<br><b>Other episodes in progress that day:</b> {len(others)}"
        if len(others):
            text += (
                " (" + ", ".join(f"{n} flights" for n in others.n_flights.head(8)) + ")"
            )
        return text

    # -- table and styles ----------------------------------------------------------

    def _populate_table(self, scene):
        """One row per member, its number in the colour of the pilot."""
        ep = self._episode
        confs = ep.configurations
        mine = ep.membership[ep.membership.conf_id.isin(confs.conf_id)]
        in_episode = (mine.t_end - mine.t_start).groupby(mine.flight).sum()
        counts = confs.explode("members").groupby("members").size()
        self._table.blockSignals(True)
        self._table.setRowCount(len(ep.members))
        for i, row in enumerate(ep.members.itertuples(index=False)):
            values = [
                str(i + 1),
                row.discipline,
                str(row.flight_id),
                _flight_datetime(row.takeoff),
                _flight_datetime(row.landing),
                f"{in_episode.get(row.flight, 0) / 60:.0f}",
                str(int(counts.get(row.flight, 0))),
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col == 0:
                    self._make_flight_checkable(item)
                    item.setBackground(QColor(PILOTS[i % len(PILOTS)]))
                    item.setForeground(QColor("#061725"))
                self._table.setItem(i, col, item)
        self._table.blockSignals(False)
        self._table.show()
        self._lower.show()  # replaceWidget gave it the hidden state of the table

    def _set_scene_summary(self, scene):
        """Identify the episode."""
        row = self._episode.row
        distance = (
            f" · {row.distance_m / 1000:.0f} km" if np.isfinite(row.distance_m) else ""
        )
        end = _flight_datetime(row.t_end).split(" ")[1]
        self._summary.setText(
            f"Episode of {row.date}: {_flight_datetime(row.t_start)} to {end} · "
            f"{int(row.n_flights)} flights, largest configuration {int(row.n_max)}, "
            f"{int(row.n_config)} configurations{distance}"
        )

    def _flight_source(self, scene):
        """Say which flights and which part of them are drawn."""
        return (
            "Members: every flight in a configuration of the episode, drawn from "
            "10 minutes before the episode to 10 minutes after it; gaps not connected. "
        )

    def _style_changed(self, *_):
        """Highlight selected members without moving the camera; redraw the instant."""
        if self._scene is None or self._episode is None:
            return
        selected = {item.row() for item in self._table.selectedItems()}
        shown = self._shown()
        flights = self._episode.members.flight.to_list()
        whole = self._extent.currentData() == "whole"
        for i, lines in enumerate(self._lines):
            inside = self._episode.inside.get(flights[i], [])
            for j, line in enumerate(lines):
                in_episode = j < len(inside) and inside[j]
                if i in selected:
                    colour, alpha = SELECTED, 1.0
                else:
                    colour = self._track_colour(i, in_episode)
                    alpha = 0.9 if in_episode else 0.35
                line.setVisible(shown[i] and whole)
                line.outline = min(1, 0.4 * self._line_width.value())
                line.setData(
                    color=_rgba(colour, alpha),
                    width=self._line_width.value() * (1.75 if i in selected else 1),
                )
                line.setDepthValue(15 if i in selected else 5)
        self._update_visible_count(shown)
        if self._surface is not None:
            self._surface.set_terrain_visible(self._terrain.isChecked())
        if self._annotations is not None:
            self._annotations.set_flights(shown, selected)
        self._flown_at = None
        for line in self._flown:
            line.setVisible(False)
        self._redraw()

    def shutdown(self):
        """Stop the replay before the shared OpenGL view goes."""
        self._timer.stop()
        if self._who_window is not None:
            self._who_window.close()
        super().shutdown()
