"""Controls for how the trajectory is drawn: frame, axes, dimensionality, DMS, PDF.

Column names, not fixed axis pairs: the combo boxes offer whichever columns the
selected frame actually has (:mod:`xc_viewer.data`'s contract --
``lon``/``lat``/
``alt`` for the geographic frame, ``E``/``N``/``z`` for the local ENU one), so the user
picks any two (2D) or three (3D) of them freely, in either order.

Controls wrap within each group as the window narrows, keeping the plot resizable
without hiding the axes, 3D view or visibility/export settings.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .flow_layout import FlowLayout, labeled_control
from .help import help_buttons

_GEOGRAPHIC_AXES = [("lon", "Longitude"), ("lat", "Latitude"), ("alt", "Altitude")]
_ENU_AXES = [("E", "East"), ("N", "North"), ("z", "Altitude (z)")]

# mplot3d's own defaults (Axes3D.view_init with no arguments) -- "Reset view" restores
# exactly what a freshly created 3D plot would show.
_DEFAULT_AZIM_DEG = -60
_DEFAULT_ELEV_DEG = 30
_DEFAULT_ZOOM_PERCENT = 100


class PlotControls(QWidget):
    """Emits ``changed`` whenever a setting affecting the drawing changes.

    Also emits ``view_changed`` for the 3D rotation/zoom sliders specifically: unlike
    ``changed`` (frame, axes, DMS, raw/cleaned visibility), adjusting the viewing
    angle or zoom never changes *what* data is drawn, only how the camera looks at
    it -- so the caller can react to it without re-plotting the trajectory, just
    re-orienting the existing one (cheaper, and it keeps the view from jumping while
    dragging a slider).
    """

    changed = pyqtSignal()
    view_changed = pyqtSignal()
    reset_view_requested = pyqtSignal()
    save_pdf_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the frame/axis/DMS/visibility controls and the Save PDF button."""
        super().__init__(parent)

        self._frame_combo = QComboBox()
        self._frame_combo.addItem("Geographic (lat/lon)", "geographic")
        self._frame_combo.addItem("Local (ENU, metres)", "enu")

        self._x_combo = QComboBox()
        self._y_combo = QComboBox()
        self._z_combo = QComboBox()
        self._chk_3d = QCheckBox("3D")

        axes_row = FlowLayout()
        axes_row.addWidget(labeled_control("Frame", self._frame_combo))
        axes_row.addWidget(labeled_control("X", self._x_combo))
        axes_row.addWidget(labeled_control("Y", self._y_combo))
        axes_row.addWidget(self._chk_3d)
        axes_row.addWidget(labeled_control("Z", self._z_combo))
        axes_box = QGroupBox("Axes")
        axes_box.setLayout(axes_row)

        self._slider_azim, self._lbl_azim, azim_unit = self._make_slider(
            -180, 180, _DEFAULT_AZIM_DEG, "°"
        )
        self._slider_elev, self._lbl_elev, elev_unit = self._make_slider(
            -90, 90, _DEFAULT_ELEV_DEG, "°"
        )
        self._slider_zoom, self._lbl_zoom, zoom_unit = self._make_slider(
            20, 300, _DEFAULT_ZOOM_PERCENT, "%"
        )
        self._btn_reset_view = QPushButton("Reset view")

        view3d_row = FlowLayout()
        view3d_row.addWidget(labeled_control("Azimuth", azim_unit))
        view3d_row.addWidget(labeled_control("Elevation", elev_unit))
        view3d_row.addWidget(labeled_control("Zoom", zoom_unit))
        view3d_row.addWidget(self._btn_reset_view)
        # A group box is never narrower than its title: keep it short.
        self._box_3d_view = QGroupBox("3D view")
        self._box_3d_view.setToolTip("Drag the plot, or use the sliders.")
        self._box_3d_view.setLayout(view3d_row)

        self._chk_dms = QCheckBox("Degrees-minutes-seconds")
        self._chk_raw = QCheckBox("Show raw")
        self._chk_cleaned = QCheckBox("Show cleaned")
        self._chk_raw.setChecked(True)
        self._chk_cleaned.setChecked(True)
        self._color_combo = QComboBox()
        # Short items: the combo is as wide as its longest one.
        self._color_combo.addItem("Flight phase", "phase")
        self._color_combo.addItem("Cleaning segment", "segment")
        self._color_combo.addItem("Discipline", "single")
        self._color_combo.setToolTip(
            "Colour of the cleaned track: flight phase, preprocessing segment, or "
            "one colour per discipline."
        )
        self._method_combo = QComboBox()
        self._method_combo.addItem("Vilpellet", "vilpellet")
        self._method_combo.addItem("Gaussian", "gaussian")
        self._method_combo.addItem("Both", "both")
        self._method_combo.setToolTip(
            "Which segmentation labels the phases. Both: Vilpellet as a wide band, "
            "Gaussian as a thin line over it."
        )
        self._phase_combo = QComboBox()
        self._phase_combo.addItem("All", None)
        self._phase_combo.addItem("Climb", "climb")
        self._phase_combo.addItem("Search", "search")
        self._phase_combo.addItem("Transition", "transition")
        self._phase_combo.setToolTip("Show every phase, or only one of them.")
        self._btn_save_pdf = QPushButton("Save PDF…")
        self._info, self._howto = help_buttons("trajectory", self)

        display_row = FlowLayout()
        display_row.addWidget(self._chk_dms)
        display_row.addWidget(self._chk_raw)
        display_row.addWidget(self._chk_cleaned)
        display_row.addWidget(labeled_control("Cleaned colour", self._color_combo))
        display_row.addWidget(labeled_control("Segmentation", self._method_combo))
        display_row.addWidget(labeled_control("Phases", self._phase_combo))
        display_row.addWidget(self._btn_save_pdf)
        display_row.addWidget(self._info)
        display_row.addWidget(self._howto)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(axes_box)
        layout.addWidget(self._box_3d_view)
        layout.addLayout(display_row)

        self._populate_axes()
        self._on_3d_toggled(False)
        self._sync_phase_controls()

        self._frame_combo.currentIndexChanged.connect(self._on_frame_changed)
        self._chk_3d.toggled.connect(self._on_3d_toggled)
        self._chk_3d.toggled.connect(self.changed.emit)
        for combo in (self._x_combo, self._y_combo, self._z_combo):
            combo.currentIndexChanged.connect(self.changed.emit)
        self._chk_dms.toggled.connect(self.changed.emit)
        self._chk_raw.toggled.connect(self.changed.emit)
        self._chk_cleaned.toggled.connect(self.changed.emit)
        self._color_combo.currentIndexChanged.connect(self._sync_phase_controls)
        self._color_combo.currentIndexChanged.connect(self.changed.emit)
        self._method_combo.currentIndexChanged.connect(self.changed.emit)
        self._phase_combo.currentIndexChanged.connect(self.changed.emit)
        self._btn_save_pdf.clicked.connect(self.save_pdf_requested.emit)

        for slider, label, suffix in (
            (self._slider_azim, self._lbl_azim, "°"),
            (self._slider_elev, self._lbl_elev, "°"),
            (self._slider_zoom, self._lbl_zoom, "%"),
        ):
            slider.valueChanged.connect(
                lambda value, lbl=label, sfx=suffix: lbl.setText(f"{value}{sfx}")
            )
            slider.valueChanged.connect(self.view_changed.emit)
        self._btn_reset_view.clicked.connect(self._reset_view)

    # -- state exposed to the caller ----------------------------------------------
    @property
    def frame_kind(self) -> str:
        """``"geographic"`` or ``"enu"``."""
        return self._frame_combo.currentData()

    @property
    def is_3d(self) -> bool:
        """Whether a third (Z) axis should be plotted."""
        return self._chk_3d.isChecked()

    @property
    def x_column(self) -> str:
        """Column name to plot on the X axis, for the current frame.

        Named ``x_column``, not ``x``: ``QWidget`` already has an ``x()`` meaning the
        widget's screen position, and shadowing it would be a trap for anyone reading
        this class from its Qt side rather than its plotting side.
        """
        return self._x_combo.currentData()

    @property
    def y_column(self) -> str:
        """Column name to plot on the Y axis (see ``x_column``)."""
        return self._y_combo.currentData()

    @property
    def z_column(self) -> str:
        """Column name to plot on the Z axis (see ``x_column``)."""
        return self._z_combo.currentData()

    @property
    def dms(self) -> bool:
        """Whether lat/lon ticks should read as degrees-minutes-seconds."""
        return self._chk_dms.isChecked() and self.frame_kind == "geographic"

    @property
    def show_raw(self) -> bool:
        """Whether the raw trajectory should be drawn."""
        return self._chk_raw.isChecked()

    @property
    def show_cleaned(self) -> bool:
        """Whether the cleaned trajectory should be drawn."""
        return self._chk_cleaned.isChecked()

    @property
    def color_mode(self) -> str:
        """``phase``, ``segment``, or ``single`` for the cleaned trajectory."""
        return self._color_combo.currentData()

    @property
    def segmentation(self) -> str:
        """``vilpellet``, ``gaussian`` or ``both``."""
        return self._method_combo.currentData()

    @property
    def phase_filter(self) -> str | None:
        """The only phase to show, or ``None`` for all phases.

        Always ``None`` unless the cleaned track is coloured by phase.
        """
        if self.color_mode != "phase":
            return None
        return self._phase_combo.currentData()

    @property
    def azim_deg(self) -> int:
        """3D camera azimuth, in degrees (only meaningful when ``is_3d``)."""
        return self._slider_azim.value()

    @property
    def elev_deg(self) -> int:
        """3D camera elevation, in degrees (only meaningful when ``is_3d``)."""
        return self._slider_elev.value()

    @property
    def zoom_percent(self) -> int:
        """3D zoom level, as a percentage of the trajectory's natural extent."""
        return self._slider_zoom.value()

    # -- internals ------------------------------------------------------------------
    def _make_slider(
        self, minimum: int, maximum: int, default: int, suffix: str
    ) -> tuple[QSlider, QLabel, QWidget]:
        """A ``(slider, value-label, row-widget)`` triple sharing one horizontal row."""
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(default)
        slider.setFixedWidth(110)
        label = QLabel(f"{default}{suffix}")
        label.setMinimumWidth(36)
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(slider)
        row_layout.addWidget(label)
        return slider, label, row

    def _reset_view(self) -> None:
        self._slider_azim.setValue(_DEFAULT_AZIM_DEG)
        self._slider_elev.setValue(_DEFAULT_ELEV_DEG)
        self._slider_zoom.setValue(_DEFAULT_ZOOM_PERCENT)
        self.reset_view_requested.emit()

    def _populate_axes(self) -> None:
        axes = _GEOGRAPHIC_AXES if self.frame_kind == "geographic" else _ENU_AXES
        for combo, default_index in (
            (self._x_combo, 0),
            (self._y_combo, 1),
            (self._z_combo, 2),
        ):
            combo.blockSignals(True)
            combo.clear()
            for value, label in axes:
                combo.addItem(label, value)
            combo.setCurrentIndex(default_index)
            combo.blockSignals(False)
        self._chk_dms.setEnabled(self.frame_kind == "geographic")

    def _on_frame_changed(self) -> None:
        self._populate_axes()
        self.changed.emit()

    def _on_3d_toggled(self, checked: bool) -> None:
        self._z_combo.setEnabled(checked)
        self._box_3d_view.setVisible(checked)

    def _sync_phase_controls(self) -> None:
        """Enable the phase-only inputs exactly while phase colouring is selected."""
        for combo in (self._method_combo, self._phase_combo):
            combo.setEnabled(self.color_mode == "phase")
