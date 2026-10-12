"""The shared route renderer with launch-order styling and group-flight metrics."""

import numpy as np
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QTableWidgetItem

from ..thermal_geometry import unproject
from .route_comparison import RouteComparison, _flight_datetime


class GroupScene(RouteComparison):
    """Reuse terrain, heat draping, navigation and full clean geometry from Routes."""

    def __init__(self, parent=None):
        """Expose only map controls; the parent owns cell and launch-group selection."""
        super().__init__(parent, scene_only=True)
        self._mode.clear()
        self._mode.addItem("All group flights", "all")
        self._mode.addItem("Selected rows only", "selected")
        self._line_width.setValue(1.2)
        self._line_width.setToolTip(
            "Stroke width in screen pixels; selected flights are 75% wider"
        )
        self._legend.setText(
            "Departure order: blue → cyan → lime · White: selected flights · "
            "○ First clean fix · ■ Last clean fix"
        )
        self._placeholder.setText("Select a departure cell and a launch group.")
        self._summary.setText("Same departure cell, independent destinations.")
        self._density.resolution.setToolTip(
            "Thermal pixels are independent of the 5 or 10 km departure grid"
        )
        self._table.setColumnCount(11)
        self._table.setHorizontalHeaderLabels(
            [
                "Show / order",
                "Discipline",
                "Flight ID",
                "Departure (Paris)",
                "Arrival (Paris)",
                "Delay (min)",
                "Elapsed",
                "Path (km)",
                "Net (km)",
                "Gaps (min)",
                "Segments",
            ]
        )
        self._table.horizontalHeaderItem(0).setToolTip(
            "Tick to show a flight; untick to hide it. The number is its launch order."
        )
        self._table.horizontalHeaderItem(7).setToolTip(
            "Horizontal length of supported cleaned edges in Lambert-93; gaps excluded."
        )
        self._table.horizontalHeaderItem(8).setToolTip(
            "Horizontal separation of first and last cleaned fixes"
        )
        self._table.horizontalHeaderItem(9).setToolTip(
            "Elapsed time without a supported cleaned edge"
        )
        self._colours = []

    def _clear_scene(self):
        """Clear the previous group immediately when the selected definition changes."""
        super()._clear_scene()
        self._summary.setText("Select a departure cell and a launch group.")

    def _populate_table(self, scene):
        """Keep every member in departure order, using the same colours as its path."""
        self._colours = [
            QColor.fromHsvF(float(h), 0.88, 1)
            for h in np.linspace(0.61, 0.27, len(scene.selected))
        ]
        self._table.blockSignals(True)
        self._table.setRowCount(len(scene.selected))
        for i, row in enumerate(scene.selected.itertuples(index=False)):
            seconds = round(row.duration_s)
            values = [
                str(row.rank),
                row.discipline,
                row.flight_id,
                _flight_datetime(row.departure_utc),
                _flight_datetime(row.arrival_utc),
                f"{row.departure_delay_s / 60:.1f}",
                f"{seconds // 3600}:{seconds % 3600 // 60:02}:{seconds % 60:02}",
                f"{row.path_km:.1f}",
                f"{row.net_km:.1f}",
                f"{row.gap_s / 60:.1f}",
                str(row.clean_segments),
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col == 0:
                    self._make_flight_checkable(item)
                    item.setBackground(self._colours[i])
                    item.setForeground(QColor("#061725"))
                self._table.setItem(i, col, item)
        self._table.blockSignals(False)
        self._table.show()

    def _set_scene_summary(self, scene):
        """Identify the displayed cohort independently of pending selectors."""
        ix, iy = scene.pair
        lon, lat = unproject((ix + 0.5) * scene.cell_m, (iy + 0.5) * scene.cell_m)
        first, last = scene.selected.departure_utc.iloc[[0, -1]]
        self._summary.setText(
            f"{len(scene.selected)} loaded flights · "
            f"{scene.cell_m / 1000:g} x {scene.cell_m / 1000:g} km departure cell "
            f"[{ix}, {iy}] · {float(lat):.3f}° N, {float(lon):.3f}° E\n"
            f"First departure: {_flight_datetime(first)} · "
            f"Last departure: {_flight_datetime(last)} "
            f"· Actual launch span: {(last - first) / 60:.1f} min"
        )

    def _flight_source(self, scene):
        """State timing and distance semantics while preserving shared IGN credits."""
        return (
            "Departure/arrival = first/last retained cleaned fix. "
            "Elapsed time includes gaps. "
            "Path = horizontal Lambert-93 length of supported cleaned edges, "
            "excluding gaps. "
            "All group members are loaded; checkboxes control visibility. "
            "Path length and destination are unrestricted. "
        )

    def _style_changed(self, *_):
        """Highlight selected members without changing geometry or camera position."""
        if self._scene is None or len(self._colours) != len(self._lines):
            return
        selected = {item.row() for item in self._table.selectedItems()}
        visible = []
        for i, lines in enumerate(self._lines):
            show = self._flight_checked(i) and (
                self._mode.currentData() == "all" or i in selected
            )
            visible.append(show)
            colour = QColor("#ffffff") if i in selected else self._colours[i]
            for line in lines:
                line.setVisible(show)
                line.outline = min(1, 0.4 * self._line_width.value())
                line.setData(
                    color=colour.getRgbF(),
                    width=self._line_width.value() * (1.75 if i in selected else 1),
                )
                line.setDepthValue(15 if i in selected else 5)
        self._update_visible_count(visible)
        if self._surface is not None:
            self._surface.set_terrain_visible(self._terrain.isChecked())
        if self._annotations is not None:
            self._annotations.set_flights(visible, selected)
