"""Choose simultaneous-departure candidates from a complete directed route cohort."""

from PyQt6.QtCore import QTime, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QLabel,
    QPushButton,
    QSpinBox,
    QTimeEdit,
    QVBoxLayout,
    QWidget,
)

from ..route_times import DepartureWindow, filter_departures, local_departures
from .flow_layout import FlowLayout, labeled_control


class DepartureFilter(QWidget):
    """A civil-day selector and editable minute window, with an honest live count."""

    changed = pyqtSignal()
    apply_requested = pyqtSignal()

    def __init__(self, parent=None):
        """Keep the optional filter inactive until a pair's full dates are loaded."""
        super().__init__(parent)
        self._cohort = None
        self._busy = False
        self._active = QCheckBox("Departure window")
        self._active.setToolTip(
            "Load a pair, then choose a day and departure interval in Europe/Paris. "
            "Departure means the first retained fix, as in the flight table."
        )
        self._day = QComboBox()
        self._day.setMinimumContentsLength(20)
        self._day.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self._start = QTimeEdit(QTime(12, 0))
        self._start.setDisplayFormat("HH:mm")
        self._start.setKeyboardTracking(False)
        self._minutes = QSpinBox()
        self._minutes.setRange(1, 1440)
        self._minutes.setValue(30)
        self._minutes.setSuffix(" min")
        self._minutes.setKeyboardTracking(False)
        self._minutes.setToolTip(
            "Window length in local clock minutes; ends on the same day"
        )
        self._apply = QPushButton("Apply")
        self._apply.setToolTip(
            "Load all matching departures, then apply the 300-flight limit"
        )
        self._note = QLabel()
        self._note.setWordWrap(True)
        self._fields = [
            labeled_control("Day (Paris)", self._day),
            labeled_control("From", self._start),
            labeled_control("Window", self._minutes),
        ]
        row = FlowLayout()
        for widget in (self._active, *self._fields, self._apply):
            row.addWidget(widget)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(row)
        layout.addWidget(self._note)
        self._active.toggled.connect(self._changed)
        self._day.currentIndexChanged.connect(self._day_changed)
        self._start.timeChanged.connect(self._changed)
        self._minutes.valueChanged.connect(self._changed)
        self._apply.clicked.connect(self.apply_requested.emit)
        self.reset()

    def reset(self):
        """A different route or discipline starts with all of its departures."""
        self.blockSignals(True)
        self._cohort = None
        self._active.setChecked(False)
        self._day.clear()
        self._start.setTime(QTime(12, 0))
        self._minutes.setValue(30)
        self._update()
        self.blockSignals(False)

    def set_cohort(self, cohort):
        """Offer dates from every candidate, including flights outside the sample."""
        if cohort is None:
            self.reset()
            return
        self.blockSignals(True)
        first = self._cohort is None
        selected_day = self._day.currentData()
        self._cohort = cohort
        local = local_departures(cohort)
        counts = local.dt.date.value_counts().sort_index()
        self._day.blockSignals(True)
        self._day.clear()
        for day, count in counts.items():
            self._day.addItem(f"{day:%d/%m/%Y} · {count} flights", day)
        if len(counts):
            day = selected_day if selected_day in counts.index else counts.idxmax()
            # Qt's findData does not compare Python date objects by value.
            # Rebuilding the menu creates new objects, even for the same dates.
            self._day.setCurrentIndex(int(counts.index.get_loc(day)))
            if first or day != selected_day:
                self._suggest_start()
        self._day.blockSignals(False)
        self._update()
        self.blockSignals(False)

    def selection(self):
        """Capture an immutable request independently of later GUI edits."""
        if not self._active.isChecked() or self._day.currentData() is None:
            return None
        return DepartureWindow(
            self._day.currentData(),
            self._start.time().toPyTime(),
            self._minutes.value(),
        )

    def set_busy(self, busy):
        """Lock source selection while its worker is running."""
        self._busy = busy
        self._update()

    def has_matches(self):
        """Whether the preview contains departures that can actually be loaded."""
        return self._matches > 0

    def _suggest_start(self):
        """Start at a real departure, including when the window is only one minute."""
        local = local_departures(self._cohort)
        earliest = local.loc[local.dt.date.eq(self._day.currentData())].min()
        self._start.blockSignals(True)
        self._start.setTime(QTime(earliest.hour, earliest.minute))
        self._start.blockSignals(False)

    def _day_changed(self, *_):
        """Keep a useful time, or move to the first departure on the new day."""
        if self._cohort is not None and self._day.currentData() is not None:
            window = DepartureWindow(
                self._day.currentData(),
                self._start.time().toPyTime(),
                self._minutes.value(),
            )
            if filter_departures(self._cohort, window).empty:
                self._suggest_start()
        self._changed()

    def _changed(self, *_):
        """Update the draft preview without applying it to displayed trajectories."""
        self._update()
        self.changed.emit()

    def _update(self):
        """Keep boundaries, availability and full-cohort counts visible."""
        maximum = 1440 - self._start.time().hour() * 60 - self._start.time().minute()
        self._minutes.blockSignals(True)
        self._minutes.setMaximum(maximum)
        self._minutes.blockSignals(False)
        active = self._active.isChecked()
        ready = self._cohort is not None
        for field in self._fields:
            field.setVisible(active)
            field.setEnabled(not self._busy)
        self._active.setEnabled(not self._busy and self._day.count() > 0)
        self._apply.setVisible(ready)
        self._note.setVisible(ready)
        matches = 0
        if ready:
            window = self.selection()
            matches = len(filter_departures(self._cohort, window))
            unknown = int(self._cohort.departure_utc.isna().sum())
            text = (
                f"Preview · {window.label} · {matches} / {len(self._cohort)} departures"
                if window
                else f"Preview · All departures · {len(self._cohort)} flights"
            )
            if not matches:
                text += " · No departures: choose another time or day"
            if unknown:
                text += f" · unavailable dates: {unknown}"
            self._note.setText(text)
        else:
            self._note.clear()
        self._matches = matches
        self._apply.setEnabled(not self._busy and matches > 0)
