"""Picking a flight: browse the disk directly, or filter the FFVL catalog.

Two independent ways in, one signal out: whichever path a flight is found through,
:class:`FlightPicker` reduces it to the same ``(path, discipline, flight_id)`` triple
the rest of the viewer needs.

Every metadata filter is a searchable dropdown (:func:`_make_searchable_combo`),
populated from :func:`xc_viewer.catalog_index.distinct_values` for the currently
selected discipline: the user picks (or types, with live filtering) a value the
archive actually contains, rather than having to guess a department code or the exact
spelling of a site name.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pandas as pd
from PyQt6.QtCore import (
    QAbstractTableModel,
    QLocale,
    QModelIndex,
    Qt,
    QTimer,
    pyqtSignal,
)
from PyQt6.QtGui import QKeySequence
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QCompleter,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QScrollArea,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from .. import catalog_index, data, datafolder, geography
from ..core.disciplines import DISCIPLINES, Discipline
from ..core.naming import parse_igc_filename
from .english import ENGLISH

_RESULT_COLUMNS = [
    "flight_id",
    "date",
    "dept",
    "takeoff",
    "flight_type",
    "wing_class",
    "kept",
]
_RESULT_HEADERS = [
    "Flight ID",
    "Date",
    "Dept.",
    "Takeoff",
    "Type",
    "Wing class",
    "Pipeline status",
]
# A folder can hold the entire archive (186,052 .igc files for paragliders per
# docs/guide/data-on-disk.md): walking all of it just to fill a picker list would
# make "Browse folder..." itself the slow path. Capped at the walk, not after it.
_MAX_FOLDER_SCAN = 2000

# (catalog column, field label, distinct_values() populates it) -- season_year is
# included here too even though its values are years, not text: the same searchable
# widget and the same "type or pick" interaction work just as well for it.
_FILTER_FIELDS = [
    ("dept", "Department"),
    ("season_year", "Season year"),
    ("flight_type", "Flight type"),
    ("wing_class", "Wing class"),
    ("wing", "Wing model"),
    ("takeoff", "Takeoff site"),
    ("landing", "Landing site"),
    ("club", "Club"),
    ("pilot", "Pilot"),
]


class FlightResultsModel(QAbstractTableModel):
    """Expose every matching flight without allocating a widget item per cell."""

    def __init__(self, parent=None):
        """Start with an empty result table."""
        super().__init__(parent)
        self.rows = pd.DataFrame(columns=_RESULT_COLUMNS)

    def set_rows(self, rows: pd.DataFrame) -> None:
        """Replace the search results atomically."""
        self.beginResetModel()
        self.rows = rows.reset_index(drop=True)
        self.endResetModel()

    def rowCount(self, parent=None):  # noqa: N802
        """Make all matches addressable by the view."""
        return 0 if parent is not None and parent.isValid() else len(self.rows)

    def columnCount(self, parent=None):  # noqa: N802
        """Return the fixed number of result columns."""
        return 0 if parent is not None and parent.isValid() else len(_RESULT_COLUMNS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):  # noqa: N802
        """Label the columns and number the rows."""
        if role == Qt.ItemDataRole.DisplayRole:
            return (
                _RESULT_HEADERS[section]
                if orientation == Qt.Orientation.Horizontal
                else str(section + 1)
            )
        return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        """Format boolean verdicts and explain genuinely absent pipeline results."""
        if not index.isValid() or role not in (
            Qt.ItemDataRole.DisplayRole,
            Qt.ItemDataRole.ToolTipRole,
        ):
            return None
        row = self.rows.iloc[index.row()]
        key = _RESULT_COLUMNS[index.column()]
        value = row.get(key)
        if key == "kept":
            if pd.isna(value):
                return str(row.get("pipeline_status", "Not evaluated"))
            if role == Qt.ItemDataRole.ToolTipRole and not bool(value):
                return f"Dropped: {row.get('drop_reason', 'see pipeline result')}"
            return "Kept" if bool(value) else "Dropped"
        return "" if pd.isna(value) else str(value)


def _make_searchable_combo() -> QComboBox:
    """An editable combo box with type-ahead, substring-matching completion."""
    combo = QComboBox()
    combo.setEditable(True)
    combo.setMinimumContentsLength(8)
    combo.setSizeAdjustPolicy(
        QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
    )
    combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
    completer = combo.completer()
    if completer is not None:
        completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
    combo.setCurrentText("")
    return combo


class FlightPicker(QWidget):
    """Emits ``flight_chosen(path, discipline, flight_id)`` when a flight is picked.

    Also emits ``folders_changed`` whenever the data folder changes
    (:meth:`refresh_folders`) -- so a sibling widget with its own cache of the same
    archive (:class:`~xc_viewer.widgets.map_view.MapView`) knows to drop it too,
    rather than keep showing whichever root was current when it last loaded. Its
    "Choose data folder" button only emits ``data_folder_requested``: the window owns
    the dialog and the remembered setting.
    """

    flight_chosen = pyqtSignal(object, object, str)
    folders_changed = pyqtSignal()
    data_folder_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the discipline/browse controls, the catalog filter form and results."""
        super().__init__(parent)
        self._rows = pd.DataFrame(columns=_RESULT_COLUMNS)
        self._forced_discipline: Discipline | None = None

        self._btn_data_folder = QPushButton("Choose data folder…")
        self._btn_data_folder.setShortcut(QKeySequence(QKeySequence.StandardKey.Open))
        self._btn_data_folder.setToolTip(
            "Select the xc-viewer-data folder, the one holding paragliders/ "
            "and hang_gliders/. The viewer remembers it."
        )
        self._btn_data_folder.clicked.connect(self.data_folder_requested)
        # Wrapped, so a long folder name never widens the picker.
        self._data_folder_label = QLabel()
        self._data_folder_label.setWordWrap(True)
        self._data_folder_label.setProperty("role", "muted")

        self._discipline_combo = QComboBox()
        for disc in DISCIPLINES.values():
            self._discipline_combo.addItem(disc.name.capitalize(), disc)
        self._discipline_label = QLabel()

        self._btn_browse = QPushButton("Browse .igc file…")
        self._btn_browse_folder = QPushButton("Browse folder of .igc files…")

        self._filter_combos: dict[str, QComboBox] = {
            column: _make_searchable_combo() for column, _label in _FILTER_FIELDS
        }
        self._month_combo = QComboBox()
        self._month_combo.addItem("(any)", None)
        for month in range(1, 13):
            self._month_combo.addItem(
                ENGLISH.monthName(month, QLocale.FormatType.LongFormat), month
            )
        # Not a searchable combo like _filter_combos: these aren't catalog columns
        # (distinct_values() has nothing to offer for them) but a fixed, small set of
        # boxes/bands computed from the pipeline's own lat0/lon0/alt0 -- see
        # catalog_index.filter_flights().
        self._region_combo = QComboBox()
        self._region_combo.addItem("(any)", None)
        for name in geography.REGIONS:
            self._region_combo.addItem(name, name)
        self._terrain_combo = QComboBox()
        self._terrain_combo.addItem("(any)", None)
        for name in geography.TERRAIN_ORDER:
            self._terrain_combo.addItem(name, name)
        self._kept_only_check = QCheckBox("Kept by pipeline only")
        self._btn_search = QPushButton("Search catalog")
        self._btn_search.setProperty("emphasis", "primary")

        filter_form = QFormLayout()
        filter_form.setContentsMargins(6, 8, 6, 6)
        filter_form.setHorizontalSpacing(8)
        filter_form.setVerticalSpacing(6)
        filter_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        for column, label in _FILTER_FIELDS:
            filter_form.addRow(label, self._filter_combos[column])
            if column == "wing_class":
                # Grouped with the other always-present, calendar-fixed field.
                filter_form.addRow("Month", self._month_combo)
            if column == "takeoff":
                # Grouped with the site name they refine (pipeline-derived, unlike it).
                filter_form.addRow("Region", self._region_combo)
                filter_form.addRow("Terrain", self._terrain_combo)
        filter_form.addRow(self._kept_only_check)
        filter_box = QGroupBox("Filter the catalog")
        filter_box.setToolTip("Catalog metadata comes from the FFVL and can be wrong.")
        filter_box.setLayout(filter_form)
        filter_scroll = QScrollArea()
        filter_scroll.setFrameShape(QFrame.Shape.NoFrame)
        filter_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        filter_scroll.setWidget(filter_box)
        filter_scroll.setWidgetResizable(True)
        filter_scroll.setMaximumHeight(420)

        self._results = QTableView()
        self._results_model = FlightResultsModel(self._results)
        self._results.setModel(self._results_model)
        self._results.setAlternatingRowColors(True)
        self._results.setShowGrid(False)
        self._results.verticalHeader().hide()
        header = self._results.horizontalHeader()
        if header is not None:
            header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
            self._results.setColumnWidth(0, 100)
            self._results.setColumnWidth(1, 100)
            self._results.setColumnWidth(6, 170)
        self._results.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._results.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._results.doubleClicked.connect(self._on_result_double_clicked)

        self._status = QLabel("No flight loaded.")
        self._status.setWordWrap(True)
        self._status.setProperty("role", "muted")

        heading = QLabel("Flight browser")
        heading.setProperty("role", "heading")
        data_box = QGroupBox("Data")
        data_layout = QVBoxLayout(data_box)
        data_layout.setContentsMargins(6, 8, 6, 6)
        data_layout.setSpacing(6)
        data_layout.addWidget(self._btn_data_folder)
        data_layout.addWidget(self._data_folder_label)
        self._discipline_caption = QLabel("Discipline")
        discipline_row = QHBoxLayout()
        discipline_row.addWidget(self._discipline_caption)
        discipline_row.addWidget(self._discipline_combo, 1)
        discipline_row.addWidget(self._discipline_label, 1)
        data_layout.addLayout(discipline_row)
        data_layout.addWidget(self._btn_browse)
        data_layout.addWidget(self._btn_browse_folder)
        results_heading = QLabel("Results · double-click a flight")
        results_heading.setProperty("role", "section")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)
        layout.addWidget(heading)
        layout.addWidget(data_box)
        layout.addWidget(filter_scroll)
        layout.addWidget(self._btn_search)
        layout.addWidget(results_heading)
        layout.addWidget(self._results, 1)
        layout.addWidget(self._status)

        self._btn_browse.clicked.connect(self._on_browse)
        self._btn_browse_folder.clicked.connect(self._on_browse_folder)
        self._btn_search.clicked.connect(self._on_search)
        self._discipline_combo.currentIndexChanged.connect(
            self._repopulate_filter_combos
        )

        self._update_discipline_selector()
        self._update_data_folder_label()
        # Deferred rather than called here directly: at construction time the window
        # is not shown yet, so a "Loading metadata..." status would paint on nothing
        # and the whole app would appear frozen for the ~2-3 s a first paraglider
        # catalog read takes. Queuing it for the next spin of the event loop lets the
        # window appear first.
        QTimer.singleShot(0, self._repopulate_filter_combos)

    def set_status(self, text: str) -> None:
        """Show a message below the results table (which flight loaded, or why not)."""
        self._status.setText(text)

    def current_discipline(self) -> Discipline:
        """The discipline every action (search, browse) currently applies to.

        The forced discipline (see :meth:`_update_discipline_selector`) when there is
        one; the combo box's selection otherwise. Never reads the combo directly
        elsewhere in this class, since a *hidden* combo can still hold a stale
        selection (e.g. "hang gliders" left over from before its folder stopped being
        reachable) that must not leak back in.
        """
        if self._forced_discipline is not None:
            return self._forced_discipline
        return self._discipline_combo.currentData()

    def _update_discipline_selector(self) -> None:
        """Show the discipline combo only when picking one is a real choice.

        With both archives reachable, both remain offered. With exactly one, there is
        nothing to choose -- the combo is replaced by a plain label naming it. With
        neither, a choice is still shown: an arbitrary browsed file still needs a
        discipline for the pipeline's speed threshold, and nothing here can guess it.
        """
        reachable = data.reachable_disciplines()
        if len(reachable) == 1:
            self._forced_discipline = reachable[0]
            self._discipline_combo.setVisible(False)
            self._discipline_caption.setVisible(False)
            name = reachable[0].name.capitalize()
            self._discipline_label.setText(f"Discipline: {name} (the only one)")
            self._discipline_label.setVisible(True)
        else:
            self._forced_discipline = None
            self._discipline_combo.setVisible(True)
            self._discipline_caption.setVisible(True)
            self._discipline_label.setVisible(False)

    def _update_data_folder_label(self) -> None:
        folder = datafolder.current()
        if folder is None:
            self._data_folder_label.setText("No data folder chosen.")
            self._data_folder_label.setToolTip("")
        else:
            self._data_folder_label.setText(f"Data folder: {folder.name}")
            self._data_folder_label.setToolTip(str(folder))

    def refresh_folders(self) -> None:
        """Re-read both archive roots after the data folder changed."""
        catalog_index.clear_cache()
        self._update_data_folder_label()
        self._update_discipline_selector()
        self._repopulate_filter_combos()
        self.folders_changed.emit()

    def _repopulate_filter_combos(self) -> None:
        discipline = self.current_discipline()
        self.set_status("Loading metadata…")
        QApplication.processEvents()
        try:
            for column, combo in self._filter_combos.items():
                values = catalog_index.distinct_values(discipline, column)
                current = combo.currentText()
                combo.blockSignals(True)
                combo.clear()
                combo.addItems(values)
                combo.setCurrentText(current)
                combo.blockSignals(False)
        except FileNotFoundError as exc:
            self.set_status(str(exc))
            return
        self.set_status("Ready — pick a filter or browse a file.")

    def _on_browse(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self, "Choose an .igc file", "", "IGC files (*.igc)"
        )
        if not path_str:
            return
        path = Path(path_str)
        discipline = data.resolve_discipline(path)
        if discipline is not None and self._forced_discipline is None:
            index = self._discipline_combo.findData(discipline)
            if index >= 0:
                self._discipline_combo.setCurrentIndex(index)
        if discipline is None:
            discipline = self.current_discipline()
        self.flight_chosen.emit(path, discipline, path.stem)

    def _on_browse_folder(self) -> None:
        dir_str = QFileDialog.getExistingDirectory(
            self, "Choose a folder of .igc files"
        )
        if not dir_str:
            return
        root = Path(dir_str)
        self.set_status("Scanning folder…")
        QApplication.processEvents()
        # islice on the walk itself, not on a fully materialised+sorted list: a
        # season (or the whole archive) can hold tens of thousands of files, and this
        # is the difference between the cap actually bounding the work and only
        # bounding the display.
        paths = sorted(itertools.islice(root.rglob("*.igc"), _MAX_FOLDER_SCAN))

        if paths and self._forced_discipline is None:
            discipline = data.resolve_discipline(paths[0])
            if discipline is not None:
                index = self._discipline_combo.findData(discipline)
                if index >= 0:
                    self._discipline_combo.setCurrentIndex(index)

        records = []
        for path in paths:
            try:
                date, flight_id = parse_igc_filename(path.name)
            except ValueError:
                date, flight_id = "—", path.stem
            records.append(
                {
                    "flight_id": flight_id,
                    "date": date,
                    "dept": "—",
                    "takeoff": "—",
                    "flight_type": "—",
                    "wing_class": "—",
                    "kept": pd.NA,
                    "path": path,
                }
            )
        self._rows = pd.DataFrame(records, columns=[*_RESULT_COLUMNS, "path"])
        self._results_model.set_rows(self._rows)

        truncated = len(paths) == _MAX_FOLDER_SCAN
        suffix = (
            f" (stopped at {_MAX_FOLDER_SCAN}; narrow the folder for more)"
            if truncated
            else ""
        )
        self.set_status(f"{len(paths)} .igc file(s) found in {root.name}{suffix}.")

    def _on_search(self) -> None:
        discipline = self.current_discipline()
        year_text = self._filter_combos["season_year"].currentText().strip()
        text_values = {
            column: (self._filter_combos[column].currentText().strip() or None)
            for column, _label in _FILTER_FIELDS
            if column != "season_year"
        }
        # The first search of a session reads catalog.csv in full (tens of MB,
        # ~2-3 s for paragliders) before caching it; every later search is instant.
        # Without repainting first, that one-time read looks like a frozen window.
        self.set_status("Searching…")
        self._btn_search.setEnabled(False)
        QApplication.processEvents()
        try:
            rows = catalog_index.filter_flights(
                discipline,
                season_year=int(year_text) if year_text else None,
                month=self._month_combo.currentData(),
                region=self._region_combo.currentData(),
                terrain=self._terrain_combo.currentData(),
                kept_only=self._kept_only_check.isChecked(),
                **text_values,
            )
        except FileNotFoundError as exc:
            self._rows = pd.DataFrame(columns=_RESULT_COLUMNS)
            self._results_model.set_rows(self._rows)
            self.set_status(str(exc))
            return
        except ValueError:
            self.set_status("Season year must be a whole number, e.g. 2022.")
            return
        finally:
            self._btn_search.setEnabled(True)

        self._rows = rows
        self._results_model.set_rows(rows)
        self.set_status(f"{len(rows)} flight(s) match — all available in the table.")

    def _on_result_double_clicked(self, item: QModelIndex) -> None:
        row = self._rows.iloc[item.row()]
        discipline = self.current_discipline()
        # A folder-scan row already carries its own path (found by rglob, not looked
        # up); a catalog-search row does not and needs resolve_igc_path.
        path = row["path"] if "path" in row.index and pd.notna(row["path"]) else None
        if path is None:
            path = catalog_index.resolve_igc_path(discipline, row)
        if path is None:
            self.set_status(f"Flight {row['flight_id']}: .igc file not found on disk.")
            return
        self.flight_chosen.emit(path, discipline, str(row["flight_id"]))
