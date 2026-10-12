"""Tests for the discipline-selection logic in xc_viewer.widgets.flight_picker.

Qt widget tests: need ``qapp`` (tests/viewer/conftest.py). ``Discipline.config()`` is
monkeypatched the same way tests/viewer/test_catalog_index.py does, pointing at
tmp_path roots instead of the real archive. A picker is ``.show()``n before any
visibility assertion: an un-shown top-level widget reports every descendant as
invisible regardless of ``setVisible()``, so that alone would make every assertion
below pass by accident.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from xc_viewer import catalog_index
from xc_viewer.core import disciplines as disciplines_mod
from xc_viewer.core.disciplines import DataRoot
from xc_viewer.widgets.flight_picker import FlightPicker


def _fake_config_for(root_by_name: dict):
    def fake_config(self):
        root = root_by_name.get(self.name)
        if root is None:
            raise FileNotFoundError("not configured in this test")
        return DataRoot(data_root=root)

    return fake_config


def _make_root(tmp_path, name):
    root = tmp_path / name
    (root / "raw" / "igc").mkdir(parents=True)
    return root


@pytest.fixture(autouse=True)
def _clear_caches():
    catalog_index._catalog_cache.clear()
    catalog_index._flights_meta_cache.clear()
    yield
    catalog_index._catalog_cache.clear()
    catalog_index._flights_meta_cache.clear()


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    # Discipline.config() reads XC_VIEWER_*_ROOT ahead of the data folder;
    # isolate the tests from whatever this shell happens to have set.
    monkeypatch.delenv("XC_VIEWER_PARA_ROOT", raising=False)
    monkeypatch.delenv("XC_VIEWER_HANG_ROOT", raising=False)
    monkeypatch.delenv(disciplines_mod.DATA_FOLDER_ENV, raising=False)


def test_discipline_choice_hidden_when_only_paragliders_reachable(
    qapp, tmp_path, monkeypatch
):
    para_root = _make_root(tmp_path, "para")
    monkeypatch.setattr(
        disciplines_mod.Discipline,
        "config",
        _fake_config_for({"paragliders": para_root}),
    )
    picker = FlightPicker()
    picker.show()
    assert picker._discipline_combo.isVisible() is False
    assert picker._discipline_label.isVisible() is True
    assert "Paragliders" in picker._discipline_label.text()
    assert picker.current_discipline().name == "paragliders"


def test_discipline_choice_shown_when_both_reachable(qapp, tmp_path, monkeypatch):
    para_root = _make_root(tmp_path, "para")
    hang_root = _make_root(tmp_path, "hang")
    monkeypatch.setattr(
        disciplines_mod.Discipline,
        "config",
        _fake_config_for({"paragliders": para_root, "hang gliders": hang_root}),
    )
    picker = FlightPicker()
    picker.show()
    assert picker._discipline_combo.isVisible() is True
    assert picker._discipline_label.isVisible() is False


def test_discipline_choice_shown_when_neither_reachable(qapp, monkeypatch):
    def raising_config(self):
        raise FileNotFoundError("nothing configured")

    monkeypatch.setattr(disciplines_mod.Discipline, "config", raising_config)
    picker = FlightPicker()
    picker.show()
    # Still a real choice here: an arbitrary browsed file still needs a discipline
    # for the pipeline's speed threshold, and nothing can guess it from no archive.
    assert picker._discipline_combo.isVisible() is True
    assert picker._discipline_label.isVisible() is False


def test_refresh_after_a_new_data_folder_updates_selector_and_label(
    qapp, tmp_path, monkeypatch
):
    from xc_viewer import datafolder

    def folder_config(self):
        # A reduced stand-in for the real Discipline.config(): only the data folder,
        # so the "before" state is deterministic whatever this machine has mounted.
        folder = datafolder.current()
        if folder is None or not (folder / self.folder / "raw" / "igc").is_dir():
            raise FileNotFoundError("not in the data folder")
        return DataRoot(data_root=folder / self.folder)

    monkeypatch.delenv(disciplines_mod.DATA_FOLDER_ENV, raising=False)
    monkeypatch.setattr(disciplines_mod.Discipline, "config", folder_config)
    picker = FlightPicker()
    picker.show()
    received = []
    picker.folders_changed.connect(lambda: received.append(1))
    assert picker._discipline_combo.isVisible() is True  # neither reachable yet
    assert picker._data_folder_label.text() == "No data folder chosen."

    folder = tmp_path / "xc-viewer-data"
    _make_root(folder, "paragliders")
    monkeypatch.setenv(disciplines_mod.DATA_FOLDER_ENV, str(folder))
    picker.refresh_folders()

    assert picker._discipline_combo.isVisible() is False
    assert picker._discipline_label.isVisible() is True
    assert picker.current_discipline().name == "paragliders"
    assert picker._data_folder_label.text() == "Data folder: xc-viewer-data"
    assert picker._data_folder_label.toolTip() == str(folder)
    # A sibling widget with its own cache (MapView) needs this to know the archive
    # roots just changed -- catalog_index's own cache being cleared isn't enough,
    # since that cache is invisible to whatever a widget cached on its own.
    assert received == [1]


def test_data_folder_button_asks_the_window(qapp, monkeypatch):
    monkeypatch.setattr(FlightPicker, "_repopulate_filter_combos", lambda self: None)
    picker = FlightPicker()
    received = []
    picker.data_folder_requested.connect(lambda: received.append(1))
    picker._btn_data_folder.click()
    assert received == [1]
    assert picker._btn_data_folder.text() == "Choose data folder…"


def test_search_stays_clickable_without_scrolling_a_small_sidebar(qapp, monkeypatch):
    from PyQt6.QtCore import QSize, Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QWidget

    searches = []
    monkeypatch.setattr(FlightPicker, "_repopulate_filter_combos", lambda self: None)
    monkeypatch.setattr(FlightPicker, "_on_search", lambda self: searches.append(1))
    picker = FlightPicker()
    picker.resize(260, 550)
    picker.show()
    qapp.processEvents()
    widest = sorted(
        (
            child.minimumSizeHint().width(),
            type(child).__name__,
            getattr(child, "text", lambda: "")(),
        )
        for child in picker.findChildren(QWidget)
        if child.isVisibleTo(picker)
    )[-8:]
    assert picker.size() == QSize(260, 550), (
        f"font {picker.font().toString()}, DPI {picker.logicalDpiX()}, "
        f"widest controls {widest}"
    )
    button = picker._btn_search
    center = button.mapTo(picker, button.rect().center())
    assert picker.rect().contains(center)
    assert picker.childAt(center) is button
    QTest.mouseClick(button, Qt.MouseButton.LeftButton)
    assert searches == [1]
    picker.close()


def test_all_catalog_matches_are_selectable_and_verdicts_are_explicit(
    qapp, monkeypatch
):
    import numpy as np
    import pandas as pd
    from PyQt6.QtCore import Qt

    monkeypatch.setattr(FlightPicker, "_repopulate_filter_combos", lambda self: None)
    rows = pd.DataFrame({"flight_id": [str(i) for i in range(200_001)]})
    rows["kept"] = pd.Series(
        [True, False, pd.NA] + [True] * (len(rows) - 3), dtype="boolean"
    )
    rows["pipeline_status"] = "No archived result"
    rows["drop_reason"] = "altitude unavailable"
    monkeypatch.setattr(catalog_index, "filter_flights", lambda *a, **kw: rows)
    chosen_path = Path("last.igc")
    monkeypatch.setattr(catalog_index, "resolve_igc_path", lambda d, row: chosen_path)
    picker = FlightPicker()
    picker._on_search()
    model = picker._results.model()
    assert model.rowCount() == len(rows)
    assert isinstance(rows.iloc[0]["kept"], np.bool_)
    assert [model.data(model.index(i, 6)) for i in range(3)] == [
        "Kept",
        "Dropped",
        "No archived result",
    ]
    assert "altitude unavailable" in model.data(
        model.index(1, 6), Qt.ItemDataRole.ToolTipRole
    )
    received = []
    picker.flight_chosen.connect(lambda *args: received.append(args))
    picker._on_result_double_clicked(model.index(len(rows) - 1, 0))
    assert received[0][0] == chosen_path
    assert received[0][2] == "200000"
    picker.close()
