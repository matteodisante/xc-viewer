"""Every tab opens on a minimal data folder: what CI checks on each OS.

The folder holds one paraglider track and nothing prepared, the state of a fresh data
folder. Each tab must still open and say what is missing rather than fail.
"""

import shutil
import sys
from pathlib import Path

import pytest

from xc_viewer.core.disciplines import (
    DATA_FOLDER_ENV,
    HANG_GLIDERS,
    PARAGLIDERS,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "sample_flight.igc"


@pytest.fixture
def data_folder(tmp_path, monkeypatch):
    folder = tmp_path / "data"
    track = folder / "paragliders/raw/igc/2020-2021/2021-05-01_20311250.igc"
    track.parent.mkdir(parents=True)
    shutil.copyfile(FIXTURE, track)
    monkeypatch.setenv(DATA_FOLDER_ENV, str(folder))
    for discipline in (PARAGLIDERS, HANG_GLIDERS):
        monkeypatch.delenv(discipline.env, raising=False)
    return folder


def test_every_tab_opens_and_a_track_loads(qapp, data_folder, monkeypatch):
    from xc_viewer.main_window import MainWindow

    errors = []
    monkeypatch.setattr(sys, "excepthook", lambda *exc: errors.append(exc))
    window = MainWindow()
    window.show()
    qapp.processEvents()
    assert not window._figure.axes[0].axison
    assert str(data_folder) in window.windowTitle()
    for index in range(window._tabs.count()):
        window._tabs.setCurrentIndex(index)
        qapp.processEvents()
    window._tabs.setCurrentIndex(0)
    track = next(data_folder.rglob("*.igc"))
    window._on_flight_chosen(track, PARAGLIDERS, "20311250")
    qapp.processEvents()
    assert window._raw is not None
    assert window._figure.axes[0].axison
    window.close()
    assert errors == []
