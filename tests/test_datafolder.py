"""The data folder is validated, published to every discipline, and remembered."""

from pathlib import Path

import pytest

from xc_viewer import datafolder
from xc_viewer.core.disciplines import (
    DATA_FOLDER_ENV,
    HANG_GLIDERS,
    PARAGLIDERS,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (DATA_FOLDER_ENV, PARAGLIDERS.env, HANG_GLIDERS.env):
        monkeypatch.delenv(name, raising=False)


class FakeSettings:
    def __init__(self, values=None):
        self.values = dict(values or {})

    def value(self, key, default="", type=str):
        return self.values.get(key, default)


def test_a_folder_needs_at_least_one_discipline(tmp_path):
    assert datafolder.missing_parts(tmp_path / "absent")
    assert datafolder.missing_parts(tmp_path)
    (tmp_path / "hang_gliders").mkdir()
    assert datafolder.missing_parts(tmp_path) == []


def test_use_points_both_disciplines_and_drops_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv(PARAGLIDERS.env, str(tmp_path / "elsewhere"))
    folder = datafolder.use(tmp_path)
    assert datafolder.current() == folder
    assert PARAGLIDERS.config().data_root == folder / "paragliders"
    assert HANG_GLIDERS.config().derived_dir == folder / "hang_gliders" / "derived"


def test_no_folder_means_no_discipline_is_reachable():
    with pytest.raises(FileNotFoundError):
        PARAGLIDERS.config()
    assert PARAGLIDERS.derived_dir() is None


def test_startup_order_argument_environment_then_remembered(tmp_path, monkeypatch):
    from xc_viewer.app import _initial_data_folder

    good = tmp_path / "good"
    (good / "paragliders").mkdir(parents=True)
    other = tmp_path / "other"
    (other / "paragliders").mkdir(parents=True)
    remembered = FakeSettings({datafolder.SETTINGS_KEY: str(other)})

    assert _initial_data_folder(good, remembered) == (good, True)
    with pytest.raises(SystemExit):
        _initial_data_folder(tmp_path / "missing", remembered)
    assert _initial_data_folder(None, remembered) == (Path(other), False)
    monkeypatch.setenv(DATA_FOLDER_ENV, str(good))
    assert _initial_data_folder(None, remembered) == (good, False)


def test_a_stale_remembered_folder_asks_again(tmp_path, monkeypatch):
    from xc_viewer import app

    asked = []
    monkeypatch.setattr(
        app, "choose_data_folder", lambda parent=None, start="": asked.append(start)
    )
    stale = FakeSettings({datafolder.SETTINGS_KEY: str(tmp_path / "moved")})
    assert app._initial_data_folder(None, stale) == (None, True)
    assert asked == [str(tmp_path / "moved")]


def test_a_relative_folder_becomes_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data/paragliders").mkdir(parents=True)
    monkeypatch.setenv(DATA_FOLDER_ENV, "data")
    root = PARAGLIDERS.config().data_root
    assert root.is_absolute()
    assert root == (tmp_path / "data/paragliders").resolve()
