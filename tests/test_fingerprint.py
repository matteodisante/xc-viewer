"""Saved products stay valid when the data folder moves to another computer."""

import os
import shutil

import pandas as pd
import pytest

from xc_viewer import datafolder, route_index, thermal_index
from xc_viewer.core.disciplines import DATA_FOLDER_ENV, PARAGLIDERS
from xc_viewer.fingerprint import SMALL_FILE_BYTES, file_identity


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (DATA_FOLDER_ENV, PARAGLIDERS.env):
        monkeypatch.delenv(name, raising=False)


def _archive(folder):
    root = folder / "paragliders"
    (root / "catalog").mkdir(parents=True)
    (root / "derived").mkdir()
    pd.DataFrame({"flight_id": ["a"]}).to_csv(root / "catalog/catalog.csv")
    pd.DataFrame({"t": [0.0, 1.0]}).to_parquet(root / "derived/fixes.parquet")
    pd.DataFrame({"flight_id": ["a"]}).to_parquet(root / "derived/flights_meta.parquet")
    return folder


def _signatures(folder):
    datafolder.use(folder)
    return (
        thermal_index.archive_signature([PARAGLIDERS]),
        route_index.archive_signature([PARAGLIDERS]),
    )


def test_a_copy_elsewhere_with_new_times_keeps_every_signature(tmp_path):
    here = _archive(tmp_path / "here")
    # copyfile, not copy2: modification times are not carried over.
    there = tmp_path / "elsewhere" / "data"
    shutil.copytree(here, there, copy_function=shutil.copyfile)
    for path in there.rglob("*"):
        if path.is_file():
            os.utime(path, ns=(10**18, 10**18))
    assert _signatures(here) == _signatures(there)


def test_changed_content_changes_the_signature(tmp_path):
    folder = _archive(tmp_path)
    before = _signatures(folder)
    pd.DataFrame({"flight_id": ["b"]}).to_csv(
        folder / "paragliders/catalog/catalog.csv"
    )
    assert _signatures(folder)[0] != before[0]


def test_large_files_are_identified_by_size(tmp_path):
    big = tmp_path / "big.bin"
    with big.open("wb") as out:
        out.truncate(SMALL_FILE_BYTES + 1)
    assert file_identity(big, tmp_path) == ["big.bin", SMALL_FILE_BYTES + 1]


def test_a_rewrite_with_same_size_and_time_is_still_seen(tmp_path):
    # Windows can give two quick writes the same modification time.
    path = tmp_path / "config.yaml"
    path.write_text("version: 1")
    stamp = path.stat().st_mtime_ns
    before = file_identity(path)
    path.write_text("version: 2")
    os.utime(path, ns=(stamp, stamp))
    assert file_identity(path) != before
