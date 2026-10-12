"""Packing copies what the viewer reads and keeps saved products usable elsewhere."""

import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from xc_viewer import route_index, thermal_index
from xc_viewer.core.disciplines import (
    DATA_FOLDER_ENV,
    HANG_GLIDERS,
    PARAGLIDERS,
)
from xc_viewer.sqlite import connect

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/pack_data_folder.py"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (DATA_FOLDER_ENV, PARAGLIDERS.env, HANG_GLIDERS.env):
        monkeypatch.delenv(name, raising=False)


def _thesis_archive(root):
    """The thesis layout: tracks, tables, Vilpellet products and viewer products."""
    track = root / "raw/igc/2020-2021/2021-05-01_1.igc"
    track.parent.mkdir(parents=True)
    track.write_text("AXXX\n")
    (root / "raw/raw_xml").mkdir()
    (root / "raw/raw_xml/not-read.xml").write_text("<x/>")
    (root / "catalog").mkdir()
    pd.DataFrame({"flight_id": ["1"]}).to_csv(root / "catalog/catalog.csv")
    derived = root / "derived"
    vilpellet = derived / "segmentation/vilpellet"
    (vilpellet / "model").mkdir(parents=True)
    pd.DataFrame({"t": [0.0]}).to_parquet(derived / "fixes.parquet")
    pd.DataFrame({"flight_id": ["1"]}).to_parquet(derived / "flights_meta.parquet")
    pd.DataFrame({"flight_id": ["1"]}).to_parquet(vilpellet / "phase_segments.parquet")
    pd.DataFrame({"flight_id": ["1"]}).to_parquet(vilpellet / "phase_coverage.parquet")
    (vilpellet / "model/parameters.json").write_text("{}")
    products = derived / "viewer/thermal-planes"
    products.mkdir(parents=True)
    with connect(products / "thermal-cells.sqlite3") as db:
        db.execute("CREATE TABLE metadata (signature TEXT)")
        db.execute("INSERT INTO metadata VALUES ('old-census')")
    with connect(products / "route-cells.sqlite3") as db:
        db.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
        db.executemany(
            "INSERT INTO metadata VALUES (?,?)",
            [("signature", "old-routes"), ("complete", "yes")],
        )
    with connect(products / "thermal-planes.sqlite3") as db:
        db.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
        db.executemany(
            "INSERT INTO metadata VALUES (?,?)",
            [
                ("archive_signature", "old-census"),
                ("segmentation_signatures", '{"own":"hmm","vilpellet":"old-climbs"}'),
            ],
        )
    with connect(products / "thermal-climbs.sqlite3") as db:
        db.execute(
            "CREATE TABLE climbs (cache_key TEXT, discipline TEXT, flight_id TEXT, "
            "ix INTEGER, iy INTEGER, status TEXT, edges BLOB)"
        )
        db.executemany(
            "INSERT INTO climbs VALUES (?,?,?,?,?,?,?)",
            [
                ("old-climbs", "paragliders", "1", 1, 2, "decoded", b"x"),
                ("hmm", "paragliders", "1", 1, 2, "decoded", b"y"),
            ],
        )
    metadata = np.array(json.dumps({"signature": "old-density", "complete": True}))
    np.savez(
        products / "route-thermal-duration-vilpellet.npz",
        metadata=metadata,
        version=np.array(1),
    )
    table = pa.table({"flight_id": ["1"]}).replace_schema_metadata(
        {b"group_signature": b"old-group"}
    )
    pq.write_table(table, products / "group-flight-catalog.parquet")
    (products / ".prepare.lock").write_text("")
    return root


def test_pack_copies_reads_only_and_restamps_for_another_computer(
    tmp_path, monkeypatch
):
    source = _thesis_archive(tmp_path / "ssd/paragliders/ffvl_cfd_igc")
    destination = tmp_path / "data"
    argv = ["pack", "--para", str(source), "--to", str(destination)]
    argv += ["--thesis-repo", str(tmp_path / "no-thesis")]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as done:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    assert done.value.code == 0

    root = destination / "paragliders"
    assert (root / "raw/igc/2020-2021/2021-05-01_1.igc").is_file()
    assert not (root / "raw/raw_xml").exists()
    products = root / "derived/viewer/thermal-planes"
    assert not (products / ".prepare.lock").exists()

    # The viewer accepts the copied census and route index without preparing them.
    assert thermal_index.load_saved_index([PARAGLIDERS]) is not None
    assert route_index.load_saved_index([PARAGLIDERS]) is not None
    census = thermal_index.archive_signature([PARAGLIDERS])
    with connect(products / "thermal-planes.sqlite3") as db:
        metadata = dict(db.execute("SELECT key,value FROM metadata"))
    assert metadata["archive_signature"] == census
    keys = json.loads(metadata["segmentation_signatures"])
    assert list(keys) == ["vilpellet"]
    with connect(products / "thermal-climbs.sqlite3") as db:
        rows = db.execute("SELECT cache_key, edges FROM climbs").fetchall()
    assert rows == [(keys["vilpellet"], b"x")]
    with np.load(products / "route-thermal-duration-vilpellet.npz") as saved:
        density = json.loads(str(saved["metadata"]))["signature"]
    assert density != "old-density"
    group = pq.read_schema(products / "group-flight-catalog.parquet").metadata
    assert group[b"group_signature"] != b"old-group"
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["products"]["census"] == "ready"


def test_pack_links_tracks_from_an_identical_copy(tmp_path, monkeypatch):
    import shutil

    source = _thesis_archive(tmp_path / "ssd/paragliders/ffvl_cfd_igc")
    copy = tmp_path / "other/paragliders/ffvl_cfd_igc"
    shutil.copytree(source / "raw", copy / "raw")
    shutil.copytree(source / "catalog", copy / "catalog")
    (copy / "catalog/catalog.csv").write_text("changed,but\n")
    destination = tmp_path / "data"
    argv = ["pack", "--para", str(source), "--to", str(destination)]
    argv += ["--para-copy", str(copy), "--thesis-repo", str(tmp_path / "none")]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        runpy.run_path(str(SCRIPT), run_name="__main__")
    track = "raw/igc/2020-2021/2021-05-01_1.igc"
    assert (destination / "paragliders" / track).samefile(copy / track)
    # A differing file is copied from the archive, never linked.
    catalog = destination / "paragliders/catalog/catalog.csv"
    assert catalog.read_bytes() == (source / "catalog/catalog.csv").read_bytes()


def _run(argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["pack", *argv])
    with pytest.raises(SystemExit) as done:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    return done.value.code


def test_a_disk_without_hard_links_refuses_instead_of_copying(tmp_path, monkeypatch):
    import os

    source = _thesis_archive(tmp_path / "ssd/paragliders/ffvl_cfd_igc")
    destination = tmp_path / "data"

    def no_links(*_):
        raise OSError("exFAT")

    monkeypatch.setattr(os, "link", no_links)
    argv = ["--para", str(source), "--to", str(destination)]
    argv += ["--para-copy", str(source), "--thesis-repo", str(tmp_path / "none")]
    assert _run(argv, monkeypatch) == 2
    assert not (destination / "paragliders").exists()


def test_not_enough_space_stops_before_copying(tmp_path, monkeypatch):
    import shutil
    from collections import namedtuple

    source = _thesis_archive(tmp_path / "ssd/paragliders/ffvl_cfd_igc")
    destination = tmp_path / "data"
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda _: usage(10, 10, 0))
    argv = ["--para", str(source), "--to", str(destination)]
    argv += ["--thesis-repo", str(tmp_path / "none")]
    assert "space" in str(_run(argv, monkeypatch))
    assert not destination.exists()
