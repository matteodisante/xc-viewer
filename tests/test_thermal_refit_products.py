"""Each saved segmentation prepares thermal products of its own.

Vilpellet keeps the file names and keys of the data folders already prepared; the refit
writes beside them and never into their files.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from xc_viewer import segmentation
from xc_viewer.core.disciplines import DATA_FOLDER_ENV, PARAGLIDERS
from xc_viewer.sqlite import connect
from xc_viewer.thermal_cache import ClimbCache, climbs_path, segmentation_signature
from xc_viewer.thermal_geometry import CELL_M, ThermalCell, project
from xc_viewer.thermal_index import ThermalIndex, _create_tables

PLANES = "thermal-planes.sqlite3"
FLIGHTS = ("10", "11")


@pytest.fixture(autouse=True)
def _vilpellet_default():
    segmentation.use(segmentation.DEFAULT)
    yield
    segmentation.use(segmentation.DEFAULT)


@pytest.fixture
def archive(tmp_path, monkeypatch):
    """A data folder with two flights, their saved refit runs and a census of them.

    Flight 10 climbs from t = 10 s to 49 s in one cell; flight 11 never climbs. The
    saved segment numbers (7) are not those of the cleaned fixes (0): runs join by time.
    """
    monkeypatch.setenv(DATA_FOLDER_ENV, str(tmp_path))
    monkeypatch.delenv(PARAGLIDERS.env, raising=False)
    derived = tmp_path / "paragliders/derived"
    refit = derived / "segmentation/refit"
    (refit / "model").mkdir(parents=True)
    t = np.arange(60, dtype=float)
    fixes = pd.concat(
        pd.DataFrame(
            {
                "flight_id": fid,
                "segment_id": 0,
                "t": t,
                "E": t * 5,
                "N": t * 0.0,
                "z": 100 + t * 2,
            }
        )
        for fid in FLIGHTS
    )
    fixes.to_parquet(derived / "fixes.parquet")
    pd.DataFrame(
        {
            "source": "paraglider",
            "flight_id": ["10", "10", "10", "11"],
            "segment_id": 7,
            "phase": ["search", "climb", "transition", "search"],
            "t_start": [0.0, 10.0, 50.0, 0.0],
            "t_end": [9.0, 49.0, 59.0, 59.0],
            "n_fixes": [10, 40, 10, 60],
        }
    ).to_parquet(refit / "phase_segments.parquet")
    pd.DataFrame(
        {
            "flight_id": ["10", "11"],
            "n_native_fixes": [60, 60],
            "status": "classified",
        }
    ).to_parquet(refit / "phase_coverage.parquet")
    (refit / "model/parameters.json").write_text(
        json.dumps({"discipline": PARAGLIDERS.slug})
    )
    # The saved runs must postdate the cleaned archive they label.
    later = (derived / "fixes.parquet").stat().st_mtime + 10
    for name in ("phase_segments.parquet", "phase_coverage.parquet"):
        os.utime(refit / name, (later, later))

    x, y = project(6.0, 45.0)
    ix, iy = int(x // CELL_M), int(y // CELL_M)
    census = derived / "viewer/thermal-planes/thermal-cells.sqlite3"
    census.parent.mkdir(parents=True)
    with connect(census) as db:
        _create_tables(db)
        db.executemany(
            "INSERT INTO flights VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            [("paragliders", f, "x", 1000, 0, 45, 6, 0, ix, iy, 50) for f in FLIGHTS],
        )
        db.executemany(
            "INSERT INTO visits VALUES (?,?,?,?,?,?,?)",
            [("paragliders", f, ix, iy, 0, 59, 300) for f in FLIGHTS],
        )
        db.executemany(
            "INSERT INTO flight_groups VALUES (?,?,?)",
            [("paragliders", f, 0) for f in FLIGHTS],
        )
    index = ThermalIndex(census, ("paragliders",), "census-signature")
    cell = ThermalCell(ix, iy, "Plains", 2, 2, 50.0, 300.0, launch_median_m=50.0)
    return index, cell


def test_product_names_keep_vilpellet_and_suffix_the_others():
    name = segmentation.product_name
    assert name(PLANES, "vilpellet") == PLANES
    assert name(PLANES, "refit") == "thermal-planes-refit.sqlite3"
    assert name("thermal-duration.npz", "refit") == "thermal-duration-refit.npz"
    assert name(PLANES) == PLANES
    segmentation.use("refit")
    assert name(PLANES) == "thermal-planes-refit.sqlite3"
    with pytest.raises(ValueError, match="Unknown segmentation"):
        name(PLANES, "nope")


def test_each_segmentation_has_its_own_climb_cache_and_key(archive):
    index, cell = archive
    assert climbs_path(index.path, "vilpellet").name == "thermal-climbs.sqlite3"
    assert climbs_path(index.path, "refit").name == "thermal-climbs-refit.sqlite3"
    vilpellet = segmentation_signature(index, "vilpellet")
    refit = segmentation_signature(index, "refit")
    assert vilpellet != refit
    edges = pd.DataFrame(columns=["x0", "y0", "z0", "utc0", "x1", "y1", "z1", "utc1"])
    with ClimbCache(index, "refit") as cache:
        cache.put("paragliders", "10", cell, "decoded", edges)
    assert climbs_path(index.path, "refit").is_file()
    assert not climbs_path(index.path, "vilpellet").exists()


def test_refit_key_follows_the_saved_runs(archive):
    index, _ = archive
    before = segmentation_signature(index, "refit")
    runs = (
        PARAGLIDERS.config().derived_dir / "segmentation/refit/phase_segments.parquet"
    )
    runs.write_bytes(runs.read_bytes() + b"\0")
    assert segmentation_signature(index, "refit") != before


def test_census_joins_runs_by_time_when_segment_numbers_differ(monkeypatch):
    from xc_viewer.thermal_activity import flight_activity

    monkeypatch.setattr(
        "xc_viewer.thermal_activity.enu_to_geodetic",
        lambda e, n, z, origin: (n.to_numpy(), e.to_numpy()),
    )
    monkeypatch.setattr(
        "xc_viewer.thermal_activity.project", lambda lon, lat: (lon, lat)
    )
    fixes = pd.DataFrame(
        {
            "segment_id": [0] * 6,
            "t": [0, 1, 2, 50, 51, 52],
            "E": [3] * 6,
            "N": [45] * 6,
            "z": [100] * 6,
        }
    )
    # Segment 7 is not the fixes' segment 0, and the saved count (99) is not checked.
    runs = np.array([[7, 0, 52, 99]])
    assert flight_activity(fixes, runs, (0, 0, 0), by_time=True) == [(0, 0, 2)]
    with pytest.raises(ValueError, match="do not match"):
        flight_activity(fixes, runs, (0, 0, 0))
    # Two runs close to each other stay two runs.
    two = np.array([[7, 0, 1, 2], [8, 2, 52, 4]])
    assert flight_activity(fixes, two, (0, 0, 0), by_time=True) == [(0, 0, 2)]
    # A run in the gap holds no fix: nothing to count.
    assert (
        flight_activity(fixes, np.array([[7, 10, 40, 5]]), (0, 0, 0), by_time=True)
        == []
    )


def test_refit_census_is_saved_beside_vilpellets_and_counts_runs(archive):
    from xc_viewer.thermal_activity import activity_path, prepare_activity

    index, cell = archive
    path = prepare_activity(index, source="refit", workers=1, progress=lambda _: None)
    assert path == activity_path(index.path, "refit")
    assert path.name == "thermal-refit-activity.sqlite3"
    assert activity_path(index.path, "vilpellet").name == (
        "thermal-vilpellet-activity.sqlite3"
    )
    assert not activity_path(index.path, "vilpellet").exists()
    with connect(path) as db:
        metadata = dict(db.execute("SELECT key,value FROM metadata"))
        rows = db.execute("SELECT discipline,flight_id,ix,iy,climb_runs FROM activity")
        assert rows.fetchall() == [("paragliders", "10", cell.ix, cell.iy, 1)]
        assert db.execute("SELECT COUNT(*) FROM completed").fetchone() == (2,)
    assert metadata["ready"] == "1"
    assert metadata["version"] == "refit-continuous-climb-runs-v1"


def test_refit_census_refuses_runs_older_than_the_cleaned_archive(archive):
    from xc_viewer.thermal_activity import _validated_products

    index, _ = archive
    derived = PARAGLIDERS.config().derived_dir
    old = (derived / "fixes.parquet").stat().st_mtime - 10
    os.utime(derived / "segmentation/refit/phase_segments.parquet", (old, old))
    with pytest.raises(ValueError, match="Regenerate stale Refit products"):
        _validated_products(index, "refit")


def test_refit_census_refuses_parameters_of_another_discipline(archive):
    from xc_viewer.thermal_activity import _validated_products

    index, _ = archive
    folder = PARAGLIDERS.config().derived_dir / "segmentation/refit/model"
    (folder / "parameters.json").write_text(json.dumps({"discipline": "hang"}))
    with pytest.raises(ValueError, match="Saved Refit parameters differ"):
        _validated_products(index, "refit")


def test_refit_ranking_is_named_after_the_refit(archive, monkeypatch):
    from xc_viewer.thermal_activity import prepare_activity
    from xc_viewer.thermal_ground import climb_ranking, is_climb_ranking
    from xc_viewer.thermal_ranking import rank_cells

    index, cell = archive
    activity = prepare_activity(index, source="refit", workers=1, progress=print)
    monkeypatch.setattr(
        "xc_viewer.thermal_ranking._cell_terrain",
        lambda *a: {"minimum_m": 0, "maximum_m": 300},
    )
    ranked = rank_cells(index, activity=activity, source="refit", per_category=1)
    summary = ranked.quality_summary
    assert summary["ranking_reference"] == climb_ranking("refit")
    assert summary["ranking_method"] == "refit"
    assert is_climb_ranking(summary["ranking_reference"])
    assert is_climb_ranking("vilpellet-climb-runs-dem-max-top-v1")
    assert not is_climb_ranking("ign-dem-cell-max-top-v1")
    assert ranked.activity_counts[cell.ix, cell.iy] == {
        "climb_runs": 1,
        "climb_flights": 1,
    }


def test_refit_climbs_go_to_their_own_file_and_the_snapshot_is_named_after_it(
    archive, monkeypatch
):
    from xc_viewer.thermal_activity import prepare_activity
    from xc_viewer.thermal_prepare import prepare_climbs
    from xc_viewer.thermal_ranking import rank_cells
    from xc_viewer.thermal_store import (
        ThermalStore,
        export_store,
        find_store_folder,
        find_store_path,
        load_store,
    )

    index, cell = archive
    monkeypatch.setattr(
        "xc_viewer.thermal_ranking._cell_terrain",
        lambda *a: {"minimum_m": 0, "maximum_m": 300},
    )
    monkeypatch.setattr(
        "xc_viewer.thermal_store.terrain_reference",
        lambda c: pytest.fail("the ranking's reference must be reused"),
    )
    activity = prepare_activity(index, source="refit", workers=1, progress=print)
    ranked = rank_cells(index, activity=activity, source="refit", per_category=1)
    prepare_climbs(index, workers=1, progress=print, cells=[cell], sources=("refit",))
    assert not climbs_path(index.path, "vilpellet").exists()
    with connect(climbs_path(index.path, "refit")) as db:
        found = db.execute(
            "SELECT flight_id,status,ix,iy FROM climbs ORDER BY flight_id"
        ).fetchall()
    assert found == [
        ("10", "decoded", cell.ix, cell.iy),
        ("11", "decoded", cell.ix, cell.iy),
    ]

    path = export_store(ranked, relief=[], source="refit")
    assert path.name == "thermal-planes-refit.sqlite3"
    assert not path.with_name(PLANES).exists()
    store = ThermalStore(path)
    assert store.sources == ("refit",)
    assert store.has_climb_ranking
    assert store.activity_counts[cell.ix, cell.iy]["climb_runs"] == 1

    # The climbs of flight 10 are the refit's climb run, from t = 10 s to 49 s.
    plane = store.read_plane(store.cells()[0], 0, 1e9, "refit")
    assert plane.selected == plane.decoded == 2
    assert len(plane.edges) == 39
    assert set(plane.edges.flight_id) == {"10"}
    with pytest.raises(ValueError, match=r"prepared for Refit only.*Vilpellet"):
        store.read_plane(store.cells()[0], 0, 1e9, "vilpellet")

    # Each segmentation finds the snapshot of its own; the folder is found by either.
    assert find_store_path("refit") == path
    assert find_store_path("vilpellet") is None
    assert find_store_folder() == path.parent
    assert load_store("vilpellet") is None
    segmentation.use("refit")
    assert load_store().path == path


def test_vilpellet_snapshot_keeps_its_name_and_its_key(archive, monkeypatch):
    from xc_viewer.thermal_ranking import rank_cells
    from xc_viewer.thermal_store import export_store, find_store_path

    index, cell = archive
    monkeypatch.setattr(
        "xc_viewer.thermal_ranking._cell_terrain",
        lambda *a: {"minimum_m": 0, "maximum_m": 300},
    )
    edges = pd.DataFrame(
        [[1.0, 1.0, 100.0, 1000.0, 2.0, 2.0, 120.0, 1010.0]],
        columns=["x0", "y0", "z0", "utc0", "x1", "y1", "z1", "utc1"],
    )
    with ClimbCache(index, "vilpellet") as cache:
        for fid in ("10", "11"):
            cache.put("paragliders", fid, cell, "decoded", edges)
    path = export_store(rank_cells(index, per_category=1), relief=[])
    assert path.name == PLANES
    with connect(path) as db:
        keys = json.loads(
            db.execute(
                "SELECT value FROM metadata WHERE key='segmentation_signatures'"
            ).fetchone()[0]
        )
    assert keys == {"vilpellet": segmentation_signature(index, "vilpellet")}
    assert find_store_path() == path
    assert not climbs_path(index.path, "refit").exists()
