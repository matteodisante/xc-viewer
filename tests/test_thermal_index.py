"""Crossing census and UTC reconstruction must use different flight populations."""

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from xc_viewer import thermal_index
from xc_viewer.core.disciplines import PARAGLIDERS
from xc_viewer.core.igc import first_fix
from xc_viewer.sqlite import connect
from xc_viewer.thermal_index import (
    ThermalIndex,
    _create_tables,
    load_plane_data,
)


@pytest.fixture
def index(tmp_path):
    path = tmp_path / "cells.sqlite3"
    with connect(path) as db:
        _create_tables(db)
        # Only launches a and b set the ground (median 200). Visitor c starts in a
        # different cell at 2000 m. Repeated visits by c cannot inflate the census.
        flights = [
            ("paragliders", "a", "a.igc", 1000, 60, 45, 6, 400, 0, 0, 100),
            ("paragliders", "b", "b.igc", 1000, 60, 45, 6, 500, 0, 0, 300),
            ("paragliders", "c", "c.igc", 1000, 60, 45, 6, 2000, 1, 1, 2000),
        ]
        db.executemany("INSERT INTO flights VALUES (?,?,?,?,?,?,?,?,?,?,?)", flights)
        db.executemany(
            "INSERT INTO visits VALUES (?,?,?,?,?,?,?)",
            [
                ("paragliders", fid, 0, 0, 0, 100, peak)
                for fid, peak in [("a", 800), ("b", 900), ("c", 1800)]
            ],
        )
        db.executemany(
            "INSERT INTO flight_groups VALUES (?,?,?)",
            [("paragliders", fid, 0) for fid in ("a", "b", "c")],
        )
        db.commit()
    return ThermalIndex(path, ("paragliders",))


def test_all_crossing_flights_count_but_only_internal_starts_set_ground(index):
    cell = index.cells()[0]
    assert cell.terrain == "Plains"
    assert cell.flights == 3
    assert cell.launches == 2
    assert cell.ground_m == 200
    assert cell.max_agl_m == 1600
    assert len(index.flights(cell)) == 3


def test_raw_clock_is_header_plus_first_accepted_record(tmp_path):
    path = tmp_path / "bad-catalog-date.igc"
    path.write_text(
        "AXXX\nHFDTE150624\nB9961994432469N00542796EA010000010000\n"
        "B2359584432469N00542796EA010000010000\n"
        "B0000014432469N00542796EA010000010000\n"
    )
    origin = first_fix(path, include_utc=True)
    assert (
        origin["start_utc"] == datetime(2024, 6, 15, 23, 59, 58, tzinfo=UTC).timestamp()
    )
    assert "start_utc" not in first_fix(path)
    path.write_text(path.read_text().replace("HFDTE150624", "HFDTE000000"))
    assert np.isnan(first_fix(path, include_utc=True)["start_utc"])


def test_decode_whole_flight_and_restore_trim_offset_before_utc_filter(
    index, monkeypatch
):
    fixes = pd.DataFrame(
        {
            "flight_id": [fid for fid in ("a", "b", "c") for _ in range(3)],
            "segment_id": [0] * 9,
            "t": [0.0, 50.0, 100.0] * 3,
            "E": [1000.0, 2000.0, 3000.0] * 3,
            "N": [1000.0] * 9,
            "z": [300.0, 500.0, 700.0] * 3,
        }
    )
    fake = SimpleNamespace(
        read_row_group=lambda _: SimpleNamespace(to_pandas=lambda: fixes)
    )
    monkeypatch.setattr(thermal_index.pq, "ParquetFile", lambda _: fake)
    monkeypatch.setattr(
        type(PARAGLIDERS), "config", lambda self: SimpleNamespace(derived_dir=Path())
    )
    monkeypatch.setattr(thermal_index, "enu_to_geodetic", lambda e, n, z, frame: (n, e))
    monkeypatch.setattr(thermal_index, "project", lambda lon, lat: (lon, lat))
    calls = []

    def decode(frame, discipline):
        calls.append((frame.t.tolist(), discipline.name))
        return SimpleNamespace(fixes=frame.assign(phase="climb"))

    monkeypatch.setattr(thermal_index.data, "load_vilpellet_phases", decode)
    result = load_plane_data(index, index.cells()[0], 1080, 1090, "vilpellet")
    assert result.selected == result.decoded == 3
    assert calls == [([0.0, 50.0, 100.0], "paragliders")] * 3
    assert result.edges.utc0.tolist() == [1060.0] * 3
    assert result.edges.utc1.tolist() == [1110.0] * 3
    # A new viewer/session and a different interval reuse the whole-flight product.
    monkeypatch.setattr(
        thermal_index.pq, "ParquetFile", lambda _: pytest.fail("reread")
    )
    monkeypatch.setattr(
        thermal_index.data, "load_vilpellet_phases", lambda *_: pytest.fail("redecoded")
    )
    reopened = ThermalIndex(index.path, index.disciplines)
    later = load_plane_data(reopened, reopened.cells()[0], 1130, 1140, "vilpellet")
    assert later.cached == 3
    assert later.edges.utc0.tolist() == [1110.0] * 3
    assert later.edges.utc1.tolist() == [1160.0] * 3
    assert (
        load_plane_data(index, index.cells()[0], 1000, 1050, "vilpellet").selected == 0
    )


def test_missing_segmentation_is_explicit_not_substituted(index, monkeypatch):
    fake = SimpleNamespace(
        read_row_group=lambda _: SimpleNamespace(
            to_pandas=lambda: pd.DataFrame(
                {
                    "flight_id": ["a", "b", "c"],
                    "t": [0.0] * 3,
                    "segment_id": [0] * 3,
                }
            )
        )
    )
    monkeypatch.setattr(thermal_index.pq, "ParquetFile", lambda _: fake)
    monkeypatch.setattr(
        type(PARAGLIDERS), "config", lambda self: SimpleNamespace(derived_dir=Path())
    )
    monkeypatch.setattr(thermal_index.data, "load_vilpellet_phases", lambda *_: None)
    result = load_plane_data(index, index.cells()[0], 1060, 1160, "vilpellet")
    assert result.unavailable == 3
    assert result.decoded == 0
    assert result.edges.empty


def test_index_streams_row_groups_once_and_reuses_completed_cache(
    tmp_path, monkeypatch
):
    pytest.importorskip("pyproj")
    import pyarrow as pa
    import pyarrow.parquet as pq

    from xc_viewer.core.preproc.enu import geodetic_to_enu
    from xc_viewer.thermal_geometry import project, unproject

    derived, raw = tmp_path / "derived", tmp_path / "raw"
    derived.mkdir()
    raw.mkdir()
    catalog = tmp_path / "catalog.csv"
    pd.DataFrame({"flight_id": ["a", "b"]}).to_csv(catalog, index=False)
    config = SimpleNamespace(
        derived_dir=derived, data_root=tmp_path, igc_dir=raw, catalog_path=catalog
    )
    monkeypatch.setattr(type(PARAGLIDERS), "config", lambda self: config)
    x0, y0 = project(6, 45)
    ix, iy = int(x0 // 5000), int(y0 // 5000)
    x0, y0 = ix * 5000 + 1000, iy * 5000 + 1000
    lon0, lat0 = unproject(x0, y0)
    meta = pd.DataFrame(
        {
            "flight_id": ["a", "b"],
            "drop_reason": [None, None],
            "lat0": [lat0, lat0],
            "lon0": [lon0, lon0],
            "alt0": [100.0, 100.0],
        }
    )
    meta.to_parquet(derived / "flights_meta.parquet")
    lon, lat = unproject([x0, x0 + 10000, x0 + 10000, x0], [y0] * 4)
    altitude = np.array([100.0, 1100.0, 1100.0, 100.0])
    e, n, _ = geodetic_to_enu(lat, lon, altitude, lat0, lon0, 100)
    fixes = pd.DataFrame(
        {
            "flight_id": ["a", "a", "b", "b"],
            "segment_id": [0] * 4,
            "t": [0.0, 10.0, 0.0, 10.0],
            "E": e,
            "N": n,
            "z": altitude,
        }
    )
    # Every edge straddles row-group boundaries: carry-over is essential.
    pq.write_table(
        pa.Table.from_pandas(fixes), derived / "fixes.parquet", row_group_size=1
    )

    def origins(db, disc, meta, progress, cancel):
        db.executemany(
            "INSERT INTO flights VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    disc.name,
                    fid,
                    "unused",
                    1000.0,
                    0.0,
                    lat0,
                    lon0,
                    100.0,
                    ix + offset,
                    iy,
                    alt,
                )
                for fid, offset, alt in [("a", 0, 100.0), ("b", 2, 2000.0)]
            ],
        )

    monkeypatch.setattr(thermal_index, "_index_origins", origins)
    path = tmp_path / "index.sqlite3"
    index = thermal_index.build_index([PARAGLIDERS], path=path)
    with connect(path) as db:
        assert db.execute("SELECT COUNT(*) FROM visits").fetchone()[0] == 6
        assert db.execute("SELECT COUNT(*) FROM flight_groups").fetchone()[0] == 4
        assert db.execute("SELECT COUNT(*) FROM metadata").fetchone()[0] == 1
    assert {cell.terrain for cell in index.cells()} == {"Plains", "High mountains"}
    assert all(cell.flights == 2 for cell in index.cells())
    stamp = path.stat().st_mtime_ns
    monkeypatch.setattr(
        thermal_index, "_index_origins", lambda *_: pytest.fail("cache miss")
    )
    assert thermal_index.build_index([PARAGLIDERS], path=path) == index
    assert path.stat().st_mtime_ns == stamp
    assert thermal_index.load_saved_index([PARAGLIDERS], path=path) == index

    # A stopped first census retains origins and committed groups in the data folder.
    resume_path = tmp_path / "resumed.sqlite3"
    checkpoint = resume_path.with_suffix(".building.sqlite3")
    with connect(path) as source, connect(checkpoint) as dest:
        source.backup(dest)
        dest.execute("DELETE FROM metadata")
        dest.execute("DROP TABLE selected_cells")
        dest.execute("DELETE FROM scan_complete")
        dest.execute("DELETE FROM visits")
        dest.execute("DELETE FROM flight_groups WHERE row_group > 0")
        dest.commit()
    # The origin loader above is still set to fail: resume must never call it.
    resumed = thermal_index.build_index([PARAGLIDERS], path=resume_path)
    assert resumed.cells() == index.cells()
    assert not checkpoint.exists()
    with connect(resume_path) as db:
        assert db.execute("SELECT COUNT(*) FROM visits").fetchone()[0] == 6


def test_unknown_clocks_remain_in_census_but_not_calendar_selection(index, monkeypatch):
    with connect(index.path) as db:
        db.execute("UPDATE flights SET start_utc=NULL")
    result = load_plane_data(index, index.cells()[0], 0, 10000, "vilpellet")
    assert index.cells()[0].flights == 3
    assert result.selected == 0
    assert result.unknown_clock == 3


def test_stale_archive_is_not_used_for_row_group_lookups(index, monkeypatch):
    stale = ThermalIndex(index.path, index.disciplines, "old-signature")
    monkeypatch.setattr(thermal_index, "archive_signature", lambda _: "new-signature")
    with pytest.raises(RuntimeError, match="Archive changed"):
        load_plane_data(stale, index.cells()[0], 0, 10000, "vilpellet")


def test_default_cache_lives_on_the_archive_disk(tmp_path, monkeypatch):
    derived = tmp_path / "SSD" / "derived"
    monkeypatch.setattr(
        type(PARAGLIDERS), "config", lambda _: SimpleNamespace(derived_dir=derived)
    )
    monkeypatch.delenv("XC_VIEWER_CACHE_DIR", raising=False)
    assert thermal_index.cache_path([PARAGLIDERS]) == (
        derived / "viewer/thermal-planes/thermal-cells.sqlite3"
    )


def test_config_change_invalidates_climb_products(index, monkeypatch, tmp_path):
    from xc_viewer import thermal_cache

    config = tmp_path / "vilpellet.yaml"
    config.write_text("version: 1")
    monkeypatch.setattr(thermal_cache, "DEFAULT_VILPELLET_CONFIG_PATH", config)
    before = thermal_cache.segmentation_signature(index, "vilpellet")
    config.write_text("version: 2")
    assert thermal_cache.segmentation_signature(index, "vilpellet") != before


def test_offline_export_requires_complete_climbs_and_is_standalone(index, monkeypatch):
    import runpy
    from dataclasses import replace

    from xc_viewer.thermal_cache import ClimbCache
    from xc_viewer.thermal_ranking import rank_cells
    from xc_viewer.thermal_store import ThermalStore, export_store

    cell = index.cells()[0]
    monkeypatch.setattr(
        "xc_viewer.thermal_store.terrain_reference",
        lambda c: {"minimum_m": c.ground_m, "maximum_m": c.ground_m},
    )
    edges = pd.DataFrame(
        {
            "x0": [100.0],
            "y0": [100.0],
            "z0": [300.0],
            "utc0": [1060.0],
            "x1": [200.0],
            "y1": [200.0],
            "z1": [700.0],
            "utc1": [1160.0],
        }
    )
    with ClimbCache(index, "vilpellet") as cache:
        for fid in ("a", "b"):
            cache.put("paragliders", fid, cell, "decoded", edges)
    with pytest.raises(RuntimeError, match="1 flights still need preparation"):
        export_store(index)
    assert not index.path.with_name("thermal-planes.sqlite3").exists()
    with ClimbCache(index, "vilpellet") as cache:
        cache.put("paragliders", "c", cell, "decoded", edges)
    path = export_store(index)
    assert not ThermalStore(path).has_terrain_ranking
    # Publication must reuse the exact DEM used to rank, even when the
    # launch category differs and no repository raster exists for the winner.
    monkeypatch.setattr(
        "xc_viewer.thermal_ranking._cell_terrain",
        lambda *a: {"minimum_m": 350, "maximum_m": 700},
    )
    ranked = rank_cells(index)
    monkeypatch.setattr(
        "xc_viewer.thermal_store.terrain_reference",
        lambda c: pytest.fail("ranking reference must be reused"),
    )
    published = path
    original = published.read_bytes()
    path = export_store(
        ranked, relief=[], destination=published.with_name("staged.sqlite3")
    )
    assert published.read_bytes() == original
    script = Path(__file__).resolve().parents[1] / "scripts/prepare_thermal_planes.py"
    matches = runpy.run_path(str(script))["_matches_prepared_selection"]
    assert matches(path, ranked)
    changed = replace(ranked, selected=(replace(ranked.cells()[0], ground_m=351),))
    assert not matches(path, changed)
    assert not matches(published, ranked)
    index.path.unlink()
    index.path.with_name("thermal-climbs.sqlite3").unlink()
    store = ThermalStore(path)
    assert store.has_terrain_ranking
    assert store.cells() == [
        replace(cell, terrain="Hills", ground_m=350, launch_median_m=cell.ground_m)
    ]
    assert store.terrain_reference(store.cells()[0])["minimum_m"] == 350
    assert store.defaults(cell) == (0, 150)
    result = store.read_plane(cell, 1080, 1090, "vilpellet")
    assert result.cached == result.selected == 3
    assert result.edges.utc0.tolist() == [1060.0] * 3


def test_quality_changes_launch_audit_without_changing_terrain_or_crossing_flights(
    index, tmp_path, monkeypatch
):
    from xc_viewer.thermal_ranking import rank_cells

    monkeypatch.setattr(
        "xc_viewer.thermal_ranking._cell_terrain",
        lambda *a: {"minimum_m": 200, "maximum_m": 250},
    )
    assert rank_cells(index).cells()[0].terrain == "Plains"
    quality = tmp_path / "quality.sqlite3"
    with connect(quality) as db:
        db.execute(
            "CREATE TABLE origins(policy TEXT,discipline TEXT,"
            "flight_id TEXT,status TEXT)"
        )
        db.executemany(
            "INSERT INTO origins VALUES (?,?,?,?)",
            [
                ("p", "paragliders", fid, status)
                for fid, status in [
                    ("a", "altitude_jump"),
                    ("b", "accepted"),
                    ("c", "accepted"),
                ]
            ],
        )
    ranked = rank_cells(index, quality=(quality, "p"))
    cell = ranked.cells()[0]
    assert cell.terrain == "Plains"
    assert cell.ground_m == 200
    assert cell.launch_median_m == 300
    assert cell.launches == 1
    assert cell.flights == len(ranked.flights(cell)) == 3
    assert ranked.path == index.path
    assert ranked.quality_summary["excluded"] == 1


def test_top_three_use_all_crossers_and_break_ties_by_grid_position(index, monkeypatch):
    from xc_viewer.thermal_ranking import rank_cells

    monkeypatch.setattr(
        "xc_viewer.thermal_ranking._cell_terrain",
        lambda *a: {"minimum_m": 200, "maximum_m": 250},
    )
    with connect(index.path) as db:
        for ix in range(2, 5):
            db.execute(
                "INSERT INTO flights VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                ("paragliders", f"start{ix}", "x", 1000, 0, 45, 6, 200, ix, 0, 200),
            )
            for fid in ("a", "b", "c"):
                db.execute(
                    "INSERT INTO visits VALUES (?,?,?,?,?,?,?)",
                    ("paragliders", fid, ix, 0, 0, 100, 900),
                )
    cells = rank_cells(index, per_category=3).cells()
    assert [c.ix for c in cells] == [0, 2, 3]
    assert [c.flights for c in cells] == [3, 3, 3]
