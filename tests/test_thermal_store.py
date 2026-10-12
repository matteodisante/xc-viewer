"""The delivered file works alone and the actual Qt worker loads it end to end."""

import io
import json
import time
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from xc_viewer.sqlite import connect
from xc_viewer.thermal_geometry import ThermalCell
from xc_viewer.thermal_store import ThermalStore, load_store


@pytest.fixture
def store(tmp_path):
    cell = ThermalCell(193, 1304, "Plains", 2, 1, 260, 2000, launch_median_m=185)
    west, south, _, _ = cell.bounds
    blob = io.BytesIO()
    np.savez_compressed(
        blob,
        edges=np.array(
            [[west + 100, south + 100, 400, 100, west + 200, south + 200, 800, 200.0]]
        ),
    )
    path = tmp_path / "thermal-planes.sqlite3"
    with connect(path) as db:
        db.executescript("""
        CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT);
        CREATE TABLE terrain(ix INTEGER,iy INTEGER,metadata TEXT,PRIMARY KEY(ix,iy));
        CREATE TABLE cells(position INTEGER,ix INTEGER,iy INTEGER,
                payload TEXT,day REAL,height REAL);
        CREATE TABLE visitors(ix INTEGER,iy INTEGER,discipline TEXT,flight_id TEXT,
                start REAL,end REAL);
        CREATE TABLE climbs(source TEXT,discipline TEXT,flight_id TEXT,
                ix INTEGER,iy INTEGER,
                status TEXT,edges BLOB);
        """)
        db.executemany(
            "INSERT INTO metadata VALUES (?,?)",
            [
                ("ready", "1"),
                ("version", "3"),
                ("disciplines", '["paragliders"]'),
                ("ground_reference", "ign-dem-cell-min-v1"),
                ("ranking_reference", "ign-dem-cell-max-top-v1"),
            ],
        )
        db.execute(
            "INSERT INTO terrain VALUES (?,?,?)",
            (
                cell.ix,
                cell.iy,
                json.dumps(
                    {
                        "mean_m": 275,
                        "minimum_m": 260,
                        "maximum_m": 290,
                        "grid_m": [25, 25],
                        "samples": 40000,
                        "retrieved_utc": "2026-09-22",
                        "source_url": "https://data.geopf.fr/",
                    }
                ),
            ),
        )
        db.execute(
            "INSERT INTO cells VALUES (0,?,?,?,?,?)",
            (cell.ix, cell.iy, json.dumps(asdict(cell)), 0, 340),
        )
        db.execute(
            "INSERT INTO visitors VALUES (?,?,?,?,?,?)",
            (cell.ix, cell.iy, "paragliders", "a", 100, 200),
        )
        db.execute(
            "INSERT INTO visitors VALUES (?,?,?,?,NULL,NULL)",
            (cell.ix, cell.iy, "paragliders", "unknown"),
        )
        for source in ("vilpellet",):
            db.execute(
                "INSERT INTO climbs VALUES (?,?,?,?,?,?,?)",
                (
                    source,
                    "paragliders",
                    "a",
                    cell.ix,
                    cell.iy,
                    "decoded",
                    blob.getvalue(),
                ),
            )
    path.chmod(0o444)
    return ThermalStore(path)


def test_standalone_read_only_file_needs_no_archive_or_model(store, monkeypatch):
    monkeypatch.setenv("XC_VIEWER_CACHE_DIR", str(store.path.parent))
    before = store.path.stat().st_mtime_ns
    loaded = load_store()
    cell = loaded.cells()[0]
    for source in ("vilpellet",):
        result = loaded.read_plane(cell, 125, 175, source)
        assert result.cached == result.decoded == result.selected == 1
        assert result.unknown_clock == 1
        assert len(result.edges) == 1
        assert loaded.read_plane(cell, 300, 400, source).edges.empty
    assert store.path.stat().st_mtime_ns == before
    assert {p.name for p in store.path.parent.iterdir()} == {"thermal-planes.sqlite3"}


def test_climb_ranking_survives_loading_without_visitor_resort(store):
    from dataclasses import replace

    from xc_viewer.thermal_ground import CLIMB_RANKING

    first = store.cells()[0]
    second = replace(first, ix=first.ix + 1, flights=1)
    store.path.chmod(0o644)
    with connect(store.path) as db:
        db.execute(
            "UPDATE metadata SET value=? WHERE key='ranking_reference'",
            (CLIMB_RANKING,),
        )
        db.execute(
            "CREATE TABLE cell_activity(ix INTEGER,iy INTEGER,"
            "climb_runs INTEGER,climb_flights INTEGER)"
        )
        db.executemany(
            "INSERT INTO cell_activity VALUES (?,?,?,?)",
            [(first.ix, first.iy, 3, 2), (second.ix, second.iy, 10, 1)],
        )
        db.execute(
            "INSERT INTO cells VALUES (1,?,?,?,?,?)",
            (second.ix, second.iy, json.dumps(asdict(second)), 0, 340),
        )
        db.execute(
            "INSERT INTO terrain SELECT ?,?,metadata FROM terrain",
            (second.ix, second.iy),
        )
    loaded = ThermalStore(store.path)
    assert loaded.has_climb_ranking
    assert loaded.cells() == [second, first]
    assert loaded.activity_counts[second.ix, second.iy]["climb_runs"] == 10


@pytest.mark.parametrize("change", [None, "ground", "archive", "method"])
def test_reuse_points_requires_unchanged_geometry_and_method(store, change):
    import shutil

    from xc_viewer.thermal_daily import prepare_daily, reuse_prepared_points

    store.path.chmod(0o644)
    with connect(store.path) as db:
        db.executemany(
            "INSERT INTO metadata VALUES (?,?)",
            [
                ("archive_signature", "archive"),
                ("segmentation_signatures", '{"vilpellet":"vil"}'),
            ],
        )
    prepare_daily(store.path, progress=lambda _: None)
    target = store.path.with_name("changed-selection.sqlite3")
    shutil.copyfile(store.path, target)
    with connect(target) as db:
        db.execute("DELETE FROM plane_points")
        db.execute(
            "DELETE FROM metadata WHERE key IN "
            "('point_lattice','point_build_signature')"
        )
        if change == "ground":
            row = json.loads(db.execute("SELECT payload FROM cells").fetchone()[0])
            row["ground_m"] += 10
            db.execute("UPDATE cells SET payload=?", (json.dumps(row),))
        elif change in ("archive", "method"):
            key = (
                "archive_signature"
                if change == "archive"
                else "segmentation_signatures"
            )
            db.execute("UPDATE metadata SET value='different' WHERE key=?", (key,))
    reuse_prepared_points(target, store.path)
    with connect(target) as db:
        assert db.execute("SELECT count(*) FROM plane_points").fetchone()[0] == (
            1 if change is None else 0
        )
        assert (
            db.execute(
                "SELECT value FROM metadata WHERE key='point_lattice'"
            ).fetchone()
            is None
        )
    if change is None:
        messages = []
        prepare_daily(target, progress=messages.append)
        assert ThermalStore(target).has_points
        assert any("0 products to prepare" in m for m in messages)


def test_real_worker_auto_load_and_reload_button(qapp, store, monkeypatch):
    from xc_viewer.widgets.thermal_plane import ThermalPlane

    monkeypatch.setenv("XC_VIEWER_CACHE_DIR", str(store.path.parent))
    view = ThermalPlane()

    def drain():
        deadline = time.monotonic() + 10
        while view._worker is not None and time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(0.01)
        qapp.processEvents()
        assert view._worker is None, view._status.text()
        assert view._plane is not None, view._status.text()
        assert view._load.isEnabled()
        assert len(view._plane_ax.collections) == 1
        assert "IGN RGE ALTI" in view._provenance.text()
        assert "40,000" in view._provenance.text()
        assert (
            store.terrain_reference(store.cells()[0])["source_url"]
            in view._provenance.text()
        )
        assert view._provenance.openExternalLinks()

    try:
        view.ensure_loaded()
        drain()
        assert view._build.text() == "Reload data"
        view._build.click()
        drain()
        view._load.click()
        drain()
    finally:
        view.shutdown()
        view.close()


def test_saved_relief_preserves_extent_and_never_uses_network(store, monkeypatch):
    from PIL import Image

    store.path.chmod(0o644)
    pixels = np.array(
        [[[30, 40, 50], [60, 70, 80]], [[90, 100, 110], [120, 130, 140]]],
        dtype=np.uint8,
    )
    output = io.BytesIO()
    Image.fromarray(pixels).save(output, format="PNG")
    cell = store.cells()[0]
    info = {"extent": cell.bounds, "epsg": 2154}
    with connect(store.path) as db:
        db.execute("CREATE TABLE relief(key TEXT,metadata TEXT,png BLOB)")
        db.execute(
            "INSERT INTO relief VALUES (?,?,?)",
            (f"{cell.ix}/{cell.iy}", json.dumps(info), output.getvalue()),
        )
    store.path.chmod(0o444)
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *a, **k: pytest.fail("network")
    )
    actual, metadata = store.relief(cell)
    np.testing.assert_array_equal(actual, pixels)
    assert metadata["extent"] == list(cell.bounds)
    assert metadata["epsg"] == 2154
    assert store.relief() is None


def test_prepared_points_read_without_geometry_and_resume(store, monkeypatch):
    from xc_viewer.thermal_daily import prepare_daily

    store.path.chmod(0o644)
    prepare_daily(store.path, progress=lambda _: None)
    prepared = ThermalStore(store.path)
    assert prepared.has_points
    cell = prepared.cells()[0]
    before = store.path.stat().st_size
    prepare_daily(store.path, progress=lambda _: None)
    assert store.path.stat().st_size == before
    store.path.chmod(0o444)
    monkeypatch.setattr(
        "xc_viewer.thermal_geometry.plane_intersections",
        lambda *a, **k: pytest.fail("runtime intersection calculation"),
    )
    for source in ("vilpellet",):
        data = prepared.read_plane(cell, 125, 175, source)
        assert data.edges.empty
        assert data.points is not None
        assert data.points.utc.between(125, 175).all()
        # z=340 AGL means 600 ASL, halfway along the original native edge.
        point = data.points.loc[data.points.level == 34].iloc[0]
        assert point.x == cell.bounds[0] + 150
        assert point.y == cell.bounds[1] + 150
        assert point.utc == 150
        assert data.selected == 1
        native = prepared.read_plane(cell, 125, 175, source, use_edges=True)
        assert native.points is None
        assert len(native.edges) == 1
        assert native.edges.z0.iloc[0] == 400
        assert native.edges.z1.iloc[0] == 800


def test_saved_aerial_dates_and_pixels(store, monkeypatch):
    from PIL import Image

    store.path.chmod(0o644)
    cell = store.cells()[0]
    image = io.BytesIO()
    Image.new("RGB", (4, 4), (20, 40, 60)).save(image, format="PNG")
    dates = ["2022-06-12", "2023-08-01"]
    with connect(store.path) as db:
        db.execute(
            "CREATE TABLE backgrounds(kind TEXT,key TEXT,metadata TEXT,image BLOB)"
        )
        db.execute(
            "INSERT INTO backgrounds VALUES (?,?,?,?)",
            (
                "aerial",
                f"{cell.ix}/{cell.iy}",
                json.dumps({"extent": cell.bounds, "acquisition_dates": dates}),
                image.getvalue(),
            ),
        )
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *a, **k: pytest.fail("network")
    )
    pixels, metadata = store.background(cell, "aerial")
    assert pixels[0, 0].tolist() == [20, 40, 60]
    assert metadata["acquisition_dates"] == dates


def test_terrain_upgrade_rebuilds_points_and_resumes_without_changing_source(
    store, monkeypatch
):
    from xc_viewer import thermal_daily, thermal_ground

    store.path.chmod(0o644)
    thermal_daily.prepare_daily(store.path, progress=lambda _: None)
    original = store.path.read_bytes()
    cell = store.cells()[0]
    reference = {**store.terrain_reference(cell), "minimum_m": 300, "maximum_m": 700}
    monkeypatch.setattr(thermal_ground, "terrain_reference", lambda *a: reference)
    real_prepare = thermal_daily.prepare_daily

    def interrupted(path, **kwargs):
        real_prepare(path, **kwargs)
        raise RuntimeError("interrupted before publication")

    monkeypatch.setattr(thermal_daily, "prepare_daily", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        thermal_ground.upgrade_terrain_store(store.path, progress=lambda _: None)
    assert store.path.read_bytes() == original
    monkeypatch.setattr(thermal_daily, "prepare_daily", real_prepare)
    thermal_ground.upgrade_terrain_store(store.path, progress=lambda _: None)
    upgraded = ThermalStore(store.path)
    new_cell = upgraded.cells()[0]
    assert new_cell.ground_m == 300
    assert new_cell.launch_median_m == 185
    assert new_cell.terrain == "Hills"
    assert upgraded.defaults(new_cell)[1] == 300
    for source in ("vilpellet",):
        points = upgraded.read_plane(new_cell, 125, 175, source).points
        # z=300 above the lowest terrain is 600 ASL, halfway along this edge.
        point = points.loc[points.level == 30].iloc[0]
        assert point.x == cell.bounds[0] + 150
        assert point.utc == 150
    modified = store.path.stat().st_mtime_ns
    thermal_ground.upgrade_terrain_store(store.path, progress=lambda _: None)
    assert store.path.stat().st_mtime_ns == modified


@pytest.mark.parametrize("legacy", [None, "ign-dem-cell-mean-v1"])
def test_other_ground_reference_cannot_be_displayed_as_lowest_terrain(store, legacy):
    store.path.chmod(0o644)
    with connect(store.path) as db:
        db.execute(
            "UPDATE metadata SET value=? WHERE key='ground_reference'", (legacy,)
        )
    with pytest.raises(ValueError, match="older version of the viewer"):
        ThermalStore(store.path)


def _neighbour_census(store, tmp_path, monkeypatch, products):
    """A census whose only neighbour visitor crosses the square east of the cell."""
    import pandas as pd

    cell = store.cells()[0]
    visitors = pd.DataFrame(
        [
            {
                "discipline": "paragliders",
                "flight_id": "east",
                "start_utc": 0.0,
                "trim_start": 0.0,
            },
            {
                "discipline": "paragliders",
                "flight_id": "no-clock",
                "start_utc": np.nan,
                "trim_start": 0.0,
            },
        ]
    )
    census = SimpleNamespace(
        path=tmp_path / "thermal-cells.sqlite3",
        # An uncrossed square comes back from SQL with object-typed columns.
        flights=lambda c: (
            visitors
            if (c.ix, c.iy) == (cell.ix + 1, cell.iy)
            else visitors.iloc[:0].astype(object)
        ),
    )
    with connect(tmp_path / "thermal-climbs.sqlite3") as db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS climbs(cache_key TEXT,discipline TEXT,"
            "flight_id TEXT,ix INTEGER,iy INTEGER,status TEXT,edges BLOB)"
        )
        db.executemany("INSERT INTO climbs VALUES (?,?,?,?,?,?,?)", products)
    monkeypatch.setattr(
        "xc_viewer.thermal_index.load_saved_index", lambda **_: census
    )
    monkeypatch.setattr(
        "xc_viewer.thermal_cache.segmentation_signature",
        lambda index, source: f"key-{source}",
    )
    monkeypatch.setattr(
        "xc_viewer.thermal_prepare.prepare_climbs", lambda *a, **k: None
    )
    return cell


def _edges(*rows):
    blob = io.BytesIO()
    np.savez_compressed(blob, edges=np.array(rows, dtype=float).reshape(-1, 8))
    return blob.getvalue()


def test_neighbours_lie_on_the_cell_planes_and_resume(store, tmp_path, monkeypatch):
    from xc_viewer.thermal_neighbours import prepare_neighbour_points

    store.path.chmod(0o644)
    cell = store.cells()[0]
    west, south, _, _ = cell.bounds
    east = (cell.ix + 1, cell.iy)
    crossing = _edges(
        [west + 5100, south + 100, 400, 100, west + 5200, south + 200, 800, 200]
    )
    products = [
        # Decoded for another square, this flight never visits the east one.
        ("key-vilpellet", "paragliders", "elsewhere", *east, "decoded", _edges()),
        ("key-vilpellet", "paragliders", "east", *east, "decoded", crossing),
    ]
    _neighbour_census(store, tmp_path, monkeypatch, products[:1])
    with pytest.raises(RuntimeError, match="1 flights still need climb products"):
        prepare_neighbour_points(store.path, progress=lambda _: None)
    _neighbour_census(store, tmp_path, monkeypatch, products[1:])
    prepare_neighbour_points(store.path, progress=lambda _: None)
    for source in ("vilpellet",):
        assert store.neighbour_flights(cell, source) == [("paragliders", "east")]
        # z=34 * 10 m above the selected cell's 260 m mean terrain: 600 m ASL,
        # halfway along the edge. The east square's own terrain plays no part.
        points = store.neighbour_points(cell, source, 34)
        assert points.to_numpy().tolist() == [[west + 5150, south + 150, 150, 0]]
        assert store.neighbour_points(cell, source, 60).empty  # 860 m: above the edge
    before = store.path.stat().st_mtime_ns
    prepare_neighbour_points(store.path, progress=lambda _: None)
    assert store.path.stat().st_mtime_ns == before


def test_zoom_out_reads_only_the_ssd(qapp, store, tmp_path, monkeypatch):
    from xc_viewer.thermal_neighbours import prepare_neighbour_points
    from xc_viewer.widgets.thermal_plane import NEIGHBOURS_MISSING, ThermalPlane

    monkeypatch.setenv("XC_VIEWER_CACHE_DIR", str(store.path.parent))
    view = ThermalPlane()

    def drain():
        deadline = time.monotonic() + 10
        while view._worker is not None and time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(0.01)
        qapp.processEvents()
        assert view._worker is None, view._status.text()

    try:
        view.ensure_loaded()
        drain()
        view._zoom_out.click()
        drain()
        assert NEIGHBOURS_MISSING in view._status.text()
        assert view._plane_limits == ((0, 5), (0, 5))

        store.path.chmod(0o644)
        cell = store.cells()[0]
        west, south, _, _ = cell.bounds
        east = (cell.ix + 1, cell.iy)
        crossing = _edges(
            [west + 5100, south + 100, 400, 100, west + 5200, south + 200, 800, 200]
        )
        _neighbour_census(
            store,
            tmp_path,
            monkeypatch,
            [("key-vilpellet", "paragliders", "east", *east, "decoded", crossing)],
        )
        prepare_neighbour_points(store.path, progress=lambda _: None)
        store.path.chmod(0o444)
        monkeypatch.setattr(
            "xc_viewer.thermal_index.load_saved_index",
            lambda **_: pytest.fail("census"),
        )
        monkeypatch.setattr(
            "urllib.request.urlopen", lambda *a, **k: pytest.fail("network")
        )
        view._zoom_out.click()
        drain()
        assert view._plane_limits == ((-2.5, 7.5), (-2.5, 7.5))
        offsets = view._plane_ax.collections[0].get_offsets().tolist()
        assert offsets == [pytest.approx([0.15, 0.15]), pytest.approx([5.15, 0.15])]
    finally:
        view.shutdown()
        view.close()


def test_legacy_launch_categories_are_corrected_without_touching_snapshot(store):
    from dataclasses import replace

    store.path.chmod(0o644)
    original_cell = store.cells()[0]
    cells = [
        replace(
            original_cell,
            ix=196,
            iy=1312,
            terrain="Plains",
            ground_m=1030.758,
            flights=1596,
            launch_median_m=188,
        ),
        replace(
            original_cell,
            ix=191,
            iy=1307,
            terrain="Hills",
            ground_m=875.409,
            flights=3154,
            launch_median_m=776,
        ),
        replace(
            original_cell,
            ix=185,
            iy=1294,
            terrain="Low mountains",
            ground_m=651.897,
            flights=19192,
            launch_median_m=959,
        ),
    ]
    highest = {(196, 1312): 2100.0, (191, 1307): 1400.0, (185, 1294): 1944.2}
    with connect(store.path) as db:
        db.execute("DELETE FROM metadata WHERE key='ranking_reference'")
        db.execute("DELETE FROM cells")
        db.executemany(
            "INSERT INTO cells VALUES (?,?,?,?,?,?)",
            [
                (i, c.ix, c.iy, json.dumps(asdict(c)), 0, 340)
                for i, c in enumerate(cells)
            ],
        )
        db.executemany(
            "INSERT INTO terrain VALUES (?,?,?)",
            [
                (ix, iy, json.dumps({"minimum_m": 0.0, "maximum_m": top}))
                for (ix, iy), top in highest.items()
            ],
        )
    original = store.path.read_bytes()
    saved = ThermalStore(store.path)
    corrected = saved.cells()
    assert [(c.ix, c.iy, c.terrain) for c in corrected] == [
        (191, 1307, "Low mountains"),
        (185, 1294, "High mountains"),
        (196, 1312, "High mountains"),
    ]
    for before in cells:
        after = next(c for c in corrected if (c.ix, c.iy) == (before.ix, before.iy))
        assert replace(after, terrain=before.terrain) == before
    assert not saved.has_terrain_ranking
    assert store.path.read_bytes() == original


def test_viewer_refuses_legacy_subset_instead_of_calling_it_the_terrain_top_three(
    store, monkeypatch
):
    store.path.chmod(0o644)
    with connect(store.path) as db:
        db.execute("DELETE FROM metadata WHERE key='ranking_reference'")
    monkeypatch.setenv("XC_VIEWER_CACHE_DIR", str(store.path.parent))
    with pytest.raises(ValueError, match="older version of the viewer"):
        load_store()
