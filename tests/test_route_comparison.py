"""Scientific selection, archive addressing and bounded IGN terrain behaviour."""

import io
import os
from itertools import pairwise
from threading import Event
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from xc_viewer import route_index as ri
from xc_viewer import route_scene as rs
from xc_viewer.sqlite import connect
from xc_viewer.thermal_index import CancelledError


@pytest.fixture
def archive(tmp_path):
    root = tmp_path / "derived"
    root.mkdir()
    disc = SimpleNamespace(
        name="test",
        config=lambda: SimpleNamespace(derived_dir=root, data_root=tmp_path),
    )
    fixes = pd.DataFrame(
        {
            "flight_id": ["a"] * 7 + ["b"] * 2,
            "segment_id": [0, 0, 0, 1, 1, 1, 1, 0, 0],
            "t": [5, 10, 15, 30, 35, 40, 45, 0, 10],
            "E": [0, 10000, 20000, 50000, 70000, 80000, 100000, 0, 500],
            "N": np.zeros(9),
            "z": np.full(9, 2000),
        }
    )
    for name in ("t", "E", "N", "z"):
        fixes[name] = fixes[name].astype(np.float32)
    pq.write_table(
        pa.Table.from_pandas(fixes), root / "fixes.parquet", row_group_size=3
    )
    pd.DataFrame(
        {
            "flight_id": ["a", "b"],
            "lat0": [44, 44],
            "lon0": [6, 6],
            "alt0": [2000, 2000],
            "n_segments_kept": [2, 1],
            "drop_reason": [None, None],
        }
    ).to_parquet(root / "flights_meta.parquet")
    return disc, root, fixes


def test_census_extrema_across_groups_and_scene_retains_gaps(
    archive, tmp_path, monkeypatch
):
    disc, root, _ = archive
    index = ri.build_index([disc], path=tmp_path / "index.sqlite3")
    rows = index.flights().set_index("flight_id")
    assert rows.loc["a", "duration_s"] == 40
    assert rows.loc["a", "t0"] == 5
    assert rows.loc["a", "t1"] == 45
    with connect(index.path) as db:
        assert db.execute(
            "SELECT row_group FROM parts WHERE flight_id='a'"
        ).fetchall() == [(0,), (1,), (2,)]
        assert db.execute("SELECT typeof(t0) FROM parts LIMIT 1").fetchone() == (
            "real",
        )
    pair = rows.loc["a", ri.PAIR_COLUMNS].to_numpy(dtype=int)
    assert ri.route_pairs(index.flights()).flights.sum() == 1
    monkeypatch.setattr(rs, "load_terrain", lambda *args, **kwargs: None)
    scene = rs.load_scene(index, index.flights(), pair)
    assert scene.total == 1
    assert [len(s) for s in scene.tracks[0]] == [3, 4]
    np.testing.assert_allclose(np.concatenate(scene.tracks[0])[:, 2], 2000)
    assert scene.bounds[0] <= pair[0] * ri.ROUTE_CELL_M
    assert scene.bounds[2] >= (pair[2] + 1) * ri.ROUTE_CELL_M
    assert ri.load_saved_index([disc], path=index.path) == index
    # A changed origin must invalidate endpoint cells as well as geometry reads.
    meta = pd.read_parquet(root / "flights_meta.parquet")
    meta.loc[0, "lon0"] = 7
    meta.to_parquet(root / "flights_meta.parquet")
    assert ri.load_saved_index([disc], path=index.path) is None
    with pytest.raises(ValueError, match="changed"):
        rs.load_scene(index, index.flights(), pair)


def test_clean_path_and_time_metrics_exclude_gaps(archive, tmp_path):
    disc, _, _ = archive
    index = ri.build_index([disc], path=tmp_path / "index.sqlite3")
    selected = index.flights().query("flight_id == 'a'")
    tracks, metrics = rs.read_tracks(index, selected)
    assert [len(segment) for segment in tracks[0]] == [3, 4]
    row = metrics.iloc[0]
    assert row.retained_s == 25
    assert row.gap_s == 15
    assert row.clean_segments == 2
    # 20 km + 50 km of supported edges; the 30 km jump is never added.
    assert row.path_km == pytest.approx(70, abs=0.5)
    assert row.net_km == pytest.approx(100, abs=0.5)


def test_resumes_census_after_cancellation_and_rejects_incomplete_archive(
    archive, tmp_path
):
    disc, root, _ = archive
    event = Event()
    path = tmp_path / "index.sqlite3"

    def cancel_first(message):
        if "census" in message:
            event.set()

    with pytest.raises(CancelledError):
        ri.build_index([disc], path=path, progress=cancel_first, cancel=event)
    assert not path.exists()
    progress = []
    index = ri.build_index([disc], path=path, progress=progress.append)
    assert len(index.flights()) == 2
    assert not any("census 1/" in message for message in progress)
    (root / ".run_incomplete").touch()
    with pytest.raises(ValueError, match="incomplete"):
        ri.load_saved_index([disc], path=path)


def test_scene_dates_match_retained_endpoints_after_trimming(
    archive, tmp_path, monkeypatch
):
    from xc_viewer.core.naming import igc_path

    disc, root, _ = archive
    cfg = SimpleNamespace(
        derived_dir=root,
        data_root=tmp_path,
        catalog_path=tmp_path / "catalog.csv",
        igc_dir=tmp_path / "igc",
    )
    disc.config = lambda: cfg
    meta = pd.read_parquet(root / "flights_meta.parquet")
    meta["ground_phase_start_s"] = 120.0
    meta.to_parquet(root / "flights_meta.parquet")
    pd.DataFrame(
        {
            "flight_id": ["a", "b"],
            "season_year": [2024, 2024],
            "date": ["2024-06-15", "2024-06-15"],
        }
    ).to_csv(cfg.catalog_path, index=False)
    path = igc_path(cfg.igc_dir, 2024, "2024-06-15", "a")
    path.parent.mkdir(parents=True)
    path.write_text("HFDTE150624\nB1200004432469N00542796EA010000010000\n")
    index = ri.build_index([disc], path=tmp_path / "index.sqlite3")
    flights = index.flights()
    pair = flights.set_index("flight_id").loc["a", ri.PAIR_COLUMNS].to_numpy(dtype=int)
    monkeypatch.setattr(rs, "load_terrain", lambda *args, **kwargs: None)
    scene = rs.load_scene(index, flights, pair)
    row = scene.selected.iloc[0]
    assert row.departure_utc == pd.Timestamp("2024-06-15T12:02:05Z").timestamp()
    assert row.arrival_utc == pd.Timestamp("2024-06-15T12:02:45Z").timestamp()
    assert row.arrival_utc - row.departure_utc == row.duration_s == 40


def cohort(count=301):
    rows = pd.DataFrame(
        {
            "flight_id": [f"{i:04}" for i in range(count)],
            "discipline": ["para" if i % 2 else "hang" for i in range(count)],
            "duration_s": np.arange(count) + 1000,
            "start_ix": 90,
            "start_iy": 650,
            "end_ix": 100,
            "end_iy": 650,
        }
    )
    return rows.sample(frac=1, random_state=42)


def test_scene_filters_all_departures_before_duration_sampling(
    archive, tmp_path, monkeypatch
):
    from datetime import date, time

    from xc_viewer.route_times import DepartureWindow

    disc, root, _ = archive
    ids = [str(i) for i in range(12)]
    frames = [
        pd.DataFrame(
            {
                "flight_id": fid,
                "segment_id": 0,
                "t": [0.0, 1000.0 + i],
                "E": [0.0, 100000.0],
                "N": 0.0,
                "z": 2000.0,
            }
        )
        for i, fid in enumerate(ids)
    ]
    pq.write_table(
        pa.Table.from_pandas(pd.concat(frames, ignore_index=True)),
        root / "fixes.parquet",
        row_group_size=2,
    )
    pd.DataFrame(
        {
            "flight_id": ids,
            "lat0": 44.0,
            "lon0": 6.0,
            "alt0": 2000.0,
            "n_segments_kept": 1,
            "drop_reason": None,
        }
    ).to_parquet(root / "flights_meta.parquet")
    index = ri.build_index([disc], path=tmp_path / "routes.sqlite3")
    flights = index.flights()
    pair = flights.iloc[0][ri.PAIR_COLUMNS].to_numpy(dtype=int)
    monkeypatch.setattr(ri, "MAX_FLIGHTS", 10)
    sampled, _ = ri.select_flights(flights, pair)
    assert not {"5", "6"} & set(sampled.flight_id)
    base = pd.Timestamp("2024-06-15T10:00:00Z").timestamp()

    def dates(candidates, *_args, **_kwargs):
        assert len(candidates) == 12  # Must not recover dates only for sampled tracks.
        result = candidates.assign(departure_utc=base + 86400)
        result.loc[result.flight_id.isin(["5", "6"]), "departure_utc"] = base + 300
        result.loc[result.flight_id == "0", "departure_utc"] = np.nan
        result["arrival_utc"] = result.departure_utc + result.duration_s
        return result

    monkeypatch.setattr(rs, "with_flight_times", dates)
    monkeypatch.setattr(rs, "load_terrain", lambda *a, **kw: None)
    monkeypatch.setattr(rs, "load_density", lambda *a, **kw: None)
    window = DepartureWindow(date(2024, 6, 15), time(12), 30)
    scene = rs.load_scene(index, flights, pair, departure_window=window)
    assert scene.selected.flight_id.tolist() == ["5", "6"]
    assert scene.total == len(scene.tracks) == 2
    assert scene.selected["rank"].tolist() == [1, 2]
    assert len(scene.departure_cohort) == 12 and scene.departure_window == window
    monkeypatch.setattr(
        rs.pq, "ParquetFile", lambda *_: pytest.fail("empty window read tracks")
    )
    with pytest.raises(ValueError, match=r"No departures.*1 flights have unavailable"):
        rs.load_scene(
            index,
            flights,
            pair,
            departure_window=DepartureWindow(date(2024, 6, 15), time(13), 30),
        )


def test_limit_preserves_true_extremes_direction_and_disciplines():
    flights = cohort()
    pair = (90, 650, 100, 650)
    selected, total = ri.select_flights(flights, pair)
    assert total == 301 and len(selected) == 300
    assert selected.flight_id.nunique() == 300
    assert selected.iloc[:5].flight_id.tolist() == [f"{i:04}" for i in range(5)]
    assert selected.iloc[-5:].flight_id.tolist() == [f"{i:04}" for i in range(296, 301)]
    assert (selected.speed_group == "fast").sum() == 5
    assert (selected.speed_group == "slow").sum() == 5
    pd.testing.assert_frame_equal(
        selected, ri.select_flights(flights.iloc[::-1], pair)[0]
    )
    assert ri.select_flights(flights, (100, 650, 90, 650))[1] == 0
    subset, total = ri.select_flights(flights, pair, "para")
    assert total == 150 and set(subset.discipline) == {"para"}
    pairs = ri.route_pairs(flights, 100, 100)
    assert pairs.distance_km.tolist() == [100]
    assert pairs.flights.tolist() == [301]
    assert len(ri.route_pairs(flights, require_both=True)) == 1
    assert ri.route_pairs(
        flights.loc[flights.discipline == "para"], require_both=True
    ).empty
    with pytest.raises(ValueError):
        ri.route_pairs(flights, 110, 90)


def test_small_cohort_and_ties_are_honest():
    flights = cohort(7)
    flights["duration_s"] = 1000
    selected, total = ri.select_flights(flights, (90, 650, 100, 650))
    assert total == len(selected) == 7
    assert selected.speed_group.isin(["fast", "both"]).sum() == 5
    assert selected.speed_group.isin(["slow", "both"]).sum() == 5
    assert (selected.speed_group == "both").sum() == 3
    pd.testing.assert_frame_equal(
        selected, ri.select_flights(flights.iloc[::-1], (90, 650, 100, 650))[0]
    )


def tiff(values):
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(values, dtype=np.float32)).save(buffer, format="TIFF")
    return buffer.getvalue()


def test_dem_orientation_holes_and_bounded_overview():
    bounds = (900000, 6400000, 902000, 6403000)
    raw = tiff([[1000, 1100], [500, -99999], [100, 200]])
    x, y, z, grid, coverage = rs.decode_terrain(raw, bounds, (2, 3))
    assert x[[0, -1]].tolist() == [900000, 902000]
    assert y[[0, -1]].tolist() == [6400000, 6403000]
    assert z[1, 1] == 100 and z[-2, 1] == 1000
    assert np.isnan(z[2, 2]) and coverage == pytest.approx(5 / 6)
    assert grid == (1000, 1000)
    vertices, faces = rs.terrain_mesh((x, y, z, {}), np.array(bounds[:2]))
    assert np.isfinite(vertices).all()
    assert not np.isin(faces, np.flatnonzero(~np.isfinite(z))).any()
    _, size = rs.terrain_request((0, 0, 500000, 130000))
    assert max(size) <= 600
    with pytest.raises(ValueError, match="coverage"):
        rs.decode_terrain(tiff([[-99999, -99999], [-99999, -99999]]), bounds, (2, 2))


def test_terrain_cache_is_reusable_offline_and_checks_hash(tmp_path, monkeypatch):
    bounds = (900000, 6400000, 900200, 6400200)
    raw = tiff([[10, 20], [30, 40]])
    monkeypatch.setattr(rs, "urlopen", lambda *args, **kwargs: io.BytesIO(raw))
    first = rs.load_terrain(bounds, tmp_path)
    monkeypatch.setattr(
        rs, "urlopen", lambda *args, **kwargs: pytest.fail("offline read")
    )
    second = rs.load_terrain(bounds, tmp_path)
    np.testing.assert_array_equal(first[2], second[2])
    path = next(tmp_path.glob("*.npz"))
    with np.load(path) as cached:
        reference = str(cached["reference"])
    np.savez_compressed(
        path, raw=np.array([1, 2, 3], dtype=np.uint8), reference=reference
    )
    with pytest.raises(ValueError, match="provenance"):
        rs.load_terrain(bounds, tmp_path)


def test_aerial_cache_has_matching_bounds_north_up_pixels_and_integrity(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(rs, "AERIAL_MAX_SIDE", 4)
    bounds = (900000, 6400000, 900200, 6400400)
    url, size = rs.aerial_request(bounds)
    assert size == (2, 4)
    assert "EPSG%3A2154" in url and rs.AERIAL_LAYER in url
    pixels = np.zeros((4, 2, 3), dtype=np.uint8)
    pixels[0, :, 0] = 255  # north: red
    pixels[-1, :, 2] = 255  # south: blue
    payload = io.BytesIO()
    Image.fromarray(pixels).save(payload, format="PNG")
    monkeypatch.setattr(
        rs, "urlopen", lambda *args, **kwargs: io.BytesIO(payload.getvalue())
    )
    first, ref = rs.load_aerial(bounds, tmp_path)
    np.testing.assert_array_equal(first, pixels)
    assert ref["bounds_epsg2154"] == bounds
    assert ref["grid_m"] == (100, 100)
    monkeypatch.setattr(
        rs, "urlopen", lambda *args, **kwargs: pytest.fail("offline image")
    )
    cached, _ = rs.load_aerial(bounds, tmp_path)
    np.testing.assert_array_equal(cached, pixels)
    path = next(tmp_path.glob("*.npz"))
    with np.load(path) as stored:
        reference = str(stored["reference"])
    np.savez_compressed(path, raw=np.array([0], dtype=np.uint8), reference=reference)
    with pytest.raises(ValueError, match="provenance"):
        rs.load_aerial(bounds, tmp_path)
    with pytest.raises(ValueError, match="dimensions"):
        rs.decode_aerial(payload.getvalue(), (4, 2))


def test_distance_presets_update_interval_and_choose_actual_pairs(qapp, monkeypatch):
    from xc_viewer.widgets.route_comparison import RouteComparison

    view = RouteComparison()
    calls = []
    monkeypatch.setattr(
        view, "_load_pair", lambda: calls.append(view._pairs.currentData())
    )
    frames = []
    for distance in ri.DISTANCE_PRESETS_KM:
        frame = cohort(12)
        frame["end_ix"] = frame.start_ix + distance // 10
        frames.append(frame)
    view._flights = pd.concat(frames, ignore_index=True)
    try:
        for distance in (50, 100, 200, 300):
            view._distance.setCurrentIndex(view._distance.findData(distance))
            assert (view._minimum.value(), view._maximum.value()) == (
                distance - 10,
                distance + 10,
            )
            assert view._pairs.count() == 1
            assert calls[-1] == (90, 650, 90 + distance // 10, 650)
    finally:
        view.close()


def test_aerial_mesh_switch_preserves_geometry_camera_and_terrain_holes(qapp):
    pytest.importorskip("pyqtgraph.opengl")
    from xc_viewer.widgets.route_comparison import RouteComparison

    selected, total = ri.select_flights(cohort(12), (90, 650, 100, 650))
    bounds = (900000, 6500000, 1010000, 6510000)
    terrain = (
        np.linspace(bounds[0], bounds[2], 4),
        np.linspace(bounds[1], bounds[3], 4),
        np.array(
            [
                [100, 200, 200, 100],
                [200, np.nan, 200, 100],
                [100, 200, 300, 100],
                [100, 100, 100, 100],
            ]
        ),
        {
            "grid_m": (100, 100),
            "coverage": 15 / 16,
            "dataset_url": "https://www.data.gouv.fr/datasets/rge-alti-r",
        },
    )
    scene = rs.RouteScene(
        selected,
        total,
        (90, 650, 100, 650),
        [
            [np.array([[-50000, 0, 1000], [50000, 0, 2000]], dtype=np.float32)]
            for _ in range(12)
        ],
        bounds,
        np.array([955000, 6505000]),
        terrain=terrain,
        aerial=np.zeros((4, 8, 3), dtype=np.uint8),
        aerial_reference={"grid_m": (100, 100), "dataset_url": rs.AERIAL_DATASET_URL},
    )
    view = RouteComparison()
    try:
        view.set_scene(scene)
        surface = view._surface
        mesh = surface.opts["meshdata"]
        camera = view._view.cameraParams()
        assert surface._bounds == [-55000, -5000, 55000, 5000]
        assert len(mesh.faces()) < 18  # missing elevation stays a hole in both modes
        for mode in (1, 0, 2, 1):
            view._surface_mode.setCurrentIndex(mode)
            assert surface._aerial == (mode == 1)
            assert surface.opts["meshdata"] is mesh
            assert view._view.cameraParams() == camera
            assert ("BD ORTHO" in view._source.text()) == (mode == 1)
            if mode == 2:
                colors = mesh.vertexColors()
                np.testing.assert_allclose(colors[:, 0], colors[:, 1])
                np.testing.assert_allclose(colors[:, 1], colors[:, 2])
        view._terrain.setChecked(False)
        assert not surface.visible()
        view._surface_mode.setCurrentIndex(0)
        assert not surface.visible()
    finally:
        view.shutdown()
        view.close()


def test_widget_speed_filters_keep_camera_and_clear_stale_scene(qapp):
    pytest.importorskip("pyqtgraph.opengl")
    from xc_viewer.widgets.route_comparison import RouteComparison

    selected, total = ri.select_flights(cohort(12), (90, 650, 100, 650))
    selected["departure_utc"] = pd.Timestamp("2024-06-15T21:59:55Z").timestamp()
    selected["arrival_utc"] = selected.departure_utc + selected.duration_s
    selected.loc[1, ["departure_utc", "arrival_utc"]] = np.nan
    scene = rs.RouteScene(
        selected,
        total,
        (90, 650, 100, 650),
        [
            [np.array([[-50000, 0, 1000], [50000, 0, 2000]], dtype=np.float32)]
            for _ in range(12)
        ],
        (900000, 6500000, 1010000, 6510000),
        np.array([955000, 6505000]),
    )
    widget = RouteComparison()
    try:
        widget.set_scene(scene)
        assert widget._table.columnCount() == 7
        assert widget._table.item(0, 3).text() == "15/06/2024 23:59:55 CEST"
        assert widget._table.item(0, 4).text() == "16/06/2024 00:16:35 CEST"
        assert widget._table.item(0, 5).text() == "0:16:40"
        assert widget._table.item(1, 3).text() == "Unavailable"
        assert widget._table.item(1, 4).text() == "Unavailable"
        assert "Displayed departures: All dates" in widget._summary.text()
        before = widget._view.cameraParams()
        for mode, count in (("fast", 5), ("slow", 5), ("extremes", 10), ("all", 12)):
            widget._mode.setCurrentIndex(widget._mode.findData(mode))
            assert sum(lines[0].visible() for lines in widget._lines) == count
            assert sum(not widget._table.isRowHidden(i) for i in range(12)) == count
            assert widget._view.cameraParams() == before
        widget._table.selectRow(3)
        assert widget._lines[3][0].width == widget._line_width.value() * 1.75
        positions = widget._lines[3][0].pos.copy()
        widget._line_width.setValue(6)
        assert widget._lines[3][0].width == 10.5
        widget._line_width.setValue(0.1)
        assert widget._lines[3][0].width == pytest.approx(0.175)
        assert widget._lines[3][0].outline < 0.1
        np.testing.assert_array_equal(widget._lines[3][0].pos, positions)
        assert widget._view.cameraParams() == before
        np.testing.assert_array_equal(
            widget._annotations.endpoints[3], scene.tracks[3][0][[0, -1]]
        )
        assert widget._annotations.selected == {3}
        widget._pair_changed()
        assert widget._scene is None and widget._table.rowCount() == 0
    finally:
        widget.shutdown()
        widget.close()


def _vilpellet_products(root, fixes):
    """Every fix of the archive fixture inside one saved climb run per segment."""
    folder = root / "segmentation/vilpellet"
    (folder / "model").mkdir(parents=True)
    runs = (
        fixes.groupby(["flight_id", "segment_id"], sort=False)
        .t.agg(t_start="min", t_end="max", n_fixes="size")
        .reset_index()
    )
    runs["phase"] = "climb"
    runs.to_parquet(folder / "phase_segments.parquet")
    pd.DataFrame({"flight_id": ["a", "b"], "n_native_fixes": [7, 2]}).to_parquet(
        folder / "phase_coverage.parquet"
    )
    (folder / "model/parameters.json").write_text("{}")
    return folder


def test_all_flight_density_preparation_preserves_gaps_and_detects_stale_sources(
    archive, tmp_path, monkeypatch
):
    import os

    from xc_viewer import route_density as rd
    from xc_viewer.thermal_time import load_grids

    disc, root, fixes = archive
    folder = _vilpellet_products(root, fixes)
    # Batches end inside flights: the held-back flight must still be whole.
    monkeypatch.setattr(rd, "BATCH_ROWS", 2)
    path = rd.prepare_density([disc], path=tmp_path / "density.npz")
    grids, report = load_grids(path)
    assert report["complete"]
    assert report["version"] == rd.VERSION
    # 25 supported seconds from flight a, plus 10 from b. The 15-second
    # segment boundary in a and the boundary between flights never contribute.
    assert grids["archive/vilpellet"].hours * 3600 == pytest.approx(35)
    with monkeypatch.context() as m:
        m.setattr(rd.pq, "ParquetFile", lambda *_: pytest.fail("cache reuse"))
        assert rd.prepare_density([disc], path=path) == path
    with pytest.raises(ValueError, match="exceeds"):
        rd.raster_window(grids["archive/vilpellet"], (-600000, 6000000, 0, 6100000))
    meta = pd.read_parquet(root / "flights_meta.parquet")
    meta.loc[0, "lon0"] += 1
    meta.to_parquet(root / "flights_meta.parquet")
    with pytest.raises(ValueError, match="changed"):
        rd.load_density((900000, 6400000, 1010000, 6500000), [disc], path=path)
    old = (root / "fixes.parquet").stat().st_mtime_ns - 10**9
    os.utime(folder / "phase_segments.parquet", ns=(old, old))
    with pytest.raises(ValueError, match="predate"):
        rd.prepare_density([disc], path=path)


def test_density_preparation_rejects_incomplete_vilpellet_coverage(archive, tmp_path):
    from xc_viewer import route_density as rd

    disc, root, fixes = archive
    folder = _vilpellet_products(root, fixes)
    pd.DataFrame({"flight_id": ["a"], "n_native_fixes": [7]}).to_parquet(
        folder / "phase_coverage.parquet"
    )
    with pytest.raises(ValueError, match="Incomplete Vilpellet coverage"):
        rd.prepare_density([disc], path=tmp_path / "density.npz")


def test_density_coarsening_conserves_time_and_geographic_footprint():
    from xc_viewer.route_density import raster_window
    from xc_viewer.thermal_time import TimeGrid

    grid = TimeGrid((0, 0, 1000, 1000))
    # The same thermal visited by two pilots contributes both residence times.
    grid.add(
        [[25, 25], [25, 25], [25, 525]],
        [[75, 25], [75, 25], [975, 525]],
        [36, 36, 3600],
    )
    grid.flush()
    for step in (50, 250, 500, 1000):
        raster = raster_window(grid, grid.bounds, pixel_m=step)
        assert raster.hours == pytest.approx(3672 / 3600)
        assert raster.values.sum() * (step / 1000) ** 2 == pytest.approx(raster.hours)
        np.testing.assert_array_equal(raster.bounds, grid.bounds)
    assert raster_window(grid, grid.bounds, pixel_m=1000).values[0, 0] == pytest.approx(
        1.02
    )


def test_thermal_colour_uses_one_dem_mesh_and_ignores_route_speed_filters(qapp):
    from xc_viewer.route_density import DensityAtlas, raster_window
    from xc_viewer.thermal_time import TimeGrid
    from xc_viewer.widgets.route_comparison import RouteComparison

    selected, total = ri.select_flights(cohort(12), (90, 650, 100, 650))
    bounds = (900000, 6500000, 1010000, 6510000)
    grid = TimeGrid(bounds)
    grid.add([[950025, 6500025]], [[960025, 6500025]], [3600])
    grid.flush()
    density = DensityAtlas(
        {step: raster_window(grid, bounds, pixel_m=step) for step in (250, 1000)}
    )
    scene = rs.RouteScene(
        selected,
        total,
        (90, 650, 100, 650),
        [
            [np.array([[-50000, 0, 1000], [50000, 0, 2000]], dtype=np.float32)]
            for _ in range(12)
        ],
        bounds,
        np.array([955000, 6505000]),
        terrain=(
            np.linspace(bounds[0], bounds[2], 4),
            np.linspace(bounds[1], bounds[3], 4),
            np.arange(16).reshape(4, 4) * 100,
            {"grid_m": (100, 100), "coverage": 1, "dataset_url": "https://ign.fr"},
        ),
        density=density,
    )
    view = RouteComparison()
    try:
        view.set_scene(scene)
        surface = view._surface
        mesh = surface.opts["meshdata"]
        image = surface._density_image
        assert view._density.mesh is surface
        np.testing.assert_array_equal(mesh.vertexes()[:, 2], scene.terrain[2].ravel())
        camera = view._view.cameraParams()
        for mode in ("fast", "slow", "all"):
            view._mode.setCurrentIndex(view._mode.findData(mode))
            assert surface._density_image is image
        view._terrain.setChecked(False)
        assert (
            not surface.terrain_visible
            and surface.density_visible
            and surface.visible()
        )
        view._density.enabled.setChecked(False)
        assert not surface.visible()
        view._density.enabled.setChecked(True)
        view._density.resolution.setCurrentIndex(1)
        assert surface._density_image.shape != image.shape
        assert surface.opts["meshdata"] is mesh
        assert view._view.cameraParams() == camera
        view._locator.scope.setCurrentIndex(1)
        assert view._locator.ax.get_xlim() == (-180, 180)
        view._locator.expand()
        assert view._locator._dialog.locator._scene is scene
        view._clear_scene()
        assert view._locator._dialog.locator._scene is None
    finally:
        view.shutdown()
        view.close()


@pytest.mark.skipif(
    os.environ.get("XC_VIEWER_NATIVE_OPENGL") != "1",
    reason="Requires a native display and GPU to check terrain/density compositing",
)
def test_native_thermal_colour_is_not_occluded_by_terrain(qapp):
    """Opaque thermal colour stays pixel-identical as the background is toggled."""
    from PyQt6.QtGui import QImage, QVector3D
    from PyQt6.QtTest import QTest

    from xc_viewer.widgets.terrain_drape import DrapedTerrain
    from xc_viewer.widgets.thermal_3d import TerrainView

    view = TerrainView()
    view.resize(700, 500)
    x = y = np.linspace(-500, 500, 20)
    xx, yy = np.meshgrid(x, y)
    z = 500 + 400 * np.cos(xx / 400) * np.sin(yy / 400)
    vertices, faces = rs.terrain_mesh((x, y, z, {}), np.zeros(2))
    heat = np.full((16, 16, 4), [220, 25, 70, 255], dtype=np.uint8)
    mesh = DrapedTerrain(
        image=np.full((16, 16, 3), 200, dtype=np.uint8),
        bounds=(-500, -500, 500, 500),
        vertexes=vertices,
        faces=faces,
        vertexColors=np.ones((len(vertices), 4), dtype=np.float32),
        smooth=True,
        computeNormals=False,
    )
    mesh.set_density(heat, (-500, -500, 500, 500))
    mesh.density_opacity = 1
    view.addItem(mesh)
    view.opts["center"] = QVector3D(0, 0, 500)
    view.setCameraPosition(distance=3000, elevation=45, azimuth=-90)

    def pixels():
        qapp.processEvents()
        image = view.grabFramebuffer().convertToFormat(QImage.Format.Format_RGBA8888)
        return (
            np.frombuffer(image.bits().asstring(image.sizeInBytes()), np.uint8)
            .copy()
            .reshape(image.height(), image.width(), 4)
        )

    try:
        view.show()
        assert QTest.qWaitForWindowExposed(view)
        for aerial in (False, True):
            mesh.set_aerial(aerial)
            for angle in (0, 35):
                view.orbit(angle, 0)
                mesh.set_terrain_visible(True)
                with_ground = pixels()
                mesh.set_terrain_visible(False)
                without_ground = pixels()
                np.testing.assert_array_equal(with_ground, without_ground)
                assert (
                    np.count_nonzero(
                        (with_ground[:, :, 0] > 180) & (with_ground[:, :, 1] < 60)
                    )
                    > 1000
                )
        assert mesh._density_texture is not None
        view.makeCurrent()
        mesh.release_texture()
        view.doneCurrent()
        assert mesh._density_texture is None and mesh._texture is None
    finally:
        view.close()


@pytest.mark.skipif(
    os.environ.get("XC_VIEWER_NATIVE_OPENGL") != "1",
    reason="Requires a native core-profile GPU to verify actual stroke widths",
)
def test_native_subpixel_strokes_and_endpoint_annotations(qapp):
    from PyQt6.QtGui import QImage
    from PyQt6.QtTest import QTest

    from xc_viewer.widgets.route_annotations import RouteAnnotations
    from xc_viewer.widgets.route_lines import RouteLine
    from xc_viewer.widgets.thermal_3d import TerrainView

    view = TerrainView()
    view.resize(700, 500)
    view.setCameraPosition(distance=2000, elevation=90, azimuth=-90)
    positions = np.array([[-400, 0, 0], [0, 0, 0], [400, 0, 0]], np.float32)
    line = RouteLine(
        pos=positions, color=(0, 0.9, 1, 1), glOptions="translucent", width=2.5
    )
    view.addItem(line)

    def pixels():
        qapp.processEvents()
        image = view.grabFramebuffer().convertToFormat(QImage.Format.Format_RGBA8888)
        return (
            np.frombuffer(image.bits().asstring(image.sizeInBytes()), np.uint8)
            .copy()
            .reshape(image.height(), image.width(), 4)
        )

    try:
        view.show()
        assert QTest.qWaitForWindowExposed(view)
        masses = []
        for width in (0.1, 0.5, 2.5, 6):
            line.outline = min(1, 0.4 * width)
            line.setData(width=width)
            frame = pixels()
            column = frame[:, frame.shape[1] // 2 + 30, :3].astype(float)
            masses.append(abs(column - column[0]).sum())
        assert all(a < b for a, b in pairwise(masses))
        assert masses[-1] > 4 * masses[0]
        # The coloured fill must survive overlap between consecutive short edges.
        middle = frame[frame.shape[0] // 2, frame.shape[1] // 2, :3]
        assert middle[1] > 150 and middle[2] > 180 and middle[0] < 80
        camera = view.cameraParams()
        annotations = RouteAnnotations([[positions]], positions[[0, -1]])
        view.addItem(annotations)
        labelled = pixels()
        assert np.count_nonzero(labelled != frame) > 1000
        assert view.cameraParams() == camera
        np.testing.assert_array_equal(line.pos, positions)
    finally:
        view.close()
