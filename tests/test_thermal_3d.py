"""The 3D cloud uses the correct cell, altitude datum, saved lattice and DEM."""

import hashlib
import json
from dataclasses import replace
from threading import Event
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from xc_viewer.thermal_3d import (
    cell_label,
    load_scene,
    points_every_20m,
    read_aerial,
    read_area_aerial,
    read_area_surface,
    read_neighbour_points,
    read_surface,
)
from xc_viewer.thermal_geometry import ThermalCell
from xc_viewer.thermal_ground import terrain_reference
from xc_viewer.thermal_store import CancelledError, PlaneData, neighbour_frames


@pytest.fixture
def cell():
    return ThermalCell(193, 1309, "High mountains", 100, 5, 1550.25, 1585.25)


@pytest.fixture
def points(cell):
    west, south, east, _ = cell.bounds
    return pd.DataFrame(
        {
            "level": [0, 1, 2, 3, 4, 2, 2],
            "x": [west + 100] * 6 + [east],
            "y": [south + 200] * 7,
            "utc": [100, 110, 120, 130, 140, 201, 150],
            "discipline": ["paragliders", "paragliders", "hang_gliders"]
            + ["paragliders"] * 4,
            "flight_id": ["a"] * 7,
        }
    )


def test_rank_uses_vilpellet_climbs_and_stable_cell_ties(cell):
    cells = [cell, replace(cell, ix=192, flights=1), replace(cell, ix=194, flights=900)]
    store = SimpleNamespace(
        has_climb_ranking=True,
        cells=lambda: cells,
        activity_counts={
            (193, 1309): {"climb_runs": 10},
            (192, 1309): {"climb_runs": 10},
            (194, 1309): {"climb_runs": 5},
        },
    )
    assert cell_label(store, cell) == "High mountains #2"
    assert cell_label(store, cells[1]) == "High mountains #1"
    with pytest.raises(ValueError, match="missing or changed"):
        cell_label(store, replace(cell, ground_m=cell.ground_m + 1))
    store.has_climb_ranking = False
    with pytest.raises(ValueError, match="climb-ranked"):
        cell_label(store, cell)


def test_20m_levels_exclude_irregular_ceiling_and_keep_absolute_altitude(cell, points):
    original = points.copy()
    xyz, flights = points_every_20m(points, cell, 100, 200)
    np.testing.assert_allclose(xyz, [[-2400, -2300, 1550.25], [-2400, -2300, 1570.25]])
    assert flights == 2  # Same ID in distinct disciplines means distinct flights.
    pd.testing.assert_frame_equal(points, original)
    assert xyz.dtype == np.float32
    empty, flights = points_every_20m(points, cell, 300, 400)
    assert empty.shape == (0, 3) and flights == 0


def test_daily_windows_keep_only_their_points(cell, points):
    # Levels 0 and 2 are 20 m planes, at utc 100 and 120.
    xyz, _ = points_every_20m(
        points, cell, 100, 200, windows=np.array([[115.0, 125.0]])
    )
    np.testing.assert_allclose(xyz, [[-2400, -2300, 1570.25]])
    empty, _ = points_every_20m(points, cell, 100, 200, windows=np.empty((0, 2)))
    assert empty.shape == (0, 3)


def test_regular_ceiling_is_kept_and_invalid_level_is_rejected(cell, points):
    exact = replace(cell, max_alt_m=cell.ground_m + 40)
    xyz, _ = points_every_20m(points, exact, 100, 200)
    assert xyz[:, 2].tolist() == [1550.25, 1570.25, 1590.25]
    points.loc[0, "level"] = 999
    with pytest.raises(ValueError, match="Invalid saved"):
        points_every_20m(points, cell, 100, 200)


@pytest.fixture
def raster_store(tmp_path, cell):
    z = (1500 + np.arange(200)[:, None] + np.arange(200)[None, :] / 2).astype("f4")
    cell = replace(cell, ground_m=float(z.min()))
    path = tmp_path / "exploration/terrain/ign-terrain-193-1309.tif"
    path.parent.mkdir(parents=True)
    Image.fromarray(z).save(path)
    reference = {
        "response_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bounds_epsg2154": cell.bounds,
        "raster_bounds_epsg2154": cell.bounds,
    }
    store = SimpleNamespace(
        path=tmp_path / "thermal-planes.sqlite3",
        terrain_reference=lambda _: reference,
    )
    return store, cell, z, path, reference


def test_dem_rows_become_south_to_north_and_surface_covers_exact_cell(raster_store):
    store, cell, original, _, _ = raster_store
    x, y, terrain, _ = read_surface(store, cell)
    assert len(x) == len(y) == 202
    assert (x[0], x[-1], y[0], y[-1]) == (-2500, 2500, -2500, 2500)
    assert x[1] == y[1] == -2487.5
    np.testing.assert_array_equal(terrain[1:-1, 1:-1], original[::-1])
    assert terrain[0, 0] == original[-1, 0]
    assert terrain[-1, -1] == original[0, -1]


def test_changed_dem_and_inconsistent_minimum_are_rejected(raster_store):
    store, cell, _, path, _ = raster_store
    with pytest.raises(ValueError, match="DEM minimum"):
        read_surface(store, replace(cell, ground_m=cell.ground_m + 1))
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="missing or changed"):
        read_surface(store, cell)


@pytest.mark.parametrize(
    "terrain", ["Plains", "Hills", "Low mountains", "High mountains"]
)
def test_scene_reads_selected_cell_vilpellet_and_requested_dates(
    cell, points, monkeypatch, tmp_path, terrain
):
    cell = replace(cell, terrain=terrain)
    calls = []
    plane = PlaneData(pd.DataFrame(), 4, 3, 1, 1, 2, points=points)

    def read(selected, start, end, source, **kw):
        calls.append((selected, start, end, source))
        return plane

    store = SimpleNamespace(
        has_points=True,
        has_climb_ranking=True,
        path=tmp_path / "snapshot",
        cells=lambda: [replace(cell, ix=192, terrain="High mountains"), cell],
        activity_counts={
            (192, 1309): {"climb_runs": 20},
            (193, 1309): {"climb_runs": 10},
        },
        read_plane=read,
        background=lambda *_: None,
    )
    monkeypatch.setattr(
        "xc_viewer.thermal_3d.read_surface",
        lambda *args: (np.arange(2), np.arange(2), np.ones((2, 2)), {}),
    )
    scene = load_scene(store, cell, 100, 200)
    assert calls == [(cell, 100, 200, "vilpellet")]
    assert scene.cell == cell
    assert scene.label == f"{terrain} #{2 if terrain == 'High mountains' else 1}"
    assert len(scene.points) == 2 and scene.contributing_flights == 2
    assert scene.climb_runs == 10 and scene.unavailable_flights == 1
    cancel = Event()
    cancel.set()
    with pytest.raises(CancelledError):
        load_scene(store, cell, 100, 200, cancel=cancel)
    assert len(calls) == 1


def test_orthophoto_requires_matching_crs_and_bounds(cell):
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    reference = {"crs": "EPSG:2154", "extent": cell.bounds}
    store = SimpleNamespace(background=lambda *_: (image, reference))
    actual, metadata = read_aerial(store, cell)
    assert actual is image and metadata is reference
    reference["crs"] = "CRS:84"
    with pytest.raises(ValueError, match="coordinates"):
        read_aerial(store, cell)
    reference.update(crs="EPSG:2154", extent=(0, 0, 5000, 5000))
    with pytest.raises(ValueError, match="coordinates"):
        read_aerial(store, cell)
    store.background = lambda *_: None
    assert read_aerial(store, cell) == (None, None)


def test_10km_dem_keeps_original_pixels_and_neighbours_own_terrain(tmp_path, cell):
    folder = tmp_path / "exploration/terrain"
    folder.mkdir(parents=True)
    coordinates = -2500 + (np.arange(200) + 0.5) * 25
    for tile in [cell, *neighbour_frames(cell)]:
        x = coordinates + (tile.ix - cell.ix) * 5000
        y = coordinates + (tile.iy - cell.iy) * 5000
        z = (1500 + 0.05 * x[None, :] + 0.03 * y[::-1, None]).astype("f4")
        if tile == cell:
            lowest = float(z.min())
        path = folder / f"ign-terrain-{tile.ix}-{tile.iy}.tif"
        Image.fromarray(z).save(path)
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "provenance": {
                        "bounds_epsg2154": tile.bounds,
                        "raster_bounds_epsg2154": tile.bounds,
                        "terrain_file": path.name,
                        "response_sha256": hashlib.sha256(
                            path.read_bytes()
                        ).hexdigest(),
                        "source_url": "https://example.test/dem",
                        "dataset_url": "https://example.test/terrain",
                        "retrieved_utc": "2026-10-05",
                    }
                }
            )
        )
    cell = replace(cell, ground_m=lowest)
    store = SimpleNamespace(
        path=tmp_path / "snapshot",
        cells=lambda: [cell],
        terrain_reference=lambda c: terrain_reference(c, folder),
    )
    x, y, z, ref = read_area_surface(store, cell, lambda: None, lambda _: None)
    assert z.shape == (402, 402)
    assert (x[0], x[-1], y[0], y[-1]) == (-5000, 5000, -5000, 5000)
    assert np.all(np.diff(x[1:-1]) == 25)
    expected = 1500 + 0.05 * x[None, 1:-1] + 0.03 * y[1:-1, None]
    np.testing.assert_allclose(z[1:-1, 1:-1], expected)
    assert len(ref["tiles"]) == 9
    assert len({r["minimum_m"] for r in ref["tiles"]}) > 1
    path.write_bytes(b"corrupt neighbour")
    with pytest.raises(ValueError, match="hash"):
        read_area_surface(store, cell, lambda: None, lambda _: None)


def test_10km_aerial_mosaic_keeps_north_up_crops_and_acquisition_dates(cell):
    def background(tile, kind):
        assert kind == "aerial"
        col = np.arange(4) + (tile.ix - cell.ix + 1) * 4
        row = np.arange(4) + (1 - tile.iy + cell.iy) * 4
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        image[:, :, 0], image[:, :, 1] = col[None, :], row[:, None]
        return image, {
            "crs": "EPSG:2154",
            "extent": tile.bounds,
            "acquisition_dates": [f"2025-06-{tile.ix - cell.ix + 2:02d}"],
        }

    store = SimpleNamespace(background=background)
    image, ref = read_area_aerial(store, cell, lambda: None)
    assert image.shape == (8, 8, 3)
    np.testing.assert_array_equal(image[:, :, 0], np.tile(np.arange(2, 10), (8, 1)))
    np.testing.assert_array_equal(image[:, :, 1], np.tile(np.arange(2, 10), (8, 1)).T)
    assert ref["acquisition_dates"] == ["2025-06-01", "2025-06-02", "2025-06-03"]
    assert len(ref["tiles"]) == 9
    store.background = lambda tile, kind: (
        background(tile, kind) if tile == cell else None
    )
    with pytest.raises(ValueError, match="Missing aerial"):
        read_area_aerial(store, cell, lambda: None)


def test_10km_cloud_uses_central_planes_filters_dates_and_counts_unique_flights(
    cell, points, monkeypatch
):
    west, south, east, north = cell.bounds
    levels = []

    def neighbours(selected, source, level):
        assert selected == cell and source == "vilpellet"
        levels.append(level)
        return pd.DataFrame(
            {
                "x": [west - 2500, east + 2499, east + 2500, west - 1, west - 1],
                "y": [south] * 4 + [north + 2500],
                "utc": [100, 200, 150, 99, 150],
                "flight": [0, 1, 1, 1, 1],
            }
        )

    store = SimpleNamespace(
        has_points=True,
        has_climb_ranking=True,
        cells=lambda: [cell],
        activity_counts={(cell.ix, cell.iy): {"climb_runs": 10}},
        background=lambda *_: None,
        read_plane=lambda *_args, **_kw: PlaneData(
            pd.DataFrame(), 4, 3, 1, 1, 2, points=points.iloc[:6]
        ),
        neighbour_flights=lambda *_: [("paragliders", "a"), ("paragliders", "b")],
        neighbour_points=neighbours,
    )

    def surface(*_):
        return np.arange(2), np.arange(2), np.ones((2, 2)), {}

    monkeypatch.setattr("xc_viewer.thermal_3d.read_area_surface", surface)
    monkeypatch.setattr("xc_viewer.thermal_3d.read_surface", surface)
    scene = load_scene(store, cell, 100, 200, area_km=10)
    assert levels == [0, 2]  # An irregular 35 m ceiling is not a 20 m plane.
    assert scene.area_km == 10 and len(scene.points) == 6
    assert scene.contributing_flights == 3  # Same flight in centre and neighbour.
    np.testing.assert_allclose(scene.points[2:, 0], [-5000, 4999, -5000, 4999])
    np.testing.assert_allclose(scene.points[2:, 2], [1550.25] * 2 + [1570.25] * 2)
    scene = load_scene(store, cell, 100, 200)
    assert scene.area_km == 5 and len(scene.points) == 2
    assert levels == [0, 2]  # The smaller area never reads neighbours.
    store.neighbour_flights = lambda *_: None
    with pytest.raises(ValueError, match="no prepared neighbouring cells"):
        load_scene(store, cell, 100, 200, area_km=10)
    with pytest.raises(ValueError, match="Choose a 5"):
        load_scene(store, cell, 100, 200, area_km=15)


def test_neighbour_read_cancels_between_levels(cell):
    cancel = Event()

    def check_cancel():
        if cancel.is_set():
            raise CancelledError("Cancelled")

    def read(*_):
        cancel.set()
        return pd.DataFrame(columns=["x", "y", "utc", "flight"])

    store = SimpleNamespace(neighbour_flights=lambda *_: [], neighbour_points=read)
    with pytest.raises(CancelledError):
        read_neighbour_points(store, cell, 100, 200, check_cancel, lambda _: None)
