"""Neighbour exploration includes its own visitors and caches complete flights."""

import io
from threading import Event
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from xc_viewer.thermal_explorer import ThermalExplorer
from xc_viewer.thermal_geometry import ThermalCell, plane_intersections
from xc_viewer.thermal_store import EDGE_COLUMNS, CancelledError


def test_exploration_decodes_whole_flight_before_clipping_and_reuses_cache(
    tmp_path, monkeypatch
):
    cell = ThermalCell(0, 0, "Neighbour", 2, 0, 100, 1000)
    visitors = pd.DataFrame(
        [
            {
                "discipline": "paragliders",
                "flight_id": "outside-central",
                "start_utc": 100.0,
                "trim_start": 0.0,
                "t_min": 0.0,
                "t_max": 100.0,
            },
            {
                "discipline": "paragliders",
                "flight_id": "unknown",
                "start_utc": np.nan,
                "trim_start": 0.0,
                "t_min": 0.0,
                "t_max": 100.0,
            },
        ]
    )
    index = SimpleNamespace(
        path=tmp_path / "thermal-cells.sqlite3", flights=lambda _: visitors
    )
    store = SimpleNamespace(path=tmp_path / "thermal-planes.sqlite3")
    explorer = ThermalExplorer(store)
    explorer._census = index
    explorer._extras[0, 0] = (cell, {"mean_m": 100})
    monkeypatch.setattr(
        "xc_viewer.thermal_cache.segmentation_signature", lambda *a: "current"
    )
    fixes = pd.DataFrame({"t": [0, 50, 100]})
    calls = []

    def frames(index, rows, cancel):
        for row in rows:
            yield row, fixes

    def decode(task):
        row, whole, sources = task
        calls.append(whole.t.tolist())
        blob = io.BytesIO()
        values = pd.DataFrame(
            [[1000, 1000, 400, 100, 2000, 2000, 800, 200]], columns=EDGE_COLUMNS
        )
        np.savez_compressed(blob, edges=values.to_numpy(float))
        return [
            (
                key,
                row["discipline"],
                row["flight_id"],
                c.ix,
                c.iy,
                "decoded",
                blob.getvalue(),
            )
            for key, c in sources["vilpellet"]
        ]

    monkeypatch.setattr("xc_viewer.thermal_explorer._flight_fixes", frames)
    monkeypatch.setattr("xc_viewer.thermal_prepare._decode", decode)
    first = explorer.read_plane(cell, 149, 151, "vilpellet")
    assert calls == [[0, 50, 100]]
    assert first.selected == 1 and first.unknown_clock == 1 and first.cached == 0
    points = plane_intersections(first.edges, cell, 500, 149, 151)
    assert points.x.tolist() == [1500]
    assert points.flight_id.tolist() == ["outside-central"]
    again = explorer.read_plane(cell, 149, 151, "vilpellet")
    assert again.cached == 1 and len(calls) == 1
    cancelled = Event()
    cancelled.set()
    with pytest.raises(CancelledError):
        explorer.read_plane(cell, 149, 151, "vilpellet", cancel=cancelled)


def test_redraw_never_downloads_missing_neighbour_background(tmp_path, monkeypatch):
    explorer = ThermalExplorer(
        SimpleNamespace(path=tmp_path / "thermal-planes.sqlite3")
    )
    cell = ThermalCell(0, 0, "Neighbour", 0, 0, 100, 100)
    explorer._extras[0, 0] = (cell, {})
    monkeypatch.setattr(
        "xc_viewer.thermal_explorer.fetch_image",
        lambda *a: pytest.fail("network during redraw"),
    )
    assert explorer.needs_background(cell, "topography")
    assert explorer.background(cell, "topography") is None
