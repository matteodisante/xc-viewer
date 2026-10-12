"""Launch cohorts must be disjoint, day-aware and independent of destinations."""

from threading import Event
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from xc_viewer import group_flights as gf
from xc_viewer import route_index as ri
from xc_viewer import route_scene as rs
from xc_viewer.sqlite import connect
from xc_viewer.thermal_index import CancelledError


def catalog(minutes, *, cells=None, days=None):
    base = pd.Timestamp("2024-07-01T10:00:00Z").timestamp()
    times = base + np.asarray(minutes, dtype=float) * 60
    return pd.DataFrame(
        {
            "discipline": "para",
            "flight_id": [str(i) for i in range(len(times))],
            "start_ix": 190 if cells is None else cells,
            "start_iy": 1280,
            "end_ix": np.arange(len(times)) + 195,
            "end_iy": 1300,
            "departure_utc": times,
            "arrival_utc": times + 1000,
            "day": "2024-07-01" if days is None else days,
            "duration_s": 1000.0,
            "t0": 0.0,
            "t1": 1000.0,
            "lat0": 44.0,
            "lon0": 6.0,
            "alt0": 1000.0,
            "segments": 1,
        }
    )


def test_windows_do_not_overlap_or_chain_and_members_are_chronological():
    data = catalog([0, 20, 40, 60, 70]).sample(frac=1, random_state=42)
    ranking = gf.rank_groups(data, window_minutes=30)
    assert ranking.groups.flights.tolist() == [2, 3]
    members = [
        ranking.members(g).flight_id.tolist() for _, g in ranking.groups.iterrows()
    ]
    assert members == [["0", "1"], ["2", "3", "4"]]
    assert ranking.cells.grouped_flights.tolist() == [5]
    assert np.all(
        ranking.groups.last_departure_utc - ranking.groups.departure_utc <= 1800
    )
    # Changing minimum size filters the same windows instead of reassigning launches.
    higher = gf.rank_groups(data, minimum=3)
    assert higher.groups.flights.tolist() == [3]
    assert higher.members(higher.groups.iloc[0]).flight_id.tolist() == ["2", "3", "4"]
    shorter = gf.rank_groups(data, window_minutes=20)
    assert shorter.groups.flights.tolist() == [2, 2]
    assert shorter.cells.grouped_flights.tolist() == [4]
    assert gf.rank_groups(data, minimum=100).cells.empty


def test_cells_merge_on_the_same_grid_and_ranking_criteria_differ():
    data = catalog([0, 5, 10, 100, 105, 110, 0, 5, 10, 15], cells=[190] * 6 + [191] * 4)
    small = gf.rank_groups(data)
    assert gf.ordered_cells(small).ix.tolist() == [190, 191]
    assert gf.ordered_cells(small, "largest").ix.tolist() == [191, 190]
    merged = gf.rank_groups(data, cell_m=10000)
    assert merged.cells.ix.tolist() == [95]
    assert merged.cells.grouped_flights.tolist() == [10]
    assert merged.cells.largest.tolist() == [7]
    # Arrival cells and journey duration are deliberately irrelevant.
    changed = data.assign(end_ix=-5000, end_iy=10000, duration_s=1)
    pd.testing.assert_frame_equal(gf.rank_groups(changed).groups, small.groups)


def test_separate_days_disciplines_unknown_times_and_year_filter():
    data = catalog(
        [0, 5, 6, 7, np.nan],
        days=["2024-07-01", "2024-07-01", "2024-07-02", "2023-07-01", None],
    )
    data.loc[1, "discipline"] = "hang"
    all_groups = gf.rank_groups(data)
    assert all_groups.unknown_clocks == 1
    assert all_groups.groups.flights.tolist() == [2]
    assert gf.rank_groups(data, discipline="para").groups.empty
    assert gf.rank_groups(data, year=2023).groups.empty
    assert len(gf.rank_groups(data, year=2024).flights) == 3
    # Two occurrences of the same Paris wall-clock hour remain one hour apart.
    data = catalog([0, 0])
    data["departure_utc"] = [
        pd.Timestamp(t).timestamp() for t in ("2024-10-27T00:15Z", "2024-10-27T01:15Z")
    ]
    data["arrival_utc"] = data.departure_utc + 1000
    data["day"] = "2024-10-27"
    assert gf.rank_groups(data, window_minutes=30).groups.empty
    assert gf.rank_groups(data, window_minutes=60).groups.flights.tolist() == [2]


def test_invalid_definitions_and_cancellation():
    for options in ({"cell_m": 7000}, {"minimum": 1}, {"window_minutes": 0}):
        with pytest.raises(ValueError):
            gf.rank_groups(catalog([0, 2]), **options)
    stop = Event()
    stop.set()
    with pytest.raises(CancelledError):
        gf.rank_groups(catalog([0, 2]), cancel=stop)


def test_catalog_uses_trimmed_clean_endpoints_cached_utc_and_atomic_publication(
    tmp_path, monkeypatch
):
    index = SimpleNamespace(
        path=tmp_path / "routes.sqlite3",
        signature="v1",
        disciplines=(),
        verify=lambda: None,
    )
    monkeypatch.setattr(gf, "_signature", lambda idx: idx.signature)
    rows = catalog([0, 1])[gf.COLUMNS].copy()
    rows["t0"], rows["t1"] = [5, 10], [1005, 1010]
    monkeypatch.setattr(
        gf, "_endpoint_rows", lambda *a, **k: rows.itertuples(index=False, name=None)
    )
    clock = tmp_path / "clocks.sqlite3"
    origins = pd.DataFrame(
        {
            "discipline": ["para"] * 2,
            "flight_id": ["0", "1"],
            "start_utc": [1700000000.0, 1700000000.0],
            "trim_start": [300.0, 360.0],
        }
    )
    with connect(clock) as db:
        origins.to_sql("flights", db, index=False)
    monkeypatch.setattr(
        gf.thermal_index, "load_saved_index", lambda _: SimpleNamespace(path=clock)
    )
    monkeypatch.setattr(
        gf, "with_flight_times", lambda *a, **k: pytest.fail("raw clock reread")
    )
    flights = gf.load_catalog(index)
    assert flights.departure_utc.tolist() == [1700000305.0, 1700000370.0]
    assert flights.arrival_utc.tolist() == [1700001305.0, 1700001370.0]
    monkeypatch.setattr(
        gf, "_endpoint_rows", lambda *a, **k: pytest.fail("cached catalog rebuild")
    )
    pd.testing.assert_frame_equal(gf.load_catalog(index), flights)
    saved = index.path.with_name("group-flight-catalog.parquet").read_bytes()
    index.signature = "v2"
    monkeypatch.setattr(
        gf, "_endpoint_rows", lambda *a, **k: rows.itertuples(index=False, name=None)
    )
    stop = Event()
    stop.set()
    with pytest.raises(CancelledError):
        gf.load_catalog(index, cancel=stop)
    assert index.path.with_name("group-flight-catalog.parquet").read_bytes() == saved


def test_departure_only_census_keeps_destinations_outside_france(tmp_path):
    meta = pd.DataFrame(
        {
            "flight_id": ["out", "in"],
            "lat0": 44.0,
            "lon0": 6.0,
            "alt0": 1000.0,
            "n_segments_kept": 1,
            "drop_reason": None,
        }
    )
    meta.to_parquet(tmp_path / "flights_meta.parquet")
    disc = SimpleNamespace(
        name="para", config=lambda: SimpleNamespace(derived_dir=tmp_path)
    )
    fixes = pd.DataFrame(
        {
            "flight_id": ["out", "out", "in", "in"],
            "t": [0.0, 1000.0, 0.0, 1000.0],
            "E": [0.0, 600000.0, 600000.0, 0.0],
            "N": 0.0,
            "z": 1000.0,
        }
    )
    with connect(tmp_path / "index.sqlite3") as db:
        db.execute(
            "CREATE TABLE parts(discipline,flight_id,row_group,t0,e0,n0,z0,t1,e1,n1,z1)"
        )
        db.executemany(
            "INSERT INTO parts VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            ri.reduce_group(fixes, "para", 0),
        )
        assert ri._endpoint_rows(db, [disc], lambda _: None, None) == []
        groups = ri._endpoint_rows(
            db, [disc], lambda _: None, None, departure_only=True
        )
        assert [row[1] for row in groups] == ["out"]


def test_scene_loads_every_member_and_independent_destinations(monkeypatch):
    ranking = gf.rank_groups(catalog(np.arange(305) / 20), window_minutes=30)
    selected = ranking.members(ranking.groups.iloc[0])
    tracks = [
        [np.array([[900000.0, 6400000.0, 1000.0], [910000.0 + i, 6420000.0, 2000.0]])]
        for i in range(305)
    ]
    metrics = pd.DataFrame(
        {
            "path_km": 23.0,
            "net_km": 22.0,
            "retained_s": 1000.0,
            "gap_s": 0.0,
            "clean_segments": 1,
        },
        index=selected.index,
    )
    monkeypatch.setattr(gf, "read_tracks", lambda *a, **k: (tracks, metrics))
    monkeypatch.setattr(gf, "load_background", lambda scene, *a, **k: scene)
    scene = gf.load_group_scene(None, ranking, ranking.groups.iloc[0])
    assert len(scene.selected) == len(scene.tracks) == 305  # No Routes sampling cap.
    assert scene.pair == (190, 1280) and scene.cell_m == 5000
    assert scene.selected.departure_utc.is_monotonic_increasing
    assert scene.bounds[2] >= 191 * 5000


def small_scene():
    """Two real-looking displayed rows for the shared renderer's UI contract."""
    selected = catalog([0, 3]).assign(
        rank=[1, 2],
        departure_delay_s=[0.0, 180.0],
        path_km=[10.0, 20.0],
        net_km=[8.0, 12.0],
        gap_s=[0.0, 30.0],
        clean_segments=1,
    )
    return rs.RouteScene(
        selected,
        2,
        (190, 1280),
        [
            [
                np.array(
                    [[0.0, 0.0, 1000.0], [1000.0, 2000.0 + i * 100, 2000.0]], np.float32
                )
            ]
            for i in range(2)
        ],
        (950000, 6400000, 955000, 6405000),
        np.array([952500.0, 6402500.0]),
        cell_m=5000,
    )


def test_group_renderer_retains_order_selects_rows_and_has_only_departure_cell(qapp):
    from xc_viewer.widgets.group_scene import GroupScene

    view = GroupScene()
    try:
        view.set_scene(small_scene())
        assert view._table.rowCount() == 2
        assert view._table.item(0, 2).text() == "0"
        assert view._table.item(1, 5).text() == "3.0"
        assert len(view._annotations.anchors) == 1
        assert view._queries.isHidden()
        camera = view._view.cameraParams()
        view._table.selectRow(1)
        view._mode.setCurrentIndex(1)
        assert [lines[0].visible() for lines in view._lines] == [False, True]
        assert view._view.cameraParams() == camera
        view._clear_scene()
        assert view._table.rowCount() == 0
    finally:
        view.shutdown()
        view.close()


def test_group_tab_recomputes_definition_and_clears_stale_scene(qapp, monkeypatch):
    from time import monotonic

    from PyQt6.QtTest import QTest

    from xc_viewer.widgets import group_flights as widget_module

    data = catalog([0, 20, 40, 60, 70])
    index = SimpleNamespace(disciplines=[SimpleNamespace(name="para")])
    monkeypatch.setattr(
        widget_module.GroupFlights,
        "_read_catalog",
        staticmethod(lambda *a, **k: (index, data)),
    )

    def load_scene(index, ranking, group, **kwargs):
        scene = small_scene()
        scene.selected = ranking.members(group).assign(
            rank=np.arange(1, int(group.flights) + 1),
            departure_delay_s=0,
            path_km=10,
            net_km=8,
            gap_s=0,
            clean_segments=1,
        )
        scene.tracks = [scene.tracks[0] for _ in range(int(group.flights))]
        scene.pair = (int(group.ix), int(group.iy))
        scene.cell_m = ranking.cell_m
        return scene

    monkeypatch.setattr(widget_module, "load_group_scene", load_scene)
    tab = widget_module.GroupFlights()

    def wait_scene():
        deadline = monotonic() + 10
        while monotonic() < deadline:
            QTest.qWait(10)
            if (
                tab._worker is None
                and not tab._timer.isActive()
                and tab._viewer._scene is not None
            ):
                return
        pytest.fail(tab._status.text())

    try:
        assert tab._worker is None and tab._catalog is None  # Lazy tab.
        tab.ensure_loaded()
        wait_scene()
        assert tab._ranking.cells.grouped_flights.tolist() == [5]
        assert tab._viewer._scene.selected.flight_id.tolist() == ["2", "3", "4"]
        tab._minutes.setValue(20)
        assert tab._viewer._scene is None and tab._groups.count() == 0
        assert not tab._export.isEnabled()
        wait_scene()
        assert tab._ranking.cells.grouped_flights.tolist() == [4]
        assert tab._viewer._scene.selected.flight_id.tolist() == ["0", "1"]
        assert tab._viewer._table.rowCount() == 2
        tab._cell_size.setCurrentIndex(1)
        wait_scene()
        assert tab._viewer._scene.cell_m == 10000
        assert tab._viewer._scene.pair == (95, 640)
    finally:
        tab.shutdown()
        tab.close()
