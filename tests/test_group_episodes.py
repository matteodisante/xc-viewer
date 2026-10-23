"""Group episodes: the tables of the group identification, their ranking and replay."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from xc_viewer import group_episodes as ge
from xc_viewer import route_scene as rs

DAY, OTHER = "2020-06-01", "2021-07-01"
T0 = 1_591_000_000


def write_tables(folder, distance=True):
    """Day one: flights 0 and 1 meet at +600 s, flight 2 joins at +1800 s, all part at
    +3000 s: one episode of two configurations. Another day: flights 3 and 4."""
    folder.mkdir(exist_ok=True)
    pd.DataFrame(
        {
            "flight": [0, 1, 2, 3, 4],
            "discipline": ["para", "para", "hang", "para", "para"],
            "flight_id": ["a", "b", "c", "d", "e"],
            "utc0": float(T0),
            "takeoff": float(T0),
            "departure": T0 + 300.0,
            "landing": T0 + 4000.0,
            "date": [DAY, DAY, DAY, OTHER, OTHER],
        }
    ).to_parquet(folder / "flights.parquet")
    pd.DataFrame(
        {
            "date": [DAY, DAY, OTHER],
            "conf_id": [0, 1, 2],
            "episode_id": [0, 0, 1],
            "t_start": [T0 + 600, T0 + 1800, T0 + 1000],
            "t_end": [T0 + 1800, T0 + 3000, T0 + 5000],
            "n": [2, 3, 2],
            "members": [[0, 1], [0, 1, 2], [3, 4]],
        }
    ).to_parquet(folder / "configurations.parquet")
    pd.DataFrame(
        {
            "episode_id": [0, 1],
            "date": [DAY, OTHER],
            "t_start": [T0 + 600, T0 + 1000],
            "t_end": [T0 + 3000, T0 + 5000],
            "n_max": [3, 2],
            "n_config": [2, 1],
            "n_flights": [3, 2],
            "block": [0, 1],
        }
    ).to_parquet(folder / "episodes.parquet")
    pd.DataFrame(
        {
            "date": [DAY, DAY, DAY, OTHER],
            "a": [0, 1, 0, 3],
            "b": [1, 2, 2, 4],
            "t_on": [T0 + 600, T0 + 1800, T0 + 2400, T0 + 1000],
            "t_off": [T0 + 3000] * 3 + [T0 + 5000],
        }
    ).to_parquet(folder / "pair_intervals.parquet")
    rows = []
    for f, joins in ((0, 600), (1, 600), (2, 1800)):
        rows += [(f, T0, T0 + 300, "L", -1, -1), (f, T0 + 300, T0 + joins, "S", -1, -1)]
        if joins == 600:
            rows.append((f, T0 + 600, T0 + 1800, "G", 0, 0))
        rows += [
            (f, T0 + 1800, T0 + 3000, "G", 1, 0),
            (f, T0 + 3000, T0 + 4000, "S", -1, -1),
        ]
    pd.DataFrame(
        rows, columns=["flight", "t_start", "t_end", "state", "conf_id", "episode_id"]
    ).to_parquet(folder / "membership.parquet")
    if distance:
        pd.DataFrame(
            {"episode_id": [0, 1], "distance_m": [12000.0, 30000.0], "flight": [0, 3]}
        ).to_parquet(folder / "episode_distance.parquet")
    (folder / "calibration.json").write_text(
        json.dumps(
            {"tau_star": 5370, "params": {"eps_h": 500.0, "eps_z": 300.0, "delta": 300}}
        )
    )
    return folder


def test_tables_are_read_with_viewer_names_and_a_missing_file_is_named(tmp_path):
    tables = ge.load_tables(write_tables(tmp_path / "out"))
    assert tables.flights.discipline.tolist()[:3] == [
        "paragliders",
        "paragliders",
        "hang gliders",
    ]
    assert tables.episodes.duration_s.tolist() == [2400, 4000]
    assert tables.has_distance and tables.episodes.distance_m.tolist() == [
        12000.0,
        30000.0,
    ]
    assert tables.episodes.disciplines[0] == {"paragliders", "hang gliders"}
    assert tables.calibration["tau_star"] == 5370
    assert not ge.load_tables(
        write_tables(tmp_path / "plain", distance=False)
    ).has_distance
    (tmp_path / "out" / "membership.parquet").unlink()
    with pytest.raises(FileNotFoundError, match=r"membership\.parquet"):
        ge.load_tables(tmp_path / "out")


def test_ranking_orders_and_filters(tmp_path):
    tables = ge.load_tables(write_tables(tmp_path / "out"))
    order = lambda **kw: ge.rank_episodes(tables, **kw).episode_id.tolist()  # noqa: E731
    assert order(order="flights") == [0, 1]
    assert order(order="largest") == [0, 1]
    assert order(order="duration") == [1, 0]
    assert order(order="distance") == [1, 0]
    assert order(discipline="hang gliders") == [0]
    assert order(year=2021) == [1]
    assert order(minimum=3) == [0]


def test_state_of_an_episode_at_every_instant(tmp_path):
    ep = ge.episode(ge.load_tables(write_tables(tmp_path / "out")), 0)
    assert ep.members.flight.tolist() == [0, 1, 2]
    assert ep.window == (T0 + 600 - ge.MARGIN_S, T0 + 3000 + ge.MARGIN_S)
    assert ep.configurations_at(T0 + 1000).conf_id.tolist() == [0]
    assert ep.configurations_at(T0 + 2000).conf_id.tolist() == [1]
    assert ep.configurations_at(T0 + 3000).empty
    assert ep.edges_at(T0 + 2000).tolist() == [[0, 1], [1, 2]]
    assert len(ep.edges_at(T0 + 2500)) == 3
    assert ep.episodes_at(T0 + 2000).episode_id.tolist() == [
        0
    ]  # the other day is not listed
    assert [ep.state_of(2, T0 + t) for t in (100, 1000, 2000, 3500, 5000)] == [
        "L",
        "S",
        "G",
        "S",
        None,
    ]


def test_members_are_identified_from_the_catalog(tmp_path, monkeypatch):
    ep = ge.episode(ge.load_tables(write_tables(tmp_path / "out")), 0)
    catalog = pd.DataFrame(
        {
            "flight_id": ["b", "a"],
            "pilot": ["Bob", "Alice"],
            "club": "C",
            "wing": "W",
            "takeoff": "Site",
            "landing": "Field",
            "distance_km": [51.0, 42.0],
        }
    )

    def rows(disc, ids):
        if disc.name == "hang gliders":
            raise FileNotFoundError
        return catalog[catalog.flight_id.isin(ids)]

    monkeypatch.setattr(ge.catalog_index, "catalog_rows", rows)
    para, hang = (SimpleNamespace(name=n) for n in ("paragliders", "hang gliders"))
    members = ge.identify(ep.members, [para, hang])
    assert members.flight.tolist() == [0, 1, 2]  # order kept
    assert members.pilot.tolist()[:2] == ["Alice", "Bob"]
    assert pd.isna(members.pilot[2])  # no catalog for hang gliders
    assert members.takeoff.tolist() == ep.members.takeoff.tolist()  # UTC, not the site
    assert members.takeoff_site.tolist()[:2] == ["Site", "Site"]


def test_timed_track_interpolates_within_pieces_and_trails_end_at_the_instant():
    first = np.array([[0, 0, 0, 100], [10, 100, 0, 110], [20, 200, 0, 120]], float)
    second = np.array([[100, 0, 500, 300], [110, 0, 600, 300]], float)
    track = ge.TimedTrack([first, second])
    np.testing.assert_allclose(track.at(5), [50, 0, 105])
    assert track.at(50) is None  # in the gap
    np.testing.assert_allclose(track.at(105), [0, 550, 300])
    trail = track.trail(15, 10)
    assert len(trail) == 1
    np.testing.assert_allclose(trail[0], [[100, 0, 110], [150, 0, 115]])
    assert len(track.trail(105, 100)) == 2


def replay(tmp_path):
    ep = ge.episode(ge.load_tables(write_tables(tmp_path / "out")), 0)
    t = np.arange(T0, T0 + 4001, 10.0)
    for k, f in enumerate(ep.members.flight):
        ep.tracks[f] = ge.TimedTrack(
            [
                np.column_stack(
                    [
                        t,
                        (t - T0) * 8.0,
                        np.full(len(t), 200.0 * k),
                        np.full(len(t), 1500.0),
                    ]
                )
            ]
        )
    lines = []
    for f in ep.members.flight:
        m = ep.membership[(ep.membership.flight == f) & (ep.membership.state == "G")]
        pieces, ep.inside[f] = ge.split_inside(
            ep.tracks[f].pieces[0], m[["t_start", "t_end"]].to_numpy(float)
        )
        lines.append([p[:, 1:].astype(np.float32) for p in pieces])
    scene = rs.RouteScene(
        ep.members, 3, (), lines, (-1000, -1000, 33000, 1000), np.zeros(2)
    )
    return scene, ep


def drawn(curves):
    """Points drawn by plot curves."""
    return sum(0 if c.xData is None else len(c.xData) for c in curves)


def test_scene_colours_configurations_pilots_and_contacts(qapp, tmp_path):
    pytest.importorskip("pyqtgraph.opengl")
    from xc_viewer.widgets.group_scene import (
        EPISODE,
        OTHER,
        PALETTE,
        PILOTS,
        SOLO,
        TRACK,
        EpisodeScene,
        _rgba,
    )

    scene, ep = replay(tmp_path)
    ep.members["pilot"] = ["Alice", "$2A$07$FFVL0123", np.nan]
    view = EpisodeScene()
    try:
        view.set_episode(scene, ep)
        assert view._table.rowCount() == 3 and view._slider.isEnabled()
        assert not view._lower.isHidden()  # Timeline, Reeb graph and Members tabs
        view._who.click()
        who = view._who_window.table
        assert [who.item(k, 1).text() for k in range(3)] == [
            "Alice",
            "(anonymised)",
            "",
        ]
        assert who.item(2, 2).text() == "c"
        assert view._annotations.anchors.shape == (0, 3)  # no cells to mark
        assert ep.inside[0] == [False, True, False]  # before, in and after the episode
        np.testing.assert_allclose(view._lines[0][1].color, _rgba(PILOTS[0], 0.9))
        np.testing.assert_allclose(view._lines[0][0].color, _rgba(PILOTS[0], 0.35))
        view._track_by.setCurrentIndex(view._track_by.findData("episode"))
        np.testing.assert_allclose(view._lines[0][1].color, _rgba(EPISODE, 0.9))
        np.testing.assert_allclose(view._lines[0][0].color, _rgba(TRACK, 0.35))
        low = ep.window[0]  # the replay opens at the start of the slider
        assert view._now == low and view._cursor.value() == low
        assert view._slider.value() == 0
        view.seek(T0 + 1000)
        np.testing.assert_allclose(
            view._markers.color, [_rgba(c) for c in (PALETTE[0], PALETTE[0], SOLO)]
        )
        assert view._markers.size.tolist() == [13, 13, 7]  # large in a group
        assert view._edges[PALETTE[0]].visible()  # in the colour of the configuration
        assert drawn([view._graph_edges[PALETTE[0]]]) == 2  # members 1 and 2
        assert "1 pair in contact" in view._graph.plotItem.titleLabel.text
        assert view._numbers_item.texts == ["1", "2", "3"]
        text = view._now_text.text()
        assert "1 configuration" in text and "[1, 2]" in text
        view._colour_by.setCurrentIndex(view._colour_by.findData("pilot"))
        np.testing.assert_allclose(view._markers.color, [_rgba(c) for c in PILOTS[:3]])
        assert view._edges["#ffffff"].visible()
        view._colour_by.setCurrentIndex(view._colour_by.findData("episode"))
        np.testing.assert_allclose(
            view._markers.color, [_rgba(c) for c in (EPISODE, EPISODE, SOLO)]
        )
        view._colour_by.setCurrentIndex(view._colour_by.findData("configuration"))
        view.seek(T0 + 2500)
        np.testing.assert_allclose(view._markers.color, [_rgba(PALETTE[1])] * 3)
        assert len(view._edges[PALETTE[1]].pos) == 6  # three pairs in contact
        assert drawn(view._graph_edges.values()) == 6
        assert view._reeb_future.isVisible()  # live: the future is hidden
        assert view._reeb_future.rect().left() == T0 + 2500
        view._reeb_mode.setCurrentIndex(view._reeb_mode.findData("whole"))
        assert not view._reeb_future.isVisible()
        view._reeb_cursor.setValue(T0 + 2000)  # dragging the Reeb graph moves it too
        assert view._now == T0 + 2000 and view._cursor.value() == T0 + 2000
        view._extent.setCurrentIndex(view._extent.findData("flown"))
        track = ep.tracks[0]
        assert not view._lines[0][1].visible() and view._flown[0].visible()
        np.testing.assert_allclose(view._flown[0].pos[-1], track.at(T0 + 2000))
        view.seek(T0 + 2010)  # the flown track waits, the trail bridges the gap
        np.testing.assert_allclose(view._flown[0].pos[-1], track.at(T0 + 2000))
        np.testing.assert_allclose(view._trails[0].pos[-1], track.at(T0 + 2010))
        view.seek(T0 + 1500)  # back in time: rebuilt
        np.testing.assert_allclose(view._flown[0].pos[-1], track.at(T0 + 1500))
        view._extent.setCurrentIndex(view._extent.findData("trail"))
        assert not view._lines[0][1].visible() and not view._flown[0].visible()
        assert view._trails[0].visible()  # the last minutes, nothing before them
        view._extent.setCurrentIndex(view._extent.findData("whole"))
        assert view._lines[0][1].visible() and not view._flown[0].visible()
        view._cursor.setValue(T0 + 1000)  # dragging the timeline moves the instant
        assert (
            view._now == T0 + 1000 and view._slider.value() == (1000 - 600 + 600) // 5
        )
        view._follow.setChecked(True)
        centre = view._view.opts["center"]
        assert (centre.x(), centre.y()) == pytest.approx((8000.0, 200.0))  # mean of 3
        ep.membership = pd.concat(
            [
                ep.membership,
                pd.DataFrame(
                    [(2, T0 + 3000, T0 + 3500, "G", 9, 9)],
                    columns=ep.membership.columns[:6],
                ),
            ],
            ignore_index=True,
        )
        view.seek(T0 + 3200)  # member 3 in a configuration of another episode
        np.testing.assert_allclose(view._markers.color[2], _rgba(OTHER))
        assert "In another episode:</b> 3" in view._now_text.text()
        unchecked = view._table.item(2, 0).checkState().Unchecked
        view._table.item(2, 0).setCheckState(unchecked)
        assert view._markers.color[2][3] == 0  # hidden member
        assert "3" not in view._numbers_item.texts
        view.seek(T0 + 10**6)
        assert view._now == ep.window[1]
        view._play.setChecked(True)
        assert view._now == ep.window[0]  # playing from the end starts again
        view._advance()
        assert view._timer.isActive()  # each frame arms the next
        view.seek(ep.window[1] - 1)
        view._advance()
        assert not view._play.isChecked() and not view._timer.isActive()  # the end
        view._clear_scene()
        assert view._markers is None and not view._slider.isEnabled()
        assert who.rowCount() == 0 and view._lower.isHidden()
        view.seek(T0 + 1000)  # nothing to draw, nothing fails
    finally:
        view.shutdown()
        view.close()


def test_a_click_picks_the_nearest_visible_dot_within_reach():
    from xc_viewer.widgets.group_scene import nearest_member

    points = np.array([[10.0, 10.0], [50.0, 10.0], [52.0, 12.0]])
    visible = np.array([True, True, False])
    assert nearest_member(points, visible, 12, 11) == 0
    assert nearest_member(points, visible, 51, 12) == 1  # the hidden dot is skipped
    assert nearest_member(points, visible, 300, 300) is None  # out of reach
    assert nearest_member(np.zeros((0, 2)), np.zeros(0, bool), 0, 0) is None


def test_the_view_reports_a_click_but_not_a_drag(qapp):
    from PyQt6.QtCore import QPoint, Qt
    from PyQt6.QtTest import QTest

    from xc_viewer.widgets.thermal_3d import TerrainView

    view = TerrainView()
    clicks = []
    view.clicked.connect(lambda x, y: clicks.append((x, y)))
    try:
        left = Qt.MouseButton.LeftButton
        QTest.mouseClick(view, left, pos=QPoint(10, 12))
        assert clicks == [(10.0, 12.0)]
        QTest.mousePress(view, left, pos=QPoint(10, 12))
        QTest.mouseMove(view, QPoint(80, 12))
        QTest.mouseRelease(view, left, pos=QPoint(80, 12))
        assert len(clicks) == 1  # the pointer moved: a drag, not a click
    finally:
        view.close()


def test_clicking_a_pilot_follows_its_configuration(qapp, tmp_path, monkeypatch):
    from xc_viewer.widgets import group_scene as gs

    scene, ep = replay(tmp_path)
    view = gs.EpisodeScene()
    try:
        view.set_episode(scene, ep)
        view.seek(T0 + 1000)  # flights 0 and 1 in a configuration, flight 2 alone
        assert set(view._located) == {0, 1, 2}
        monkeypatch.setattr(  # member k at the pixel (100 k, 0)
            gs,
            "screen_points",
            lambda item, xyz: (
                np.column_stack([100.0 * np.arange(len(xyz)), np.zeros(len(xyz))]),
                np.ones(len(xyz), bool),
            ),
        )
        view._picked(200, 3)  # member 3 is alone
        assert view._focus == () and "Member 3" in view._focus_text.text()
        view._picked(500, 500)  # nothing there
        assert view._focus == ()
        view._picked(105, 3)  # member 2, in the configuration of members 1 and 2
        assert view._focus == (0, 1) and view._focus_clear.isEnabled()
        assert "1, 2" in view._focus_text.text()
        centre = view._view.opts["center"]
        both = np.mean([ep.tracks[f].at(T0 + 1000) for f in (0, 1)], axis=0)
        np.testing.assert_allclose([centre.x(), centre.y()], both[:2])
        assert view._markers.color[2][3] > 0  # all members stay drawn...
        view._focus_only.setChecked(True)  # ...unless asked
        assert view._markers.color[2][3] == 0
        assert view._numbers_item.texts == ["1", "2"]
        view.seek(T0 + 2500)  # the configuration is over, its members are followed
        assert view._focus == (0, 1) and view._markers.color[2][3] == 0
        view._focus_clear.click()
        assert view._focus == () and not view._focus_only.isChecked()
        assert view._markers.color[2][3] > 0
        view._picked(105, 3)
        view._clear_scene()  # a new episode starts without focus
        assert view._focus == () and not view._focus_clear.isEnabled()
    finally:
        view.shutdown()
        view.close()


def test_the_replay_holds_while_a_mouse_button_is_down(qapp, monkeypatch):
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QGuiApplication

    from xc_viewer.widgets.group_scene import EpisodeScene

    view = EpisodeScene()
    try:
        down = Qt.MouseButton.LeftButton
        monkeypatch.setattr(QGuiApplication, "mouseButtons", lambda: down)
        view._advance()
        assert view._timer.isActive()  # no frame drawn, the next one is armed
        view._timer.stop()
        up = Qt.MouseButton.NoButton
        monkeypatch.setattr(QGuiApplication, "mouseButtons", lambda: up)
        view._advance()
        assert not view._timer.isActive()  # no episode: the replay stops
    finally:
        view.shutdown()
        view.close()


def test_reeb_graph_continues_lanes_through_splits_and_merges():
    # 0 splits into 1 (three members) and 2 (one, with 5 joining); 3 forms apart and
    # merges with 1 into 4.
    confs = pd.DataFrame(
        {
            "conf_id": [0, 1, 2, 3, 4],
            "t_start": [0, 100, 100, 150, 200],
            "t_end": [100, 200, 300, 200, 400],
            "n": [4, 3, 2, 2, 5],
            "members": [[1, 2, 3, 4], [1, 2, 3], [4, 5], [6, 7], [1, 2, 3, 6, 7]],
        }
    )
    edges, links = ge.reeb_graph(confs)
    assert dict(zip(edges.conf_id, edges.lane, strict=True)) == {
        0: 0,
        1: 0,  # the larger part keeps the lane
        2: 1,
        3: 2,  # lanes 0 and 1 are busy at 150 s
        4: 0,  # nearest to (0 x 3 + 2 x 2) / 5 = 0.8
    }
    assert links.values.tolist() == [[100, 0, 0], [100, 0, 1], [200, 0, 0], [200, 2, 0]]


def test_tables_are_the_groups_folder_of_the_data_folder(qapp, tmp_path, monkeypatch):
    from xc_viewer.core.disciplines import DATA_FOLDER_ENV
    from xc_viewer.widgets import group_flights as gf

    monkeypatch.setenv(DATA_FOLDER_ENV, str(tmp_path / "data"))
    (tmp_path / "data").mkdir()
    assert gf.groups_folder() is None  # nothing exported yet
    write_tables(tmp_path / "data" / "groups")
    assert gf.groups_folder() == tmp_path / "data" / "groups"


def test_tab_ranks_the_episodes_of_the_chosen_tables(qapp, tmp_path):
    from xc_viewer.widgets.group_flights import GroupFlights

    tab = GroupFlights()
    try:
        tab._tables = ge.load_tables(write_tables(tmp_path / "out"))
        tab._show_sources()
        assert tab._episodes.count() == 2 and tab._episodes.currentData() == 0
        assert "89.5 min" in tab._sources
        assert not tab._load.isEnabled()  # no census of cleaned tracks yet
        tab._order.setCurrentIndex(tab._order.findData("duration"))
        assert tab._episodes.currentData() == 1
        tab._minimum.setValue(3)
        assert tab._episodes.count() == 1
    finally:
        tab.shutdown()
        tab.close()


def zoned_tables(folder):
    """The tables of `write_tables` with zones: day one starts in cell (10, 20) with
    two flights of three; the other day has a tie between (30, 40) and (31, 40)."""
    cells = pd.DataFrame(
        {
            "discipline": ["paragliders"] * 2 + ["hang gliders"] + ["paragliders"] * 2,
            "flight_id": ["a", "b", "c", "d", "e"],
            "start_ix": [10, 10, 11, 31, 30],
            "start_iy": [20, 20, 20, 40, 40],
        }
    )
    return ge.attach_zones(ge.load_tables(write_tables(folder)), cells)


def test_an_episode_is_in_the_cell_most_of_its_members_took_off_from(tmp_path):
    tables = zoned_tables(tmp_path / "out")
    e = tables.episodes.set_index("episode_id")
    assert e.loc[0, ["zone_x", "zone_y"]].tolist() == [10, 20]
    assert e.loc[1, ["zone_x", "zone_y"]].tolist() == [30, 40]  # a tie: lowest cell
    first = ge.rank_episodes(tables, "flights", zone=(10, 20))
    assert first.episode_id.tolist() == [0]
    assert ge.rank_episodes(tables, "flights", day=OTHER).episode_id.tolist() == [1]
    assert ge.rank_episodes(tables, "flights", zone=(10, 20), day=OTHER).empty
    by_day = ge.activity(tables.episodes, ["date"]).set_index("date")
    assert by_day.loc[DAY, ["episodes", "flights"]].tolist() == [1, 3]
    assert by_day.loc[DAY, "hours"] == pytest.approx(2400 / 3600)
    assert "°N" in ge.zone_label((10, 20)) and "°E" in ge.zone_label((10, 20))


def test_an_episode_without_a_located_member_has_no_zone(tmp_path):
    tables = ge.load_tables(write_tables(tmp_path / "out"))
    cells = pd.DataFrame(
        {
            "discipline": ["paragliders"],
            "flight_id": ["d"],
            "start_ix": [5],
            "start_iy": [6],
        }
    )
    e = ge.attach_zones(tables, cells).episodes.set_index("episode_id")
    assert e.zone_x.isna().tolist() == [True, False]  # only flight d is in the index
    assert e.loc[1, ["zone_x", "zone_y"]].tolist() == [5, 6]


def test_the_browser_narrows_days_to_a_zone_and_the_map_to_a_day(qapp, tmp_path):
    from xc_viewer.widgets.group_zones import ZoneDayDialog, cell_corners, zone_at

    corners = cell_corners([(10, 20)])
    assert zone_at(*corners[0].mean(axis=0)) == (10, 20)  # a click inside the cell
    tables = zoned_tables(tmp_path / "out")
    dialog = ZoneDayDialog()
    chosen = []
    dialog.chosen.connect(lambda zone, day: chosen.append((zone, day)))
    try:
        dialog.set_data(
            tables, {"discipline": None, "year": None, "minimum": 2}, None, None
        )
        assert dialog._days.rowCount() == 2 and dialog._zones == {(10, 20), (30, 40)}
        dialog.choose((10, 20), None)
        assert dialog._days.rowCount() == 1  # the days of that zone only
        dialog._days.selectRow(0)
        assert dialog._day == DAY and dialog._zones == {(10, 20)}  # the map of that day
        assert not dialog._clear_zone.isHidden() and dialog._clear_day.isEnabled()
        dialog._show.click()
        assert chosen == [((10, 20), DAY)]
        dialog.choose(None, None)
        assert dialog._days.rowCount() == 2 and len(dialog._zones) == 2
    finally:
        dialog.close()


def test_tab_lists_the_episodes_of_a_zone_and_a_day(qapp, tmp_path):
    from xc_viewer.widgets.group_flights import GroupFlights

    tab = GroupFlights()
    try:
        tab._tables = zoned_tables(tmp_path / "out")
        tab._show_sources()
        assert tab._episodes.count() == 2 and "°N" in tab._episodes.itemText(0)
        tab._choose_where((30, 40), None)
        assert tab._episodes.count() == 1 and tab._episodes.currentData() == 1
        assert "zone" in tab._where.text() and tab._clear_where.isEnabled()
        tab._choose_where(None, DAY)
        assert tab._episodes.currentData() == 0 and "day " + DAY in tab._where.text()
        tab._clear_where.click()
        assert tab._episodes.count() == 2 and not tab._clear_where.isEnabled()
    finally:
        tab.shutdown()
        tab.close()


def test_tracks_split_where_the_flight_enters_and_leaves_the_episode():
    t = np.arange(0, 101, 10.0)
    piece = np.column_stack([t, t, 0 * t, 0 * t])
    pieces, inside = ge.split_inside(piece, np.array([[30.0, 60.0]]))
    assert inside == [False, True, False]
    assert [p[[0, -1], 0].tolist() for p in pieces] == [[0, 30], [30, 60], [60, 100]]
    pieces, inside = ge.split_inside(piece, np.array([[95.0, 200.0]]))
    assert inside == [False, True] and pieces[1][:, 0].tolist() == [90, 100]
    assert ge.split_inside(piece, np.empty((0, 2)))[1] == [False]
