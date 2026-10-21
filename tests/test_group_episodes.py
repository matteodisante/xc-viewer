"""Group episodes: the tables of the group identification, their ranking and replay."""

import json

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
    view = EpisodeScene()
    try:
        view.set_episode(scene, ep)
        assert view._table.rowCount() == 3 and view._slider.isEnabled()
        assert view._annotations.anchors.shape == (0, 3)  # no cells to mark
        assert ep.inside[0] == [False, True, False]  # before, in and after the episode
        np.testing.assert_allclose(view._lines[0][1].color, _rgba(PILOTS[0], 0.9))
        np.testing.assert_allclose(view._lines[0][0].color, _rgba(PILOTS[0], 0.35))
        view._track_by.setCurrentIndex(view._track_by.findData("episode"))
        np.testing.assert_allclose(view._lines[0][1].color, _rgba(EPISODE, 0.9))
        np.testing.assert_allclose(view._lines[0][0].color, _rgba(TRACK, 0.35))
        assert view._now == T0 + 600 and view._cursor.value() == T0 + 600
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
        view._whole.setChecked(False)  # only what has been flown
        track = ep.tracks[0]
        assert not view._lines[0][1].visible() and view._flown[0].visible()
        np.testing.assert_allclose(view._flown[0].pos[-1], track.at(T0 + 2000))
        view.seek(T0 + 2010)  # the flown track waits, the trail bridges the gap
        np.testing.assert_allclose(view._flown[0].pos[-1], track.at(T0 + 2000))
        np.testing.assert_allclose(view._trails[0].pos[-1], track.at(T0 + 2010))
        view.seek(T0 + 1500)  # back in time: rebuilt
        np.testing.assert_allclose(view._flown[0].pos[-1], track.at(T0 + 1500))
        view._whole.setChecked(True)
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
        view.seek(ep.window[1] - 1)
        view._advance()
        assert not view._play.isChecked()  # and stops at the end
        view._clear_scene()
        assert view._markers is None and not view._slider.isEnabled()
        view.seek(T0 + 1000)  # nothing to draw, nothing fails
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


def test_tables_default_to_the_groups_folder_of_the_data_folder(
    qapp, tmp_path, monkeypatch
):
    from xc_viewer.core.disciplines import DATA_FOLDER_ENV
    from xc_viewer.widgets import group_flights as gf

    class Settings:
        def value(self, key, default):
            return str(tmp_path / "chosen")

    monkeypatch.setattr(gf, "QSettings", Settings)
    monkeypatch.setenv(DATA_FOLDER_ENV, str(tmp_path / "data"))
    (tmp_path / "data").mkdir()
    assert gf.default_tables() == tmp_path / "chosen"  # no groups/: the last folder
    write_tables(tmp_path / "data" / "groups")
    assert gf.default_tables() == tmp_path / "data" / "groups"


def test_tab_ranks_the_episodes_of_the_chosen_tables(qapp, tmp_path):
    from xc_viewer.widgets.group_flights import GroupFlights

    tab = GroupFlights()
    try:
        tab._tables = ge.load_tables(write_tables(tmp_path / "out"))
        tab._show_sources()
        assert tab._episodes.count() == 2 and tab._episodes.currentData() == 0
        assert "89.5 min" in tab._sources and "89.5 min" in tab._choose.toolTip()
        assert not tab._load.isEnabled()  # no census of cleaned tracks yet
        tab._order.setCurrentIndex(tab._order.findData("duration"))
        assert tab._episodes.currentData() == 1
        tab._minimum.setValue(3)
        assert tab._episodes.count() == 1
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
