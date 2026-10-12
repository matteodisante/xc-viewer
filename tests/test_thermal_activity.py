"""Rank ascent episodes, without counting fixes, gaps or repeated cell entries."""

import numpy as np
import pandas as pd
import pytest

from xc_viewer.thermal_activity import edge_run_cells, flight_activity


def test_one_episode_counts_once_per_cell_and_distinct_episodes_accumulate():
    a = np.array([[100, 100], [4900, 100], [6000, 100], [100, 100]])
    b = np.array([[4900, 100], [6000, 100], [100, 100], [900, 100]])
    assert edge_run_cells(a, b, np.array([0, 0, 0, 1])) == [(0, 0, 2), (1, 0, 1)]


def test_between_fix_cells_count_but_terminal_and_corner_contacts_do_not():
    a = np.array([[100, 100], [100, 100]])
    b = np.array([[15000, 100], [10000, 10000]])
    assert edge_run_cells(a, b, np.array([0, 1])) == [
        (0, 0, 2),
        (1, 0, 1),
        (1, 1, 1),
        (2, 0, 1),
    ]


def test_saved_runs_must_match_native_fixes_and_gaps_split_runs(monkeypatch):
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
    assert flight_activity(fixes, [[0, 0, 52, 6]], (0, 0, 0)) == [(0, 0, 2)]
    with pytest.raises(ValueError, match="do not match"):
        flight_activity(fixes, [[0, 0, 52, 7]], (0, 0, 0))
    # Two saved runs cannot be joined simply because every fix is climb-labelled.
    assert flight_activity(fixes, [[0, 0, 1, 2], [0, 2, 52, 4]], (0, 0, 0)) == [
        (0, 0, 2)
    ]


def test_single_fix_run_cannot_intersect_a_plane(monkeypatch):
    fixes = pd.DataFrame(
        {
            "segment_id": [0, 0, 0],
            "t": [0, 1, 2],
            "E": [3] * 3,
            "N": [45] * 3,
            "z": [100] * 3,
        }
    )
    assert flight_activity(fixes, [[0, 1, 1, 1]], (0, 0, 0)) == []
