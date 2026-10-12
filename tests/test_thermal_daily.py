"""Saved height lattice agrees with native edge intersections and civil time."""

import numpy as np
import pandas as pd

from xc_viewer.thermal_daily import height_levels, lattice_points, local_bounds
from xc_viewer.thermal_geometry import ThermalCell, plane_intersections
from xc_viewer.thermal_store import EDGE_COLUMNS


def test_lattice_matches_native_edges():
    cell = ThermalCell(0, 0, "Plains", 1, 1, 100, 153)
    # Ascending, descending, terminal/shared vertices, coplanar, and spatial cuts.
    values = np.array(
        [
            [0, 0, 100, 0, 100, 100, 130, 30],
            [100, 100, 130, 30, 6000, 100, 120, 60],
            [0, 100, 120, 70, 100, 100, 120, 80],
            [100, 100, 120, 80, 100, 100, 153, 90],
            [-100, 100, 99, 100, 5100, 100, 155, 110],
        ],
        dtype=float,
    )
    frame = pd.DataFrame(values, columns=EDGE_COLUMNS)
    frame["flight_id"], frame["discipline"] = "one", "paragliders"
    prepared = lattice_points(values, cell)
    for i, z in enumerate(height_levels(cell.max_agl_m)):
        expected = plane_intersections(frame, cell, z, -np.inf, np.inf)
        np.testing.assert_allclose(
            prepared[prepared[:, 0] == i, 1:], expected[["x", "y", "utc"]].to_numpy()
        )


def test_step_and_dst():
    assert height_levels(53, 20).tolist() == [0, 20, 40, 53]
    assert height_levels(50, 10).tolist() == [0, 10, 20, 30, 40, 50]
    assert np.diff(local_bounds("2026-03-29"))[0] == 23 * 3600
    assert np.diff(local_bounds("2026-10-25"))[0] == 25 * 3600
    assert np.diff(local_bounds("2026-07-01", (11, 15)))[0] == 4 * 3600


def test_season_days_wrap_years_and_clamp_leap_day():
    from datetime import date

    from xc_viewer.thermal_daily import season_days

    winter = season_days((12, 30), (1, 2), range(2023, 2025))
    assert winter == [
        date(2023, 12, 30),
        date(2023, 12, 31),
        date(2024, 1, 1),
        date(2024, 1, 2),
        date(2024, 12, 30),
        date(2024, 12, 31),
        date(2025, 1, 1),
        date(2025, 1, 2),
    ]
    leap = season_days((2, 28), (2, 29), range(2023, 2025))
    assert leap == [date(2023, 2, 28), date(2024, 2, 28), date(2024, 2, 29)]
    assert season_days((6, 1), (8, 31), range(2015, 2014)) == []


def test_wall_windows_follow_the_paris_clock_across_dst_changes():
    from datetime import date, datetime

    from xc_viewer.thermal_daily import PARIS, wall_windows

    days = [date(2024, 3, 30), date(2024, 3, 31), date(2024, 10, 27)]
    windows = wall_windows(days, (8, 18.5))
    for (start, end), day in zip(windows, days, strict=True):
        assert datetime.fromtimestamp(start, PARIS).strftime("%H:%M") == "08:00"
        assert datetime.fromtimestamp(end, PARIS).strftime("%H:%M") == "18:30"
        assert datetime.fromtimestamp(start, PARIS).date() == day
    whole = wall_windows([date(2024, 3, 31)], (0, 24))
    assert whole[0, 1] - whole[0, 0] == 23 * 3600


def test_window_membership_is_half_open_and_spans_count_on_overlap():
    from xc_viewer.thermal_daily import in_windows, overlaps_windows

    windows = np.array([[10.0, 20.0], [30.0, 40.0]])
    utc = [5, 10, 19.9, 20, 25, 30, 40, 50]
    assert in_windows(utc, windows).tolist() == [
        False,
        True,
        True,
        False,
        False,
        True,
        False,
        False,
    ]
    start = np.array([0, 15, 20, 21, 39, 40])
    end = np.array([9, 16, 29, 30, 45, 50])
    assert overlaps_windows(start, end, windows).tolist() == [
        False,
        True,
        False,
        True,
        True,
        False,
    ]
    assert not in_windows(utc, np.empty((0, 2))).any()
