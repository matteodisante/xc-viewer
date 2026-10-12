"""Scientific contracts of the thermal-plane slice, independent of the GUI."""

import numpy as np
import pandas as pd
import pytest

from xc_viewer.thermal_geometry import (
    ThermalCell,
    cell_visits,
    climb_edges,
    plane_intersections,
    project,
    unproject,
)


def track(x, z, *, times=None, segments=None, phases=None):
    size = len(x)
    t = np.arange(size, dtype=float) if times is None else np.array(times)
    return pd.DataFrame(
        {
            "x": x,
            "y": np.full(size, 1000),
            "z": z,
            "t": t,
            "utc": t + 1000,
            "segment_id": np.zeros(size) if segments is None else segments,
            "phase": ["climb"] * size if phases is None else phases,
        }
    )


CELL = ThermalCell(0, 0, "Plains", 2, 1, 100, 1000)


def edges(fixes):
    result = climb_edges(fixes, CELL)
    result["flight_id"] = "a"
    result["discipline"] = "paragliders"
    return result


def test_projected_grid_is_metric_and_geographic_roundtrip():
    pyproj = pytest.importorskip("pyproj")
    x, y = project(6, 45)
    lon, lat = unproject([x, x + 5000, x], [y, y, y + 5000])
    assert lon[0] == pytest.approx(6)
    assert lat[0] == pytest.approx(45)
    geod = pyproj.Geod(ellps="WGS84")
    for i in (1, 2):
        _, _, distance = geod.inv(lon[0], lat[0], lon[i], lat[i])
        assert distance == pytest.approx(5000, rel=0.002)


def test_crossed_cell_between_fixes_counts_and_has_clipped_peak():
    # Neither fix falls in ix=1, but the supported line crosses it.
    visits = cell_visits(track([1000, 11000], [100, 1100], times=[0, 10]))
    assert visits.ix.tolist() == [0, 1, 2]
    middle = visits.loc[visits.ix == 1].iloc[0]
    assert middle.t_min == pytest.approx(4)
    assert middle.t_max == pytest.approx(9)
    assert middle.max_alt == pytest.approx(1000)


def test_revisits_count_once_and_segments_never_create_visits():
    visits = cell_visits(track([1000, 6000, 1000], [100, 200, 300]))
    assert visits.ix.tolist() == [0, 1]
    split = cell_visits(track([1000, 11000], [100, 1100], segments=[0, 1]))
    assert split.ix.tolist() == [0, 2]


def test_crossing_interpolates_time_and_position_before_spatial_or_time_cut():
    result = plane_intersections(
        edges(track([-1000, 6000], [100, 800], times=[0, 7])),
        CELL,
        350,
        1003,
        1004,
    )
    assert len(result) == 1
    assert result.x.iloc[0] == pytest.approx(2500)
    assert result.utc.iloc[0] == pytest.approx(1003.5)
    assert plane_intersections(
        edges(track([-1000, 6000], [100, 800], times=[0, 7])),
        CELL,
        350,
        1004,
        1005,
    ).empty


def test_no_plane_crossing_across_gap_segment_or_phase_boundary():
    for fixes in (
        track([1000, 1100], [200, 400], segments=[0, 1]),
        track([1000, 1100], [200, 400], phases=["climb", "search"]),
        track([1000, 1100, 1200, 1300], [100, 200, 400, 500], times=[0, 1, 101, 102]),
    ):
        assert plane_intersections(edges(fixes), CELL, 200, 0, 2000).empty


def test_vertex_once_terminal_vertex_and_downward_crossings():
    fixes = track([1000, 1100, 1200], [200, 300, 400])
    assert len(plane_intersections(edges(fixes), CELL, 200, 0, 2000)) == 1
    assert len(plane_intersections(edges(fixes), CELL, 300, 0, 2000)) == 1
    downward = track([1000, 1100], [400, 200])
    assert len(plane_intersections(edges(downward), CELL, 200, 0, 2000)) == 1
    flat = track([1000, 1100], [300, 300])
    assert plane_intersections(edges(flat), CELL, 200, 0, 2000).empty


def test_plane_outside_cell_not_thermal_inside_cell():
    fixes = track([-1000, 6000], [100, 800])
    assert plane_intersections(edges(fixes), CELL, 10, 0, 2000).empty


def test_nan_and_invalid_bounds():
    fixes = track([np.nan, 1000], [100, 800])
    assert plane_intersections(edges(fixes), CELL, 100, 0, 2000).empty
    with pytest.raises(ValueError):
        plane_intersections(edges(fixes), CELL, CELL.max_agl_m + 1, 0, 2000)


def test_exact_plane_between_fixes_has_no_altitude_band():
    fixes = edges(track([1000, 2000], [197, 203]))
    point = plane_intersections(fixes, CELL, 100, 0, 2000).iloc[0]
    assert point.x == 1500
    assert point.utc == 1000.5
    # Proximity alone never supplies a crossing outside the supported heights.
    assert plane_intersections(fixes, CELL, 104, 0, 2000).empty
