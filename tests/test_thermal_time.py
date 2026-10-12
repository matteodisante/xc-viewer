"""Thermal time maps conserve duration across clipping, sampling and zoom."""

import numpy as np
import pandas as pd
import pytest

from xc_viewer.thermal_time import (
    TimeGrid,
    duration_bins,
    load_grids,
    save_grids,
)


def test_crossed_pixels_receive_elapsed_time_and_outside_portions_are_clipped():
    flat, seconds = duration_bins([[-50, 25]], [[150, 25]], [20], (0, 0, 100, 100))
    assert flat.tolist() == [0, 1]
    assert seconds.tolist() == pytest.approx([5, 5])


def test_stationary_and_downward_climb_edges_keep_their_time():
    flat, seconds = duration_bins(
        [[25, 25], [25, 25]], [[25, 25], [25, 75]], [10, 20], (0, 0, 100, 100)
    )
    assert flat.tolist() == [0, 2]
    assert seconds.tolist() == pytest.approx([20, 10])


def test_diagonal_corner_and_subdivision_conserve_identical_weights():
    args = ((0, 0, 100, 100),)
    first = duration_bins([[0, 0]], [[100, 100]], [20], *args)
    second = duration_bins([[0, 0], [25, 25]], [[25, 25], [100, 100]], [5, 15], *args)
    assert first[0].tolist() == second[0].tolist() == [0, 3]
    np.testing.assert_allclose(first[1], second[1])
    assert first[1].sum() == pytest.approx(20)


def test_no_mass_on_upper_boundary_or_from_invalid_edges():
    flat, seconds = duration_bins(
        [[100, 0], [0, 100], [0, 0], [0, 0]],
        [[100, 50], [50, 100], [50, 0], [np.nan, 1]],
        [5, 5, 0, 5],
        (0, 0, 100, 100),
    )
    assert len(flat) == len(seconds) == 0


def test_hour_units_and_coarsening_include_partial_edge_pixel_area(tmp_path):
    grid = TimeGrid((0, 0, 150, 100))
    grid.add([[0, 25]], [[150, 25]], [3600])
    grid.flush()
    assert grid.hours == pytest.approx(1)
    for bins in (100, 1):
        values, xe, ye, _ = grid.window(0, 0.15, 0, 0.1, bins=bins)
        assert (
            values * np.diff(ye)[:, None] * np.diff(xe)[None, :]
        ).sum() == pytest.approx(1)
    path = tmp_path / "duration.npz"
    save_grids({"test": grid}, path, {"complete": True})
    loaded, _ = load_grids(path)
    assert loaded["test"].hours == pytest.approx(1)


@pytest.fixture
def flat_earth(monkeypatch):
    """East becomes Lambert-93 x, so edges can be read off the fixes."""
    pytest.importorskip("pyproj")
    monkeypatch.setattr(
        "xc_viewer.thermal_time_prepare.enu_to_geodetic",
        lambda e, n, z, f: (np.asarray(n), np.asarray(e)),
    )
    monkeypatch.setattr(
        "xc_viewer.thermal_time_prepare.project", lambda x, y: (x, y)
    )


def _runs(rows):
    return pd.DataFrame(
        rows, columns=["flight_id", "segment_id", "t_start", "t_end", "n_fixes"]
    )


def test_climb_edges_stay_inside_runs_and_skip_gaps_and_flights(flat_earth):
    from xc_viewer.thermal_time_prepare import climb_edges

    fixes = pd.DataFrame(
        {
            "flight_id": ["a"] * 10 + ["b"] * 3,
            "segment_id": [0] * 8 + [1] * 2 + [0] * 3,
            "t": [0, 1, 2, 3, 4, 5, 10, 11, 20, 21, 11, 12, 13],
            "E": np.arange(13.0),
            "N": 0.0,
            "z": 1000.0,
        }
    )
    runs = _runs(
        [
            ("b", 0, 11, 13, 3),
            ("a", 0, 3, 11, 5),
            ("a", 0, 0, 2, 3),
            ("c", 0, 0, 9, 4),  # Another batch's flight is ignored.
        ]
    )
    origins = {"a": (0, 0, 0), "b": (0, 0, 0)}
    a, b, dt = climb_edges(fixes, runs, origins)
    # Never between runs (2-3), across the 5 s gap (5-10), into an unlabelled
    # segment (8-9) or between flights (9-10).
    assert a[:, 0].tolist() == [0, 1, 3, 4, 6, 10, 11]
    assert b[:, 0].tolist() == [1, 2, 4, 5, 7, 11, 12]
    assert dt.tolist() == [1] * 7
    runs.loc[1, "n_fixes"] = 6
    with pytest.raises(ValueError, match="do not match"):
        climb_edges(fixes, runs, origins)
    with pytest.raises(ValueError, match="origin"):
        climb_edges(fixes, runs.iloc[[0]], {"a": (0, 0, 0)})


def test_climb_edges_match_the_per_flight_census_definition(flat_earth):
    from xc_viewer.thermal_geometry import continuous_edges
    from xc_viewer.thermal_time_prepare import climb_edges

    rng = np.random.default_rng(7)
    parts, runs = [], []
    for flight in range(40):
        t0 = 0.0
        for segment in range(rng.integers(1, 4)):
            step = rng.choice([1.0, 2.0, 5.0])
            dt = np.where(rng.random(60) < 0.05, step * 4, step)
            t = t0 + np.cumsum(dt)
            t0 = t[-1] + 100
            parts.append(
                pd.DataFrame(
                    {
                        "flight_id": f"f{flight}",
                        "segment_id": segment,
                        "t": t,
                        "E": rng.normal(size=60).cumsum(),
                        "N": rng.normal(size=60).cumsum(),
                        "z": 1000.0,
                    }
                )
            )
            edges = np.sort(rng.choice(np.arange(1, 59), 6, replace=False))
            for lo, hi in zip(edges[::2], edges[1::2], strict=True):
                runs.append((f"f{flight}", segment, t[lo], t[hi], hi - lo + 1))
    fixes = pd.concat(parts, ignore_index=True)
    runs = _runs(runs)
    a, b, dt = climb_edges(fixes, runs, dict.fromkeys(fixes.flight_id, (0, 0, 0)))
    expected = []
    for fid, flight in fixes.groupby("flight_id", sort=False):
        labels = np.full(len(flight), -1)
        for i, run in enumerate(runs.loc[runs.flight_id == fid].itertuples()):
            labels[
                (flight.segment_id.to_numpy() == run.segment_id)
                & flight.t.between(run.t_start, run.t_end).to_numpy()
            ] = i
        use = continuous_edges(flight) & (labels[:-1] >= 0)
        use &= labels[:-1] == labels[1:]
        index = np.flatnonzero(use)
        east, t = flight.E.to_numpy(), flight.t.to_numpy()
        expected += zip(
            east[index], east[index + 1], t[index + 1] - t[index], strict=True
        )
    assert sorted(zip(a[:, 0], b[:, 0], dt, strict=True)) == sorted(expected)
    assert len(expected) > 100


def test_regions_are_cropped_without_changing_pixel_time():
    from xc_viewer.thermal_time import TimeGrid

    national = TimeGrid((0, 0, 1000, 1000))
    national.add([[10, 10], [510, 510]], [[90, 10], [590, 510]], [40, 60])
    direct = TimeGrid((450, 450, 700, 650))
    direct.add([[10, 10], [510, 510]], [[90, 10], [590, 510]], [40, 60])
    direct.flush()
    crop = national.crop((450, 450, 700, 650), metadata={"name": "test"})
    assert crop.flat.tolist() == direct.flat.tolist()
    assert crop.seconds.tolist() == pytest.approx(direct.seconds.tolist())
    assert crop.hours * 3600 == pytest.approx(60)
    assert crop.metadata == {"name": "test"}
    with pytest.raises(ValueError, match="lattice"):
        national.crop((25, 0, 500, 500))


def test_grid_origin_need_not_be_a_multiple_of_the_pixel_size():
    flat, seconds = duration_bins([[25, 50]], [[175, 50]], [30], (25, 25, 175, 125))
    assert flat.tolist() == [0, 1, 2]
    assert seconds.tolist() == pytest.approx([10, 10, 10])


def test_incomplete_benchmark_cannot_be_displayed_as_the_full_archive(tmp_path):
    path = tmp_path / "benchmark.npz"
    save_grids({"test": TimeGrid((0, 0, 100, 100))}, path, {"complete": False})
    with pytest.raises(ValueError, match="Incomplete"):
        load_grids(path)
