"""Smoke tests for xc_viewer.plotting (matplotlib only, no display needed)."""

from __future__ import annotations

from itertools import pairwise

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from xc_viewer.plotting import (
    PHASE_COLORS,
    center_message,
    make_axes,
    plot_trajectory,
    save_pdf,
)


def _raw_geo(n=50):
    return pd.DataFrame(
        {
            "lat": 45.0 + np.linspace(0, 0.01, n),
            "lon": 7.0 + np.linspace(0, 0.02, n),
            "alt": np.linspace(1000, 1500, n),
        }
    )


def _cleaned_geo(n=40, n_segments=3):
    seg = np.repeat(np.arange(n_segments), n // n_segments + 1)[:n]
    return pd.DataFrame(
        {
            "lat": 45.0 + np.linspace(0, 0.01, n),
            "lon": 7.0 + np.linspace(0, 0.02, n),
            "alt": np.linspace(1000, 1500, n),
            "segment_id": seg,
        }
    )


def test_2d_geographic_plot_with_raw_and_cleaned_renders():
    fig = plt.figure()
    ax = make_axes(fig, is_3d=False)
    plot_trajectory(ax, raw=_raw_geo(), cleaned=_cleaned_geo(), x="lon", y="lat")
    assert ax.lines
    plt.close(fig)


def test_3d_plot_renders():
    fig = plt.figure()
    ax = make_axes(fig, is_3d=True)
    plot_trajectory(
        ax, raw=_raw_geo(), cleaned=_cleaned_geo(), x="lon", y="lat", z="alt"
    )
    assert ax.lines
    plt.close(fig)


def test_cleaned_segments_are_drawn_as_separate_lines_not_bridged():
    fig = plt.figure()
    ax = make_axes(fig, is_3d=False)
    cleaned = _cleaned_geo(n=40, n_segments=3)
    plot_trajectory(ax, cleaned=cleaned, x="lon", y="lat")
    assert len(ax.lines) == cleaned["segment_id"].nunique()
    plt.close(fig)


def test_only_the_first_segment_gets_a_legend_label():
    fig = plt.figure()
    ax = make_axes(fig, is_3d=False)
    plot_trajectory(ax, cleaned=_cleaned_geo(n_segments=3), x="lon", y="lat")
    labels = [line.get_label() for line in ax.lines]
    assert labels.count("cleaned") == 1
    assert labels.count("_nolegend_") == 2
    plt.close(fig)


def test_color_by_none_draws_cleaned_as_a_single_line():
    fig = plt.figure()
    ax = make_axes(fig, is_3d=False)
    plot_trajectory(
        ax, cleaned=_cleaned_geo(n_segments=3), x="lon", y="lat", color_by=None
    )
    assert len(ax.lines) == 1
    plt.close(fig)


def test_phase_colours_keep_repeated_noncontiguous_runs_separate():
    phase_track = _cleaned_geo(n=6, n_segments=1)
    phase_track["phase"] = [
        "transition",
        "transition",
        "search",
        "search",
        "transition",
        "climb",
    ]
    phase_track["phase_run"] = [0, 0, 1, 1, 2, 3]
    fig = plt.figure()
    ax = make_axes(fig, is_3d=False)

    plot_trajectory(
        ax,
        cleaned=phase_track,
        x="lon",
        y="lat",
        color_by="phase",
        group_by="phase_run",
        color_map=PHASE_COLORS,
    )

    assert len(ax.lines) == 4
    assert [line.get_color() for line in ax.lines] == [
        PHASE_COLORS["transition"],
        PHASE_COLORS["search"],
        PHASE_COLORS["transition"],
        PHASE_COLORS["climb"],
    ]
    assert [line.get_label() for line in ax.lines] == [
        "transition",
        "search",
        "_nolegend_",
        "climb",
    ]
    assert len(ax.lines[-1].get_xdata()) == 1
    assert ax.lines[-1].get_marker() == "."
    plt.close(fig)


def test_dms_ticks_only_apply_to_lat_lon_axes():
    fig = plt.figure()
    ax = make_axes(fig, is_3d=False)
    plot_trajectory(ax, raw=_raw_geo(), x="lon", y="lat", dms=True)
    formatter = ax.xaxis.get_major_formatter()
    assert "°" in formatter(7.001, 0)
    plt.close(fig)


def test_enu_axes_ignore_the_dms_toggle():
    raw = pd.DataFrame({"E": np.linspace(0, 100, 20), "N": np.linspace(0, 50, 20)})
    fig = plt.figure()
    ax = make_axes(fig, is_3d=False)
    plot_trajectory(ax, raw=raw, x="E", y="N", dms=True)
    assert ax.get_xlabel() == "East [m]"
    plt.close(fig)


def test_raw_endpoints_are_marked_at_the_true_first_and_last_rows():
    fig = plt.figure()
    ax = make_axes(fig, is_3d=False)
    raw = _raw_geo(n=50)
    plot_trajectory(ax, raw=raw, x="lon", y="lat")
    labels = {c.get_label() for c in ax.collections}
    assert {"raw start", "raw end"} <= labels
    for collection in ax.collections:
        (point,) = collection.get_offsets()
        if collection.get_label() == "raw start":
            assert point[0] == pytest.approx(raw["lon"].iloc[0])
            assert point[1] == pytest.approx(raw["lat"].iloc[0])
        elif collection.get_label() == "raw end":
            assert point[0] == pytest.approx(raw["lon"].iloc[-1])
            assert point[1] == pytest.approx(raw["lat"].iloc[-1])
    plt.close(fig)


def test_raw_endpoints_can_be_turned_off():
    fig = plt.figure()
    ax = make_axes(fig, is_3d=False)
    plot_trajectory(ax, raw=_raw_geo(), x="lon", y="lat", mark_raw_endpoints=False)
    assert not ax.collections
    plt.close(fig)


def test_raw_endpoints_are_marked_in_3d_too():
    fig = plt.figure()
    ax = make_axes(fig, is_3d=True)
    plot_trajectory(ax, raw=_raw_geo(), x="lon", y="lat", z="alt")
    labels = {c.get_label() for c in ax.collections}
    assert {"raw start", "raw end"} <= labels
    plt.close(fig)


def test_no_endpoint_markers_when_raw_is_not_drawn():
    fig = plt.figure()
    ax = make_axes(fig, is_3d=False)
    plot_trajectory(ax, cleaned=_cleaned_geo(), x="lon", y="lat")
    assert not ax.collections
    plt.close(fig)


def test_center_message_on_a_2d_axes():
    fig = plt.figure()
    ax = make_axes(fig, is_3d=False)
    center_message(ax, "Pick a flight to plot.", is_3d=False)
    assert ax.texts
    plt.close(fig)


def test_center_message_on_a_3d_axes_does_not_raise():
    # Regression: Axes3D.text() takes (x, y, z, s), not (x, y, s) -- calling it the
    # 2D way silently reinterprets the message string as a z-coordinate and then
    # raises on the missing `s`. This is exactly the "3D checked, no flight loaded
    # yet" state the app starts in if a user toggles 3D before picking a flight.
    fig = plt.figure()
    ax = make_axes(fig, is_3d=True)
    center_message(ax, "Pick a flight to plot.", is_3d=True)
    assert ax.texts
    plt.close(fig)


def test_save_pdf_writes_a_vector_pdf(tmp_path):
    fig = plt.figure()
    ax = make_axes(fig, is_3d=False)
    plot_trajectory(ax, raw=_raw_geo(), x="lon", y="lat")
    out = tmp_path / "traj.pdf"
    save_pdf(fig, out)
    assert out.is_file()
    assert out.stat().st_size > 0
    plt.close(fig)


@pytest.mark.parametrize("is_3d", [False, True])
def test_phase_colour_changes_cover_every_native_edge_without_crossing_gaps(is_3d):
    from xc_viewer.data import phases_on_cleaned_fixes

    cleaned = pd.DataFrame(
        {
            "segment_id": [0] * 31 + [1] * 3,
            "t": [*range(31), 0, 1, 2],
            "E": np.cos(np.arange(34)),
            "N": np.sin(np.arange(34)),
            "z": np.arange(34),
        }
    )
    decisions = pd.DataFrame(
        {
            "segment_id": [0, 0, 0],
            "t": [0.0, 10.0, 20.0],
            "phase": ["climb", "search", "climb"],
        }
    )
    coloured = phases_on_cleaned_fixes(cleaned, decisions, decision_step_s=10.0)
    fig = plt.figure()
    ax = make_axes(fig, is_3d=is_3d)
    plot_trajectory(
        ax,
        cleaned=coloured,
        x="E",
        y="N",
        z="z" if is_3d else None,
        color_by="phase",
        group_by="phase_run",
        color_map=PHASE_COLORS,
    )
    actual = []
    for line in ax.lines:
        xs, ys = line.get_data_3d()[:2] if is_3d else line.get_data()
        actual.extend(pairwise(zip(xs, ys, strict=True)))
    expected = []
    for _, segment in cleaned.groupby("segment_id"):
        xy = list(zip(segment.E, segment.N, strict=True))
        expected.extend(pairwise(xy))
    assert actual == expected
    plt.close(fig)
