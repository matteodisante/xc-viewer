"""Tests for xc_viewer.geography (matplotlib only, no display needed)."""

from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from xc_viewer import geography


def test_load_basemap_reads_the_committed_file_and_has_a_world_panel():
    panels = geography.load_basemap()
    assert panels is not None
    assert "world" in panels
    assert len(panels["world"]["rings"]) > 0


def test_draw_land_frames_the_axes_to_the_given_extent():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    rings = [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]]
    geography.draw_land(ax, rings, geography.WORLD_EXTENT)
    assert ax.get_xlim() == geography.WORLD_EXTENT[0::2]
    assert ax.get_ylim() == geography.WORLD_EXTENT[1::2]
    assert len(ax.collections) == 1
    plt.close(fig)


def test_draw_density_returns_a_mesh_when_points_fall_inside_the_extent():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    rng = np.random.default_rng(0)
    lon = rng.uniform(5.0, 6.0, 500)
    lat = rng.uniform(44.0, 45.0, 500)
    mesh = geography.draw_density(ax, lon, lat, geography.FRANCE_EXTENT)
    assert mesh is not None
    plt.close(fig)


def test_draw_density_returns_none_when_nothing_falls_inside_the_extent():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # Points nowhere near FRANCE_EXTENT: the mesh must be empty rather than error.
    lon = np.array([170.0, 171.0])
    lat = np.array([-50.0, -51.0])
    mesh = geography.draw_density(ax, lon, lat, geography.FRANCE_EXTENT)
    assert mesh is None
    plt.close(fig)


def test_classify_region_labels_the_three_named_boxes():
    lat = np.array([45.0, 42.8, 50.0, 10.0])
    lon = np.array([7.0, 0.5, 0.0, 0.0])
    labels = geography.classify_region(lat, lon)
    assert list(labels) == ["Alps", "Pyrenees", "Channel Coast", ""]


def test_classify_terrain_bands_match_the_thresholds():
    alt = np.array([-10.0, 300.0, 800.0, 1500.0, np.nan])
    labels = geography.classify_terrain(alt)
    assert list(labels) == [
        "Plains",
        "Hills",
        "Low mountains",
        "High mountains",
        "",
    ]
