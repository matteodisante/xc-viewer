"""Terrain extraction must locate crests, not valleys or the flight cloud."""

import numpy as np
import pytest

from xc_viewer.thermal_ridges import clip_line, derive_ridges


@pytest.mark.parametrize("angle", [0, 30, 90])
def test_synthetic_ridge_is_located_independently_of_orientation_and_datum(angle):
    grid = (np.arange(120) + 0.5) * 25
    x, y = np.meshgrid(grid - 1500, grid - 1500)
    across = x * np.cos(np.deg2rad(angle)) + y * np.sin(np.deg2rad(angle))
    z = 200 * np.exp(-0.5 * (across / 150) ** 2)
    lines = derive_ridges(z, (0, 0, 3000, 3000), (500, 500, 2500, 2500))
    shifted = derive_ridges(z + 2000, (0, 0, 3000, 3000), (500, 500, 2500, 2500))
    assert lines
    assert len(lines) == len(shifted)
    for a, b in zip(lines, shifted, strict=True):
        np.testing.assert_allclose(a, b, atol=1e-6)
    xy = np.concatenate(lines) - 1500
    distance = xy[:, 0] * np.cos(np.deg2rad(angle)) + xy[:, 1] * np.sin(
        np.deg2rad(angle)
    )
    assert abs(distance).max() < 5


@pytest.mark.parametrize("shape", ["plane", "valley", "flat"])
def test_plane_valley_and_flat_terrain_do_not_acquire_crest_lines(shape):
    x = np.broadcast_to((np.arange(120) + 0.5) * 25 - 1500, (120, 120))
    z = {
        "plane": 0.1 * x,
        "valley": -200 * np.exp(-0.5 * (x / 150) ** 2),
        "flat": np.zeros_like(x),
    }[shape]
    assert derive_ridges(z, (0, 0, 3000, 3000), (500, 500, 2500, 2500)) == []


def test_clipping_does_not_connect_crests_across_an_outside_excursion():
    points = [(-1, 2), (3, 2), (6, 2), (6, 4), (3, 4)]
    lines = clip_line(points, (0, 0, 5, 5))
    assert lines == [[[0, 2], [3, 2], [5, 2]], [[5, 4], [3, 4]]]
