"""Drawing a basemap and a take-off density mesh on it: matplotlib only.

It mirrors the static launch-density figure of the thesis this viewer comes from: same
basemap file, same colours, same binned-density approach, so the interactive map and the
static figure never disagree about what "where flights launch" looks like.

No Qt import (see the package docstring).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import QuadMesh

# repo root: src/soaring/viewer/geography.py -> viewer -> soaring -> src -> root.
_BASEMAP_PATH = Path(__file__).resolve().parent / "assets" / "basemap.json"

#: ``(lon_min, lat_min, lon_max, lat_max)``.
Extent = tuple[float, float, float, float]

# Same values as generate_prelim_figure.py, for the reason in the module docstring.
LAND, SEA, COAST = "#efece6", "#dce7ef", "#9aa5ae"
WORLD_EXTENT: Extent = (-180.0, -60.0, 180.0, 75.0)
FRANCE_EXTENT: Extent = (-5.5, 41.0, 10.0, 51.5)
REUNION_EXTENT: Extent = (55.15, -21.42, 55.92, -20.82)
CELL_DEG = 0.15

# The same three take-off boxes as ``TERRAIN_GROUPS`` in
# generate_kinematic_isotropy_figure.py (themselves a subset of ``REGIONS`` in
# generate_prelim_figure.py: this drops "Massif Central" and the two other flat
# controls, which those chapter-3 figures use but this picker does not offer).
# Boxes, not polygons traced from the basemap, for the same reason given there:
# a box that is written down can be checked against a gazetteer.
REGIONS: dict[str, Extent] = {
    "Alps": (5.4, 43.8, 10.0, 46.6),
    "Pyrenees": (-1.9, 42.0, 3.3, 43.5),
    "Channel Coast": (-1.8, 48.3, 2.0, 51.2),
}

# Descriptive elevation bands from arXiv:2608.00241 Eq. 1, same labels and
# thresholds as generate_terrain_figure.py -- applied here to the pipeline's own
# ``alt0`` (the cleaned trajectory's origin altitude) rather than that figure's
# raw-first-fix proxy, since ``alt0`` is what the rest of the viewer already uses.
TERRAIN_BANDS = [
    ("Plains", -np.inf, 300.0),
    ("Hills", 300.0, 800.0),
    ("Low mountains", 800.0, 1500.0),
    ("High mountains", 1500.0, np.inf),
]
TERRAIN_ORDER = [name for name, _, _ in TERRAIN_BANDS]


def classify_region(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Label each point by the :data:`REGIONS` box it falls in, ``""`` if none."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    labels = np.full(lat.shape, "", dtype=object)
    for name, (lon_min, lat_min, lon_max, lat_max) in REGIONS.items():
        hit = (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)
        labels[hit] = name
    return labels


def classify_terrain(alt: np.ndarray) -> np.ndarray:
    """Label each altitude by the :data:`TERRAIN_BANDS` band it falls in.

    ``""`` if not finite.
    """
    alt = np.asarray(alt, dtype=float)
    labels = np.full(alt.shape, "", dtype=object)
    finite = np.isfinite(alt)
    for name, lo, hi in TERRAIN_BANDS:
        labels[finite & (alt >= lo) & (alt < hi)] = name
    return labels


def load_basemap() -> dict | None:
    """The committed coastline/border geometry (``data/basemap.json``), or ``None``.

    ``None`` when the file is missing, so the map still draws (as a plain sea-coloured
    rectangle) rather than fail outright -- the basemap is a visual aid, not something
    a click-to-load flight depends on.
    """
    if not _BASEMAP_PATH.is_file():
        return None
    return json.loads(_BASEMAP_PATH.read_text(encoding="utf-8"))["panels"]


def draw_land(ax: Axes, rings: list[list[list[float]]], extent: Extent) -> None:
    """Fill land polygons, colour the sea, and frame ``ax`` to ``extent``.

    Args:
        ax: Target axes.
        rings: Polygon rings, e.g. ``basemap["world"]["rings"]`` from
            :func:`load_basemap`.
        extent: ``(lon_min, lat_min, lon_max, lat_max)``.
    """
    from matplotlib.collections import PolyCollection

    ax.add_collection(
        PolyCollection(
            [np.asarray(ring) for ring in rings],
            facecolors=LAND,
            edgecolors=COAST,
            linewidths=0.35,
            zorder=0,
        )
    )
    ax.set_facecolor(SEA)
    ax.set_xlim(extent[0], extent[2])
    ax.set_ylim(extent[1], extent[3])
    # A degree of longitude is cos(phi) of a degree of latitude, so this is what keeps
    # a coastline the right shape rather than a stretched one.
    ax.set_aspect(1 / np.cos(np.deg2rad(0.5 * (extent[1] + extent[3]))))


def draw_density(
    ax: Axes,
    lon: np.ndarray,
    lat: np.ndarray,
    extent: Extent,
    cell: float = CELL_DEG,
    cmap: str = "magma_r",
) -> QuadMesh | None:
    """Take-off points as a binned mesh, on a log colour scale.

    A mesh, not one marker per flight: at 1e5+ flights markers saturate over a busy
    site and the panel stops answering whether the ensemble samples broadly or
    concentrates -- the same reasoning as the static thesis figure.

    Returns:
        The mesh artist (for a colorbar), or ``None`` if no cell has a flight in it.
    """
    from matplotlib.colors import LogNorm

    lon_edges = np.arange(extent[0], extent[2] + cell, cell)
    lat_edges = np.arange(extent[1], extent[3] + cell, cell)
    counts, _, _ = np.histogram2d(lon, lat, bins=(lon_edges, lat_edges))
    if counts.max() < 1:
        return None
    return ax.pcolormesh(
        lon_edges,
        lat_edges,
        np.ma.masked_less(counts.T, 1),
        cmap=cmap,
        norm=LogNorm(vmin=1, vmax=max(counts.max(), 2)),
        zorder=2,
    )
