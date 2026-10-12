"""Approximate crest lines derived from small, saved IGN terrain windows.

The line detector uses terrain only, never flight locations. These are local
transverse maxima at a stated smoothing scale, not an official ridge inventory.
"""

from __future__ import annotations

import json
from functools import lru_cache
from itertools import pairwise
from pathlib import Path

import contourpy
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from .thermal_orography import DATA_DIRECTORY, SUMMIT_COLOR

SERVICE = "https://data.geopf.fr/wms-r/wms"
LAYER = "ELEVATION.ELEVATIONGRIDCOVERAGE.HIGHRES"
DATASET_URL = "https://www.data.gouv.fr/datasets/rge-alti-r"
ATTRIBUTION = "Crests derived from © IGN RGE ALTI · Licence Ouverte 2.0"
PARAMETERS = {
    "grid_m": 25,
    "buffer_m": 500,
    "gaussian_sigma_m": 50,
    "minimum_transverse_curvature_per_m": 0.0003,
    "transverse_probe_m": 200,
    "minimum_drop_each_side_m": 5,
    "minimum_line_length_m": 200,
}


def clip_line(points, bounds):
    """Clip a polyline to a square without joining disjoint surviving pieces."""
    west, south, east, north = bounds
    pieces, current = [], []
    for a, b in pairwise(points):
        a, b = np.asarray(a), np.asarray(b)
        d = b - a
        lo, hi = 0.0, 1.0
        for p, q in zip(
            (-d[0], d[0], -d[1], d[1]),
            (a[0] - west, east - a[0], a[1] - south, north - a[1]),
            strict=True,
        ):
            if p == 0:
                if q < 0:
                    hi = -1
                    break
            elif p < 0:
                lo = max(lo, q / p)
            else:
                hi = min(hi, q / p)
        if lo <= hi and np.linalg.norm(d) > 0:
            start, end = a + lo * d, a + hi * d
            if not current or not np.allclose(current[-1], start, rtol=0, atol=1e-6):
                if len(current) > 1:
                    pieces.append(current)
                current = [start.tolist()]
            current.append(end.tolist())
        elif current:
            if len(current) > 1:
                pieces.append(current)
            current = []
    if len(current) > 1:
        pieces.append(current)
    return pieces


def derive_ridges(z, raster_bounds, cell_bounds, parameters=None):
    """Return scale-dependent height ridges; z rows run south to north.

    Zero transverse derivative + negative transverse curvature defines a
    height ridge. The normal is the Hessian's most concave eigenvector.
    Two orientation charts prevent its arbitrary sign from creating false
    zero crossings. Drop and length cutoffs suppress flat/noisy features.
    """
    p = PARAMETERS if parameters is None else parameters
    z = np.asarray(z, dtype=float)
    if z.ndim != 2 or not np.isfinite(z).all():
        raise ValueError("Terrain must be a finite, two-dimensional height grid")
    step, sigma = p["grid_m"], p["gaussian_sigma_m"] / p["grid_m"]
    west, south, east, north = raster_bounds
    if not np.allclose(
        (east - west, north - south), (z.shape[1] * step, z.shape[0] * step)
    ):
        raise ValueError("Terrain dimensions do not match its metric extent")
    smooth = gaussian_filter(z, sigma)
    centered = z - z.mean()
    gx = gaussian_filter(centered, sigma, order=(0, 1)) / step
    gy = gaussian_filter(centered, sigma, order=(1, 0)) / step
    hessian = np.empty((*z.shape, 2, 2))
    hessian[..., 0, 0] = gaussian_filter(centered, sigma, order=(0, 2)) / step**2
    hessian[..., 1, 1] = gaussian_filter(centered, sigma, order=(2, 0)) / step**2
    hessian[..., 0, 1] = hessian[..., 1, 0] = (
        gaussian_filter(centered, sigma, order=(1, 1)) / step**2
    )
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    nx, ny = eigenvectors[..., 0, 0], eigenvectors[..., 1, 0]
    y, x = np.indices(z.shape, dtype=float)
    probe = p["transverse_probe_m"] / step
    a = map_coordinates(
        smooth, [y + probe * ny, x + probe * nx], order=1, mode="nearest"
    )
    b = map_coordinates(
        smooth, [y - probe * ny, x - probe * nx], order=1, mode="nearest"
    )
    valid = (
        (eigenvalues[..., 0] < -p["minimum_transverse_curvature_per_m"])
        & (smooth - a > p["minimum_drop_each_side_m"])
        & (smooth - b > p["minimum_drop_each_side_m"])
    )
    # Exclude the raster boundary, where differentiation lacks outside support.
    margin = int(np.ceil(max(4 * sigma, probe)))
    valid[:margin] = valid[-margin:] = False
    valid[:, :margin] = valid[:, -margin:] = False
    xgrid = west + (np.arange(z.shape[1]) + 0.5) * step
    ygrid = south + (np.arange(z.shape[0]) + 0.5) * step
    lines = []
    horizontal_normal = np.abs(nx) >= np.abs(ny)
    for chart, component in ((horizontal_normal, nx), (~horizontal_normal, ny)):
        derivative = (gx * nx + gy * ny) * np.where(component < 0, -1, 1)
        contours = contourpy.contour_generator(
            x=xgrid,
            y=ygrid,
            z=np.ma.masked_where(~(valid & chart), derivative),
            corner_mask=False,
        ).lines(0)
        for line in contours:
            if (
                np.linalg.norm(np.diff(line, axis=0), axis=1).sum()
                < p["minimum_line_length_m"]
            ):
                continue
            lines.extend(clip_line(line, cell_bounds))
    return lines


@lru_cache(maxsize=24)
def _read_ridges(path, modified_ns):
    return json.loads(path.read_text(encoding="utf-8"))


def load_ridges(cell, folder=None):
    """Read an attributed offline extract, distinguishing missing from empty."""
    folder = DATA_DIRECTORY if folder is None else Path(folder)
    path = folder / f"ign-ridges-{cell.ix}-{cell.iy}.geojson"
    if not path.exists():
        return None
    from .thermal_ground import terrain_reference

    # These are our estimates from official elevations. Verify the exact saved
    # source raster before displaying a line, including after a local file edit.
    reference = terrain_reference(cell, folder)
    if not reference["source_url"].startswith(SERVICE + "?"):
        raise ValueError("Crest elevation source is not the official IGN WMS")
    payload = _read_ridges(path, path.stat().st_mtime_ns)
    if tuple(payload["provenance"]["bounds_epsg2154"]) != tuple(cell.bounds):
        raise ValueError("Saved crest extent does not match this cell")
    if (
        payload.get("type") != "FeatureCollection"
        or payload.get("crs", {}).get("properties", {}).get("name") != "EPSG:2154"
    ):
        raise ValueError("Crest geometry must be a Lambert-93 FeatureCollection")
    west, south, east, north = cell.bounds
    for f in payload["features"]:
        xy = np.asarray(f["geometry"]["coordinates"], dtype=float)
        if (
            f["geometry"]["type"] != "LineString"
            or xy.ndim != 2
            or xy.shape[1] != 2
            or len(xy) < 2
        ):
            raise ValueError("Invalid crest line geometry")
        if (
            not np.isfinite(xy).all()
            or not (
                (xy >= (west, south) - np.ones(2) * 1e-6)
                & (xy <= (east, north) + np.ones(2) * 1e-6)
            ).all()
        ):
            raise ValueError("Crest coordinates are invalid or outside this cell")
    return payload


def draw_ridges(ax, cell, payload, *, linewidth=1.15):
    """Keep ground crest locations fixed when the flight plane changes height."""
    import matplotlib.patheffects as effects

    for i, feature in enumerate(payload["features"]):
        xy = (np.asarray(feature["geometry"]["coordinates"]) - cell.bounds[:2]) / 1000
        ax.plot(
            xy[:, 0],
            xy[:, 1],
            color=SUMMIT_COLOR,
            linewidth=linewidth,
            zorder=5,
            label="Estimated crests from IGN DEM" if i == 0 else None,
            path_effects=[
                effects.Stroke(linewidth=linewidth + 0.8, foreground="white"),
                effects.Normal(),
            ],
        )
