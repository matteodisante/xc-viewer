"""Metric cells and exact piecewise-linear climb/plane intersections.

Horizontal coordinates use Lambert-93 (EPSG:2154), appropriate to the metropolitan
France panel in the thesis. Heights remain the adopted GNSS altitude, not ENU up.
No Qt dependency; pyproj is part of the optional viewer dependency group.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import pairwise

import numpy as np
import pandas as pd

CELL_M = 5000.0


@lru_cache(maxsize=2)
def _projection(inverse: bool = False):
    """Construct the cached WGS84/Lambert-93 transformer on first use."""
    from pyproj import Transformer

    source, target = (2154, 4326) if inverse else (4326, 2154)
    return Transformer.from_crs(source, target, always_xy=True)


def project(lon, lat):
    """Longitude/latitude in degrees to metric Lambert-93 coordinates."""
    return _projection().transform(lon, lat)


def unproject(x, y):
    """Metric Lambert-93 coordinates to longitude/latitude in degrees."""
    return _projection(True).transform(x, y)


@dataclass(frozen=True)
class ThermalCell:
    """A square and its reference elevation; published cells use the DEM mean.

    The census initially uses launch medians for ranking. Publication replaces
    ground_m with mean terrain and retains the ranking median separately.
    """

    ix: int
    iy: int
    terrain: str
    flights: int
    launches: int
    ground_m: float
    max_alt_m: float
    launch_median_m: float | None = None

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """West, south, east and north bounds in projected metres."""
        return (
            self.ix * CELL_M,
            self.iy * CELL_M,
            (self.ix + 1) * CELL_M,
            (self.iy + 1) * CELL_M,
        )

    @property
    def max_agl_m(self) -> float:
        """Highest supported in-cell altitude relative to the cell reference."""
        return max(0.0, self.max_alt_m - self.ground_m)


def continuous_edges(fixes: pd.DataFrame) -> np.ndarray:
    """Adjacent native fixes that can be joined without crossing a data gap."""
    t = fixes.t.to_numpy(dtype=float)
    dt = np.diff(t)
    segments = fixes.segment_id.to_numpy()
    same = segments[1:] == segments[:-1]
    # Sampling cadence can differ by segment. A gap never acquires a line merely
    # because both endpoints happen to have the same phase label.
    edge = np.zeros(len(dt), dtype=bool)
    for segment in np.unique(segments):
        supported = same & (segments[:-1] == segment) & (dt > 0)
        if supported.any():
            edge |= supported & (dt <= 1.5 * np.median(dt[supported]))
    for column in ("track_run", "phase_run"):
        if column in fixes:
            values = fixes[column].to_numpy()
            edge &= values[1:] == values[:-1]
    return edge


def cell_visits(fixes: pd.DataFrame) -> pd.DataFrame:
    """Distinct visited squares, time bounds and peak altitude for one flight.

    Input columns are x/y/z/t/segment_id. Edges are split at every grid boundary,
    so a cell crossed between fixes counts too. Time/altitude extrema include the
    clipped edge endpoints. Zero-measure contact with a corner is not a visit.
    """
    cols = ["ix", "iy", "t_min", "t_max", "max_alt"]
    if fixes.empty:
        return pd.DataFrame(columns=cols)
    coords = fixes[["x", "y", "z", "t"]].to_numpy(dtype=float)
    finite = np.isfinite(coords).all(axis=1)
    bins = np.floor(np.where(np.isfinite(coords[:, :2]), coords[:, :2], 0) / CELL_M)
    bins = bins.astype(np.int64)
    samples = np.column_stack(
        (bins[finite], coords[finite, 3], coords[finite, 3], coords[finite, 2])
    )
    changed = np.any(bins[1:] != bins[:-1], axis=1)
    edges = continuous_edges(fixes) & finite[:-1] & finite[1:] & changed
    pieces = []
    for i in np.flatnonzero(edges):
        a, b = coords[i], coords[i + 1]
        fractions = [0.0, 1.0]
        for axis in (0, 1):
            low, high = sorted((a[axis], b[axis]))
            boundaries = (
                np.arange(np.floor(low / CELL_M) + 1, np.ceil(high / CELL_M)) * CELL_M
            )
            if b[axis] != a[axis]:
                fractions.extend(
                    ((boundaries - a[axis]) / (b[axis] - a[axis])).tolist()
                )
        for lo, hi in pairwise(np.unique(fractions)):
            middle = a + (lo + hi) / 2 * (b - a)
            ix, iy = np.floor(middle[:2] / CELL_M).astype(int)
            left, right = a + lo * (b - a), a + hi * (b - a)
            pieces.append((ix, iy, left[3], right[3], max(left[2], right[2])))
    if pieces:
        samples = np.concatenate((samples, np.asarray(pieces)))
    if len(samples) == 0:
        return pd.DataFrame(columns=cols)
    # Avoid several pandas GroupBy constructions for every archived flight.
    samples = samples[np.lexsort((samples[:, 1], samples[:, 0]))]
    starts = np.r_[
        0, np.flatnonzero(np.any(np.diff(samples[:, :2], axis=0), axis=1)) + 1
    ]
    return pd.DataFrame(
        {
            "ix": samples[starts, 0].astype(np.int64),
            "iy": samples[starts, 1].astype(np.int64),
            "t_min": np.minimum.reduceat(samples[:, 2], starts),
            "t_max": np.maximum.reduceat(samples[:, 3], starts),
            "max_alt": np.maximum.reduceat(samples[:, 4], starts),
        }
    )


def climb_edges(fixes: pd.DataFrame, cell: ThermalCell) -> pd.DataFrame:
    """Retain continuous climb edges whose horizontal bounds overlap the cell.

    Phase labels must match at both endpoints. Call before applying a time window
    or a spatial cut, otherwise edge crossings between fixes would be lost.
    """
    columns = [f"{axis}{end}" for end in (0, 1) for axis in ("x", "y", "z", "utc")]
    if len(fixes) < 2 or "phase" not in fixes:
        return pd.DataFrame(columns=columns)
    values = fixes[["x", "y", "z", "utc"]].to_numpy(dtype=float)
    a, b = values[:-1], values[1:]
    labels = fixes.phase.to_numpy()
    west, south, east, north = cell.bounds
    use = continuous_edges(fixes) & (labels[:-1] == "climb") & (labels[1:] == "climb")
    use &= np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    use &= (np.maximum(a[:, 0], b[:, 0]) >= west) & (
        np.minimum(a[:, 0], b[:, 0]) < east
    )
    use &= (np.maximum(a[:, 1], b[:, 1]) >= south) & (
        np.minimum(a[:, 1], b[:, 1]) < north
    )
    return pd.DataFrame(np.column_stack((a[use], b[use])), columns=columns)


def plane_intersections(
    edges: pd.DataFrame,
    cell: ThermalCell,
    z_agl: float,
    start_utc: float,
    end_utc: float,
    *,
    altitude_m: float | None = None,
) -> pd.DataFrame:
    """Intersect climb polylines with an AGL plane, then clip position and UTC.

    A plane vertex belongs to the outgoing nonhorizontal edge (half-open edge
    convention), avoiding duplicates. Horizontal coplanar edges are not isolated
    intersections and are excluded. Both upward and downward crossings within a
    climb-labelled phase are included; no inference of thermal centres is made.
    """
    if (altitude_m is None and not 0 <= z_agl <= cell.max_agl_m) or end_utc < start_utc:
        raise ValueError("Invalid altitude or UTC interval")
    if altitude_m is not None and not np.isfinite(altitude_m):
        raise ValueError("Invalid absolute plane altitude")
    columns = ["x", "y", "utc", "flight_id", "discipline"]
    if edges.empty:
        return pd.DataFrame(columns=columns)
    z = cell.ground_m + z_agl if altitude_m is None else altitude_m
    delta = edges.z1.to_numpy() - edges.z0.to_numpy()
    fraction = np.divide(
        z - edges.z0.to_numpy(),
        delta,
        out=np.full(len(edges), np.nan),
        where=delta != 0,
    )
    use = (fraction >= 0) & (fraction < 1)
    # Include the terminal vertex of each disconnected edge run as well. Adjacent
    # edges share exact native coordinates; a terminal at z must not disappear.
    next_connected = np.zeros(len(edges), dtype=bool)
    if len(edges) > 1:
        next_connected[:-1] = (
            (edges.utc1.to_numpy()[:-1] == edges.utc0.to_numpy()[1:])
            & (edges.flight_id.to_numpy()[:-1] == edges.flight_id.to_numpy()[1:])
            & (edges.discipline.to_numpy()[:-1] == edges.discipline.to_numpy()[1:])
            & (edges.z1.to_numpy()[:-1] == edges.z0.to_numpy()[1:])
        )
    use |= (fraction == 1) & ~next_connected
    selected = edges.loc[use]
    f = fraction[use]
    result = selected[["flight_id", "discipline"]].reset_index(drop=True).copy()
    for axis in ("x", "y", "utc"):
        result[axis] = selected[f"{axis}0"].to_numpy() + f * (
            selected[f"{axis}1"].to_numpy() - selected[f"{axis}0"].to_numpy()
        )
    west, south, east, north = cell.bounds
    inside = (
        (result.x >= west)
        & (result.x < east)
        & (result.y >= south)
        & (result.y < north)
        & result.utc.between(start_utc, end_utc)
    )
    return result.loc[inside, columns].reset_index(drop=True)
