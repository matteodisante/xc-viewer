"""Conservative thermal residence time on a metric grid, pooling every height."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

BASE_M = 50.0
VERSION = 1
FILE_NAME = "thermal-duration.npz"


def duration_bins(a, b, seconds, bounds, step=BASE_M):
    """Clip linear edges to the map and split their duration at every pixel boundary.

    Inputs are Lambert-93 endpoints and elapsed seconds. Repeated positions retain
    their whole duration. Out-of-frame portions, nonfinite edges and nonpositive
    durations contribute nothing. Grid boundaries are half open.
    """
    a, b, seconds = (
        np.asarray(a, float),
        np.asarray(b, float),
        np.asarray(seconds, float),
    )
    valid = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    valid &= np.isfinite(seconds) & (seconds > 0)
    a, b, seconds = a[valid], b[valid], seconds[valid]
    delta = b - a
    lo, hi = np.zeros(len(a)), np.ones(len(a))
    for axis in (0, 1):
        moving = delta[:, axis] != 0
        left = np.divide(
            bounds[axis] - a[:, axis],
            delta[:, axis],
            out=np.full(len(a), -np.inf),
            where=moving,
        )
        right = np.divide(
            bounds[axis + 2] - a[:, axis],
            delta[:, axis],
            out=np.full(len(a), np.inf),
            where=moving,
        )
        lo = np.maximum(lo, np.minimum(left, right))
        hi = np.minimum(hi, np.maximum(left, right))
        outside = ~moving & (
            (a[:, axis] < bounds[axis]) | (a[:, axis] >= bounds[axis + 2])
        )
        hi[outside] = -1
    valid = hi > lo
    seconds = seconds[valid] * (hi[valid] - lo[valid])
    start = a[valid] + lo[valid, None] * delta[valid]
    end = a[valid] + hi[valid, None] * delta[valid]
    delta = end - start
    n = len(start)
    if not n:
        return np.empty(0, np.int64), np.empty(0, float)
    ids, fractions = [np.arange(n), np.arange(n)], [np.zeros(n), np.ones(n)]
    for axis in (0, 1):
        origin = bounds[axis]
        first = np.floor((np.minimum(start[:, axis], end[:, axis]) - origin) / step) + 1
        last = np.ceil((np.maximum(start[:, axis], end[:, axis]) - origin) / step)
        counts = np.maximum(last - first, 0).astype(np.int64)
        rows = np.repeat(np.arange(n), counts)
        offsets = np.repeat(np.cumsum(counts) - counts, counts)
        lines = origin + (first[rows] + np.arange(len(rows)) - offsets) * step
        ids.append(rows)
        fractions.append((lines - start[rows, axis]) / delta[rows, axis])
    ids, fractions = np.concatenate(ids), np.concatenate(fractions)
    order = np.lexsort((fractions, ids))
    ids, fractions = ids[order], fractions[order]
    width = np.diff(fractions)
    use = (ids[1:] == ids[:-1]) & (width > 0)
    rows = ids[:-1][use]
    middle = (fractions[:-1][use] + fractions[1:][use]) / 2
    xy = start[rows] + middle[:, None] * delta[rows]
    ij = np.floor((xy - np.asarray(bounds[:2])) / step).astype(np.int64)
    nx, ny = np.rint((np.asarray(bounds[2:]) - bounds[:2]) / step).astype(int)
    inside = (ij[:, 0] >= 0) & (ij[:, 0] < nx) & (ij[:, 1] >= 0) & (ij[:, 1] < ny)
    flat = ij[inside, 1] * nx + ij[inside, 0]
    unique, inverse = np.unique(flat, return_inverse=True)
    return unique, np.bincount(inverse, weights=(seconds[rows] * width[use])[inside])


class TimeGrid:
    """Sparse seconds on a fixed metric lattice, aggregated conservatively for zoom."""

    def __init__(self, bounds, flat=None, seconds=None, *, step=BASE_M, metadata=None):
        """Keep only occupied pixels; no regional dense raster is retained in RAM."""
        self.step = float(step)
        self.bounds = np.asarray(bounds, float)
        self.nx, self.ny = np.rint((self.bounds[2:] - self.bounds[:2]) / step).astype(
            int
        )
        self.flat = np.asarray([] if flat is None else flat, dtype=np.int64)
        self.seconds = np.asarray([] if seconds is None else seconds, dtype=float)
        self.metadata = metadata or {}
        self._pending = []
        self._levels = {}

    def add(self, a, b, seconds):
        """Accumulate an input batch without expanding the sparse map."""
        flat, values = duration_bins(a, b, seconds, self.bounds, self.step)
        if len(flat):
            self._pending.append((flat, values))
        if len(self._pending) >= 32:
            self.flush()

    def flush(self):
        """Merge pending batches, conserving elapsed seconds."""
        if not self._pending:
            return
        flat = np.concatenate([self.flat, *(p[0] for p in self._pending)])
        values = np.concatenate([self.seconds, *(p[1] for p in self._pending)])
        self.flat, inverse = np.unique(flat, return_inverse=True)
        self.seconds = np.bincount(inverse, weights=values)
        self._pending.clear()
        self._levels.clear()

    def crop(self, bounds, *, metadata=None):
        """The pixels inside ``bounds`` as a new grid, with their seconds unchanged.

        ``bounds`` must lie on this grid's pixel lattice: each pixel then holds the
        same time it would hold if the edges had been binned into it directly.
        """
        bounds = np.asarray(bounds, float)
        offset = (bounds - np.tile(self.bounds[:2], 2)) / self.step
        if (
            (offset != np.round(offset)).any()
            or (bounds[:2] < self.bounds[:2]).any()
            or (bounds[2:] > self.bounds[2:]).any()
        ):
            raise ValueError("Crop bounds must lie on the grid's lattice and inside it")
        self.flush()
        grid = TimeGrid(bounds, step=self.step, metadata=metadata)
        rows, cols = np.divmod(self.flat, self.nx)
        rows, cols = rows - int(offset[1]), cols - int(offset[0])
        inside = (cols >= 0) & (cols < grid.nx) & (rows >= 0) & (rows < grid.ny)
        # Row-major order survives the shift, so the sparse index stays sorted.
        grid.flat = rows[inside] * grid.nx + cols[inside]
        grid.seconds = self.seconds[inside]
        return grid

    @property
    def hours(self):
        """Total in-frame thermal time, counted once per flight trajectory."""
        return float(self.seconds.sum() / 3600)

    def window(self, x0, x1, y0, y1, bins=700):
        """Return hours/km² for the visible Lambert-93 window (axes in km).

        Powers-of-two aggregation sums seconds and divides by the new pixel area;
        zoom never stretches a coarse density image or manufactures sub-grid detail.
        """
        factor = 2 ** max(
            0,
            int(
                np.ceil(
                    np.log2(max(max(x1 - x0, y1 - y0) * 1000 / bins / self.step, 1))
                )
            ),
        )
        step = self.step * factor
        nx, ny = -(-self.nx // factor), -(-self.ny // factor)
        c0, c1 = np.clip(
            np.floor((np.array([x0, x1]) * 1000 - self.bounds[0]) / step), 0, nx
        ).astype(int)
        r0, r1 = np.clip(
            np.floor((np.array([y0, y1]) * 1000 - self.bounds[1]) / step), 0, ny
        ).astype(int)
        c1, r1 = min(c1 + 1, nx), min(r1 + 1, ny)
        if c1 <= c0 or r1 <= r0:
            return np.zeros((1, 1)), np.array([x0, x1]), np.array([y0, y1]), ""
        if factor not in self._levels:
            if len(self._levels) >= 3:
                del self._levels[next(iter(self._levels))]
            if factor == 1:
                self._levels[factor] = self.flat, self.seconds
            else:
                rows, cols = np.divmod(self.flat, self.nx)
                flat, inverse = np.unique(
                    (rows // factor) * nx + cols // factor, return_inverse=True
                )
                self._levels[factor] = (
                    flat,
                    np.bincount(inverse, weights=self.seconds),
                )
        flat, weights = self._levels[factor]
        start, stop = np.searchsorted(flat, [r0 * nx, r1 * nx])
        flat, weights = flat[start:stop], weights[start:stop]
        rows, cols = np.divmod(flat, nx)
        inside = (cols >= c0) & (cols < c1) & (rows >= r0) & (rows < r1)
        values = np.zeros((max(r1 - r0, 0), max(c1 - c0, 0)))
        values[rows[inside] - r0, cols[inside] - c0] = weights[inside]
        # Edge pixels can contain fewer fine cells than interior pixels.
        widths = np.minimum(factor, self.nx - np.arange(c0, c1) * factor) * self.step
        heights = np.minimum(factor, self.ny - np.arange(r0, r1) * factor) * self.step
        values = values / 3600 / (heights[:, None] * widths[None, :] / 1e6)
        xe = (
            np.minimum(self.bounds[0] + np.arange(c0, c1 + 1) * step, self.bounds[2])
            / 1000
        )
        ye = (
            np.minimum(self.bounds[1] + np.arange(r0, r1 + 1) * step, self.bounds[3])
            / 1000
        )
        return values, xe, ye, f"{step:g} m"


def cache_path():
    """Locate the small density product beside the standalone plane snapshot."""
    from .thermal_store import find_store_path

    override = os.environ.get("XC_VIEWER_CACHE_DIR")
    if override:
        return Path(override).expanduser() / FILE_NAME
    store = find_store_path()
    if store is None:
        raise FileNotFoundError("Choose a data folder with thermal-planes.sqlite3")
    return store.with_name(FILE_NAME)


def save_grids(grids, path, metadata):
    """Publish complete maps atomically; old point counts are never read as hours."""
    arrays = {"version": np.array(VERSION), "metadata": np.array(json.dumps(metadata))}
    records = []
    for i, (key, grid) in enumerate(grids.items()):
        grid.flush()
        records.append(
            {
                "key": key,
                "bounds": grid.bounds.tolist(),
                "step": grid.step,
                "metadata": grid.metadata,
            }
        )
        arrays[f"flat_{i}"] = grid.flat.astype(np.uint32)
        arrays[f"seconds_{i}"] = grid.seconds
    arrays["grids"] = np.array(json.dumps(records))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".building-" + path.name)
    with temporary.open("wb") as out:
        np.savez_compressed(out, **arrays)
    temporary.replace(path)


def load_grids(path=None, *, allow_incomplete=False):
    """Read a duration product, rejecting incompatible count-based versions."""
    path = cache_path() if path is None else Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            "Prepare thermal-duration.npz with prepare_thermal_density.py"
        )
    with np.load(path, allow_pickle=False) as saved:
        if int(saved["version"]) != VERSION:
            raise ValueError("Incompatible thermal duration product; prepare it again")
        metadata = json.loads(str(saved["metadata"]))
        if metadata.get("complete") is False and not allow_incomplete:
            raise ValueError("Incomplete benchmark product; prepare the full density")
        records = json.loads(str(saved["grids"]))
        grids = {
            r["key"]: TimeGrid(
                r["bounds"],
                saved[f"flat_{i}"],
                saved[f"seconds_{i}"],
                step=r["step"],
                metadata=r["metadata"],
            )
            for i, r in enumerate(records)
        }
        return grids, metadata
