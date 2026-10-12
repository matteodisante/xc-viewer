"""Stream archived climb trajectories into bounded-memory residence-time products."""

from __future__ import annotations

import io
import json
import time
from contextlib import suppress
from datetime import UTC, datetime
from itertools import pairwise

import numpy as np
import pandas as pd
from pyproj import Transformer

from .core.preproc.enu import LocalFrame
from .geodesy import enu_to_geodetic
from .sqlite import connect
from .thermal_geometry import project
from .thermal_regions import REGIONS, extent
from .thermal_store import load_store
from .thermal_time import BASE_M, TimeGrid, cache_path, load_grids, save_grids

# The climb labels behind every residence-time product.
SOURCE = "vilpellet"


def region_grids(national):
    """Crop the national grid to metric viewports enclosing the five regional boxes."""
    transform = Transformer.from_crs(4326, 2154, always_xy=True)
    result = {}
    for name, box in REGIONS.items():
        bounds = np.asarray(transform.transform_bounds(*extent(box), densify_pts=21))
        bounds[:2] = np.floor(bounds[:2] / BASE_M) * BASE_M
        bounds[2:] = np.ceil(bounds[2:] / BASE_M) * BASE_M
        result[f"region/{name}/{SOURCE}"] = national.crop(
            bounds,
            metadata={
                "name": name,
                "source": SOURCE,
                "population": "all crossing trajectories",
            },
        )
    return result


def climb_edges(fixes, runs, origins):
    """Lambert-93 edges inside saved Vilpellet climb runs, and their seconds.

    ``fixes`` holds whole flights, each contiguous and ordered by segment and time;
    ``runs`` holds their climb runs (``flight_id``, ``segment_id``, ``t_start``,
    ``t_end``, ``n_fixes``). An edge joins two fixes of one run across a continuous
    step, as in :func:`~xc_viewer.thermal_geometry.continuous_edges`: positive
    and at most 1.5 times the segment's median step. Gaps, run boundaries and
    separate flights never contribute.
    """
    empty = np.empty((0, 2)), np.empty((0, 2)), np.empty(0)
    if len(fixes) < 2 or not len(runs):
        return empty
    codes, flights = pd.factorize(fixes.flight_id.astype(str), sort=False)
    segments = fixes.segment_id.to_numpy(np.int64)
    if (segments < 0).any() or (segments >= 2**31).any():
        raise ValueError("Unexpected preprocessing segment identifier")
    key = codes.astype(np.int64) * 2**32 + segments
    t = fixes.t.to_numpy(float)
    dt = np.diff(t)
    supported = (key[1:] == key[:-1]) & (dt > 0)
    median = pd.Series(dt[supported]).groupby(key[:-1][supported]).median()
    limit = median.reindex(key[:-1]).to_numpy()
    edge = supported & (dt <= 1.5 * limit)

    run_codes = flights.get_indexer(runs.flight_id.astype(str))
    runs = runs.loc[run_codes >= 0]
    run_codes = run_codes[run_codes >= 0].astype(np.int64)
    run_key = run_codes * 2**32 + runs.segment_id.to_numpy(np.int64)
    t0, t1 = runs.t_start.to_numpy(float), runs.t_end.to_numpy(float)
    order = np.lexsort((t0, run_key))
    run_key, t0, t1 = run_key[order], t0[order], t1[order]
    expected = runs.n_fixes.to_numpy(np.int64)[order]
    # Merge run starts into the fixes: each fix takes the last run starting at or
    # before it in the same flight segment, if that run has not ended yet.
    n = len(run_key)
    is_fix = np.r_[np.zeros(n, bool), np.ones(len(key), bool)]
    merged = np.lexsort((is_fix, np.r_[t0, t], np.r_[run_key, key]))
    last = np.maximum.accumulate(np.where(is_fix[merged], -1, merged))
    at_fix = is_fix[merged]
    fix, run = merged[at_fix] - n, last[at_fix]
    ok = run >= 0
    ok[ok] = (run_key[run[ok]] == key[fix[ok]]) & (t[fix[ok]] <= t1[run[ok]])
    labels = np.full(len(key), -1, dtype=np.int64)
    labels[fix[ok]] = run[ok]
    if not np.array_equal(np.bincount(labels[labels >= 0], minlength=n), expected):
        raise ValueError("Saved Vilpellet runs do not match the native archive")

    indices = np.flatnonzero(edge & (labels[:-1] >= 0) & (labels[:-1] == labels[1:]))
    if not len(indices):
        return empty
    positions = np.unique(np.r_[indices, indices + 1])
    lat, lon = np.empty(len(positions)), np.empty(len(positions))
    east, north, z = (fixes[c].to_numpy(float)[positions] for c in ("E", "N", "z"))
    owner = codes[positions]
    # Flights are contiguous, so each owner occupies one slice of ``positions``.
    for a, b in pairwise(np.flatnonzero(np.r_[True, owner[1:] != owner[:-1], True])):
        fid = str(flights[owner[a]])
        if fid not in origins:
            raise ValueError(f"Missing geographic origin for flight {fid}")
        lat[a:b], lon[a:b] = enu_to_geodetic(
            east[a:b], north[a:b], z[a:b], LocalFrame(*origins[fid])
        )
    xy = np.column_stack(project(lon, lat))
    return (
        xy[np.searchsorted(positions, indices)],
        xy[np.searchsorted(positions, indices + 1)],
        dt[indices],
    )


def prepare(
    *, regions=True, cells=True, limit_batches=None, output=None, progress=print
):
    """Crop regions from the national Vilpellet product; reuse saved cell edges.

    No decoder runs, no new plane intersections, no changes to the source archives.
    Limited builds (cells only) require an explicit output and are marked incomplete.
    """
    if limit_batches is not None and output is None:
        raise ValueError("A benchmark must use a separate output path")
    if limit_batches is not None and regions:
        raise ValueError("Regions crop the complete national product; limit cells only")
    start = time.perf_counter()
    target = cache_path() if output is None else output
    grids, report = (
        {},
        {
            "created": datetime.now(UTC).isoformat(),
            "base_m": BASE_M,
            "complete": limit_batches is None,
            "inputs": [],
            "stages": {},
        },
    )
    if not regions or not cells:
        with suppress(FileNotFoundError):
            existing = (
                target
                if target.exists()
                else target.with_name(".regional-" + target.name)
            )
            grids, previous_report = load_grids(existing)
            report["inputs"] = previous_report.get("inputs", [])
            report["stages"] = previous_report.get("stages", {})
    if regions:
        from .route_density import prepare_density

        stage_start = time.perf_counter()
        national_path = prepare_density(progress=progress)
        national, info = load_grids(national_path)
        grids.update(region_grids(national[f"archive/{SOURCE}"]))
        report["inputs"].append(
            {"path": str(national_path), "signature": info.get("signature")}
        )
        report["stages"]["regions"] = {
            "seconds": time.perf_counter() - stage_start,
            "national": str(national_path),
        }
        # The private checkpoint is never opened automatically by the viewer.
        save_grids(grids, target.with_name(".regional-" + target.name), report)
        progress("Regional duration checkpoint saved")
    if cells:
        store = load_store()
        if store is None:
            raise FileNotFoundError("Prepare thermal-planes.sqlite3 first")
        cell_grids = {}
        for cell in store.cells():
            cell_grids[cell.ix, cell.iy] = grids[
                f"cell/{cell.ix}/{cell.iy}/{SOURCE}"
            ] = TimeGrid(
                cell.bounds,
                metadata={
                    "source": SOURCE,
                    "cell": [cell.ix, cell.iy],
                    "terrain": cell.terrain,
                    "ground_m": cell.ground_m,
                    "visiting_flights": cell.flights,
                },
            )
        stage_start = time.perf_counter()
        with connect(store.path.as_uri() + "?mode=ro", uri=True) as db:
            rows = db.execute(
                "SELECT ix,iy,edges FROM climbs WHERE source=?", (SOURCE,)
            )
            products = 0
            for ix, iy, blob in rows:
                if limit_batches is not None and products >= limit_batches * 100:
                    break
                with np.load(io.BytesIO(blob), allow_pickle=False) as saved:
                    values = saved["edges"]
                grid = cell_grids[ix, iy]
                grid.add(values[:, :2], values[:, 4:6], values[:, 7] - values[:, 3])
                products += 1
                if products % 10000 == 0:
                    progress(
                        f"Cells: {products:,} saved flight/cell products; "
                        f"{time.perf_counter() - stage_start:.1f} s"
                    )
            metadata = dict(db.execute("SELECT key,value FROM metadata"))
            report["snapshot"] = {
                k: metadata.get(k)
                for k in ("archive_signature", "segmentation_signatures")
            }
        report["stages"]["cells"] = {
            "seconds": time.perf_counter() - stage_start,
            "products": products,
        }
    save_grids(grids, target, report)
    if cells and limit_batches is None:
        target.with_name(".regional-" + target.name).unlink(missing_ok=True)
    report["elapsed_s"] = time.perf_counter() - start
    report["output_bytes"] = target.stat().st_size
    report["grids"] = {
        key: {
            "hours": grid.hours,
            "occupied_pixels": len(grid.flat),
            "bounds": grid.bounds.tolist(),
        }
        for key, grid in grids.items()
    }
    target.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    progress(
        f"Wrote {target}: {report['output_bytes'] / 1e6:.2f} MB "
        f"in {report['elapsed_s']:.1f} s"
    )
    return report
