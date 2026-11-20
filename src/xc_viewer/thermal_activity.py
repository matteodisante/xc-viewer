"""Offline national census of continuous climb runs in 5 km cells.

One census per saved segmentation (:mod:`xc_viewer.segmentation`), each in its own file.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
from itertools import pairwise

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import segmentation
from .core.disciplines import DISCIPLINES
from .core.preproc.enu import LocalFrame
from .core.vilpellet import load_vilpellet_config
from .fingerprint import file_identity
from .geodesy import enu_to_geodetic
from .geography import FRANCE_EXTENT
from .sqlite import connect
from .thermal_geometry import CELL_M, continuous_edges, project

ACTIVITY_VERSION = "vilpellet-continuous-climb-runs-v1"


def activity_version(source):
    """The census version of one segmentation; Vilpellet's is the one already saved."""
    if source == segmentation.VILPELLET.key:
        return ACTIVITY_VERSION
    return f"{source}-continuous-climb-runs-v1"


def activity_path(index_path, source):
    """The census file of ``source``, beside the cell census."""
    return index_path.with_name(f"thermal-{source}-activity.sqlite3")


def edge_run_cells(a, b, runs):
    """Count each continuous run once per cell, including crossings between fixes.

    Only positive-duration portions of an edge count. A corner touch or terminal
    vertex on a grid boundary cannot manufacture a run in another cell.
    """
    if not len(a):
        return []
    first = np.floor(a / CELL_M).astype(np.int64)
    last = np.floor(b / CELL_M).astype(np.int64)
    same = np.all(first == last, axis=1)
    pieces = np.column_stack((first[same], runs[same])).tolist()
    for start, end, run in zip(a[~same], b[~same], runs[~same], strict=True):
        fractions = [0.0, 1.0]
        for axis in (0, 1):
            lo, hi = sorted((start[axis], end[axis]))
            boundaries = (
                np.arange(np.floor(lo / CELL_M) + 1, np.ceil(hi / CELL_M)) * CELL_M
            )
            if end[axis] != start[axis]:
                fractions.extend(
                    ((boundaries - start[axis]) / (end[axis] - start[axis])).tolist()
                )
        for lo, hi in pairwise(np.unique(fractions)):
            middle = start + (lo + hi) / 2 * (end - start)
            ix, iy = np.floor(middle / CELL_M).astype(np.int64)
            pieces.append([int(ix), int(iy), int(run)])
    unique = np.unique(np.asarray(pieces, dtype=np.int64), axis=0)
    cells, counts = np.unique(unique[:, :2], axis=0, return_counts=True)
    return [
        (int(ix), int(iy), int(n)) for (ix, iy), n in zip(cells, counts, strict=True)
    ]


def flight_activity(fixes, segments, origin, *, by_time=False):
    """Map saved whole-flight climb runs onto the unchanged native geometry.

    A fix belongs to the run of its segment holding its time, and each run must hold
    the number of fixes it saved. With ``by_time`` the segment numbers of the runs are
    not those of ``fixes``: a fix belongs to the run holding its time, whatever its
    segment, and the counts are not checked.
    """
    fixes = fixes.sort_values(["segment_id", "t"], kind="stable")
    t = fixes.t.to_numpy()
    ids = fixes.segment_id.to_numpy()
    labels = np.full(len(fixes), -1, dtype=np.int64)
    if by_time:
        order = np.argsort(segments[:, 1].astype(float), kind="stable")
        start, end = (segments[order, k].astype(float) for k in (1, 2))
        run = np.searchsorted(start, t, side="right") - 1
        safe = np.clip(run, 0, max(len(start) - 1, 0))
        inside = (run >= 0) & (t <= end[safe])
        labels[inside] = order[safe[inside]]
    else:
        for run, (segment, start, end, expected) in enumerate(segments):
            positions = (ids == segment) & (t >= start) & (t <= end)
            if int(positions.sum()) != int(expected):
                raise ValueError("Saved runs do not match the native archive")
            if np.any(labels[positions] >= 0):
                raise ValueError("Overlapping saved climb runs")
            labels[positions] = run
    supported = continuous_edges(fixes)
    use = supported & (labels[:-1] >= 0) & (labels[:-1] == labels[1:])
    indices = np.flatnonzero(use)
    if not len(indices):
        return []
    # Split at every missing edge, even within a saved run; never bridge gaps.
    runs = np.cumsum(
        np.r_[True, (np.diff(indices) != 1) | (np.diff(labels[indices]) != 0)]
    )
    positions = np.unique(np.r_[indices, indices + 1])
    native = fixes.iloc[positions]
    lat, lon = enu_to_geodetic(native.E, native.N, native.z, LocalFrame(*origin))
    x, y = project(lon, lat)
    xy = np.column_stack((x, y))
    west, south, east, north = FRANCE_EXTENT
    inside = (
        np.isfinite(xy).all(axis=1)
        & (lon >= west)
        & (lon < east)
        & (lat >= south)
        & (lat < north)
    )
    left, right = (
        np.searchsorted(positions, indices),
        np.searchsorted(positions, indices + 1),
    )
    valid = inside[left] & inside[right]
    return edge_run_cells(xy[left[valid]], xy[right[valid]], runs[valid])


def _task(task):
    discipline, fid, fixes, segments, origin, by_time = task
    return discipline, fid, flight_activity(fixes, segments, origin, by_time=by_time)


def _validated_products(index, source=segmentation.VILPELLET.key):
    """Require complete saved runs from the current cleaned input and protocol.

    Vilpellet's runs are checked against his configuration and the native fixes. The
    saved runs of another segmentation carry their own parameters: they must belong to
    the discipline, postdate the cleaned archive and share flights with the census.
    """
    own = source == segmentation.VILPELLET.key
    label = segmentation.get(source).label
    config = load_vilpellet_config()
    if own and (
        config.recommendations.active or config.input_policy.source != "cleaned"
    ):
        raise ValueError("Climb census requires the unfiltered cleaned-input protocol")
    products = {}
    fingerprint = [activity_version(source), index.signature]
    for name in index.disciplines:
        root = DISCIPLINES[name].config().derived_dir
        saved = segmentation.paths(DISCIPLINES[name], source)
        runs, coverage = saved["runs"], saved["coverage"]
        record = json.loads(saved["parameters"].read_text(encoding="utf-8"))
        if own:
            expected = {
                "discipline": name,
                "alpha_straight_rad": config.alpha_for(name),
                "state_names": list(config.state_names),
                "features": asdict(config.features),
                "majority_fixes": config.majority_fixes,
                "eligibility": {
                    "max_mean_dt_s": config.eligibility.max_mean_dt_s,
                    "min_fixes": config.eligibility.min_fixes,
                },
                "input": asdict(config.input_policy),
                "recommendations_active": False,
                "parameters": {
                    key: getattr(config.parameters, key).tolist()
                    for key in ("emission", "transition", "initial")
                },
            }
            if any(record.get(key) != value for key, value in expected.items()):
                raise ValueError(f"Saved {label} parameters differ for {name}")
            if not (
                config.eligibility.require_unique_times
                and config.eligibility.require_increasing_times
            ):
                raise ValueError(f"Saved runs require the standard {label} time guards")
        else:
            expected = {"discipline": DISCIPLINES[name].slug}
            if record.get("discipline") != expected["discipline"]:
                raise ValueError(f"Saved {label} parameters differ for {name}")
        native = root / "fixes.parquet"
        if (
            min(runs.stat().st_mtime_ns, coverage.stat().st_mtime_ns)
            < native.stat().st_mtime_ns
        ):
            raise ValueError(f"Regenerate stale {label} products for {name}")
        support = pq.read_table(
            coverage, columns=["flight_id", "n_native_fixes"]
        ).to_pandas()
        if own and (
            int(support.n_native_fixes.sum())
            != pq.ParquetFile(native).metadata.num_rows
        ):
            raise ValueError(f"Incomplete {label} coverage for {name}")
        with connect(index.path) as db:
            flights = {
                r[0]
                for r in db.execute(
                    "SELECT flight_id FROM flights WHERE discipline=?", (name,)
                )
            }
        covered = set(support.flight_id.astype(str))
        # The other segmentations come from their own preprocessing of the tracks:
        # they may miss census flights (these stay unclassified) and label others.
        if (covered != flights) if own else not (covered & flights):
            raise ValueError(f"{label} flight coverage differs from census for {name}")
        products[name] = runs
        data_root = DISCIPLINES[name].config().data_root
        for path in (runs, coverage, saved["parameters"]):
            fingerprint.append(file_identity(path, data_root))
        fingerprint.append(expected)
    from .thermal_cache import segmentation_signature

    fingerprint.append(segmentation_signature(index, source))
    return products, hashlib.sha256(
        json.dumps(fingerprint, sort_keys=True).encode()
    ).hexdigest()


def prepare_activity(
    index, *, source=segmentation.VILPELLET.key, workers=4, progress=print
):
    """Resume a census of all in-area flights, independent of selected cells.

    The census counts the climb runs of segmentation ``source`` and is saved in its
    own file (:func:`activity_path`).
    """
    label = segmentation.get(source).label
    by_time = source != segmentation.VILPELLET.key
    products, signature = _validated_products(index, source)
    path = activity_path(index.path, source)
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = "1"
    with (
        connect(path, timeout=120) as db,
        connect(index.path) as census,
    ):
        db.executescript("""
            CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT);
            CREATE TABLE IF NOT EXISTS completed(discipline TEXT,flight_id TEXT,
                PRIMARY KEY(discipline,flight_id));
            CREATE TABLE IF NOT EXISTS activity(discipline TEXT,flight_id TEXT,
                ix INTEGER,iy INTEGER,climb_runs INTEGER,
                PRIMARY KEY(discipline,flight_id,ix,iy));
        """)
        if db.execute(
            "SELECT value FROM metadata WHERE key='signature'"
        ).fetchone() != (signature,):
            db.executescript(
                "DELETE FROM activity; DELETE FROM completed; DELETE FROM metadata;"
            )
            db.execute("INSERT INTO metadata VALUES ('signature',?)", (signature,))
            db.commit()
        if db.execute("SELECT value FROM metadata WHERE key='ready'").fetchone() == (
            "1",
        ):
            progress(f"National {label} climb census already complete")
            return path
        pending = set()
        completed = 0
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:

            def collect():
                nonlocal pending, completed
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    name, fid, rows = future.result()
                    db.executemany(
                        "INSERT INTO activity VALUES (?,?,?,?,?)",
                        [(name, fid, *row) for row in rows],
                    )
                    db.execute("INSERT INTO completed VALUES (?,?)", (name, fid))
                    completed += 1
                db.commit()
                progress(f"Mapped {completed:,} {label} flights in this run")

            for name, runs_path in products.items():
                saved = {
                    r[0]
                    for r in db.execute(
                        "SELECT flight_id FROM completed WHERE discipline=?", (name,)
                    )
                }
                rows = census.execute(
                    """SELECT flight_id,lat0,lon0,alt0 FROM flights
                    WHERE discipline=? AND flight_id IN
                    (SELECT flight_id FROM visits WHERE discipline=?)""",
                    (name, name),
                ).fetchall()
                origins = {
                    fid: (lat, lon, alt)
                    for fid, lat, lon, alt in rows
                    if fid not in saved
                }
                segments = pq.read_table(
                    runs_path,
                    columns=["flight_id", "segment_id", "t_start", "t_end", "n_fixes"],
                    filters=[("phase", "=", "climb")],
                ).to_pandas()
                segments.flight_id = segments.flight_id.astype(str)
                segments = segments.loc[segments.flight_id.isin(origins)]
                by_flight = {
                    fid: frame[["segment_id", "t_start", "t_end", "n_fixes"]].to_numpy()
                    for fid, frame in segments.groupby("flight_id", sort=False)
                }
                no_climbs = origins.keys() - by_flight.keys()
                db.executemany(
                    "INSERT INTO completed VALUES (?,?)",
                    [(name, fid) for fid in no_climbs],
                )
                db.commit()
                progress(
                    f"{name}: {len(by_flight):,} flights with saved climbs to map; "
                    f"{len(no_climbs):,} without climbs"
                )
                groups, last = {}, {}
                for fid, group in census.execute(
                    "SELECT flight_id,row_group FROM flight_groups "
                    "WHERE discipline=? ORDER BY row_group",
                    (name,),
                ):
                    if fid in by_flight:
                        groups.setdefault(group, set()).add(fid)
                        last[fid] = group
                archive = pq.ParquetFile(
                    DISCIPLINES[name].config().derived_dir / "fixes.parquet"
                )
                pieces = {}
                for group, identifiers in groups.items():
                    frame = archive.read_row_group(
                        group, columns=["flight_id", "segment_id", "t", "E", "N", "z"]
                    ).to_pandas()
                    frame.flight_id = frame.flight_id.astype(str)
                    frame = frame.loc[frame.flight_id.isin(identifiers)]
                    for fid, part in frame.groupby("flight_id", sort=False):
                        pieces.setdefault(fid, []).append(part.copy())
                        if last[fid] == group:
                            pending.add(
                                pool.submit(
                                    _task,
                                    (
                                        name,
                                        fid,
                                        pd.concat(pieces.pop(fid), ignore_index=True),
                                        by_flight.pop(fid),
                                        origins[fid],
                                        by_time,
                                    ),
                                )
                            )
                            if len(pending) >= workers * 2:
                                collect()
                if pieces or by_flight:
                    raise ValueError(f"Incomplete geometry for {label} climb census")
                while pending:
                    collect()
        expected = census.execute(
            "SELECT count(*) FROM (SELECT DISTINCT discipline,flight_id FROM visits)"
        ).fetchone()[0]
        actual = db.execute("SELECT count(*) FROM completed").fetchone()[0]
        if actual != expected:
            raise ValueError(f"Incomplete climb census: {actual}/{expected} flights")
        db.execute("CREATE INDEX IF NOT EXISTS activity_by_cell ON activity(ix,iy)")
        db.execute("INSERT INTO metadata VALUES ('ready','1')")
        db.execute(
            "INSERT INTO metadata VALUES ('version',?)", (activity_version(source),)
        )
        db.commit()
    progress(f"National {label} climb census ready: {path}")
    return path
