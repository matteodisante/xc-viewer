"""Offline, resumable preparation; never imported by the viewer widget."""

from __future__ import annotations

import io
import multiprocessing
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import data
from .core.disciplines import DISCIPLINES
from .core.preproc.enu import LocalFrame
from .geodesy import enu_to_geodetic
from .sqlite import connect
from .thermal_cache import EDGE_COLUMNS, ClimbCache
from .thermal_geometry import climb_edges, project


def _decode(task):
    """Decode each complete flight once per source, then extract every target cell."""
    row, fixes, requests = task
    fixes = fixes.sort_values(["segment_id", "t"], kind="stable")
    result = []
    for _source, targets in requests.items():
        phase = data.load_vilpellet_phases(fixes, DISCIPLINES[row["discipline"]])
        status = "unavailable"
        if phase is not None:
            frame = phase.fixes.copy()
            status = (
                "decoded" if frame.phase.ne("unclassified").any() else "unclassified"
            )
            lat, lon = enu_to_geodetic(
                frame.E,
                frame.N,
                frame.z,
                LocalFrame(row["lat0"], row["lon0"], row["alt0"]),
            )
            frame["x"], frame["y"] = project(lon, lat)
            frame["utc"] = frame.t + row["trim_start"] + row["start_utc"]
        for key, cell in targets:
            edges = climb_edges(frame, cell) if phase is not None else pd.DataFrame()
            buffer = io.BytesIO()
            np.savez_compressed(
                buffer, edges=edges.reindex(columns=EDGE_COLUMNS).to_numpy(dtype=float)
            )
            result.append(
                (
                    key,
                    row["discipline"],
                    row["flight_id"],
                    cell.ix,
                    cell.iy,
                    status,
                    buffer.getvalue(),
                )
            )
    return result


def prepare_climbs(
    index, *, workers=4, progress=print, cells=None, sources=("vilpellet",)
):
    """Stream needed row groups once; parent alone commits complete products.

    ``cells`` narrows the work, e.g. to explored squares; by default every selected
    cell is prepared.
    """
    todo = {}
    for source in sources:
        with ClimbCache(index, source) as cache:
            existing = set(
                cache.db.execute(
                    "SELECT discipline,flight_id,ix,iy FROM climbs WHERE cache_key=?",
                    (cache.key,),
                )
            )
            for cell in index.cells() if cells is None else cells:
                for row in index.flights(cell).to_dict("records"):
                    if not np.isfinite(row["start_utc"] + row["trim_start"]):
                        continue
                    if (
                        row["discipline"],
                        row["flight_id"],
                        cell.ix,
                        cell.iy,
                    ) in existing:
                        continue
                    identity = (row["discipline"], row["flight_id"])
                    entry = todo.setdefault(identity, (row, {}))
                    entry[1].setdefault(source, []).append((cache.key, cell))
    total = len(todo)
    progress(
        f"{total:,} distinct flights still need products (requested methods and cells)"
    )
    if not total:
        return
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = "1"
    completed = 0
    pending = set()
    with (
        connect(index.path) as census,
        # A read-only audit can briefly hold a shared lock. Wait for it instead
        # of aborting a long preparation after SQLite's default five seconds.
        connect(index.path.with_name("thermal-climbs.sqlite3"), timeout=120) as saved,
        ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool,
    ):

        def collect():
            nonlocal pending, completed
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                saved.executemany(
                    "INSERT OR REPLACE INTO climbs VALUES (?,?,?,?,?,?,?)",
                    future.result(),
                )
                completed += 1
            saved.commit()
            progress(f"Prepared {completed:,}/{total:,} distinct flights")

        for discipline in index.disciplines:
            wanted = {fid for disc, fid in todo if disc == discipline}
            if not wanted:
                continue
            groups = {}
            last_group = {}
            for fid, group in census.execute(
                "SELECT flight_id,row_group FROM flight_groups "
                "WHERE discipline=? ORDER BY row_group",
                (discipline,),
            ):
                if fid in wanted:
                    groups.setdefault(group, set()).add(fid)
                    last_group[fid] = group
            archive = pq.ParquetFile(
                DISCIPLINES[discipline].config().derived_dir / "fixes.parquet"
            )
            pieces = {}
            for group, identifiers in groups.items():
                frame = archive.read_row_group(group).to_pandas()
                frame["flight_id"] = frame.flight_id.astype(str)
                frame = frame.loc[frame.flight_id.isin(identifiers)]
                for fid, part in frame.groupby("flight_id", sort=False):
                    pieces.setdefault(fid, []).append(part.copy())
                    if last_group[fid] == group:
                        row, requests = todo[(discipline, fid)]
                        pending.add(
                            pool.submit(
                                _decode,
                                (
                                    row,
                                    pd.concat(pieces.pop(fid), ignore_index=True),
                                    requests,
                                ),
                            )
                        )
                        if len(pending) >= workers * 2:
                            collect()
            if pieces:
                raise RuntimeError("Incomplete archived flight at end of scan")
        while pending:
            collect()
