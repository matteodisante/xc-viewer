"""On-demand neighbouring cells, with official IGN terrain and persistent climbs.

Used by the offsite measurement scripts. The thermal-plane viewer does not call
it: Zoom - reads the neighbourhoods saved by thermal_neighbours.py.

The twelve published cells stay read-only. Explicit navigation uses the saved
archive census and processes missing whole flights in the worker thread. Reuse
the same decoder and edge cache as offline preparation; never infer neighbours
from the subset of flights or intersections in the starting cell.
"""

from __future__ import annotations

import io
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from PIL import Image

from .sqlite import connect
from .thermal_daily import PARIS
from .thermal_geometry import ThermalCell
from .thermal_ground import cell_ground, fetch_terrain_reference
from .thermal_imagery import acquisition_dates, fetch_image
from .thermal_store import CancelledError, PlaneData, load_store


def _check(cancel):
    """Cancel between network requests, row groups or complete flight products."""
    if cancel is not None and cancel.is_set():
        raise CancelledError("Cancelled")


class ThermalExplorer:
    """Delegate published cells and load additional cells only on explicit request."""

    def __init__(self, store):
        """Keep the standalone store usable without connected source archives."""
        self.store = store
        self._extras = {}
        self._backgrounds = {}
        self._census = None

    def __getattr__(self, name):
        """Preserve the standalone store interface for the ranked cells."""
        return getattr(self.store, name)

    def _index(self):
        """Require a current archive census only when leaving the saved selection."""
        from .thermal_index import load_saved_index

        if self._census is None:
            self._census = load_saved_index(
                path=self.path.with_name("thermal-cells.sqlite3")
            )
        if self._census is None:
            raise ValueError(
                "Neighbour exploration needs the connected processed archives "
                "and a current thermal-cells.sqlite3 census."
            )
        return self._census

    def resolve_cell(
        self, ix, iy, *, kind="topography", progress=lambda _: None, cancel=None
    ):
        """Load the target square's own population and complete terrain summary."""
        _check(cancel)
        for cell in self.store.cells():
            if (cell.ix, cell.iy) == (ix, iy):
                return cell
        if (ix, iy) in self._extras:
            cell = self._extras[ix, iy][0]
        else:
            index = self._index()
            progress(f"Reading all flights crossing cell {ix}/{iy}")
            with connect(index.path) as db:
                flights, maximum = db.execute(
                    "SELECT COUNT(*),MAX(max_alt) FROM visits WHERE ix=? AND iy=?",
                    (ix, iy),
                ).fetchone()
            cell = ThermalCell(ix, iy, "Neighbour", flights, 0, 0, maximum or 0)
            progress(f"Loading official IGN terrain for cell {ix}/{iy}")
            reference = fetch_terrain_reference(
                cell, self.path.parent / "exploration/terrain"
            )
            cell = replace(
                cell,
                ground_m=cell_ground(reference),
                max_alt_m=maximum if maximum is not None else cell_ground(reference),
            )
            _check(cancel)
            self._extras[ix, iy] = (cell, reference)
        self.prepare_background(cell, kind, progress=progress, cancel=cancel)
        return cell

    def terrain_reference(self, cell):
        """Return the same attributed terrain metadata for ranked and explored cells."""
        item = self._extras.get((cell.ix, cell.iy))
        return item[1] if item else self.store.terrain_reference(cell)

    def reference_audit(self, cell):
        """Explored cells use DEM minima and carry no launch-based ranking audit."""
        if (cell.ix, cell.iy) in self._extras:
            return None
        return self.store.reference_audit(cell)

    def defaults(self, cell):
        """Supply a bounded initial height for a newly explored square."""
        if (cell.ix, cell.iy) in self._extras:
            return (self.store.time_extent()[0], min(500, cell.max_agl_m))
        return self.store.defaults(cell)

    def summer_days(self, cell):
        """Compute the same distinct visitor/day census for an explored square."""
        if (cell.ix, cell.iy) not in self._extras:
            return self.store.summer_days(cell)
        frame = self._index().flights(cell)
        counts = Counter()
        for row in frame.itertuples(index=False):
            origin = row.start_utc + row.trim_start
            if not np.isfinite(origin):
                continue
            day = datetime.fromtimestamp(origin + row.t_min, PARIS).date()
            last = datetime.fromtimestamp(origin + row.t_max, PARIS).date()
            while day <= last:
                if day.month in (6, 7, 8):
                    counts[day.isoformat()] += 1
                day += timedelta(days=1)
        return sorted(counts.items(), key=lambda item: (-item[1], item[0]))

    def prepare_background(self, cell, kind, *, progress=lambda _: None, cancel=None):
        """Download only the requested background, outside the GUI thread."""
        if (cell.ix, cell.iy) not in self._extras or kind == "none":
            return
        key = (cell.ix, cell.iy, kind)
        if key in self._backgrounds:
            return
        _check(cancel)
        progress(f"Loading IGN {kind} map for cell {cell.ix}/{cell.iy}")
        if kind == "relief":
            from .thermal_relief import fetch_relief

            payload, info = fetch_relief(
                self.path.parent, cell.bounds, 2154, (1500, 1500)
            )
        else:
            info, payload = fetch_image(
                self.path.parent / "imagery",
                cell.bounds,
                "EPSG:2154",
                (4000, 4000),
                kind,
            )
            if kind == "aerial":
                info["acquisition_dates"], info["acquisition_source"] = (
                    acquisition_dates(cell.bounds, "EPSG:2154")
                )
        _check(cancel)
        # Compressed bytes keep neighbouring cells from retaining many full rasters.
        self._backgrounds[key] = (payload, info)

    def background(self, cell=None, kind="relief", key=None):
        """Return ready imagery only; redraws never perform a network request."""
        if cell is not None and (cell.ix, cell.iy) in self._extras:
            saved = self._backgrounds.get((cell.ix, cell.iy, kind))
            if saved:
                payload, info = saved
                return np.asarray(Image.open(io.BytesIO(payload)).convert("RGB")), info
            return None
        return self.store.background(cell, kind, key)

    def needs_background(self, cell, kind):
        """Check readiness without decoding a full raster on the GUI thread."""
        return (
            kind != "none"
            and (cell.ix, cell.iy) in self._extras
            and (cell.ix, cell.iy, kind) not in self._backgrounds
        )

    def read_plane(
        self,
        cell,
        start,
        end,
        source,
        *,
        progress=lambda _: None,
        cancel=None,
        use_edges=False,
    ):
        """Use every eligible visitor; decode whole flights before spatial/time cuts."""
        if (cell.ix, cell.iy) not in self._extras:
            return self.store.read_plane(
                cell,
                start,
                end,
                source,
                progress=progress,
                cancel=cancel,
                use_edges=use_edges,
            )
        if source != "vilpellet" or end < start:
            raise ValueError("Invalid segmentation or time interval")
        from .thermal_cache import ClimbCache

        index = self._index()
        visitors = index.flights(cell)
        origin = visitors.start_utc + visitors.trim_start
        unknown = int(origin.isna().sum())
        visits = pd.DataFrame(
            {"start": origin + visitors.t_min, "end": origin + visitors.t_max}
        )
        use = visits.end.ge(start) & visits.start.le(end)
        rows = visitors.loc[use]
        decoded = unclassified = unavailable = cached = 0
        frames, missing = [], []
        with ClimbCache(index, source) as cache:

            def collect(row, status, edges):
                """Count complete flight status and retain time-overlapping edges."""
                nonlocal decoded, unclassified, unavailable
                decoded += status != "unavailable"
                unclassified += status == "unclassified"
                unavailable += status == "unavailable"
                if len(edges):
                    edges = edges.loc[
                        (edges.utc1 >= start) & (edges.utc0 <= end)
                    ].copy()
                    edges["flight_id"], edges["discipline"] = (
                        row["flight_id"],
                        row["discipline"],
                    )
                    frames.append(edges)

            for row in rows.to_dict("records"):
                _check(cancel)
                product = cache.get(row["discipline"], row["flight_id"], cell)
                if product is None:
                    missing.append(row)
                else:
                    cached += 1
                    collect(row, *product)
            progress(
                f"{cached:,} flights cached; {len(missing):,} whole flights to label"
            )
            for row, fixes in _flight_fixes(index, missing, cancel):
                from .thermal_prepare import _decode

                # Cache adjacent squares from the same complete flight too.
                targets = [
                    (cache.key, replace(cell, ix=cell.ix + dx, iy=cell.iy + dy))
                    for dx in (-1, 0, 1)
                    for dy in (-1, 0, 1)
                ]
                products = _decode((row, fixes, {source: targets}))
                cache.db.executemany(
                    "INSERT OR REPLACE INTO climbs VALUES (?,?,?,?,?,?,?)", products
                )
                cache.db.commit()
                collect(row, *cache.get(row["discipline"], row["flight_id"], cell))
                if (decoded + unavailable) % 25 == 0:
                    progress(f"Loaded {decoded + unavailable:,}/{len(rows):,} flights")
            _check(cancel)
        return PlaneData(
            pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(),
            len(rows),
            decoded,
            unclassified,
            unavailable,
            unknown,
            cached,
            visits=visits.loc[use].reset_index(drop=True),
        )

    def read_neighborhood(
        self, cell, start, end, source, kind, *, progress=lambda _: None, cancel=None
    ):
        """Prepare the 3 x 3 cells supporting bounded pan/zoom at one absolute plane."""
        self._index()
        targets = []
        with ThreadPoolExecutor(max_workers=3) as pool:
            pending = [
                pool.submit(
                    self.resolve_cell,
                    cell.ix + dx,
                    cell.iy + dy,
                    kind=kind,
                    progress=progress,
                    cancel=cancel,
                )
                for dy in (-1, 0, 1)
                for dx in (-1, 0, 1)
            ]
            for future in as_completed(pending):
                _check(cancel)
                targets.append(future.result())
        result = {}
        for target in sorted(targets, key=lambda c: (c.iy, c.ix)):
            _check(cancel)
            plane = self.read_plane(
                target,
                start,
                end,
                source,
                progress=progress,
                cancel=cancel,
                use_edges=True,
            )
            result[target.ix, target.iy] = (target, plane)
        return result


def _flight_fixes(index, rows, cancel):
    """Read each required parquet row group once and assemble whole flights."""
    import pyarrow.parquet as pq

    from .core.disciplines import DISCIPLINES

    with connect(index.path) as db:
        for discipline in sorted({r["discipline"] for r in rows}):
            wanted = {r["flight_id"]: r for r in rows if r["discipline"] == discipline}
            groups, last = {}, {}
            for fid, group in db.execute(
                "SELECT flight_id,row_group FROM flight_groups "
                "WHERE discipline=? ORDER BY row_group",
                (discipline,),
            ):
                if fid in wanted:
                    groups.setdefault(group, set()).add(fid)
                    last[fid] = group
            if set(last) != set(wanted):
                raise ValueError(
                    "Flight census lacks parquet locations; rebuild the archive census"
                )
            archive = pq.ParquetFile(
                DISCIPLINES[discipline].config().derived_dir / "fixes.parquet"
            )
            pieces = {}
            for group, identifiers in groups.items():
                _check(cancel)
                frame = archive.read_row_group(group).to_pandas()
                frame["flight_id"] = frame.flight_id.astype(str)
                for fid, part in frame.loc[frame.flight_id.isin(identifiers)].groupby(
                    "flight_id", sort=False
                ):
                    pieces.setdefault(fid, []).append(part.copy())
                    if last[fid] == group:
                        _check(cancel)
                        yield wanted[fid], pd.concat(pieces.pop(fid), ignore_index=True)


def load_explorer():
    """Open the standalone selection with optional on-demand neighbour support."""
    store = load_store()
    return ThermalExplorer(store) if store is not None else None
