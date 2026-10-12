"""Cached archive index for the four busiest metric cells and their climb planes.

The expensive pass scans geometry, never phase products. Each (discipline, flight,
cell) is counted once, irrespective of repeated visits. Raw first fixes supply only
the separate ground reference and the UTC origin. An atomic SQLite cache keeps the
GUI usable on later launches without scanning the multi-gigabyte archive again.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from threading import Event

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import data
from .catalog_index import resolve_igc_path
from .core.config import load_preproc_config
from .core.disciplines import DISCIPLINES, Discipline
from .core.igc import first_fix
from .core.naming import igc_path
from .core.preproc.enu import LocalFrame
from .fingerprint import file_identity
from .geodesy import enu_to_geodetic
from .geography import FRANCE_EXTENT, TERRAIN_ORDER, classify_terrain
from .locking import exclusive
from .sqlite import connect
from .thermal_geometry import CELL_M, ThermalCell, cell_visits, climb_edges, project

INDEX_VERSION = 1


def cache_path(disciplines: list[Discipline] | None = None) -> Path:
    """Place both caches on the archive disk, never silently on the system disk."""
    override = os.environ.get("XC_VIEWER_CACHE_DIR")
    if override:
        return Path(override).expanduser() / "thermal-cells.sqlite3"
    if disciplines is None:
        disciplines = [d for d in DISCIPLINES.values() if d.derived_dir() is not None]
    if not disciplines:
        raise FileNotFoundError("Choose a data folder with the processed archives")
    return (
        disciplines[0].config().derived_dir
        / "viewer/thermal-planes"
        / "thermal-cells.sqlite3"
    )


def load_saved_index(
    disciplines: list[Discipline] | None = None,
    *,
    path: Path | None = None,
) -> ThermalIndex | None:
    """Read a valid prepared index; opening a viewer never starts a census."""
    if disciplines is None:
        disciplines = [d for d in DISCIPLINES.values() if d.derived_dir() is not None]
    if not disciplines:
        return None
    path = cache_path(disciplines) if path is None else path
    if not path.is_file():
        return None
    signature = archive_signature(disciplines)
    try:
        with connect(path) as db:
            saved = db.execute("SELECT signature FROM metadata").fetchone()
        if saved == (signature,):
            return ThermalIndex(path, tuple(d.name for d in disciplines), signature)
    except sqlite3.DatabaseError:
        pass
    return None


class CancelledError(Exception):
    """The user cancelled a background archive operation."""


def _check_cancel(cancel: Event | None) -> None:
    """Stop between bounded pieces of I/O or decoding work."""
    if cancel is not None and cancel.is_set():
        raise CancelledError("Cancelled")


def archive_signature(disciplines: list[Discipline]) -> str:
    """Fingerprint the archive inputs, geometry/metadata versions and ground policy.

    Portable (see :mod:`xc_viewer.fingerprint`): bump ``INDEX_VERSION`` when
    the census code changes what it saves.
    """
    parts: list = [INDEX_VERSION]
    cfg = load_preproc_config()
    parts.extend([cfg.fix.min_altitude_m, cfg.fix.max_altitude_m])
    for disc in disciplines:
        config = disc.config()
        parts.append(disc.name)
        for path in (
            config.catalog_path,
            config.derived_dir / "fixes.parquet",
            config.derived_dir / "flights_meta.parquet",
        ):
            parts.append(file_identity(path, config.data_root))
    return hashlib.sha256(json.dumps(parts).encode()).hexdigest()


@dataclass(frozen=True)
class ThermalIndex:
    """A complete cache and its source disciplines; no open cross-thread handle."""

    path: Path
    disciplines: tuple[str, ...]
    signature: str | None = None

    def cells(self) -> list[ThermalCell]:
        """Busiest crossing cell in each band of median raw launch altitude.

        Cells without an internal usable GNSS launch have no defensible ground
        reference and cannot be candidates. All crossing flights still contribute
        to the counts, including flights launched outside the candidate cell.
        """
        with connect(self.path) as db:
            if db.execute(
                "SELECT 1 FROM sqlite_master WHERE name='selected_cells'"
            ).fetchone():
                return [
                    ThermalCell(**json.loads(row[0]))
                    for row in db.execute(
                        "SELECT payload FROM selected_cells ORDER BY position"
                    )
                ]
            starts = pd.read_sql_query(
                "SELECT launch_x AS ix, launch_y AS iy, launch_alt FROM flights "
                "WHERE launch_alt IS NOT NULL",
                db,
            )
            visits = pd.read_sql_query(
                "SELECT ix, iy, COUNT(*) AS flights, MAX(max_alt) AS max_alt "
                "FROM visits GROUP BY ix, iy",
                db,
            )
        if starts.empty or visits.empty:
            return []
        ground = starts.groupby(["ix", "iy"], as_index=False).agg(
            ground=("launch_alt", "median"), launches=("launch_alt", "size")
        )
        cells = visits.merge(ground, on=["ix", "iy"])
        cells["terrain"] = classify_terrain(cells.ground.to_numpy())
        cells = cells.sort_values(
            ["flights", "ix", "iy"], ascending=[False, True, True]
        )
        result = []
        for name in TERRAIN_ORDER:
            group = cells.loc[cells.terrain == name]
            if group.empty:
                continue
            c = group.iloc[0]
            result.append(
                ThermalCell(
                    int(c.ix),
                    int(c.iy),
                    name,
                    int(c.flights),
                    int(c.launches),
                    float(c.ground),
                    float(c.max_alt),
                )
            )
        return result

    def save_cells(self) -> None:
        """Persist the four reductions so startup only reads four small records."""
        with connect(self.path) as db:
            if db.execute(
                "SELECT 1 FROM sqlite_master WHERE name='selected_cells'"
            ).fetchone():
                return
        cells = self.cells()
        with connect(self.path) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS selected_cells "
                "(position INTEGER PRIMARY KEY, payload TEXT)"
            )
            db.executemany(
                "INSERT OR REPLACE INTO selected_cells VALUES (?,?)",
                [(i, json.dumps(asdict(c))) for i, c in enumerate(cells)],
            )
            db.commit()

    def flights(self, cell: ThermalCell) -> pd.DataFrame:
        """All-time visitors, including those whose raw UTC cannot be recovered."""
        with connect(self.path) as db:
            return pd.read_sql_query(
                "SELECT f.*, v.t_min, v.t_max FROM visits v JOIN flights f "
                "ON f.discipline=v.discipline AND f.flight_id=v.flight_id "
                "WHERE v.ix=? AND v.iy=? ORDER BY f.discipline, f.flight_id",
                db,
                params=(cell.ix, cell.iy),
            )


def _create_tables(db: sqlite3.Connection) -> None:
    """Create a fresh index, including source metadata for archived UTC recovery."""
    db.executescript("""
        CREATE TABLE metadata (signature TEXT);
        CREATE TABLE flights (
            discipline TEXT, flight_id TEXT, path TEXT, start_utc REAL,
            trim_start REAL, lat0 REAL, lon0 REAL, alt0 REAL,
            launch_x INTEGER, launch_y INTEGER, launch_alt REAL,
            PRIMARY KEY (discipline, flight_id)
        );
        CREATE TABLE flight_groups (
            discipline TEXT, flight_id TEXT, row_group INTEGER,
            PRIMARY KEY (discipline, flight_id, row_group)
        );
        CREATE TABLE visits (
            discipline TEXT, flight_id TEXT, ix INTEGER, iy INTEGER,
            t_min REAL, t_max REAL, max_alt REAL,
            PRIMARY KEY (discipline, flight_id, ix, iy)
        );
    """)


def _index_origins(db, disc, meta, progress, cancel) -> None:
    """Read pre-trim launch position/altitude and the true UTC date from raw IGC."""
    catalog = pd.read_csv(
        disc.config().catalog_path, dtype={"flight_id": str}, low_memory=False
    ).drop_duplicates("flight_id", keep="last")
    rows = meta.merge(catalog, on="flight_id", how="left", suffixes=("", "_catalog"))
    cfg = load_preproc_config()
    igc_root = disc.config().igc_dir
    values = []
    for i, row in enumerate(rows.itertuples(index=False)):
        _check_cancel(cancel)
        if i % 500 == 0:
            progress(f"{disc.name}: raw launch fixes {i:,}/{len(rows):,}")
        try:
            path: Path | None = igc_path(
                igc_root, int(row.season_year), str(row.date), row.flight_id
            )
            if path is not None and not path.is_file():
                path = resolve_igc_path(disc, pd.Series(row._asdict()))
            origin = first_fix(path, include_utc=True) if path else None
        except (OSError, ValueError, TypeError, OverflowError):
            path, origin = None, None
        utc = altitude = ix = iy = None
        if origin is not None:
            utc = origin["start_utc"]
            lon, lat, alt = origin["lon"], origin["lat"], origin["gnss_alt"]
            west, south, east, north = FRANCE_EXTENT
            if (
                west <= lon < east
                and south <= lat < north
                and np.isfinite(alt)
                and alt != 0
                and cfg.fix.min_altitude_m <= alt <= cfg.fix.max_altitude_m
            ):
                x, y = project(lon, lat)
                ix, iy, altitude = int(x // CELL_M), int(y // CELL_M), alt
        values.append(
            (
                disc.name,
                str(row.flight_id),
                str(path) if path else None,
                utc,
                row.ground_phase_start_s,
                row.lat0,
                row.lon0,
                row.alt0,
                ix,
                iy,
                altitude,
            )
        )
    db.executemany("INSERT INTO flights VALUES (?,?,?,?,?,?,?,?,?,?,?)", values)
    db.commit()


def _index_geometry(db, disc, meta, progress, cancel) -> None:
    """Stream all supported archived edges, preserving row-group continuations."""
    db.execute("CREATE TABLE IF NOT EXISTS scan_complete (discipline TEXT PRIMARY KEY)")
    if db.execute(
        "SELECT 1 FROM scan_complete WHERE discipline=?", (disc.name,)
    ).fetchone():
        return
    frames = {
        str(r.flight_id): LocalFrame(r.lat0, r.lon0, r.alt0)
        for r in meta.itertuples(index=False)
    }
    archive = pq.ParquetFile(disc.config().derived_dir / "fixes.parquet")
    columns = ["flight_id", "segment_id", "t", "E", "N", "z"]
    carry = pd.DataFrame()
    west, south, east, north = FRANCE_EXTENT

    def consume(flight_id, fixes):
        """Reduce one complete flight to its distinct visited cells."""
        _check_cancel(cancel)
        if flight_id not in frames:
            return
        fixes = fixes.sort_values(["segment_id", "t"], kind="stable").copy()
        lat, lon = enu_to_geodetic(fixes.E, fixes.N, fixes.z, frames[flight_id])
        inside = (lon >= west) & (lon < east) & (lat >= south) & (lat < north)
        if not inside.any():
            return
        x, y = project(lon, lat)
        fixes["x"] = np.where(inside, x, np.nan)
        fixes["y"] = np.where(inside, y, np.nan)
        visits = cell_visits(fixes)
        db.executemany(
            "INSERT INTO visits VALUES (?,?,?,?,?,?,?) "
            "ON CONFLICT(discipline,flight_id,ix,iy) DO UPDATE SET "
            "t_min=MIN(t_min,excluded.t_min), t_max=MAX(t_max,excluded.t_max), "
            "max_alt=MAX(max_alt,excluded.max_alt)",
            [
                (
                    disc.name,
                    flight_id,
                    int(r.ix),
                    int(r.iy),
                    r.t_min,
                    r.t_max,
                    r.max_alt,
                )
                for r in visits.itertuples(index=False)
            ],
        )

    # Re-read the pending boundary flight's first row group on resume. UPSERTs
    # make replaying its already committed neighbours harmless to distinct counts.
    last_group = db.execute(
        "SELECT MAX(row_group) FROM flight_groups WHERE discipline=?", (disc.name,)
    ).fetchone()[0]
    first_group = 0
    if last_group is not None:
        pending_id = str(
            archive.read_row_group(last_group, columns=["flight_id"])
            .column("flight_id")[-1]
            .as_py()
        )
        first_group = db.execute(
            "SELECT MIN(row_group) FROM flight_groups "
            "WHERE discipline=? AND flight_id=?",
            (disc.name, pending_id),
        ).fetchone()[0]
        progress(f"Resuming {disc.name} from saved block {first_group + 1:,}")
    for group in range(first_group, archive.num_row_groups):
        _check_cancel(cancel)
        progress(
            f"{disc.name}: trajectory blocks {group + 1:,}/{archive.num_row_groups:,}"
        )
        frame = archive.read_row_group(group, columns=columns).to_pandas()
        frame["flight_id"] = frame.flight_id.astype(str)
        db.executemany(
            "INSERT OR IGNORE INTO flight_groups VALUES (?,?,?)",
            [(disc.name, fid, group) for fid in frame.flight_id.unique()],
        )
        frame = pd.concat([carry, frame], ignore_index=True)
        last_id = frame.flight_id.iloc[-1]
        carry = frame.loc[frame.flight_id == last_id].copy()
        for fid, flight in frame.loc[frame.flight_id != last_id].groupby(
            "flight_id", sort=False
        ):
            consume(fid, flight)
        db.commit()
    if not carry.empty:
        consume(str(carry.flight_id.iloc[0]), carry)
    db.execute("INSERT OR IGNORE INTO scan_complete VALUES (?)", (disc.name,))
    db.commit()


def build_index(
    disciplines: list[Discipline] | None = None,
    *,
    path: Path | None = None,
    progress: Callable[[str], None] = lambda message: None,
    cancel: Event | None = None,
    force: bool = False,
) -> ThermalIndex:
    """Serialize resumable builds; a second viewer never duplicates the scan."""
    if disciplines is None:
        disciplines = [d for d in DISCIPLINES.values() if d.derived_dir() is not None]
    path = cache_path(disciplines) if path is None else path
    path.parent.mkdir(parents=True, exist_ok=True)
    with exclusive(path, "The cell cache is already being prepared"):
        return _build_index(
            disciplines, path=path, progress=progress, cancel=cancel, force=force
        )


def _build_index(
    disciplines: list[Discipline] | None = None,
    *,
    path: Path | None = None,
    progress: Callable[[str], None] = lambda message: None,
    cancel: Event | None = None,
    force: bool = False,
) -> ThermalIndex:
    """Build once in the data folder, reusing a valid saved census later."""
    if disciplines is None:
        disciplines = [d for d in DISCIPLINES.values() if d.derived_dir() is not None]
    if not disciplines:
        raise FileNotFoundError("No processed flight archive is reachable")
    path = cache_path(disciplines) if path is None else path
    signature = archive_signature(disciplines)
    if path.is_file() and not force:
        try:
            with connect(path) as db:
                cached = db.execute("SELECT signature FROM metadata").fetchone()
            if cached == (signature,):
                index = ThermalIndex(
                    path, tuple(d.name for d in disciplines), signature
                )
                index.save_cells()
                return index
        except sqlite3.DatabaseError:
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".building.sqlite3")
    # A cancelled census is a reusable checkpoint, never a selectable result.
    if temporary.is_file():
        with connect(temporary) as db:
            try:
                pending_signature = db.execute(
                    "SELECT signature FROM build_info"
                ).fetchone()
            except sqlite3.DatabaseError:
                pending_signature = None
        if pending_signature != (signature,) or force:
            temporary.unlink()
    fresh = not temporary.is_file()
    with connect(temporary) as db:
        if fresh:
            _create_tables(db)
            db.execute("CREATE TABLE build_info (signature TEXT)")
            db.execute("INSERT INTO build_info VALUES (?)", (signature,))
            db.commit()
        for disc in disciplines:
            meta = pd.read_parquet(disc.config().derived_dir / "flights_meta.parquet")
            meta["flight_id"] = meta.flight_id.astype(str)
            meta = meta.drop_duplicates("flight_id", keep="last")
            meta = meta.loc[meta.drop_reason.isna()].copy()
            origins = db.execute(
                "SELECT COUNT(*) FROM flights WHERE discipline=?", (disc.name,)
            ).fetchone()[0]
            if origins != len(meta):
                db.execute("DELETE FROM flights WHERE discipline=?", (disc.name,))
                _index_origins(db, disc, meta, progress, cancel)
            _index_geometry(db, disc, meta, progress, cancel)
        db.execute("CREATE INDEX IF NOT EXISTS visits_by_cell ON visits(ix,iy)")
        db.execute("DELETE FROM metadata")
        db.execute("INSERT INTO metadata VALUES (?)", (signature,))
        db.commit()
    ThermalIndex(temporary, tuple(d.name for d in disciplines), signature).save_cells()
    if archive_signature(disciplines) != signature:
        raise RuntimeError("Archive changed while indexing; rebuild the cells")
    temporary.replace(path)
    return ThermalIndex(path, tuple(d.name for d in disciplines), signature)


@dataclass(frozen=True)
class PlaneData:
    """Climb edges for an interval and explicit coverage of the requested flights."""

    edges: pd.DataFrame
    selected: int
    decoded: int
    unclassified: int
    unavailable: int
    unknown_clock: int
    cached: int = 0


def load_plane_data(
    index: ThermalIndex,
    cell: ThermalCell,
    start_utc: float,
    end_utc: float,
    source: str,
    *,
    progress: Callable[[str], None] = lambda message: None,
    cancel: Event | None = None,
    collect_edges: bool = True,
) -> PlaneData:
    """Read saved climbs; decode and persist only previously unprepared flights."""
    from .thermal_cache import ClimbCache

    if source != "vilpellet":
        raise ValueError(f"Unknown segmentation: {source}")
    if (
        index.signature is not None
        and archive_signature([DISCIPLINES[name] for name in index.disciplines])
        != index.signature
    ):
        raise RuntimeError("Archive changed. Build / refresh the cell index first.")
    visitors = index.flights(cell)
    origins = visitors.start_utc + visitors.trim_start
    unknown = int(origins.isna().sum())
    selected = visitors.loc[
        (origins + visitors.t_max >= start_utc) & (origins + visitors.t_min <= end_utc)
    ]
    edges = []
    decoded = unclassified = unavailable = cached = 0
    archives = {}

    @lru_cache(maxsize=2)
    def read_group(discipline, group):
        """Reuse neighbouring selected flights from the same physical row group."""
        if discipline not in archives:
            archives[discipline] = pq.ParquetFile(
                DISCIPLINES[discipline].config().derived_dir / "fixes.parquet"
            )
        return archives[discipline].read_row_group(group).to_pandas()

    loader = data.load_vilpellet_phases
    with connect(index.path) as db, ClimbCache(index, source) as cache:
        for i, row in enumerate(selected.itertuples(index=False)):
            _check_cancel(cancel)
            saved = cache.get(
                row.discipline, row.flight_id, cell, geometry=collect_edges
            )
            if saved is not None:
                status, flight_edges = saved
                cached += 1
                unavailable += status == "unavailable"
                decoded += status != "unavailable"
                unclassified += status == "unclassified"
                if collect_edges and not flight_edges.empty:
                    flight_edges = flight_edges.loc[
                        (flight_edges.utc1 >= start_utc)
                        & (flight_edges.utc0 <= end_utc)
                    ].copy()
                    flight_edges["flight_id"] = row.flight_id
                    flight_edges["discipline"] = row.discipline
                    edges.append(flight_edges)
                progress(f"Reading saved cache: {i + 1}/{len(selected)} ({source})")
                continue
            progress(f"Preparing once: {i + 1}/{len(selected)} ({source})")
            groups = db.execute(
                "SELECT row_group FROM flight_groups "
                "WHERE discipline=? AND flight_id=? "
                "ORDER BY row_group",
                (row.discipline, row.flight_id),
            ).fetchall()
            pieces = []
            for (group,) in groups:
                _check_cancel(cancel)
                frame = read_group(row.discipline, group)
                pieces.append(frame.loc[frame.flight_id.astype(str) == row.flight_id])
            if not pieces:
                unavailable += 1
                continue
            fixes = pd.concat(pieces, ignore_index=True).sort_values(
                ["segment_id", "t"], kind="stable"
            )
            phase = loader(fixes, DISCIPLINES[row.discipline])
            if phase is None:
                unavailable += 1
                cache.put(
                    row.discipline, row.flight_id, cell, "unavailable", pd.DataFrame()
                )
                continue
            decoded += 1
            status = "decoded"
            if not phase.fixes.phase.ne("unclassified").any():
                unclassified += 1
                status = "unclassified"
            fixes = phase.fixes.copy()
            lat, lon = enu_to_geodetic(
                fixes.E, fixes.N, fixes.z, LocalFrame(row.lat0, row.lon0, row.alt0)
            )
            fixes["x"], fixes["y"] = project(lon, lat)
            fixes["utc"] = fixes.t + row.trim_start + row.start_utc
            flight_edges = climb_edges(fixes, cell)
            # Persist the entire flight's cell geometry, not the requested window.
            # A new day/time range must never require the same decode again.
            cache.put(row.discipline, row.flight_id, cell, status, flight_edges)
            if collect_edges:
                flight_edges = flight_edges.loc[
                    (flight_edges.utc1 >= start_utc) & (flight_edges.utc0 <= end_utc)
                ].copy()
                flight_edges["flight_id"] = row.flight_id
                flight_edges["discipline"] = row.discipline
                edges.append(flight_edges)
    return PlaneData(
        pd.concat(edges, ignore_index=True) if edges else pd.DataFrame(),
        len(selected),
        decoded,
        unclassified,
        unavailable,
        unknown,
        cached,
    )


def prepare_cache(
    *,
    progress: Callable[[str], None] = lambda message: None,
    cancel: Event | None = None,
    force: bool = False,
) -> ThermalIndex:
    """Prepare the census and Vilpellet climbs for all four cells, across all dates.

    Each completed climb product is committed separately: cancel and resume only
    computes missing flights. This is an explicit preparation action, never a task
    automatically started when opening the viewer.
    """
    index = build_index(progress=progress, cancel=cancel, force=force)
    for cell in index.cells():
        for source in ("vilpellet",):
            _check_cancel(cancel)

            def cell_progress(message: str, terrain: str = cell.terrain) -> None:
                """Identify the cell being prepared without changing cached inputs."""
                progress(f"{terrain}: {message}")

            load_plane_data(
                index,
                cell,
                -np.inf,
                np.inf,
                source,
                collect_edges=False,
                progress=cell_progress,
                cancel=cancel,
            )
    return index
