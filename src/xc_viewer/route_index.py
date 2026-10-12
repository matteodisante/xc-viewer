"""Small endpoint census and row-group lookup for cleaned flight comparisons.

Only the first and last retained fixes define a directed pair of 10 km cells.
The stored 5 km endpoint lattice is coarsened exactly on read, so the existing
census remains usable without rescanning the trajectory archive.
The census streams six Parquet columns once, never copies the trajectory archive,
and checkpoints each row group. No raw IGC or segmentation product is read.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .core.disciplines import DISCIPLINES
from .core.preproc.enu import LocalFrame
from .fingerprint import file_identity
from .geodesy import enu_to_geodetic
from .geography import FRANCE_EXTENT
from .locking import exclusive
from .sqlite import connect
from .thermal_geometry import CELL_M as INDEX_CELL_M
from .thermal_geometry import project
from .thermal_index import _check_cancel, cache_path

VERSION = 1
ROUTE_CELL_M = 10000.0
MAX_FLIGHTS = 300
DISTANCE_PRESETS_KM = (50, 100, 200, 300)
FIX_COLUMNS = ["flight_id", "segment_id", "t", "E", "N", "z"]
PAIR_COLUMNS = ["start_ix", "start_iy", "end_ix", "end_iy"]
META_COLUMNS = ["flight_id", "lat0", "lon0", "alt0", "n_segments_kept", "drop_reason"]


def available_archives():
    """Return configured disciplines with complete processed geometry available."""
    return [d for d in DISCIPLINES.values() if d.derived_dir() is not None]


def route_cache_path(disciplines=None):
    """Keep the compact cache and terrain on the configured archive disk."""
    return cache_path(disciplines).with_name("route-cells.sqlite3")


def archive_signature(disciplines):
    """Reject incomplete archives and fingerprint the inputs and the code version.

    Portable (see :mod:`xc_viewer.fingerprint`): bump ``VERSION`` when the
    index code, geodesy.py or thermal_geometry.py changes what is saved.
    """
    parts = [VERSION]
    for disc in disciplines:
        config = disc.config()
        root = config.derived_dir
        if (root / ".run_incomplete").exists():
            raise ValueError(f"{disc.name}: preprocessing is incomplete")
        for name in ("fixes.parquet", "flights_meta.parquet"):
            parts.append([disc.name, file_identity(root / name, config.data_root)])
    return hashlib.sha256(json.dumps(parts).encode()).hexdigest()


@dataclass(frozen=True)
class RouteIndex:
    """A completed census with no connection shared between GUI and worker."""

    path: Path
    disciplines: tuple
    signature: str

    def flights(self):
        """Read endpoints on 10 km cells by merging aligned 2 x 2 index squares."""
        with connect(self.path) as db:
            flights = pd.read_sql_query("SELECT * FROM flights", db)
        for column in PAIR_COLUMNS:
            flights[column] = flights[column] // int(ROUTE_CELL_M / INDEX_CELL_M)
        return flights

    def verify(self):
        """Prevent a stale index from addressing a replaced Parquet archive."""
        if archive_signature(self.disciplines) != self.signature:
            raise ValueError("Processed archive changed; rebuild the route index")


def load_saved_index(disciplines=None, *, path=None):
    """Load an existing census only; opening a tab never starts a full scan."""
    disciplines = available_archives() if disciplines is None else disciplines
    if not disciplines:
        return None
    path = route_cache_path(disciplines) if path is None else Path(path)
    if not path.is_file():
        return None
    signature = archive_signature(disciplines)
    try:
        with connect(path) as db:
            saved = dict(db.execute("SELECT key,value FROM metadata"))
        if saved.get("signature") == signature and saved.get("complete") == "yes":
            return RouteIndex(path, tuple(disciplines), signature)
    except sqlite3.DatabaseError:
        pass
    return None


def reduce_group(frame, discipline, group):
    """Retain exact temporal extrema even when a flight spans Parquet groups."""
    frame = frame.reset_index(drop=True)
    grouped = frame.groupby("flight_id", sort=False)
    extrema = grouped.t.agg(["idxmin", "idxmax"])
    start = frame.loc[extrema["idxmin"], ["t", "E", "N", "z"]].to_numpy()
    end = frame.loc[extrema["idxmax"], ["t", "E", "N", "z"]].to_numpy()
    return [
        (discipline, str(fid), group, *map(float, a), *map(float, b))
        for fid, a, b in zip(extrema.index, start, end, strict=True)
    ]


def _endpoint_rows(db, disciplines, progress, cancel, *, departure_only=False):
    """Recover geographic endpoints from each flight's own ENU origin."""
    result = []
    west, south, east, north = FRANCE_EXTENT
    for disc in disciplines:
        parts = pd.read_sql_query(
            "SELECT * FROM parts WHERE discipline=?", db, params=(disc.name,)
        )
        if parts.empty:
            continue
        first = parts.sort_values(["t0", "row_group"]).drop_duplicates("flight_id")
        last = parts.sort_values(["t1", "row_group"]).drop_duplicates(
            "flight_id", keep="last"
        )
        ends = first[["flight_id", "t0", "e0", "n0", "z0"]].merge(
            last[["flight_id", "t1", "e1", "n1", "z1"]], on="flight_id"
        )
        meta = pd.read_parquet(
            disc.config().derived_dir / "flights_meta.parquet", columns=META_COLUMNS
        )
        meta["flight_id"] = meta.flight_id.astype(str)
        meta = meta.drop_duplicates("flight_id", keep="last")
        ends = ends.merge(meta.loc[meta.drop_reason.isna()], on="flight_id")
        for i, row in enumerate(ends.itertuples(index=False)):
            if i % 1000 == 0:
                _check_cancel(cancel)
                progress(f"{disc.name}: locating endpoints {i:,}/{len(ends):,}")
            values = [
                row.t0,
                row.t1,
                row.e0,
                row.n0,
                row.z0,
                row.e1,
                row.n1,
                row.z1,
                row.lat0,
                row.lon0,
                row.alt0,
            ]
            if (
                pd.isna(values).any()
                or not np.isfinite(values).all()
                or row.t1 <= row.t0
            ):
                continue
            lat, lon = enu_to_geodetic(
                [row.e0, row.e1],
                [row.n0, row.n1],
                [row.z0, row.z1],
                LocalFrame(row.lat0, row.lon0, row.alt0),
            )
            inside = (lon >= west) & (lon < east) & (lat >= south) & (lat < north)
            if not (inside[0] if departure_only else inside.all()):
                continue
            x, y = project(lon, lat)
            cells = np.floor(np.column_stack((x, y)) / INDEX_CELL_M).astype(int).ravel()
            result.append(
                (
                    disc.name,
                    row.flight_id,
                    *map(int, cells),
                    row.t1 - row.t0,
                    row.t0,
                    row.t1,
                    row.lat0,
                    row.lon0,
                    row.alt0,
                    int(row.n_segments_kept),
                )
            )
    return result


def build_index(disciplines=None, *, path=None, progress=lambda _: None, cancel=None):
    """Atomically publish a resumable census, serializing writers with a lock."""
    disciplines = available_archives() if disciplines is None else disciplines
    if not disciplines:
        raise FileNotFoundError("Choose a data folder with the processed archives")
    path = route_cache_path(disciplines) if path is None else Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with exclusive(path, "The route census is already being prepared"):
        saved = load_saved_index(disciplines, path=path)
        if saved is not None:
            return saved
        signature = archive_signature(disciplines)
        temporary = path.with_suffix(".building.sqlite3")
        with connect(temporary) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT)"
            )
            previous = dict(db.execute("SELECT key,value FROM metadata"))
            if previous.get("signature") != signature:
                db.executescript(
                    "DROP TABLE IF EXISTS parts; DROP TABLE IF EXISTS scanned; "
                    "DROP TABLE IF EXISTS flights; DELETE FROM metadata;"
                )
                db.execute("INSERT INTO metadata VALUES ('signature',?)", (signature,))
            db.executescript("""
                CREATE TABLE IF NOT EXISTS parts(
                    discipline TEXT,flight_id TEXT,row_group INTEGER,
                    t0 REAL,e0 REAL,n0 REAL,z0 REAL,t1 REAL,e1 REAL,n1 REAL,z1 REAL,
                    PRIMARY KEY(discipline,flight_id,row_group));
                CREATE TABLE IF NOT EXISTS scanned(
                    discipline TEXT,row_group INTEGER,
                    PRIMARY KEY(discipline,row_group));
            """)
            for disc in disciplines:
                archive = pq.ParquetFile(disc.config().derived_dir / "fixes.parquet")
                scanned = {
                    r[0]
                    for r in db.execute(
                        "SELECT row_group FROM scanned WHERE discipline=?", (disc.name,)
                    )
                }
                for group in range(archive.num_row_groups):
                    _check_cancel(cancel)
                    if group in scanned:
                        continue
                    progress(
                        f"{disc.name}: endpoint census "
                        f"{group + 1:,}/{archive.num_row_groups:,}"
                    )
                    frame = archive.read_row_group(
                        group, columns=FIX_COLUMNS
                    ).to_pandas()
                    db.executemany(
                        "INSERT INTO parts VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        reduce_group(frame, disc.name, group),
                    )
                    db.execute("INSERT INTO scanned VALUES (?,?)", (disc.name, group))
                    db.commit()
            rows = _endpoint_rows(db, disciplines, progress, cancel)
            _check_cancel(cancel)
            if archive_signature(disciplines) != signature:
                raise ValueError("Archive changed during the endpoint census")
            db.execute("DROP TABLE IF EXISTS flights")
            db.execute("""CREATE TABLE flights(
                discipline TEXT,flight_id TEXT,start_ix INTEGER,start_iy INTEGER,
                end_ix INTEGER,end_iy INTEGER,duration_s REAL,t0 REAL,t1 REAL,
                lat0 REAL,lon0 REAL,alt0 REAL,segments INTEGER,
                PRIMARY KEY(discipline,flight_id))""")
            db.executemany(
                "INSERT INTO flights VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows
            )
            db.execute("INSERT OR REPLACE INTO metadata VALUES ('complete','yes')")
            db.commit()
        temporary.replace(path)
    return RouteIndex(path, tuple(disciplines), signature)


def route_pairs(
    flights, minimum_km=90, maximum_km=110, discipline=None, *, require_both=False
):
    """Rank directed pairs by population, then proximity to the interval centre."""
    if (
        not np.isfinite([minimum_km, maximum_km]).all()
        or not 0 <= minimum_km <= maximum_km
    ):
        raise ValueError("Choose an ordered, nonnegative distance interval")
    if discipline is not None:
        flights = flights.loc[flights.discipline == discipline]
    pairs = flights.groupby(PAIR_COLUMNS).size().reset_index(name="flights")
    if require_both:
        shared = flights.groupby(PAIR_COLUMNS).discipline.nunique()
        pairs = pairs.merge(shared.loc[shared >= 2].reset_index(), on=PAIR_COLUMNS)
    pairs["distance_km"] = (
        np.hypot(pairs.end_ix - pairs.start_ix, pairs.end_iy - pairs.start_iy)
        * ROUTE_CELL_M
        / 1000
    )
    pairs = pairs.loc[pairs.distance_km.between(minimum_km, maximum_km)].copy()
    pairs["target_error"] = abs(pairs.distance_km - (minimum_km + maximum_km) / 2)
    return pairs.sort_values(
        ["flights", "target_error", *PAIR_COLUMNS],
        ascending=[False, True, True, True, True, True],
    ).reset_index(drop=True)


def matching_flights(flights, pair, discipline=None):
    """Return every flight in the directed pair before any sampling or ranking."""
    mask = (flights[PAIR_COLUMNS].to_numpy() == np.asarray(pair)).all(axis=1)
    if discipline is not None:
        mask &= flights.discipline.eq(discipline).to_numpy()
    return flights.loc[mask].copy()


def select_flights(flights, pair, discipline=None):
    """Choose at most 300 reproducible flights, always including both extremes.

    Five fastest and five slowest are kept from the entire matching population;
    remaining slots are evenly spaced duration ranks. This is a comparison sample,
    not a random population estimate. Ties break by discipline and flight ID.
    """
    matching = (
        matching_flights(flights, pair, discipline)
        .sort_values(["duration_s", "discipline", "flight_id"])
        .reset_index(drop=True)
    )
    count = len(matching)
    if count > MAX_FLIGHTS:
        ranks = np.r_[
            np.arange(5),
            np.linspace(5, count - 6, MAX_FLIGHTS - 10, dtype=int),
            np.arange(count - 5, count),
        ]
        matching = matching.iloc[ranks].copy()
    matching["rank"] = matching.index + 1
    matching["speed_group"] = "other"
    fast = matching["rank"] <= 5
    slow = matching["rank"] > count - 5
    matching.loc[fast, "speed_group"] = "fast"
    matching.loc[slow, "speed_group"] = "slow"
    matching.loc[fast & slow, "speed_group"] = "both"
    return matching.reset_index(drop=True), count
