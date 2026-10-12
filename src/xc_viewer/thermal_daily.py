"""Offline intersection lattice and civil-day summaries from saved climb edges."""

from __future__ import annotations

import hashlib
import io
import json
from calendar import monthrange
from collections import Counter
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np

from .sqlite import connect
from .thermal_geometry import ThermalCell

PARIS = ZoneInfo("Europe/Paris")
POINT_COLUMNS = ["level", "x", "y", "utc"]
POINT_LATTICE_VERSION = "10m-v2"


def _point_signature(cells):
    return hashlib.sha256(
        json.dumps(
            [
                POINT_LATTICE_VERSION,
                [(c.ix, c.iy, c.ground_m, c.max_alt_m) for c in cells],
            ]
        ).encode()
    ).hexdigest()


def reuse_prepared_points(path, previous):
    """Retain identical cell lattices when only the selected set/ranking changes."""
    from pathlib import Path

    from .thermal_neighbours import NEIGHBOUR_LATTICE_VERSION
    from .thermal_store import ThermalStore

    if not Path(previous).exists() or Path(path).resolve() == Path(previous).resolve():
        return
    current = ThermalStore(path)
    try:
        old = ThermalStore(previous)
    except ValueError:
        # Another ground reference shifts every level: nothing can be reused.
        return
    cells = current.cells()
    old_cells = {(c.ix, c.iy): c for c in old.cells()}
    shared = [
        c
        for c in cells
        if (c.ix, c.iy) in old_cells
        and c.ground_m == old_cells[c.ix, c.iy].ground_m
        and c.max_alt_m == old_cells[c.ix, c.iy].max_alt_m
    ]
    if not shared:
        return
    with connect(path, uri=True) as db:
        db.execute(
            "ATTACH DATABASE ? AS previous",
            (Path(previous).resolve().as_uri() + "?mode=ro",),
        )
        meta = dict(db.execute("SELECT key,value FROM main.metadata"))
        prior = dict(db.execute("SELECT key,value FROM previous.metadata"))
        if not all(
            meta.get(key) is not None and meta[key] == prior.get(key)
            for key in (
                "archive_signature",
                "segmentation_signatures",
                "ground_reference",
            )
        ):
            return
        signature = _point_signature(cells)
        if meta.get("point_build_signature") not in (None, signature):
            return
        if prior.get("point_lattice") == POINT_LATTICE_VERSION:
            db.execute("""CREATE TABLE IF NOT EXISTS plane_points(
                source TEXT,ix INTEGER,iy INTEGER,discipline TEXT,flight_id TEXT,
                points BLOB,count INTEGER,
                PRIMARY KEY(source,ix,iy,discipline,flight_id))""")
            for cell in shared:
                db.execute(
                    "INSERT OR IGNORE INTO main.plane_points SELECT * "
                    "FROM previous.plane_points WHERE ix=? AND iy=?",
                    (cell.ix, cell.iy),
                )
            db.execute(
                "INSERT OR REPLACE INTO metadata VALUES ('point_build_signature',?)",
                (signature,),
            )
        if prior.get("neighbour_segmentations") == meta["segmentation_signatures"]:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS neighbour_points(
                    source TEXT,ix INTEGER,iy INTEGER,level INTEGER,points BLOB,
                    PRIMARY KEY(source,ix,iy,level));
                CREATE TABLE IF NOT EXISTS neighbour_flights(
                    source TEXT,ix INTEGER,iy INTEGER,flights TEXT,signature TEXT,
                    PRIMARY KEY(source,ix,iy));
            """)
            for cell in shared:
                for source, key in json.loads(meta["segmentation_signatures"]).items():
                    expected = json.dumps(
                        [NEIGHBOUR_LATTICE_VERSION, key, cell.ground_m, cell.max_alt_m]
                    )
                    identity = source, cell.ix, cell.iy
                    row = db.execute(
                        "SELECT signature FROM previous.neighbour_flights "
                        "WHERE source=? AND ix=? AND iy=?",
                        identity,
                    ).fetchone()
                    if row != (expected,):
                        continue
                    for table in ("neighbour_points", "neighbour_flights"):
                        db.execute(
                            f"INSERT OR IGNORE INTO main.{table} SELECT * "
                            f"FROM previous.{table} WHERE source=? AND ix=? AND iy=?",
                            identity,
                        )
        db.commit()


def height_levels(maximum, step=10):
    """Regular AGL levels plus the exact supported ceiling, even off the grid."""
    values = np.arange(0, maximum, step, dtype=float)
    return np.r_[values, maximum]


def local_bounds(day, hours=(0, 24)):
    """Paris wall-clock bounds, including 23/25-hour civil days at DST changes."""
    if isinstance(day, str):
        day = date.fromisoformat(day)
    return tuple(
        (datetime.combine(day, time(), PARIS) + timedelta(hours=h)).timestamp()
        for h in hours
    )


def season_days(first, last, years):
    """Civil days from ``first`` to ``last`` (month, day) in each start year.

    A season whose end precedes its start runs into the next year. A 29 February
    bound becomes 28 February in common years.
    """

    def on(year, month_day):
        month, number = month_day
        return date(year, month, min(number, monthrange(year, month)[1]))

    days = set()
    for year in years:
        start, end = on(year, first), on(year + (last < first), last)
        days.update(start + timedelta(days=i) for i in range((end - start).days + 1))
    return sorted(days)


def wall_windows(days, hours):
    """UTC ``[start, end)`` of the same Paris clock hours on each civil day.

    ``hours`` are fractional; 24 is the next midnight. Unlike ``local_bounds``, an
    hour is a clock reading, also on the 23/25-hour days of DST changes.
    """
    return np.array(
        [
            [
                (datetime.combine(day, time()) + timedelta(hours=h))
                .replace(tzinfo=PARIS)
                .timestamp()
                for h in hours
            ]
            for day in days
        ],
        dtype=float,
    ).reshape(-1, 2)


def in_windows(utc, windows):
    """Epoch seconds inside any of the sorted, disjoint ``[start, end)`` windows."""
    utc = np.asarray(utc, dtype=float)
    if not len(windows):
        return np.zeros(len(utc), dtype=bool)
    k = np.searchsorted(windows[:, 0], utc, side="right") - 1
    return (k >= 0) & (utc < windows[np.maximum(k, 0), 1])


def overlaps_windows(start, end, windows):
    """Closed spans ``[start, end]`` that overlap any sorted, disjoint window."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    if not len(windows):
        return np.zeros(len(start), dtype=bool)
    k = np.searchsorted(windows[:, 1], start, side="right")
    return (k < len(windows)) & (windows[np.minimum(k, len(windows) - 1), 0] <= end)


def lattice_points(values, cell):
    """Vectorized equivalent of plane_intersections at every prepared height.

    Input is one flight's continuous saved edges. Preserve the existing half-open
    edge convention and terminal vertices. Never create new trajectory edges.
    """
    levels = height_levels(cell.max_agl_m)
    if not len(values):
        return np.empty((0, 4))
    a, b = values[:, :4], values[:, 4:]
    low = np.minimum(a[:, 2], b[:, 2]) - cell.ground_m
    high = np.maximum(a[:, 2], b[:, 2]) - cell.ground_m
    first = np.searchsorted(levels, low, side="left")
    last = np.searchsorted(levels, high, side="right")
    counts = np.maximum(last - first, 0)
    counts[a[:, 2] == b[:, 2]] = 0
    rows = np.repeat(np.arange(len(a)), counts)
    if not len(rows):
        return np.empty((0, 4))
    offsets = np.repeat(np.cumsum(counts) - counts, counts)
    level = np.repeat(first, counts) + np.arange(len(rows)) - offsets
    f = (cell.ground_m + levels[level] - a[rows, 2]) / (b[rows, 2] - a[rows, 2])
    connected = np.zeros(len(a), dtype=bool)
    connected[:-1] = (b[:-1, 3] == a[1:, 3]) & (b[:-1, 2] == a[1:, 2])
    use = ((f >= 0) & (f < 1)) | ((f == 1) & ~connected[rows])
    rows, level, f = rows[use], level[use], f[use]
    points = a[rows] + f[:, None] * (b[rows] - a[rows])
    w, s, e, n = cell.bounds
    inside = (
        (points[:, 0] >= w)
        & (points[:, 0] < e)
        & (points[:, 1] >= s)
        & (points[:, 1] < n)
    )
    return np.column_stack((level[inside], points[inside][:, [0, 1, 3]]))


def prepare_daily(path, progress=print):
    """Add resumable products to the ready file without opening raw archives.

    Each flight is committed in batches. Readers use the old format until the
    final capability flag is published, so an interrupted build stays usable.
    """
    with connect(path) as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS plane_points(
                source TEXT, ix INTEGER, iy INTEGER, discipline TEXT,
                flight_id TEXT, points BLOB, count INTEGER,
                PRIMARY KEY(source,ix,iy,discipline,flight_id));
            CREATE TABLE IF NOT EXISTS summer_days(
                ix INTEGER,iy INTEGER,day TEXT,flights INTEGER,
                PRIMARY KEY(ix,iy,day));
        """)
        cells = [
            ThermalCell(**json.loads(r[0]))
            for r in db.execute("SELECT payload FROM cells ORDER BY position")
        ]
        signature = _point_signature(cells)
        previous = db.execute(
            "SELECT value FROM metadata WHERE key='point_build_signature'"
        ).fetchone()
        if previous != (signature,):
            db.execute("DELETE FROM plane_points")
            db.execute("DELETE FROM metadata WHERE key='point_lattice'")
            db.execute(
                "INSERT OR REPLACE INTO metadata VALUES ('point_build_signature',?)",
                (signature,),
            )
            db.commit()
        ready = db.execute(
            "SELECT value FROM metadata WHERE key='point_lattice'"
        ).fetchone()
        if ready and ready[0] == POINT_LATTICE_VERSION:
            expected = db.execute("SELECT COUNT(*) FROM climbs").fetchone()[0]
            actual = db.execute("SELECT COUNT(*) FROM plane_points").fetchone()[0]
            if expected == actual:
                progress(f"All {actual:,} point products already prepared")
                return
        for cell in cells:
            days = Counter()
            for start, end in db.execute(
                "SELECT start,end FROM visitors "
                "WHERE ix=? AND iy=? AND start IS NOT NULL",
                (cell.ix, cell.iy),
            ):
                day = datetime.fromtimestamp(start, PARIS).date()
                last = datetime.fromtimestamp(end, PARIS).date()
                while day <= last:
                    if day.month in (6, 7, 8):
                        days[day.isoformat()] += 1
                    day += timedelta(days=1)
            db.executemany(
                "INSERT OR REPLACE INTO summer_days VALUES (?,?,?,?)",
                [(cell.ix, cell.iy, d, n) for d, n in days.items()],
            )
            db.commit()
            rows = db.execute(
                """SELECT c.source,c.discipline,c.flight_id,c.edges
                FROM climbs c LEFT JOIN plane_points p
                ON p.source=c.source AND p.ix=c.ix AND p.iy=c.iy
                AND p.discipline=c.discipline AND p.flight_id=c.flight_id
                WHERE c.ix=? AND c.iy=? AND p.source IS NULL""",
                (cell.ix, cell.iy),
            ).fetchall()
            progress(
                f"{cell.terrain} {cell.ix}/{cell.iy}: {len(rows):,} products to prepare"
            )
            for i, (source, discipline, fid, blob) in enumerate(rows):
                with np.load(io.BytesIO(blob), allow_pickle=False) as saved:
                    points = lattice_points(saved["edges"], cell)
                output = io.BytesIO()
                np.savez_compressed(output, points=points)
                db.execute(
                    "INSERT INTO plane_points VALUES (?,?,?,?,?,?,?)",
                    (
                        source,
                        cell.ix,
                        cell.iy,
                        discipline,
                        fid,
                        output.getvalue(),
                        len(points),
                    ),
                )
                if (i + 1) % 1000 == 0:
                    db.commit()
                    progress(f"  {i + 1:,}/{len(rows):,} saved")
            db.commit()
        missing = db.execute("""SELECT COUNT(*) FROM climbs c LEFT JOIN plane_points p
            ON p.source=c.source AND p.ix=c.ix AND p.iy=c.iy
            AND p.discipline=c.discipline AND p.flight_id=c.flight_id
            WHERE p.source IS NULL""").fetchone()[0]
        if missing:
            raise ValueError(f"Missing {missing} prepared point products")
        db.execute(
            "INSERT OR REPLACE INTO metadata VALUES ('point_lattice',?)",
            (POINT_LATTICE_VERSION,),
        )
        db.commit()
        count = db.execute("SELECT SUM(count) FROM plane_points").fetchone()[0]
        progress(f"Prepared points: {count:,}")


def prepare_reference_audit(path):
    """Expose raw versus screened launch support without changing any reference."""
    import statistics
    from pathlib import Path

    census = Path(path).with_name("thermal-cells.sqlite3")
    if not census.exists():
        return
    with (
        connect(census.as_uri() + "?mode=ro", uri=True) as source,
        connect(path) as db,
    ):
        db.execute("""CREATE TABLE IF NOT EXISTS ground_audit(
            ix INTEGER,iy INTEGER,raw_count INTEGER,raw_median REAL,
            PRIMARY KEY(ix,iy))""")
        for ix, iy in db.execute("SELECT ix,iy FROM cells").fetchall():
            values = [
                r[0]
                for r in source.execute(
                    "SELECT launch_alt FROM flights WHERE launch_x=? AND launch_y=? "
                    "AND launch_alt IS NOT NULL",
                    (ix, iy),
                )
            ]
            db.execute(
                "INSERT OR REPLACE INTO ground_audit VALUES (?,?,?,?)",
                (ix, iy, len(values), statistics.median(values) if values else None),
            )
