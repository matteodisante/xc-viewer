"""Self-contained thermal-plane delivery file: the viewer opens it read-only."""

from __future__ import annotations

import io
import json
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .core.disciplines import DISCIPLINES
from .sqlite import connect
from .thermal_geometry import ThermalCell
from .thermal_ground import (
    CLIMB_RANKING,
    GROUND_REFERENCE,
    TERRAIN_RANKING,
    cell_band,
    order_terrain_cells,
    terrain_cell,
    terrain_reference,
)

EDGE_COLUMNS = [f"{axis}{end}" for end in (0, 1) for axis in ("x", "y", "z", "utc")]
STORE_NAME = "thermal-planes.sqlite3"
NEIGHBOUR_OFFSETS = tuple(
    (dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dx, dy) != (0, 0)
)


def neighbour_frames(cell):
    """The eight squares around ``cell``, carrying its ground and ceiling.

    Zoom - shows them on the selected cell's planes, so only ix/iy change: their
    own terrain minima never tilt or step the plane.
    """
    return [
        replace(cell, ix=cell.ix + dx, iy=cell.iy + dy) for dx, dy in NEIGHBOUR_OFFSETS
    ]


class CancelledError(Exception):
    """A saved-data read was cancelled."""


@dataclass(frozen=True)
class PlaneData:
    """Saved edges and their coverage for the chosen UTC interval."""

    edges: pd.DataFrame
    selected: int
    decoded: int
    unclassified: int
    unavailable: int
    unknown_clock: int
    cached: int = 0
    points: pd.DataFrame | None = None
    visits: pd.DataFrame | None = None


@contextmanager
def _connect(path):
    db = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    db.execute("PRAGMA query_only=ON")
    try:
        yield db
    finally:
        db.close()


class ThermalStore:
    """Only a filename crosses threads; each read owns its SQLite connection."""

    def __init__(self, path):
        """Validate the completed snapshot without probing any archive inputs."""
        self.path = Path(path)
        with _connect(self.path) as db:
            metadata = dict(db.execute("SELECT key,value FROM metadata"))
            if metadata.get("ready") != "1" or metadata.get("version") not in (
                "1",
                "2",
                "3",
            ):
                raise ValueError(
                    "The prepared thermal-plane file is incomplete or incompatible"
                )
            if metadata.get("ground_reference") != GROUND_REFERENCE:
                raise ValueError(
                    "The Thermal planes data in this data folder come from an "
                    "older version of the viewer. Use an up-to-date data folder."
                )
            from .thermal_daily import POINT_LATTICE_VERSION

            self.has_points = metadata.get("point_lattice") == POINT_LATTICE_VERSION
            self.disciplines = tuple(json.loads(metadata["disciplines"]))
            self.quality_summary = json.loads(metadata.get("launch_quality", "null"))
            self.has_terrain_ranking = metadata.get("ranking_reference") in (
                TERRAIN_RANKING,
                CLIMB_RANKING,
            )
            self.has_climb_ranking = metadata.get("ranking_reference") == CLIMB_RANKING
            self.activity_counts = (
                {
                    (ix, iy): {"climb_runs": runs, "climb_flights": flights}
                    for ix, iy, runs, flights in db.execute(
                        "SELECT * FROM cell_activity"
                    )
                }
                if self.has_climb_ranking
                else {}
            )

    def cells(self):
        """Classify the saved subset by its saved DEM maxima, never by payload labels.

        Reordering saved cells does not claim a new archive-wide top three.
        Heights, grid IDs and point products remain unchanged.
        """
        with _connect(self.path) as db:
            bands = {
                (ix, iy): cell_band(json.loads(metadata))
                for ix, iy, metadata in db.execute("SELECT ix,iy,metadata FROM terrain")
            }
            cells = [
                ThermalCell(**json.loads(row[0]))
                for row in db.execute("SELECT payload FROM cells ORDER BY position")
            ]
        cells = [replace(c, terrain=bands[c.ix, c.iy]) for c in cells]
        if self.has_climb_ranking:
            from .geography import TERRAIN_ORDER

            return sorted(
                cells,
                key=lambda c: (
                    TERRAIN_ORDER.index(c.terrain),
                    -self.activity_counts[c.ix, c.iy]["climb_runs"],
                    c.ix,
                    c.iy,
                ),
            )
        return order_terrain_cells(cells)

    def relief(self, cell=None, key=None):
        """Read prepared relief and its exact coordinates, without networking."""
        key = key or ("france" if cell is None else f"{cell.ix}/{cell.iy}")
        with _connect(self.path) as db:
            if not db.execute(
                "SELECT 1 FROM sqlite_master WHERE name='relief'"
            ).fetchone():
                return None
            row = db.execute(
                "SELECT metadata,png FROM relief WHERE key=?", (key,)
            ).fetchone()
        if row is None:
            return None
        from PIL import Image

        return np.asarray(Image.open(io.BytesIO(row[1])).convert("RGB")), json.loads(
            row[0]
        )

    def background(self, cell=None, kind="relief", key=None):
        """Read saved imagery, including the acquisition-date provenance."""
        if kind == "relief":
            return self.relief(cell, key)
        key = key or ("france" if cell is None else f"{cell.ix}/{cell.iy}")
        with _connect(self.path) as db:
            if not db.execute(
                "SELECT 1 FROM sqlite_master WHERE name='backgrounds'"
            ).fetchone():
                return None
            row = db.execute(
                "SELECT metadata,image FROM backgrounds WHERE kind=? AND key=?",
                (kind, key),
            ).fetchone()
        if row is None:
            return None
        from PIL import Image

        return np.asarray(Image.open(io.BytesIO(row[1])).convert("RGB")), json.loads(
            row[0]
        )

    def neighbour_flights(self, cell, source):
        """Flights behind the saved neighbourhood of ``cell``; None if unprepared."""
        with _connect(self.path) as db:
            if not db.execute(
                "SELECT 1 FROM sqlite_master WHERE name='neighbour_flights'"
            ).fetchone():
                return None
            row = db.execute(
                "SELECT flights FROM neighbour_flights "
                "WHERE source=? AND ix=? AND iy=?",
                (source, cell.ix, cell.iy),
            ).fetchone()
        return None if row is None else [tuple(f) for f in json.loads(row[0])]

    def neighbour_points(self, cell, source, level):
        """Saved crossings of the eight surrounding squares with one plane of ``cell``.

        Columns x, y (Lambert-93 m), utc, and ``flight``: the position of the
        flight in neighbour_flights. One indexed row per level, so a height
        change reads only that plane.
        """
        with _connect(self.path) as db:
            row = db.execute(
                "SELECT points FROM neighbour_points "
                "WHERE source=? AND ix=? AND iy=? AND level=?",
                (source, cell.ix, cell.iy, level),
            ).fetchone()
        if row is None:
            points, flight = np.empty((0, 3)), np.empty(0, dtype=np.int32)
        else:
            with np.load(io.BytesIO(row[0]), allow_pickle=False) as saved:
                points, flight = saved["points"], saved["flight"]
        frame = pd.DataFrame(points, columns=["x", "y", "utc"])
        frame["flight"] = flight
        return frame

    def reference_audit(self, cell):
        """Return the saved unfiltered support beside the screened reference."""
        with _connect(self.path) as db:
            if not db.execute(
                "SELECT 1 FROM sqlite_master WHERE name='ground_audit'"
            ).fetchone():
                return None
            return db.execute(
                "SELECT raw_count,raw_median FROM ground_audit WHERE ix=? AND iy=?",
                (cell.ix, cell.iy),
            ).fetchone()

    def terrain_reference(self, cell):
        """Read the DEM summary, sampling, coverage and attribution saved offline."""
        with _connect(self.path) as db:
            row = db.execute(
                "SELECT metadata FROM terrain WHERE ix=? AND iy=?",
                (cell.ix, cell.iy),
            ).fetchone()
        if row is None:
            raise ValueError("Missing saved terrain reference")
        return json.loads(row[0])

    def time_extent(self):
        """Saved global coverage; no per-cell selection can narrow the user's dates."""
        with _connect(self.path) as db:
            return db.execute("SELECT MIN(start),MAX(end) FROM visitors").fetchone()

    def summer_days(self, cell):
        """Busiest summer dates, ranked by distinct crossing flights (ties: date)."""
        if not self.has_points:
            return []
        with _connect(self.path) as db:
            return db.execute(
                "SELECT day,flights FROM summer_days WHERE ix=? AND iy=? "
                "ORDER BY flights DESC,day",
                (cell.ix, cell.iy),
            ).fetchall()

    def defaults(self, cell):
        """Return a prepared day and height with climb data for immediate display."""
        with _connect(self.path) as db:
            return db.execute(
                "SELECT day,height FROM cells WHERE ix=? AND iy=?", (cell.ix, cell.iy)
            ).fetchone()

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
        """Read saved products only: no raw files, parquet, models or write handles."""
        if source != "vilpellet":
            raise ValueError(f"Unknown segmentation: {source}")
        if end < start:
            raise ValueError("The end time precedes the start time")
        edges = []
        point_frames = []
        visit_spans = []
        selected = decoded = unclassified = unavailable = 0
        use_points = self.has_points and not use_edges
        with _connect(self.path) as db:
            unknown = db.execute(
                "SELECT COUNT(*) FROM visitors WHERE ix=? AND iy=? AND start IS NULL",
                (cell.ix, cell.iy),
            ).fetchone()[0]
            product = "p.points" if use_points else "c.edges"
            join = (
                (
                    " LEFT JOIN plane_points p ON p.source=c.source "
                    "AND p.ix=c.ix AND p.iy=c.iy "
                    "AND p.discipline=c.discipline AND p.flight_id=c.flight_id "
                )
                if use_points
                else ""
            )
            rows = db.execute(
                f"""SELECT v.discipline,v.flight_id,c.status,{product},v.start,v.end
                FROM visitors v LEFT JOIN climbs c
                ON c.discipline=v.discipline AND c.flight_id=v.flight_id
                AND c.ix=v.ix AND c.iy=v.iy AND c.source=?
                {join} WHERE v.ix=? AND v.iy=? AND v.end>=? AND v.start<=?""",
                (source, cell.ix, cell.iy, start, end),
            )
            for discipline, fid, status, blob, v_start, v_end in rows:
                if cancel is not None and cancel.is_set():
                    raise CancelledError("Cancelled")
                if status is None:
                    raise ValueError(
                        "Missing saved product; rerun the offline preparation"
                    )
                selected += 1
                visit_spans.append((v_start, v_end))
                decoded += status != "unavailable"
                unclassified += status == "unclassified"
                unavailable += status == "unavailable"
                if blob is None:
                    raise ValueError(
                        "Missing prepared intersections; finish offline preparation"
                    )
                if use_points:
                    with np.load(io.BytesIO(blob), allow_pickle=False) as saved:
                        values = saved["points"]
                    values = values[(values[:, 3] >= start) & (values[:, 3] <= end)]
                    if len(values):
                        frame = pd.DataFrame(values, columns=["level", "x", "y", "utc"])
                        frame["discipline"], frame["flight_id"] = discipline, fid
                        point_frames.append(frame)
                    continue
                with np.load(io.BytesIO(blob), allow_pickle=False) as saved:
                    values = saved["edges"]
                values = values[(values[:, 7] >= start) & (values[:, 3] <= end)]
                if len(values):
                    frame = pd.DataFrame(values, columns=EDGE_COLUMNS)
                    frame["discipline"], frame["flight_id"] = discipline, fid
                    edges.append(frame)
        progress(f"Read {selected:,} saved flights from the data folder")
        return PlaneData(
            pd.concat(edges, ignore_index=True) if edges else pd.DataFrame(),
            selected,
            decoded,
            unclassified,
            unavailable,
            unknown,
            selected,
            (
                pd.concat(point_frames, ignore_index=True)
                if point_frames
                else pd.DataFrame(
                    columns=["level", "x", "y", "utc", "discipline", "flight_id"]
                )
            )
            if use_points
            else None,
            pd.DataFrame(visit_spans, columns=["start", "end"], dtype=float),
        )


def find_store_path():
    """Locate a snapshot, including legacy files awaiting an offline upgrade."""
    override = os.environ.get("XC_VIEWER_CACHE_DIR")
    paths = (
        [Path(override).expanduser() / STORE_NAME]
        if override
        else [
            d.config().derived_dir / "viewer/thermal-planes" / STORE_NAME
            for d in DISCIPLINES.values()
        ]
    )
    for path in paths:
        if path.is_file():
            return path
    return None


def load_store():
    """Find the ready file, without checking or opening its original archives."""
    path = find_store_path()
    if path is None:
        return None
    store = ThermalStore(path)
    if not store.has_terrain_ranking:
        raise ValueError(
            "The Thermal planes data in this data folder come from an older "
            "version of the viewer. Use an up-to-date data folder."
        )
    return store


def export_store(index, *, relief=None, destination=None):
    """Export checked edges, optionally staging before imagery/point enrichment."""
    from .thermal_cache import segmentation_signature

    references = getattr(index, "terrain_references", None)
    if references is None:
        references = {
            (cell.ix, cell.iy): terrain_reference(cell) for cell in index.cells()
        }
    if relief is None and hasattr(index, "quality_summary"):
        from .thermal_relief import prepare_relief

        relief = prepare_relief(index)
    target = (
        index.path.with_name(STORE_NAME) if destination is None else Path(destination)
    )
    temporary = target.with_suffix(".building.sqlite3")
    temporary.unlink(missing_ok=True)
    with (
        connect(temporary) as out,
        _connect(index.path.with_name("thermal-climbs.sqlite3")) as products,
    ):
        out.executescript("""
            CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT);
            CREATE TABLE relief(key TEXT PRIMARY KEY,metadata TEXT,png BLOB);
            CREATE TABLE terrain(ix INTEGER,iy INTEGER,metadata TEXT,
                PRIMARY KEY(ix,iy));
            CREATE TABLE cells(position INTEGER PRIMARY KEY,ix INTEGER,iy INTEGER,
                payload TEXT,day REAL,height REAL);
            CREATE TABLE cell_activity(ix INTEGER,iy INTEGER,climb_runs INTEGER,
                climb_flights INTEGER,PRIMARY KEY(ix,iy));
            CREATE TABLE visitors(ix INTEGER,iy INTEGER,discipline TEXT,flight_id TEXT,
                start REAL,end REAL,
                PRIMARY KEY(ix,iy,discipline,flight_id));
            CREATE TABLE climbs(source TEXT,discipline TEXT,flight_id TEXT,
                ix INTEGER,iy INTEGER,
                status TEXT,edges BLOB,
                PRIMARY KEY(source,ix,iy,discipline,flight_id));
        """)
        keys = {"vilpellet": segmentation_signature(index, "vilpellet")}
        out.executemany(
            "INSERT INTO metadata VALUES (?,?)",
            [
                ("version", "3"),
                ("ground_reference", GROUND_REFERENCE),
                ("disciplines", json.dumps(index.disciplines)),
                ("archive_signature", index.signature),
                ("segmentation_signatures", json.dumps(keys)),
            ],
        )
        if relief:
            out.executemany("INSERT INTO relief VALUES (?,?,?)", relief)
        if hasattr(index, "quality_summary"):
            out.execute(
                "INSERT INTO metadata VALUES (?,?)",
                ("launch_quality", json.dumps(index.quality_summary)),
            )
            if index.quality_summary.get("ranking_reference") in (
                TERRAIN_RANKING,
                CLIMB_RANKING,
            ):
                out.execute(
                    "INSERT INTO metadata VALUES ('ranking_reference',?)",
                    (index.quality_summary["ranking_reference"],),
                )
        for position, cell in enumerate(index.cells()):
            reference = references[(cell.ix, cell.iy)]
            cell = terrain_cell(cell, reference)
            activity = getattr(index, "activity_counts", {}).get((cell.ix, cell.iy))
            if activity is not None:
                out.execute(
                    "INSERT INTO cell_activity VALUES (?,?,?,?)",
                    (
                        cell.ix,
                        cell.iy,
                        activity["climb_runs"],
                        activity["climb_flights"],
                    ),
                )
            out.execute(
                "INSERT INTO terrain VALUES (?,?,?)",
                (cell.ix, cell.iy, json.dumps(reference)),
            )
            visitors = index.flights(cell)
            utc = visitors.start_utc + visitors.trim_start
            out.executemany(
                "INSERT INTO visitors VALUES (?,?,?,?,?,?)",
                [
                    (
                        cell.ix,
                        cell.iy,
                        r.discipline,
                        r.flight_id,
                        float(o + r.t_min) if np.isfinite(o) else None,
                        float(o + r.t_max) if np.isfinite(o) else None,
                    )
                    for r, o in zip(visitors.itertuples(index=False), utc, strict=True)
                ],
            )
            expected = set(
                zip(
                    visitors.loc[utc.notna(), "discipline"],
                    visitors.loc[utc.notna(), "flight_id"],
                    strict=True,
                )
            )
            days = {}
            for source, key in keys.items():
                found = set()
                for discipline, fid, status, blob in products.execute(
                    "SELECT discipline,flight_id,status,edges FROM climbs "
                    "WHERE cache_key=? AND ix=? AND iy=?",
                    (key, cell.ix, cell.iy),
                ):
                    if (discipline, fid) not in expected:
                        continue
                    found.add((discipline, fid))
                    out.execute(
                        "INSERT INTO climbs VALUES (?,?,?,?,?,?,?)",
                        (source, discipline, fid, cell.ix, cell.iy, status, blob),
                    )
                    if source == "vilpellet":
                        with np.load(io.BytesIO(blob), allow_pickle=False) as saved:
                            a = saved["edges"]
                        if len(a):
                            for day in np.unique(np.floor(a[:, 3] / 86400) * 86400):
                                subset = a[np.floor(a[:, 3] / 86400) * 86400 == day]
                                n, z = days.get(float(day), (0, 0.0))
                                days[float(day)] = (
                                    n + len(subset),
                                    z
                                    + float(((subset[:, 2] + subset[:, 6]) / 2).sum()),
                                )
                if found != expected:
                    raise RuntimeError(
                        f"{cell.terrain}/{source}: {len(expected - found)} "
                        "flights still need preparation"
                    )
            if days:
                day, (n, z) = max(days.items(), key=lambda item: item[1][0])
                height = float(np.clip(z / n - cell.ground_m, 0, cell.max_agl_m))
            else:
                day, height = 0.0, 0.0
            out.execute(
                "INSERT INTO cells VALUES (?,?,?,?,?,?)",
                (position, cell.ix, cell.iy, json.dumps(asdict(cell)), day, height),
            )
        out.execute("INSERT INTO metadata VALUES ('ready','1')")
        out.commit()
        if out.execute("PRAGMA quick_check").fetchone() != ("ok",):
            raise RuntimeError("Prepared file failed SQLite integrity check")
    temporary.replace(target)
    return target
