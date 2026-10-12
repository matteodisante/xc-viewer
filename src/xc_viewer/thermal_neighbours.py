"""Offline 3 x 3 neighbourhoods around each published square; never imported by Qt.

Zoom - shows the eight squares around the selected cell on the selected cell's
planes: one absolute altitude per level, so a neighbour's own DEM minimum plays no
part. Their crossings are ``lattice_points`` of the neighbour's climb products,
with the neighbour's bounds and the centre's levels. A neighbour that is itself
published reuses the snapshot's own products; any other square takes the shared
climb cache, completed here from the processed archives. Backgrounds are saved
under the neighbour's own ``ix/iy`` key. The viewer only reads the result.
"""

from __future__ import annotations

import io
import json
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .sqlite import connect
from .thermal_daily import lattice_points
from .thermal_geometry import ThermalCell
from .thermal_store import ThermalStore, neighbour_frames

NEIGHBOUR_LATTICE_VERSION = "10m-v1"
SOURCES = ("vilpellet",)


def prepare_neighbour_terrain(path, *, progress=print):
    """Complete the attributed DEM cache for the selectable 10 km 3D areas."""
    from .thermal_ground import fetch_terrain_reference, terrain_reference
    from .thermal_orography import DATA_DIRECTORY

    store = ThermalStore(path)
    folder = store.path.parent / "exploration/terrain"
    squares = _squares(store)
    for done, cell in enumerate(squares, 1):
        # Reuse attributed repository rasters before requesting another copy.
        try:
            terrain_reference(cell, DATA_DIRECTORY)
        except (OSError, ValueError):
            fetch_terrain_reference(cell, folder)
        progress(f"Terrain {done}/{len(squares)}: {cell.ix}/{cell.iy}")


def _squares(store):
    """Distinct unpublished squares next to any published cell.

    The census query, climb_edges and background keys use only ix/iy.
    """
    cells = store.cells()
    published = {(c.ix, c.iy) for c in cells}
    near = {(f.ix, f.iy) for c in cells for f in neighbour_frames(c)}
    return [
        ThermalCell(ix, iy, "Neighbour", 0, 0, 0.0, 0.0)
        for ix, iy in sorted(near - published)
    ]


def prepare_neighbour_points(path, *, workers=4, progress=print):
    """Label every flight crossing a neighbour, then save the centre-level lattice.

    Resumable: climb products are committed per flight, and each (source, cell)
    neighbourhood is published in one transaction with its build signature.
    """
    from .thermal_cache import segmentation_signature
    from .thermal_index import load_saved_index
    from .thermal_prepare import prepare_climbs

    store = ThermalStore(path)
    census = load_saved_index(path=Path(path).with_name("thermal-cells.sqlite3"))
    if census is None:
        raise ValueError(
            "Neighbour preparation needs the connected processed archives "
            "and a current thermal-cells.sqlite3 census."
        )
    cells = store.cells()
    published = {(c.ix, c.iy) for c in cells}
    targets = _squares(store)
    progress(f"{len(targets)} neighbouring squares around {len(cells)} cells")
    prepare_climbs(census, workers=workers, progress=progress, cells=targets)
    keys = {source: segmentation_signature(census, source) for source in SOURCES}
    with (
        connect(path, timeout=120) as db,
        connect(
            census.path.with_name("thermal-climbs.sqlite3").as_uri() + "?mode=ro",
            uri=True,
        ) as cache,
    ):
        db.executescript("""
            CREATE TABLE IF NOT EXISTS neighbour_points(
                source TEXT, ix INTEGER, iy INTEGER, level INTEGER, points BLOB,
                PRIMARY KEY(source,ix,iy,level));
            CREATE TABLE IF NOT EXISTS neighbour_flights(
                source TEXT, ix INTEGER, iy INTEGER, flights TEXT, signature TEXT,
                PRIMARY KEY(source,ix,iy));
        """)
        for cell in cells:
            for source, key in keys.items():
                signature = json.dumps(
                    [NEIGHBOUR_LATTICE_VERSION, key, cell.ground_m, cell.max_alt_m]
                )
                if db.execute(
                    "SELECT signature FROM neighbour_flights "
                    "WHERE source=? AND ix=? AND iy=?",
                    (source, cell.ix, cell.iy),
                ).fetchone() == (signature,):
                    continue
                flights, parts = {}, []
                for frame in neighbour_frames(cell):
                    if (frame.ix, frame.iy) in published:
                        # Export already checked these against the square's visitors.
                        rows = db.execute(
                            "SELECT discipline,flight_id,edges FROM climbs "
                            "WHERE source=? AND ix=? AND iy=?",
                            (source, frame.ix, frame.iy),
                        ).fetchall()
                    else:
                        visitors = census.flights(frame)
                        # A square nobody crossed (e.g. at sea) comes back from
                        # SQL with object columns; cast before testing the clock.
                        utc = visitors.start_utc.astype(float) + visitors.trim_start
                        known = visitors.loc[np.isfinite(utc.astype(float))]
                        expected = set(
                            zip(known.discipline, known.flight_id, strict=True)
                        )
                        # A flight decoded for another square also leaves an empty
                        # product here; only this square's visitors belong to it.
                        rows = [
                            r
                            for r in cache.execute(
                                "SELECT discipline,flight_id,edges FROM climbs "
                                "WHERE cache_key=? AND ix=? AND iy=?",
                                (key, frame.ix, frame.iy),
                            )
                            if (r[0], r[1]) in expected
                        ]
                        if len(rows) != len(expected):
                            raise RuntimeError(
                                f"{frame.ix}/{frame.iy}/{source}: "
                                f"{len(expected) - len(rows)} flights still need "
                                "climb products"
                            )
                    for discipline, fid, blob in rows:
                        with np.load(io.BytesIO(blob), allow_pickle=False) as saved:
                            points = lattice_points(saved["edges"], frame)
                        if len(points):
                            number = flights.setdefault((discipline, fid), len(flights))
                            parts.append(
                                np.column_stack((points, np.full(len(points), number)))
                            )
                values = np.concatenate(parts) if parts else np.empty((0, 5))
                values = values[np.argsort(values[:, 0], kind="stable")]
                levels, starts = np.unique(values[:, 0], return_index=True)
                db.execute(
                    "DELETE FROM neighbour_points WHERE source=? AND ix=? AND iy=?",
                    (source, cell.ix, cell.iy),
                )
                for level, chunk in zip(
                    levels, np.split(values, starts[1:]), strict=True
                ):
                    buffer = io.BytesIO()
                    np.savez_compressed(
                        buffer,
                        points=chunk[:, 1:4],
                        flight=chunk[:, 4].astype(np.int32),
                    )
                    db.execute(
                        "INSERT INTO neighbour_points VALUES (?,?,?,?,?)",
                        (source, cell.ix, cell.iy, int(level), buffer.getvalue()),
                    )
                db.execute(
                    "INSERT OR REPLACE INTO neighbour_flights VALUES (?,?,?,?,?)",
                    (
                        source,
                        cell.ix,
                        cell.iy,
                        json.dumps([list(f) for f in flights]),
                        signature,
                    ),
                )
                db.commit()
                progress(
                    f"{cell.terrain} {cell.ix}/{cell.iy} {source}: "
                    f"{len(values):,} neighbour crossings, {len(flights):,} flights"
                )
        provenance = json.dumps(keys)
        if db.execute(
            "SELECT value FROM metadata WHERE key='neighbour_segmentations'"
        ).fetchone() != (provenance,):
            db.execute(
                "INSERT OR REPLACE INTO metadata VALUES ('neighbour_segmentations',?)",
                (provenance,),
            )
            db.commit()


def prepare_neighbour_imagery(path, *, size=4000, progress=print):
    """Save each background kind for every neighbouring square, under its ix/iy key.

    ``size`` is the image width in pixels for a 5 km square (published cells use
    4000, 1.25 m/px). Hillshade keeps the published cells' 600 px. Downloads go
    to a temporary folder: the snapshot alone keeps them.
    """
    from .thermal_imagery import LAYERS, acquisition_dates, fetch_image
    from .thermal_relief import fetch_relief

    store = ThermalStore(path)
    squares = _squares(store)

    def download(cell, kind):
        """Fetch one image concurrently; the caller alone writes SQLite."""
        with tempfile.TemporaryDirectory() as folder:
            if kind == "relief":
                payload, info = fetch_relief(
                    Path(path).parent, cell.bounds, 2154, (600, 600)
                )
            else:
                info, payload = fetch_image(
                    Path(folder), cell.bounds, "EPSG:2154", (size, size), kind
                )
                if kind == "aerial":
                    info["acquisition_dates"], info["acquisition_source"] = (
                        acquisition_dates(cell.bounds, "EPSG:2154")
                    )
        return kind, f"{cell.ix}/{cell.iy}", json.dumps(info), payload

    with (
        connect(path, timeout=120) as db,
        ThreadPoolExecutor(max_workers=3) as pool,
    ):
        db.execute("""CREATE TABLE IF NOT EXISTS backgrounds(
            kind TEXT,key TEXT,metadata TEXT,image BLOB,PRIMARY KEY(kind,key))""")
        db.execute("""CREATE TABLE IF NOT EXISTS relief(
            key TEXT PRIMARY KEY,metadata TEXT,png BLOB)""")
        pending = []
        for cell in squares:
            key = f"{cell.ix}/{cell.iy}"
            for kind in LAYERS:
                saved = db.execute(
                    "SELECT metadata FROM backgrounds WHERE kind=? AND key=?",
                    (kind, key),
                ).fetchone()
                if saved and tuple(json.loads(saved[0]).get("size", ())) == (
                    size,
                    size,
                ):
                    continue
                pending.append(pool.submit(download, cell, kind))
            if not db.execute("SELECT 1 FROM relief WHERE key=?", (key,)).fetchone():
                pending.append(pool.submit(download, cell, "relief"))
        progress(f"{len(pending)} neighbour backgrounds to download")
        for done, future in enumerate(as_completed(pending), 1):
            kind, key, info, payload = future.result()
            if kind == "relief":
                db.execute(
                    "INSERT OR REPLACE INTO relief VALUES (?,?,?)", (key, info, payload)
                )
            else:
                db.execute(
                    "INSERT OR REPLACE INTO backgrounds VALUES (?,?,?,?)",
                    (kind, key, info, payload),
                )
            db.commit()
            progress(f"Saved {done}/{len(pending)}: IGN {kind} {key}")
