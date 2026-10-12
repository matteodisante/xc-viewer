"""Cell terrain references from attributed, offline IGN elevation rasters.

A cell's altitude band comes from its highest DEM pixel; its planes start at its
lowest DEM pixel. Both are read from the same area-weighted summary.
"""

from __future__ import annotations

import hashlib
import io
import json
import sqlite3
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image

from .geography import TERRAIN_ORDER, classify_terrain
from .sqlite import connect
from .thermal_orography import DATA_DIRECTORY

GROUND_REFERENCE = "ign-dem-cell-min-v1"
TERRAIN_RANKING = "ign-dem-cell-max-top-v1"
CLIMB_RANKING = "vilpellet-climb-runs-dem-max-top-v1"


class IncompleteTerrainError(ValueError):
    """The IGN DEM leaves part of the cell empty, e.g. beyond the French border."""


def terrain_category(highest_m):
    """Band a cell by its finite highest DEM elevation."""
    category = classify_terrain([highest_m])[0]
    if not category:
        raise ValueError(
            "Cell classification requires a finite highest terrain elevation"
        )
    return category


def cell_band(reference):
    """The altitude band of the cell a DEM summary describes."""
    return terrain_category(reference["maximum_m"])


def cell_ground(reference):
    """The plane reference: the lowest DEM elevation inside the cell."""
    return reference["minimum_m"]


def order_terrain_cells(cells):
    """Order DEM categories and visitor counts, breaking ties by grid coordinates."""
    return sorted(
        cells,
        key=lambda c: (TERRAIN_ORDER.index(c.terrain), -c.flights, c.ix, c.iy),
    )


def mean_terrain(z, raster_bounds, cell_bounds):
    """Area-weight unsmoothed raster pixels over exactly the requested square.

    Rows run north to south. Missing or implausible elevations anywhere inside
    the cell are errors: never substitute zero or a flight's launch altitude.
    """
    z = np.asarray(z, dtype=float)
    if z.ndim != 2 or not z.size:
        raise ValueError("Terrain must be a two-dimensional elevation raster")
    w, s, e, n = map(float, raster_bounds)
    cw, cs, ce, cn = map(float, cell_bounds)
    if not (w <= cw < ce <= e and s <= cs < cn <= n):
        raise ValueError("Terrain raster does not cover the complete cell")
    dx, dy = (e - w) / z.shape[1], (n - s) / z.shape[0]
    if max(dx, dy) > 25:
        raise ValueError("Terrain sampling must be 25 m or finer")
    x = w + np.arange(z.shape[1]) * dx
    y = n - np.arange(z.shape[0]) * dy
    wx = np.maximum(0, np.minimum(x + dx, ce) - np.maximum(x, cw))
    wy = np.maximum(0, np.minimum(y, cn) - np.maximum(y - dy, cs))
    weights = wy[:, None] * wx[None, :]
    inside = weights > 0
    values = z[inside]
    if not np.isfinite(values).all() or (values < -500).any() or (values > 9000).any():
        raise IncompleteTerrainError(
            "Missing or invalid terrain elevations inside the cell"
        )
    area = (ce - cw) * (cn - cs)
    if not np.isclose(weights.sum(), area, rtol=1e-10):
        raise ValueError("Terrain does not cover the complete cell area")
    return {
        "mean_m": float(np.sum(values * weights[inside]) / area),
        "minimum_m": float(values.min()),
        "maximum_m": float(values.max()),
        "samples": int(inside.sum()),
        "grid_m": [dx, dy],
        "coverage_fraction": 1.0,
    }


def terrain_reference(cell, folder=None):
    """Validate the saved raster; return its elevation summary and provenance."""
    folder = DATA_DIRECTORY if folder is None else Path(folder)
    terrain_path = folder / f"ign-terrain-{cell.ix}-{cell.iy}.json"
    provenance_path = (
        terrain_path
        if terrain_path.exists()
        else folder / f"ign-ridges-{cell.ix}-{cell.iy}.geojson"
    )
    if not provenance_path.exists():
        raise ValueError(
            f"Missing IGN terrain for cell {cell.ix}/{cell.iy}; "
            "prepare its elevation raster with prepare_thermal_ridges.py first"
        )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))["provenance"]
    if tuple(provenance["bounds_epsg2154"]) != cell.bounds:
        raise ValueError("Terrain provenance does not match the cell extent")
    raw = (folder / provenance["terrain_file"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != provenance["response_sha256"]:
        raise ValueError("Terrain raster hash differs from its provenance")
    with Image.open(io.BytesIO(raw)) as im:
        if im.mode != "F":
            raise ValueError("Terrain must contain float elevations, not map colours")
        summary = mean_terrain(
            np.asarray(im), provenance["raster_bounds_epsg2154"], cell.bounds
        )
    return {
        **summary,
        "reference": GROUND_REFERENCE,
        "bounds_epsg2154": cell.bounds,
        "raster_bounds_epsg2154": provenance["raster_bounds_epsg2154"],
        "response_sha256": provenance["response_sha256"],
        "source_url": provenance["source_url"],
        "dataset_url": provenance["dataset_url"],
        "retrieved_utc": provenance["retrieved_utc"],
        "attribution": "© IGN RGE ALTI · Licence Ouverte 2.0",
        "method": (
            "Area-weighted mean, minimum and maximum over the unsmoothed DEM "
            "pixels inside the cell"
        ),
        "vertical_reference": "IGN normal heights; recorder GNSS datum not harmonised",
    }


def fetch_terrain_reference(cell, folder):
    """Cache an official IGN float elevation window for an explicitly visited cell."""
    from urllib.parse import urlencode

    from .thermal_imagery import _download
    from .thermal_ridges import DATASET_URL, LAYER, SERVICE

    folder = Path(folder)
    if (folder / f"ign-terrain-{cell.ix}-{cell.iy}.json").exists():
        return terrain_reference(cell, folder)
    bounds = cell.bounds
    url = (
        SERVICE
        + "?"
        + urlencode(
            {
                "SERVICE": "WMS",
                "VERSION": "1.3.0",
                "REQUEST": "GetMap",
                "LAYERS": LAYER,
                "STYLES": "normal",
                "CRS": "EPSG:2154",
                "BBOX": ",".join(map(str, bounds)),
                "WIDTH": 200,
                "HEIGHT": 200,
                "FORMAT": "image/tiff",
            }
        )
    )
    raw = _download(url)
    with Image.open(io.BytesIO(raw)) as im:
        if im.mode != "F" or im.size != (200, 200):
            raise ValueError("IGN response must be a 200 x 200 float elevation raster")
        mean_terrain(np.asarray(im), bounds, bounds)
    provenance = {
        "bounds_epsg2154": bounds,
        "raster_bounds_epsg2154": bounds,
        "terrain_file": f"ign-terrain-{cell.ix}-{cell.iy}.tif",
        "response_sha256": hashlib.sha256(raw).hexdigest(),
        "source_url": url,
        "dataset_url": DATASET_URL,
        "retrieved_utc": datetime.now(UTC).isoformat(),
    }
    folder.mkdir(parents=True, exist_ok=True)
    (folder / provenance["terrain_file"]).write_bytes(raw)
    path = folder / f"ign-terrain-{cell.ix}-{cell.iy}.json"
    temporary = path.with_suffix(".building.json")
    temporary.write_text(json.dumps({"provenance": provenance}), encoding="utf-8")
    temporary.replace(path)
    return terrain_reference(cell, folder)


def terrain_cell(cell, reference):
    """Band by the DEM maximum, start planes at the DEM minimum.

    The old launch median is retained only as an audit.
    """
    return replace(
        cell,
        terrain=cell_band(reference),
        ground_m=cell_ground(reference),
        launch_median_m=(
            cell.ground_m
            if cell.launch_median_m is None and cell.launches
            else cell.launch_median_m
        ),
    )


def upgrade_terrain_store(path, *, folder=None, progress=print):
    """Atomically replace a legacy snapshot after rebuilding its height lattice.

    The caller holds .prepare.lock. A separate SQLite copy retains all imagery,
    edges and dates; committed lattice batches resume after an interruption.
    The original snapshot remains intact until the copy passes verification.
    """
    from .thermal_daily import POINT_LATTICE_VERSION, prepare_daily
    from .thermal_geometry import ThermalCell
    from .thermal_store import _connect

    path = Path(path).resolve()
    with _connect(path) as source:
        metadata = dict(source.execute("SELECT key,value FROM metadata"))
        cells = [
            ThermalCell(**json.loads(r[0]))
            for r in source.execute("SELECT payload FROM cells ORDER BY position")
        ]
        references = [terrain_reference(cell, folder) for cell in cells]
        reference_signature = hashlib.sha256(
            json.dumps(references, sort_keys=True).encode()
        ).hexdigest()
        if (
            metadata.get("terrain_signature") == reference_signature
            and metadata.get("point_lattice") == POINT_LATTICE_VERSION
            and metadata.get("ground_reference") == GROUND_REFERENCE
        ):
            progress("Terrain references and intersections already prepared")
            return path
        identity = json.dumps(
            [path.stat().st_size, path.stat().st_mtime_ns, reference_signature]
        )
        temporary = path.with_suffix(".terrain-building.sqlite3")
        resume = False
        if temporary.exists():
            try:
                with _connect(temporary) as previous:
                    resume = previous.execute(
                        "SELECT value FROM metadata WHERE key='terrain_upgrade_source'"
                    ).fetchone() == (identity,)
            except sqlite3.DatabaseError:
                # An interruption during SQLite backup can leave no schema yet.
                resume = False
        if not resume:
            temporary.unlink(missing_ok=True)
            progress("Copying saved edges and images for the terrain upgrade")
            with connect(temporary) as target:
                source.backup(target)
                target.execute(
                    "INSERT OR REPLACE INTO metadata "
                    "VALUES ('terrain_upgrade_source',?)",
                    (identity,),
                )
    with connect(temporary) as db:
        db.execute("""CREATE TABLE IF NOT EXISTS terrain(
            ix INTEGER,iy INTEGER,metadata TEXT,PRIMARY KEY(ix,iy))""")
        rebased = db.execute(
            "SELECT value FROM metadata WHERE key='terrain_signature'"
        ).fetchone() == (reference_signature,)
        if not rebased:
            for cell, reference in zip(cells, references, strict=True):
                updated = terrain_cell(cell, reference)
                db.execute(
                    "UPDATE cells SET payload=?,height=MIN(?,MAX(0,height+?)) "
                    "WHERE ix=? AND iy=?",
                    (
                        json.dumps(asdict(updated)),
                        updated.max_agl_m,
                        cell.ground_m - updated.ground_m,
                        cell.ix,
                        cell.iy,
                    ),
                )
                db.execute(
                    "INSERT OR REPLACE INTO terrain VALUES (?,?,?)",
                    (cell.ix, cell.iy, json.dumps(reference)),
                )
                progress(
                    f"{cell.ix}/{cell.iy}: lowest terrain {updated.ground_m:.2f} m "
                    f"({reference['samples']:,} DEM pixels)"
                )
            db.executemany(
                "INSERT OR REPLACE INTO metadata VALUES (?,?)",
                [
                    ("version", "3"),
                    ("ground_reference", GROUND_REFERENCE),
                    ("terrain_signature", reference_signature),
                ],
            )
        db.commit()
    prepare_daily(temporary, progress=progress)
    with connect(temporary) as db:
        if db.execute("PRAGMA quick_check").fetchone() != ("ok",):
            raise ValueError("Terrain snapshot failed SQLite integrity check")
    temporary.replace(path)
    progress(f"Published terrain-referenced intersections: {path}")
    return path
