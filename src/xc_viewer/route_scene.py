"""Selected cleaned trajectories and bounded, cached IGN terrain windows."""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.parse import urlencode
from urllib.request import urlopen
from uuid import uuid4

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

from .core.preproc.enu import LocalFrame
from .geodesy import enu_to_geodetic
from .route_density import DensityAtlas, load_density
from .route_index import FIX_COLUMNS, ROUTE_CELL_M, matching_flights, select_flights
from .route_times import DepartureWindow, filter_departures, with_flight_times
from .sqlite import connect
from .thermal_geometry import continuous_edges, project
from .thermal_index import _check_cancel
from .thermal_ridges import DATASET_URL, LAYER, SERVICE

AERIAL_LAYER = "ORTHOIMAGERY.ORTHOPHOTOS"
AERIAL_DATASET_URL = "https://www.data.gouv.fr/datasets/bd-ortho-r"
AERIAL_MAX_SIDE = 3072


@dataclass
class RouteScene:
    """Selected full cleaned tracks in one common metric frame."""

    selected: pd.DataFrame
    total: int
    pair: tuple
    tracks: list
    bounds: tuple
    origin: np.ndarray
    terrain: tuple | None = None
    terrain_error: str | None = None
    aerial: np.ndarray | None = None
    aerial_reference: dict | None = None
    aerial_error: str | None = None
    density: DensityAtlas | None = None
    density_error: str | None = None
    departure_cohort: pd.DataFrame | None = None
    departure_window: DepartureWindow | None = None
    cell_m: float = ROUTE_CELL_M


def track_segments(fixes, frame):
    """Project cleaned fixes and split every unsupported edge before drawing."""
    fixes = fixes.sort_values("t", kind="stable").reset_index(drop=True)
    lat, lon = enu_to_geodetic(fixes.E, fixes.N, fixes.z, frame)
    x, y = project(lon, lat)
    xyz = np.column_stack((x, y, fixes.z)).astype(float)
    finite = np.isfinite(xyz).all(axis=1)
    edges = continuous_edges(fixes) & finite[:-1] & finite[1:]
    return [
        part
        for part in np.split(xyz, np.flatnonzero(~edges) + 1)
        if len(part) and np.isfinite(part).all()
    ]


def scene_bounds(tracks, pair, cell_m=ROUTE_CELL_M):
    """Cover all retained geometry and both full endpoint cells, with a margin."""
    minima = [segment.min(axis=0)[:2] for track in tracks for segment in track]
    maxima = [segment.max(axis=0)[:2] for track in tracks for segment in track]
    cells = np.asarray(pair).reshape(-1, 2) * cell_m
    low = np.min([*minima, *cells], axis=0) - 2000
    high = np.max([*maxima, *(cells + cell_m)], axis=0) + 2000
    return (*np.floor(low / 1000) * 1000, *np.ceil(high / 1000) * 1000)


def terrain_request(bounds):
    """Limit the overview DEM to 600 pixels per side, sampled at >=100 m."""
    west, south, east, north = bounds
    step = max(100, (east - west) / 600, (north - south) / 600)
    width, height = (
        max(2, int(np.ceil((east - west) / step))),
        max(2, int(np.ceil((north - south) / step))),
    )
    params = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetMap",
        "LAYERS": LAYER,
        "STYLES": "normal",
        "CRS": "EPSG:2154",
        "BBOX": ",".join(map(str, bounds)),
        "WIDTH": width,
        "HEIGHT": height,
        "FORMAT": "image/tiff",
    }
    return SERVICE + "?" + urlencode(params), (width, height)


def decode_terrain(raw, bounds, size):
    """Validate numeric elevations, preserving missing terrain as holes."""
    with Image.open(io.BytesIO(raw)) as image:
        if image.mode != "F" or image.size != size:
            raise ValueError(
                "IGN must return floating-point elevations at the requested size"
            )
        z = np.asarray(image).copy()[::-1]
    valid = np.isfinite(z) & (z >= -500) & (z <= 9000)
    if not valid.any():
        raise ValueError("No IGN elevation coverage for this flight area")
    z[~valid] = np.nan
    west, south, east, north = bounds
    width, height = size
    dx, dy = (east - west) / width, (north - south) / height
    # Extend half a pixel to cover the entire requested rectangle.
    x = np.r_[west, west + (np.arange(width) + 0.5) * dx, east]
    y = np.r_[south, south + (np.arange(height) + 0.5) * dy, north]
    return x, y, np.pad(z, 1, mode="edge"), (dx, dy), float(valid.mean())


def load_terrain(bounds, folder, *, cancel=None):
    """Download one small official DEM window, then reuse its verified cache.

    A single response is at most 600 x 600 floats. A 30 s network timeout keeps
    cancellation and application shutdown bounded. Missing coverage stays absent;
    no flight altitude is altered to force it above the ground.
    """
    url, size = terrain_request(bounds)
    key = hashlib.sha256(url.encode()).hexdigest()[:24]
    path = folder / f"ign-route-{key}.npz"
    _check_cancel(cancel)
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            raw = saved["raw"].tobytes()
            reference = json.loads(str(saved["reference"]))
        if (
            reference["source_url"] != url
            or hashlib.sha256(raw).hexdigest() != reference["response_sha256"]
        ):
            raise ValueError("Saved IGN terrain differs from its provenance")
    else:
        with urlopen(url, timeout=30) as response:
            raw = response.read(8_000_001)
        if len(raw) > 8_000_000:
            raise ValueError("IGN response exceeded the bounded terrain download")
        decode_terrain(raw, bounds, size)
        _check_cancel(cancel)
        reference = {
            "source_url": url,
            "dataset_url": DATASET_URL,
            "response_sha256": hashlib.sha256(raw).hexdigest(),
            "retrieved_utc": datetime.now(UTC).isoformat(),
            "bounds_epsg2154": bounds,
            "attribution": "© IGN RGE ALTI · Licence Ouverte 2.0",
        }
        folder.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{key}-{uuid4().hex}.npz")
        try:
            np.savez_compressed(
                temporary,
                raw=np.frombuffer(raw, dtype=np.uint8),
                reference=json.dumps(reference),
            )
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    x, y, z, grid, coverage = decode_terrain(raw, bounds, size)
    return x, y, z, {**reference, "grid_m": grid, "coverage": coverage}


def aerial_request(bounds):
    """Request one north-up overview orthophoto over exactly the DEM bounds."""
    west, south, east, north = bounds
    step = max(east - west, north - south) / AERIAL_MAX_SIDE
    size = tuple(
        max(2, min(AERIAL_MAX_SIDE, round(span / step)))
        for span in (east - west, north - south)
    )
    params = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetMap",
        "LAYERS": AERIAL_LAYER,
        "STYLES": "normal",
        "CRS": "EPSG:2154",
        "BBOX": ",".join(map(str, bounds)),
        "WIDTH": size[0],
        "HEIGHT": size[1],
        "FORMAT": "image/jpeg",
    }
    return SERVICE + "?" + urlencode(params), size


def decode_aerial(raw, size):
    """Validate image dimensions and preserve the service's north-first row order."""
    with Image.open(io.BytesIO(raw)) as image:
        if image.size != size:
            raise ValueError("IGN orthophoto dimensions differ from the request")
        return np.asarray(image.convert("RGB")).copy()


def load_aerial(bounds, folder, *, cancel=None):
    """Cache a verified aerial mosaic without downloading native-resolution tiles."""
    url, size = aerial_request(bounds)
    key = hashlib.sha256(url.encode()).hexdigest()[:24]
    path = folder / f"ign-aerial-{key}.npz"
    _check_cancel(cancel)
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            raw = saved["raw"].tobytes()
            reference = json.loads(str(saved["reference"]))
        if (
            reference["source_url"] != url
            or tuple(reference["bounds_epsg2154"]) != tuple(bounds)
            or hashlib.sha256(raw).hexdigest() != reference["response_sha256"]
        ):
            raise ValueError("Saved IGN orthophoto differs from its provenance")
    else:
        with urlopen(url, timeout=30) as response:
            raw = response.read(32_000_001)
        if len(raw) > 32_000_000:
            raise ValueError("IGN orthophoto exceeded the bounded download")
        decode_aerial(raw, size)
        _check_cancel(cancel)
        reference = {
            "source_url": url,
            "dataset_url": AERIAL_DATASET_URL,
            "response_sha256": hashlib.sha256(raw).hexdigest(),
            "retrieved_utc": datetime.now(UTC).isoformat(),
            "bounds_epsg2154": bounds,
            "crs": "EPSG:2154",
            "size": size,
            "attribution": "© IGN BD ORTHO · Licence Ouverte 2.0",
            "note": "Mosaic acquisition dates vary and differ from flight dates.",
        }
        folder.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{key}-{uuid4().hex}.npz")
        try:
            np.savez_compressed(
                temporary,
                raw=np.frombuffer(raw, dtype=np.uint8),
                reference=json.dumps(reference),
            )
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    west, south, east, north = bounds
    return decode_aerial(raw, size), {
        **reference,
        "grid_m": ((east - west) / size[0], (north - south) / size[1]),
    }


def load_scene(
    index,
    flights,
    pair,
    discipline=None,
    *,
    departure_window=None,
    progress=lambda _: None,
    cancel=None,
):
    """Filter complete endpoint cohorts before sampling or reading their geometry."""
    index.verify()
    candidates = matching_flights(flights, pair, discipline)
    if candidates.empty:
        raise ValueError("No cleaned flights match these directed cells")
    cohort = with_flight_times(
        candidates, index.disciplines, progress=progress, cancel=cancel
    )
    selected, total = select_flights(filter_departures(cohort, departure_window), pair)
    if selected.empty:
        unknown = int(cohort.departure_utc.isna().sum())
        raise ValueError(
            f"No departures in {departure_window.label}. "
            f"{unknown} flights have unavailable departure dates. "
            "Choose another window or disable the departure filter."
        )
    tracks, _ = read_tracks(index, selected, progress=progress, cancel=cancel)
    index.verify()
    bounds = scene_bounds(tracks, pair)
    origin = (np.asarray(bounds[:2]) + bounds[2:]) / 2
    for track in tracks:
        for i, segment in enumerate(track):
            segment[:, :2] -= origin
            track[i] = segment.astype(np.float32)
    scene = RouteScene(
        selected,
        total,
        tuple(pair),
        tracks,
        bounds,
        origin,
        departure_cohort=cohort,
        departure_window=departure_window,
    )
    return load_background(scene, index, progress=progress, cancel=cancel)


def read_tracks(index, selected, *, progress=lambda _: None, cancel=None):
    """Read selected cleaned fixes once, retaining boundaries and measuring paths."""
    index.verify()
    pieces = {(r.discipline, r.flight_id): [] for r in selected.itertuples()}
    with connect(index.path) as db:
        for disc in index.disciplines:
            ids = selected.loc[selected.discipline == disc.name, "flight_id"].tolist()
            if not ids:
                continue
            query = (
                "SELECT DISTINCT row_group FROM parts "
                "WHERE discipline=? AND flight_id IN ("
                + ",".join("?" for _ in ids)
                + ") ORDER BY row_group"
            )
            groups = [r[0] for r in db.execute(query, (disc.name, *ids))]
            archive = pq.ParquetFile(disc.config().derived_dir / "fixes.parquet")
            for i, group in enumerate(groups):
                _check_cancel(cancel)
                progress(
                    f"{disc.name}: reading selected tracks {i + 1}/{len(groups)} blocks"
                )
                frame = archive.read_row_group(group, columns=FIX_COLUMNS).to_pandas()
                frame["flight_id"] = frame.flight_id.astype(str)
                frame = frame.loc[frame.flight_id.isin(ids)]
                for fid, flight in frame.groupby("flight_id", sort=False):
                    pieces[disc.name, fid].append(flight)
    tracks, metrics = [], []
    for row in selected.itertuples():
        _check_cancel(cancel)
        parts = pieces[row.discipline, row.flight_id]
        if not parts:
            raise ValueError(
                f"Missing cleaned geometry for {row.discipline} {row.flight_id}"
            )
        fixes = pd.concat(parts, ignore_index=True)
        if not np.allclose(
            [fixes.t.min(), fixes.t.max()], [row.t0, row.t1], rtol=0, atol=1e-6
        ):
            raise ValueError(
                "Track endpoints disagree with the route census; rebuild it"
            )
        track = track_segments(fixes, LocalFrame(row.lat0, row.lon0, row.alt0))
        tracks.append(track)
        edges = continuous_edges(fixes.sort_values("t", kind="stable"))
        retained_s = float(np.diff(np.sort(fixes.t.to_numpy()))[edges].sum())
        metrics.append(
            {
                "path_km": sum(
                    np.linalg.norm(np.diff(s[:, :2], axis=0), axis=1).sum()
                    for s in track
                )
                / 1000,
                "net_km": float(np.linalg.norm(track[-1][-1, :2] - track[0][0, :2]))
                / 1000,
                "retained_s": retained_s,
                "gap_s": max(0.0, row.duration_s - retained_s),
                "clean_segments": len(track),
            }
        )
    index.verify()
    return tracks, pd.DataFrame(metrics, index=selected.index)


def load_background(scene, index, *, progress=lambda _: None, cancel=None):
    """Attach the same all-flight heat texture and bounded IGN layers to any cohort."""
    _check_cancel(cancel)
    progress("Reading thermal hours from all archived flights crossing the area…")
    try:
        scene.density = load_density(scene.bounds, index.disciplines)
    except (OSError, ValueError) as exc:
        scene.density_error = str(exc)
    _check_cancel(cancel)
    progress("Reading IGN terrain for the selected flight area…")
    try:
        scene.terrain = load_terrain(
            scene.bounds, index.path.parent / "route-terrain", cancel=cancel
        )
    except (OSError, ValueError) as exc:
        scene.terrain_error = str(exc)
    _check_cancel(cancel)
    if scene.terrain is not None:
        progress("Reading IGN aerial imagery for the same terrain extent…")
        try:
            scene.aerial, scene.aerial_reference = load_aerial(
                scene.bounds, index.path.parent / "route-terrain", cancel=cancel
            )
        except (OSError, ValueError) as exc:
            scene.aerial_error = str(exc)
    _check_cancel(cancel)
    return scene


def terrain_mesh(terrain, origin):
    """Return only triangles with three valid DEM vertices; never fill holes."""
    x, y, z, _ = terrain
    xx, yy = np.meshgrid(x - origin[0], y - origin[1])
    vertices = np.column_stack((xx.ravel(), yy.ravel(), z.ravel())).astype(np.float32)
    ids = np.arange(z.size).reshape(z.shape)
    a, b, c, d = (
        ids[:-1, :-1].ravel(),
        ids[:-1, 1:].ravel(),
        ids[1:, :-1].ravel(),
        ids[1:, 1:].ravel(),
    )
    faces = np.concatenate((np.column_stack((a, b, c)), np.column_stack((b, d, c))))
    faces = faces[np.isfinite(vertices[faces, 2]).all(axis=1)]
    vertices[~np.isfinite(vertices[:, 2]), 2] = 0
    return vertices, faces
