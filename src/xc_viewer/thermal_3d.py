"""Verified terrain and saved 20 m climb intersections for any prepared cell."""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from PIL import Image

from .thermal_daily import height_levels, in_windows
from .thermal_geometry import CELL_M, ThermalCell
from .thermal_ground import cell_ground, mean_terrain, terrain_reference
from .thermal_orography import DATA_DIRECTORY
from .thermal_store import CancelledError, neighbour_frames


@dataclass(frozen=True)
class TerrainScene:
    """Metres east/north of the cell centre, with absolute altitude in metres."""

    cell: ThermalCell
    x: np.ndarray
    y: np.ndarray
    terrain: np.ndarray
    points: np.ndarray
    reference: dict
    climb_runs: int
    contributing_flights: int
    selected_flights: int
    unavailable_flights: int
    unclassified_flights: int
    unknown_clock_flights: int
    start: float
    end: float
    aerial: np.ndarray | None = None
    aerial_reference: dict | None = None
    aerial_error: str | None = None
    label: str = ""
    area_km: int = 5


def area_bounds(cell, area_km):
    """Centre either supported display square on the selected 5 km cell."""
    if area_km not in (5, 10):
        raise ValueError("Choose a 5 x 5 km or 10 x 10 km area")
    west, south, east, north = cell.bounds
    margin = (area_km * 1000 - CELL_M) / 2
    return west - margin, south - margin, east + margin, north + margin


def cell_label(store, cell):
    """Validate the selected saved cell and identify its Vilpellet category rank."""
    if not store.has_climb_ranking:
        raise ValueError("The 3D view requires the Vilpellet climb-ranked snapshot")
    cells = sorted(
        (c for c in store.cells() if c.terrain == cell.terrain),
        key=lambda c: (-store.activity_counts[c.ix, c.iy]["climb_runs"], c.ix, c.iy),
    )
    if cell not in cells:
        raise ValueError(
            "The selected cell is missing or changed in the prepared snapshot"
        )
    return f"{cell.terrain} #{cells.index(cell) + 1}"


def read_surface(store, cell, *, reference=None):
    """Read the original IGN floats, checking the snapshot hash and coordinates.

    Files remain offline. Raster rows initially run north to south; return a
    south-to-north mesh. Extend the outer half-pixel to the cell boundary with
    the nearest elevation, so the displayed surface covers the complete square.
    """
    if reference is None:
        reference = store.terrain_reference(cell)
    name = f"ign-terrain-{cell.ix}-{cell.iy}.tif"
    candidates = (
        store.path.parent / "exploration/terrain" / name,
        DATA_DIRECTORY / name,
    )
    expected = reference["response_sha256"]
    raw = None
    for path in candidates:
        if path.is_file():
            candidate = path.read_bytes()
            if hashlib.sha256(candidate).hexdigest() == expected:
                raw = candidate
                break
    if raw is None:
        raise ValueError(
            f"The verified IGN elevation raster {name} is missing or changed. "
            "Restore it in the data folder's exploration/terrain folder."
        )
    if tuple(reference["bounds_epsg2154"]) != cell.bounds:
        raise ValueError("Terrain reference belongs to a different cell")
    with Image.open(io.BytesIO(raw)) as im:
        if im.mode != "F":
            raise ValueError(
                "The terrain raster must contain floating-point elevations"
            )
        z = np.asarray(im).copy()
    bounds = reference["raster_bounds_epsg2154"]
    summary = mean_terrain(z, bounds, cell.bounds)
    if not np.isclose(cell_ground(summary), cell.ground_m, rtol=0, atol=1e-5):
        raise ValueError("DEM minimum differs from the saved intersection reference")
    west, south, east, north = cell.bounds
    w, s, e, n = bounds
    dx, dy = (e - w) / z.shape[1], (n - s) / z.shape[0]
    x = w + (np.arange(z.shape[1]) + 0.5) * dx
    y = n - (np.arange(z.shape[0]) + 0.5) * dy
    cols = (x >= west) & (x < east)
    rows = (y >= south) & (y < north)
    z = z[np.ix_(rows, cols)][::-1]
    x, y = x[cols], y[rows][::-1]
    if min(z.shape) < 2:
        raise ValueError("The terrain grid is too small to form a surface")
    x = np.r_[west, x, east] - (west + east) / 2
    y = np.r_[south, y, north] - (south + north) / 2
    return x, y, np.pad(z, 1, mode="edge"), reference


def read_neighbour_surface(store, cell):
    """Validate a neighbour's own DEM without changing the cloud's height datum."""
    published = {(c.ix, c.iy): c for c in store.cells()}
    if (cell.ix, cell.iy) in published:
        return read_surface(store, published[cell.ix, cell.iy])
    for folder in (store.path.parent / "exploration/terrain", DATA_DIRECTORY):
        if any(
            (folder / name).exists()
            for name in (
                f"ign-terrain-{cell.ix}-{cell.iy}.json",
                f"ign-ridges-{cell.ix}-{cell.iy}.geojson",
            )
        ):
            reference = terrain_reference(cell, folder)
            return read_surface(
                store,
                replace(cell, ground_m=cell_ground(reference)),
                reference=reference,
            )
    raise ValueError(
        f"The data folder has no IGN terrain for the neighbouring cell "
        f"{cell.ix}/{cell.iy}. Choose the 5 x 5 km area."
    )


def read_area_surface(store, cell, check_cancel, progress):
    """Join the nine aligned DEM grids and crop a centred 10 km square."""
    tiles, references = {}, []
    x, y, z, reference = read_surface(store, cell)
    tiles[cell.ix, cell.iy] = z[1:-1, 1:-1]
    references.append(reference)
    for neighbour in neighbour_frames(cell):
        check_cancel()
        progress(f"Reading IGN terrain {neighbour.ix}/{neighbour.iy}…")
        nx, ny, nz, ref = read_neighbour_surface(store, neighbour)
        if (
            nx.shape != x.shape
            or ny.shape != y.shape
            or not np.allclose(nx, x, rtol=0, atol=1e-6)
            or not np.allclose(ny, y, rtol=0, atol=1e-6)
        ):
            raise ValueError("Neighbouring terrain grids must use the same sampling")
        tiles[neighbour.ix, neighbour.iy] = nz[1:-1, 1:-1]
        references.append(ref)
    z = np.block(
        [[tiles[cell.ix + dx, cell.iy + dy] for dx in (-1, 0, 1)] for dy in (-1, 0, 1)]
    )
    x = np.concatenate([x[1:-1] + dx * CELL_M for dx in (-1, 0, 1)])
    y = np.concatenate([y[1:-1] + dy * CELL_M for dy in (-1, 0, 1)])
    cols, rows = (x >= -5000) & (x < 5000), (y >= -5000) & (y < 5000)
    return (
        np.r_[-5000, x[cols], 5000],
        np.r_[-5000, y[rows], 5000],
        np.pad(z[np.ix_(rows, cols)], 1, mode="edge"),
        {"grid_m": reference["grid_m"], "tiles": references},
    )


def points_every_20m(frame, cell, start, end, *, area_km=5, windows=None):
    """Select true 20 m levels, including neither odd levels nor an off-grid ceiling.

    The saved level is an index, not an altitude. The last 10 m lattice entry
    can be an irregular exact ceiling, so index parity alone is insufficient.
    Time and cell masks match the 2D plane view, including its optional daily
    hour ``windows``. No point is randomly thinned.
    """
    west, south, east, north = area_bounds(cell, area_km)
    if frame is None:
        raise ValueError("Saved intersection levels are required for the 3D view")
    if frame.empty:
        return np.empty((0, 3), dtype=np.float32), 0
    levels = height_levels(cell.max_agl_m)
    indices = frame.level.to_numpy(dtype=float)
    if (
        not np.isfinite(indices).all()
        or (indices != np.floor(indices)).any()
        or (indices < 0).any()
        or (indices >= len(levels)).any()
    ):
        raise ValueError("Invalid saved intersection level")
    heights = levels[indices.astype(int)]
    keep = (
        np.isclose(heights / 20, np.round(heights / 20), rtol=0, atol=1e-8)
        & frame.x.between(west, east, inclusive="left").to_numpy()
        & frame.y.between(south, north, inclusive="left").to_numpy()
        & frame.utc.between(start, end).to_numpy()
    )
    if windows is not None:
        keep &= in_windows(frame.utc, windows)
    selected = frame.loc[keep]
    xyz = np.column_stack(
        (
            selected.x.to_numpy() - (west + east) / 2,
            selected.y.to_numpy() - (south + north) / 2,
            cell.ground_m + heights[keep],
        )
    ).astype(np.float32)
    flights = len(selected[["discipline", "flight_id"]].drop_duplicates())
    return xyz, flights


def read_neighbour_points(store, cell, start, end, check_cancel, progress):
    """Read only true 20 m levels, retaining the centre's planes and flight IDs."""
    flights = store.neighbour_flights(cell, "vilpellet")
    if flights is None:
        raise ValueError(
            "The data folder has no prepared neighbouring cells. "
            "Choose the 5 x 5 km area."
        )
    identities = np.asarray(flights, dtype=object).reshape(-1, 2)
    west, south, east, north = area_bounds(cell, 10)
    parts = []
    for level, height in enumerate(height_levels(cell.max_agl_m)):
        if not np.isclose(height / 20, round(height / 20), rtol=0, atol=1e-8):
            continue
        check_cancel()
        progress(f"Reading neighbouring intersections at {height:g} m…")
        frame = store.neighbour_points(cell, "vilpellet", level)
        frame = frame.loc[
            frame.utc.between(start, end)
            & frame.x.between(west, east, inclusive="left")
            & frame.y.between(south, north, inclusive="left")
        ].copy()
        if frame.empty:
            continue
        numbers = frame.pop("flight").to_numpy(dtype=np.int64)
        if (numbers < 0).any() or (numbers >= len(identities)).any():
            raise ValueError("Invalid saved neighbour flight identity")
        frame["discipline"], frame["flight_id"] = identities[numbers].T
        frame["level"] = level
        parts.append(frame)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def read_aerial(store, cell):
    """Use the saved north-up orthophoto only when its metric bounds match."""
    saved = store.background(cell, "aerial")
    if saved is None:
        return None, None
    image, reference = saved
    if (
        reference.get("crs") != "EPSG:2154"
        or tuple(reference.get("extent", ())) != cell.bounds
    ):
        raise ValueError(
            "The saved aerial photo does not match the terrain coordinates"
        )
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("The saved aerial photo must be an RGB image")
    return image, reference


def read_area_aerial(store, cell, check_cancel):
    """Crop saved north-up photographs into a 10 km mosaic without resampling."""
    centre, reference = read_aerial(store, cell)
    if centre is None:
        return None, None
    height, width = centre.shape[:2]
    if height % 2 or width % 2:
        raise ValueError("The 10 km photo requires even-sized saved image tiles")
    mosaic = np.empty((height * 2, width * 2, 3), dtype=np.uint8)
    west, south, east, north = area_bounds(cell, 10)
    references = []
    for tile in [cell, *neighbour_frames(cell)]:
        check_cancel()
        image, ref = (centre, reference) if tile == cell else read_aerial(store, tile)
        if image is None:
            raise ValueError(f"Missing aerial photo for neighbour {tile.ix}/{tile.iy}")
        if image.shape != centre.shape:
            raise ValueError("Neighbouring aerial photos must use the same sampling")
        w, s, e, n = tile.bounds
        cw, cs, ce, cn = max(w, west), max(s, south), min(e, east), min(n, north)
        x0, x1 = [round((v - w) / CELL_M * width) for v in (cw, ce)]
        y0, y1 = [round((n - v) / CELL_M * height) for v in (cn, cs)]
        tx, ty = (
            round((cw - west) / CELL_M * width),
            round((north - cn) / CELL_M * height),
        )
        mosaic[ty : ty + y1 - y0, tx : tx + x1 - x0] = image[y0:y1, x0:x1]
        references.append(ref)
    return mosaic, {
        "crs": "EPSG:2154",
        "extent": (west, south, east, north),
        "tiles": references,
        "acquisition_dates": sorted(
            {date for ref in references for date in ref.get("acquisition_dates", [])}
        ),
    }


def load_scene(
    store,
    cell,
    start,
    end,
    *,
    area_km=5,
    windows=None,
    progress=lambda _: None,
    cancel=None,
):
    """Load the selected saved cell, Vilpellet and the requested UTC interval.

    ``windows`` optionally keeps only the daily hour windows inside it.
    """
    area_bounds(cell, area_km)
    if not np.isfinite([start, end]).all() or end < start:
        raise ValueError("Choose a valid time interval in Thermal planes")
    if not store.has_points:
        raise ValueError("Prepare the saved 10 m intersection lattice first")

    def check_cancel():
        if cancel is not None and cancel.is_set():
            raise CancelledError("Cancelled")

    check_cancel()
    label = cell_label(store, cell)
    progress(f"Reading verified IGN terrain for {label}…")
    x, y, terrain, reference = (
        read_surface(store, cell)
        if area_km == 5
        else read_area_surface(store, cell, check_cancel, progress)
    )
    check_cancel()
    progress("Reading the saved IGN aerial photo…")
    aerial, aerial_reference, aerial_error = None, None, None
    try:
        aerial, aerial_reference = (
            read_aerial(store, cell)
            if area_km == 5
            else read_area_aerial(store, cell, check_cancel)
        )
    except (OSError, ValueError) as exc:
        aerial_error = str(exc)
    check_cancel()
    progress("Reading saved Vilpellet intersections for the selected dates…")
    plane = store.read_plane(
        cell, start, end, "vilpellet", progress=progress, cancel=cancel
    )
    check_cancel()
    frame = plane.points
    if area_km == 10:
        neighbours = read_neighbour_points(
            store, cell, start, end, check_cancel, progress
        )
        if frame is not None and not neighbours.empty:
            frame = pd.concat([frame, neighbours], ignore_index=True)
    check_cancel()
    points, flights = points_every_20m(
        frame, cell, start, end, area_km=area_km, windows=windows
    )
    return TerrainScene(
        cell,
        x,
        y,
        terrain,
        points,
        reference,
        store.activity_counts[cell.ix, cell.iy]["climb_runs"],
        flights,
        plane.selected,
        plane.unavailable,
        plane.unclassified,
        plane.unknown_clock,
        start,
        end,
        aerial,
        aerial_reference,
        aerial_error,
        label,
        area_km,
    )
