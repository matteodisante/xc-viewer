"""All-archive Vilpellet residence time, shared in definition with Thermal density.

One sparse 50 m grid covers France and its surroundings. Route selection never
filters this background. Only a bounded, conservatively aggregated raster enters
the 3-D scene; neither trajectories nor dense continental rasters are duplicated.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .fingerprint import file_identity
from .locking import exclusive
from .route_index import available_archives, route_cache_path
from .thermal_index import _check_cancel
from .thermal_time import BASE_M, TimeGrid, load_grids, save_grids
from .thermal_time_prepare import SOURCE, climb_edges

# 2.5 billion possible pixels, within save_grids' uint32 index limit; only
# occupied pixels are stored. This also covers excursions beyond French borders.
BOUNDS = (-500000, 5500000, 2000000, 8000000)
FILE_NAME = "route-thermal-duration-vilpellet.npz"
# Bump on purpose whenever the calculation changes; saved products then rebuild.
VERSION = "vilpellet-climb-residence-v1"
FIX_COLUMNS = ["flight_id", "segment_id", "t", "E", "N", "z"]
BATCH_ROWS = 1 << 20
RUN_COLUMNS = ["flight_id", "segment_id", "t_start", "t_end", "n_fixes"]


def density_path(disciplines=None):
    """Keep the all-flight thermal product beside the route index in the data folder."""
    return route_cache_path(disciplines).with_name(FILE_NAME)


def _inputs(disc):
    """The cleaned archive and the saved Vilpellet products it is binned from."""
    root = disc.config().derived_dir
    folder = root / "segmentation/vilpellet"
    return {
        "fixes": root / "fixes.parquet",
        "meta": root / "flights_meta.parquet",
        "runs": folder / "phase_segments.parquet",
        "coverage": folder / "phase_coverage.parquet",
        "parameters": folder / "model/parameters.json",
    }


def _require_current_runs(disc):
    """Refuse to bin Vilpellet runs written before the cleaned archive they label.

    Checked only when preparing: the read path must not depend on modification
    times, which a copy to another computer need not preserve.
    """
    paths = _inputs(disc)
    if (
        min(paths["runs"].stat().st_mtime_ns, paths["coverage"].stat().st_mtime_ns)
        < paths["fixes"].stat().st_mtime_ns
    ):
        raise ValueError(f"{disc.name}: Vilpellet runs predate the cleaned archive")


def source_signature(disciplines):
    """Bind saved seconds to the archives, origins, saved runs and calculation."""
    parts = [VERSION, BASE_M]
    for disc in disciplines:
        config = disc.config()
        if (config.derived_dir / ".run_incomplete").exists():
            raise ValueError(f"{disc.name}: preprocessing is incomplete")
        for path in _inputs(disc).values():
            parts.append(file_identity(path, config.data_root))
    return hashlib.sha256(json.dumps(parts).encode()).hexdigest()


def _climb_runs(path):
    """Saved climb runs sorted by flight, with each flight's slice of rows."""
    runs = pq.read_table(path, columns=RUN_COLUMNS, filters=[("phase", "=", "climb")])
    runs = runs.to_pandas()
    runs["flight_id"] = runs.flight_id.astype(str)
    runs = runs.sort_values(["flight_id", "segment_id", "t_start"], kind="stable")
    runs = runs.reset_index(drop=True)
    flights, first = np.unique(runs.flight_id.to_numpy(), return_index=True)
    stops = np.r_[first[1:], len(runs)]
    return runs, {str(f): (a, b) for f, a, b in zip(flights, first, stops, strict=True)}


def _add_flights(grid, frame, runs, spans, origins):
    """Bin the climb edges of whole flights; return their count and seconds."""
    rows = [np.arange(*spans[f]) for f in pd.unique(frame.flight_id) if f in spans]
    if not rows:
        return 0, 0.0
    a, b, dt = climb_edges(frame, runs.iloc[np.concatenate(rows)], origins)
    grid.add(a, b, dt)
    return len(dt), float(dt.sum())


def prepare_density(
    disciplines=None, *, path=None, progress=lambda _: None, cancel=None
):
    """Stream each cleaned archive once with its saved Vilpellet climb runs.

    No decoder runs: the runs are the saved whole-flight Vilpellet segmentation.
    A complete product with a matching signature is reused.
    """
    disciplines = available_archives() if disciplines is None else disciplines
    if not disciplines:
        raise FileNotFoundError("Connect the processed flight archive")
    path = density_path(disciplines) if path is None else Path(path)
    signature = source_signature(disciplines)
    for disc in disciplines:
        _require_current_runs(disc)
    path.parent.mkdir(parents=True, exist_ok=True)
    with exclusive(path, "The thermal density is already being prepared"):
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                info = json.loads(str(saved["metadata"]))
                if info.get("complete") and info.get("signature") == signature:
                    return path
        grid = TimeGrid(
            BOUNDS,
            metadata={"population": "all crossing archived flights", "source": SOURCE},
        )
        stages = {}
        for disc in disciplines:
            _check_cancel(cancel)
            paths = _inputs(disc)
            archive = pq.ParquetFile(paths["fixes"])
            covered = pq.read_table(paths["coverage"], columns=["n_native_fixes"])
            if int(covered.column(0).to_numpy().sum()) != archive.metadata.num_rows:
                raise ValueError(f"Incomplete Vilpellet coverage for {disc.name}")
            meta = pd.read_parquet(
                paths["meta"], columns=["flight_id", "lat0", "lon0", "alt0"]
            )
            origins = {
                str(r.flight_id): (r.lat0, r.lon0, r.alt0)
                for r in meta.itertuples(index=False)
            }
            runs, spans = _climb_runs(paths["runs"])
            rows = edges = 0
            seconds = 0.0
            carry = None
            for batch in archive.iter_batches(
                batch_size=BATCH_ROWS, columns=FIX_COLUMNS
            ):
                _check_cancel(cancel)
                frame = batch.to_pandas()
                frame["flight_id"] = frame.flight_id.astype(str)
                rows += len(frame)
                if carry is not None:
                    frame = pd.concat([carry, frame], ignore_index=True)
                # Flights are contiguous: hold back the last one, it may continue.
                ids = frame.flight_id.to_numpy()
                others = np.flatnonzero(ids != ids[-1])
                cut = others[-1] + 1 if len(others) else 0
                carry = frame.iloc[cut:]
                count, total = _add_flights(
                    grid, frame.iloc[:cut], runs, spans, origins
                )
                edges, seconds = edges + count, seconds + total
                progress(
                    f"Thermal hours · {disc.name}: "
                    f"{rows:,}/{archive.metadata.num_rows:,} fixes"
                )
            if carry is not None:
                count, total = _add_flights(grid, carry, runs, spans, origins)
                edges, seconds = edges + count, seconds + total
            stages[disc.name] = {"rows": rows, "edges": edges, "seconds": seconds}
        _check_cancel(cancel)
        if source_signature(disciplines) != signature:
            raise ValueError("Thermal source archive changed during preparation")
        save_grids(
            {f"archive/{SOURCE}": grid},
            path,
            {
                "complete": True,
                "signature": signature,
                "version": VERSION,
                "stages": stages,
                "created": datetime.now(UTC).isoformat(),
                "base_m": BASE_M,
                "method": "consecutive fixes inside one saved Vilpellet climb run, "
                "along continuous edges; same calculation as Thermal density "
                "regions",
            },
        )
    return path


@dataclass
class DensityRaster:
    """South-first hours/km² and its uniform metric footprint."""

    values: np.ndarray
    bounds: np.ndarray
    step: float
    hours: float
    maximum: float
    metadata: dict


@dataclass
class DensityAtlas:
    """A few bounded overview resolutions from the same all-flight time grid."""

    rasters: dict[float, DensityRaster]


def raster_window(grid, bounds, *, bins=700, pixel_m=None, metadata=None):
    """Sum fine-cell seconds into aligned overview pixels, conserving their time."""
    bounds = np.asarray(bounds, float)
    if (bounds[:2] < grid.bounds[:2]).any() or (bounds[2:] > grid.bounds[2:]).any():
        raise ValueError(
            "Route exceeds the prepared thermal coverage (France and surroundings)"
        )
    factor = 2 ** max(
        0,
        int(np.ceil(np.log2(max(max(bounds[2:] - bounds[:2]) / bins / grid.step, 1)))),
    )
    if pixel_m is not None:
        factor = max(1, round(pixel_m / grid.step))
    step = grid.step * factor
    origin = grid.bounds[:2]
    lo = np.floor((bounds[:2] - origin) / step).astype(int)
    hi = np.ceil((bounds[2:] - origin) / step).astype(int)
    raster_bounds = np.r_[origin + lo * step, origin + hi * step]
    nx, ny = hi - lo
    values = np.zeros((ny, nx))
    # Slice sorted sparse row indices before materialising coordinate arrays.
    r0 = max(0, lo[1] * factor)
    r1 = min(grid.ny, hi[1] * factor)
    first, last = np.searchsorted(grid.flat, [r0 * grid.nx, r1 * grid.nx])
    rows, cols = np.divmod(grid.flat[first:last], grid.nx)
    rows, cols = rows // factor - lo[1], cols // factor - lo[0]
    inside = (cols >= 0) & (cols < nx)
    np.add.at(values, (rows[inside], cols[inside]), grid.seconds[first:last][inside])
    hours = float(values.sum() / 3600)
    values /= 3600 * (step / 1000) ** 2
    maximum = max(
        1.0, float(grid.seconds.max(initial=0)) / 3600 / (grid.step / 1000) ** 2
    )
    return DensityRaster(values, raster_bounds, step, hours, maximum, metadata or {})


def load_density(bounds, disciplines=None, *, path=None):
    """Read the all-flight source and return only the small scene raster."""
    disciplines = available_archives() if disciplines is None else disciplines
    path = density_path(disciplines) if path is None else path
    grids, metadata = load_grids(path)
    if metadata.get("signature") != source_signature(disciplines):
        raise ValueError("Thermal archive changed; use Prepare / refresh index")
    grid = grids[f"archive/{SOURCE}"]
    span = max(np.asarray(bounds[2:]) - bounds[:2])
    steps = [step for step in (250, 500, 1000, 2000) if span / step <= 1600]
    if not steps:
        steps = [BASE_M * np.ceil(span / 1600 / BASE_M)]
    return DensityAtlas(
        {
            step: raster_window(grid, bounds, pixel_m=step, metadata=metadata)
            for step in steps
        }
    )
