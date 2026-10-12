"""Prepare approximate terrain crest lines only for the twelve saved viewer cells."""

import argparse
import hashlib
import io
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
from PIL import Image

from xc_viewer.thermal_geometry import ThermalCell
from xc_viewer.thermal_ridges import (
    ATTRIBUTION,
    DATA_DIRECTORY,
    DATASET_URL,
    LAYER,
    PARAMETERS,
    SERVICE,
    derive_ridges,
    load_ridges,
)
from xc_viewer.thermal_store import _connect, find_store_path


def main():
    """Fetch small terrain windows or rederive their lines entirely offline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DATA_DIRECTORY)
    parser.add_argument("--store", type=Path)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--rederive",
        action="store_true",
        help="Recompute lines from saved terrain, without downloads",
    )
    args = parser.parse_args()
    path = args.store or find_store_path()
    if path is None:
        raise SystemExit("Connect the prepared store with at most twelve saved cells.")
    with _connect(path) as db:
        cells = [
            ThermalCell(**json.loads(r[0]))
            for r in db.execute("SELECT payload FROM cells ORDER BY position")
        ]
    if len(cells) > 12:
        raise SystemExit("Expected at most twelve saved cells.")
    args.output.mkdir(parents=True, exist_ok=True)
    for cell in cells:
        existing = load_ridges(cell, args.output)
        if not args.refresh and not args.rederive and existing is not None:
            continue
        w, s, e, n = cell.bounds
        pad = PARAMETERS["buffer_m"]
        bounds = (w - pad, s - pad, e + pad, n + pad)
        width = int((bounds[2] - bounds[0]) / PARAMETERS["grid_m"])
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
                    "WIDTH": width,
                    "HEIGHT": width,
                    "FORMAT": "image/tiff",
                }
            )
        )
        stem = f"ign-terrain-{cell.ix}-{cell.iy}.tif"
        if args.rederive:
            if existing is None:
                raise ValueError("Offline rederivation needs the saved provenance")
            raw = (args.output / stem).read_bytes()
            old = existing["provenance"]
            if hashlib.sha256(raw).hexdigest() != old["response_sha256"]:
                raise ValueError("Saved terrain hash differs from its provenance")
            retrieved = old["retrieved_utc"]
            url = old["source_url"]
        else:
            raw = subprocess.check_output(
                [
                    "curl",
                    "--fail",
                    "--silent",
                    "--show-error",
                    "--location",
                    "--max-time",
                    "40",
                    "--retry",
                    "2",
                    url,
                ]
            )
            retrieved = datetime.now(UTC).isoformat()
        im = Image.open(io.BytesIO(raw))
        z = np.asarray(im)
        if (
            im.mode != "F"
            or z.shape != (width, width)
            or not np.isfinite(z).all()
            or z.min() < -500
            or z.max() > 9000
        ):
            raise ValueError("IGN response is not a valid float32 elevation window")
        stem = f"ign-terrain-{cell.ix}-{cell.iy}.tif"
        (args.output / stem).write_bytes(raw)
        lines = derive_ridges(z[::-1], bounds, cell.bounds)
        payload = {
            "type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": "EPSG:2154"}},
            "features": [
                {
                    "type": "Feature",
                    "properties": {"method": "transverse height maximum"},
                    "geometry": {"type": "LineString", "coordinates": p},
                }
                for p in lines
            ],
            "provenance": {
                "source_url": url,
                "dataset_url": DATASET_URL,
                "layer": LAYER,
                "attribution": ATTRIBUTION,
                "licence": "Licence Ouverte 2.0",
                "retrieved_utc": retrieved,
                "derived_utc": datetime.now(UTC).isoformat(),
                "derivation_code_sha256": hashlib.sha256(
                    Path(derive_ridges.__code__.co_filename).read_bytes()
                ).hexdigest(),
                "response_bytes": len(raw),
                "response_sha256": hashlib.sha256(raw).hexdigest(),
                "terrain_file": stem,
                "raster_bounds_epsg2154": bounds,
                "bounds_epsg2154": cell.bounds,
                "parameters": PARAMETERS,
                "terrain_height_range_m": [float(z.min()), float(z.max())],
                "method": (
                    "Gaussian-smoothed DEM; zero derivative along most concave "
                    "Hessian eigenvector; negative curvature, two-sided drop "
                    "and length filters. No flight coordinates used."
                ),
                "limitations": (
                    "Approximate, scale-dependent crests; not an official "
                    "or exhaustive ridge network. Gaps remain at weak "
                    "ridges, orientation transitions and junctions; "
                    "no invented links."
                ),
            },
        }
        path = args.output / f"ign-ridges-{cell.ix}-{cell.iy}.geojson"
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        load_ridges(cell, args.output)
        print(
            f"{cell.ix}/{cell.iy}: {len(lines)} crest pieces; {len(raw)} terrain bytes",
            flush=True,
        )


if __name__ == "__main__":
    main()
