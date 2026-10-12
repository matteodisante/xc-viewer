"""Download only named summits in the viewer's twelve saved 5 km cells."""

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlencode

from xc_viewer.thermal_orography import (
    ATTRIBUTION,
    DATA_DIRECTORY,
    DATASET_URL,
    LAYER,
    SERVICE,
    load_summits,
    select_cell_summits,
    validate_summits,
)
from xc_viewer.thermal_store import load_store


def main():
    """Keep complete, attributed local extracts; never modify the flight store."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DATA_DIRECTORY)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    store = load_store()
    if store is None:
        raise SystemExit("Choose the data folder with the prepared thermal planes.")
    args.output.mkdir(parents=True, exist_ok=True)
    cells = store.cells()
    # This command prepares only the existing viewer selection, never France.
    if len(cells) > 12:
        raise SystemExit("Expected at most twelve viewer cells; review the selection.")
    for cell in cells:
        path = args.output / f"ign-summits-{cell.ix}-{cell.iy}.geojson"
        if not args.refresh and load_summits(cell, args.output) is not None:
            print(f"Already saved: {cell.ix}/{cell.iy}")
            continue
        bounds = ",".join(f"{v:g}" for v in cell.bounds)
        params = {
            "SERVICE": "WFS",
            "VERSION": "2.0.0",
            "REQUEST": "GetFeature",
            "TYPENAMES": LAYER,
            "SRSNAME": "EPSG:2154",
            "CQL_FILTER": f"BBOX(geometrie,{bounds},'EPSG:2154') "
            "AND nature IN ('Sommet','Pic')",
            "OUTPUTFORMAT": "application/json",
            "COUNT": 1000,
        }
        url = SERVICE + "?" + urlencode(params)
        # Use the system certificate store; TLS verification remains enabled.
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
                "--retry-delay",
                "1",
                url,
            ]
        )
        response = json.loads(raw)
        payload = select_cell_summits(response, cell.bounds)
        validate_summits(payload, cell.bounds)
        payload["provenance"] = {
            "source_url": url,
            "dataset_url": DATASET_URL,
            "layer": LAYER,
            "attribution": ATTRIBUTION,
            "licence": "Licence Ouverte 2.0",
            "retrieved_utc": datetime.now(UTC).isoformat(),
            "response_sha256": hashlib.sha256(raw).hexdigest(),
            "response_bytes": len(raw),
            "response_feature_count": len(response["features"]),
            "bounds_epsg2154": cell.bounds,
            "selection": "Sommet/Pic only; exact Lambert-93 cell bounds.",
            "note": "Named summits; not a complete ridge network or DEM maxima.",
        }
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
        print(
            f"{cell.ix}/{cell.iy}: {len(payload['features'])} summits, {len(raw)} bytes"
        )


if __name__ == "__main__":
    main()
