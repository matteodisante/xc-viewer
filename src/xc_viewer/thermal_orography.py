"""Small, offline IGN summit extracts shared by the viewer and figure exports."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

DATA_DIRECTORY = Path(__file__).resolve().parent / "assets" / "thermal_orography"
SERVICE = "https://data.geopf.fr/wfs/ows"
DATASET_URL = "https://www.data.gouv.fr/datasets/bd-topo-r"
LAYER = "BDTOPO_V3:detail_orographique"
ATTRIBUTION = "Summits: © IGN · BD TOPO · Licence Ouverte 2.0"
SUMMIT_COLOR = "#D20A32"
SUMMIT_NATURES = frozenset(("Sommet", "Pic"))


def validate_summits(payload, bounds=None):
    """Reject truncated, misprojected or unrelated features before plotting."""
    if payload.get("type") != "FeatureCollection":
        raise ValueError("IGN response is not a FeatureCollection")
    features = payload["features"]
    if int(payload["numberMatched"]) != len(features):
        raise ValueError("IGN summit response is incomplete")
    crs = (payload.get("crs") or {}).get("properties", {}).get("name", "")
    # IGN omits the CRS on empty responses, which contain no coordinates to project.
    if features and crs not in ("EPSG:2154", "urn:ogc:def:crs:EPSG::2154"):
        raise ValueError("IGN summits must use Lambert-93 (EPSG:2154)")
    for feature in features:
        geometry = feature["geometry"]
        if geometry["type"] != "Point":
            raise ValueError("A named summit must have point geometry")
        x, y = geometry["coordinates"][:2]
        if not np.isfinite(x + y):
            raise ValueError("IGN summit coordinates are not finite")
        if bounds is not None:
            west, south, east, north = bounds
            if not (west <= x <= east and south <= y <= north):
                raise ValueError(f"IGN summit {(x, y)} lies outside cell {bounds}")
        if feature["properties"]["nature"] not in SUMMIT_NATURES:
            raise ValueError("Named ridges, slopes and passes are not summit points")


def select_cell_summits(payload, bounds):
    """Clip WFS bounding-envelope candidates to the exact Lambert-93 square."""
    validate_summits(payload)
    west, south, east, north = bounds
    features = []
    for feature in payload["features"]:
        x, y = feature["geometry"]["coordinates"][:2]
        if west <= x <= east and south <= y <= north:
            features.append(feature)
    return {
        **payload,
        "features": features,
        "numberMatched": len(features),
        "numberReturned": len(features),
        "totalFeatures": len(features),
    }


@lru_cache(maxsize=24)
def _read_extract(path, modified_ns):
    """Reuse parsed local data across height changes, refreshing after edits."""
    return json.loads(path.read_text(encoding="utf-8"))


def load_summits(cell, folder=None):
    """Return an exact saved cell extract; None means not prepared, not empty."""
    folder = DATA_DIRECTORY if folder is None else Path(folder)
    path = folder / f"ign-summits-{cell.ix}-{cell.iy}.geojson"
    if not path.exists():
        return None
    payload = _read_extract(path, path.stat().st_mtime_ns)
    if tuple(payload["provenance"]["bounds_epsg2154"]) != tuple(cell.bounds):
        raise ValueError("Saved IGN summit extent does not match this cell")
    validate_summits(payload, cell.bounds)
    return payload


def draw_summits(ax, cell, payload, *, size=45, labels=False):
    """Draw red triangles at fixed ground locations, independently of plane height."""
    features = payload["features"]
    if not features:
        return
    west, south, _, _ = cell.bounds
    positions = np.asarray([f["geometry"]["coordinates"][:2] for f in features])
    positions = (positions - (west, south)) / 1000
    ax.scatter(
        positions[:, 0],
        positions[:, 1],
        s=size,
        marker="^",
        color=SUMMIT_COLOR,
        edgecolors="white",
        linewidths=0.65,
        zorder=5,
        label="IGN named summits",
    )
    if labels:
        for (x, y), feature in zip(positions, features, strict=True):
            name = feature["properties"].get("toponyme") or "Summit"
            # Keep the label within the map near the eastern edge.
            right = x > 3.8
            high = y > 4.7
            ax.annotate(
                name,
                (x, y),
                xytext=(-5 if right else 5, -6 if high else 9 if y < 0.15 else 4),
                textcoords="offset points",
                ha="right" if right else "left",
                va="top" if high else "bottom",
                fontsize=6.5,
                color=SUMMIT_COLOR,
                zorder=6,
                clip_on=True,
                bbox={
                    "facecolor": "white",
                    "alpha": 0.85,
                    "edgecolor": "none",
                    "pad": 1,
                },
            )
