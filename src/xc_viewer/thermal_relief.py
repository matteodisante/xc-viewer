"""Offline acquisition of georeferenced hillshade; no networking in the viewer."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

from PIL import Image

from .geography import FRANCE_EXTENT

SERVICE = "https://services.arcgisonline.com/arcgis/rest/services/Elevation/World_Hillshade/MapServer"
ATTRIBUTION = "Relief: Esri World Hillshade and data contributors"
SOURCE_URL = "https://services.arcgisonline.com/arcgis/rest/services/Elevation/World_Hillshade/MapServer"


def fetch_relief(cache_dir, bounds, epsg, size, *, timeout=90):
    """Keep the service's exact returned extent alongside each cached PNG."""
    params = {
        "bbox": ",".join(map(str, bounds)),
        "bboxSR": epsg,
        "imageSR": epsg,
        "size": ",".join(map(str, size)),
        "format": "png",
        "f": "json",
    }
    digest = hashlib.sha256(
        json.dumps([SERVICE, params], sort_keys=True).encode()
    ).hexdigest()
    folder = Path(cache_dir) / "relief"
    folder.mkdir(exist_ok=True)
    png, metadata = folder / f"{digest}.png", folder / f"{digest}.json"
    if png.is_file() and metadata.is_file():
        info = json.loads(metadata.read_text(encoding="utf-8"))
        info.setdefault("cache_key", digest)
        return png.read_bytes(), info
    with urlopen(SERVICE + "/export?" + urlencode(params), timeout=timeout) as response:
        exported = json.load(response)
    if "error" in exported or "href" not in exported:
        raise RuntimeError(f"Hillshade service: {exported}")
    extent = exported["extent"]
    actual = [extent[k] for k in ("xmin", "ymin", "xmax", "ymax")]
    with urlopen(exported["href"], timeout=timeout) as response:
        payload = response.read()
    with Image.open(io.BytesIO(payload)) as image:
        image.verify()
    info = {
        "cache_key": digest,
        "extent": actual,
        "epsg": epsg,
        "attribution": ATTRIBUTION,
        "source_url": SOURCE_URL,
    }
    png.write_bytes(payload)
    metadata.write_text(json.dumps(info), encoding="utf-8")
    return payload, info


def prepare_relief(index):
    """Return the France overview plus an exact Lambert-93 image for every cell."""
    requests = [("france", FRANCE_EXTENT, 4326, (1550, 1050))]
    requests.extend(
        (f"{c.ix}/{c.iy}", c.bounds, 2154, (600, 600)) for c in index.cells()
    )
    result = []
    for key, bounds, epsg, size in requests:
        payload, info = fetch_relief(index.path.parent, bounds, epsg, size)
        result.append((key, json.dumps(info), payload))
    return result
