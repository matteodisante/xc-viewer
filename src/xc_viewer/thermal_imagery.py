"""Offline IGN colour maps and aerial photographs, embedded in the saved snapshot."""

import hashlib
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from http.client import HTTPException
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

from PIL import Image

from .geography import FRANCE_EXTENT
from .sqlite import connect
from .thermal_store import ThermalStore, neighbour_frames

SERVICE = "https://data.geopf.fr/wms-r/wms"
LAYERS = {
    "aerial": "ORTHOIMAGERY.ORTHOPHOTOS",
    "colour": "GEOGRAPHICALGRIDSYSTEMS.PLANIGNV2",
    "topography": "GEOGRAPHICALGRIDSYSTEMS.PLANIGNV2,ELEVATION.CONTOUR.LINE",
}


def reuse_backgrounds(path, previous, *, with_neighbours=False):
    """Copy unchanged grid backgrounds from an earlier selection, with provenance."""
    previous = Path(previous)
    if not previous.exists() or Path(path).resolve() == previous.resolve():
        return
    cells = ThermalStore(path).cells()
    keys = {"france", *(f"{c.ix}/{c.iy}" for c in cells)}
    if with_neighbours:
        keys.update(f"{f.ix}/{f.iy}" for c in cells for f in neighbour_frames(c))
    with connect(path, uri=True) as db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS backgrounds(kind TEXT,key TEXT,"
            "metadata TEXT,image BLOB,PRIMARY KEY(kind,key))"
        )
        db.execute(
            "ATTACH DATABASE ? AS previous", (previous.resolve().as_uri() + "?mode=ro",)
        )
        tables = {
            r[0]
            for r in db.execute(
                "SELECT name FROM previous.sqlite_master WHERE type='table'"
            )
        }
        placeholders = ",".join("?" for _ in keys)
        for table in ("backgrounds", "relief"):
            if table in tables:
                db.execute(
                    f"INSERT OR IGNORE INTO main.{table} SELECT * "
                    f"FROM previous.{table} WHERE key IN ({placeholders})",
                    tuple(keys),
                )
        db.commit()


def _download(url):
    """Retry transient service errors; never publish a partial response."""
    for attempt in range(4):
        try:
            with urlopen(url, timeout=120) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError, HTTPException, ConnectionError):
            if attempt == 3:
                raise
            time.sleep(2**attempt)


def fetch_image(folder, bounds, crs, size, kind, *, downloader=None):
    """Request exactly one georeferenced image; retain attribution and provenance."""
    params = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetMap",
        "LAYERS": LAYERS[kind],
        "STYLES": ",".join("normal" for _ in LAYERS[kind].split(",")),
        "CRS": crs,
        "BBOX": ",".join(map(str, bounds)),
        "WIDTH": size[0],
        "HEIGHT": size[1],
        "FORMAT": "image/png" if kind == "topography" else "image/jpeg",
    }
    query = urlencode(params)
    digest = hashlib.sha256(query.encode()).hexdigest()
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{digest}.{'png' if kind == 'topography' else 'jpg'}"
    meta = folder / f"{digest}.json"
    if target.exists() and meta.exists():
        info = json.loads(meta.read_text(encoding="utf-8"))
        info.setdefault("request_url", SERVICE + "?" + query)
        info.setdefault("cache_key", digest)
        return info, target.read_bytes()
    requests = []
    if kind == "topography" and max(size) > 1024:
        # Large combined-layer PNG responses can be truncated by the service.
        # Request smaller windows at exactly the same ground sampling and stitch
        # without resampling. Retain every actual query as provenance.
        w, s, e, n = bounds
        width, height = size
        mosaic = Image.new("RGB", size)
        xs, ys = (0, width // 2, width), (0, height // 2, height)
        for row in range(2):
            for col in range(2):
                x0, x1, y0, y1 = xs[col], xs[col + 1], ys[row], ys[row + 1]
                tile_bounds = (
                    w + (e - w) * x0 / width,
                    n - (n - s) * y1 / height,
                    w + (e - w) * x1 / width,
                    n - (n - s) * y0 / height,
                )
                tile_info, tile = fetch_image(
                    folder,
                    tile_bounds,
                    crs,
                    (x1 - x0, y1 - y0),
                    kind,
                    downloader=downloader,
                )
                requests.extend(
                    tile_info.get("request_urls", [tile_info["request_url"]])
                )
                with Image.open(io.BytesIO(tile)) as image:
                    mosaic.paste(image.convert("RGB"), (x0, y0))
        output = io.BytesIO()
        mosaic.save(output, format="PNG")
        payload = output.getvalue()
    else:
        payload = (downloader or _download)(SERVICE + "?" + query)
    with Image.open(io.BytesIO(payload)) as img:
        img.verify()
        if img.size != tuple(size):
            raise ValueError(
                "IGN image dimensions differ from the requested resolution"
            )
    info = {
        "cache_key": digest,
        "extent": bounds,
        "crs": crs,
        "size": size,
        "attribution": "© IGN / Géoplateforme · "
        + {
            "aerial": "BD ORTHO",
            "colour": "Plan IGN",
            "topography": "Plan IGN + official elevation contours",
        }[kind],
        "source_url": SERVICE,
        "request_url": SERVICE + "?" + query,
        "layer": LAYERS[kind],
        "retrieved": datetime.now(UTC).isoformat(),
        "note": "Mosaic acquisition dates vary spatially and differ from flight dates.",
    }
    if requests:
        info.update(
            request_url=requests[0],
            request_urls=requests,
            assembly="Georeferenced tiles joined without resampling",
        )
    target.write_bytes(payload)
    meta.write_text(json.dumps(info), encoding="utf-8")
    return info, payload


def acquisition_dates(bounds, crs):
    """Read acquisition dates from the official mosaic graph for this extent."""
    params = {
        "SERVICE": "WFS",
        "VERSION": "2.0.0",
        "REQUEST": "GetFeature",
        "TYPENAMES": "ORTHOIMAGERY.ORTHOPHOTOS.GRAPHE-MOSAIQUAGE:graphe_bdortho",
        "SRSNAME": crs,
        "BBOX": ",".join(map(str, bounds)) + "," + crs,
        "OUTPUTFORMAT": "application/json",
        "COUNT": 10000,
        "PROPERTYNAME": "date_vol",
    }
    url = "https://data.geopf.fr/wfs/ows?" + urlencode(params)
    result = json.loads(_download(url))
    if "features" not in result:
        raise ValueError(f"IGN mosaic dates unavailable: {result}")
    if int(result.get("numberMatched", 0)) > len(result["features"]):
        raise ValueError("IGN date query truncated; cannot publish incomplete dates")
    return sorted(
        {
            f["properties"]["date_vol"][:10]
            for f in result["features"]
            if f["properties"].get("date_vol")
        }
    ), url


def prepare_imagery(path, progress=print):
    """Save 1.25 m/px aerial imagery for each square plus colour maps of France."""
    store = ThermalStore(path)
    requests = [("france", FRANCE_EXTENT, "CRS:84", (1550, 1050))]
    requests += [
        (f"{c.ix}/{c.iy}", c.bounds, "EPSG:2154", (4000, 4000)) for c in store.cells()
    ]

    def download(key, bounds, crs, size, kind):
        """Fetch independent images concurrently; the caller alone writes SQLite."""
        info, payload = fetch_image(
            store.path.parent / "imagery", bounds, crs, size, kind
        )
        if kind == "aerial":
            if key == "france":
                info["acquisition_note"] = "France is a multi-date mosaic."
            else:
                dates, provenance = acquisition_dates(bounds, crs)
                info["acquisition_dates"] = dates
                info["acquisition_source"] = provenance
        return kind, key, json.dumps(info), payload

    with (
        connect(path, timeout=120) as db,
        ThreadPoolExecutor(max_workers=3) as pool,
    ):
        db.execute("""CREATE TABLE IF NOT EXISTS backgrounds(
            kind TEXT,key TEXT,metadata TEXT,image BLOB,PRIMARY KEY(kind,key))""")
        pending = []
        for key, bounds, crs, size in requests:
            for kind in LAYERS:
                dimensions = size
                saved = db.execute(
                    "SELECT metadata FROM backgrounds WHERE kind=? AND key=?",
                    (kind, key),
                ).fetchone()
                if saved and tuple(json.loads(saved[0]).get("size", ())) == dimensions:
                    continue
                progress(f"Downloading IGN {kind}: {key}")
                pending.append(
                    pool.submit(download, key, bounds, crs, dimensions, kind)
                )
        for future in as_completed(pending):
            row = future.result()
            db.execute("INSERT OR REPLACE INTO backgrounds VALUES (?,?,?,?)", row)
            db.commit()
            progress(f"Saved IGN {row[0]}: {row[1]}")


def prepare_region_backgrounds(path, progress=print):
    """Save a background of each regional box, all three kinds, into the store."""
    from .thermal_regions import REGIONS, extent
    from .thermal_relief import fetch_relief

    with connect(path, timeout=120) as db:
        db.execute("""CREATE TABLE IF NOT EXISTS backgrounds(
            kind TEXT,key TEXT,metadata TEXT,image BLOB,PRIMARY KEY(kind,key))""")
        db.execute("""CREATE TABLE IF NOT EXISTS relief(
            key TEXT PRIMARY KEY,metadata TEXT,png BLOB)""")
        for name, box in REGIONS.items():
            lon_min, lat_min, lon_max, lat_max = extent(box)
            key = f"region/{name}"
            width = 4000
            size = (width, round(width * (lat_max - lat_min) / (lon_max - lon_min)))
            for kind in LAYERS:
                if db.execute(
                    "SELECT 1 FROM backgrounds WHERE kind=? AND key=?", (kind, key)
                ).fetchone():
                    continue
                progress(f"Downloading IGN {kind}: {name}")
                info, payload = fetch_image(
                    Path(path).parent / "imagery",
                    (lon_min, lat_min, lon_max, lat_max),
                    "CRS:84",
                    size,
                    kind,
                )
                if kind == "aerial":
                    info["acquisition_note"] = (
                        "Multi-date mosaic; acquisition dates vary by location "
                        "and differ from flight dates."
                    )
                db.execute(
                    "INSERT INTO backgrounds VALUES (?,?,?,?)",
                    (kind, key, json.dumps(info), payload),
                )
                db.commit()
            if not db.execute("SELECT 1 FROM relief WHERE key=?", (key,)).fetchone():
                progress(f"Downloading hillshade: {name}")
                payload, info = fetch_relief(
                    Path(path).parent, (lon_min, lat_min, lon_max, lat_max), 4326, size
                )
                db.execute(
                    "INSERT INTO relief VALUES (?,?,?)",
                    (key, json.dumps(info), payload),
                )
                db.commit()
