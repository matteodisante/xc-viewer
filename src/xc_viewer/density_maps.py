"""Screen-resolution backgrounds shared with Thermal planes, in a bounded cache."""

from __future__ import annotations

import hashlib
import io
import threading
from contextlib import suppress
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen

import numpy as np
from PIL import Image

from .thermal_imagery import fetch_image
from .thermal_relief import fetch_relief
from .thermal_store import ThermalStore

CACHE_BYTES = 512 * 1024**2
_CACHE_LOCK = threading.Lock()


def trim_cache(folder: Path, limit=CACHE_BYTES):
    """Evict least-recently-used response files, only inside this dedicated cache."""
    files = []
    for path in folder.rglob("*"):
        if path.is_file():
            stat = path.stat()
            files.append((stat.st_mtime_ns, path, stat.st_size))
    total = sum(size for _, _, size in files)
    for _, path, size in sorted(files):
        if total <= limit:
            break
        with suppress(FileNotFoundError):
            path.unlink()
        total -= size


def _download(url):
    # Interaction must not wait through the offline preparer's long retry policy.
    with urlopen(url, timeout=15) as response:
        return response.read()


def _saved_background(folder, request):
    """Reuse the planes' existing high-resolution image when it covers this view."""
    kind, bounds, size = request
    w, s, e, n = bounds
    if max(e - w, n - s) > 5000:
        return None
    path = folder.parent / "thermal-planes.sqlite3"
    if not path.is_file():
        return None
    ix, iy = int((w + e) / 10000), int((s + n) / 10000)
    saved = ThermalStore(path).background(kind=kind, key=f"{ix}/{iy}")
    if saved is None:
        return None
    pixels, info = saved
    left, bottom, right, top = info["extent"]
    if not (left <= w < e <= right and bottom <= s < n <= top):
        return None
    height, width = pixels.shape[:2]
    dx, dy = (right - left) / width, (top - bottom) / height
    if dx > (e - w) / size[0] * 1.01 or dy > (n - s) / size[1] * 1.01:
        return None
    crop = ((w - left) / dx, (top - n) / dy, (e - left) / dx, (top - s) / dy)
    image = Image.fromarray(pixels).transform(
        size, Image.Transform.EXTENT, crop, resample=Image.Resampling.BICUBIC
    )
    return {**info, "extent": bounds, "size": size, "saved_planes": True}, np.asarray(
        image
    )


def fetch_background(folder, request):
    """Read cached pixels or request current detail; retain exact map georeferencing.

    The caller limits concurrency. Serial cache access prevents eviction races with
    the split requests used for large IGN contour maps.
    """
    kind, bounds, size = request
    folder = Path(folder)
    saved = _saved_background(folder, request)
    if saved is not None:
        return saved
    with _CACHE_LOCK:
        folder.mkdir(parents=True, exist_ok=True)
        try:
            if kind == "relief":
                payload, info = fetch_relief(folder, bounds, 2154, size, timeout=15)
            else:
                info, payload = fetch_image(
                    folder, bounds, "EPSG:2154", size, kind, downloader=_download
                )
            # Touch all pieces of this request, including contour mosaic children.
            # Other entries retain their age for eviction.
            urls = info.get("request_urls", [info.get("request_url", "")])
            cache_key = info.get("cache_key")
            if cache_key:
                for path in folder.rglob(cache_key + ".*"):
                    path.touch()
            for url in urls:
                if url:
                    digest = hashlib.sha256(urlsplit(url).query.encode()).hexdigest()
                    for path in folder.glob(digest + ".*"):
                        path.touch()
            with Image.open(io.BytesIO(payload)) as image:
                pixels = np.asarray(image.convert("RGB")).copy()
            return info, pixels
        finally:
            trim_cache(folder)


def view_request(kind, ax, device_ratio=1):
    """Match ground resolution to the viewport, down to the planes' 1.25 m/pixel."""
    x0, x1 = sorted(ax.get_xlim())
    y0, y1 = sorted(ax.get_ylim())
    bounds = tuple(round(v * 1000, 1) for v in (x0, y0, x1, y1))
    width = min(2048, max(256, round(ax.bbox.width * device_ratio)))
    height = min(2048, max(256, round(ax.bbox.height * device_ratio)))
    width = max(2, min(width, int((bounds[2] - bounds[0]) / 1.25)))
    height = max(2, min(height, int((bounds[3] - bounds[1]) / 1.25)))
    return kind, bounds, (width, height)
