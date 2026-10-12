"""IGN requests recover from incomplete responses without caching partial images."""

import io
from http.client import IncompleteRead
from urllib.parse import parse_qs, urlparse

from PIL import Image

from xc_viewer import thermal_imagery


def test_topography_retries_truncated_response_and_caches_valid_png(
    tmp_path, monkeypatch
):
    buffer = io.BytesIO()
    Image.new("RGBA", (32, 32), (100, 120, 140, 255)).save(buffer, format="PNG")
    payload = buffer.getvalue()
    requests = []

    class Response(io.BytesIO):
        def read(self):
            if len(requests) == 1:
                raise IncompleteRead(b"partial", 100)
            return payload

    def open_url(url, **kwargs):
        requests.append(url)
        assert not list(tmp_path.glob("*.png"))
        return Response()

    monkeypatch.setattr(thermal_imagery, "urlopen", open_url)
    monkeypatch.setattr(thermal_imagery.time, "sleep", lambda _: None)
    args = (tmp_path, (0, 0, 5000, 5000), "EPSG:2154", (32, 32), "topography")
    metadata, image = thermal_imagery.fetch_image(*args)
    assert len(requests) == 2 and image == payload
    query = parse_qs(urlparse(requests[-1]).query)
    assert query["FORMAT"] == ["image/png"]
    assert query["LAYERS"] == [
        "GEOGRAPHICALGRIDSYSTEMS.PLANIGNV2,ELEVATION.CONTOUR.LINE"
    ]
    assert query["STYLES"] == ["normal,normal"]
    assert metadata["request_url"] == requests[-1]
    cached_metadata, cached_image = thermal_imagery.fetch_image(*args)
    assert cached_image == image
    assert cached_metadata["request_url"] == metadata["request_url"]
    assert cached_metadata["retrieved"] == metadata["retrieved"]
    assert tuple(cached_metadata["extent"]) == metadata["extent"]
    assert len(requests) == 2


def test_large_topography_tiles_preserve_north_and_exact_pixel_sampling(
    tmp_path, monkeypatch
):
    requests = []

    def download(url):
        query = parse_qs(urlparse(url).query)
        bounds = tuple(map(float, query["BBOX"][0].split(",")))
        size = (int(query["WIDTH"][0]), int(query["HEIGHT"][0]))
        requests.append((url, bounds, size))
        colour = (255 if bounds[0] == 0 else 0, 255 if bounds[1] > 0 else 0, 0)
        buffer = io.BytesIO()
        Image.new("RGB", size, colour).save(buffer, format="PNG")
        return buffer.getvalue()

    monkeypatch.setattr(thermal_imagery, "_download", download)
    args = (tmp_path, (0, 0, 5000, 5000), "EPSG:2154", (2048, 2048), "topography")
    info, payload = thermal_imagery.fetch_image(*args)
    assert [r[1] for r in requests] == [
        (0, 2500, 2500, 5000),
        (2500, 2500, 5000, 5000),
        (0, 0, 2500, 2500),
        (2500, 0, 5000, 2500),
    ]
    assert all(r[2] == (1024, 1024) for r in requests)
    assert info["request_urls"] == [r[0] for r in requests]
    with Image.open(io.BytesIO(payload)) as image:
        assert image.size == (2048, 2048)
        assert image.getpixel((0, 0)) == (255, 255, 0)
        assert image.getpixel((2047, 0)) == (0, 255, 0)
        assert image.getpixel((0, 2047)) == (255, 0, 0)
        assert image.getpixel((2047, 2047)) == (0, 0, 0)
    cached, _ = thermal_imagery.fetch_image(*args)
    assert cached["request_urls"] == info["request_urls"]
    assert cached["request_url"] == info["request_url"]
    assert len(requests) == 4
