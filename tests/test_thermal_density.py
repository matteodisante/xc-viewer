"""The displayed thermal measure, navigation and map requests stay consistent."""

from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pytest
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure

pytest.importorskip("PyQt6")
pytest.importorskip("pyproj")

from xc_viewer.density_maps import trim_cache, view_request
from xc_viewer.density_view import AdaptiveDensity
from xc_viewer.thermal_time import TimeGrid
from xc_viewer.widgets.density_background import DensityBackgrounds
from xc_viewer.widgets.thermal_density import ThermalDensity


def test_fractional_hours_are_visible_and_colour_scale_survives_zoom():
    grid = TimeGrid((0, 0, 150, 100))
    grid.add([[25, 25]], [[25, 25]], [0.9])  # 0.1 h/km²
    grid.flush()
    fig = Figure()
    ax = fig.subplots()
    ax.set(xlim=(0, 0.15), ylim=(0, 0.1))
    ax.set_autoscale_on(False)
    norm = LogNorm(0.01, 100)
    density = AdaptiveDensity(ax, grid.window, norm=norm)
    density.refresh()
    density.add_colorbar(fig)
    assert density.image.get_array().compressed().tolist() == pytest.approx([0.1])
    ax.set_xlim(0, 0.05)
    assert density.image.norm is norm
    assert norm.vmin == 0.01 and norm.vmax == 100
    ax.set_xlim(10, 11)
    assert len(density.image.get_array().compressed()) == 0
    density.close()


def test_background_opacity_and_layer_preserve_zoom_and_history(qapp):
    view = ThermalDensity()
    view._background.setCurrentIndex(view._background.findData("none"))
    grid = TimeGrid((800000, 6400000, 805000, 6405000))
    grid.add([[802000, 6402000]], [[802100, 6402100]], [60])
    grid.flush()
    view._grids = {"region/Alps/vilpellet": grid}
    view._draw()
    ax = view._axes_by_key["region/Alps/vilpellet"]
    view._toolbar.push_current()
    ax.set_xlim(801, 803)
    ax.set_ylim(6401, 6403)
    view._toolbar.push_current()
    view._terrain.setValue(30)
    view._strength.setValue(20)
    view._background.setCurrentIndex(0)
    assert view._axes_by_key["region/Alps/vilpellet"] is ax
    assert ax.get_xlim() == pytest.approx((801, 803))
    assert ax.get_ylim() == pytest.approx((6401, 6403))
    view._map_received(
        "region/Alps/vilpellet",
        (
            {
                "extent": (801000, 6401000, 803000, 6403000),
                "attribution": "test",
            },
            np.zeros((10, 10, 3)),
        ),
    )
    assert ax.get_xlim() == pytest.approx((801, 803))
    view._toolbar.back()
    assert ax.get_xlim() == pytest.approx((800, 805))
    view.shutdown()
    view.close()


def test_requests_refine_the_same_layer_and_limit_pixel_work():
    fig = Figure()
    ax = fig.subplots()
    ax.set(xlim=(800, 1200), ylim=(6400, 6600))
    request = view_request("topography", ax, 2)
    ax.set(xlim=(801, 803), ylim=(6401, 6403))
    zoom = view_request("topography", ax, 2)
    assert request[0] == zoom[0] == "topography"
    assert zoom[1] == (801000, 6401000, 803000, 6403000)
    assert zoom[2][0] <= 2048 and zoom[2][1] <= 2048
    assert (request[1][2] - request[1][0]) / request[2][0] > (
        zoom[1][2] - zoom[1][0]
    ) / zoom[2][0]


def test_stale_network_result_never_overwrites_new_view(qapp, tmp_path):
    loader = DensityBackgrounds(tmp_path)
    calls, received = [], []
    loader._pool.shutdown()

    def submit(*args):
        future = Future()
        calls.append(future)
        return future

    loader._pool = SimpleNamespace(submit=submit, shutdown=lambda **kw: None)
    loader.ready.connect(lambda key, result: received.append((key, result)))
    loader.request({"Alps": "old"})
    loader._poll()
    loader.request({"Alps": "new"})
    calls[0].set_result("old pixels")
    loader._poll()
    assert received == []
    calls[1].set_result("new pixels")
    loader._poll()
    assert received == [("Alps", "new pixels")]
    loader.close()


def test_disk_cache_has_a_size_limit_and_does_not_touch_siblings(tmp_path):
    folder = tmp_path / "density-maps"
    folder.mkdir()
    outside = tmp_path / "thermal-duration.npz"
    outside.write_bytes(b"data")
    for i in range(4):
        (folder / f"{i}.png").write_bytes(bytes(100))
    trim_cache(folder, limit=150)
    assert sum(p.stat().st_size for p in folder.iterdir()) <= 150
    assert outside.read_bytes() == b"data"


def test_saved_cell_maps_are_reused_offline_with_exact_zoom_extent(
    monkeypatch, tmp_path
):
    from xc_viewer.density_maps import fetch_background

    (tmp_path / "thermal-planes.sqlite3").touch()
    pixels = np.ones((100, 100, 3), dtype=np.uint8) * 128
    saved = (
        pixels,
        {
            "extent": (0, 0, 5000, 5000),
            "attribution": "IGN",
            "crs": "EPSG:2154",
        },
    )
    monkeypatch.setattr(
        "xc_viewer.density_maps.ThermalStore",
        lambda _: SimpleNamespace(background=lambda **kw: saved),
    )
    monkeypatch.setattr(
        "xc_viewer.density_maps.fetch_image",
        lambda *a, **kw: pytest.fail("unexpected network request"),
    )
    request = ("topography", (1000, 2000, 2000, 3000), (20, 20))
    info, result = fetch_background(tmp_path / "density-maps", request)
    assert info["extent"] == request[1]
    assert info["saved_planes"]
    assert result.shape == (20, 20, 3)
    assert (result == 128).all()
