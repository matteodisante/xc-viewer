"""Tests for xc_viewer.widgets.map_view.

Qt widget tests: need ``qapp`` (tests/viewer/conftest.py). ``Discipline.config()`` is
monkeypatched the same way tests/viewer/test_catalog_index.py does.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from xc_viewer import catalog_index, geography
from xc_viewer.core import disciplines as disciplines_mod
from xc_viewer.core.disciplines import DataRoot
from xc_viewer.widgets.map_view import MapView, _field, _tooltip_text

# -- pure helpers: no Qt needed -----------------------------------------------


def test_field_treats_nan_and_blank_strings_as_absent():
    row = pd.Series({"pilot": "  ", "distance_km": float("nan"), "wing_class": "A"})
    assert _field(row, "pilot") is None
    assert _field(row, "distance_km") is None
    assert _field(row, "missing_column") is None
    assert _field(row, "wing_class") == "A"


def test_tooltip_text_includes_the_requested_fields():
    row = pd.Series(
        {
            "flight_id": "42",
            "date": "2020-07-15",
            "pilot": "Alice",
            "flight_type": "Dist libre",
            "wing_class": "A",
            "distance_km": 51.3,
            "duration_s": 7200.0,
            "takeoff": "SAINT HILAIRE",
            "landing": "LUMBIN",
            "dept": "38",
        }
    )
    text = _tooltip_text(row)
    assert "Flight 42" in text
    assert "2020-07-15" in text
    assert "Alice" in text
    assert "Dist libre" in text
    assert "51 km" in text
    assert "2.0 h" in text
    assert "SAINT HILAIRE" in text
    assert "LUMBIN" in text


def test_tooltip_text_replaces_an_anonymised_pilot():
    row = pd.Series({"flight_id": "1", "pilot": "$2A$07$FFVL0RGPD1SALT2345"})
    assert "(anonymised)" in _tooltip_text(row)
    assert "$2A$07$FFVL" not in _tooltip_text(row)


def test_tooltip_text_survives_every_field_missing():
    row = pd.Series({"flight_id": "1"})
    text = _tooltip_text(row)
    assert "Flight 1" in text
    assert "unknown" in text


# -- widget behaviour: needs qapp ----------------------------------------------


@pytest.fixture(autouse=True)
def _clear_caches():
    catalog_index._catalog_cache.clear()
    catalog_index._flights_meta_cache.clear()
    yield
    catalog_index._catalog_cache.clear()
    catalog_index._flights_meta_cache.clear()


def _points_frame(
    n: int, lon0=6.0, lat0=45.0, spread=0.05, alt0=1000.0
) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "flight_id": [str(i) for i in range(n)],
            "season_year": 2020,
            "date": "2020-07-15",
            "lat0": lat0 + rng.uniform(-spread, spread, n),
            "lon0": lon0 + rng.uniform(-spread, spread, n),
            "alt0": alt0,
            "pilot": "Alice",
            "flight_type": "Dist libre",
            "wing_class": "A",
            "distance_km": 40.0,
            "duration_s": 6000.0,
            "takeoff": "SITE",
            "landing": "SITE2",
            "dept": "38",
        }
    )


def test_mode_switches_between_mesh_and_points_on_point_count(qapp, monkeypatch):
    map_view = MapView()
    map_view._points = {"paragliders": _points_frame(500)}
    map_view._full_redraw()
    ax = map_view._figure.axes[0]
    ax.set_xlim(*geography.FRANCE_EXTENT[0::2])
    ax.set_ylim(*geography.FRANCE_EXTENT[1::2])
    map_view._recompute_view()
    assert map_view._mode == "mesh"
    assert map_view._mesh_artist is not None
    assert map_view._scatter_artist is None

    map_view._points = {"paragliders": _points_frame(50)}
    map_view._plotted = map_view._current_points()
    map_view._recompute_view()
    assert map_view._mode == "points"
    assert map_view._scatter_artist is not None
    assert map_view._mesh_artist is None
    assert len(map_view._visible) == 50


def test_many_mesh_points_cycles_do_not_crash(qapp):
    # Regression: matplotlib's Colorbar.remove() corrupts its host axes' gridspec
    # bookkeeping on a second create/remove cycle (confirmed against a bare
    # ax.pcolormesh()/fig.colorbar()/mesh.remove()/cb.remove() loop, independent of
    # this widget) -- the fix keeps one persistent, hidden colorbar axes
    # (self._cax) instead of creating and destroying a Colorbar every recompute.
    # This exercises many more cycles than the crash needed, on both branches.
    map_view = MapView()
    for i in range(8):
        n = 500 if i % 2 == 0 else 20
        map_view._points = {"paragliders": _points_frame(n)}
        map_view._plotted = map_view._current_points()
        map_view._recompute_view()
    assert map_view._mode == "points"


def test_zone_selection_jumps_the_view(qapp):
    map_view = MapView()
    map_view._zone_combo.setCurrentIndex(1)  # "France", per _ZONES order
    ax = map_view._figure.axes[0]
    assert ax.get_xlim() == pytest.approx(geography.FRANCE_EXTENT[0::2])
    assert ax.get_ylim() == pytest.approx(geography.FRANCE_EXTENT[1::2])


def test_click_only_selects_in_points_mode(qapp, tmp_path, monkeypatch):
    root = tmp_path / "para"
    (root / "raw" / "igc" / "2020-2021").mkdir(parents=True)
    igc_file = root / "raw" / "igc" / "2020-2021" / "2020-07-15_0.igc"
    igc_file.touch()

    def fake_config(self):
        return DataRoot(data_root=root)

    monkeypatch.setattr(disciplines_mod.Discipline, "config", fake_config)

    map_view = MapView()
    map_view._points = {"paragliders": _points_frame(5, spread=0.001)}
    map_view._full_redraw()
    ax = map_view._figure.axes[0]
    assert map_view._mode == "points"

    row = map_view._visible.iloc[0]
    px = ax.transData.transform([[row["lon0"], row["lat0"]]])[0]

    received = []
    map_view.flight_chosen.connect(lambda p, d, f: received.append((p, d, f)))

    class FakeEvent:
        pass

    event = FakeEvent()
    event.inaxes = ax
    event.x, event.y = float(px[0]), float(px[1])
    map_view._on_click(event)
    assert received and received[0][2] == row["flight_id"]

    # Force mesh mode (many points) and confirm the same click now does nothing.
    map_view._mode = "mesh"
    received.clear()
    map_view._on_click(event)
    assert received == []


def test_aspect_adapts_to_a_manual_pan_not_only_a_zone_jump(qapp):
    # Regression: only _on_zone_selected used to correct the aspect ratio, so a
    # manual pan/zoom/scroll into a region far in latitude from the world view's
    # own mean latitude (~7.5 deg) rendered visibly stretched.
    map_view = MapView()
    ax = map_view._figure.axes[0]
    world_aspect = ax.get_aspect()

    ax.set_xlim(*geography.FRANCE_EXTENT[0::2])
    ax.set_ylim(*geography.FRANCE_EXTENT[1::2])
    map_view._recompute_view()

    france_mean_lat = 0.5 * (geography.FRANCE_EXTENT[1] + geography.FRANCE_EXTENT[3])
    expected = 1.0 / np.cos(np.deg2rad(france_mean_lat))
    assert ax.get_aspect() == pytest.approx(expected)
    assert ax.get_aspect() != pytest.approx(world_aspect)


def test_invalidate_makes_ensure_loaded_reload(qapp, tmp_path, monkeypatch):
    root = tmp_path / "para"
    (root / "raw" / "igc").mkdir(parents=True)

    def fake_config(self):
        return DataRoot(data_root=root)

    monkeypatch.setattr(disciplines_mod.Discipline, "config", fake_config)

    map_view = MapView()
    calls = []

    def fake_reload():
        calls.append(1)
        map_view._loaded = True  # the real reload_points()'s own side effect

    monkeypatch.setattr(map_view, "reload_points", fake_reload)

    map_view.ensure_loaded()
    assert calls == [1]
    map_view.ensure_loaded()  # already loaded: no second reload
    assert calls == [1]

    map_view.invalidate()
    map_view.ensure_loaded()
    assert calls == [1, 1]


def test_region_filter_keeps_only_points_inside_the_chosen_box(qapp):
    map_view = MapView()
    alps = _points_frame(5, lon0=7.0, lat0=45.0, spread=0.01, alt0=1000.0)
    coast = _points_frame(3, lon0=0.0, lat0=50.0, spread=0.01, alt0=100.0)
    coast["flight_id"] = [f"coast-{i}" for i in range(3)]
    map_view._points = {"paragliders": pd.concat([alps, coast], ignore_index=True)}

    index = map_view._region_combo.findData("Alps")
    assert index >= 0
    map_view._region_combo.setCurrentIndex(index)
    points = map_view._current_points()
    assert len(points) == 5
    assert (points["region"] == "Alps").all()


def test_terrain_filter_keeps_only_the_chosen_elevation_band(qapp):
    map_view = MapView()
    low = _points_frame(4, alt0=100.0)
    high = _points_frame(2, alt0=2000.0)
    high["flight_id"] = [f"high-{i}" for i in range(2)]
    map_view._points = {"paragliders": pd.concat([low, high], ignore_index=True)}

    index = map_view._terrain_combo.findData("High mountains")
    assert index >= 0
    map_view._terrain_combo.setCurrentIndex(index)
    points = map_view._current_points()
    assert len(points) == 2
    assert (points["terrain"] == "High mountains").all()


def test_empty_view_in_points_mode_titles_no_flights(qapp):
    map_view = MapView()
    map_view._points = {"paragliders": _points_frame(5, lon0=6.0, lat0=45.0)}
    map_view._full_redraw()
    ax = map_view._figure.axes[0]
    # Zoom somewhere with no points at all -- still few enough to be "points" mode,
    # just zero of them.
    ax.set_xlim(150.0, 151.0)
    ax.set_ylim(10.0, 11.0)
    map_view._recompute_view()
    assert map_view._mode == "points"
    assert len(map_view._visible) == 0
    # loc="left", matching the repo's panel-title convention (e.g.
    # generate_prelim_figure.py) -- ax.get_title() alone reads the *centre* title,
    # which this code never sets.
    assert ax.get_title(loc="left") == "No flights in view"
