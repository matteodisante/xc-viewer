"""Interactive height and time controls must refer to the same loaded slice."""

from datetime import UTC, datetime
from types import SimpleNamespace

import pandas as pd
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyproj")

from xc_viewer.thermal_geometry import ThermalCell, project
from xc_viewer.thermal_store import PlaneData
from xc_viewer.widgets.thermal_plane import ThermalPlane


@pytest.fixture
def widget(qapp, monkeypatch):
    x, y = project(6, 45)
    cell = ThermalCell(int(x // 5000), int(y // 5000), "Hills", 5, 2, 500, 1500)
    day = datetime(2024, 6, 15, 12, tzinfo=UTC).timestamp()
    index = SimpleNamespace(
        disciplines=("paragliders",),
        cells=lambda: [cell],
        defaults=lambda _: (day - 43200, 500),
    )
    view = ThermalPlane()
    with monkeypatch.context() as m:
        m.setattr(view, "_start_plane", lambda: None)
        view._index_ready(index)
    view._set_busy(False)
    west, south, _, _ = cell.bounds
    edges = pd.DataFrame(
        {
            "x0": [west + 1000],
            "x1": [west + 2000],
            "y0": [south + 1000],
            "y1": [south + 2000],
            "z0": [600],
            "z1": [1400],
            "utc0": [day + 60],
            "utc1": [day + 100],
            "flight_id": ["a"],
            "discipline": ["paragliders"],
        }
    )
    view._plane_ready(PlaneData(edges, 1, 1, 0, 0, 0))
    view._set_busy(False)
    yield view
    view.shutdown()
    view.close()


def test_height_slider_updates_points_without_reloading(widget):
    assert len(widget._plane_ax.collections) == 1
    widget._slider.setValue(0)
    assert widget._height.value() == 0
    assert len(widget._plane_ax.collections) == 0
    widget._slider.setValue(50)
    assert widget._height.value() == 500
    offsets = widget._plane_ax.collections[0].get_offsets()
    assert offsets[0].tolist() == pytest.approx([1.5, 1.5])
    widget._slider.setValue(widget._slider.maximum())
    assert widget._height.value() == 1000
    assert widget._plane_ax.get_xlim() == (0, 5)
    assert widget._plane_ax.get_ylim() == (0, 5)


def test_exact_terminal_height_survives_spinbox_rounding(widget, monkeypatch):
    from dataclasses import replace

    cell = replace(widget._cells.currentData(), max_alt_m=1500.006)
    plane = widget._plane
    plane.edges["z1"] = cell.max_alt_m
    index = SimpleNamespace(
        disciplines=("paragliders",),
        cells=lambda: [cell],
        defaults=lambda _: (widget._utc_bounds()[0], 500),
    )
    monkeypatch.setattr(widget, "_start_plane", lambda: None)
    widget._index_ready(index)
    widget._plane_ready(plane)
    widget._slider.setValue(widget._slider.maximum())
    assert widget._height.value() == 1000.01
    assert widget._plane_ax.collections[0].get_offsets()[0].tolist() == [2, 2]


def test_neighbour_zoom_uses_one_absolute_plane_and_limits_counts_to_view(widget):
    cell = widget._cells.currentData()
    west, south, _, _ = cell.bounds
    day = widget._utc_bounds()[0] + 43200
    reads = []

    def neighbour_points(c, source, level):
        reads.append(level)
        # Saved on the selected cell's lattice: level 50 is H = 500 + 500 m.
        if level != 50:
            return pd.DataFrame({"x": [], "y": [], "utc": [], "flight": []})
        return pd.DataFrame(
            {
                "x": [west + 6500, west + 6600],
                "y": [south + 1500, south + 1600],
                "utc": [day + 80, day + 10 * 86400],
                "flight": [0, 0],
            }
        )

    widget._index = SimpleNamespace(neighbour_points=neighbour_points)
    widget._plane_limits = ((-2.5, 7.5), (-2.5, 7.5))
    widget._neighborhood_ready((None, [("paragliders", "neighbour-only")], {}))
    # The second crossing lies outside the selected interval.
    offsets = widget._plane_ax.collections[0].get_offsets()
    assert offsets.tolist() == [[1.5, 1.5], [6.5, 1.5]]
    widget._height.setValue(600)
    widget._height.setValue(500)
    assert reads == [50, 60]
    before = (widget._height.value(), widget._utc_bounds())
    widget._zoom_plane(2)
    assert widget._plane_ax.get_xlim() == (-2.5, 7.5)
    widget._reset_plane_view()
    assert widget._plane_ax.get_xlim() == (0, 5)
    assert widget._plane_ax.collections[0].get_offsets().tolist() == [[1.5, 1.5]]
    assert before == (widget._height.value(), widget._utc_bounds())
    for _ in range(8):
        widget._zoom_plane(0.5)
    assert widget._plane_ax.get_xlim()[1] - widget._plane_ax.get_xlim()[0] == 0.5


def test_pan_and_zoom_are_preserved_by_height_and_background_redraw(widget):
    widget._plane_ax.set_xlim(1, 3)
    widget._plane_ax.set_ylim(1, 3)
    widget._pan_finished(SimpleNamespace(inaxes=widget._plane_ax))
    widget._height.setValue(600)
    widget._relief_strength.setValue(100)
    assert widget._plane_ax.get_xlim() == (1, 3)
    assert widget._plane_ax.get_ylim() == (1, 3)


def test_default_background_is_topography(widget):
    assert widget._background.currentData() == "topography"
    assert widget._relief_strength.value() == 85


def test_vilpellet_climb_counts_are_separate_from_visitors(widget, monkeypatch):
    cell = widget._cells.currentData()
    index = SimpleNamespace(
        disciplines=("paragliders",),
        cells=lambda: [cell],
        defaults=lambda _: (widget._utc_bounds()[0], 500),
        has_terrain_ranking=True,
        has_climb_ranking=True,
        activity_counts={(cell.ix, cell.iy): {"climb_runs": 40, "climb_flights": 3}},
    )
    monkeypatch.setattr(widget, "_start_plane", lambda: None)
    widget._index_ready(index)
    assert "40 Vilpellet climbs" in widget._cells.currentText()
    assert "40 Vilpellet climb runs from 3 flights" in widget._summary.text()
    assert "5 distinct crossing flights" in widget._summary.text()
    assert "40" in widget._map_labels[0][0].get_text()
    # paper_style() elsewhere in the run moves untitled-loc titles to the left.
    titles = [widget._map_ax.get_title(loc=loc) for loc in ("left", "center")]
    assert any("Vilpellet climb" in title for title in titles)


def test_dem_ranked_cell_without_internal_starts_can_be_selected(widget, monkeypatch):
    from dataclasses import replace

    cell = replace(widget._cells.currentData(), launches=0, launch_median_m=None)
    index = SimpleNamespace(
        disciplines=("paragliders",),
        cells=lambda: [cell],
        defaults=lambda _: (widget._utc_bounds()[0], 500),
        reference_audit=lambda _: (0, None),
        terrain_reference=lambda _: {
            "minimum_m": cell.ground_m,
            "maximum_m": 1234.5,
            "grid_m": [25, 25],
            "samples": 40000,
            "retrieved_utc": "2026-09-22",
            "source_url": "https://data.geopf.fr/",
        },
        has_terrain_ranking=True,
    )
    monkeypatch.setattr(widget, "_start_plane", lambda: None)
    widget._index_ready(index)
    assert "Category bands use highest terrain" in widget._summary.text()
    assert "Highest terrain (category): 1234.50 m ASL" in widget._summary.text()
    assert "no usable launch altitude" in widget._summary.text()
    assert "Launch median" not in widget._summary.text()
    assert widget._height.maximum() == cell.max_agl_m


def test_missing_contours_use_saved_colour_map_with_explicit_attribution(widget):
    import numpy as np

    cell = widget._cells.currentData()
    pixels = np.full((4, 4, 3), 180, dtype=np.uint8)
    metadata = {"extent": cell.bounds, "attribution": "Plan IGN"}
    widget._index = SimpleNamespace(
        background=lambda c, kind: (pixels, metadata) if kind == "colour" else None
    )
    widget._reliefs.clear()
    image, info = widget._saved_relief(cell)
    assert image is pixels
    assert info["extent"] == cell.bounds
    assert info["attribution"] == "Plan IGN · Elevation contours unavailable"
    assert metadata["attribution"] == "Plan IGN"


def test_height_does_not_change_the_loaded_climbs(widget):
    widget._height.setValue(500)
    loaded = widget._plane
    edges = loaded.edges.copy()
    widget._draw_plane()
    widget._height.setValue(600)
    assert widget._plane is loaded
    pd.testing.assert_frame_equal(widget._plane.edges, edges)


def test_only_vilpellet_climbs_are_offered(widget):
    assert not hasattr(widget, "_source")
    assert "Vilpellet" in widget._provenance.text()
    assert "HMM" not in widget._provenance.text()


def test_datetime_controls_use_real_utc_and_invalidate_cached_slice(widget):
    start, end = widget._utc_bounds()
    assert start == datetime(2024, 6, 15, tzinfo=UTC).timestamp()
    assert end == start + 86399
    widget._start.setDateTime(widget._start.dateTime().addSecs(3600))
    assert widget._utc_bounds()[0] == start + 3600
    assert widget._plane is None


def test_reversed_interval_has_no_background_request(widget):
    widget._start.setDateTime(widget._end.dateTime().addSecs(1))
    widget._start_plane()
    assert widget._worker is None
    assert "end time" in widget._status.text()


@pytest.mark.parametrize(
    "first,last,hours",
    [
        ("2024-06-15", "2024-06-15", 24),
        ("2024-03-30", "2024-03-31", 47),
        ("2024-10-26", "2024-10-27", 49),
        ("2024-12-31", "2025-01-01", 48),
    ],
)
def test_daily_date_range_uses_whole_paris_days_and_loads_on_request(
    widget, monkeypatch, first, last, hours
):
    from PyQt6.QtCore import QDate

    from xc_viewer.thermal_daily import local_bounds

    original = widget._utc_bounds()
    operations, reads = [], []
    monkeypatch.setattr(widget, "_run", lambda op, _: operations.append(op))
    widget._index.read_plane = lambda *args, **kw: reads.append(args)
    widget._mode.setCurrentIndex(1)
    operations.clear()
    widget._daily_start.setDate(QDate.fromString(first, "yyyy-MM-dd"))
    widget._daily_end.setDate(QDate.fromString(last, "yyyy-MM-dd"))
    assert operations == []
    assert widget._plane is None
    start, end = widget._read_bounds()
    assert start == local_bounds(first)[0]
    assert end == local_bounds(last)[1] - 1e-6
    assert end - start == pytest.approx(hours * 3600)
    assert widget._utc_bounds() == original
    assert f"{first} → {last}" in widget._figure._suptitle.get_text()
    widget._load.click()
    assert len(operations) == 1
    operations[0]()
    assert reads == [(widget._cells.currentData(), start, end, "vilpellet")]


def test_reversed_daily_dates_reject_plane_and_3d_loads(widget, monkeypatch):
    from PyQt6.QtCore import QDate

    with monkeypatch.context() as m:
        m.setattr(widget, "_start_plane", lambda: None)
        widget._mode.setCurrentIndex(1)
    widget._daily_start.setDate(QDate(2024, 6, 20))
    widget._daily_end.setDate(QDate(2024, 6, 19))
    assert "end date" in widget._status.text()
    assert "Invalid date range" in widget._figure._suptitle.get_text()
    monkeypatch.setattr(widget, "_run", lambda *_: pytest.fail("invalid dates loaded"))
    widget._start_plane()
    assert widget._worker is None
    widget._show_terrain_3d()
    assert widget._terrain_3d_panel is None
    assert "end date" in widget._status.text()


def test_daily_range_survives_cell_switch_reload_and_relative_shortcuts(
    widget, monkeypatch
):
    from dataclasses import replace

    from PyQt6.QtCore import QDate

    first = widget._cells.currentData()
    second = replace(first, ix=first.ix + 1)
    index = SimpleNamespace(
        disciplines=("paragliders",),
        cells=lambda: [first, second],
        defaults=lambda _: (widget._utc_bounds()[0], 500),
        summer_days=lambda cell: [("2024-06-15" if cell == first else "2024-07-10", 5)],
    )
    monkeypatch.setattr(widget, "_start_plane", lambda: None)
    widget._index_ready(index)
    widget._mode.setCurrentIndex(1)
    assert widget._daily_start.date() == widget._daily_end.date() == QDate(2024, 6, 15)
    widget._daily_start.setDate(QDate(2024, 5, 1))
    widget._daily_end.setDate(QDate(2024, 9, 30))
    chosen = widget._read_bounds()
    widget._cells.setCurrentIndex(1)
    assert widget._daily_start.date() == QDate(2024, 7, 10)
    widget._cells.setCurrentIndex(0)
    widget._index_ready(index)
    assert widget._read_bounds() == chosen
    widget._best_day.click()
    assert widget._daily_start.date() == widget._daily_end.date() == QDate(2024, 6, 15)
    widget._daily_date_mode.setCurrentIndex(1)
    widget._before.setValue(3)
    widget._after.setValue(5)
    around = widget._read_bounds()
    widget._daily_date_mode.setCurrentIndex(0)
    assert widget._read_bounds() == around
    assert widget._daily_start.date() == QDate(2024, 6, 12)
    assert widget._daily_end.date() == QDate(2024, 6, 20)
    widget._daily_date_mode.setCurrentIndex(1)
    widget._cells.setCurrentIndex(1)
    around_second = widget._read_bounds()
    widget._daily_date_mode.setCurrentIndex(0)
    assert widget._read_bounds() == around_second
    assert widget._daily_start.date() == QDate(2024, 7, 7)
    assert widget._daily_end.date() == QDate(2024, 7, 15)
    widget._set_busy(True)
    assert not widget._daily_start.isEnabled() and not widget._daily_end.isEnabled()
    assert not widget._daily_date_mode.isEnabled()


def test_archive_change_clears_four_cells_and_old_results(widget):
    widget.invalidate()
    assert widget._index is None
    assert widget._plane is None
    assert widget._cells.count() == 0
    assert not widget._load.isEnabled()


def test_twelve_category_ranks_clickable_labels_and_relief_do_not_change_points(
    widget, monkeypatch, qapp
):
    from dataclasses import replace

    import numpy as np

    from xc_viewer.geography import TERRAIN_ORDER

    old = widget._cells.currentData()
    cells = [
        replace(old, ix=old.ix + i, terrain=band, flights=300 - rank)
        for i, band in enumerate(TERRAIN_ORDER)
        for rank in range(3)
    ]
    # Give all cells distinct geometry while deliberately keeping map markers close.
    cells = [
        replace(c, ix=old.ix + i % 4, iy=old.iy + i // 4) for i, c in enumerate(cells)
    ]
    images_read = []

    def relief(cell=None):
        images_read.append(cell)
        extent = (-5.5, 41, 10, 51.5) if cell is None else cell.bounds
        return np.full((3, 3, 3), 180, dtype=np.uint8), {"extent": extent}

    proxy = SimpleNamespace(
        disciplines=("paragliders",),
        cells=lambda: cells,
        defaults=lambda c: (widget._utc_bounds()[0], 500),
        relief=relief,
    )
    with monkeypatch.context() as m:
        m.setattr(widget, "_start_plane", lambda: None)
        widget._index_ready(proxy)
        assert widget._cells.count() == 12
        assert [widget._rank(c) for c in cells] == [1, 2, 3] * 4
        widget._canvas.draw()
        assert len(widget._map_labels) == 12
        for i in range(12):
            assert f"#{i % 3 + 1}" in widget._cells.itemText(i)
        label, target = widget._map_labels[8]
        box = label.get_bbox_patch().get_window_extent(widget._canvas.get_renderer())
        widget._map_clicked(
            SimpleNamespace(
                inaxes=widget._map_ax, x=(box.x0 + box.x1) / 2, y=(box.y0 + box.y1) / 2
            )
        )
        assert widget._cells.currentIndex() == target == 8
        assert len(widget._plane_ax.images) == 1
        assert list(widget._plane_ax.images[0].get_extent()) == [0, 5, 0, 5]
        loaded = len(images_read)
        widget._relief_strength.setValue(60)
        widget._slider.setValue(7000)
        assert len(images_read) == loaded
        assert widget._plane_ax.images[0].get_alpha() == 0.6


def test_dates_survive_other_controls_and_reload(widget, monkeypatch):
    from PyQt6.QtCore import QTime

    widget._start.setDateTime(widget._start.dateTime().addDays(-20))
    widget._end.setDateTime(widget._end.dateTime().addDays(20))
    bounds = widget._utc_bounds()
    with monkeypatch.context() as m:
        m.setattr(widget, "_start_plane", lambda: None)
        widget._step.setCurrentIndex(2)
        widget._height.setValue(273)
        assert widget._height.value() == 250
        widget._background.setCurrentIndex(1)
        widget._relief_strength.setValue(70)
        widget._before.setValue(3)
        widget._after.setValue(5)
        widget._hours[1].setTime(QTime(16, 0))
        widget._mode.setCurrentIndex(1)
        widget._index_ready(widget._index)
        assert widget._utc_bounds() == bounds
        assert len(widget._plane_axes) == 1
        assert widget._hours[1].time() == QTime(16, 0)
        start, end = widget._read_bounds()
        assert end - start == pytest.approx(9 * 86400)


def test_daily_window_draws_one_panel_and_focus_preserves_selection(
    widget, monkeypatch
):
    plane = widget._plane
    dates = widget._utc_bounds()
    height = widget._height.value()
    monkeypatch.setattr(widget, "_start_plane", lambda: None)
    widget._mode.setCurrentIndex(1)
    widget._plane_ready(plane)
    assert widget._map_ax is not None
    assert len(widget._plane_axes) == 1
    assert widget._time_settings.isHidden()
    assert not widget._daily_settings.isHidden()
    for focus in ("planes", "france", "overview"):
        widget._view.setCurrentIndex(widget._view.findData(focus))
        assert widget._plane is plane
        assert widget._utc_bounds() == dates
        assert widget._height.value() == height
        assert len(widget._plane_axes) == (focus != "france")
        assert (widget._map_ax is not None) == (focus != "planes")


def test_daily_hour_window_filters_points_on_the_paris_clock(widget, monkeypatch):
    from PyQt6.QtCore import QDate, QTime

    # The fixture's climb crosses z = 500 m at 14:01:20 Paris time (CEST).
    plane = widget._plane
    monkeypatch.setattr(widget, "_start_plane", lambda: None)
    widget._mode.setCurrentIndex(1)
    widget._daily_start.setDate(QDate(2024, 6, 15))
    widget._daily_end.setDate(QDate(2024, 6, 15))
    widget._plane_ready(plane)
    widget._height.setValue(500)
    for start, end, count in ((8, 18, 1), (14, 15, 1), (15, 18, 0), (8, 14, 0)):
        widget._hours[0].setTime(QTime(start, 0))
        widget._hours[1].setTime(QTime(end, 0))
        assert len(widget._plane_ax.collections) == count
    widget._hours[0].setTime(QTime(14, 0))
    widget._hours[1].setTime(QTime(0, 0))  # Midnight at the end of the day.
    assert len(widget._plane_ax.collections) == 1
    widget._hours[0].setTime(QTime(18, 0))
    widget._hours[1].setTime(QTime(17, 0))
    assert "start before it ends" in widget._plane_ax.texts[-1].get_text()


def test_same_dates_every_year_pool_only_the_season(widget, monkeypatch):
    from datetime import date

    from PyQt6.QtCore import QDate

    from xc_viewer.thermal_daily import local_bounds

    operations, reads = [], []
    monkeypatch.setattr(widget, "_run", lambda op, _: operations.append(op))
    widget._index.read_plane = lambda *args, **kw: reads.append(args)
    widget._mode.setCurrentIndex(1)
    operations.clear()
    widget._daily_date_mode.setCurrentIndex(widget._daily_date_mode.findData("yearly"))
    assert widget._season_start.isVisibleTo(widget)
    assert not widget._daily_start.isVisibleTo(widget)
    assert widget._best_day.isHidden()
    widget._season_start.setDate(QDate(2000, 6, 1))
    widget._season_end.setDate(QDate(2000, 8, 31))
    widget._first_year.setValue(2015)
    widget._last_year.setValue(2022)
    assert operations == []
    days = widget._selected_days()
    assert len(days) == 8 * 92
    assert days[0] == date(2015, 6, 1) and days[-1] == date(2022, 8, 31)
    assert date(2016, 1, 1) not in days
    start, end = widget._read_bounds()
    assert start == local_bounds("2015-06-01")[0]
    assert end == local_bounds("2022-08-31")[1] - 1e-6
    title = widget._figure._suptitle.get_text()
    assert "of each year 2015-2022" in title and "736 days pooled" in title
    widget._load.click()
    operations[0]()
    assert reads == [(widget._cells.currentData(), start, end, "vilpellet")]
    widget._last_year.setValue(2014)
    assert "last year" in widget._status.text()
    assert "Invalid date range" in widget._figure._suptitle.get_text()


def test_cell_flights_counts_every_visit_overlapping_the_paris_window(widget):
    from xc_viewer.thermal_daily import local_bounds, wall_windows

    day = datetime(2024, 6, 15).date()

    def at(hour):
        return local_bounds(day, (hour, hour))[0]

    spans = [(13.2, 13.8), (13.8, 14.2), (15, 15.5), (20, 20.2)]
    visits = pd.DataFrame(
        {"start": [at(a) for a, _ in spans], "end": [at(b) for _, b in spans]}
    )
    widget._plane = PlaneData(pd.DataFrame(), 4, 4, 0, 0, 0, visits=visits)
    assert widget._cell_flights() == 4
    assert widget._cell_flights(wall_windows([day], (13, 14))) == 2
    assert widget._cell_flights(wall_windows([day], (14, 18))) == 2
    assert widget._cell_flights(wall_windows([day], (18, 19))) == 0
    widget._plane = PlaneData(pd.DataFrame(), 0, 0, 0, 0, 0)
    assert widget._cell_flights() is None


def test_backdrops_are_cropped_to_the_view_at_screen_resolution():
    import numpy as np

    from xc_viewer.widgets.thermal_plane import _visible_part

    pixels = np.arange(64 * 64).reshape(64, 64)
    extent = (0, 8, 0, 8)  # 0.125 km per pixel
    whole, box = _visible_part(pixels, extent, ((0, 8), (0, 8)), 16)
    assert whole.shape == (16, 16)
    assert box == (0, 8, 0, 8)
    # A 2 km view keeps half a view of margin on each side, at full resolution.
    part, box = _visible_part(pixels, extent, ((2, 4), (2, 4)), 1000)
    assert box == (1, 5, 1, 5)
    assert part.shape == (32, 32)
    assert part[0, 0] == pixels[24, 8]
    assert _visible_part(pixels, extent, ((20, 22), (2, 4)), 1000) is None
