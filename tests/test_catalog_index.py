"""Tests for xc_viewer.catalog_index (Qt-free logic layer).

Never against the real multi-MB/GB catalog.csv / flights_meta.parquet -- small
synthetic tables written under tmp_path, with Discipline.config() monkeypatched to
point at them.
"""

from __future__ import annotations

import pandas as pd
import pytest

from xc_viewer import catalog_index
from xc_viewer.core import disciplines as disciplines_mod
from xc_viewer.core.disciplines import DataRoot

_CATALOG_ROWS = [
    {
        "flight_id": 1,
        "season_year": 2020,
        "date": "2020-07-15",
        "dept": "Isere",
        "flight_type": "Dist libre",
        "wing_class": "A",
        "takeoff": "SAINT HILAIRE",
        "landing": "LUMBIN",
        "club": "GCVL",
        "wing": "Atos V",
        "pilot": "Alice",
        "distance_km": 42.0,
        "duration_s": 7200.0,
    },
    {
        "flight_id": 2,
        "season_year": 2020,
        "date": "2020-08-20",
        "dept": "Isere",
        "flight_type": "Dist libre",
        "wing_class": "B",
        "takeoff": "SAINT HILAIRE",
        "landing": "LUMBIN",
        "club": "GCVL",
        "wing": "Atos V",
        "pilot": "Bob",
        "distance_km": 51.0,
        "duration_s": 9000.0,
    },
    {
        "flight_id": 3,
        "season_year": 2021,
        "date": "2021-07-15",
        "dept": "Savoie",
        "flight_type": "Triangle",
        "wing_class": "A",
        "takeoff": "COL DE LA CROIX",
        "landing": "ALBERTVILLE",
        "club": None,
        "wing": "Astir",
        "pilot": "Carla",
        "distance_km": 88.0,
        "duration_s": 12600.0,
    },
    {
        # A placeholder date, exactly like real FFVL rows -- must match nothing on
        # month, and still be resolvable on disk via the glob fallback.
        "flight_id": 4,
        "season_year": 2020,
        "date": "0000-00-00",
        "dept": "Isere",
        "flight_type": "Dist libre",
        "wing_class": "A",
        "takeoff": "SAINT HILAIRE",
        "landing": "LUMBIN",
        "club": "GCVL",
        "wing": "Atos V",
        "pilot": "Alice",
        "distance_km": 30.0,
        "duration_s": 5400.0,
    },
]

_FLIGHTS_META_ROWS = [
    {"flight_id": "1", "lat0": 45.31, "lon0": 5.89, "alt0": 980.0, "drop_reason": None},
    {
        "flight_id": "2",
        "lat0": 45.40,
        "lon0": 6.10,
        "alt0": 1010.0,
        "drop_reason": "duration_below_minimum",
    },
    # flights 3 and 4 never reached the pipeline: absent here on purpose.
]


@pytest.fixture(autouse=True)
def _clear_caches():
    catalog_index._catalog_cache.clear()
    catalog_index._flights_meta_cache.clear()
    yield
    catalog_index._catalog_cache.clear()
    catalog_index._flights_meta_cache.clear()


@pytest.fixture
def discipline(tmp_path, monkeypatch):
    root = tmp_path / "para_root"
    (root / "catalog").mkdir(parents=True)
    (root / "derived").mkdir(parents=True)

    def fake_config(self):
        return DataRoot(data_root=root)

    monkeypatch.setattr(disciplines_mod.Discipline, "config", fake_config)
    disc = disciplines_mod.PARAGLIDERS
    pd.DataFrame(_CATALOG_ROWS).to_csv(disc.catalog_path(), index=False)
    return disc


def test_filter_by_dept_year_month(discipline):
    result = catalog_index.filter_flights(
        discipline, dept="isere", season_year=2020, month=7
    )
    # Flight 2 (August) and 4 (placeholder date) are excluded; only flight 1 matches.
    assert result["flight_id"].tolist() == ["1"]


def test_catalog_rows_by_flight_id(discipline):
    result = catalog_index.catalog_rows(discipline, [3, "1", "99"])
    assert sorted(result["flight_id"]) == ["1", "3"]
    assert "_month" not in result.columns


def test_filter_by_flight_type_and_wing_class(discipline):
    result = catalog_index.filter_flights(
        discipline, flight_type="Triangle", wing_class="A"
    )
    assert result["flight_id"].tolist() == ["3"]


def test_filter_by_takeoff_landing_club_wing_pilot(discipline):
    result = catalog_index.filter_flights(discipline, takeoff="saint hilaire")
    assert result["flight_id"].tolist() == ["1", "2", "4"]

    result = catalog_index.filter_flights(discipline, landing="albertville")
    assert result["flight_id"].tolist() == ["3"]

    result = catalog_index.filter_flights(discipline, club="gcvl", wing="atos v")
    assert result["flight_id"].tolist() == ["1", "2", "4"]

    result = catalog_index.filter_flights(discipline, pilot="Alice")
    assert result["flight_id"].tolist() == ["1", "4"]


def test_distinct_values_are_sorted_and_exclude_blanks(discipline):
    assert catalog_index.distinct_values(discipline, "dept") == ["Isere", "Savoie"]
    pilots = catalog_index.distinct_values(discipline, "pilot")
    assert pilots == ["Alice", "Bob", "Carla"]
    # flight 3's club is None -- excluded, not turned into a bogus "None" entry.
    assert catalog_index.distinct_values(discipline, "club") == ["GCVL"]


def test_distinct_values_excludes_ffvl_anonymised_pilot_placeholders(discipline):
    root = discipline.config().data_root
    rows = [
        *_CATALOG_ROWS,
        {
            "flight_id": 5,
            "season_year": 2020,
            "date": "2020-07-15",
            "dept": "Isere",
            "flight_type": "Dist libre",
            "wing_class": "A",
            "takeoff": "SAINT HILAIRE",
            "landing": "LUMBIN",
            "club": "GCVL",
            "wing": "Atos V",
            "pilot": "$2A$07$FFVL0RGPD1SALT2345678U.FOBPSYACWGVBWKRYQXQSFQKY9ZOQVM",
        },
    ]
    pd.DataFrame(rows).to_csv(root / "catalog" / "catalog.csv", index=False)
    pilots = catalog_index.distinct_values(discipline, "pilot")
    assert pilots == ["Alice", "Bob", "Carla"]


def test_distinct_values_of_every_column_match_something(discipline):
    # Every value distinct_values offers must be usable as-is by filter_flights: a
    # dropdown built from one must never propose a value that then matches nothing.
    for column in catalog_index._TEXT_FILTER_COLUMNS:
        for value in catalog_index.distinct_values(discipline, column):
            result = catalog_index.filter_flights(discipline, **{column: value})
            assert len(result) > 0, f"{column}={value!r} matched no rows"


def test_kept_column_reflects_the_pipelines_verdict(discipline):
    pd.DataFrame(_FLIGHTS_META_ROWS).to_parquet(
        discipline.config().derived_dir / "flights_meta.parquet"
    )
    result = catalog_index.filter_flights(discipline)
    kept = dict(zip(result["flight_id"], result["kept"], strict=True))
    assert kept["1"] is True
    assert kept["2"] is False
    assert pd.isna(kept["3"])  # never processed by the pipeline


def test_kept_column_is_na_when_flights_meta_is_unreachable(discipline):
    result = catalog_index.filter_flights(discipline)
    assert result["kept"].isna().all()


def test_kept_only_excludes_unprocessed_and_dropped_flights(discipline):
    pd.DataFrame(_FLIGHTS_META_ROWS).to_parquet(
        discipline.config().derived_dir / "flights_meta.parquet"
    )
    result = catalog_index.filter_flights(discipline, kept_only=True)
    assert result["flight_id"].tolist() == ["1"]


def test_filter_by_region_keeps_only_take_offs_inside_that_box(discipline):
    meta_rows = [
        {
            "flight_id": "1",
            "lat0": 45.31,
            "lon0": 5.89,
            "alt0": 980.0,
            "drop_reason": None,
        },
        {
            "flight_id": "2",
            "lat0": 50.0,
            "lon0": 0.0,
            "alt0": 80.0,
            "drop_reason": None,
        },
        {
            "flight_id": "3",
            "lat0": 0.0,
            "lon0": 0.0,
            "alt0": 500.0,
            "drop_reason": None,
        },
    ]
    pd.DataFrame(meta_rows).to_parquet(
        discipline.config().derived_dir / "flights_meta.parquet"
    )
    result = catalog_index.filter_flights(discipline, region="Alps")
    assert result["flight_id"].tolist() == ["1"]


def test_filter_by_terrain_keeps_only_that_elevation_band(discipline):
    meta_rows = [
        {
            "flight_id": "1",
            "lat0": 45.31,
            "lon0": 5.89,
            "alt0": 980.0,
            "drop_reason": None,
        },
        {
            "flight_id": "2",
            "lat0": 45.40,
            "lon0": 6.10,
            "alt0": 80.0,
            "drop_reason": None,
        },
        {
            "flight_id": "3",
            "lat0": 45.50,
            "lon0": 6.20,
            "alt0": 2200.0,
            "drop_reason": None,
        },
    ]
    pd.DataFrame(meta_rows).to_parquet(
        discipline.config().derived_dir / "flights_meta.parquet"
    )
    result = catalog_index.filter_flights(discipline, terrain="Plains")
    assert result["flight_id"].tolist() == ["2"]


def test_filter_by_region_excludes_flights_never_reaching_the_pipeline(discipline):
    # flight 3 exists in the catalog (see _CATALOG_ROWS) but has no flights_meta row.
    meta_rows = [
        {
            "flight_id": "1",
            "lat0": 45.31,
            "lon0": 5.89,
            "alt0": 980.0,
            "drop_reason": None,
        },
    ]
    pd.DataFrame(meta_rows).to_parquet(
        discipline.config().derived_dir / "flights_meta.parquet"
    )
    result = catalog_index.filter_flights(discipline, region="Alps")
    assert result["flight_id"].tolist() == ["1"]


def test_resolve_igc_path_via_the_functional_name(discipline):
    igc_dir = discipline.config().igc_dir / "2020-2021"
    igc_dir.mkdir(parents=True)
    (igc_dir / "2020-07-15_1.igc").touch()

    row = pd.Series({"flight_id": "1", "season_year": 2020, "date": "2020-07-15"})
    expected = igc_dir / "2020-07-15_1.igc"
    assert catalog_index.resolve_igc_path(discipline, row) == expected


def test_resolve_igc_path_falls_back_to_a_glob_on_a_placeholder_date(discipline):
    igc_dir = discipline.config().igc_dir / "2020-2021"
    igc_dir.mkdir(parents=True)
    # The file's real name carries the true date; the catalog only has a placeholder.
    (igc_dir / "2020-07-19_4.igc").touch()

    row = pd.Series({"flight_id": "4", "season_year": 2020, "date": "0000-00-00"})
    expected = igc_dir / "2020-07-19_4.igc"
    assert catalog_index.resolve_igc_path(discipline, row) == expected


def test_resolve_igc_path_is_none_when_the_file_was_never_downloaded(discipline):
    row = pd.Series({"flight_id": "999", "season_year": 2020, "date": "2020-07-15"})
    assert catalog_index.resolve_igc_path(discipline, row) is None


def test_takeoff_points_joins_kept_flights_with_their_origin(discipline):
    meta_rows = [
        {
            "flight_id": "1",
            "lat0": 45.31,
            "lon0": 5.89,
            "alt0": 980.0,
            "drop_reason": None,
        },
        {
            "flight_id": "2",
            "lat0": 45.40,
            "lon0": 6.10,
            "alt0": 1010.0,
            "drop_reason": "too_short",
        },
        # flight 3 was retained by the pipeline but never got a local frame (dropped
        # before stage (v) after all -- meta rows can carry that combination): no
        # origin to plot, so it must be excluded even though "kept" elsewhere.
        {
            "flight_id": "3",
            "lat0": None,
            "lon0": None,
            "alt0": None,
            "drop_reason": None,
        },
        # flight 4 never reached the pipeline at all: absent here, same as elsewhere.
    ]
    pd.DataFrame(meta_rows).to_parquet(
        discipline.config().derived_dir / "flights_meta.parquet"
    )
    points = catalog_index.takeoff_points(discipline)
    assert points["flight_id"].tolist() == ["1"]
    assert points.iloc[0]["lat0"] == pytest.approx(45.31)
    assert points.iloc[0]["lon0"] == pytest.approx(5.89)
    assert points.iloc[0]["alt0"] == pytest.approx(980.0)
    assert points.iloc[0]["season_year"] == 2020
    assert points.iloc[0]["date"] == "2020-07-15"


def test_takeoff_points_is_empty_when_flights_meta_is_unreachable(discipline):
    points = catalog_index.takeoff_points(discipline)
    assert points.empty
    assert {"flight_id", "season_year", "date", "lat0", "lon0"} <= set(points.columns)


def test_pipeline_status_distinguishes_missing_results_from_rejection(discipline):
    absent = catalog_index.filter_flights(discipline)
    assert absent.pipeline_status.eq("Pipeline results unavailable").all()
    pd.DataFrame(_FLIGHTS_META_ROWS).to_parquet(
        discipline.config().derived_dir / "flights_meta.parquet"
    )
    catalog_index.clear_cache()
    result = catalog_index.filter_flights(discipline)
    assert result.pipeline_status.tolist() == [
        "Kept",
        "Dropped",
        "No archived result",
        "No archived result",
    ]
    assert result.loc[1, "drop_reason"] == "duration_below_minimum"
