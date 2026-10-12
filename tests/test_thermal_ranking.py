"""Ground quality screening must not manufacture a new launch altitude."""

from xc_viewer.sqlite import connect
from xc_viewer.thermal_ranking import launch_quality


def _records(path, altitudes, valid="A"):
    lines = ["HFDTE010120"]
    for i, z in enumerate(altitudes):
        seconds = i * 30
        lines.append(
            f"B12{seconds // 60:02d}{seconds % 60:02d}"
            f"4544836N00626542E{valid}01638{z:05d}"
        )
    path.write_text("\n".join(lines))
    return path


def test_first_altitude_jump_requires_two_witnesses(tmp_path):
    path = _records(tmp_path / "bad.igc", [-7, 1724, 1726])
    assert launch_quality(path, 45, 10) == "altitude_jump"
    path = _records(path, [500, 1724, 502])
    assert launch_quality(path, 45, 10) == "accepted"


def test_receiver_invalid_origin_is_excluded_but_missing_support_is_explicit(tmp_path):
    path = _records(tmp_path / "invalid.igc", [500, 501, 502], valid="V")
    assert launch_quality(path, 45, 10) == "invalid_gnss"
    path = _records(path, [500])
    assert launch_quality(path, 45, 10) == "unchecked"


def test_dem_ranking_includes_cells_without_starts_and_stops_after_all_bands(
    tmp_path, monkeypatch
):

    from xc_viewer.thermal_ground import TERRAIN_RANKING
    from xc_viewer.thermal_index import ThermalIndex, _create_tables
    from xc_viewer.thermal_ranking import rank_cells

    path = tmp_path / "census.sqlite3"
    with connect(path) as db:
        _create_tables(db)
        # Highest-population cell has a misleading 188 m start, and all other
        # cells have no starts at all. None may be assigned by launch altitude.
        db.execute(
            "INSERT INTO flights VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            ("paragliders", "0", "a", 0, 0, 45, 6, 188, 0, 0, 188),
        )
        for ix, population in enumerate([6, 5, 4, 3, 2, 1]):
            db.executemany(
                "INSERT INTO visits VALUES (?,?,?,?,?,?,?)",
                [
                    ("paragliders", str(i), ix, 0, 0, 100, 2000)
                    for i in range(population)
                ],
            )
    highest = [1030.758, 875.409, 1500, 300, 299.999]
    fetched = []

    def terrain(cell, index):
        fetched.append(cell.ix)
        return {"minimum_m": highest[cell.ix] - 100, "maximum_m": highest[cell.ix]}

    monkeypatch.setattr("xc_viewer.thermal_ranking._cell_terrain", terrain)
    index = ThermalIndex(path, ("paragliders",))
    ranked = rank_cells(index, per_category=1)
    assert fetched == [0, 1, 2, 3, 4]  # lower-population sixth cell is unnecessary
    cells = ranked.cells()
    assert [c.ix for c in cells] == [4, 3, 0, 2]
    assert [c.terrain for c in cells] == [
        "Plains",
        "Hills",
        "Low mountains",
        "High mountains",
    ]
    assert [c.ground_m for c in cells] == [highest[c.ix] - 100 for c in cells]
    assert cells[2].launch_median_m == 188
    assert cells[0].launch_median_m is None
    assert cells[0].launches == 0
    assert ranked.quality_summary["ranking_reference"] == TERRAIN_RANKING
    assert ranked.terrain_references[0, 0]["maximum_m"] == 1030.758


def test_vilpellet_runs_determine_ranking_instead_of_visitors(tmp_path, monkeypatch):

    from xc_viewer.thermal_ground import CLIMB_RANKING
    from xc_viewer.thermal_index import ThermalIndex, _create_tables
    from xc_viewer.thermal_ranking import rank_cells

    path, activity = tmp_path / "census.sqlite3", tmp_path / "activity.sqlite3"
    with connect(path) as db:
        _create_tables(db)
        for ix, population in enumerate([100, 2, 3, 4, 5, 6]):
            db.executemany(
                "INSERT INTO visits VALUES (?,?,?,?,?,?,?)",
                [
                    ("paragliders", str(i), ix, 0, 0, 100, 2500)
                    for i in range(population)
                ],
            )
    with connect(activity) as db:
        db.executescript("""
            CREATE TABLE metadata(key TEXT,value TEXT);
            INSERT INTO metadata VALUES ('ready','1'),('signature','test');
            CREATE TABLE activity(discipline TEXT,flight_id TEXT,ix INTEGER,
                iy INTEGER,climb_runs INTEGER);
        """)
        db.executemany(
            "INSERT INTO activity VALUES ('paragliders','a',?,0,?)",
            enumerate([10, 30, 30, 29, 28, 27]),
        )
    highest = [100, 200, 250, 600, 1200, 1800]
    examined = []

    def terrain(cell, index):
        examined.append(cell.ix)
        return {"minimum_m": 0, "maximum_m": highest[cell.ix]}

    monkeypatch.setattr("xc_viewer.thermal_ranking._cell_terrain", terrain)
    ranked = rank_cells(
        ThermalIndex(path, ("paragliders",)), activity=activity, per_category=1
    )
    assert [c.ix for c in ranked.cells()] == [1, 3, 4, 5]
    assert examined == [1, 2, 3, 4, 5]
    assert ranked.cells()[0].flights == 2
    assert ranked.activity_counts[1, 0] == {"climb_runs": 30, "climb_flights": 1}
    assert ranked.quality_summary["ranking_reference"] == CLIMB_RANKING


def test_cell_beyond_dem_coverage_is_skipped_but_other_terrain_errors_abort(
    tmp_path, monkeypatch
):

    import pytest

    from xc_viewer.thermal_ground import IncompleteTerrainError
    from xc_viewer.thermal_index import ThermalIndex, _create_tables
    from xc_viewer.thermal_ranking import rank_cells

    path = tmp_path / "census.sqlite3"
    with connect(path) as db:
        _create_tables(db)
        for ix, population in enumerate([6, 5, 4, 3, 2]):
            db.executemany(
                "INSERT INTO visits VALUES (?,?,?,?,?,?,?)",
                [
                    ("paragliders", str(i), ix, 0, 0, 100, 2000)
                    for i in range(population)
                ],
            )
    highest = [None, 200, 500, 1000, 2000]  # cell 0 lies beyond the border

    def terrain(cell, index):
        if highest[cell.ix] is None:
            raise IncompleteTerrainError("Missing or invalid terrain elevations")
        return {"minimum_m": 0, "maximum_m": highest[cell.ix]}

    monkeypatch.setattr("xc_viewer.thermal_ranking._cell_terrain", terrain)
    index = ThermalIndex(path, ("paragliders",))
    ranked = rank_cells(index, per_category=1)
    assert [c.ix for c in ranked.cells()] == [1, 2, 3, 4]
    assert ranked.quality_summary["terrain_excluded"] == [[0, 0]]
    assert ranked.quality_summary["terrain_candidates_examined"] == 5

    def tampered(cell, index):
        raise ValueError("Terrain raster hash differs from its provenance")

    monkeypatch.setattr("xc_viewer.thermal_ranking._cell_terrain", tampered)
    with pytest.raises(ValueError, match="hash"):
        rank_cells(index, per_category=1)
