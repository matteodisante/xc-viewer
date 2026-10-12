import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from xc_viewer.thermal_orography import (
    load_summits,
    select_cell_summits,
    validate_summits,
)


@pytest.fixture
def extract():
    return {
        "type": "FeatureCollection",
        "numberMatched": 1,
        "crs": {"properties": {"name": "urn:ogc:def:crs:EPSG::2154"}},
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [926000, 6477000]},
                "properties": {"nature": "Sommet", "toponyme": "Test summit"},
            }
        ],
        "provenance": {"bounds_epsg2154": [925000, 6475000, 930000, 6480000]},
    }


def test_missing_extract_is_distinct_from_verified_empty_extract(tmp_path, extract):
    cell = SimpleNamespace(
        ix=185, iy=1295, bounds=extract["provenance"]["bounds_epsg2154"]
    )
    assert load_summits(cell, tmp_path) is None
    extract["features"] = []
    extract["numberMatched"] = 0
    extract["crs"] = None
    (tmp_path / "ign-summits-185-1295.geojson").write_text(json.dumps(extract))
    assert load_summits(cell, tmp_path)["features"] == []


@pytest.mark.parametrize("defect", ["truncated", "crs", "outside", "ridge", "line"])
def test_invalid_geographic_extract_cannot_be_plotted(extract, defect):
    bad = deepcopy(extract)
    if defect == "truncated":
        bad["numberMatched"] = 2
    elif defect == "crs":
        bad["crs"]["properties"]["name"] = "EPSG:4326"
    elif defect == "outside":
        bad["features"][0]["geometry"]["coordinates"][0] = 900000
    elif defect == "ridge":
        bad["features"][0]["properties"]["nature"] = "Crête"
    else:
        bad["features"][0]["geometry"]["type"] = "LineString"
    with pytest.raises(ValueError):
        validate_summits(bad, extract["provenance"]["bounds_epsg2154"])


def test_saved_extract_cannot_silently_use_another_cells_extent(tmp_path, extract):
    cell = SimpleNamespace(ix=185, iy=1295, bounds=(0, 0, 5000, 5000))
    (tmp_path / "ign-summits-185-1295.geojson").write_text(json.dumps(extract))
    with pytest.raises(ValueError, match="extent"):
        load_summits(cell, tmp_path)


def test_wfs_envelope_candidates_are_clipped_to_the_exact_metric_square(extract):
    outside = deepcopy(extract["features"][0])
    outside["geometry"]["coordinates"][0] = 930111
    extract["features"].append(outside)
    extract["numberMatched"] = 2
    bounds = extract["provenance"]["bounds_epsg2154"]
    selected = select_cell_summits(extract, bounds)
    assert selected["numberMatched"] == 1
    assert len(selected["features"]) == 1
    validate_summits(selected, bounds)
    assert len(extract["features"]) == 2
