"""Round-trip tests for xc_viewer.geodesy against the pipeline's own forward
projection (xc_viewer.core.preproc.enu) -- the two must agree, since the inverse
is
built from the same WGS84 constants and rotation.
"""

from __future__ import annotations

import numpy as np
import pytest

from xc_viewer.core.preproc.enu import LocalFrame, geodetic_to_enu
from xc_viewer.geodesy import enu_to_geodetic

LAT0, LON0, ALT0 = 45.0, 7.0, 1000.0
FRAME = LocalFrame(lat0_deg=LAT0, lon0_deg=LON0, alt0_m=ALT0)


def test_the_origin_itself_round_trips_exactly():
    zeros = np.array([0.0])
    lat, lon = enu_to_geodetic(zeros, zeros, np.array([ALT0]), FRAME)
    assert lat[0] == pytest.approx(LAT0, abs=1e-9)
    assert lon[0] == pytest.approx(LON0, abs=1e-9)


def test_points_across_a_flights_extent_round_trip_to_within_a_metre():
    # A grid spanning roughly +-20 km in each horizontal direction and +-1000 m of
    # altitude relative to the origin -- comfortably past what any real flight covers
    # (docs/guide/data-on-disk.md: extent_km rarely exceeds a few tens of km).
    rng = np.random.default_rng(0)
    n = 500
    lat = LAT0 + rng.uniform(-0.18, 0.18, n)  # ~ +-20 km in latitude
    lon = LON0 + rng.uniform(-0.25, 0.25, n)  # ~ +-20 km in longitude at this latitude
    alt = ALT0 + rng.uniform(-1000.0, 1000.0, n)

    east, north, _up = geodetic_to_enu(lat, lon, alt, LAT0, LON0, ALT0)
    # z is the trajectory's own vertical coordinate -- the same `alt` fed into the
    # forward projection, exactly as xc_viewer.core.preproc.enu.to_local_frame
    # uses
    # it (never the rotation's "up").
    lat_est, lon_est = enu_to_geodetic(east, north, alt, FRAME)

    m_per_deg_lat = 111_320.0
    m_per_deg_lon = m_per_deg_lat * np.cos(np.radians(LAT0))
    err_m = np.hypot((lat_est - lat) * m_per_deg_lat, (lon_est - lon) * m_per_deg_lon)
    assert err_m.max() < 1.0


def test_a_raw_and_cleaned_pair_projected_from_the_same_geographic_point_agree():
    # The overlay guarantee this function exists for: a point common to both the raw
    # and cleaned trajectories must land on the same (lat, lon) whichever direction it
    # is reached from.
    lat, lon, alt = 45.02, 7.03, 1450.0
    east, north, _up = geodetic_to_enu(lat, lon, alt, LAT0, LON0, ALT0)
    lat_back, lon_back = enu_to_geodetic(
        np.array([east]), np.array([north]), np.array([alt]), FRAME
    )
    assert lat_back[0] == pytest.approx(lat, abs=1e-6)
    assert lon_back[0] == pytest.approx(lon, abs=1e-6)


def test_float32_clean_coordinates_recover_locations_300km_away():
    """Long-route display inversion stays within 2 m in this geographic sweep."""
    from pyproj import Geod

    geod = Geod(ellps="WGS84")
    directions = np.arange(0, 360, 5)
    for lat0 in (42, 45, 50):
        for altitude in (0, 1000, 4000):
            lon, lat, _ = geod.fwd(
                np.full(len(directions), LON0),
                np.full(len(directions), lat0),
                directions,
                np.full(len(directions), 300000),
            )
            z = np.full(len(directions), altitude)
            e, n, _ = geodetic_to_enu(lat, lon, z, lat0, LON0, ALT0)
            la, lo = enu_to_geodetic(
                e.astype(np.float32),
                n.astype(np.float32),
                z,
                LocalFrame(lat0, LON0, ALT0),
            )
            assert np.max(geod.inv(lon, lat, lo, la)[2]) < 2
