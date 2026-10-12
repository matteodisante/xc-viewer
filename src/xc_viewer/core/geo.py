"""Great-circle distance on the mean Earth sphere."""

from __future__ import annotations

import numpy as np

# Mean Earth radius (IUGG), the sphere the great-circle distance is measured on.
_EARTH_RADIUS_M = 6371008.8


def great_circle_m(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Haversine great-circle distance (metres) between two (arrays of) points.

    Scalars or equal-length arrays; ``(lat1, lon1)`` may be a single reference point
    broadcast against arrays ``(lat2, lon2)``, or two aligned arrays for consecutive
    steps. All angles in degrees.

    Returns:
        Distances in metres, broadcast to the common shape.
    """
    p1 = np.radians(np.asarray(lat1, dtype=float))
    p2 = np.radians(np.asarray(lat2, dtype=float))
    dphi = np.radians(np.asarray(lat2, dtype=float) - np.asarray(lat1, dtype=float))
    dlam = np.radians(np.asarray(lon2, dtype=float) - np.asarray(lon1, dtype=float))
    a = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2.0) ** 2
    return 2.0 * _EARTH_RADIUS_M * np.arcsin(np.sqrt(a))
