"""The inverse of the pipeline's geodetic-to-ENU projection, for display only.

The pipeline (:mod:`xc_viewer.core.preproc.enu`) only ever goes geodetic -> ENU:
nothing downstream of stage (v) needs the reverse, so the cleaned trajectory it
produces (``FlightResult.fixes``) carries only ``(E, N, z)``, never ``(lat, lon)``. The
viewer wants to show that same trajectory in the geographic frame too, overlaid
against the raw one -- this module is exactly that inverse, built from the same WGS84
ellipsoid and the same rotation the forward projection uses
(:func:`~xc_viewer.core.preproc.enu.geodetic_to_ecef`,
:func:`~xc_viewer.core.preproc.enu.enu_rotation_matrix`,
:func:`~xc_viewer.core.preproc.enu.prime_vertical_radius_m`): reused, not
re-derived, so the two directions cannot silently disagree. No Qt or matplotlib
import (see the package docstring).
"""

from __future__ import annotations

import numpy as np

from .core.preproc.enu import (
    WGS84_A_M,
    WGS84_E2,
    LocalFrame,
    enu_rotation_matrix,
    geodetic_to_ecef,
    prime_vertical_radius_m,
)


def enu_to_geodetic(
    east: np.ndarray, north: np.ndarray, z: np.ndarray, frame: LocalFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Local ENU coordinates back to geodetic ``(lat, lon)``, for display.

    ``z`` here is the trajectory's own vertical coordinate -- the adopted altitude
    channel at its measured value, exactly as ``FlightResult.fixes["z"]`` carries it --
    never the rotation's "up" (``enu.py``'s module docstring): the two differ by the
    sagitta ``d^2 / (2R)``, tens of metres at 20 km, because "up" is a height above the
    flat *tangent plane* and ``z`` is the true altitude on the curved surface. That
    sagitta is corrected for here (using the same prime-vertical radius the forward
    projection is built on) before the rotation is undone, so the round trip recovers
    ``(lat, lon)`` to the residual of the forward projection itself -- centimetres at
    20 km, ~13 cm at 50 km (the next-order term the same docstring derives) -- not to
    the much larger, uncorrected sagitta.

    Args:
        east: The trajectory's ``E`` column, in metres.
        north: The trajectory's ``N`` column, in metres.
        z: The trajectory's ``z`` column (adopted altitude, metres) -- keep using it,
            not this function's return value, for the vertical coordinate of any plot;
            this function only recovers the horizontal position.
        frame: The same :class:`~xc_viewer.core.preproc.enu.LocalFrame` the
            trajectory was projected with
            (:func:`xc_viewer.data.frame_from_meta`).

    Returns:
        ``(lat_deg, lon_deg)``, each broadcast to the shape of the inputs.
    """
    east = np.asarray(east, dtype=float)
    north = np.asarray(north, dtype=float)
    z = np.asarray(z, dtype=float)

    radius = prime_vertical_radius_m(frame.lat0_deg)
    sagitta = (east**2 + north**2) / (2.0 * radius)
    up = (z - frame.alt0_m) - sagitta

    rot = enu_rotation_matrix(frame.lat0_deg, frame.lon0_deg)
    # rot's rows are (East, North, Up) in ECEF components; rot.T undoes the rotation.
    dx = rot[0, 0] * east + rot[1, 0] * north + rot[2, 0] * up
    dy = rot[0, 1] * east + rot[1, 1] * north + rot[2, 1] * up
    dz = rot[0, 2] * east + rot[1, 2] * north + rot[2, 2] * up

    x0, y0, z0 = geodetic_to_ecef(frame.lat0_deg, frame.lon0_deg, frame.alt0_m)
    lat, lon, _alt = _ecef_to_geodetic(x0 + dx, y0 + dy, z0 + dz)
    return lat, lon


def _ecef_to_geodetic(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ECEF ``(X, Y, Z)`` to geodetic ``(lat, lon, alt)`` on the WGS84 ellipsoid.

    Bowring's closed-form approximation (Bowring 1976): one reduced-latitude
    substitution, no iteration, sub-millimetre accurate at the altitudes and latitudes
    a flight is ever at.
    """
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    b = WGS84_A_M * np.sqrt(1.0 - WGS84_E2)
    ep2 = (WGS84_A_M**2 - b**2) / b**2
    theta = np.arctan2(z * WGS84_A_M, p * b)
    lat = np.arctan2(
        z + ep2 * b * np.sin(theta) ** 3,
        p - WGS84_E2 * WGS84_A_M * np.cos(theta) ** 3,
    )
    n_rad = WGS84_A_M / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - n_rad
    return np.degrees(lat), np.degrees(lon), alt
