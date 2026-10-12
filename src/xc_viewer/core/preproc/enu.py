"""Map geodetic fixes into the fixed local Cartesian frame used for analysis.

WGS84 geodetic latitude, longitude and a height proxy are first converted to ECEF.
Subtracting the origin and applying an orthogonal rotation gives East, North and Up.
The rotation preserves the full three-dimensional chord; retaining East and North
projects it into the tangent plane, so it does not preserve all surface distances.
The frame is anchored at the first retained fix after trimming and does not move.

For a spherical surface point at arc distance d from the origin, the horizontal
projection is R*sin(d/R), with shortening d^3/(6*R^2) to leading order: 0.03 m at
20 km, 0.51 m at 50 km, and 33 m at 200 km. Arc versus full ECEF chord instead has
leading deficit d^3/(24*R^2). Remote within-flight pairs require a separate estimate.

The logged GNSS altitude is used as a proxy for ellipsoidal height. IGC-approved and
CIVL recorder specifications differ in vertical datum, and the parser does not yet
harmonize them. A height error delta_h gives horizontal error of order abs(delta_h)*d/R
near the origin; there is no established 100 m bound on that error across the archive.

The analysis stores logged altitude z, not the rotated Up coordinate and not altitude
relative to the origin. A surface point's Up coordinate falls below the tangent plane
by approximately d^2/(2*R). Constant offsets cancel from z increments; changing datum
corrections or sensor biases do not. Missing altitude is temporarily interpolated only
for the geometric conversion; its missingness remains available to later resampling.
This stage carries elapsed time through unchanged. See thesis sec:enu and app:geodesy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# WGS84 reference ellipsoid (NGA TR8350.2): the datum the GPS receiver works in and the
# one the IGC B records are written in, so the fixes reach this module already in it and
# no datum transformation appears in the pipeline. Semi-major axis a, flattening f, and
# the first eccentricity squared e^2 = f(2 - f).
WGS84_A_M = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)

# Columns of the table this stage produces, in order (the rest of the input table is
# carried through behind them; see :func:`to_local_frame`).
LOCAL_COLUMNS = ["t", "E", "N", "z"]


@dataclass(frozen=True)
class LocalFrame:
    """Origin and orientation of a flight's fixed ENU frame.

    These values locate the frame in ECEF. Inverting a complete ENU vector also requires
    its Up component; East and North alone do not uniquely specify a geodetic position.
    alt0_m is the height proxy used at the origin, including temporary interpolation if
    its logged altitude is missing. It does not re-zero the stored altitude channel.
    """

    lat0_deg: float
    lon0_deg: float
    alt0_m: float


def prime_vertical_radius_m(lat_deg: np.ndarray | float) -> np.ndarray:
    """Prime-vertical radius of curvature ``N(phi)`` of the WGS84 ellipsoid, in metres.

    Away from the poles, this is the distance along the ellipsoid normal from the
    surface point to the polar axis (segment ``Q``--``P`` in thesis fig:ellipsoid).
    At the poles that intersection is degenerate and the formula extends by continuity.
    It is what replaces "the radius of the Earth" on an ellipsoid, and
    it is *not* the distance to the centre: the normal misses the centre except at the
    equator and the poles.

    Args:
        lat_deg: Geodetic latitude in degrees (scalar or array).

    Returns:
        ``a / sqrt(1 - e^2 sin^2 phi)``, between the polar ``a / sqrt(1 - e^2)``
        (~6399.6 km) and the equatorial ``a`` (~6378.1 km).
    """
    sin_lat = np.sin(np.radians(np.asarray(lat_deg, dtype=float)))
    return WGS84_A_M / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)


def geodetic_to_ecef(
    lat_deg: np.ndarray | float,
    lon_deg: np.ndarray | float,
    alt_m: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Geodetic ``(phi, lambda, h)`` to ECEF ``(X, Y, Z)`` (thesis, Eq. eq:ecef).

    The height ``h`` is measured along the ellipsoid normal, which is why it enters the
    horizontal components only through the radius factor ``N + h`` and the vertical one
    through ``N(1 - e^2) + h``; the asymmetry between the two is the flattening at work
    (derivation: Appendix app:geodesy).

    Args:
        lat_deg: Geodetic latitude in degrees.
        lon_deg: Longitude in degrees.
        alt_m: Height above the ellipsoid, in metres. The pipeline supplies logged
            GNSS altitude as a proxy without a recorder-datum correction; its
            horizontal geometric effect scales as ``abs(delta_h) * d / R``.

    Returns:
        ``(x, y, z)`` in metres, broadcast to the common shape of the inputs.
    """
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    alt = np.asarray(alt_m, dtype=float)
    n_rad = WGS84_A_M / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
    x = (n_rad + alt) * np.cos(lat) * np.cos(lon)
    y = (n_rad + alt) * np.cos(lat) * np.sin(lon)
    z = (n_rad * (1.0 - WGS84_E2) + alt) * np.sin(lat)
    return x, y, z


def enu_rotation_matrix(lat0_deg: float, lon0_deg: float) -> np.ndarray:
    """Rotation from ECEF components to the ENU frame tangent at the origin fix.

    The rows are the unit East, North and Up vectors of the local frame, written in ECEF
    components: East is tangent to the parallel, Up is the ellipsoid *normal* (hence the
    geodetic latitude), and North completes the right-handed triad (Appendix
    app:geodesy). Being orthonormal rows, the matrix is a rotation --
    ``R.T @ R == I``, ``det R == +1`` -- so it changes the axes without touching lengths
    or angles. It is evaluated once per flight, at the origin fix: the frame is fixed,
    not one that travels with the wing.

    Args:
        lat0_deg: Geodetic latitude of the origin fix, in degrees.
        lon0_deg: Longitude of the origin fix, in degrees.

    Returns:
        The ``(3, 3)`` matrix ``R`` with rows ``(East, North, Up)``.
    """
    lat0 = np.radians(float(lat0_deg))
    lon0 = np.radians(float(lon0_deg))
    sin_lat, cos_lat = np.sin(lat0), np.cos(lat0)
    sin_lon, cos_lon = np.sin(lon0), np.cos(lon0)
    return np.array(
        [
            [-sin_lon, cos_lon, 0.0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
        ]
    )


def geodetic_to_enu(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    alt_m: np.ndarray,
    lat0_deg: float,
    lon0_deg: float,
    alt0_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map geographic fixes onto the local ENU frame of an origin fix (sec:enu).

    Composes the two steps: each fix and the origin go to ECEF, and the displacement
    between them is rotated into the tangent frame at the origin. The ECEF coordinates
    never leave this function.

    Args:
        lat_deg: Geodetic latitudes of the fixes, in degrees.
        lon_deg: Longitudes of the fixes, in degrees.
        alt_m: Adopted altitude channel at the fixes, in metres.
        lat0_deg: Geodetic latitude of the origin fix, in degrees.
        lon0_deg: Longitude of the origin fix, in degrees.
        alt0_m: Adopted altitude at the origin fix, in metres.

    Returns:
        ``(east, north, up)`` in metres, each broadcast to the shape of the inputs and
        all three zero at the origin fix. Only ``east`` and ``north`` are analysis
        coordinates: the vertical the analysis uses is the adopted altitude itself,
        ``z = h`` (sec:enu), never ``up`` -- which is a height above the *tangent plane*
        and so drifts from the altitude by the sagitta, tens of metres over the extent
        of a flight.
    """
    x, y, z = geodetic_to_ecef(lat_deg, lon_deg, alt_m)
    x0, y0, z0 = geodetic_to_ecef(lat0_deg, lon0_deg, alt0_m)
    dx, dy, dz = x - x0, y - y0, z - z0
    # Applied row by row rather than as a matrix product: the shape of the input is
    # then preserved exactly (a scalar fix in, a scalar component out).
    rot = enu_rotation_matrix(lat0_deg, lon0_deg)
    east = rot[0, 0] * dx + rot[0, 1] * dy + rot[0, 2] * dz
    north = rot[1, 0] * dx + rot[1, 1] * dy + rot[1, 2] * dz
    up = rot[2, 0] * dx + rot[2, 1] * dy + rot[2, 2] * dz
    return east, north, up


def to_local_frame(
    fixes: pd.DataFrame, *, alt_column: str = "alt"
) -> tuple[pd.DataFrame, LocalFrame]:
    """Run stage (v) over one flight's fixes: geographic in, local Cartesian out.

    The origin is the flight's **first fix** -- which, at this point in the pipeline, is
    the first fix of the trimmed track, i.e. the start of free flight (sec:trimming) --
    so the trajectory starts at the horizontal origin, ``r(0) = 0`` (sec:notation). The
    time column is carried through untouched: its zero was set by trimming, not here.

    The vertical coordinate ``z`` is the adopted altitude channel at its measured value,
    copied over rather than rotated (sec:enu). Nothing is gained by re-zeroing it: every
    vertical quantity downstream (``v_z``, the climb/sink discrimination of the
    segmentation) depends on increments, which are invariant to the reference, while the
    measured value preserves the absolute height, an observable in its own right.

    Args:
        fixes: One flight's per-fix table, with columns ``t``, ``lat``, ``lon`` and the
            adopted altitude; any further columns (the per-fix cleaning flags of stage
            (ii), for instance) are carried through unchanged, so the ``fixes`` table
            can be assembled in one pass.
        alt_column: Name of the adopted altitude channel, the output of stage (i).

    Returns:
        ``(local, frame)``: the per-fix table with columns :data:`LOCAL_COLUMNS`
        (``t``, ``E``, ``N``, ``z``) followed by the carried-through columns, and the
        :class:`LocalFrame` that pins the frame to the Earth. The index of ``fixes`` is
        preserved. Values stay ``float64`` here; the cast to the ``float32`` of the
        stored table happens once, at write time.

    Raises:
        ValueError: If the table is empty (an origin fix is exactly what it lacks), or
            if a required column is missing.
    """
    required = ["t", "lat", "lon", alt_column]
    missing = [c for c in required if c not in fixes.columns]
    if missing:
        raise ValueError(f"fixes is missing the column(s) {missing}")
    if len(fixes) == 0:
        raise ValueError("cannot place a local frame on an empty track: no origin fix")

    lat = fixes["lat"].to_numpy(dtype=float)
    lon = fixes["lon"].to_numpy(dtype=float)
    alt = fixes[alt_column].to_numpy(dtype=float)
    # Height enters every ECEF component, so a missing value would also invalidate
    # the otherwise retained horizontal coordinates. Interpolate a height proxy for
    # this transform only, leaving the altitude-channel gap marked in the output.
    # Its horizontal influence is approximately abs(delta_h)*d/R near the origin;
    # this sensitivity is not a uniform sub-metre guarantee across all flights.
    for_frame = _altitude_for_frame(fixes["t"].to_numpy(dtype=float), alt)
    frame = LocalFrame(
        lat0_deg=float(lat[0]), lon0_deg=float(lon[0]), alt0_m=float(for_frame[0])
    )
    east, north, _up = geodetic_to_enu(
        lat, lon, for_frame, frame.lat0_deg, frame.lon0_deg, frame.alt0_m
    )

    carried = [c for c in fixes.columns if c not in {"lat", "lon", alt_column, "t"}]
    local = pd.DataFrame(
        {
            "t": fixes["t"].to_numpy(dtype=float),
            "E": east,
            "N": north,
            "z": alt,
        },
        index=fixes.index,
    )
    for c in carried:
        local[c] = fixes[c].to_numpy()
    return local, frame


def _altitude_for_frame(t: np.ndarray, alt: np.ndarray) -> np.ndarray:
    """The altitude series to feed the ECEF transform, with no gaps left in it.

    Missing values are interpolated from the flight's own altitudes, and a flight with
    none at all falls back to sea level. This series is used *only* to place the fixes
    in the local frame, never as the vertical coordinate: ``z`` keeps the gap, which
    stage (vi) fills and flags only within the gap bound, excluding longer holes.
    """
    finite = np.isfinite(alt)
    if finite.all():
        return alt
    if not finite.any():
        return np.zeros_like(alt)
    return np.interp(t, t[finite], alt[finite])
