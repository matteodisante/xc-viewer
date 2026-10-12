"""The three binary observations the Vilpellet HMM emits, built exactly as published.

The reference implementation reduces a track to one bit per fix per question:

* is the glider flying persistently straight?
* is its mean vertical speed over the window positive?
* does it keep turning the same way?

Every window below is counted in fixes rather than seconds, which is what the reference
implementation does, and the eligibility gate in :mod:`.pipeline` is what makes the two
readings interchangeable.  The operations are deliberately expressed as the same
``pandas`` calls in the same order as the source, so that a transcription test can
compare the two outputs value by value rather than approximately.

Column names here are the repository's; the source names they correspond to are

=================================  ===========================================
this module                        reference implementation
=================================  ===========================================
``turn_increment_rad``             ``radius_curvature_lat_long_sgn``
``straight``                       ``radius_straight_indic``
``v_z``                            ``vertical_speed``
``mean_v_z``                       ``vertical_speed_mean``
``persistent_straight``            ``persistence_radius_straight_indic``
``positive_mean_v_z``              ``persistence_vertical_speed_indic``
``persistent_turn_sign``           ``persistence_radius_sgn_indic``
=================================  ===========================================

The quantity the source calls a "signed radius of curvature" is the signed increment of
the heading between two consecutive steps, in radians per fix; it is an angle, and the
straightness threshold is compared against it directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import VilpelletConfig

# The three emitted indicators, in the order the fitted emission vectors enumerate them.
OBSERVATION_COLUMNS = [
    "persistent_straight",
    "positive_mean_v_z",
    "persistent_turn_sign",
]

# WGS84, as used by the reference implementation's geodetic conversion.
_WGS84_A = 6378137.0
_WGS84_F = 1 / 298.257223563
_WGS84_E2 = _WGS84_F * (2 - _WGS84_F)


def geodetic_to_local_enu(
    lat_deg: np.ndarray, lon_deg: np.ndarray, alt_m: np.ndarray
) -> np.ndarray:
    """Project geodetic coordinates into a local ENU frame at the first fix.

    The reference implementation's own conversion: WGS84 geodetic to ECEF, then a
    rotation into the east/north/up triad of the first fix, with that fix as origin.

    Args:
        lat_deg: Latitude in signed degrees.
        lon_deg: Longitude in signed degrees.
        alt_m: Height above the ellipsoid in metres.

    Returns:
        An ``(n, 3)`` array of east, north and up displacements in metres.
    """
    phi = np.deg2rad(lat_deg)
    lam = np.deg2rad(lon_deg)
    sin_phi, cos_phi = np.sin(phi), np.cos(phi)
    curvature = _WGS84_A / np.sqrt(1 - _WGS84_E2 * sin_phi**2)
    ecef = np.stack(
        (
            (curvature + alt_m) * cos_phi * np.cos(lam),
            (curvature + alt_m) * cos_phi * np.sin(lam),
            (curvature * (1 - _WGS84_E2) + alt_m) * sin_phi,
        ),
        axis=-1,
    )
    offset = ecef - ecef[0]
    phi0, lam0 = np.deg2rad(lat_deg[0]), np.deg2rad(lon_deg[0])
    sin_phi0, cos_phi0 = np.sin(phi0), np.cos(phi0)
    sin_lam0, cos_lam0 = np.sin(lam0), np.cos(lam0)
    rotation = np.array(
        [
            [-sin_lam0, cos_lam0, 0.0],
            [-sin_phi0 * cos_lam0, -sin_phi0 * sin_lam0, cos_phi0],
            [cos_phi0 * cos_lam0, cos_phi0 * sin_lam0, sin_phi0],
        ]
    )
    return offset.dot(rotation.T)


def gaussian_smooth(values: pd.Series, window_fixes: int) -> pd.Series:
    """Centred Gaussian rolling mean of standard deviation ``window_fixes / 6``.

    The endpoints are left missing rather than shortened, exactly as the source does:
    the first and last partial windows produce ``NaN``, which later propagates into a
    ``False`` indicator instead of a shorter-window estimate.

    Args:
        values: One coordinate series.
        window_fixes: Window width in fixes.

    Returns:
        The smoothed series, indexed like ``values``.
    """
    return values.rolling(window=window_fixes, win_type="gaussian", center=True).mean(
        std=window_fixes / 6
    )


def signed_heading_increment(x: pd.Series, y: pd.Series) -> np.ndarray:
    """The signed turn between the step into a fix and the step out of it.

    With ``theta_k = atan2(y_k - y_{k-1}, x_k - x_{k-1})`` the heading of the step
    arriving at fix ``k``, the increment at ``k`` is ``theta_{k+1} - theta_k`` wrapped
    into ``(-pi, pi]``.  It is undefined at both ends of the track.

    Args:
        x: Eastward coordinate in metres.
        y: Northward coordinate in metres.

    Returns:
        An array of radians per fix, ``NaN`` at the first and last fix.
    """
    x_pre, x_now, x_post = x.shift(1).to_numpy(), x.to_numpy(), x.shift(-1).to_numpy()
    y_pre, y_now, y_post = y.shift(1).to_numpy(), y.to_numpy(), y.shift(-1).to_numpy()
    theta_in = np.arctan2(y_now - y_pre, x_now - x_pre)
    theta_out = np.arctan2(y_post - y_now, x_post - x_now)
    return (theta_out - theta_in + np.pi) % (2 * np.pi) - np.pi


def persistent_turn_sign(
    turn_increment: pd.Series, window_fixes: int, beta_persistence: float
) -> np.ndarray:
    """Whether the largest turns in a window agree on a direction.

    Within each window the ``window_fixes // 2`` increments of largest magnitude are
    selected, their signs counted, and the indicator fires when the majority sign
    covers at least ``ceil(beta * window_fixes // 2)`` of them.  Selecting by magnitude
    is what keeps the near-zero increments of straight flight, whose sign is noise,
    from diluting the count.

    The window at fix ``k`` spans ``k - window_fixes // 2`` to
    ``k + window_fixes // 2 - 1`` for an even width, with the track reflected at both
    ends; this off-by-one asymmetry is the source implementation's and is kept.

    Args:
        turn_increment: Signed heading increments in radians per fix.
        window_fixes: Window width in fixes.
        beta_persistence: Required agreeing fraction of the selected increments.

    Returns:
        An integer array of zeros and ones.
    """
    values = turn_increment.to_numpy(dtype=float)
    half_window = window_fixes // 2
    padded = np.pad(values, (half_window, half_window), mode="reflect")
    windows = np.lib.stride_tricks.as_strided(
        padded,
        shape=(values.size, window_fixes),
        strides=(padded.strides[0], padded.strides[0]),
    )
    largest = np.argpartition(-np.abs(windows), kth=half_window - 1, axis=1)[
        :, :half_window
    ]
    selected = windows[np.arange(windows.shape[0])[:, None], largest]
    signs = np.sign(selected)
    agreeing = np.maximum(np.sum(signs > 0, axis=1), np.sum(signs < 0, axis=1))
    return (agreeing >= np.ceil(beta_persistence * half_window)).astype(int)


def build_observation_frame(
    track: pd.DataFrame, config: VilpelletConfig, *, alpha_straight_rad: float
) -> pd.DataFrame:
    """Build the per-fix feature table of one contiguous, uniformly logged track.

    Args:
        track: One track with a ``t`` column in seconds and either the local ENU
            columns ``E``/``N``/``z`` or the geodetic columns ``lat``/``lon``/``alt``.
            Rows must already be sorted by ``t``.
        config: The loaded protocol.
        alpha_straight_rad: The discipline's straightness threshold.

    Returns:
        A copy of ``track`` with the intermediate quantities and the three columns of
        :data:`OBSERVATION_COLUMNS` appended, indexed like ``track``.

    Raises:
        ValueError: If neither coordinate triple is present.
    """
    params = config.features
    out = track.reset_index(drop=True).copy()
    if {"E", "N", "z"}.issubset(out.columns):
        east, north, up = (out["E"], out["N"], out["z"])
    elif {"lat", "lon", "alt"}.issubset(out.columns):
        enu = geodetic_to_local_enu(
            out["lat"].to_numpy(dtype=float),
            out["lon"].to_numpy(dtype=float),
            out["alt"].to_numpy(dtype=float),
        )
        east, north, up = (
            pd.Series(enu[:, axis], index=out.index) for axis in range(3)
        )
    else:
        raise ValueError("a Vilpellet track needs either E/N/z or lat/lon/alt columns")

    if config.input_policy.pre_smooth:
        window = params.position_smoothing_fixes
        east, north, up = (
            gaussian_smooth(east, window),
            gaussian_smooth(north, window),
            gaussian_smooth(up, window),
        )
    out["x"], out["y"], out["z_enu"] = east, north, up

    turn = pd.Series(signed_heading_increment(east, north), index=out.index)
    out["turn_increment_rad"] = turn.rolling(
        window=params.turn_smoothing_fixes, center=True, min_periods=1
    ).mean()

    time = out["t"]
    out["v_z"] = (out["z_enu"] - out["z_enu"].shift(1)) / (time - time.shift(1))
    out["mean_v_z"] = (
        out["v_z"]
        .rolling(window=params.persistence_fixes, center=True, min_periods=1)
        .mean()
    )

    # A NaN increment compares false against both bounds, so the edges of the track are
    # "not straight" rather than missing.  That is the source behaviour and it only
    # affects fixes the smoothing window already left undefined.
    out["straight"] = np.where(
        (out["turn_increment_rad"] < alpha_straight_rad)
        & (out["turn_increment_rad"] > -alpha_straight_rad),
        1,
        0,
    )
    out["persistent_straight"] = np.where(
        out["straight"]
        .rolling(window=params.persistence_fixes, center=True, min_periods=1)
        .mean()
        > params.beta_persistence,
        1,
        0,
    )
    out["positive_mean_v_z"] = np.where(out["mean_v_z"] > 0, 1, 0)
    out["persistent_turn_sign"] = persistent_turn_sign(
        out["turn_increment_rad"],
        window_fixes=params.persistence_fixes,
        beta_persistence=params.beta_persistence,
    )
    # The source forward-fills the whole table here before reading the three
    # indicators out of it. That fill cannot reach them: each is produced by a
    # comparison that already maps an undefined edge to zero, so the columns hold no
    # missing value to carry forward. It is omitted so that the intermediate
    # quantities keep the edge gaps the windows genuinely left, which is also what the
    # source's own retained frame holds.
    return out


def observation_matrix(frame: pd.DataFrame) -> np.ndarray:
    """The ``(n, 3)`` integer observation array the decoder consumes."""
    return frame[OBSERVATION_COLUMNS].to_numpy(dtype=np.int64)
