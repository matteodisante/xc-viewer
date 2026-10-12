"""Stage (iv): flight-level filtering (thesis, sec:flightfilter).

Five criteria restrict the analysed population, in this order:

* duration: a floor on recorded time summed within blocks, and a ceiling on the
  elapsed first-to-last span, including gaps;
* path length: a floor on great-circle steps summed within those same blocks;
* altitude activity: a floor on the finite adopted-altitude range, with abstention
  if no finite altitude remains;
* plausibility: a ceiling on path divided by recorded duration;
* reach: each fix's haversine distance from the first must be at most the configured
  speed times its elapsed time, plus ``REACH_TOLERANCE_M``.

The criteria act on the trimmed parent and are not reapplied to its individual
segments. Duration and path omit declared boundaries and gaps exceeding the later
resampling limit. The reach clock and origin do not reset at a boundary.

These are operational restrictions, not a classification of all real soaring.
For example, nearly level soaring can fail the altitude-range condition. Duration
selection can change the population contributing at long lags and the estimated
scaling, without a predetermined direction of bias. A flat retained-fraction curve
indicates low sensitivity to nearby cuts; it does not identify errors or rule out
selection bias.

The mean-speed criterion checks aggregation consistency: if every included step
satisfies the same speed bound, its duration-weighted mean also does. The reach
criterion can catch an inconsistent first fix, but cannot identify which fix is
wrong. It acts before ENU conversion, interpolation and smoothing; final-table
Cartesian checks use diagnostic tolerances and are not the same exact inequality.
Every rejected parent carries the first criterion it failed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..geo import great_circle_m

if TYPE_CHECKING:
    from ..config import FlightLevelThresholds, SamplingThresholds

DROP_TOO_SHORT = "duration_below_minimum"
DROP_TOO_LONG = "duration_above_maximum"
DROP_SHORT_PATH = "path_below_minimum"
DROP_FLAT = "altitude_range_below_minimum"
DROP_IMPLAUSIBLE = "mean_ground_speed_above_the_fix_level_bound"
DROP_UNREACHABLE = "extent_out_of_reach_of_the_first_fix"

# Additive allowance on raw-coordinate haversine reach, in metres. Later Cartesian
# checks are diagnostics after projection and filtering, not this exact predicate.
REACH_TOLERANCE_M = 50.0


@dataclass(frozen=True)
class FlightVerdict:
    """Whole-flight summary quantities and the first failed criterion.

    Attributes:
        duration_s: Airborne duration, summed over the blocks a split left.
        path_km: Flown path length, summed the same way -- never across a cut.
        alt_range_m: Total range of the adopted altitude channel over the airborne
            track (``nan`` if the channel has no value at all).
        extent_km: The farthest any fix lies from the first, in kilometres.
        drop_reason: ``None`` when the flight passes, otherwise the first failure in
            the order duration, path, altitude activity, mean speed and reach.
    """

    duration_s: float
    path_km: float
    alt_range_m: float
    extent_km: float
    drop_reason: str | None


def flight_quantities(
    fixes: pd.DataFrame,
    sampling: SamplingThresholds | None = None,
) -> tuple[float, float, float, float, np.ndarray, np.ndarray]:
    """Duration, path length and altitude range of one trimmed flight.

    Blocks are delimited by the ``split_before`` markers stages (ii) and (iii) left --
    an excised frozen-lock run, a re-acquisition offset, a mid-flight landing -- and by
    any inter-fix gap the resampling would later split at. Duration and path are
    accumulated inside a block and never across the cut that ends it: the trajectory
    there is unknown, and a straight line drawn over it would be counted as flown.

    The gap limit is computed from the flight's native cadence using the same rule
    as resampling. Later stages can still reject short or unsupported segments, so
    these quantities need not equal sums over the final stored table.

    Args:
        fixes: The trimmed flight, with ``t``, ``lat``, ``lon``, ``alt`` and optionally
            ``split_before``.
        sampling: The resampling thresholds, used only for ``g_max``. ``None`` keeps the
            markers as the only cuts, which is what a caller that has no configuration
            can honestly do.

    Returns:
        ``(duration_s, path_km, alt_range_m, extent_km, reach_m, elapsed_s)``, the
        last two being per-fix arrays: the great-circle distance from the first fix, and
        the time elapsed since it.
    """
    t = fixes["t"].to_numpy(dtype=float)
    if t.size < 2:
        empty = np.zeros(t.size)
        return 0.0, 0.0, float("nan"), 0.0, empty, empty
    lat = fixes["lat"].to_numpy(dtype=float)
    lon = fixes["lon"].to_numpy(dtype=float)
    alt = fixes["alt"].to_numpy(dtype=float)
    cut = (
        fixes["split_before"].to_numpy(dtype=bool)[1:]
        if "split_before" in fixes.columns
        else np.zeros(t.size - 1, dtype=bool)
    )
    gaps = np.diff(t)
    if sampling is not None:
        from .resample import split_bound_s

        dt_native = float(np.median(gaps[gaps > 0])) if (gaps > 0).any() else 0.0
        if dt_native > 0:
            cut = cut | (gaps > float(split_bound_s(dt_native, sampling)))
    step = great_circle_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
    within = ~cut
    duration = float(gaps[within].sum())
    path_km = float(step[within].sum()) / 1000.0
    alt_range = (
        float(np.nanmax(alt) - np.nanmin(alt))
        if np.isfinite(alt).any()
        else float("nan")
    )
    # Per-fix reach uses each observation's own elapsed time. Comparing only the
    # maximum extent with the full duration would miss some early inconsistencies.
    reach = great_circle_m(lat[0], lon[0], lat, lon)
    extent_km = float(reach.max()) / 1000.0
    return duration, path_km, alt_range, extent_km, reach, t - t[0]


def filter_flight(
    fixes: pd.DataFrame,
    flight_level: FlightLevelThresholds,
    *,
    max_mean_speed_mps: float | None = None,
    sampling: SamplingThresholds | None = None,
) -> FlightVerdict:
    """Run stage (iv) over one trimmed flight.

    Args:
        fixes: The trimmed flight (stage (iii) output).
        flight_level: The adopted cuts.
        max_mean_speed_mps: The plausibility bound on the mean ground speed; the driver
            passes the discipline's fix-level ``max_horizontal_speed_mps``. ``None``
            skips the check.
        sampling: The resampling thresholds. Passed only so that the duration and the
            path stop at the gaps stage (vi) will split at, and not merely at the
            markers the earlier stages left; ``None`` keeps the markers alone.

    Returns:
        The :class:`FlightVerdict`. A flight with an altitude channel that carries no
        value at all cannot be judged on activity and is not dropped for it: the
        criterion abstains rather than condemning on missing evidence.
    """
    duration, path_km, alt_range, extent_km, reach_m, elapsed_s = flight_quantities(
        fixes, sampling
    )
    if duration < flight_level.min_duration_s:
        reason = DROP_TOO_SHORT
    elif elapsed_s[-1] > flight_level.max_duration_s:
        # The ceiling constrains elapsed span including gaps; the floor above
        # constrains recorded duration accumulated within blocks.
        reason = DROP_TOO_LONG
    elif path_km < flight_level.min_path_km:
        reason = DROP_SHORT_PATH
    elif np.isfinite(alt_range) and alt_range < flight_level.min_alt_range_m:
        reason = DROP_FLAT
    elif (
        max_mean_speed_mps is not None
        and duration > 0
        and 1000.0 * path_km / duration > max_mean_speed_mps
    ):
        reason = DROP_IMPLAUSIBLE
    elif max_mean_speed_mps is not None and bool(
        (reach_m > max_mean_speed_mps * elapsed_s + REACH_TOLERANCE_M).any()
    ):
        # The retained record violates the adopted reach allowance. A wrong first
        # fix is one possible cause; this predicate does not establish attribution.
        reason = DROP_UNREACHABLE
    else:
        reason = None
    return FlightVerdict(
        duration_s=duration,
        path_km=path_km,
        alt_range_m=alt_range,
        extent_km=extent_km,
        drop_reason=reason,
    )
