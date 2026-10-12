"""Stage (iii): trimming of ground phases (thesis, sec:trimming).

The outer trimming rule uses observed horizontal step speeds to estimate the
airborne interval (thesis, Eq. eq:trimming)::

    t_on  = min{ t : v_xy > v0 throughout [t, t + T0] }
    t_off = max{ t : v_xy > v0 throughout [t - T0, t] }

The rule takes the first and last qualifying fast runs, so an interior slow run
does not define the outer boundaries. Persistence reduces sensitivity to short
excursions but does not bound trimming error by ``T0``. Long slow flight at either
end can be clipped, and sustained fast ground movement can be retained. Speeds are
inferred between recorded fixes; they do not observe motion throughout a gap.

The clock is re-zeroed at the estimated onset. The original-clock boundaries are
stored for the integrity audit. They are processing estimates, not independently
verified take-off and landing times.

Interior slow runs are candidates for ground stops. Removal requires a duration
of at least ``T_ground`` and complete local coverage from an eligible raw barometer.
The pressure-altitude residual's 95th-minus-5th percentile span and fitted linear
slope must both satisfy their bounds. Neither test identifies pressure drift or
real vertical motion uniquely. Missing pressure evidence makes this guard abstain;
GNSS altitude does not replace it. A removed stint creates a segment boundary.

The driver uses ``frozen_tau_s`` as the minimum span worth reporting as suspect.
With the current threshold ordering, slow-and-flat runs at least this long but
shorter than ``T_ground`` are returned for future sensitivity analyses. Briefer
runs are not reported. This module does not perform waiting-time fits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..geo import great_circle_m

if TYPE_CHECKING:
    from ..config import TrimmingThresholds

# A flight with no sustained airborne stretch at all has no airborne segment.
DROP_NO_FLIGHT = "no_sustained_flight"

# The driver reuses this configured duration as the suspect-reporting floor.
# It is a working threshold, not a universal separation of ground and airborne motion.
_SUSPECT_MIN_SPAN_KEY = "frozen_tau_s"


@dataclass(frozen=True)
class Trimmed:
    """One flight after stage (iii).

    Attributes:
        fixes: The estimated airborne stretch, with the clock re-zeroed at onset and
            ``split_before`` set where an interior ground stint was excised.
        t_on: Estimated onset, in the input's recorded clock.
        t_off: Estimated end, in the same clock.
        trimmed_fraction: Fraction of the received elapsed span outside the outer
            window. Interior excisions do not enter this fraction. It describes
            processing impact, not the fraction of genuine flight wrongly removed.
        suspect_intervals: Slow-and-flat stints too short to excise, as ``t_start`` /
            ``t_end`` in the re-zeroed clock.
        n_interior_excised: How many interior ground stints were cut.
        drop_reason: :data:`DROP_NO_FLIGHT` when no sustained stretch exists.
    """

    fixes: pd.DataFrame
    t_on: float
    t_off: float
    trimmed_fraction: float
    suspect_intervals: pd.DataFrame
    n_interior_excised: int
    drop_reason: str | None


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs of ``True`` in ``mask``, as half-open index ranges."""
    edges = np.flatnonzero(np.diff(np.concatenate([[False], mask, [False]])))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2], strict=True)]


def airborne_window(
    t: np.ndarray, lat: np.ndarray, lon: np.ndarray, trimming: TrimmingThresholds
) -> tuple[float, float] | None:
    """Estimate ``[t_on, t_off]``, or return ``None`` if no run meets the speed rule.

    Args:
        t: Fix times, strictly increasing.
        lat: Latitudes in degrees.
        lon: Longitudes in degrees.
        trimming: The adopted thresholds.

    Returns:
        The window in the input clock, or ``None`` when no stretch holds ``v_xy > v0``
        on consecutive observed steps for at least ``T0``.
    """
    if t.size < 2:
        return None
    dt = np.diff(t)
    step = great_circle_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
    with np.errstate(divide="ignore", invalid="ignore"):
        fast = np.where(dt > 0, step / dt, 0.0) > trimming.takeoff_speed_mps
    sustained = [
        (start, stop)
        for start, stop in _runs(fast)
        if t[stop] - t[start] >= trimming.sustained_s
    ]
    if not sustained:
        return None
    return float(t[sustained[0][0]]), float(t[sustained[-1][1]])


def _is_flat(
    t: np.ndarray, alt: np.ndarray, tolerance_m: float, max_drift_mps: float
) -> bool:
    """Test pressure-altitude residual spread and fitted slope on a covered stint.

    Every candidate fix needs finite pressure altitude, with at least three samples.
    Ordinary least squares gives a linear trend; its absolute slope must not exceed
    ``max_drift_mps``. The driver sets this to
    ``frozen_delta_z_m / frozen_tau_s``. The residual's 95th-minus-5th percentile span
    must not exceed ``tolerance_m``.

    A linear climb and a linear pressure drift both disappear from the residual.
    The slope bound constrains their observed rate without identifying its cause.
    Slow real motion can pass; rapid pressure drift can fail.

    For an independent Gaussian reference model, the population central span is
    about ``3.29 sigma`` and the leading large-sample full-range scale is
    ``2 sigma sqrt(2 log n)``. Finite-sample percentile estimates fluctuate; neither
    expression is a distribution-free guarantee. A central span can also hide short
    excursions. Passing this test does not establish that a glider was on the ground.
    """
    finite = np.isfinite(alt)
    if finite.sum() < 3 or not finite.all():
        # A flight-level usable sensor need not cover this candidate stint. Sparse
        # observed endpoints cannot certify that the unobserved interior was flat.
        # Keep the stint when any contemporaneous barometric witness is missing.
        return False
    slope, intercept = np.polyfit(t[finite], alt[finite], 1)
    residual = alt[finite] - (slope * t[finite] + intercept)
    spread = float(np.percentile(residual, 95) - np.percentile(residual, 5))
    return bool(spread <= tolerance_m and abs(slope) <= max_drift_mps)


def trim_flight(
    fixes: pd.DataFrame,
    trimming: TrimmingThresholds,
    *,
    suspect_min_span_s: float,
    max_drift_mps: float,
    baro_witness: bool = False,
) -> Trimmed:
    """Run stage (iii) over one cleaned flight.

    Args:
        fixes: The flight after stage (ii): ``t``, ``lat``, ``lon``, ``alt``, plus the
            raw ``baro_alt`` channel and the per-fix cleaning columns. ``t`` must be
            strictly increasing.
        trimming: The adopted thresholds.
        suspect_min_span_s: Shortest slow-and-flat stint worth reporting as suspect;
            the driver passes ``fix_level.frozen_tau_s`` (see the module note).
        max_drift_mps: Largest fitted altitude slope a stint may show and still count as
            flat; the driver passes ``frozen_delta_z_m / frozen_tau_s`` (see
            :func:`_is_flat`).
        baro_witness: Whether this flight's raw barometric channel is usable as the
            flatness witness (stage (i) decides). Without one the interior-ground guard
            abstains; see the module note.

    Returns:
        The :class:`Trimmed` record. When no airborne window exists the fixes come back
        empty and ``drop_reason`` says so -- counted and recorded, never silently lost.

    Raises:
        ValueError: If a required column is missing.
    """
    missing = [c for c in ("t", "lat", "lon", "alt") if c not in fixes.columns]
    if missing:
        raise ValueError(f"the per-fix table is missing the column(s) {missing}")

    work = fixes.reset_index(drop=True)
    t = work["t"].to_numpy(dtype=float)
    lat = work["lat"].to_numpy(dtype=float)
    lon = work["lon"].to_numpy(dtype=float)
    recorded_span = float(t[-1] - t[0]) if t.size >= 2 else 0.0

    window = airborne_window(t, lat, lon, trimming)
    if window is None:
        return Trimmed(
            fixes=work.iloc[:0].copy(),
            t_on=float("nan"),
            t_off=float("nan"),
            trimmed_fraction=1.0,
            suspect_intervals=pd.DataFrame({"t_start": [], "t_end": []}),
            n_interior_excised=0,
            drop_reason=DROP_NO_FLIGHT,
        )

    t_on, t_off = window
    airborne = (t >= t_on) & (t <= t_off)
    out = work.loc[airborne].reset_index(drop=True)
    t = out["t"].to_numpy(dtype=float)
    excise, suspect = _interior_ground(
        t,
        out["lat"].to_numpy(dtype=float),
        out["lon"].to_numpy(dtype=float),
        _witness_altitude(out, baro_witness),
        trimming,
        suspect_min_span_s,
        max_drift_mps,
    )
    n_excised = len(_runs(excise))
    if "split_before" not in out.columns:
        out["split_before"] = False
    split = out["split_before"].to_numpy(dtype=bool).copy()
    for _, stop in _runs(excise):
        if stop < split.size:
            split[stop] = True
    out["split_before"] = split
    out = out.loc[~excise].reset_index(drop=True)
    # Elapsed time from the estimated onset; this is not verified take-off time.
    out["t"] = out["t"].to_numpy(dtype=float) - t_on

    airborne_span = t_off - t_on
    return Trimmed(
        fixes=out,
        t_on=t_on,
        t_off=t_off,
        trimmed_fraction=(
            1.0 - airborne_span / recorded_span if recorded_span > 0 else 0.0
        ),
        suspect_intervals=pd.DataFrame(
            {
                "t_start": [t[a] - t_on for a, _ in suspect],
                "t_end": [t[b - 1] - t_on for _, b in suspect],
            }
        ),
        n_interior_excised=n_excised,
        drop_reason=None,
    )


def _witness_altitude(fixes: pd.DataFrame, baro_witness: bool) -> np.ndarray:
    """The raw barometric channel as a flatness witness, all-``nan`` if there is none.

    The IGC zero means "absent" (see altchannel), so it must not read as a measurement
    of sea level; a flight whose barometer stage (i) judged unusable gets no witness at
    all, which :func:`_is_flat` turns into an abstention.
    """
    n = len(fixes)
    if not baro_witness or "baro_alt" not in fixes.columns:
        return np.full(n, np.nan)
    baro = fixes["baro_alt"].to_numpy(dtype=float)
    return np.where(baro == 0.0, np.nan, baro)


def _interior_ground(
    t: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    witness: np.ndarray,
    trimming: TrimmingThresholds,
    suspect_min_span_s: float,
    max_drift_mps: float,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Interior ground stints: which fixes to excise, and which to merely flag.

    Returns:
        ``(excise, suspect)``: a boolean array over the fixes, and the half-open index
        ranges of the slow-and-flat stints that were too short to cut.
    """
    excise = np.zeros(t.size, dtype=bool)
    suspect: list[tuple[int, int]] = []
    if t.size < 3:
        return excise, suspect
    dt = np.diff(t)
    step = great_circle_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
    with np.errstate(divide="ignore", invalid="ignore"):
        slow = np.where(dt > 0, step / dt, 0.0) < trimming.takeoff_speed_mps
    for start, stop in _runs(slow):
        stop = stop + 1  # a run of steps [start, stop) spans the fixes [start, stop]
        span = t[stop - 1] - t[start]
        if span < min(suspect_min_span_s, trimming.interior_ground_s):
            continue
        if not _is_flat(
            t[start:stop],
            witness[start:stop],
            trimming.ground_flatness_m,
            max_drift_mps,
        ):
            continue
        if span >= trimming.interior_ground_s:
            excise[start:stop] = True
        else:
            suspect.append((start, stop))
    return excise, suspect
