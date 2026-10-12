"""Stage (vi): a uniform time base within each flight (thesis, sec:uniform).

The smoothing of stage (vii) assumes a *uniform* sampling interval, so each flight's
time base is examined here. Loggers record at a fixed nominal rate, but the realised
series carries jitter and drop-outs. Each flight keeps its **native** rate -- the median
inter-fix interval ``dt`` -- because the derivatives are not finite differences across
flights but coefficients of a local polynomial fitted to each flight separately
(sec:savgol); what matters is uniformity *within* a flight, not across flights.

Three cases, decided by the split bound
``g_max = min(max_gap_factor * dt, max(max_gap_seconds, 2 * dt))``
(:func:`split_bound_s`):

* **uniform** -- every interval is ``dt``: the series passes through unchanged, since
  the interpolants below reproduce the samples exactly at the samples;
* **mildly irregular** -- jitter and isolated drop-outs, no gap past ``g_max``: the
  series is resampled onto the uniform grid at its own ``dt``, the short holes are
  filled, and every filled point is flagged;
* **broken by a long gap** -- a gap past ``g_max``, native or opened where cleaning
  excised a frozen-lock run: it is **not** bridged. Interpolating across it would
  fabricate dynamics the smoothing could not tell from real motion, so the flight is
  *split* there into independently analysed segments.

**A segment keeps its parent's origin and clock.** A split assigns a ``segment_id`` and
nothing else: the ENU conversion of stage (v) ran first, so every fix already carries
coordinates and elapsed time referred to the parent's origin, and ``t`` is never
re-zeroed. A segment starting at parent time ``t0 > 0`` is an observation window onto a
process already ``t0`` old, not a fresh flight -- re-zeroing would mix young and aged
trajectories, which the aging analysis must keep apart (sec:uniform, "Consequences of a
split").

Two gates act **per segment**, never on the parent (the flight-level cuts of stage (iv)
were passed by the parent and are not re-applied -- gaps concentrate in particular
conditions, so re-applying a duration cut would discard precisely the data recorded
around gaps):

* ``min_segment_duration_s`` -- below it a segment cannot support the smoothing window
  and the within-segment statistics;
* ``max_missing_fraction`` -- the share of grid points that had to be reconstructed.

A segment failing either is dropped **alone**; the flight is lost only if no segment
survives. Every segment the split produced is reported, kept or not, with the reason --
the material for the gap-cap sweep that sec:uniform owes.

The fill is per channel: shape-preserving piecewise cubic (PCHIP) on the horizontal
coordinates and linear on altitude. Both preserve the scalar range between consecutive
finite endpoints. Componentwise range preservation does not guarantee a physically
correct two-dimensional path. For a twice-differentiable true altitude and exact
endpoints, the linear interpolation error is bounded by max|z''|*g^2/8; missing altitude
gaps between finite altitude readings are subject to the same split bound as temporal
gaps. Longer vertical gaps remove the unsupported interval from all three coordinates.
Leading and trailing missing altitude is excluded, never held at the endpoint value.
Short reconstructions remain flagged; a duration cap does not guarantee accuracy.

``scipy`` is imported lazily, as elsewhere in this package, so importing this module
never requires the ``analysis`` dependency group.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..igc import median_sampling_period
from .enu import LOCAL_COLUMNS

if TYPE_CHECKING:
    from ..config import SamplingThresholds

# Columns of the resampled per-fix table, in order (carried-through columns follow),
# with the dtypes an empty result must still advertise.
FIX_DTYPES = {
    "segment_id": "int16",
    "t": "float64",
    "E": "float64",
    "N": "float64",
    "z": "float64",
    "interpolated": "bool",
    "z_reconstructed": "bool",
}
FIX_COLUMNS = list(FIX_DTYPES)

# Columns of the per-segment table (:class:`Resampled.segments`).
SEGMENT_COLUMNS = [
    "segment_id",
    "t_start",
    "t_end",
    "n_fix",
    "n_fix_raw",
    "frac_interpolated",
    "frac_z_reconstructed",
    "censored_start",
    "censored_end",
    "kept",
    "drop_reason",
]

# Why a flight, or one of its segments, was dropped here. Recorded, never merely acted
# on: the census of removals is what makes a cut auditable (sec:uniform).
DROP_NO_CADENCE = "no_native_cadence"  # flight: fewer than two fixes, or dt <= 0
DROP_NO_SEGMENT = "no_segment_survived"  # flight: every segment failed a gate
DROP_TOO_SHORT = "shorter_than_min_segment_duration"
DROP_TOO_SPARSE = "missing_fraction_above_max"
DROP_INCOMPLETE = "channel_not_reconstructable"  # a channel with nothing to fill from

# A grid point counts as *measured* when a fix lies within half a native step of it
# (sec:uniform), interpolated otherwise. The slack absorbs the floating-point dust of
# building the grid by repeated addition; it is nanoseconds against a step of seconds.
_HALF_STEP_SLACK_S = 1e-9


@dataclass(frozen=True)
class Resampled:
    """One flight after stage (vi): the two tables and the per-flight record.

    Attributes:
        fixes: One row per **grid point** of every retained segment, columns
            :data:`FIX_COLUMNS` followed by the carried-through columns of the input.
            Empty when the flight was dropped.
        segments: One row per segment the split produced -- retained or not -- with
            columns :data:`SEGMENT_COLUMNS`. ``drop_reason`` is null for a retained
            segment; ``n_fix`` counts the rows the segment contributed to ``fixes`` (so
            zero for a dropped one) against ``n_fix_raw``, its measured fixes.
        dt_s: The flight's native sampling interval (``nan`` if it has none).
        g_max_s: The split bound this flight was cut at, in seconds.
        frac_interpolated: Share of reconstructed grid points over the retained
            segments; the per-flight value ``flights_meta`` records.
        frac_z_reconstructed: The same share for the *vertical* channel, which can be
            absent while the horizontal record is intact and is therefore not implied by
            the one above.
        z_gap_max_s: Longest unbroken run of reconstructed altitude, in seconds -- the
            number that separates a two-second hole from the 500 s one that made the
            flag necessary. A fraction alone cannot tell them apart.
        was_resampled: Whether any grid point had to be reconstructed. Its complement is
            the thesis' "uniform as recorded" case (sec:uniform).
        drop_reason: ``None`` when at least one segment survived, otherwise
            :data:`DROP_NO_CADENCE` or :data:`DROP_NO_SEGMENT`.
    """

    fixes: pd.DataFrame
    segments: pd.DataFrame
    dt_s: float
    g_max_s: float
    frac_interpolated: float
    frac_z_reconstructed: float
    z_gap_max_s: float
    was_resampled: bool
    drop_reason: str | None


def split_bound_s(
    dt_s: np.ndarray | float, sampling: SamplingThresholds
) -> np.ndarray | float:
    """The largest inter-fix gap a flight of cadence ``dt_s`` may bridge, in seconds.

    ``g_max = min(max_gap_factor * dt, max(max_gap_seconds, 2 * dt))`` -- two scales,
    each pinned by its own argument (sec:uniform). The relative part tolerates isolated
    dropped fixes in proportion to the cadence and caps the number of interpolated grid
    points; the absolute cap stops a slow logger from bridging a hole long enough to
    hide a full thermalling circle, and sits below ``tau_freeze`` so that a native gap
    cannot be bridged where an excised frozen-lock run of the same length would split;
    the ``2 * dt`` floor keeps "a single missed fix never splits a flight" true at every
    cadence. Which term binds depends on the rate: the relative bound at the dominant
    1 s, the absolute cap from 2 s to 10 s, the floor above that.

    Args:
        dt_s: Native sampling interval(s), in seconds (scalar or array).
        sampling: The adopted thresholds.

    Returns:
        The bound, of the same shape as ``dt_s``.
    """
    return np.minimum(
        sampling.max_gap_factor * dt_s,
        np.maximum(sampling.max_gap_seconds, 2.0 * dt_s),
    )


def segment_bounds(
    t: np.ndarray, g_max_s: float, *, forced: np.ndarray | None = None
) -> list[tuple[int, int]]:
    """Cut a flight's fix times into segments at the gaps that cannot be bridged.

    Args:
        t: Fix times in seconds, strictly increasing.
        g_max_s: The split bound (:func:`split_bound_s`).
        forced: Optional per-fix boolean; ``True`` cuts immediately *before* that fix
            even if no time gap opened there. This is how stage (ii) hands over a
            position discontinuity -- a re-acquisition offset, where the track jumps and
            stays: both sides are genuine and only the transition between them is
            unknown, so it is handled exactly like a long gap (sec:uniform).

    Returns:
        ``[(start, stop), ...]``, half-open index ranges covering ``t`` in order.
    """
    cut = np.diff(t) > g_max_s
    if forced is not None:
        cut = cut | np.asarray(forced, dtype=bool)[1:]
    edges = [0, *(np.flatnonzero(cut) + 1).tolist(), len(t)]
    return list(itertools.pairwise(edges))


def _altitude_bounds(
    t: np.ndarray, z: np.ndarray, bounds: list[tuple[int, int]], g_max_s: float
) -> list[tuple[int, int]]:
    """Partition temporal segments into bounded altitude support and excluded runs.

    Consecutive finite readings may share a segment only up to ``g_max_s`` apart.
    Missing prefixes, suffixes and long interior holes become separate, incomplete
    candidates so their exclusion remains visible in the segment table. Refining each
    temporal segment separately preserves mandatory cleaning boundaries, even inside
    a missing-altitude run. Returned ranges still partition every input fix.
    """
    result = []
    for start, stop in bounds:
        finite = np.flatnonzero(np.isfinite(z[start:stop])) + start
        if finite.size == 0:
            result.append((start, stop))
            continue
        groups = np.split(finite, np.flatnonzero(np.diff(t[finite]) > g_max_s) + 1)
        cursor = start
        for group in groups:
            first, end = int(group[0]), int(group[-1]) + 1
            if cursor < first:
                result.append((cursor, first))
            result.append((first, end))
            cursor = end
        if cursor < stop:
            result.append((cursor, stop))
    return result


def _uniform_grid(t: np.ndarray, dt_s: float) -> np.ndarray:
    """The uniform grid of a segment: anchored on its first fix, stepping ``dt_s``.

    Anchored per segment, not on the parent's grid: the first fix after a gap is a
    genuine measurement and deserves to be a grid point. This does not touch the clock,
    which stays the parent's -- the anchor is a time *value* carried over from the
    parent, not a new zero.

    The span is rounded *down* to a whole number of steps, so the grid never reaches
    past the last fix and every point is interpolated strictly inside the measured
    range -- never extrapolated, and never held constant past the end. A uniform series
    lands on the grid exactly (the epsilon absorbs the floating-point dust of the
    division); where jitter leaves the last fix more than half a step past the final
    grid point, that point is honestly flagged as reconstructed.
    """
    n = int(np.floor((t[-1] - t[0]) / dt_s + 1e-9)) + 1
    return t[0] + np.arange(n) * dt_s


def _nearest_fix(t: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each grid point, the index of the nearest fix and its distance in seconds."""
    pos = np.searchsorted(t, grid)
    left = np.clip(pos - 1, 0, t.size - 1)
    right = np.clip(pos, 0, t.size - 1)
    take_left = np.abs(t[left] - grid) <= np.abs(t[right] - grid)
    nearest = np.where(take_left, left, right)
    return nearest, np.abs(t[nearest] - grid)


def _fill_channel(
    t: np.ndarray, values: np.ndarray, grid: np.ndarray, *, monotone: bool
) -> np.ndarray:
    """Resample one scalar channel onto ``grid``, over its own finite samples.

    Per channel, not per fix: a fix whose altitude cleaning invalidated still carries a
    valid position; within bounded support its altitude is restored here (sec:uniform).
    Each channel is interpolated over its own finite samples.

    Args:
        t: Fix times of the segment.
        values: The channel's values at those times, possibly with holes (``nan``).
        grid: The uniform grid to evaluate on.
        monotone: ``True`` for the monotone piecewise cubic (PCHIP) of the horizontal
            coordinates, ``False`` for the linear interpolant of the altitude.

    Returns:
        The channel on the grid, or an all-``nan`` array if fewer than two samples of it
        are finite -- nothing to interpolate between, which costs the segment its place
        (:data:`DROP_INCOMPLETE`).
    """
    ok = np.isfinite(values)
    if ok.sum() < 2:
        return np.full(grid.size, np.nan)
    if not monotone:
        return np.interp(grid, t[ok], values[ok], left=np.nan, right=np.nan)
    # Lazy, as elsewhere in this package: importing the module must not require the
    # optional `analysis` dependency group.
    from scipy.interpolate import PchipInterpolator

    return PchipInterpolator(t[ok], values[ok], extrapolate=True)(grid)


def resample_flight(
    local: pd.DataFrame,
    sampling: SamplingThresholds,
    *,
    dt_s: float | None = None,
    split_column: str = "split_before",
) -> Resampled:
    """Run stage (vi) over one flight: split at long gaps, resample what remains.

    Args:
        local: The flight in its local frame -- the output of stage (v): columns ``t``,
            ``E``, ``N``, ``z``, plus anything carried through. ``t`` must be strictly
            increasing, which is what stage (ii) guarantees by merging duplicate
            timestamps and repairing backward ones. An optional boolean column named by
            ``split_column`` forces a cut before the fixes it marks.
        sampling: The adopted sampling thresholds.
        dt_s: Native interval, if it is already known; by default the median inter-fix
            interval of this flight.
        split_column: Name of the optional forced-split column.

    Returns:
        The :class:`Resampled` record. Carried-through columns are sampled at the
        nearest fix within half a step and blanked (``False`` for a boolean flag,
        ``nan``/``NA`` otherwise) at reconstructed grid points, which carry no
        measurement to describe.

    Raises:
        ValueError: If a required column is missing or ``t`` is not strictly increasing.
    """
    missing = [c for c in LOCAL_COLUMNS if c not in local.columns]
    if missing:
        raise ValueError(f"the local-frame table is missing the column(s) {missing}")
    t = local["t"].to_numpy(dtype=float)
    if t.size >= 2 and not np.all(np.diff(t) > 0):
        raise ValueError(
            "t must be strictly increasing: duplicate or backward timestamps are "
            "stage (ii)'s to repair, not this stage's to absorb"
        )

    dt = float(dt_s) if dt_s is not None else median_sampling_period(local)
    if t.size < 2 or not np.isfinite(dt) or dt <= 0.0:
        return _dropped_flight(dt, float("nan"), DROP_NO_CADENCE, local)
    g_max = float(split_bound_s(dt, sampling))

    forced = local[split_column].to_numpy(dtype=bool) if split_column in local else None
    bounds = _altitude_bounds(
        t,
        local["z"].to_numpy(dtype=float),
        segment_bounds(t, g_max, forced=forced),
        g_max,
    )
    carried = [c for c in local.columns if c not in {*LOCAL_COLUMNS, split_column}]

    fix_frames: list[pd.DataFrame] = []
    records: list[dict] = []
    for segment_id, (start, stop) in enumerate(bounds):
        seg = local.iloc[start:stop]
        record = {
            "segment_id": segment_id,
            "t_start": float(t[start]),
            "t_end": float(t[stop - 1]),
            "n_fix": 0,
            "n_fix_raw": int(stop - start),
            "frac_interpolated": float("nan"),
            "frac_z_reconstructed": float("nan"),
            # A boundary that a split created truncates whatever phase was in progress
            # there, so the adjoining durations are lower bounds; the parent flight's
            # own first and last boundary are truncations of a different kind, and are
            # told apart by these being False (sec:uniform, "Censoring").
            "censored_start": start > 0,
            "censored_end": stop < t.size,
            "kept": False,
            "drop_reason": (
                DROP_TOO_SHORT if np.isfinite(seg["z"]).any() else DROP_INCOMPLETE
            ),
        }
        if stop - start >= 2:
            grid_fixes, frac, frac_z = _resample_segment(seg, dt, carried, segment_id)
            record["n_fix"] = len(grid_fixes)
            record["t_end"] = float(grid_fixes["t"].iloc[-1])
            record["frac_interpolated"] = frac
            record["frac_z_reconstructed"] = frac_z
            duration = record["t_end"] - record["t_start"]
            complete = bool(
                np.all(np.isfinite(grid_fixes[["E", "N", "z"]].to_numpy(dtype=float)))
            )
            if not complete:
                # The completeness invariant of sec:uniform -- "within a retained
                # segment, every grid time carries a defined (E, N, z)" -- is a promise
                # to everything downstream, which is never handed a missing value. A
                # channel with fewer than two samples to interpolate between cannot keep
                # it, so the segment goes rather than the promise.
                record["n_fix"] = 0
                record["drop_reason"] = DROP_INCOMPLETE
            elif duration < sampling.min_segment_duration_s:
                record["n_fix"] = 0
            elif frac > sampling.max_missing_fraction:
                record["n_fix"] = 0
                record["drop_reason"] = DROP_TOO_SPARSE
            else:
                record["kept"] = True
                record["drop_reason"] = None
                fix_frames.append(grid_fixes)
        records.append(record)

    segments = pd.DataFrame(records, columns=SEGMENT_COLUMNS)
    # Null-aware rather than object: a retained segment has no reason to give, and the
    # column must survive a Parquet round trip saying exactly that.
    segments["drop_reason"] = segments["drop_reason"].astype("string")
    if not fix_frames:
        return _dropped_flight(dt, g_max, DROP_NO_SEGMENT, local, segments=segments)

    fixes = pd.concat(fix_frames, ignore_index=True)
    kept = segments["kept"].to_numpy(dtype=bool)
    n_grid = segments.loc[kept, "n_fix"].sum()
    frac = float(
        (segments.loc[kept, "frac_interpolated"] * segments.loc[kept, "n_fix"]).sum()
        / n_grid
    )
    frac_z = float(
        (segments.loc[kept, "frac_z_reconstructed"] * segments.loc[kept, "n_fix"]).sum()
        / n_grid
    )
    return Resampled(
        fixes=fixes,
        segments=segments,
        dt_s=dt,
        g_max_s=g_max,
        frac_interpolated=frac,
        frac_z_reconstructed=frac_z,
        z_gap_max_s=_longest_run_s(fixes, dt),
        was_resampled=bool(fixes["interpolated"].any()),
        drop_reason=None,
    )


def _longest_run_s(fixes: pd.DataFrame, dt_s: float) -> float:
    """Longest unbroken run of reconstructed altitude, in seconds, over one flight.

    Runs are measured inside a segment, since a segment boundary is where the record
    stops rather than where the reconstruction does. Reported per flight because it is
    what a fraction cannot say: 5 % of a four-hour flight is either two hundred separate
    holes of a few seconds, which the linear bridge handles well, or one uninterrupted
    twelve minutes of invented altitude, which it does not.
    """
    worst = 0
    for _, segment in fixes.groupby("segment_id", sort=False):
        run = 0
        for flag in segment["z_reconstructed"].to_numpy(dtype=bool):
            run = run + 1 if flag else 0
            worst = max(worst, run)
    return float(worst * dt_s)


def _resample_segment(
    seg: pd.DataFrame, dt_s: float, carried: list[str], segment_id: int
) -> tuple[pd.DataFrame, float, float]:
    """One segment onto its uniform grid, with its two reconstruction fractions."""
    t = seg["t"].to_numpy(dtype=float)
    grid = _uniform_grid(t, dt_s)
    nearest, distance = _nearest_fix(t, grid)
    measured = distance <= 0.5 * dt_s + _HALF_STEP_SLACK_S

    # Time coverage and altitude support are distinct: a short altitude-only hole
    # has an existing horizontal fix and still needs its own reconstruction flag.
    # Long vertical holes and unsupported endpoints were excluded by _altitude_bounds.
    z_measured = measured & np.isfinite(seg["z"].to_numpy(dtype=float)[nearest])

    out = pd.DataFrame(
        {
            "segment_id": np.full(grid.size, segment_id, dtype=np.int16),
            "t": grid,
            # PCHIP on the horizontal, linear on the altitude: sec:uniform, and the
            # module docstring for why the two differ.
            "E": _fill_channel(t, seg["E"].to_numpy(float), grid, monotone=True),
            "N": _fill_channel(t, seg["N"].to_numpy(float), grid, monotone=True),
            "z": _fill_channel(t, seg["z"].to_numpy(float), grid, monotone=False),
            "interpolated": ~measured,
            "z_reconstructed": ~z_measured,
        },
        columns=FIX_COLUMNS,
    )
    for c in carried:
        column = seg[c].reset_index(drop=True).iloc[nearest].reset_index(drop=True)
        # A reconstructed grid point has no measurement for a flag to describe.
        out[c] = (
            column.to_numpy() & measured
            if pd.api.types.is_bool_dtype(column)
            else column.where(measured)
        )
    return out, float(np.mean(~measured)), float(np.mean(~z_measured))


def _dropped_flight(
    dt_s: float,
    g_max_s: float,
    reason: str,
    local: pd.DataFrame,
    *,
    segments: pd.DataFrame | None = None,
) -> Resampled:
    """A :class:`Resampled` carrying no fixes, for a flight that did not survive."""
    carried = [c for c in local.columns if c not in LOCAL_COLUMNS]
    empty = pd.DataFrame(
        {
            **{c: pd.Series(dtype=d) for c, d in FIX_DTYPES.items()},
            **{c: pd.Series(dtype=local[c].dtype) for c in carried},
        }
    )
    return Resampled(
        fixes=empty,
        segments=(
            segments if segments is not None else pd.DataFrame(columns=SEGMENT_COLUMNS)
        ),
        dt_s=dt_s,
        g_max_s=g_max_s,
        frac_interpolated=float("nan"),
        frac_z_reconstructed=float("nan"),
        z_gap_max_s=float("nan"),
        was_resampled=False,
        drop_reason=reason,
    )
