"""Fix-level cleaning before projection, resampling and smoothing.

Position and time rules remove selected fixes; altitude rules invalidate only that
channel. These are operational decisions, not a guarantee that every removed fix
is erroneous or every retained fix is correct. Slowly varying errors can survive,
and interpolation cannot reconstruct unobserved manoeuvres.

After timestamp handling, Hampel residuals identify and attribute candidates, while
the horizontal impossibility/rejoin gate controls deletion independently of the
flag. Collapsed or exactly repeated coordinate runs are tested for duration and
witness support. Removed frozen-run intervals force segment boundaries. Altitude
uses absolute bounds, adjacent windowed speed tests and isolated return-spike rules.

Distances here are spherical great-circle approximations on raw geodetic
coordinates. The later ENU transform and smoothing define different coordinates,
so a speed bound enforced here is not a bound on every smoothed output step.
The integrity gate is called after trimming and measures reconstruction burden
over the retained airborne interval, without estimating a true defect rate.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..geo import great_circle_m

if TYPE_CHECKING:
    from ..config import FixLevelThresholds

# Per-fix columns this stage adds. `split_before` is consumed by stage (vi), which
# realises the cut; the other two are the per-fix record of what the cleaning did, and
# travel all the way into the `fixes` table.
CLEANING_COLUMNS = ["hampel_flagged", "alt_invalidated", "split_before"]

# Why a fix was deleted. Recorded per fix, not merely counted: the removal audit of
# sec:fixlevel needs to know which rule fired, and where.
REMOVED_BACKWARD_TIME = "backward_timestamp"
REMOVED_POSITION_SPIKE = "position_spike"
REMOVED_FROZEN_RUN = "frozen_lock_run"

# The flight-level verdict of :func:`integrity_gate`.
DROP_INTEGRITY = "cleaning_rebuilt_too_much"

# Conventional scalar Gaussian MAD normalization; radial local residuals do not
# inherit an exact Gaussian standard-deviation interpretation (Eq. eq:hampel).
_MAD_TO_SIGMA = 1.4826


@dataclass(frozen=True)
class CleaningReport:
    """What the cleaning did to one flight: the per-rule counters of the audit.

    Every field is a count, and every count is over the whole flight; the integrity gate
    re-counts the deletions over the airborne window alone. ``n_flagged_kept`` is the
    number the thesis calls the per-flight anomaly count: fixes the Hampel identifier
    flagged but the speed bound refused to condemn, kept and recorded.
    """

    n_fix_raw: int
    n_fix_clean: int
    n_merged_duplicates: int
    n_removed_backward: int
    n_removed_spike: int
    n_removed_frozen: int
    n_alt_out_of_band: int
    n_alt_vz_sustained: int
    n_alt_vz_spike: int
    n_flagged_kept: int
    n_splits: int
    n_vz_runs: int
    n_alt_level_shift: int
    split_jump_max_m: float
    n_v_flag: int
    n_boundaried: int


@dataclass(frozen=True)
class Cleaned:
    """One flight after stage (ii).

    Attributes:
        fixes: The surviving fixes, with :data:`CLEANING_COLUMNS` added and the adopted
            ``alt`` set to ``nan`` wherever the cleaning invalidated it.
        removed: One row per deleted fix -- ``t`` and ``reason`` -- so the removals can
            be counted inside any window and attributed to the rule that made them.
        report: The per-rule counters.
    """

    fixes: pd.DataFrame
    removed: pd.DataFrame
    report: CleaningReport


def longest_non_decreasing(values: np.ndarray) -> np.ndarray:
    """Indices of a longest non-decreasing subsequence of ``values``.

    The minimal-removal rule for backward timestamps (sec:fixlevel): delete the
    smallest set of fixes whose removal leaves a clock that never goes back, i.e. keep a
    longest non-decreasing subsequence. Ties are kept, because two fixes in one UTC
    second are a *duplicate*, handled by the merge that runs next, not a backward step.

    The naive alternative -- clamping with a running maximum, as the parser used to --
    would condemn every fix after a single forward-jumped clock instead of the one fix
    that carries the corrupt time.

    Args:
        values: The (possibly non-monotonic) timestamps.

    Returns:
        The kept indices, ascending. Patience sorting, ``O(n log n)``.
    """
    tails: list[float] = []
    tail_index: list[int] = []
    predecessor = np.full(values.size, -1, dtype=np.int64)
    for i, value in enumerate(values):
        pos = bisect.bisect_right(tails, value)
        if pos:
            predecessor[i] = tail_index[pos - 1]
        if pos == len(tails):
            tails.append(value)
            tail_index.append(i)
        else:
            tails[pos] = value
            tail_index[pos] = i
    keep: list[int] = []
    node = tail_index[-1] if tail_index else -1
    while node >= 0:
        keep.append(node)
        node = int(predecessor[node])
    return np.array(keep[::-1], dtype=np.int64)


def unwrap_longitude(lon_deg: np.ndarray) -> np.ndarray:
    """Place longitude on a continuous branch before componentwise calculations.

    Medians, bounding boxes and duplicate-time centroids must not average values just
    below +180 degrees with values just above -180 degrees as if they were far apart.
    Unwrapping avoids that coordinate discontinuity; trigonometric distance and ENU
    calculations remain periodic. Stored longitude is wrapped again at the end.

    Args:
        lon_deg: Longitude in degrees.

    Returns:
        A continuous branch, possibly outside [-180, 180].
    """
    return np.degrees(np.unwrap(np.radians(lon_deg)))


def wrap_longitude(lon_deg: np.ndarray) -> np.ndarray:
    """Bring longitudes back into ``[-180, 180)``, the convention readers expect."""
    return ((np.asarray(lon_deg, dtype=float) + 180.0) % 360.0) - 180.0


def hampel_flags(
    t: np.ndarray, lat: np.ndarray, lon: np.ndarray, fix_level: FixLevelThresholds
) -> np.ndarray:
    """Identify local horizontal-position residuals; do not delete fixes.

    At each fix, compute the componentwise median latitude/longitude in the inclusive
    physical-time window +/- w and the great-circle residual r_k from that median.
    The local scale is 1.4826 times the median of neighbouring residuals r_j, each of
    which was formed around its own local median. Flag r_k above the greater of
    k times this scale and the configured metre floor, only if enough fixes exist.

    The factor is the conventional scalar Gaussian MAD normalization; it does not
    make this radial, correlated residual an exactly calibrated Gaussian statistic.
    The flag attributes a candidate. The separate impossibility/rejoin gate makes
    the horizontal deletion decision, and no median is substituted for a position.
    """
    index = pd.to_timedelta(t, unit="s")
    roll = {
        "window": pd.Timedelta(seconds=2.0 * fix_level.hampel_window_s),
        "center": True,
        "closed": "both",  # the inclusive |t_j - t_k| <= w of eq:hampelresid
        "min_periods": 1,
    }
    med_lat = pd.Series(lat, index=index).rolling(**roll).median().to_numpy()
    med_lon = pd.Series(lon, index=index).rolling(**roll).median().to_numpy()
    populated = pd.Series(lat, index=index).rolling(**roll).count().to_numpy()
    residual = great_circle_m(lat, lon, med_lat, med_lon)
    mad = pd.Series(residual, index=index).rolling(**roll).median().to_numpy()
    threshold = np.maximum(
        fix_level.hampel_k * _MAD_TO_SIGMA * mad, fix_level.hampel_eps_min_m
    )
    return (populated >= fix_level.hampel_min_window_fixes) & (residual > threshold)


def _step_speeds(t: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Horizontal speed between consecutive fixes, in m/s (``inf`` on a zero step)."""
    dt = np.diff(t)
    step = great_circle_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(dt > 0, step / dt, np.inf)


def _scan_positions(
    t: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    flagged: np.ndarray,
    max_speed_mps: float,
    block_span_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scan left to right, removing bounded excursions and splitting discontinuities.

    The last accepted fix anchors the impossibility gate. An unreachable block is
    deleted if its removal lets the bracketing fixes rejoin within the speed bound.
    This assigns a removal using local geometry; it cannot prove which position is
    erroneous. A real manoeuvre whose recorded steps respect the bound is retained.

    The Hampel flag is recorded but does not gate deletion (ADR 0001). Contamination
    of most of a median window can hide the anomalous block from that identifier;
    the absolute-speed gate does not depend on its local scale. This avoids that
    particular failure, without guaranteeing correct attribution in every record.

    When a candidate excursion cannot rejoin within the permitted span, the scan
    retains both sides and marks a discontinuity. Keeping them does not certify
    their positions; the transition is excluded from subsequent path accumulation.

    Args:
        t: Fix times, strictly increasing.
        lat: Latitudes in degrees.
        lon: Longitudes in degrees.
        flagged: The Hampel flags, recorded per fix.
        max_speed_mps: The discipline's absolute horizontal-speed bound.
        block_span_s: Permitted span of a pending excursion before an unresolved
            continuation becomes a discontinuity. A persistent offset can become
            reachable before this cap and be treated as an excursion; the parameter
            therefore affects attribution as well as computational work.

    Returns:
        ``(delete, split_before, anchor)``. The first two are boolean arrays over the
        fixes. The third is the index of the fix each one was judged against, which is
        its predecessor wherever no block was open; it carries no decision and exists so
        that the explainer figure of sec:fixlevel can draw the reach test the scan
        actually applied rather than a restatement of it. The invariant that no
        impossible step survives inside a segment is *not* established here: two later
        passes still delete fixes, so it is enforced once, on the final surviving set,
        by :func:`_boundary_impossible_steps`.
    """
    n = t.size
    delete = np.zeros(n, dtype=bool)
    split = np.zeros(n, dtype=bool)
    # Every fix is judged against its predecessor until a block opens and freezes it.
    anchors = np.maximum(np.arange(n) - 1, 0)
    step_speed = _step_speeds(t, lat, lon)
    # The overwhelmingly common case: nothing off the trend and no impossible step, so
    # the scan below would accept every fix in turn. Skipping it keeps the archive-wide
    # cost in the parser rather than in this loop.
    if not flagged.any() and np.all(step_speed <= max_speed_mps):
        return delete, split, anchors

    anchor = 0
    pending: list[int] = []
    for i in range(1, n):
        anchors[i] = anchor
        if not pending and step_speed[i - 1] <= max_speed_mps:
            anchor = i
            continue
        gap = t[i] - t[anchor]
        reach = great_circle_m(lat[anchor], lon[anchor], lat[i], lon[i])
        speed = reach / gap if gap > 0 else np.inf
        if speed <= max_speed_mps:
            delete[pending] = True  # this block's removal restored the trend
            pending = []
            anchor = i
        elif not pending or t[i] - t[pending[0]] <= block_span_s:
            pending.append(i)
        else:
            split[pending[0]] = True
            pending = []
            anchor = i
    if pending:
        # A block still open when the track ends is the mirror of one that opens it, and
        # takes the mirror decision. The scan never got to finish its argument here:
        # a block is confirmed corrupt by the track *rejoining* after it, and there is
        # no after. What can be said is that the block is short -- the loop splits and
        # resets as soon as a block outlives `block_span_s`, so one still pending at the
        # end spans at most that -- and that the two readings of it cost very different
        # amounts. If the block is corrupt, deleting it costs seconds near the landing,
        # which the trimming would very likely take anyway. If instead the *anchor* were
        # the corrupt one, keeping the block would mean trusting a handful of fixes over
        # the whole flight that preceded them. And keeping it is not free: the fix stays
        # in the record, becomes the farthest point from the origin, and the flight is
        # refused whole by the reach bound of sec:flightfilter -- one corrupt trailing
        # fix discarding four thousand seconds of perfect flight.
        delete[pending] = True

    # There is no rule here for a corrupt block that *opens* the record, and its
    # absence is the decision rather than an omission. Such a block is real: the
    # first fix of the trimmed track becomes the origin of the local frame
    # (sec:enu), so a corrupt one displaces every coordinate of the flight, and five
    # paraglider flights sat 4500 km from their own origin because of it. The
    # temptation is to delete the prefix.
    #
    # It was tried twice, and both rules were wrong, in opposite directions. Firing
    # on the first impossible step accuses the *earlier* end always, so an ordinary
    # spike three seconds in deleted the good fixes and promoted the spike to
    # origin. Firing instead on a split the scan declares near the start looks sound
    # -- a split means no bounded removal after the step restored the trend -- but
    # the split index is `pending[0]`, the first fix of the block that could not be
    # rejoined, and that block is *after* the step whenever the corruption simply
    # outlasts `block_span_s`. A 50 s corrupt run five seconds in therefore deleted
    # five good fixes, kept fifty bad ones, and left the origin 5039 km out. The
    # scan's verdict says a discontinuity exists; it does not say which side of it
    # the record belongs to, and at the very start there is no trend behind the step
    # to settle it.
    #
    # So the discontinuity is left as a discontinuity, and the guarantee is taken
    # where it can be stated without attributing blame: at the flight level, where a
    # record whose own first fix is out of reach of the rest is refused whole
    # (sec:flightfilter). That bound never depended on this rule. What the rule
    # would have bought is the recovery of two flights in 156017 -- against a
    # failure mode that has now been demonstrated twice.

    # Postcondition, and the reason it is stated as one rather than left to the scan:
    # whatever the scan decided, no step *between two surviving fixes* may still break
    # the absolute bound. Where the scan keeps a block it could neither rejoin nor
    # explain, the step back out of that block is exactly such a step, and without this
    # it would enter a segment as flown motion -- a 40 km jump inside one second, which
    # then carries every average computed over that segment. A step the bound rejects
    # is a transition of unknown course, so it is made a boundary, on the same footing
    # as a long gap. It is enforced on the final surviving set, in `clean_flight`.
    return delete, split, anchors


def _frozen_runs(
    t: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    alt: np.ndarray,
    baro: np.ndarray,
    valid: np.ndarray,
    fix_level: FixLevelThresholds,
    baro_witness: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect candidate repeated-position intervals with joint operational criteria.

    Candidates are exact runs of repeated decoded coordinates and greedy intervals
    whose bounding-box diameter stays below the configured distance. They must last
    at least frozen_tau_s and pass _is_witnessed. Those criteria are not statistically
    independent and do not prove the receiver's internal lock state. Slow or quantized
    physical motion and shared recorder failures remain possible confounders.

    Exact coordinate repetition is accepted as a witness by the current policy.
    Otherwise a flight with an eligible barometer needs a finite reading at every
    candidate fix and a small difference between its first/last-quarter medians;
    missing local support causes abstention. Without an eligible barometer, a majority
    of GNSS-invalid or missing-altitude declarations supplies the fallback witness.
    The barometer comparison measures net change, not the absence of interior motion.

    Returns:
        Deletion mask and split-before mask. Every removed block ends in a forced
        boundary at its next surviving fix, preventing interpolation across that block.
    """
    n = t.size
    delete = np.zeros(n, dtype=bool)
    split = np.zeros(n, dtype=bool)
    if n < 2:
        return delete, split

    # Exact-repeat candidates are tracked separately: a greedy collapsed interval
    # can include a nearby nonidentical fix and would then fail the equality test.
    runs = _identical_runs(lat, lon) + _collapsed_runs(t, lat, lon, fix_level)
    for i, j in runs:
        if t[j] - t[i] < fix_level.frozen_tau_s:
            continue
        if _is_witnessed(
            lat[i : j + 1],
            lon[i : j + 1],
            alt[i : j + 1],
            baro[i : j + 1],
            valid[i : j + 1],
            fix_level,
            baro_witness,
        ):
            delete[i : j + 1] = True
    # One split per deleted block, at the first fix that survives it.
    edges = np.flatnonzero(np.diff(np.concatenate([[False], delete, [False]])))
    for stop in edges[1::2]:
        if stop < n:
            split[stop] = True
    return delete, split


def _identical_runs(lat: np.ndarray, lon: np.ndarray) -> list[tuple[int, int]]:
    """Find inclusive runs of exactly equal decoded latitude/longitude values.

    This is equality without a distance tolerance, not a comparison of whole raw IGC
    records. Coordinate quantization can produce equal values during physical motion.
    The frozen-run policy additionally requires the configured minimum duration.
    """
    same = (lat[1:] == lat[:-1]) & (lon[1:] == lon[:-1])
    edges = np.flatnonzero(np.diff(np.concatenate([[False], same, [False]])))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2], strict=True)]


def _collapsed_runs(
    t: np.ndarray, lat: np.ndarray, lon: np.ndarray, fix_level: FixLevelThresholds
) -> list[tuple[int, int]]:
    """Find greedy candidate intervals with bounding-box diameter below delta_xy.

    A trailing-window bounding-box prefilter limits the exact scan to potentially
    collapsed stretches. Such geometry identifies candidates; it does not establish
    that a slow, spatially confined interval is a recorder defect.
    """
    eps = fix_level.frozen_eps_m
    index = pd.to_timedelta(t, unit="s")
    roll = {
        "window": pd.Timedelta(seconds=fix_level.frozen_tau_s),
        "closed": "both",
        "min_periods": 1,
    }
    lat_s, lon_s = pd.Series(lat, index=index), pd.Series(lon, index=index)
    diagonal = great_circle_m(
        lat_s.rolling(**roll).min().to_numpy(),
        lon_s.rolling(**roll).min().to_numpy(),
        lat_s.rolling(**roll).max().to_numpy(),
        lon_s.rolling(**roll).max().to_numpy(),
    )
    # A trailing window only means what it says once it *is* a window: with
    # `min_periods=1` the first fix of every flight has a bounding box of zero and would
    # open a candidate region, from which the expansion below walks the whole track --
    # on a 1 Hz flight every consecutive step is under delta_xy, so nothing stops it.
    # Requiring the window to span tau_freeze is the same condition the rule applies
    # anyway, moved to where it saves the work.
    candidate = (diagonal < eps) & ((t - t[0]) >= fix_level.frozen_tau_s)
    if not candidate.any():
        return []

    step = great_circle_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
    small = np.concatenate([step < eps, [False]])
    runs: list[tuple[int, int]] = []
    edges = np.flatnonzero(np.diff(np.concatenate([[False], candidate, [False]])))
    for first, last in zip(edges[::2], edges[1::2], strict=True):
        # The window is a trailing one, so the run may start before the first flagged
        # fix; it may equally continue past the last, as long as the steps stay small.
        start = int(np.searchsorted(t, t[first] - fix_level.frozen_tau_s))
        stop = int(last) - 1
        while stop + 1 < t.size and small[stop]:
            stop += 1
        i = start
        while i <= stop:
            j = i
            lo_lat = hi_lat = lat[i]
            lo_lon = hi_lon = lon[i]
            while j + 1 <= stop:
                new = (
                    min(lo_lat, lat[j + 1]),
                    min(lo_lon, lon[j + 1]),
                    max(hi_lat, lat[j + 1]),
                    max(hi_lon, lon[j + 1]),
                )
                if float(great_circle_m(*new)) >= eps:
                    break
                lo_lat, lo_lon, hi_lat, hi_lon = new
                j += 1
            runs.append((i, j))
            i = j + 1
    return runs


def _is_witnessed(
    lat: np.ndarray,
    lon: np.ndarray,
    alt: np.ndarray,
    baro: np.ndarray,
    valid: np.ndarray,
    fix_level: FixLevelThresholds,
    baro_witness: bool,
) -> bool:
    """Evaluate the configured witness for one candidate interval.

    Exact decoded coordinate repetition passes directly. Otherwise a barometer deemed
    eligible at flight level must also be finite at every candidate fix; the first
    and last quarter medians are compared. This is a net-change test. No missing
    barometric reading is interpolated. If no eligible barometer exists, the fallback
    requires a majority of invalid-GNSS or missing-altitude declarations.

    Args:
        lat: Decoded latitudes over the run.
        lon: Decoded longitudes over the run.
        alt: Adopted GNSS altitude, with unavailable values represented by NaN.
        baro: Raw pressure-altitude readings, NaN where absent.
        valid: Recorder A/V flags.
        fix_level: Operational thresholds.
        baro_witness: Flight-level barometer eligibility; not local coverage.
    """
    byte_identical = bool(np.all(lat == lat[0]) and np.all(lon == lon[0]))
    if byte_identical:
        # This branch uses exact equality, with no tolerance. It is a configured
        # decision rule, not proof that physical motion was impossible.
        return True
    if baro_witness:
        # Flight-level presence does not establish coverage of this particular run.
        # Require a contemporaneous barometric reading at every candidate fix; do not
        # interpolate a missing witness or infer a flat run from two observed ends.
        # Abstain locally rather than falling back to a weaker GNSS declaration when
        # an otherwise usable independent sensor happens to be missing here.
        if not np.isfinite(baro).all():
            return False
        # Between medians of the run's ends: this tests net altitude change, not the
        # absence of every interior excursion. The geometric condition is still needed.
        end = max(1, baro.size // 4)
        head, tail = np.median(baro[:end]), np.median(baro[-end:])
        return abs(tail - head) < fix_level.frozen_delta_z_m
    # No barometer: the recorder's own declarations are the only witness left, and they
    # must speak for the run rather than for a stray fix in it.
    declared = ~valid.astype(bool) | ~np.isfinite(alt)
    return bool(declared.mean() > 0.5)


def _largest_boundary_jump(fixes: pd.DataFrame) -> float:
    """The largest displacement across a boundary this stage declared, in metres.

    A boundary is a statement that the trajectory between two fixes is unknown, and the
    two kinds this stage produces differ in a way that matters downstream. Excising a
    frozen-lock run leaves the position on either side genuine; splitting at a
    re-acquisition offset does not, because the jump is the instrument's and everything
    after it carries an unknown constant. Segments keep the parent's origin
    (sec:uniform), so that constant enters the *ensemble* MSD, which is built from
    absolute position -- while leaving the time-averaged one, built from increments,
    untouched.

    Measured on the cleaned track, which is *before* trimming, so it is an upper bound
    on the displacement that actually reaches the analysis rather than that displacement
    itself -- a jump inside a ground phase counts here and is then trimmed away. On the
    archive the two differ at the tail and nowhere else: the largest value over retained
    paraglider flights is thousands of kilometres while their largest retained extent is
    555 km, which is exactly such a ground-phase jump. The quantity that matters is the
    displacement across a boundary of a *retained* segment, and it is measured where it
    can be measured honestly -- on the written table, by the thesis repository's
    dataset verification, which streams it anyway.

    No rule is applied to either: the effect is small and bounding it is worth more than
    guessing at it. On a sample of the archive the median such displacement is 20 m for
    paragliders and 80 m for hang gliders, the 90th percentile 150 m and 450 m; only
    0.55 % and 1.25 % of flights carry one over a kilometre, and the handful over
    100 km are the corrupt records the reach bound of sec:flightfilter refuses whole.
    Reporting the maximum per flight is what lets that statement be checked rather than
    asserted, and lets an analysis of absolute position exclude on it if it must.
    """
    if "split_before" not in fixes.columns or len(fixes) < 2:
        return 0.0
    at = np.flatnonzero(fixes["split_before"].to_numpy(dtype=bool))
    at = at[at > 0]
    if not at.size:
        return 0.0
    lat = fixes["lat"].to_numpy(dtype=float)
    lon = fixes["lon"].to_numpy(dtype=float)
    return float(great_circle_m(lat[at - 1], lon[at - 1], lat[at], lon[at]).max())


def step_vz(t: np.ndarray, alt: np.ndarray) -> np.ndarray:
    """Vertical speed between consecutive fixes, in m/s (``0`` on a zero time step).

    Public: :func:`xc_viewer.core.geo` reuses it (with :func:`local_vz`) to
    measure the same windowed statistic on the raw archive, off the pipeline, when
    re-deriving ``max_vertical_speed_mps`` from data (thesis, sec:fixlevel "Validating
    the cleaning").
    """
    dt = np.diff(t)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(dt > 0, np.diff(alt) / dt, 0.0)


def local_vz(
    t: np.ndarray, v_z: np.ndarray, window_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """The local vertical speed: a rolling median of ``|v_z|`` over the steps.

    ``v_z`` lives on the *steps* between fixes rather than on the fixes, so the window
    is centred on a step's own midpoint ``(t_j + t_{j+1})/2`` and holds every step
    whose midpoint falls within ``+/- window_s`` of it.

    The median describes a neighbourhood rather than a single increment. With an
    odd number of finite steps, it exceeds a bound only if a majority exceed it;
    with an even number, the average of the two central order statistics decides.
    Isolated excursions therefore need not trigger this statistic, and are tested
    separately by the out-and-back rule in ``_clean_altitude``.

    This is not a noise-free velocity estimate: sufficiently strong GNSS noise or
    sustained real motion can exceed an operational bound. A time window keeps the
    temporal scale fixed across logging cadences. The caller falls back to per-step
    speed when too few finite steps support the median.

    Args:
        t: Fix times in seconds, strictly increasing.
        v_z: Vertical speed per step, ``t.size - 1`` values.
        window_s: Half-width of the centred window, in seconds.

    Returns:
        ``(median, populated)``: the rolling median of ``|v_z|`` at each step, and how
        many steps its window actually held. A step with a missing altitude at either
        end is ``nan``, which both the median and the count skip, so a hole in the
        channel neither biases the estimate nor counts towards the population that
        licenses it.
    """
    if v_z.size == 0:
        return np.empty(0), np.empty(0)
    midpoint = 0.5 * (t[:-1] + t[1:])
    roll = {
        "window": pd.Timedelta(seconds=2.0 * window_s),
        "center": True,
        "closed": "both",
        "min_periods": 1,
    }
    magnitude = np.abs(v_z)
    # Rolling median ignores infinities, whereas rolling count otherwise includes
    # them. Both operations must use exactly the same finite support.
    magnitude = np.where(np.isfinite(magnitude), magnitude, np.nan)
    series = pd.Series(magnitude, index=pd.to_timedelta(midpoint, unit="s"))
    return (
        series.rolling(**roll).median().to_numpy(),
        series.rolling(**roll).count().to_numpy(),
    )


def _clean_altitude(
    t: np.ndarray, alt: np.ndarray, fix_level: FixLevelThresholds
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Operational cuts on GNSS altitude (thesis, tab:cleaning).

    Three rules mark altitude missing without deleting horizontal fixes:

    * out of band: altitude outside the configured range;
    * sustained excess: both adjacent local magnitudes from :func:`local_vz`
      exceed the configured vertical-speed cutoff;
    * out-and-back spike: both adjacent per-step magnitudes exceed that cutoff
      and their signed velocities have opposite signs.

    The sustained rule tests the neighbourhood, without additionally requiring the
    fix's own increments to be excessive. If fewer than ``vz_min_window_fixes``
    finite steps support a median, its per-step magnitude stands in. Such windows
    occur at sparse cadence, record boundaries, or around missing altitudes; this
    fallback is less robust than the populated-window test.

    A threshold exceedance does not identify its cause: sustained real motion and
    GNSS noise can both trigger censoring. Unreturned isolated level shifts are
    counted separately, without correction; their treatment remains an explicit
    methodological decision (sec:fixlevel).

    Returns:
        ``(out_of_band, vz_sustained, vz_spike, n_runs, n_level_shifts)``: three boolean
        arrays over the fixes, the number of distinct sustained runs the second one
        censored, and the number of unreturned single steps.
    """
    out_of_band = (alt < fix_level.min_altitude_m) | (alt > fix_level.max_altitude_m)
    v_z = step_vz(t, alt)
    excess = np.abs(v_z) > fix_level.max_vertical_speed_mps
    excess = np.nan_to_num(excess, nan=False).astype(bool)

    # The sustained rule, on the local median; the per-step value stands in wherever the
    # window is too thin to estimate one.
    median_vz, populated = local_vz(t, v_z, fix_level.vz_window_s)
    thin = populated < fix_level.vz_min_window_fixes
    local = np.where(thin, np.abs(v_z), median_vz)
    with np.errstate(invalid="ignore"):
        bad_step = local > fix_level.max_vertical_speed_mps
    bad_step = np.nan_to_num(bad_step, nan=False).astype(bool)
    sustained = np.zeros(alt.size, dtype=bool)
    if bad_step.size >= 2:
        sustained[1:-1] = bad_step[:-1] & bad_step[1:]
    # How many distinct stretches that censored, not how many fixes: the fix count is
    # already in the report, and what a reader needs beside it is whether the removals
    # are one long event or a scatter of short ones.
    edges = np.flatnonzero(np.diff(np.concatenate([[False], sustained, [False]])))
    n_runs = int(edges.size // 2)

    spike = np.zeros(alt.size, dtype=bool)
    if excess.size >= 2:
        both = excess[:-1] & excess[1:]
        opposite = np.sign(v_z[:-1]) * np.sign(v_z[1:]) < 0
        spike[1:-1] = both & opposite
    # The two rules can agree on a fix -- a spike sitting inside a sustained stretch --
    # and the sustained verdict wins, so the audit never counts one censoring twice.
    spike &= ~sustained
    # A level shift: an isolated excessive step that the record does not come back from
    # within a few samples. "Comes back" is measured against the step itself rather than
    # against an absolute tolerance, so the test scales with the size of the jump.
    isolated = excess.copy()
    if excess.size >= 2:
        isolated[:-1] &= ~excess[1:]
        isolated[1:] &= ~excess[:-1]
    n_level_shifts = 0
    for i in np.flatnonzero(isolated):
        jump = abs(alt[i + 1] - alt[i])
        window = alt[i + 1 : min(i + 6, alt.size)]
        window = window[np.isfinite(window)]
        if not window.size or np.min(np.abs(window - alt[i])) > 0.3 * jump:
            n_level_shifts += 1
    return out_of_band & np.isfinite(alt), sustained, spike, n_runs, n_level_shifts


def clean_flight(
    fixes: pd.DataFrame,
    fix_level: FixLevelThresholds,
    *,
    discipline: str,
    baro_witness: bool = False,
) -> Cleaned:
    """Run stage (ii) over one flight, in the fixed detector order.

    Args:
        fixes: The flight after stage (i): columns ``t``, ``lat``, ``lon``, ``alt``,
            ``valid``, plus anything carried. ``t`` may hold backward steps and
            duplicate
            seconds -- repairing them is this stage's first job, not the parser's.
        fix_level: The adopted thresholds.
        discipline: Key into ``max_horizontal_speed_mps`` (``"paragliders"``,
            ``"hang gliders"``): the two types have markedly different speed envelopes,
            so one shared bound is either too loose for the slower or clips the faster.
        baro_witness: Whether this flight's raw barometric channel is usable as the
            frozen-lock witness (stage (i) decides). Local completeness is also
            required. With ``False``, exact repeats or sustained recorder declarations
            can still witness a collapsed run.

    Returns:
        The :class:`Cleaned` record.

    Raises:
        ValueError: If a required column is missing or the discipline has no bound.
    """
    required = ["t", "lat", "lon", "alt", "valid"]
    missing = [c for c in required if c not in fixes.columns]
    if missing:
        raise ValueError(f"the per-fix table is missing the column(s) {missing}")
    if discipline not in fix_level.max_horizontal_speed_mps:
        raise ValueError(
            f"no horizontal-speed bound for discipline {discipline!r}; known: "
            f"{sorted(fix_level.max_horizontal_speed_mps)}"
        )
    max_speed = fix_level.max_horizontal_speed_mps[discipline]
    n_raw = len(fixes)
    removed_t: list[np.ndarray] = []
    removed_reason: list[str] = []

    # (1) Time defects -- backward steps first, so that the duplicate merge and every
    #     window below see a clock that only ever moves forward.
    work = fixes.reset_index(drop=True)
    # Before anything reads a longitude componentwise (see :func:`unwrap_longitude`).
    work["lon"] = unwrap_longitude(work["lon"].to_numpy(dtype=float))
    t = work["t"].to_numpy(dtype=float)
    if t.size and not np.all(np.diff(t) >= 0):
        keep = longest_non_decreasing(t)
        dropped = np.setdiff1d(np.arange(t.size), keep)
        removed_t.append(t[dropped])
        removed_reason += [REMOVED_BACKWARD_TIME] * dropped.size
        work = work.iloc[keep].reset_index(drop=True)
    work, n_merged = _merge_duplicate_seconds(work)

    t = work["t"].to_numpy(dtype=float)
    lat = work["lat"].to_numpy(dtype=float)
    lon = work["lon"].to_numpy(dtype=float)
    alt = work["alt"].to_numpy(dtype=float)
    # The raw barometric channel, for the frozen-lock witness alone. The IGC zero means
    # "absent" (see altchannel), so it must not read as a measurement of sea level.
    baro = (
        work["baro_alt"].to_numpy(dtype=float)
        if "baro_alt" in work.columns
        else np.full(alt.size, np.nan)
    )
    baro = np.where(baro == 0.0, np.nan, baro)
    valid = work["valid"].to_numpy(dtype=bool)
    if t.size < 2:
        return _empty_result(work, n_raw, n_merged, removed_t, removed_reason)

    # (2) Position outliers: the identifier flags, the speed bound corroborates.
    flagged = hampel_flags(t, lat, lon, fix_level)
    spike, split, _ = _scan_positions(
        t, lat, lon, flagged, max_speed, fix_level.hampel_window_s
    )

    # (3) Frozen-lock runs, on the fixes the spike pass leaves standing.
    frozen, frozen_split = _frozen_runs(
        t, lat, lon, alt, baro, valid, fix_level, baro_witness
    )
    frozen &= ~spike
    split |= frozen_split

    # (4) Altitude: the absolute band and the local vertical-speed test, marking
    #     missing and never deleting.
    out_of_band, vz_sustained, vz_spike, n_vz_runs, n_level_shifts = _clean_altitude(
        t, alt, fix_level
    )
    invalidated = out_of_band | vz_sustained | vz_spike
    # A V flag is the recorder certifying that its own GNSS altitude is degraded. The
    # adopted channel is the GNSS one for every flight now (sec:altchannel), so this is
    # the corrupt-altitude case everywhere -- where it used to reach only the minority
    # that had fallen back to GNSS.
    invalidated |= ~valid

    deleted = spike | frozen
    # The postcondition, applied where it can hold: after *every* pass that deletes.
    # Enforcing it inside the position scan would be too early -- the frozen-lock pass
    # runs next and its deletions create new adjacencies of their own, one of which can
    # be an impossible step no rule has yet declared a boundary.
    n_boundaried = _boundary_impossible_steps(t, lat, lon, deleted, split, max_speed)
    removed_t += [t[spike], t[frozen]]
    removed_reason += [REMOVED_POSITION_SPIKE] * int(spike.sum())
    removed_reason += [REMOVED_FROZEN_RUN] * int(frozen.sum())

    out = work.loc[~deleted].copy()
    out["lon"] = wrap_longitude(out["lon"].to_numpy(dtype=float))
    out["alt"] = np.where(invalidated[~deleted], np.nan, alt[~deleted])
    out["hampel_flagged"] = flagged[~deleted]
    out["alt_invalidated"] = invalidated[~deleted]
    out["split_before"] = split[~deleted]
    out = out.reset_index(drop=True)

    report = CleaningReport(
        n_fix_raw=n_raw,
        n_fix_clean=len(out),
        n_merged_duplicates=n_merged,
        n_removed_backward=removed_reason.count(REMOVED_BACKWARD_TIME),
        n_removed_spike=int(spike.sum()),
        n_removed_frozen=int(frozen.sum()),
        n_alt_out_of_band=int((out_of_band & ~deleted).sum()),
        n_alt_vz_sustained=int((vz_sustained & ~deleted).sum()),
        n_alt_vz_spike=int((vz_spike & ~deleted).sum()),
        n_flagged_kept=int((flagged & ~deleted).sum()),
        n_splits=int(split[~deleted].sum()),
        n_vz_runs=n_vz_runs,
        n_alt_level_shift=n_level_shifts,
        split_jump_max_m=_largest_boundary_jump(out),
        n_v_flag=int((~valid).sum()),
        n_boundaried=n_boundaried,
    )
    removed = pd.DataFrame(
        {
            "t": np.concatenate(removed_t) if removed_t else np.empty(0),
            "reason": removed_reason,
        }
    )
    return Cleaned(fixes=out, removed=removed, report=report)


def _merge_duplicate_seconds(fixes: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Collapse fixes sharing one UTC second to a single fix at that second's centroid.

    Componentwise and unweighted: the mean of the latitudes and longitudes and the mean
    of the altitude over the members that have one. The ``valid`` flag of the merged fix
    is the logical *and* of the members' -- equivalently, the ``V`` flag is their
    logical *or* -- so a degraded member cannot be averaged away.

    A centroid rather than "keep the first": keeping the first privileges an arbitrary
    member and, for a genuine 2 Hz logger, would silently halve the cadence, whereas the
    centroid is order-independent and the sub-second resolution it discards is discarded
    anyway once the flight is resampled. One dilution case is accepted: if one member is
    itself a position spike, the centroid halves its amplitude before the outlier pass
    sees it. A diluted spike may escape a later threshold; smoothing does not
    establish a bound on the resulting position error.
    """
    t = fixes["t"].to_numpy(dtype=float)
    if t.size < 2 or not (np.diff(t) == 0).any():
        return fixes, 0
    # Zero is the raw IGC absence sentinel, not a pressure reading. Normalise before
    # averaging duplicates; {0, 1000} must yield the one available reading, not 500 m.
    # Two missing members stay missing, while an observed member supplies the witness
    # at their shared timestamp without temporal interpolation.
    work = fixes.copy()
    if "baro_alt" in work:
        work["baro_alt"] = work["baro_alt"].mask(work["baro_alt"] == 0.0)
    grouped = work.groupby("t", sort=True)
    merged = grouped.mean(numeric_only=True)
    if "valid" in fixes.columns:
        merged["valid"] = grouped["valid"].all()
    merged = merged.reset_index()
    return merged[[c for c in fixes.columns if c in merged.columns]], t.size - len(
        merged
    )


def _empty_result(
    work: pd.DataFrame,
    n_raw: int,
    n_merged: int,
    removed_t: list[np.ndarray],
    removed_reason: list[str],
) -> Cleaned:
    """A flight left with fewer than two fixes: nothing to test, nothing to detect."""
    out = work.copy()
    if "lon" in out.columns and len(out):
        out["lon"] = wrap_longitude(out["lon"].to_numpy(dtype=float))
    for column in CLEANING_COLUMNS:
        out[column] = np.zeros(len(out), dtype=bool)
    return Cleaned(
        fixes=out,
        removed=pd.DataFrame(
            {
                "t": np.concatenate(removed_t) if removed_t else np.empty(0),
                "reason": removed_reason,
            }
        ),
        report=CleaningReport(
            n_fix_raw=n_raw,
            n_fix_clean=len(out),
            n_merged_duplicates=n_merged,
            n_removed_backward=len(removed_reason),
            n_removed_spike=0,
            n_removed_frozen=0,
            n_alt_out_of_band=0,
            n_alt_vz_sustained=0,
            n_alt_vz_spike=0,
            n_flagged_kept=0,
            n_splits=0,
            n_vz_runs=0,
            n_alt_level_shift=0,
            split_jump_max_m=0.0,
            n_v_flag=0,
            n_boundaried=0,
        ),
    )


def integrity_gate(
    cleaned: Cleaned,
    t_on: float,
    t_off: float,
    fix_level: FixLevelThresholds,
) -> tuple[float, str | None]:
    """The flight-level integrity gate, over the airborne window (sec:fixlevel).

    Cleaning acts fix by fix, but it also keeps count of how much it had to do. Past a
    small fraction the flight is being *rebuilt* rather than cleaned, and is dropped
    whole. The counters are taken over the airborne window on purpose: defects in a
    ground phase that trimming removes anyway must not condemn the flight -- which is
    why this is called after stage (iii) and not inside :func:`clean_flight`.

    Merged duplicates are not counted against the flight. A merge repairs the record
    without losing a sample of the path, and counting it would drop every genuine 2 Hz
    logger for having recorded too well.

    Args:
        cleaned: The record :func:`clean_flight` returned.
        t_on: Start of the airborne window, in the flight's own clock.
        t_off: End of the airborne window.
        fix_level: The adopted thresholds.

    Returns:
        ``(fraction, drop_reason)``: the deleted-plus-invalidated share of the airborne
        fixes, and :data:`DROP_INTEGRITY` when it exceeds the gate.
    """
    kept = cleaned.fixes["t"].to_numpy(dtype=float)
    in_window = (kept >= t_on) & (kept <= t_off)
    removed_t = cleaned.removed["t"].to_numpy(dtype=float)
    removed_in_window = int(((removed_t >= t_on) & (removed_t <= t_off)).sum())
    invalidated = int(
        (cleaned.fixes["alt_invalidated"].to_numpy(dtype=bool) & in_window).sum()
    )
    n_airborne_raw = int(in_window.sum()) + removed_in_window
    if n_airborne_raw == 0:
        return 0.0, None
    fraction = (removed_in_window + invalidated) / n_airborne_raw
    reason = DROP_INTEGRITY if fraction > fix_level.integrity_max_fraction else None
    return fraction, reason


def _boundary_impossible_steps(
    t: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    deleted: np.ndarray,
    split: np.ndarray,
    max_speed_mps: float,
) -> int:
    """Make every impossible step between two surviving fixes a segment boundary.

    Before projection and smoothing, each surviving raw-geodetic step above the
    configured absolute speed bound must become a segment boundary. Whatever the
    detectors
    resolved, a step they leave standing past that bound is a transition whose course is
    unknown, which is what a long gap is, so it is marked as one (sec:fixlevel).

    Stated and enforced as a property of the output rather than as one more case inside
    a detector, so that it holds by construction instead of by the detectors having
    enumerated every way a track can be corrupt -- and applied after all of them, since
    each deletion creates adjacencies the previous pass never saw.

    Args:
        t: Fix times.
        lat: Latitudes in degrees.
        lon: Longitudes in degrees.
        deleted: Which fixes the detectors removed.
        split: The split markers, updated in place.
        max_speed_mps: The discipline's absolute horizontal-speed bound.

    Returns:
        How many boundaries this had to add that no detector had marked. Returned rather
        than kept quiet: a repair made in silence is a diagnosis lost, and a large count
        would indict the detectors rather than the data.
    """
    surviving = np.flatnonzero(~deleted)
    if surviving.size < 2:
        return 0
    impossible = (
        _step_speeds(t[surviving], lat[surviving], lon[surviving]) > max_speed_mps
    )
    landing = surviving[1:][impossible]
    added = int((~split[landing]).sum())
    split[landing] = True
    return added
