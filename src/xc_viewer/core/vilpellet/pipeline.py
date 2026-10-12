"""Running the transcribed segmenter over a flight, and collapsing it into runs.

The unit of work is one preprocessing segment: it is contiguous, uniformly sampled and
already the unit the rest of the repository treats as an independent trajectory, so no
feature window and no state transition crosses a boundary the cleaning drew.  A segment
whose cadence the sample-count windows cannot serve is returned unlabelled with the
reason recorded, never decoded anyway.

The output carries the same four columns the Chapter 4 segmenter's viewer path
produces -- ``phase``, ``phase_reason``, ``track_run``, ``phase_run`` -- so the two
segmentations are interchangeable everywhere a labelled trajectory is drawn.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import VilpelletConfig
from .features import build_observation_frame, observation_matrix
from .model import rolling_majority, viterbi

UNCLASSIFIED = "unclassified"

# Why a fix carries no phase.  `classified` is the complement; the rest are the
# eligibility gate and the author's optional run selection.
REASON_CLASSIFIED = "classified"
REASON_SLOW_CADENCE = "cadence_above_gate"
REASON_NON_UNIFORM = "non_increasing_or_repeated_time"
REASON_SHORT_SEGMENT = "segment_shorter_than_persistence_window"
REASON_SHORT_FLIGHT = "flight_shorter_than_author_guard"
REASON_DROPPED_TAIL = "dropped_flight_tail"
REASON_SHORT_RUN = "run_shorter_than_minimum"
REASON_UNCONFIRMED_SEARCH = "search_not_followed_by_climb"

PHASE_SEGMENT_COLUMNS = [
    "source",
    "flight_id",
    "segment_id",
    "phase",
    "t_start",
    "t_end",
    "duration_s",
    "n_fixes",
    "left_censored",
    "right_censored",
]


@dataclass(frozen=True)
class VilpelletTrack:
    """One flight labelled by the transcribed segmenter.

    Attributes:
        fixes: Every input fix, with ``phase``, ``phase_reason``, ``track_run`` and
            ``phase_run`` added.
        eligible: Whether any fix was decoded at all.
        reasons: Fix counts per ``phase_reason`` value.
        discipline: The discipline whose straightness threshold was used.
        alpha_straight_rad: That threshold, in radians per fix.
    """

    fixes: pd.DataFrame
    eligible: bool
    reasons: dict[str, int]
    discipline: str
    alpha_straight_rad: float


def segment_eligibility(segment: pd.DataFrame, config: VilpelletConfig) -> str | None:
    """Why one segment cannot be decoded, or ``None`` when it can.

    Args:
        segment: One contiguous segment with a ``t`` column in seconds.
        config: The loaded protocol.

    Returns:
        A ``phase_reason`` value, or ``None``.
    """
    gate = config.eligibility
    t = segment["t"].to_numpy(dtype=float)
    if t.size < config.features.persistence_fixes:
        return REASON_SHORT_SEGMENT
    dt = np.diff(t)
    if gate.require_increasing_times and not np.all(dt > 0.0):
        return REASON_NON_UNIFORM
    if gate.require_unique_times and pd.Series(t).duplicated().any():
        return REASON_NON_UNIFORM
    if not np.isfinite(dt).all() or float(dt.mean()) >= gate.max_mean_dt_s:
        return REASON_SLOW_CADENCE
    return None


def segment_track(
    segment: pd.DataFrame, config: VilpelletConfig, *, alpha_straight_rad: float
) -> pd.DataFrame:
    """Decode one eligible, contiguous segment.

    Args:
        segment: One segment, sorted by ``t``, carrying either ``E``/``N``/``z`` or
            ``lat``/``lon``/``alt``.
        config: The loaded protocol.
        alpha_straight_rad: The discipline's straightness threshold.

    Returns:
        The feature frame with a ``phase_component`` column of integer component
        indices and a ``phase`` column of behavioural names, both after the rolling
        majority vote.
    """
    frame = build_observation_frame(
        segment, config, alpha_straight_rad=alpha_straight_rad
    )
    raw_path = viterbi(observation_matrix(frame), config.parameters)
    smoothed = rolling_majority(
        pd.Series(raw_path, index=frame.index), config.majority_fixes
    )
    frame["phase_component_raw"] = raw_path
    frame["phase_component"] = smoothed.to_numpy(dtype=np.int64)
    frame["phase"] = [
        config.state_name(component) for component in frame["phase_component"]
    ]
    return frame


def segment_flight(
    fixes: pd.DataFrame,
    config: VilpelletConfig,
    *,
    discipline: str,
    enforce_flight_guard: bool = True,
) -> VilpelletTrack:
    """Label every fix of one already-preprocessed flight.

    Args:
        fixes: The retained fixes of exactly one flight, with ``segment_id`` and ``t``.
        config: The loaded protocol.
        discipline: The discipline name whose straightness threshold applies.
        enforce_flight_guard: Apply the author's minimum-length guard on the whole
            flight.  Switching it off decodes a shorter flight that the guard would
            otherwise leave entirely unlabelled; it does not change any label.

    Returns:
        The :class:`VilpelletTrack`.

    Raises:
        ValueError: If the table holds more than one flight.
    """
    alpha = config.alpha_for(discipline)
    out = fixes.sort_values(["segment_id", "t"], kind="stable").reset_index(drop=True)
    if "flight_id" in out and out["flight_id"].nunique() > 1:
        raise ValueError("the Vilpellet segmenter labels one flight at a time")
    out["phase"] = UNCLASSIFIED
    out["phase_reason"] = REASON_SHORT_SEGMENT
    out["phase_component"] = pd.Series(pd.NA, index=out.index, dtype="Int64")

    short_flight = enforce_flight_guard and len(out) < config.eligibility.min_fixes
    if short_flight:
        out["phase_reason"] = REASON_SHORT_FLIGHT
    else:
        for _, positions in out.groupby("segment_id", sort=False).groups.items():
            index = pd.Index(positions)
            segment = out.loc[index]
            reason = segment_eligibility(segment, config)
            if reason is not None:
                out.loc[index, "phase_reason"] = reason
                continue
            decoded = segment_track(segment, config, alpha_straight_rad=alpha)
            out.loc[index, "phase"] = decoded["phase"].to_numpy()
            out.loc[index, "phase_component"] = decoded["phase_component"].to_numpy()
            out.loc[index, "phase_reason"] = REASON_CLASSIFIED

    out = apply_recommendations(out, config)
    out = add_run_columns(out)
    reasons = {str(k): int(v) for k, v in out["phase_reason"].value_counts().items()}
    return VilpelletTrack(
        fixes=out,
        eligible=bool((out["phase"] != UNCLASSIFIED).any()),
        reasons=reasons,
        discipline=discipline,
        alpha_straight_rad=alpha,
    )


def apply_recommendations(
    points: pd.DataFrame, config: VilpelletConfig
) -> pd.DataFrame:
    """Apply the author's optional run selection, in the order he states it.

    The tail is discarded first, because it changes which runs exist; the minimum run
    length is applied next; the search confirmation last, so that it reads the runs
    that actually survive.  Each rejected fix keeps its own reason, so the discarded
    time stays countable instead of merging into one unclassified total.

    Args:
        points: A labelled fix table.
        config: The loaded protocol.

    Returns:
        The same table, with rejected fixes returned to ``unclassified``.
    """
    advice = config.recommendations
    if not advice.active:
        return points
    out = points.copy()
    if advice.drop_tail_s > 0.0:
        t = out["t"].to_numpy(dtype=float)
        tail = t > (t.max() - advice.drop_tail_s)
        _reject(out, tail, REASON_DROPPED_TAIL)
    if advice.min_run_s > 0.0 or advice.search_requires_following_climb:
        runs = _run_ids(out)
        durations = out.groupby(runs)["t"].transform(lambda s: s.max() - s.min())
        if advice.min_run_s > 0.0:
            short = (durations < advice.min_run_s) & (out["phase"] != UNCLASSIFIED)
            _reject(out, short.to_numpy(), REASON_SHORT_RUN)
    if advice.search_requires_following_climb:
        runs = _run_ids(out)
        order = out.groupby(runs)["phase"].first()
        following = order.shift(-1)
        unconfirmed = order.index[(order == "search") & (following != "climb")]
        _reject(out, runs.isin(unconfirmed).to_numpy(), REASON_UNCONFIRMED_SEARCH)
    return out


def add_run_columns(points: pd.DataFrame) -> pd.DataFrame:
    """Add the ``track_run`` and ``phase_run`` line-grouping keys the viewer needs.

    ``track_run`` changes whenever physical continuity breaks: a new preprocessing
    segment, or a gap longer than 1.5 native steps.  ``phase_run`` additionally changes
    at every colour change, so two separate occurrences of one phase are never joined
    by a straight line across the intervening flight.

    Args:
        points: A labelled fix table sorted by ``segment_id`` then ``t``.

    Returns:
        The same table with both columns.
    """
    out = points.copy()
    out["track_run"] = 0
    offset = 0
    for _, group in out.groupby("segment_id", sort=False):
        t = group["t"].to_numpy(dtype=float)
        dt = np.diff(t)
        cadence = np.median(dt[dt > 0]) if np.any(dt > 0) else np.inf
        breaks = np.r_[True, (dt <= 0) | (dt > 1.5 * cadence)]
        runs = np.cumsum(breaks) - 1 + offset
        out.loc[group.index, "track_run"] = runs
        offset = int(runs[-1]) + 1
    starts = out["track_run"].ne(out["track_run"].shift()) | out["phase"].ne(
        out["phase"].shift()
    )
    out["phase_run"] = starts.cumsum() - 1
    return out


def phase_runs(points: pd.DataFrame, config: VilpelletConfig) -> pd.DataFrame:
    """Collapse a labelled fix table into one row per uninterrupted phase run.

    A run is censored on the side where it meets the boundary of its ``track_run``
    rather than a genuine phase change, so a duration study can exclude the runs whose
    true length was cut by a gap or by the end of the flight.

    Args:
        points: A labelled fix table carrying ``track_run`` and ``phase_run``.
        config: The loaded protocol; unused beyond documenting the producing model.

    Returns:
        A table with :data:`PHASE_SEGMENT_COLUMNS`.
    """
    del config
    labelled = points.loc[points["phase"] != UNCLASSIFIED]
    if labelled.empty:
        return pd.DataFrame(columns=PHASE_SEGMENT_COLUMNS)
    rows = []
    for run_id, run in labelled.groupby("phase_run", sort=True):
        track_run = int(run["track_run"].iloc[0])
        same_track = points.loc[points["track_run"] == track_run]
        rows.append(
            {
                "source": run["source"].iloc[0] if "source" in run else pd.NA,
                "flight_id": run["flight_id"].iloc[0] if "flight_id" in run else pd.NA,
                "segment_id": run["segment_id"].iloc[0],
                "phase": run["phase"].iloc[0],
                "t_start": float(run["t"].iloc[0]),
                "t_end": float(run["t"].iloc[-1]),
                "duration_s": float(run["t"].iloc[-1] - run["t"].iloc[0]),
                "n_fixes": len(run),
                "left_censored": bool(run.index[0] == same_track.index[0]),
                "right_censored": bool(run.index[-1] == same_track.index[-1]),
            }
        )
        del run_id
    return pd.DataFrame(rows, columns=PHASE_SEGMENT_COLUMNS)


def _reject(points: pd.DataFrame, mask: np.ndarray, reason: str) -> None:
    """Return the masked, currently classified fixes to ``unclassified`` in place."""
    selected = np.asarray(mask) & (points["phase"] != UNCLASSIFIED).to_numpy()
    points.loc[selected, "phase"] = UNCLASSIFIED
    points.loc[selected, "phase_component"] = pd.NA
    points.loc[selected, "phase_reason"] = reason


def _run_ids(points: pd.DataFrame) -> pd.Series:
    """Consecutive-run identifiers over ``segment_id`` and ``phase``."""
    changed = points["phase"].ne(points["phase"].shift()) | points["segment_id"].ne(
        points["segment_id"].shift()
    )
    return changed.cumsum()
