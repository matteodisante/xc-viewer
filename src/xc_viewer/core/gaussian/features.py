"""Windowed kinematic features of one preprocessing segment, on the decision grid.

Each feature is the exact integral of the piecewise-linear interpolant of a quantity
computed at native fixes, over a centred window, divided by the window width.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import GaussianConfig

FEATURE_COLUMNS = ["mean_v_z", "mean_v_h", "mean_abs_turn_rate", "turn_coherence"]
# A window whose support touches one of these flags has no features.
QUALITY_COLUMNS = ["z_reconstructed", "z_derivative_reconstructed", "edge"]


def window_integrals(
    t: np.ndarray, values: np.ndarray, left: np.ndarray, right: np.ndarray
) -> np.ndarray:
    """Integrals of the linear interpolant of ``values`` over ``[left, right]``."""
    cumulative = np.zeros(len(t))
    cumulative[1:] = np.cumsum(0.5 * (values[:-1] + values[1:]) * np.diff(t))

    def integral_to(query: np.ndarray) -> np.ndarray:
        i = np.clip(np.searchsorted(t, query, side="right") - 1, 0, len(t) - 2)
        elapsed = query - t[i]
        slope = (values[i + 1] - values[i]) / (t[i + 1] - t[i])
        at_query = values[i] + slope * elapsed
        return cumulative[i] + 0.5 * (values[i] + at_query) * elapsed

    return integral_to(right) - integral_to(left)


def segment_features(
    segment: pd.DataFrame, config: GaussianConfig
) -> pd.DataFrame | None:
    """One row per decision time of a segment, with its four features.

    Args:
        segment: One preprocessing segment of a cleaned flight.
        config: The segmenter settings.

    Returns:
        Columns ``t``, ``feature_edge``, ``quality_masked`` and
        :data:`FEATURE_COLUMNS`. Features are NaN where the window leaves the segment
        (``feature_edge``) or its support touches a flagged fix (``quality_masked``).
        ``None`` when the segment has under two fixes, a non-uniform clock, or a
        cadence slower than ``max_native_dt_s``.
    """
    ordered = segment.sort_values("t", kind="stable")
    t = ordered["t"].to_numpy(dtype=float)
    if t.size < 2:
        return None
    dt = np.diff(t)
    cadence = float(np.median(dt))
    uniform = np.all(dt > 0.0) and np.allclose(dt, cadence, rtol=1e-7, atol=1e-8)
    if not uniform or cadence > config.max_native_dt_s:
        return None

    step, half = config.decision_step_s, config.window_s / 2.0
    n = int(np.floor((t[-1] - t[0]) / step + 1e-9)) + 1
    times = t[0] + step * np.arange(n)
    feature_edge = (times - half < t[0]) | (times + half > t[-1])
    interior = np.flatnonzero(~feature_edge)
    left, right = times[interior] - half, times[interior] + half

    # The interpolated window ends also use the fixes just outside the window.
    first = np.maximum(0, np.searchsorted(t, left, side="right") - 1)
    last = np.minimum(len(t) - 1, np.searchsorted(t, right, side="left"))
    flagged = ordered[QUALITY_COLUMNS].to_numpy(dtype=bool).any(axis=1)
    flagged_before = np.concatenate(([0], np.cumsum(flagged)))
    unsafe = flagged_before[last + 1] - flagged_before[first] > 0
    quality_masked = np.zeros(n, dtype=bool)
    quality_masked[interior[unsafe]] = True

    v_e, v_n, v_z, a_e, a_n = (
        ordered[c].to_numpy(dtype=float) for c in ("v_E", "v_N", "v_z", "a_E", "a_N")
    )
    v_h_squared = v_e**2 + v_n**2
    v_h = np.sqrt(v_h_squared)
    turn_rate = np.zeros_like(v_h)
    moving = v_h >= config.min_horizontal_speed_mps
    turn_rate[moving] = (
        v_e[moving] * a_n[moving] - v_n[moving] * a_e[moving]
    ) / v_h_squared[moving]

    features = np.full((n, len(FEATURE_COLUMNS)), np.nan)
    if np.any(~unsafe):
        lo, hi = left[~unsafe], right[~unsafe]
        signed = window_integrals(t, turn_rate, lo, hi)
        absolute = window_integrals(t, np.abs(turn_rate), lo, hi)
        coherence = np.divide(
            np.abs(signed),
            absolute,
            out=np.zeros_like(absolute),
            where=absolute > np.finfo(float).eps,
        )
        features[interior[~unsafe]] = np.column_stack(
            [
                window_integrals(t, v_z, lo, hi) / config.window_s,
                window_integrals(t, v_h, lo, hi) / config.window_s,
                absolute / config.window_s,
                np.clip(coherence, 0.0, 1.0),
            ]
        )

    frame = pd.DataFrame(
        {"t": times, "feature_edge": feature_edge, "quality_masked": quality_masked}
    )
    frame[FEATURE_COLUMNS] = features
    return frame
