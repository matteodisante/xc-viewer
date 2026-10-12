"""Stage (vii): local polynomial smoothing and derivatives within each segment.

A cubic Savitzky--Golay fit returns position, velocity and acceleration in physical
units. The configured timescale determines a sample count, rounded upward to odd and
floored at five; the sample-to-sample window span is ``(w - 1) * dt``. This heuristic
is not the filter's spectral cutoff. Its effects on observables require validation
within cadence groups, especially when the five-sample floor determines the window.

``mode='interp'`` evaluates terminal polynomials inside their fitted edge windows,
without extrapolation outside the observations. ``edge`` identifies the output rows
with off-centre weights. ``z_derivative_reconstructed`` marks every output whose
vertical fit uses at least one reconstructed input altitude; the original
``z_reconstructed`` flag remains the resampling-stage mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .resample import FIX_COLUMNS, Resampled

if TYPE_CHECKING:
    from ..config import SavgolParams

# The kinematic columns this stage adds, in order: velocity then acceleration, per ENU
# component. `E`, `N` and `z` are overwritten in place by their smoothed values -- only
# the filter's outputs are stored, never both a raw and a filtered copy of one
# quantity.
KINEMATIC_COLUMNS = ["v_E", "v_N", "v_z", "a_E", "a_N", "a_z"]

# Columns of the per-fix table this stage produces, in order (carried-through columns
# follow). `edge` joins `interpolated` as the second per-sample caution flag.
SMOOTHED_COLUMNS = [
    *FIX_COLUMNS,
    *KINEMATIC_COLUMNS,
    "edge",
    "z_derivative_reconstructed",
]

# The smallest window this stage will use, whatever the cadence says (sec:savgol): five
# samples, the least on which a cubic fit is meaningfully determined.
MIN_WINDOW = 5

# A segment the window does not fit into cannot be smoothed at all. The segment gate of
# stage (vi) covers this at every common cadence -- 90 s carries at least five samples
# up to dt = 22.5 s -- so this bites only in the thin tail of very slow loggers.
DROP_WINDOW_TOO_WIDE = "shorter_than_smoothing_window"


@dataclass(frozen=True)
class SavgolWindows:
    """The filter as it was configured for one flight: what ``flights_meta`` records.

    Attributes:
        horizontal: Window in samples for ``E`` and ``N``.
        vertical: Window in samples for ``z``. Equal to ``horizontal`` under the
            adopted working timescales; the two stay
            separate keys because they answer to different noise floors in principle,
            and only happen to agree in this archive.
        polyorder: The polynomial order, ``p = 3``.
    """

    horizontal: int
    vertical: int
    polyorder: int


@dataclass(frozen=True)
class Smoothed:
    """One flight after stage (vii): the pipeline's final per-fix table.

    Attributes:
        fixes: One row per grid point, columns :data:`SMOOTHED_COLUMNS` followed by the
            carried-through columns. ``E``, ``N`` and ``z`` are now the smoothed
            positions; the velocities and accelerations are the same fit's first and
            second derivatives, in m/s and m/s^2.
        segments: The segment table of stage (vi), with any segment too short for the
            window now marked dropped (:data:`DROP_WINDOW_TOO_WIDE`).
        windows: The filter configuration this flight was run with, or ``None`` for a
            flight that reached this stage already dropped and so was never filtered.
        drop_reason: ``None`` when at least one segment survived, otherwise
            :data:`DROP_WINDOW_TOO_WIDE` -- or, unchanged, the reason stage (vi) had
            already recorded.
    """

    fixes: pd.DataFrame
    segments: pd.DataFrame
    windows: SavgolWindows | None
    drop_reason: str | None


def savgol_window(tau_c_s: float, dt_s: float, polyorder: int) -> int:
    """The smoothing window in samples, from a noise timescale and a cadence.

    ``tau_c / dt`` rounded up to the nearest odd integer, floored at the smallest
    window a fit of this order can carry (:data:`MIN_WINDOW` for the adopted ``p = 3``,
    and ``p + 2`` rounded up to odd for any other order -- one sample more than the
    ``p + 1`` that would determine the polynomial exactly, so it stays a least-squares
    fit).

    Args:
        tau_c_s: The channel's smoothing timescale, in seconds.
        dt_s: The flight's native sampling interval, in seconds.
        polyorder: The polynomial order.

    Returns:
        An odd window length in samples.
    """
    window = int(np.ceil(tau_c_s / dt_s))
    window += window % 2 == 0  # up to the nearest odd integer: it needs a centre sample
    floor = max(MIN_WINDOW, polyorder + 2)
    floor += floor % 2 == 0
    return max(window, floor)


def savgol_windows(savgol: SavgolParams, dt_s: float) -> SavgolWindows:
    """The per-flight filter configuration (thesis, sec:savgol item (iii)).

    Args:
        savgol: The adopted order and the two smoothing timescales.
        dt_s: The flight's native sampling interval, in seconds.

    Returns:
        The :class:`SavgolWindows` for this flight.
    """
    return SavgolWindows(
        horizontal=savgol_window(savgol.tau_c_horizontal_s, dt_s, savgol.polyorder),
        vertical=savgol_window(savgol.tau_c_vertical_s, dt_s, savgol.polyorder),
        polyorder=savgol.polyorder,
    )


def smooth_segment(
    segment: pd.DataFrame, windows: SavgolWindows, dt_s: float
) -> pd.DataFrame:
    """Filter and differentiate one segment's three channels.

    Args:
        segment: The segment's grid points, with columns ``E``, ``N``, ``z`` (finite
            everywhere -- the completeness invariant of stage (vi)) on a uniform grid of
            step ``dt_s``, plus whatever else it carries.
        windows: The filter configuration.
        dt_s: The grid step, in seconds. It enters only here, as the ``delta`` that
            turns the fit's per-sample coefficients into physical units.

    Returns:
        The segment with ``E``, ``N``, ``z`` replaced by their smoothed values,
        :data:`KINEMATIC_COLUMNS` added, and the ``edge`` flag set on the samples the
        filter had to evaluate off-centre.

    Raises:
        ValueError: If the segment is shorter than the widest window, which
            ``mode='interp'`` cannot fit.
    """
    # Lazy, as elsewhere in this package: importing the module must not require the
    # optional `analysis` dependency group.
    from scipy.signal import savgol_filter

    widest = max(windows.horizontal, windows.vertical)
    if len(segment) < widest:
        raise ValueError(
            f"segment of {len(segment)} samples is shorter than the {widest}-sample "
            "smoothing window"
        )

    out = segment.copy()
    channels = {"E": windows.horizontal, "N": windows.horizontal, "z": windows.vertical}
    for channel, window in channels.items():
        values = segment[channel].to_numpy(dtype=float)
        for deriv, name in enumerate([channel, f"v_{channel}", f"a_{channel}"]):
            out[name] = savgol_filter(
                values,
                window_length=window,
                polyorder=windows.polyorder,
                deriv=deriv,
                delta=dt_s,
                # Fit the terminal half-windows on the edge window and evaluate
                # off-centre, rather than padding the segment with data it does not
                # have. The window never crosses a boundary (sec:savgol).
                mode="interp",
            )
    half = widest // 2
    edge = np.zeros(len(segment), dtype=bool)
    edge[:half] = True
    edge[len(segment) - half :] = True
    out["edge"] = edge
    reconstructed = (
        segment["z_reconstructed"].to_numpy(dtype=bool)
        if "z_reconstructed" in segment
        else np.zeros(len(segment), dtype=bool)
    )
    # Interior fits use a centred window; terminal fits use the first/last full
    # window, exactly as scipy.signal.savgol_filter(mode="interp"). The union
    # supports position and both vertical derivatives, including zero coefficients
    # that differ between derivative orders.
    starts = np.clip(
        np.arange(len(segment)) - windows.vertical // 2,
        0,
        len(segment) - windows.vertical,
    )
    prefix = np.r_[0, np.cumsum(reconstructed, dtype=np.int64)]
    out["z_derivative_reconstructed"] = (
        prefix[starts + windows.vertical] - prefix[starts]
    ) > 0
    return out


def smooth_flight(resampled: Resampled, savgol: SavgolParams) -> Smoothed:
    """Run stage (vii) over one resampled flight, segment by segment.

    Args:
        resampled: The output of stage (vi). Its ``dt_s`` is the grid step, and its
            ``segments`` table is carried forward and updated.
        savgol: The adopted order and smoothing timescales.

    Returns:
        The :class:`Smoothed` record. A segment the window does not fit into is dropped
        alone, with its reason written into the segment table, exactly as at stage (vi);
        the flight is lost only if no segment survives. A flight that arrives already
        dropped passes straight through, keeping the reason it came with.
    """
    if resampled.drop_reason is not None:
        return Smoothed(
            fixes=_empty_smoothed(resampled.fixes),
            segments=resampled.segments,
            windows=None,
            drop_reason=resampled.drop_reason,
        )
    windows = savgol_windows(savgol, resampled.dt_s)
    segments = resampled.segments.copy()
    widest = max(windows.horizontal, windows.vertical)

    smoothed: list[pd.DataFrame] = []
    for segment_id, group in resampled.fixes.groupby("segment_id", sort=True):
        if len(group) < widest:
            row = segments["segment_id"] == segment_id
            segments.loc[row, "kept"] = False
            segments.loc[row, "n_fix"] = 0
            segments.loc[row, "drop_reason"] = DROP_WINDOW_TOO_WIDE
            continue
        smoothed.append(smooth_segment(group, windows, resampled.dt_s))

    if not smoothed:
        return Smoothed(
            fixes=_empty_smoothed(resampled.fixes),
            segments=segments,
            windows=windows,
            drop_reason=DROP_WINDOW_TOO_WIDE,
        )
    return Smoothed(
        fixes=_ordered(pd.concat(smoothed, ignore_index=True)),
        segments=segments,
        windows=windows,
        drop_reason=None,
    )


def _empty_smoothed(fixes: pd.DataFrame) -> pd.DataFrame:
    """The output table of a flight that contributed no fixes, dtypes and all."""
    empty = fixes.iloc[:0].copy()
    for name in KINEMATIC_COLUMNS:
        empty[name] = pd.Series(dtype="float64")
    empty["edge"] = pd.Series(dtype="bool")
    empty["z_derivative_reconstructed"] = pd.Series(dtype="bool")
    return _ordered(empty)


def _ordered(fixes: pd.DataFrame) -> pd.DataFrame:
    """Put the table in :data:`SMOOTHED_COLUMNS` order, carried columns last."""
    carried = [c for c in fixes.columns if c not in SMOOTHED_COLUMNS]
    return fixes[[*SMOOTHED_COLUMNS, *carried]]
