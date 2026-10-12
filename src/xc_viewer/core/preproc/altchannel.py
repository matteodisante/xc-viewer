"""Stage (i): GNSS-altitude eligibility and barometric-witness eligibility.

Every admitted flight uses the numerical GNSS altitude field. Presence and full-flight
range are operational admission criteria, not proof of local accuracy or sensor health.
The barometer has separate eligibility criteria and supports the frozen-position and
interior-ground detectors; each candidate additionally requires complete local pairing.
Missing GNSS values remain missing until flagged reconstruction at resampling.

GNSS fields are not yet harmonized across geoid/ellipsoid recorder conventions, so
using one named channel does not establish a common geodetic altitude datum.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ..config import AltChannelThresholds

# The flight-level verdict when the adopted channel is unusable.
DROP_NO_ALTITUDE = "no_usable_altitude_channel"

# Zero is treated as the missing-field sentinel. A genuine rounded zero is
# indistinguishable in this field and is also marked missing. The number of affected
# real fixes is not bounded by one per flight.
_ABSENT_ALT_M = 0.0


@dataclass(frozen=True)
class AltChannel:
    """One flight's altitude verdict, and the evidence for it.

    Attributes:
        gnss_present_frac: Share of fixes carrying a non-zero GNSS altitude.
        gnss_range_m: Total GNSS range over the flight, the liveness statistic (``0``
            when the channel is absent).
        baro_witness: Whether the raw barometric channel is present and alive enough
            to witness a frozen-lock run with. Diagnostic only: it selects a *test*,
            never a measured quantity.
        baro_present_frac: Share of fixes carrying a non-zero barometric altitude, the
            witness evidence and the archive census.
        baro_range_m: Total barometric range over the flight.
        n_missing: Fixes with no value on the adopted channel, left missing here and
            restored at resampling.
        drop_reason: :data:`DROP_NO_ALTITUDE` when the GNSS channel fails either test,
            and ``None`` when the flight is admitted.
    """

    gnss_present_frac: float
    gnss_range_m: float
    baro_witness: bool
    baro_present_frac: float
    baro_range_m: float
    n_missing: int
    drop_reason: str | None


def _presence_and_range(values: np.ndarray) -> tuple[float, float]:
    """The presence fraction and the total range of one raw channel.

    ``nan`` is what a blank or unusable altitude field decodes to, and ``nan != 0``
    is ``True``, so a channel written entirely blank used to be counted as fully
    present and adopted. Four flights in the archive reached the analysis dataset that
    way.
    """
    present = np.isfinite(values) & (values != _ABSENT_ALT_M)
    fraction = float(present.mean()) if values.size else 0.0
    span = (
        float(values[present].max() - values[present].min()) if present.any() else 0.0
    )
    return fraction, span


def adopt_alt_channel(
    fixes: pd.DataFrame, alt_channel: AltChannelThresholds
) -> tuple[pd.DataFrame, AltChannel]:
    """Run stage (i) over one flight: gate the GNSS channel, rate the barometric one.

    Args:
        fixes: One flight's parsed table (``xc_viewer.core.igc.parse_igc``),
        with
            ``baro_alt`` and ``gnss_alt`` columns in metres and zero where absent.
        alt_channel: The adopted presence and liveness thresholds.

    Returns:
        ``(fixes, channel)``: the input table with an ``alt`` column added -- the
        GNSS channel, ``nan`` where it has no value at that fix -- and the
        :class:`AltChannel` record. The raw channels are left in place: the fix-level
        cleaning still needs the barometer as a frozen-lock witness and the recorder's
        own declarations besides, and only the local-frame conversion of stage (v)
        drops them.

    Raises:
        ValueError: If a required column is missing.
    """
    missing = [c for c in ("baro_alt", "gnss_alt") if c not in fixes.columns]
    if missing:
        raise ValueError(f"the parsed table is missing the column(s) {missing}")

    gnss = fixes["gnss_alt"].to_numpy(dtype=float)
    baro = fixes["baro_alt"].to_numpy(dtype=float)
    gnss_frac, gnss_range = _presence_and_range(gnss)
    baro_frac, baro_range = _presence_and_range(baro)

    drop_reason = (
        DROP_NO_ALTITUDE
        if gnss_frac < alt_channel.gnss_present_min
        or gnss_range < alt_channel.gnss_min_range_m
        else None
    )
    baro_witness = (
        baro_frac >= alt_channel.baro_witness_present_min
        and baro_range >= alt_channel.baro_witness_min_range_m
    )

    alt = gnss.copy()
    alt[alt == _ABSENT_ALT_M] = np.nan
    out = fixes.copy()
    out["alt"] = alt
    return out, AltChannel(
        gnss_present_frac=gnss_frac,
        gnss_range_m=gnss_range,
        baro_witness=baro_witness,
        baro_present_frac=baro_frac,
        baro_range_m=baro_range,
        n_missing=int(np.isnan(alt).sum()),
        drop_reason=drop_reason,
    )
