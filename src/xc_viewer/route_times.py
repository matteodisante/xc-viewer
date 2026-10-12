"""Recover the dates of retained route endpoints from the original IGC clock."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time

import numpy as np
import pandas as pd

from .catalog_index import resolve_igc_path
from .core.igc import first_fix
from .thermal_daily import PARIS
from .thermal_index import _check_cancel


@dataclass(frozen=True)
class DepartureWindow:
    """One Paris civil day and a half-open wall-clock interval within that day."""

    day: date
    start: time
    minutes: int = 30

    def __post_init__(self):
        """Require minute precision and keep the entire interval on one day."""
        if self.start.tzinfo or self.start.second or self.start.microsecond:
            raise ValueError("Departure start must be a local hour and minute")
        if not 1 <= self.minutes <= 1440 - self.start_minute:
            raise ValueError("Departure interval must stay within the selected day")

    @property
    def start_minute(self):
        """Wall-clock minutes since midnight."""
        return self.start.hour * 60 + self.start.minute

    @property
    def end_label(self):
        """Exclusive local end, allowing 24:00 at the end of the selected day."""
        end = self.start_minute + self.minutes
        return f"{end // 60:02}:{end % 60:02}"

    @property
    def label(self):
        """Human-readable interval with an explicit timezone and boundary."""
        return (
            f"{self.day:%d/%m/%Y} {self.start:%H:%M} to {self.end_label} "
            "Europe/Paris (end excluded)"
        )


def local_departures(flights):
    """Convert recovered UTC timestamps to Paris dates, retaining missing clocks."""
    return pd.to_datetime(
        flights.departure_utc, unit="s", utc=True, errors="coerce"
    ).dt.tz_convert(PARIS)


def filter_departures(flights, window):
    """Filter the full cohort before sampling; missing clocks cannot match a day.

    Wall-clock filtering includes both occurrences of a repeated autumn hour.
    Arrival times do not affect membership, including arrivals the next day.
    """
    if window is None:
        return flights
    local = local_departures(flights)
    seconds = (
        local.dt.hour * 3600
        + local.dt.minute * 60
        + local.dt.second
        + local.dt.microsecond / 1e6
    )
    mask = (
        local.dt.date.eq(window.day)
        & (seconds >= window.start_minute * 60)
        & (seconds < (window.start_minute + window.minutes) * 60)
    )
    return flights.loc[mask].copy()


def with_flight_times(selected, disciplines, *, progress=lambda _: None, cancel=None):
    """Attach UTC endpoints without mistaking relative archive times for dates.

    Cleaning stores seconds relative to the start of the trimmed airborne window.
    Restore that offset and the UTC origin of the first parsed IGC fix. Missing
    headers, files or trim offsets leave dates unavailable; catalogue dates are
    used only to locate the raw file, never to invent a clock origin.
    """
    result = selected.assign(departure_utc=np.nan, arrival_utc=np.nan)
    for disc in disciplines:
        _check_cancel(cancel)
        rows = result.loc[result.discipline == disc.name]
        if rows.empty:
            continue
        cfg = disc.config()
        catalog_path = getattr(cfg, "catalog_path", None)
        if catalog_path is None:
            continue
        try:
            meta = pd.read_parquet(
                cfg.derived_dir / "flights_meta.parquet",
                columns=["flight_id", "ground_phase_start_s"],
            )
            meta["flight_id"] = meta.flight_id.astype(str)
            offsets = (
                meta.drop_duplicates("flight_id", keep="last")
                .set_index("flight_id")
                .ground_phase_start_s
            )
            catalog = (
                pd.read_csv(
                    catalog_path,
                    usecols=["flight_id", "season_year", "date"],
                    dtype={"flight_id": str},
                )
                .drop_duplicates("flight_id", keep="last")
                .set_index("flight_id", drop=False)
            )
        except (OSError, KeyError, ValueError):
            continue
        for number, (index, row) in enumerate(rows.iterrows()):
            _check_cancel(cancel)
            if number % 25 == 0:
                progress(f"{disc.name}: reading flight dates {number + 1}/{len(rows)}")
            offset = offsets.get(row.flight_id, np.nan)
            if pd.isna(offset) or not np.isfinite(offset):
                continue
            if row.flight_id not in catalog.index:
                continue
            try:
                path = resolve_igc_path(disc, catalog.loc[row.flight_id])
                first = first_fix(path, include_utc=True) if path else None
            except (OSError, ValueError, TypeError, OverflowError):
                continue
            if first is None or not np.isfinite(first["start_utc"]):
                continue
            origin = first["start_utc"] + offset
            result.loc[index, ["departure_utc", "arrival_utc"]] = (
                origin + row.t0,
                origin + row.t1,
            )
    return result
