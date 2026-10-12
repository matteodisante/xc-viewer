"""Disjoint launch cohorts on the existing French 5 km endpoint lattice."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from uuid import uuid4

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import thermal_index
from .route_index import PAIR_COLUMNS, _endpoint_rows
from .route_scene import RouteScene, load_background, read_tracks, scene_bounds
from .route_times import with_flight_times
from .sqlite import connect
from .thermal_daily import PARIS
from .thermal_index import _check_cancel

COLUMNS = [
    "discipline",
    "flight_id",
    *PAIR_COLUMNS,
    "duration_s",
    "t0",
    "t1",
    "lat0",
    "lon0",
    "alt0",
    "segments",
]
RANK_COLUMNS = [
    "ix",
    "iy",
    "grouped_flights",
    "groups",
    "largest",
    "days",
    "departures",
]
GROUP_COLUMNS = [
    "ix",
    "iy",
    "day",
    "first",
    "stop",
    "flights",
    "departure_utc",
    "last_departure_utc",
]

#: Bump when group_flights.py or route_times.py changes what the catalog saves.
GROUP_VERSION = 1


def _signature(index):
    """Invalidate clocks after changes to the archive, catalog, origins or recovery."""
    sources = [
        GROUP_VERSION,
        index.signature,
        thermal_index.archive_signature(list(index.disciplines)),
    ]
    return hashlib.sha256(json.dumps(sources).encode()).hexdigest()


def load_catalog(index, *, prepare=False, progress=lambda _: None, cancel=None):
    """Reuse endpoint blocks and verified UTC origins, never full raw trajectories.

    A missing thermal UTC index requires explicit preparation of IGC header clocks.
    Results publish atomically. A cancelled or stale build never replaces a cache.
    """
    index.verify()
    signature = _signature(index)
    path = index.path.with_name("group-flight-catalog.parquet")
    if path.exists() and not prepare:
        metadata = pq.read_schema(path).metadata or {}
        if metadata.get(b"group_signature", b"").decode() == signature:
            return pd.read_parquet(path)
    clocks = thermal_index.load_saved_index(list(index.disciplines))
    if clocks is None and not prepare:
        raise ValueError(
            "No current launch-clock cache. Click Prepare / refresh groups "
            "to recover IGC dates."
        )
    progress("Locating cleaned departures on the existing 5 km grid…")
    with connect(index.path) as db:
        rows = _endpoint_rows(
            db, index.disciplines, progress, cancel, departure_only=True
        )
    flights = pd.DataFrame(rows, columns=COLUMNS)
    _check_cancel(cancel)
    if clocks is not None:
        progress("Reusing verified IGC clock origins from the thermal index…")
        with connect(clocks.path.resolve().as_uri() + "?mode=ro", uri=True) as db:
            origins = pd.read_sql_query(
                "SELECT discipline,flight_id,start_utc,trim_start FROM flights", db
            )
        flights = flights.merge(
            origins, on=["discipline", "flight_id"], how="left", validate="one_to_one"
        )
        flights["departure_utc"] = flights.start_utc + flights.trim_start + flights.t0
        flights["arrival_utc"] = flights.start_utc + flights.trim_start + flights.t1
        flights = flights.drop(columns=["start_utc", "trim_start"])
    else:
        flights = with_flight_times(
            flights, index.disciplines, progress=progress, cancel=cancel
        )
    local = pd.to_datetime(
        flights.departure_utc, unit="s", utc=True, errors="coerce"
    ).dt.tz_convert(PARIS)
    flights["day"] = local.dt.strftime("%Y-%m-%d")
    index.verify()
    if _signature(index) != signature:
        raise ValueError("Flight sources changed during group preparation; retry")
    _check_cancel(cancel)
    table = pa.Table.from_pandas(flights, preserve_index=False)
    table = table.replace_schema_metadata(
        {**(table.schema.metadata or {}), b"group_signature": signature.encode()}
    )
    temporary = path.with_name(f".{path.stem}-{uuid4().hex}.parquet")
    try:
        pq.write_table(table, temporary, compression="zstd")
        _check_cancel(cancel)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return flights


@dataclass(frozen=True)
class GroupRanking:
    """A sorted complete population plus disjoint index ranges for its groups."""

    flights: pd.DataFrame
    groups: pd.DataFrame
    cells: pd.DataFrame
    cell_m: int
    window_minutes: int
    minimum: int
    unknown_clocks: int

    def members(self, group):
        """Chronological members of one group, including ties deterministically."""
        return (
            self.flights.iloc[int(group["first"]) : int(group["stop"])]
            .copy()
            .reset_index(drop=True)
        )


def rank_groups(
    catalog,
    *,
    cell_m=5000,
    window_minutes=30,
    minimum=2,
    discipline=None,
    year=None,
    progress=lambda _: None,
    cancel=None,
):
    """Partition each cell/day into successive windows anchored at its next launch.

    The latest included departure is at most W elapsed minutes after the first
    (inclusive). The next unassigned departure anchors the next window. Singletons
    still consume their window. Minimum size filters these fixed groups, so raising
    the minimum does not reshuffle flights or create overlaps. Paris dates isolate
    days; comparisons use UTC so DST never changes elapsed window durations.
    """
    if cell_m not in (5000, 10000) or not 1 <= window_minutes <= 1440 or minimum < 2:
        raise ValueError(
            "Use 5 or 10 km cells, 1-1440 minutes and at least two flights"
        )
    flights = catalog.copy()
    if discipline is not None:
        flights = flights.loc[flights.discipline.eq(discipline)].copy()
    known = (
        np.isfinite(flights.departure_utc)
        & np.isfinite(flights.arrival_utc)
        & flights.day.notna()
    )
    unknown = int((~known).sum())
    flights = flights.loc[known & (flights.arrival_utc >= flights.departure_utc)].copy()
    if year is not None:
        flights = flights.loc[flights.day.str.startswith(str(year))].copy()
    factor = cell_m // 5000
    flights["ix"], flights["iy"] = (
        flights.start_ix // factor,
        flights.start_iy // factor,
    )
    flights = flights.sort_values(
        ["ix", "iy", "day", "departure_utc", "discipline", "flight_id"], kind="stable"
    ).reset_index(drop=True)
    groups = []
    for number, ((ix, iy, day), part) in enumerate(
        flights.groupby(["ix", "iy", "day"], sort=False)
    ):
        if number % 500 == 0:
            _check_cancel(cancel)
            progress(f"Grouping launch days {number:,}…")
        times = part.departure_utc.to_numpy()
        start = 0
        while start < len(part):
            stop = int(
                np.searchsorted(times, times[start] + 60 * window_minutes, side="right")
            )
            if stop - start >= minimum:
                groups.append(
                    (
                        ix,
                        iy,
                        day,
                        int(part.index[start]),
                        int(part.index[stop - 1]) + 1,
                        stop - start,
                        times[start],
                        times[stop - 1],
                    )
                )
            start = stop
    groups = pd.DataFrame(groups, columns=GROUP_COLUMNS)
    cells = pd.DataFrame(columns=RANK_COLUMNS)
    if not groups.empty:
        cells = groups.groupby(["ix", "iy"], sort=False).agg(
            grouped_flights=("flights", "sum"),
            groups=("flights", "size"),
            largest=("flights", "max"),
            days=("day", "nunique"),
        )
        cells["departures"] = flights.groupby(["ix", "iy"]).size()
        cells = cells.reset_index()[RANK_COLUMNS]
    _check_cancel(cancel)
    return GroupRanking(
        flights, groups, cells, cell_m, window_minutes, minimum, unknown
    )


def ordered_cells(ranking, metric="grouped_flights"):
    """Switch ranking criterion without recalculating group membership."""
    if metric not in ("grouped_flights", "largest", "groups"):
        raise ValueError("Unknown cell ranking criterion")
    keys = list(dict.fromkeys([metric, "grouped_flights", "largest", "ix", "iy"]))
    return ranking.cells.sort_values(keys, ascending=[k in ("ix", "iy") for k in keys])


def load_group_scene(index, ranking, group, *, progress=lambda _: None, cancel=None):
    """Display every member, regardless of its destination or total path length."""
    selected = ranking.members(group)
    if selected.empty:
        raise ValueError("The selected launch group is empty")
    tracks, metrics = read_tracks(index, selected, progress=progress, cancel=cancel)
    selected = pd.concat([selected, metrics], axis=1)
    selected["rank"] = np.arange(1, len(selected) + 1)
    selected["departure_delay_s"] = (
        selected.departure_utc - selected.departure_utc.iloc[0]
    )
    cell = (int(group.ix), int(group.iy))
    bounds = scene_bounds(tracks, cell, ranking.cell_m)
    origin = (np.asarray(bounds[:2]) + bounds[2:]) / 2
    for track in tracks:
        for i, segment in enumerate(track):
            segment[:, :2] -= origin
            track[i] = segment.astype(np.float32)
    scene = RouteScene(
        selected, len(selected), cell, tracks, bounds, origin, cell_m=ranking.cell_m
    )
    return load_background(scene, index, progress=progress, cancel=cancel)
