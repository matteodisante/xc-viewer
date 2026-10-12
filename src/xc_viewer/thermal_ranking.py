"""Offline DEM-based cell ranking and a separate audit of raw launch references."""

from __future__ import annotations

import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path

import pandas as pd

from .core.config import load_preproc_config
from .core.igc import _altitude, _lat, _lon, _valid_time_of_day
from .geography import TERRAIN_ORDER
from .sqlite import connect
from .thermal_geometry import ThermalCell
from .thermal_ground import (
    CLIMB_RANKING,
    TERRAIN_RANKING,
    IncompleteTerrainError,
    cell_band,
    cell_ground,
    fetch_terrain_reference,
    terrain_reference,
)
from .thermal_orography import DATA_DIRECTORY


def launch_quality(path, horizontal_limit, vertical_limit):
    """Reject an invalid first fix or a jump corroborated by two later raw fixes.

    Both witnesses must be valid fixes within 120 seconds. A single bad later fix
    cannot reject a good origin. The accepted altitude/position are never replaced.
    This is a conservative origin screen, not a terrain elevation measurement.
    """
    fixes = []
    with Path(path).open(encoding="latin-1") as stream:
        for line in stream:
            if not line.startswith("B") or len(line.rstrip()) < 35:
                continue
            try:
                hh, mm, ss = int(line[1:3]), int(line[3:5]), int(line[5:7])
                if not _valid_time_of_day(hh, mm, ss):
                    continue
                lat, lon = _lat(line[7:15]), _lon(line[15:24])
            except ValueError:
                continue
            fixes.append(
                (
                    hh * 3600 + mm * 60 + ss,
                    lat,
                    lon,
                    _altitude(line[30:35]),
                    line[24] in "Aa",
                )
            )
            if len(fixes) >= 20:
                break
    if not fixes:
        return "unreadable"
    origin = fixes[0]
    if not origin[4]:
        return "invalid_gnss"
    horizontal = vertical = 0
    witnesses = 0
    for t, lat, lon, z, valid in fixes[1:]:
        dt = t - origin[0]
        if origin[0] > 82800 and t < 3600:
            dt += 86400
        if dt > 120:
            break
        if dt <= 0 or not valid or not math.isfinite(z) or z == 0:
            continue
        a, b = math.radians(origin[1]), math.radians(lat)
        h = (
            math.sin((b - a) / 2) ** 2
            + math.cos(a)
            * math.cos(b)
            * math.sin(math.radians(lon - origin[2]) / 2) ** 2
        )
        distance = 6371000 * 2 * math.asin(min(1, math.sqrt(h)))
        horizontal += distance / dt > horizontal_limit
        vertical += abs(z - origin[3]) / dt > vertical_limit
        witnesses += 1
        if horizontal >= 2:
            return "position_jump"
        if vertical >= 2:
            return "altitude_jump"
        if witnesses >= 5:
            break
    return "accepted" if witnesses >= 2 else "unchecked"


def audit_launches(index, progress=print):
    """Persist the origin screen separately, keeping reusable climb keys unchanged."""
    cfg = load_preproc_config().fix
    policy = hashlib.sha256(
        json.dumps(
            [
                1,
                index.signature,
                cfg.max_horizontal_speed_mps,
                cfg.max_vertical_speed_mps,
                120,
                20,
                5,
                2,
            ]
        ).encode()
    ).hexdigest()
    path = index.path.with_name("thermal-launch-quality.sqlite3")
    with connect(index.path) as census, connect(path) as db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS origins (policy TEXT,discipline TEXT,"
            "flight_id TEXT,status TEXT, PRIMARY KEY(policy,discipline,flight_id))"
        )
        existing = set(
            db.execute(
                "SELECT discipline,flight_id FROM origins WHERE policy=?", (policy,)
            )
        )
        rows = [
            r
            for r in census.execute(
                "SELECT discipline,flight_id,path FROM flights "
                "WHERE launch_alt IS NOT NULL"
            )
            if r[:2] not in existing
        ]

        def check(row):
            disc, fid, filename = row
            status = launch_quality(
                filename, cfg.max_horizontal_speed_mps[disc], cfg.max_vertical_speed_mps
            )
            return policy, disc, fid, status

        with ThreadPoolExecutor(max_workers=8) as pool:
            for i, result in enumerate(pool.map(check, rows)):
                db.execute("INSERT OR REPLACE INTO origins VALUES (?,?,?,?)", result)
                if (i + 1) % 1000 == 0:
                    db.commit()
                    progress(f"Checked raw launch origins: {i + 1:,}/{len(rows):,}")
        db.commit()
    return path, policy


@dataclass(frozen=True)
class RankedIndex:
    """A selected subset sharing the unchanged census and climb-product identity."""

    base: object
    selected: tuple[ThermalCell, ...]
    quality_summary: dict
    terrain_references: dict
    activity_counts: dict = field(default_factory=dict)

    @property
    def path(self):
        """Keep the original census identity for reusable cell/flight geometry."""
        return self.base.path

    @property
    def signature(self):
        """Archive inputs remain unchanged by the ranking policy."""
        return self.base.signature

    @property
    def disciplines(self):
        """Disciplines included in this census."""
        return self.base.disciplines

    def cells(self):
        """Return category order followed by population rank within category."""
        return list(self.selected)

    def flights(self, cell):
        """Keep every distinct crossing flight, independently of launch quality."""
        return self.base.flights(cell)


def _cell_terrain(cell, index):
    """Reuse attributed local terrain; fetch missing candidates into the cache."""
    if any(
        (DATA_DIRECTORY / name).exists()
        for name in (
            f"ign-terrain-{cell.ix}-{cell.iy}.json",
            f"ign-ridges-{cell.ix}-{cell.iy}.geojson",
        )
    ):
        return terrain_reference(cell)
    return fetch_terrain_reference(cell, index.path.parent / "exploration/terrain")


def rank_cells(
    index, *, per_category=3, quality=None, activity=None, progress=lambda _: None
):
    """Select the busiest visited cells in each band of highest IGN terrain elevation.

    With an activity census, rank Vilpellet climb runs instead of visitors.
    Visit candidates in descending score order and stop when every band is
    full. No unexamined cell can outrank a winner. Starts are audit data only;
    a visited cell needs no internal launch to qualify. A cell the IGN DEM does
    not fully cover (beyond the border) cannot be banded: it is skipped and
    listed in the summary. Any other terrain failure aborts preparation.
    """
    if per_category < 1:
        raise ValueError("per_category must be at least 1")
    with connect(index.path) as db:
        starts = pd.read_sql_query(
            "SELECT discipline,flight_id,launch_x ix,launch_y iy,launch_alt "
            "FROM flights WHERE launch_alt IS NOT NULL",
            db,
        )
        visits = pd.read_sql_query(
            "SELECT ix,iy,COUNT(*) flights,MAX(max_alt) max_alt "
            "FROM visits GROUP BY ix,iy",
            db,
        )
    summary = {"policy": "raw_first_fix", "excluded": 0}
    if quality is not None:
        path, policy = quality
        with connect(path) as db:
            statuses = pd.read_sql_query(
                "SELECT discipline,flight_id,status FROM origins WHERE policy=?",
                db,
                params=(policy,),
            )
        starts = starts.merge(
            statuses, on=["discipline", "flight_id"], how="left", validate="one_to_one"
        )
        if starts.status.isna().any():
            raise ValueError("Launch quality audit is incomplete")
        accepted = starts.status.isin(["accepted", "unchecked"])
        summary = {
            "policy": policy,
            "excluded": int((~accepted).sum()),
            "statuses": starts.status.value_counts().to_dict(),
        }
        starts = starts.loc[accepted]
    launches = starts.groupby(["ix", "iy"], as_index=False).agg(
        launch_median=("launch_alt", "median"), launches=("launch_alt", "size")
    )
    cells = visits.merge(launches, on=["ix", "iy"], how="left")
    metric = "flights"
    if activity is not None:
        with connect(Path(activity).as_uri() + "?mode=ro", uri=True) as db:
            metadata = dict(db.execute("SELECT key,value FROM metadata"))
            if metadata.get("ready") != "1":
                raise ValueError("National Vilpellet climb census is incomplete")
            counts = pd.read_sql_query(
                "SELECT ix,iy,SUM(climb_runs) climb_runs,COUNT(*) climb_flights "
                "FROM activity GROUP BY ix,iy",
                db,
            )
        cells = cells.merge(counts, on=["ix", "iy"], validate="one_to_one")
        metric = "climb_runs"
        summary["activity_signature"] = metadata["signature"]
        summary["ranking_method"] = "vilpellet"
        summary["ranking_unit"] = "continuous_climb_runs"
    cells = cells.sort_values([metric, "ix", "iy"], ascending=[False, True, True])
    groups = {band: [] for band in TERRAIN_ORDER}
    references = {}
    activity_counts = {}
    excluded = []
    examined = 0
    for c in cells.itertuples(index=False):
        cell = ThermalCell(
            int(c.ix),
            int(c.iy),
            "",
            int(c.flights),
            int(c.launches) if pd.notna(c.launches) else 0,
            0.0,
            float(c.max_alt),
            float(c.launch_median) if pd.notna(c.launch_median) else None,
        )
        examined += 1
        try:
            reference = _cell_terrain(cell, index)
        except IncompleteTerrainError:
            excluded.append([cell.ix, cell.iy])
            progress(f"Terrain candidate {examined}: {cell.ix}/{cell.iy} excluded")
            continue
        band = cell_band(reference)
        progress(
            f"Terrain candidate {examined}: {cell.ix}/{cell.iy}, "
            + (f"{int(c.climb_runs):,} Vilpellet climbs, " if activity else "")
            + f"{cell.flights:,} flights, highest {reference['maximum_m']:.1f} m, "
            + band
        )
        if len(groups[band]) < per_category:
            groups[band].append(
                replace(cell, terrain=band, ground_m=cell_ground(reference))
            )
            references[cell.ix, cell.iy] = reference
            if activity is not None:
                activity_counts[cell.ix, cell.iy] = {
                    "climb_runs": int(c.climb_runs),
                    "climb_flights": int(c.climb_flights),
                }
        if all(len(group) == per_category for group in groups.values()):
            break
    result = [cell for group in groups.values() for cell in group]
    summary["ranking_reference"] = CLIMB_RANKING if activity else TERRAIN_RANKING
    summary["terrain_candidates_examined"] = examined
    summary["terrain_excluded"] = excluded
    summary["visited_cells"] = len(visits)
    return RankedIndex(index, tuple(result), summary, references, activity_counts)
