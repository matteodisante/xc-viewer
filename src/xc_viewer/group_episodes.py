"""Flight groups found by the delayed-contact method, for ranking and replay.

The tables are those the group identification of xc-lab writes, exported to the
``groups/`` folder of the data folder: every flight with its UTC clock, the pair
intervals of contact, the configurations (a fixed set of at least two flights that
stays one group over a maximal interval), the episodes (chains of configurations, each
starting when the previous one ends and sharing a member with it), the membership of
every flight and, when present, the distance covered by every episode.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .core.preproc.enu import LocalFrame
from .geodesy import enu_to_geodetic
from .route_index import FIX_COLUMNS
from .route_scene import RouteScene, load_background
from .sqlite import connect
from .thermal_geometry import continuous_edges, project
from .thermal_index import _check_cancel

TABLES = ("flights", "pair_intervals", "configurations", "membership", "episodes")
DISCIPLINE_NAMES = {"para": "paragliders", "hang": "hang gliders"}
ORDERS = {
    "flights": "Most flights",
    "largest": "Largest configuration",
    "duration": "Longest",
    "distance": "Longest distance",
}
SORT_KEYS = {
    "flights": ["n_flights", "duration_s"],
    "largest": ["n_max", "n_flights"],
    "duration": ["duration_s", "n_flights"],
    "distance": ["distance_m", "n_flights"],
}
#: Replay starts and ends this long before and after the episode.
MARGIN_S = 600


@dataclass(frozen=True)
class GroupTables:
    """The tables of one run of the group identification."""

    folder: Path
    flights: pd.DataFrame
    pairs: pd.DataFrame
    configurations: pd.DataFrame
    membership: pd.DataFrame
    episodes: pd.DataFrame
    calibration: dict

    @property
    def has_distance(self):
        """Whether the folder holds the distance of its episodes."""
        return bool(self.episodes.distance_m.notna().any())


def load_tables(folder):
    """Read a folder of group tables; ``FileNotFoundError`` names what is missing."""
    folder = Path(folder)
    missing = [n for n in TABLES if not (folder / f"{n}.parquet").is_file()]
    if missing:
        raise FileNotFoundError(
            f"{folder} has no " + ", ".join(f"{n}.parquet" for n in missing)
        )
    read = {n: pd.read_parquet(folder / f"{n}.parquet") for n in TABLES}
    flights = read["flights"].copy()
    flights["discipline"] = flights.discipline.map(DISCIPLINE_NAMES).fillna(
        flights.discipline
    )
    configurations = read["configurations"]
    episodes = read["episodes"].copy()
    episodes["duration_s"] = episodes.t_end - episodes.t_start
    distance = folder / "episode_distance.parquet"
    if distance.is_file():
        d = pd.read_parquet(distance)[["episode_id", "distance_m"]]
        episodes = episodes.merge(d, on="episode_id", how="left")
    else:
        episodes["distance_m"] = np.nan
    members = configurations[["episode_id", "members"]].explode("members")
    members = members.merge(
        flights[["flight", "discipline"]], left_on="members", right_on="flight"
    )
    kinds = members.groupby("episode_id").discipline.agg(lambda d: frozenset(d))
    episodes["disciplines"] = episodes.episode_id.map(kinds)
    calibration = folder / "calibration.json"
    return GroupTables(
        folder,
        flights,
        read["pair_intervals"],
        configurations,
        read["membership"],
        episodes,
        json.loads(calibration.read_text()) if calibration.is_file() else {},
    )


def rank_episodes(tables, order="flights", *, discipline=None, year=None, minimum=2):
    """Episodes in the requested order, best first, after the filters.

    The filters keep a member of the discipline, the UTC year of the date, and at least
    ``minimum`` distinct flights.
    """
    e = tables.episodes
    if discipline is not None:
        e = e[e.disciplines.map(lambda d: discipline in d)]
    if year is not None:
        e = e[e.date.str[:4] == str(year)]
    e = e[e.n_flights >= minimum]
    return e.sort_values(
        SORT_KEYS[order], ascending=False, na_position="last", kind="stable"
    ).reset_index(drop=True)


@dataclass
class TimedTrack:
    """One flight in the scene frame: continuous pieces of (UTC s, x, y, z)."""

    pieces: list = field(default_factory=list)

    def at(self, t):
        """Position at UTC ``t``, linear between fixes; None outside every piece."""
        for p in self.pieces:
            if p[0, 0] <= t <= p[-1, 0]:
                return np.array([np.interp(t, p[:, 0], p[:, k]) for k in (1, 2, 3)])
        return None

    def trail(self, t, seconds):
        """The fixes of the ``seconds`` before ``t``, up to the position at ``t``."""
        out = []
        for p in self.pieces:
            keep = (p[:, 0] >= t - seconds) & (p[:, 0] <= t)
            if keep.any():
                out.append(p[keep, 1:])
        now = self.at(t)
        if out and now is not None:
            out[-1] = np.vstack([out[-1], now])
        return out


@dataclass
class Episode:
    """One episode with everything its replay needs."""

    row: pd.Series
    members: pd.DataFrame
    configurations: pd.DataFrame
    pairs: pd.DataFrame
    membership: pd.DataFrame
    day: pd.DataFrame
    tracks: dict = field(default_factory=dict)
    #: Per member, whether each drawn piece of its track lies in the episode.
    inside: dict = field(default_factory=dict)

    @property
    def window(self):
        """First and last UTC second of the replay."""
        return self.row.t_start - MARGIN_S, self.row.t_end + MARGIN_S

    def configurations_at(self, t):
        """The configurations of this episode in progress at ``t``."""
        c = self.configurations
        return c[(c.t_start <= t) & (t < c.t_end)]

    def edges_at(self, t):
        """Pairs of members whose interval of contact contains ``t``."""
        p = self.pairs
        return p.loc[(p.t_on <= t) & (t < p.t_off), ["a", "b"]].to_numpy()

    def episodes_at(self, t):
        """Every episode of the same date in progress at ``t``."""
        d = self.day
        return d[(d.t_start <= t) & (t < d.t_end)]

    def state_of(self, flight, t):
        """'L' (local phase), 'S' (solo), 'G' (in a group), or None off the flight."""
        m = self.membership
        row = m[(m.flight == flight) & (m.t_start <= t) & (t < m.t_end)]
        return None if row.empty else row.state.iat[0]


def split_inside(piece, intervals):
    """Split a piece (UTC s, x, y, z) where it enters or leaves `intervals`.

    `intervals` are rows of [start, end). The sub-pieces share their joining fix, so
    each has two fixes or more. Returns the sub-pieces and whether each lies in the
    intervals.
    """
    t = piece[:, 0]
    inside = np.zeros(len(t), bool)
    for start, end in intervals:
        inside |= (t >= start) & (t < end)
    starts = np.r_[0, np.flatnonzero(np.diff(inside)) + 1]
    ends = np.r_[starts[1:], len(t)]
    pieces = []
    for a, b in zip(starts, ends, strict=True):
        sub = piece[a : min(b + 1, len(t))]
        pieces.append(sub if len(sub) > 1 else piece[max(a - 1, 0) : b])
    return pieces, [bool(inside[a]) for a in starts]


def reeb_graph(configurations):
    """The Reeb graph of an episode, without the edges of single members.

    Every configuration is an edge on a lane: among the lanes free when it starts and
    one new lane, the nearest to the mean lane of the configurations it continues,
    weighted by the members it keeps from each; at a split the larger part goes first.
    Returns the edges (conf_id, t0, t1, n, lane) and the links (t, lane0, lane1) from a
    configuration to each one that starts when it ends and keeps a member of it.
    """
    confs = configurations.assign(larger=-configurations.n).sort_values(
        ["t_start", "larger"], kind="stable"
    )
    members = {
        c: {int(f) for f in m}
        for c, m in zip(confs.conf_id, confs.members, strict=True)
    }
    lane, free_at, links = {}, [], []
    for c in confs.itertuples():
        before = confs.conf_id[confs.t_end == c.t_start]
        kept = [(lane[p], len(members[p] & members[c.conf_id])) for p in before]
        kept = [(lane_, k) for lane_, k in kept if k]
        want = (
            sum(lane_ * k for lane_, k in kept) / sum(k for _, k in kept)
            if kept
            else 0.0
        )
        open_ = [k for k, end in enumerate(free_at) if end <= c.t_start]
        here = min([*open_, len(free_at)], key=lambda k: abs(k - want))
        if here == len(free_at):
            free_at.append(c.t_end)
        else:
            free_at[here] = c.t_end
        lane[c.conf_id] = here
        links += [(c.t_start, lane_, here) for lane_, _ in kept]
    edges = pd.DataFrame(
        {
            "conf_id": confs.conf_id.to_numpy(int),
            "t0": confs.t_start.to_numpy(float),
            "t1": confs.t_end.to_numpy(float),
            "n": confs.n.to_numpy(int),
            "lane": confs.conf_id.map(lane).to_numpy(int),
        }
    )
    return edges, pd.DataFrame(links, columns=["t", "lane0", "lane1"], dtype=float)


def episode(tables, episode_id):
    """The members, configurations, contacts and membership of one episode."""
    row = tables.episodes.set_index("episode_id").loc[episode_id]
    row = row.copy()
    row["episode_id"] = episode_id
    configurations = tables.configurations[
        tables.configurations.episode_id == episode_id
    ].sort_values("t_start", kind="stable")
    members = sorted({int(f) for m in configurations.members for f in m})
    flights = tables.flights.set_index("flight").loc[members].reset_index()
    p = tables.pairs
    pairs = p[
        p.a.isin(members)
        & p.b.isin(members)
        & (p.t_on < row.t_end)
        & (p.t_off > row.t_start)
    ]
    membership = tables.membership[tables.membership.flight.isin(members)]
    day = tables.episodes[tables.episodes.date == row.date]
    return Episode(row, flights, configurations, pairs, membership, day)


def read_timed_tracks(index, members, *, progress=lambda _: None, cancel=None):
    """Cleaned tracks of the members in Lambert-93 with UTC times, split at gaps."""
    index.verify()
    frames = index.flights().set_index(["discipline", "flight_id"])
    parts = {}
    with connect(index.path) as db:
        for disc in index.disciplines:
            ids = members.loc[members.discipline == disc.name, "flight_id"].astype(str)
            ids = ids.tolist()
            if not ids:
                continue
            query = (
                "SELECT DISTINCT row_group FROM parts WHERE discipline=? AND flight_id "
                "IN (" + ",".join("?" for _ in ids) + ") ORDER BY row_group"
            )
            groups = [r[0] for r in db.execute(query, (disc.name, *ids))]
            archive = pq.ParquetFile(disc.config().derived_dir / "fixes.parquet")
            for i, group in enumerate(groups):
                _check_cancel(cancel)
                progress(f"{disc.name}: reading member tracks {i + 1}/{len(groups)}")
                frame = archive.read_row_group(group, columns=FIX_COLUMNS).to_pandas()
                frame["flight_id"] = frame.flight_id.astype(str)
                for fid, f in frame[frame.flight_id.isin(ids)].groupby("flight_id"):
                    parts.setdefault((disc.name, fid), []).append(f)
    tracks = {}
    for row in members.itertuples():
        key = (row.discipline, str(row.flight_id))
        if key not in parts:
            raise ValueError(f"Missing cleaned geometry for {key[0]} {key[1]}")
        fixes = pd.concat(parts[key]).sort_values("t", kind="stable")
        fixes = fixes.reset_index(drop=True)
        meta = frames.loc[key]
        lat, lon = enu_to_geodetic(
            fixes.E, fixes.N, fixes.z, LocalFrame(meta.lat0, meta.lon0, meta.alt0)
        )
        x, y = project(lon, lat)
        points = np.column_stack((row.utc0 + fixes.t.to_numpy(float), x, y, fixes.z))
        edges = continuous_edges(fixes) & np.isfinite(points).all(axis=1)[1:]
        pieces = np.split(points, np.flatnonzero(~edges) + 1)
        tracks[row.flight] = TimedTrack(
            [p for p in pieces if len(p) > 1 and np.isfinite(p).all()]
        )
    return tracks


def load_episode_scene(
    index, tables, episode_id, *, progress=lambda _: None, cancel=None
):
    """The episode and its members' tracks in its window, on their terrain."""
    ep = episode(tables, episode_id)
    progress("Reading the cleaned tracks of the members…")
    tracks = read_timed_tracks(index, ep.members, progress=progress, cancel=cancel)
    low, high = ep.window
    for t in tracks.values():
        inside = [p[(p[:, 0] >= low) & (p[:, 0] <= high)] for p in t.pieces]
        # A member flies in its episode; its whole track stays if a gap hides it.
        t.pieces = [p for p in inside if len(p) > 1] or t.pieces
    xy = np.vstack([p[:, 1:3] for t in tracks.values() for p in t.pieces])
    low, high = xy.min(axis=0) - 2000, xy.max(axis=0) + 2000
    bounds = (*np.floor(low / 1000) * 1000, *np.ceil(high / 1000) * 1000)
    origin = (np.asarray(bounds[:2]) + bounds[2:]) / 2
    for t in tracks.values():
        for p in t.pieces:
            p[:, 1:3] -= origin
    ep.tracks = tracks
    m = ep.membership[ep.membership.conf_id.isin(ep.configurations.conf_id)]
    lines = []
    for f in ep.members.flight:
        stretches = m.loc[m.flight == f, ["t_start", "t_end"]].to_numpy(float)
        parts = [split_inside(p, stretches) for p in tracks[f].pieces]
        lines.append(
            [s[:, 1:].astype(np.float32) for pieces, _ in parts for s in pieces]
        )
        ep.inside[f] = [flag for _, flags in parts for flag in flags]
    scene = RouteScene(ep.members, len(ep.members), (), lines, bounds, origin)
    return load_background(scene, index, progress=progress, cancel=cancel), ep
