"""Loading one flight for the viewer: the raw track, and the cleaned one alongside it.

No Qt or matplotlib import here (see the package docstring): this module is the
Qt-free logic layer, exercised directly by ``tests/viewer/test_data.py``.

Two independent reads of the same ``.igc`` file, not one shared parse feeding both:
:func:`load_raw` stops after stage (i)
(:mod:`xc_viewer.core.preproc.altchannel`),
:func:`load_cleaned` runs the full seven-stage pipeline
(:func:`xc_viewer.core.preproc.pipeline.run_flight`). Re-parsing a single
flight's
text is negligible next to either the pipeline itself or a mouse click, and it keeps
each function a self-contained entry point that only needs a path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from .core import igc
from .core.disciplines import DISCIPLINES, Discipline
from .core.preproc.altchannel import AltChannel, adopt_alt_channel
from .core.preproc.enu import LocalFrame, geodetic_to_enu, to_local_frame
from .core.preproc.pipeline import FlightResult, run_flight
from .geodesy import enu_to_geodetic

if TYPE_CHECKING:
    from .core.config import PreprocConfig
    from .core.gaussian import GaussianConfig
    from .core.preproc.pipeline import FlightRecord
    from .core.vilpellet import VilpelletConfig


@dataclass(frozen=True)
class RawTrack:
    """One flight's parsed track, with the adopted (GNSS) altitude added.

    Attributes:
        fixes: The output of :func:`xc_viewer.core.igc.parse_igc`, plus the
        ``alt``
            column stage (i) adds -- the GNSS channel, ``nan`` where absent, never
            gap-filled here (the viewer draws the real gaps; only the ENU projection
            of :func:`raw_to_enu` needs a gap-free proxy, and fills its own).
        alt_channel: Stage (i)'s verdict on this flight's altitude channels -- whether
            the pipeline would even admit it, and why, shown alongside a raw plot so a
            flight the archive dropped is not mistaken for one it kept.
    """

    fixes: pd.DataFrame
    alt_channel: AltChannel


@dataclass(frozen=True)
class PhaseTrack:
    """Every cleaned fix of one flight, labelled by one flight-phase segmentation.

    ``phase_run`` changes at every phase, preprocessing-segment, or temporal-gap
    boundary.  The viewer uses it as a line-group key, preventing equal phases that
    occur at different times from being joined by an artificial straight line.
    """

    fixes: pd.DataFrame
    mapping_method: str
    coverage: dict = field(default_factory=dict)


def load_raw(igc_path: str | Path, cfg: PreprocConfig | None = None) -> RawTrack:
    """Parse an ``.igc`` file and adopt its altitude channel (pipeline stage (i) only).

    Args:
        igc_path: Path to the ``.igc`` file, from a file dialog or a catalog lookup.
        cfg: The pre-processing thresholds; loaded from
            ``configs/preprocessing.yaml`` if not given.

    Returns:
        The :class:`RawTrack`.
    """
    from .core.config import load_preproc_config

    if cfg is None:
        cfg = load_preproc_config()
    fixes = igc.parse_igc(igc_path)
    with_alt, channel = adopt_alt_channel(fixes, cfg.alt_channel)
    return RawTrack(fixes=with_alt, alt_channel=channel)


def load_cleaned(
    igc_path: str | Path,
    cfg: PreprocConfig | None = None,
    *,
    source: str,
    flight_id: str,
    discipline: str,
) -> FlightResult:
    """Run the full pre-processing pipeline over one ``.igc`` file, on demand.

    Deliberately not a read of the archive's ``derived/fixes.parquet``: that table has
    no per-flight index and is 43 GB for paragliders, so a lookup would mean scanning
    row groups from the start on every click. Recomputing a single flight is
    sub-second and works identically whether the file is already in the processed
    archive or was just picked off the disk -- and it means the viewer always reflects
    the pipeline as configured *now*, not whatever version last wrote the archive.

    Args:
        igc_path: Path to the ``.igc`` file.
        cfg: The pre-processing thresholds; loaded from
            ``configs/preprocessing.yaml`` if not given.
        source: ``"paraglider"`` / ``"hangglider"`` -- the value the ``source`` column
            takes, i.e. ``Discipline.source``.
        flight_id: An identifier for this flight (the catalog's, or the file stem for
            an arbitrary disk pick); only used to key/label the result.
        discipline: The key the per-discipline speed bound is stored under
            (``Discipline.name``, e.g. ``"paragliders"``).

    Returns:
        The :class:`~xc_viewer.core.preproc.pipeline.FlightResult`. When a gate
        drops the flight, ``result.kept`` is ``False`` and ``result.meta.drop_stage`` /
        ``result.meta.drop_reason`` say which and why; the raw track (:func:`load_raw`)
        is still worth showing in that case.
    """
    from .core.config import load_preproc_config

    if cfg is None:
        cfg = load_preproc_config()
    fixes = igc.parse_igc(igc_path)
    return run_flight(
        fixes, cfg, source=source, flight_id=flight_id, discipline=discipline
    )


@lru_cache(maxsize=4)
def _load_vilpellet_config(config_path: str, config_mtime_ns: int) -> VilpelletConfig:
    """Load and cache the transcribed protocol until its YAML changes on disk."""
    del config_mtime_ns  # part of the cache key; the loader only needs the path
    from .core.vilpellet import load_vilpellet_config

    return load_vilpellet_config(config_path)


def _vilpellet_config() -> VilpelletConfig | None:
    """The configured Vilpellet protocol, or ``None`` when the file is unusable.

    Returns:
        The validated configuration, or ``None`` when ``segmentation_vilpellet.yaml``
        is missing, unreadable, or describes a model the decoder would reject.
    """
    from .core.vilpellet import config as vilpellet_config

    path = Path(vilpellet_config.DEFAULT_VILPELLET_CONFIG_PATH)
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        return None
    try:
        return _load_vilpellet_config(str(path), mtime)
    except (OSError, KeyError, TypeError, ValueError):
        return None


def load_vilpellet_phases(
    cleaned_fixes: pd.DataFrame,
    discipline: Discipline,
    *,
    enforce_flight_guard: bool = True,
    config: VilpelletConfig | None = None,
) -> PhaseTrack | None:
    """Label the selected cleaned flight with the transcribed Vilpellet segmenter.

    It reads no archive at all: the fitted parameters live in
    ``configs/segmentation_vilpellet.yaml``, and the decode runs over the
    ``FlightResult.fixes`` table already in memory, about a tenth of a second for a
    21,000-fix flight.

    Args:
        cleaned_fixes: One retained, fully preprocessed flight.
        discipline: Its selected viewer discipline; names the straightness threshold.
        enforce_flight_guard: Apply the author's own minimum-flight-length guard.
            Switching it off decodes a flight the guard would leave entirely
            unlabelled, and changes no label of a flight that passes it.
        config: An already-loaded protocol, for callers that want to vary it.  The
            configured one is loaded and cached when this is omitted.

    Returns:
        A phase track, or ``None`` when the configuration cannot be read.  A flight
        the eligibility gate rejects still returns a track: every fix is
        ``unclassified`` and ``coverage["fixes_by_reason"]`` names the guard that
        stopped it.
    """
    from .core.coverage import native_coverage_summary
    from .core.vilpellet import segment_flight

    protocol = _vilpellet_config() if config is None else config
    if protocol is None:
        return None
    track = segment_flight(
        cleaned_fixes,
        protocol,
        discipline=discipline.name,
        enforce_flight_guard=enforce_flight_guard,
    )
    return PhaseTrack(
        fixes=track.fixes,
        mapping_method="vilpellet-fitted-2019",
        coverage=native_coverage_summary(track.fixes),
    )


@lru_cache(maxsize=4)
def _load_gaussian_config(config_path: str, config_mtime_ns: int) -> GaussianConfig:
    """Load and cache the Gaussian segmenter until its YAML changes on disk."""
    del config_mtime_ns  # part of the cache key; the loader only needs the path
    from .core.gaussian import load_gaussian_config

    return load_gaussian_config(config_path)


def load_gaussian_phases(
    cleaned_fixes: pd.DataFrame,
    discipline: Discipline,
    *,
    config: GaussianConfig | None = None,
) -> PhaseTrack | None:
    """Label the selected cleaned flight with the Gaussian segmenter.

    Decisions are made on the 10-s grid and each native fix takes the label of the
    decision cell it falls in (:func:`phases_on_cleaned_fixes`).

    Args:
        cleaned_fixes: One retained, fully preprocessed flight.
        discipline: Its viewer discipline; selects the fitted model.
        config: An already-loaded configuration; the configured one when omitted.

    Returns:
        A phase track, or ``None`` when ``segmentation_gaussian.yaml`` cannot be read.
    """
    from .core.coverage import native_coverage_summary
    from .core.gaussian import config as gaussian_config
    from .core.gaussian import segment_flight

    if config is None:
        path = Path(gaussian_config.DEFAULT_GAUSSIAN_CONFIG_PATH)
        try:
            config = _load_gaussian_config(str(path), path.stat().st_mtime_ns)
        except (OSError, KeyError, TypeError, ValueError):
            return None
    points = segment_flight(cleaned_fixes, config, discipline=discipline.name)
    fixes = phases_on_cleaned_fixes(
        cleaned_fixes, points, decision_step_s=config.decision_step_s
    )
    return PhaseTrack(
        fixes=fixes,
        mapping_method="gaussian-fitted-ffvl",
        coverage=native_coverage_summary(fixes),
    )


def phase_only(fixes: pd.DataFrame, phase: str) -> pd.DataFrame:
    """The fixes of one phase in a labelled track, with every run kept apart.

    ``phase_run`` is carried through untouched, so each run remains its own line
    group.  ``track_run`` is renumbered one value per retained run: with the fixes of
    the other phases no longer displayed, two runs are no longer two parts of one
    continuous displayed track, and the drawing code must not close the edge between
    them.

    Args:
        fixes: A labelled fix table carrying a ``phase`` column.
        phase: ``climb``, ``search`` or ``transition``.

    Returns:
        A copy holding only the fixes labelled ``phase``.  An unlabelled table yields
        an empty result with the same columns.
    """
    if "phase" not in fixes.columns:
        return fixes.iloc[:0].copy()
    kept = fixes.loc[fixes["phase"] == phase].copy()
    if "track_run" in kept.columns and "phase_run" in kept.columns:
        kept["track_run"] = pd.factorize(kept["phase_run"])[0]
    return kept


def phases_on_cleaned_fixes(
    cleaned: pd.DataFrame, points: pd.DataFrame, *, decision_step_s: float
) -> pd.DataFrame:
    """Colour native cleaned fixes without replacing their geometry.

    Each decision labels its half-open time cell [t-step/2, t+step/2), matching
    the segmentation run export. Missing decisions and ineligible segments remain
    unclassified. No label crosses a preprocessing segment or a missing time cell.
    ``track_run`` marks physical continuity; ``phase_run`` also splits on colour.
    """
    out = (
        cleaned.sort_values(["segment_id", "t"], kind="stable")
        .reset_index(drop=True)
        .copy()
    )
    out["phase"] = "unclassified"
    out["phase_reason"] = "outside_decision_cells"
    out["track_run"] = 0
    run_offset = 0
    for segment_id, group in out.groupby("segment_id", sort=False):
        t = group["t"].to_numpy(dtype=float)
        dt = np.diff(t)
        cadence = np.median(dt[dt > 0]) if np.any(dt > 0) else np.inf
        breaks = np.r_[True, (dt <= 0) | (dt > 1.5 * cadence)]
        runs = np.cumsum(breaks) - 1 + run_offset
        out.loc[group.index, "track_run"] = runs
        run_offset = int(runs[-1]) + 1
        if points.empty:
            out.loc[group.index, "phase_reason"] = "no_eligible_decisions"
            continue
        decisions = points.loc[points["segment_id"] == segment_id].sort_values("t")
        if decisions.empty:
            out.loc[group.index, "phase_reason"] = "no_eligible_decisions"
            continue
        times = decisions["t"].to_numpy(dtype=float)
        indexes = np.searchsorted(times - decision_step_s / 2, t, side="right") - 1
        safe = np.clip(indexes, 0, len(times) - 1)
        covered = (indexes >= 0) & (t < times[safe] + decision_step_s / 2)
        labels = decisions["phase"].fillna("unclassified").to_numpy()
        out.loc[group.index[covered], "phase"] = labels[safe[covered]]
        from .core.coverage import decision_reasons

        reasons = decision_reasons(decisions)
        out.loc[group.index[covered], "phase_reason"] = reasons[safe[covered]]
    starts = out["track_run"].ne(out["track_run"].shift()) | out["phase"].ne(
        out["phase"].shift()
    )
    out["phase_run"] = starts.cumsum() - 1
    return out


def frame_from_meta(meta: FlightRecord) -> LocalFrame | None:
    """The cleaned trajectory's local frame, or ``None`` if it never got one.

    A flight dropped before stage (v) (the altitude-channel gate or trimming) has no
    origin: ``meta.lat0/lon0/alt0`` are all ``None``, and the ENU view has nothing to
    project onto for that flight, raw or cleaned.
    """
    if meta.lat0 is None or meta.lon0 is None or meta.alt0 is None:
        return None
    return LocalFrame(lat0_deg=meta.lat0, lon0_deg=meta.lon0, alt0_m=meta.alt0)


def raw_only_frame(raw: RawTrack) -> LocalFrame | None:
    """A local frame computed from the raw track's own first fix.

    A fallback for when the pipeline never got far enough to produce one of its own
    (:func:`frame_from_meta` is ``None``): dropped before stage (v), or the pipeline
    raised outright. A flight that early is never ``kept``, so there is no cleaned
    trajectory to align with in that situation either -- raw's own origin is exactly
    as good as any other, and it means the ENU view still has something to show
    instead of nothing at all.

    Returns:
        The frame, or ``None`` if the raw track is empty (no origin fix to place it
        at -- an IGC file with no decodable ``B`` record).
    """
    try:
        _local, frame = to_local_frame(raw.fixes, alt_column="alt")
    except ValueError:
        return None
    return frame


def _fill_altitude_gaps(t: np.ndarray, alt: np.ndarray) -> np.ndarray:
    """A gap-free altitude proxy for the ENU projection's radius correction.

    Mirrors the private helper of the same purpose in
    ``xc_viewer.core.preproc.enu.to_local_frame``, kept local rather than
    imported
    so this module does not reach into another module's private surface. A missing
    altitude enters :func:`~xc_viewer.core.preproc.enu.geodetic_to_ecef` only
    through
    the small radius term ``N + h``, so any plausible fill leaves the horizontal
    (E, N) position sound (the docstring of ``enu.py`` derives the bound: well under a
    metre across a whole flight) -- unlike the *displayed* altitude, which keeps its
    real gaps.
    """
    finite = np.isfinite(alt)
    if finite.all():
        return alt
    if not finite.any():
        return np.zeros_like(alt)
    return np.interp(t, t[finite], alt[finite])


def raw_to_enu(raw: RawTrack, frame: LocalFrame) -> pd.DataFrame:
    """Project the raw track into the *cleaned* trajectory's own local ENU frame.

    Deliberately not :func:`~xc_viewer.core.preproc.enu.to_local_frame`: that
    picks
    a *new* origin from its input's first row, which for the raw track is still on the
    ground -- misaligned against the cleaned trajectory's origin (the first fix of
    free flight). Calling
    :func:`~xc_viewer.core.preproc.enu.geodetic_to_enu` directly, with the frame
    the
    cleaned trajectory was actually projected with, is what lets raw and cleaned
    overlay correctly in ENU.

    Args:
        raw: The :class:`RawTrack` to project.
        frame: The cleaned trajectory's origin, from :func:`frame_from_meta`.

    Returns:
        A table with columns ``t``, ``E``, ``N``, ``z`` -- ``z`` is the raw adopted
        altitude at its measured value (real gaps kept), never the rotation's "up".
    """
    lat = raw.fixes["lat"].to_numpy(dtype=float)
    lon = raw.fixes["lon"].to_numpy(dtype=float)
    alt = raw.fixes["alt"].to_numpy(dtype=float)
    t = raw.fixes["t"].to_numpy(dtype=float)
    alt_for_frame = _fill_altitude_gaps(t, alt)
    east, north, _up = geodetic_to_enu(
        lat, lon, alt_for_frame, frame.lat0_deg, frame.lon0_deg, frame.alt0_m
    )
    return pd.DataFrame({"t": t, "E": east, "N": north, "z": alt})


def cleaned_to_geographic(
    cleaned_fixes: pd.DataFrame, frame: LocalFrame
) -> pd.DataFrame:
    """The cleaned trajectory's ``(E, N, z)`` fixes, back in geographic coordinates.

    ``FlightResult.fixes`` never carries ``lat``/``lon`` -- stage (v) of the pipeline
    is one-way (see :mod:`xc_viewer.geodesy`) -- so showing the cleaned
    trajectory overlaid against the raw one in the geographic frame needs this
    inverse. ``segment_id`` and every other carried-through column are kept, so the
    result still groups the same way :func:`xc_viewer.plotting.plot_trajectory`
    expects.

    Args:
        cleaned_fixes: A ``FlightResult.fixes`` table (or a subset of its rows).
        frame: The trajectory's own origin, from :func:`frame_from_meta`.

    Returns:
        A copy of ``cleaned_fixes`` with ``lat``/``lon`` columns added; ``alt`` is
        ``z`` under its geographic name, for symmetry with :class:`RawTrack`.
    """
    lat, lon = enu_to_geodetic(
        cleaned_fixes["E"].to_numpy(dtype=float),
        cleaned_fixes["N"].to_numpy(dtype=float),
        cleaned_fixes["z"].to_numpy(dtype=float),
        frame,
    )
    out = cleaned_fixes.copy()
    out["lat"] = lat
    out["lon"] = lon
    out["alt"] = cleaned_fixes["z"]
    return out


def format_dms(deg: float, axis: Literal["lat", "lon"]) -> str:
    """Decimal degrees as a degrees-minutes-seconds string with a hemisphere letter.

    Args:
        deg: Signed decimal degrees (WGS84), as
        :func:`xc_viewer.core.igc.parse_igc`
            decodes them.
        axis: ``"lat"`` for N/S, ``"lon"`` for E/W.

    Returns:
        E.g. ``'44 17m 22.03s N'`` in spirit -- the real separators are the degree,
        prime and double-prime signs, not spelled out here to keep this docstring
        plain ASCII.
    """
    if not np.isfinite(deg):
        return "n/a"
    if axis == "lat":
        hemisphere = "N" if deg >= 0 else "S"
    else:
        hemisphere = "E" if deg >= 0 else "W"
    whole = abs(deg)
    degrees = int(whole)
    minutes_full = (whole - degrees) * 60.0
    minutes = int(minutes_full)
    seconds = (minutes_full - minutes) * 60.0
    # The degree/prime/double-prime signs are the correct DMS typography, not stray
    # look-alike characters -- kept, not swapped for their ASCII lint suggestions.
    return f"{degrees}°{minutes:02d}′{seconds:05.2f}″{hemisphere}"  # noqa: RUF001


def resolve_discipline(igc_path: str | Path) -> Discipline | None:
    """Which discipline's archive root a path falls under, if any.

    Used to auto-fill (never lock) the discipline selector when a file is picked
    directly off the disk rather than through the metadata filter, which already
    knows its discipline. ``None`` when the path is outside both configured roots, or
    when neither root is reachable (config absent, disk unmounted) -- the caller falls
    back to whatever the user has selected.
    """
    path = Path(igc_path).resolve()
    for disc in DISCIPLINES.values():
        try:
            root = disc.config().data_root.resolve()
        except (FileNotFoundError, KeyError):
            continue
        if path.is_relative_to(root):
            return disc
    return None


def reachable_disciplines() -> list[Discipline]:
    """Which disciplines currently have a usable ``.igc`` archive on disk.

    "Reachable" means the same thing :func:`resolve_discipline` checks per
    discipline: its config resolves (env var or config file) *and* the directory it
    points at actually exists (data folder present, path still valid). Lets a picker
    decide
    whether choosing a discipline is a real choice -- both reachable -- or a foregone
    conclusion worth not asking about -- exactly one is.

    Returns:
        The reachable disciplines, in
        :data:`~xc_viewer.core.disciplines.DISCIPLINES`
        order. Empty if neither is (nothing configured yet, or the disk is unmounted).
    """
    reachable = []
    for disc in DISCIPLINES.values():
        try:
            cfg = disc.config()
        except (FileNotFoundError, KeyError):
            continue
        if cfg.igc_dir.is_dir():
            reachable.append(disc)
    return reachable
