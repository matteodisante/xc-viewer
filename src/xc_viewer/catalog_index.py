"""Flight metadata: ``catalog.csv`` filtered, cross-referenced against the pipeline.

No Qt import (see the package docstring). ``catalog.csv`` "is metadata, and it can be
wrong" (``docs/guide/data-on-disk.md``) -- a coarse pre-filter and provenance source,
never a basis for a scientific cut; this module treats it exactly that way, as a means
to *find* a flight to look at, never as a source of the trajectory itself (that always
comes from the ``.igc`` file, through :mod:`xc_viewer.data`).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import geography
from .core.disciplines import Discipline
from .core.naming import igc_path as build_igc_path

# One in-process cache per discipline: catalog.csv is tens of MB, worth loading once
# per session rather than once per filter call.
_catalog_cache: dict[str, pd.DataFrame] = {}
_flights_meta_cache: dict[str, pd.DataFrame | None] = {}

# The site itself writes this in place of a `pilot` name for a privacy-protected
# profile -- a bcrypt hash and salt, not a value anyone would ever filter on -- so
# distinct_values() excludes it rather than clutter a name dropdown with ~150 of them.
_ANONYMISED_MARKER = "$2A$07$FFVL"


def clear_cache() -> None:
    """Drop the cached catalog/flights_meta tables.

    Only needed after a discipline's data root is repointed mid-session (a new
    ``Discipline.config().data_root``, e.g. from a folder picker) -- the cache is
    keyed by discipline name, not by path, so without this it would keep serving the
    previous root's rows.
    """
    _catalog_cache.clear()
    _flights_meta_cache.clear()


def is_anonymised_pilot(value: str) -> bool:
    """Whether ``value`` is the FFVL privacy placeholder, not a real pilot name."""
    return _ANONYMISED_MARKER in value


def _load_catalog(discipline: Discipline) -> pd.DataFrame:
    if discipline.name not in _catalog_cache:
        path = discipline.catalog_path()
        if path is None or not path.is_file():
            raise FileNotFoundError(
                f"no catalog.csv reachable for {discipline.name} "
                f"(disk unmounted, or {discipline.env} not set)"
            )
        df = pd.read_csv(path, dtype={"flight_id": str}, low_memory=False)
        # Every non-placeholder date is ISO YYYY-MM-DD (docs/guide/data-on-disk.md);
        # a fixed format both silences pandas's per-row dateutil fallback (which is
        # what makes parsing 200k+ rows slow) and coerces every placeholder
        # ("0000-00-00", "2000-00-00": an invalid month/day) to NaT, same as before.
        parsed_date = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
        df["_month"] = parsed_date.dt.month
        _catalog_cache[discipline.name] = df
    return _catalog_cache[discipline.name]


def _load_flights_meta(discipline: Discipline) -> pd.DataFrame | None:
    """Per-flight pipeline verdict plus the region/terrain labels.

    Both derived from ``lat0``/``lon0``/``alt0``, so a flight with no local frame
    (dropped before that pipeline stage) carries ``""`` for each, the same as a
    flight absent from this table altogether.
    """
    if discipline.name not in _flights_meta_cache:
        derived = discipline.derived_dir(require="flights_meta.parquet")
        if derived is None:
            _flights_meta_cache[discipline.name] = None
        else:
            meta = pd.read_parquet(
                derived / "flights_meta.parquet",
                columns=["flight_id", "lat0", "lon0", "alt0", "drop_reason"],
            )
            meta = meta.drop_duplicates(subset="flight_id", keep="last")
            meta["kept"] = meta["drop_reason"].isna()
            meta["region"] = geography.classify_region(meta["lat0"], meta["lon0"])
            meta["terrain"] = geography.classify_terrain(meta["alt0"])
            _flights_meta_cache[discipline.name] = meta[
                ["flight_id", "kept", "drop_reason", "region", "terrain"]
            ]
    return _flights_meta_cache[discipline.name]


# Catalog columns filterable by plain case-insensitive exact match, and the keyword
# argument of filter_flights() each corresponds to (same name, checked below by a
# loop rather than a repetitive if-chain, one per column).
_TEXT_FILTER_COLUMNS = (
    "dept",
    "flight_type",
    "wing_class",
    "takeoff",
    "landing",
    "club",
    "wing",
    "pilot",
)


def filter_flights(
    discipline: Discipline,
    *,
    dept: str | None = None,
    season_year: int | None = None,
    month: int | None = None,
    flight_type: str | None = None,
    wing_class: str | None = None,
    takeoff: str | None = None,
    landing: str | None = None,
    club: str | None = None,
    wing: str | None = None,
    pilot: str | None = None,
    region: str | None = None,
    terrain: str | None = None,
    kept_only: bool = False,
) -> pd.DataFrame:
    """Catalog rows matching every given filter, joined against the pipeline's verdict.

    ``dept`` is the catalog's actual column -- the French *departement* (an INSEE
    code, e.g. ``"38"`` for Isere, not a name, and noted in
    ``docs/guide/data-on-disk.md`` as carrying its share of junk: ``"0"``, ``"999"``,
    a stray ``"38660"``), not an administrative *region*, which the catalog does not
    carry. Every text filter below matches whatever :func:`distinct_values` offers for
    that same column, so a dropdown built from it can never propose a value that then
    matches nothing.

    ``region`` and ``terrain``, unlike every other filter here, are not catalog
    columns: they are computed from ``flights_meta.parquet``'s ``lat0``/``lon0``/
    ``alt0`` (:func:`xc_viewer.geography.classify_region`/``classify_terrain``),
    so a flight excluded by ``kept_only``'s reasoning -- unprocessed, or retained but
    with no local frame -- also fails to match either one.

    Args:
        discipline: Which archive to search.
        dept: Exact match against ``dept`` (an INSEE department code), case-insensitive.
        season_year: Exact match against ``season_year``.
        month: Calendar month (1-12) parsed from ``date``; a placeholder date
            (``"0000-00-00"``) matches nothing.
        flight_type: Exact match against ``flight_type``, case-insensitive.
        wing_class: Exact match against ``wing_class``, case-insensitive.
        takeoff: Exact match against the take-off site name, case-insensitive.
        landing: Exact match against the landing site name, case-insensitive.
        club: Exact match against the pilot's club, case-insensitive.
        wing: Exact match against the wing model, case-insensitive.
        pilot: Exact match against the pilot name, case-insensitive.
        region: One of :data:`xc_viewer.geography.REGIONS` (``"Alps"``,
            ``"Pyrenees"``, ``"Channel Coast"``): keep only take-offs inside that box.
        terrain: One of :data:`xc_viewer.geography.TERRAIN_ORDER` (``"Plains"``,
            ``"Hills"``, ``"Low mountains"``, ``"High mountains"``): keep only
            take-offs in that elevation band.
        kept_only: If ``True``, keep only flights the pipeline retained (a flight not
            found in ``flights_meta.parquet`` -- never processed -- is excluded too).

    Returns:
        A copy of the matching catalog rows, with a ``kept`` column added: ``True`` /
        ``False`` from the pipeline's verdict, ``pd.NA`` where ``flights_meta.parquet``
        is unreachable or the flight is not in it. ``pipeline_status`` explains
        that distinction; ``drop_reason``, ``region`` and ``terrain`` are included
        when archive results exist.

    Raises:
        FileNotFoundError: If this discipline's ``catalog.csv`` is not reachable.
    """
    df = _load_catalog(discipline)
    mask = pd.Series(True, index=df.index)
    text_filters = {
        "dept": dept,
        "flight_type": flight_type,
        "wing_class": wing_class,
        "takeoff": takeoff,
        "landing": landing,
        "club": club,
        "wing": wing,
        "pilot": pilot,
    }
    for column in _TEXT_FILTER_COLUMNS:
        value = text_filters[column]
        if value is not None:
            mask &= df[column].astype(str).str.casefold() == value.casefold()
    if season_year is not None:
        mask &= df["season_year"] == season_year
    if month is not None:
        mask &= df["_month"] == month

    result = df.loc[mask].drop(columns="_month").copy()
    meta = _load_flights_meta(discipline)
    if meta is not None:
        result = result.merge(meta, on="flight_id", how="left")
    else:
        result["kept"] = pd.NA
        result["region"] = ""
        result["terrain"] = ""
    result["pipeline_status"] = (
        result["kept"]
        .map({True: "Kept", False: "Dropped"})
        .fillna(
            "No archived result" if meta is not None else "Pipeline results unavailable"
        )
    )
    if kept_only:
        result = result[result["kept"].fillna(False).astype(bool)]
    if region is not None:
        result = result[result["region"] == region]
    if terrain is not None:
        result = result[result["terrain"] == terrain]
    return result.reset_index(drop=True)


def catalog_rows(discipline: Discipline, flight_ids) -> pd.DataFrame:
    """The catalog rows of ``flight_ids`` (strings), one per flight.

    Raises:
        FileNotFoundError: If this discipline's ``catalog.csv`` is not reachable.
    """
    df = _load_catalog(discipline)
    rows = df[df["flight_id"].isin([str(f) for f in flight_ids])]
    return (
        rows.drop(columns="_month").drop_duplicates("flight_id").reset_index(drop=True)
    )


def distinct_values(discipline: Discipline, column: str) -> list[str]:
    """Sorted distinct non-blank values of one catalog column, for filter dropdowns.

    Lets the picker offer a menu of what the archive actually contains -- department
    codes, take-off site names, wing models -- instead of making the user guess the
    right spelling or code (the catalog's own quirks are documented in
    ``docs/guide/data-on-disk.md``).

    Args:
        discipline: Which archive to search.
        column: A column of ``catalog.csv`` (e.g. ``"dept"``, ``"takeoff"``).

    Returns:
        Sorted distinct values, as strings; blank, missing and FFVL-anonymised
        (``pilot`` for a privacy-protected profile: a bcrypt hash, not a name --
        1.7% of the paraglider ``pilot`` column) values excluded.

    Raises:
        FileNotFoundError: If this discipline's ``catalog.csv`` is not reachable.
    """
    df = _load_catalog(discipline)
    values = df[column].dropna().astype(str).str.strip()
    anonymised = values.str.contains(_ANONYMISED_MARKER, regex=False)
    values = values[(values != "") & ~anonymised]
    return sorted(values.unique().tolist())


# Catalog columns the map's hover tooltip shows alongside a point, on top of the
# flight_id/season_year/date/lat0/lon0 every takeoff_points() row already carries.
TOOLTIP_COLUMNS = [
    "pilot",
    "flight_type",
    "wing_class",
    "distance_km",
    "duration_s",
    "takeoff",
    "landing",
    "dept",
]


def takeoff_points(discipline: Discipline) -> pd.DataFrame:
    """Every retained flight's launch point, for the map view.

    Reads ``flights_meta.parquet`` afresh (not through :func:`_load_flights_meta`'s
    cache, which keeps only ``flight_id``/``kept``) for the columns the map needs on
    top of that: ``lat0``/``lon0``/``alt0`` (the last for the map's terrain-band
    filter -- see :func:`xc_viewer.geography.classify_terrain`). Joined against
    the catalog for ``season_year``/``date`` (so a clicked point can be resolved to
    its ``.igc`` file with :func:`resolve_igc_path`, the same way a catalog-search
    result is) and :data:`TOOLTIP_COLUMNS` (what a hovered point shows).

    Args:
        discipline: Which archive to read.

    Returns:
        Columns ``flight_id``, ``season_year``, ``date``, ``lat0``, ``lon0``,
        ``alt0``, plus :data:`TOOLTIP_COLUMNS` -- one row per flight the pipeline
        retained. Empty (same columns) if ``flights_meta.parquet`` is unreachable.

    Raises:
        FileNotFoundError: If this discipline's ``catalog.csv`` is not reachable.
    """
    columns = ["flight_id", "season_year", "date", *TOOLTIP_COLUMNS]
    catalog = _load_catalog(discipline)[columns]
    derived = discipline.derived_dir(require="flights_meta.parquet")
    if derived is None:
        empty = pd.Series(dtype=float)
        return catalog.iloc[0:0].assign(lat0=empty, lon0=empty, alt0=empty)

    meta = pd.read_parquet(
        derived / "flights_meta.parquet",
        columns=["flight_id", "lat0", "lon0", "alt0", "drop_reason"],
    )
    meta = meta.drop_duplicates(subset="flight_id", keep="last")
    has_origin = meta["lat0"].notna() & meta["lon0"].notna()
    meta = meta[meta["drop_reason"].isna() & has_origin]
    return catalog.merge(
        meta[["flight_id", "lat0", "lon0", "alt0"]], on="flight_id", how="inner"
    ).reset_index(drop=True)


def resolve_igc_path(discipline: Discipline, row: pd.Series) -> Path | None:
    """The ``.igc`` file for one :func:`filter_flights` row, if findable on disk.

    Tries the functional path first (``season_year`` + ``date`` + ``flight_id``,
    :func:`xc_viewer.core.naming.igc_path`); catalog dates are sometimes a
    placeholder (``"0000-00-00"``), so on a miss this falls back to searching the
    season's directory for the flight_id suffix the naming scheme guarantees.

    Returns:
        The path, or ``None`` if neither attempt finds a file (the flight was listed
        but never downloaded, the discipline's disk is unreachable, or the row's
        season/date cannot place it).
    """
    try:
        cfg = discipline.config()
    except (FileNotFoundError, KeyError):
        return None
    flight_id = str(row["flight_id"])
    candidate = build_igc_path(
        cfg.igc_dir, int(row["season_year"]), str(row["date"]), flight_id
    )
    if candidate.is_file():
        return candidate
    season_dir = candidate.parent
    if season_dir.is_dir():
        matches = list(season_dir.glob(f"*_{flight_id}.igc"))
        if matches:
            return matches[0]
    return None
