"""Functional naming for `.igc` files and on-disk paths.

Name scheme: ``{date}_{flight_id}.igc`` (e.g. ``2000-00-00_20150770.igc``).

The name is self-describing: the ``flight_id`` is unique and always links to the flight
page (see :func:`~.seasons.flight_page_url`); therefore, given a `.igc` file, the flight
can be traced back *without* any lookup dictionary.
"""

from __future__ import annotations

from pathlib import Path

from .seasons import season_label

_SAFE_DATE_CHARS = set("0123456789-")


def sanitize_date(date: str) -> str:
    """Makes a date string safe for use in a filename.

    Keeps digits and hyphens; replaces any other character with ``-``. FFVL dates are
    already in ISO form (``YYYY-MM-DD``), sometimes incomplete (``2000-00-00``):
    these are left as-is.

    Args:
        date: The raw date string from the XML.

    Returns:
        A sanitised date; ``"0000-00-00"`` if the input is empty.
    """
    date = (date or "").strip()
    if not date:
        return "0000-00-00"
    return "".join(c if c in _SAFE_DATE_CHARS else "-" for c in date)


def igc_filename(date: str, flight_id: str | int) -> str:
    """Builds the functional `.igc` filename.

    Args:
        date: Flight date (ideally ISO ``YYYY-MM-DD``).
        flight_id: Unique flight identifier.

    Returns:
        The filename, e.g. ``"2000-00-00_20150770.igc"``.
    """
    return f"{sanitize_date(date)}_{flight_id}.igc"


def parse_igc_filename(name: str) -> tuple[str, str]:
    """Inverse of :func:`igc_filename`: extracts the date and ``flight_id``.

    Args:
        name: Name (or path) of the `.igc` file.

    Returns:
        The pair ``(date, flight_id)``.

    Raises:
        ValueError: If the name does not follow the ``{date}_{flight_id}.igc`` scheme.
    """
    stem = Path(name).name
    if not stem.endswith(".igc"):
        raise ValueError(f"Not a .igc file: {name!r}")
    stem = stem[: -len(".igc")]
    if "_" not in stem:
        raise ValueError(f"Filename does not match the expected scheme: {name!r}")
    date, flight_id = stem.rsplit("_", 1)
    return date, flight_id


def igc_path(igc_root: Path, year: int, date: str, flight_id: str | int) -> Path:
    """Full path of a flight's `.igc` file.

    Files are organised in ``{igc_root}/{season}/{name}``, one subdirectory per season.

    Args:
        igc_root: Root directory for track files (``data_root/igc``).
        year: Season start year.
        date: Flight date.
        flight_id: Unique flight identifier.

    Returns:
        The `.igc` file path.
    """
    return igc_root / season_label(year) / igc_filename(date, flight_id)
