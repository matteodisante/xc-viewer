"""Year-to-season mapping and FFVL URL construction.

A 'season' is identified by its start year: the year ``1999`` corresponds to season
``1999-2000``. The module is pure (no network), therefore easy to test.
"""

from __future__ import annotations

# Canonical values for the FFVL site (overridable from configuration).
BASE_URL = "https://parapente.ffvl.fr"
LIST_PATH = "/cfd/liste/{year}"
XML_QUERY = "?xml=1"
FLIGHT_PATH = "/cfd/liste/vol/{flight_id}"


def season_label(year: int) -> str:
    """Human-readable label for the season starting in ``year``.

    Args:
        year: Season start year (e.g. ``1999``).

    Returns:
        The ``"YYYY-YYYY"`` label (e.g. ``"1999-2000"``), also used as the name
        of the track subdirectory.
    """
    return f"{year}-{year + 1}"


def list_url(year: int, base_url: str = BASE_URL, list_path: str = LIST_PATH) -> str:
    """URL of the human-readable flight list page for a season.

    Args:
        year: Season start year.
        base_url: Site base URL.
        list_path: URL path template with ``{year}`` placeholder.

    Returns:
        The full list URL (e.g. ``https://parapente.ffvl.fr/cfd/liste/1999``).
    """
    return base_url + list_path.format(year=year)


def xml_url(
    year: int,
    base_url: str = BASE_URL,
    list_path: str = LIST_PATH,
    xml_query: str = XML_QUERY,
) -> str:
    """URL of the complete XML export for a season.

    This is the canonical source we download from: it returns all flights in a single
    response.

    Args:
        year: Season start year.
        base_url: Site base URL.
        list_path: URL path template with ``{year}`` placeholder.
        xml_query: Query string that enables the XML export.

    Returns:
        The full XML export URL (e.g. ``.../cfd/liste/1999?xml=1``).
    """
    return list_url(year, base_url, list_path) + xml_query


def flight_page_url(flight_id: str | int, base_url: str = BASE_URL) -> str:
    """URL of the individual flight page given its ``flight_id``.

    This is the deterministic relationship that allows tracing from the `.igc` filename
    (which contains the ``flight_id``) back to the flight page, without any lookup
    dictionary.

    Args:
        flight_id: Numeric flight identifier.
        base_url: Site base URL.

    Returns:
        The flight page URL (e.g. ``.../cfd/liste/vol/20150770``).
    """
    return base_url + FLIGHT_PATH.format(flight_id=flight_id)


def parse_seasons_arg(value: str, default_start: int, default_end: int) -> list[int]:
    """Parses the CLI ``--seasons`` argument into a list of years.

    Accepted formats:

    * ``"all"`` -> all configured seasons ``[default_start, default_end]``;
    * a single year, e.g. ``"2014"``;
    * a range, e.g. ``"2010-2015"`` (endpoints included);
    * a comma-separated list, e.g. ``"2010,2012,2015"``.

    Args:
        value: Argument string.
        default_start: First year used for ``"all"``.
        default_end: Last year used for ``"all"``.

    Returns:
        Sorted and deduplicated list of season years.

    Raises:
        ValueError: If the string is not in a recognised format.
    """
    value = value.strip().lower()
    if value == "all":
        return list(range(default_start, default_end + 1))

    years: set[int] = set()
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_str, end_str = token.split("-", 1)
            start, end = int(start_str), int(end_str)
            if start > end:
                start, end = end, start
            years.update(range(start, end + 1))
        else:
            years.add(int(token))

    if not years:
        raise ValueError(f"Invalid --seasons argument: {value!r}")
    return sorted(years)
