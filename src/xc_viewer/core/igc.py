"""Parser for the ``B`` (fix) records of an IGC flight log.

An IGC file (Chapter "The dataset") stores one GPS fix per ``B`` record, at character
positions fixed by the FAI/IGC standard. This module turns a file into a tidy per-fix
table -- the basis of every downstream trajectory analysis, which so far has had only
the catalog metadata to work with.

A ``B`` record has the layout (positions are 0-based, half-open)::

    B HHMMSS DDMMmmm N DDDMMmmm E V ppppp ggggg  ...optional I-record extensions...
    0 1    7 7    15   15    24   24 25 30 30 35

so the fields read here are: UTC time ``[1:7]``, latitude ``[7:15]`` and longitude
``[15:24]`` (degrees, minutes*1000, hemisphere), the validity flag ``[24]``
(``A``/``V``), and the two altitudes in metres -- **barometric** (pressure) ``[25:30]``
then **GNSS** ``[30:35]``. A missing channel is written as zero by the standard.

The parser deliberately does *not* choose an altitude channel: it returns both, so the
pre-processing can adopt the GNSS one (the thesis choice) while keeping the barometric
one available as the frozen-lock witness and for the noise diagnostics.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Column layout of the returned table.
COLUMNS = ["t", "lat", "lon", "valid", "baro_alt", "gnss_alt"]

# A UTC midnight roll-over makes the time of day drop by (almost) a whole day. Only a
# drop this large is treated as a new day; a smaller backward step is an out-of-order or
# corrupted fix, not a roll-over (see the unwrap logic in :func:`parse_igc`).
# How close to midnight both ends of a drop must sit for it to be a roll-over rather
# than a broken clock. One hour: a real roll-over has a gap of at most the cadence, so
# the margin is generous, while a lost-clock 00:00:00 written in the afternoon is hours
# away from it.
_MIDNIGHT_EDGE_S = 3600


def _altitude(token: str) -> float:
    """Decode a five-character altitude field, or ``nan`` if it is not one.

    The IGC field is five digits, optionally signed in the barometric case. Loggers
    do write it blank, dashed or otherwise unusable; that is a missing altitude and
    nothing more, so it is returned as ``nan`` rather than raised, and the fix keeps
    its position and its timestamp.
    """
    token = token.strip()
    if not token or not (token.lstrip("+-").isdigit() and token.lstrip("+-")):
        return float("nan")
    return float(int(token))


_MIDNIGHT_WRAP_MIN_DROP_S = 43200


def _lat(token: str) -> float:
    """Decode an 8-char latitude token ``DDMMmmmH`` into signed degrees.

    Raises:
        ValueError: If the minutes field is not a valid ``< 60`` arc-minute value, the
            hemisphere letter is not ``N``/``S``, or the decoded value falls outside
            ``[-90, 90]`` -- i.e. the token is not a valid encoding, not merely an
            unlikely one.
    """
    # `int` accepts a leading sign and surrounding whitespace, so "-12" and " 12" both
    # parse and both land inside their bound -- decoding to a position that is merely
    # *wrong* rather than rejected, which is the one outcome a validity check must not
    # allow. Both fields are fixed-width digits by the format, so both are required to
    # be digits.
    if not token[0:2].isdigit():
        raise ValueError(f"invalid latitude degrees: {token!r}")
    deg = int(token[0:2])
    if not token[2:7].isdigit():
        raise ValueError(f"invalid latitude minutes: {token!r}")
    minutes_raw = int(token[2:7])
    if minutes_raw >= 60000:
        raise ValueError(f"invalid latitude minutes: {token!r}")
    minutes = minutes_raw / 1000.0
    hemi = token[7]
    if hemi not in "NnSs":
        raise ValueError(f"invalid latitude hemisphere: {token!r}")
    value = deg + minutes / 60.0
    if value > 90.0:
        raise ValueError(f"latitude out of range: {token!r}")
    return -value if hemi in "Ss" else value


def _lon(token: str) -> float:
    """Decode a 9-char longitude token ``DDDMMmmmH`` into signed degrees.

    Raises:
        ValueError: If the minutes field is not a valid ``< 60`` arc-minute value, the
            hemisphere letter is not ``E``/``W``, or the decoded value falls outside
            ``[-180, 180]`` -- i.e. the token is not a valid encoding, not merely an
            unlikely one.
    """
    if not token[0:3].isdigit():
        raise ValueError(f"invalid longitude degrees: {token!r}")
    deg = int(token[0:3])
    if not token[3:8].isdigit():
        raise ValueError(f"invalid longitude minutes: {token!r}")
    minutes_raw = int(token[3:8])
    if minutes_raw >= 60000:
        raise ValueError(f"invalid longitude minutes: {token!r}")
    minutes = minutes_raw / 1000.0
    hemi = token[8]
    if hemi not in "EeWw":
        raise ValueError(f"invalid longitude hemisphere: {token!r}")
    value = deg + minutes / 60.0
    if value > 180.0:
        raise ValueError(f"longitude out of range: {token!r}")
    return -value if hemi in "Ww" else value


def _valid_time_of_day(hh: int, mm: int, ss: int) -> bool:
    """Whether ``hh:mm:ss`` is a valid 24-hour UTC time-of-day."""
    return 0 <= hh <= 23 and 0 <= mm <= 59 and 0 <= ss <= 59


def _read_lines(path: str | Path) -> list[str]:
    """Read an IGC file as text, tolerant of latin-1 bytes and CRLF endings."""
    raw = Path(path).read_bytes()
    text = raw.decode("latin-1", errors="replace")
    return text.replace("\r\n", "\n").replace("\r", "\n").split("\n")


def parse_igc(path: str | Path) -> pd.DataFrame:
    """Parse the ``B`` records of an IGC file into a per-fix table.

    Args:
        path: Path to the ``.igc`` file.

    Returns:
        A DataFrame with columns ``t`` (seconds elapsed from the first fix, with the
        midnight roll-over unwrapped but a backward step left where the log has one --
        it is a defect for the cleaning to remove, not for the parser to hide),
        ``lat``/``lon`` (signed degrees, WGS84),
        ``valid`` (``True`` for an ``A`` fix, ``False`` for ``V``), and ``baro_alt``/
        ``gnss_alt`` (metres; ``0`` where the channel is absent). A record is skipped
        (not just malformed in shape) if its UTC time is not a valid 24-hour
        ``hh:mm:ss``, or if its latitude/longitude do not decode to a valid position
        (arc-minutes ``< 60``, correct hemisphere letter, within
        ``[-90, 90]``/``[-180, 180]``); this is a format-validity check, not a
        plausibility judgement on the flight dynamics, which fix-level cleaning
        (thesis, sec:fixlevel) handles separately. A file with no valid fix yields an
        empty table with the same columns.
    """
    sec_of_day: list[int] = []
    lats: list[float] = []
    lons: list[float] = []
    valids: list[bool] = []
    baros: list[float] = []
    gnsss: list[float] = []

    for line in _read_lines(path):
        if not line.startswith("B") or len(line) < 35:
            continue
        try:
            hh, mm, ss = int(line[1:3]), int(line[3:5]), int(line[5:7])
            if not _valid_time_of_day(hh, mm, ss):
                continue  # not a ValueError (ints parsed fine): explicit range check
            lat = _lat(line[7:15])
            lon = _lon(line[15:24])
        except ValueError:
            continue  # corrupted record: skip it rather than abort the file
        # The altitudes are decoded *after* the record has been accepted, and
        # separately from each other. A blank or dashed altitude field is a missing
        # value, not a corrupt fix: the whole asymmetry of stage (ii) is that a bad
        # altitude costs the altitude and never the position (sec:fixlevel). Sharing
        # one `try` with the time and the coordinates threw all three away, so a
        # logger that blank-pads the field lost every fix of the flight.

        baro = _altitude(line[25:30])
        gnss = _altitude(line[30:35])
        sec_of_day.append(hh * 3600 + mm * 60 + ss)
        lats.append(lat)
        lons.append(lon)
        valids.append(line[24] in "Aa")
        baros.append(baro)
        gnsss.append(gnss)

    if not sec_of_day:
        return pd.DataFrame({c: pd.Series(dtype="float64") for c in COLUMNS})

    sod = np.asarray(sec_of_day, dtype=np.int64)
    # Rebuild the elapsed time from the wall-clock time of day, which resets to 0 at
    # each UTC midnight. Only a drop of (nearly) a whole day is a roll-over; a *small*
    # backward step is an out-of-order or corrupted fix, not a new day. The naive "any
    # decrease adds a day" would turn a few-second GPS glitch into a spurious +86400 s
    # jump, manufacturing a multi-hour gap and a wildly wrong duration. So only large
    # drops advance the day counter.
    #
    # Residual backward jitter is deliberately *left in place*. Flattening it with a
    # running maximum, as this parser used to, is a repair, and repairing it here would
    # destroy the evidence the fix-level cleaning has to act on: a backward timestamp is
    # a defect to be recorded and removed by minimal deletion, not silently absorbed
    # (thesis, sec:fixlevel "Time-base defects"). The returned ``t`` is therefore
    # elapsed seconds from the first fix, monotonic *except* where the log itself is
    # not.
    #
    # A large drop is necessary but not sufficient. A genuine roll-over also *looks*
    # like one: the fix before it sits just under midnight and the one after it just
    # over. A logger that loses its clock writes 00:00:00 mid-flight, which from an
    # afternoon value is a drop of tens of thousands of seconds and was read as a new
    # day -- adding 86400 s to the whole remainder of the record, manufacturing a
    # day-long gap and a duration an order of magnitude too large. Requiring both ends
    # to be near midnight costs nothing on a real roll-over and refuses that one.
    diffs = np.diff(sod)
    wraps = (
        (diffs < -_MIDNIGHT_WRAP_MIN_DROP_S)
        & (sod[:-1] >= 86400 - _MIDNIGHT_EDGE_S)
        & (sod[1:] <= _MIDNIGHT_EDGE_S)
    )
    day_wraps = np.concatenate([[0], np.cumsum(wraps)])
    t = (sod + 86400 * day_wraps).astype(float)
    t -= t[0]

    return pd.DataFrame(
        {
            "t": t,
            "lat": np.asarray(lats, dtype=float),
            "lon": np.asarray(lons, dtype=float),
            "valid": np.asarray(valids, dtype=bool),
            "baro_alt": np.asarray(baros, dtype=float),
            "gnss_alt": np.asarray(gnsss, dtype=float),
        },
        columns=COLUMNS,
    )


# How much of a raw file to read looking for the first valid B record, before falling
# back to the whole file. Comfortably covers the header plus many B records even at a
# fast logging cadence, so the fallback path is rare in practice.
_FIRST_FIX_CHUNK_BYTES = 65536


def _first_valid_fix(
    lines: list[str], *, include_utc: bool = False
) -> dict[str, float] | None:
    """The first line in ``lines`` that decodes as a valid ``B`` record, or ``None``.

    Same per-line validity rules as :func:`parse_igc` (time-of-day, lat/lon range and
    hemisphere), kept in one place there and reused here rather than duplicated, since a
    fix this function accepts and ``parse_igc`` would reject (or vice versa) is exactly
    the kind of drift that makes "first raw fix" quietly stop meaning what
    :func:`parse_igc` means by a fix.
    """
    day = None
    for line in lines:
        if include_utc and (match := re.match(r"H[FO]DTE(?:DATE:)?(\d{6})", line)):
            token = match[1]
            try:
                year = int(token[4:6])
                day = datetime(
                    1900 + year if year >= 70 else 2000 + year,
                    int(token[2:4]),
                    int(token[:2]),
                    tzinfo=UTC,
                )
            except ValueError:
                day = None
        if not line.startswith("B") or len(line) < 35:
            continue
        try:
            hh, mm, ss = int(line[1:3]), int(line[3:5]), int(line[5:7])
            if not _valid_time_of_day(hh, mm, ss):
                continue
            lat = _lat(line[7:15])
            lon = _lon(line[15:24])
        except ValueError:
            continue
        result = {
            "lat": lat,
            "lon": lon,
            "baro_alt": _altitude(line[25:30]),
            "gnss_alt": _altitude(line[30:35]),
        }
        if include_utc:
            result["start_utc"] = (
                day.timestamp() + hh * 3600 + mm * 60 + ss
                if day is not None
                else float("nan")
            )
        return result
    return None


def first_fix(
    path: str | Path, *, include_utc: bool = False
) -> dict[str, float] | None:
    """The earliest valid ``B`` record of a raw IGC file, decoded but not cleaned.

    Unlike :func:`parse_igc`, this does not parse the whole flight: it reads one chunk
    from the front of the file (header plus a handful of fixes, almost always enough)
    and only falls back to the complete file for the rare log whose leading records are
    all corrupt. Scanning for "the altitude a pilot launched at" this way
    costs a few kilobytes per flight rather than the whole track.

    Args:
        path: Path to the ``.igc`` file.
        include_utc: Also return ``start_utc`` (Unix seconds) from the IGC date
            header and first accepted B record; NaN when the date is unavailable.

    Returns:
        A dict with ``lat``, ``lon``, ``baro_alt``, ``gnss_alt`` (same decoding as
        :func:`parse_igc`, ``nan`` for an absent altitude channel), or ``None`` if the
        file has no valid ``B`` record at all.
    """
    path = Path(path)
    with path.open("rb") as fh:
        chunk = fh.read(_FIRST_FIX_CHUNK_BYTES)
        chunk_is_whole_file = len(chunk) < _FIRST_FIX_CHUNK_BYTES
        text = chunk.decode("latin-1", errors="replace")
        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        # The last line may be cut mid-record at the chunk boundary; drop it unless the
        # chunk was short enough that it is the whole file and thus already complete.
        found = _first_valid_fix(
            lines if chunk_is_whole_file else lines[:-1], include_utc=include_utc
        )
        if found is not None or chunk_is_whole_file:
            return found
        rest = fh.read()
    text = (chunk + rest).decode("latin-1", errors="replace")
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    return _first_valid_fix(lines, include_utc=include_utc)


def _present_fraction(fixes: pd.DataFrame, column: str) -> float:
    """Fraction of fixes carrying a usable value on one altitude channel.

    The IGC format writes an absent altitude channel as zero, so a zero is read as "no
    value here" rather than as a measurement, and a blank or unusable field decodes to
    ``nan`` (see :func:`_altitude`). Both count as absent.

    The ``nan`` half of that is not hypothetical. ``nan != 0.0`` is ``True``, so a
    logger writing blanks used to be reported as carrying a channel at full presence --
    the opposite of the truth, and enough to have the pipeline adopt an altitude that is
    not there.

    Args:
        fixes: Table returned by :func:`parse_igc`.
        column: ``"baro_alt"`` or ``"gnss_alt"``.

    Returns:
        The fraction in ``[0, 1]`` (``0.0`` for an empty table).
    """
    if len(fixes) == 0:
        return 0.0
    alt = fixes[column].to_numpy(dtype=float)
    return float((np.isfinite(alt) & (alt != 0.0)).mean())


def baro_present_fraction(fixes: pd.DataFrame) -> float:
    """Fraction of fixes carrying a non-zero barometric altitude.

    A value near ``0`` means the logger has no pressure sensor (the whole ``baro_alt``
    channel is written as zero). The analysis reads the GNSS altitude
    (thesis, sec:altchannel), so what this measures is the *witness* channel: the
    independent instrument that tells a frozen GNSS lock from a genuine slow climb, and
    the quantity the archive census reports.

    Args:
        fixes: Table returned by :func:`parse_igc`.

    Returns:
        The fraction in ``[0, 1]`` (``0.0`` for an empty table).
    """
    return _present_fraction(fixes, "baro_alt")


def gnss_present_fraction(fixes: pd.DataFrame) -> float:
    """Fraction of fixes carrying a non-zero GNSS altitude.

    The presence test on the *adopted* channel (thesis, sec:altchannel), so unlike its
    barometric counterpart it decides whether the flight has a vertical coordinate at
    all: below the configured threshold the flight is dropped, there being no other
    channel to move it to. Some loggers write a pressure altitude and leave the GNSS
    field at ``00000``, and that is exactly the population this sizes.

    Args:
        fixes: Table returned by :func:`parse_igc`.

    Returns:
        The fraction in ``[0, 1]`` (``0.0`` for an empty table).
    """
    return _present_fraction(fixes, "gnss_alt")


def median_sampling_period(fixes: pd.DataFrame) -> float:
    """Median inter-fix interval in seconds (``nan`` if fewer than two fixes)."""
    if len(fixes) < 2:
        return float("nan")
    return float(np.median(np.diff(fixes["t"].to_numpy())))
