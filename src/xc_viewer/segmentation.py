"""The saved flight-phase segmentations the viewer can read, and the one in use.

Two sets of saved runs sit in each discipline's ``derived/segmentation/`` folder, with
the same files: ``phase_segments.parquet``, ``phase_coverage.parquet`` and
``model/parameters.json``. The choice is process-wide, like the data folder
(:mod:`xc_viewer.datafolder`): every tab reads it here. No Qt import; the main window
supplies the combo box and the remembered setting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .core.disciplines import DISCIPLINES

#: Key of the last choice, in the app's ``QSettings``.
SETTINGS_KEY = "segmentation"

#: Files of a saved segmentation, relative to its folder.
FILES = {
    "runs": "phase_segments.parquet",
    "coverage": "phase_coverage.parquet",
    "parameters": "model/parameters.json",
}


@dataclass(frozen=True)
class Segmentation:
    """One saved segmentation.

    Attributes:
        key: The name used in code, cache keys and file names.
        label: The name shown to the user.
        folder: Its folder under a discipline's ``derived/``.
        mapping_method: The tag carried by the :class:`~xc_viewer.data.PhaseTrack`.
        credit: One line crediting the model and the data it was fitted on.
    """

    key: str
    label: str
    folder: str
    mapping_method: str
    credit: str


VILPELLET = Segmentation(
    key="vilpellet",
    label="Vilpellet",
    folder="segmentation/vilpellet",
    mapping_method="vilpellet-fitted-2019",
    credit="Vilpellet et al. (2026): model fitted on 2019 Alpine paraglider flights.",
)
REFIT = Segmentation(
    key="refit",
    label="Refit",
    folder="segmentation/refit",
    mapping_method="refit-2026",
    credit="Refit of Vilpellet's model on the FFVL flights.",
)
SEGMENTATIONS = {s.key: s for s in (VILPELLET, REFIT)}
DEFAULT = VILPELLET.key

_current = DEFAULT


def get(key: str) -> Segmentation:
    """The segmentation named ``key``; ``ValueError`` if there is none."""
    try:
        return SEGMENTATIONS[key]
    except KeyError:
        raise ValueError(f"Unknown segmentation: {key}") from None


def current() -> Segmentation:
    """The segmentation every tab reads from now on."""
    return SEGMENTATIONS[_current]


def use(key: str) -> Segmentation:
    """Read every tab from the segmentation ``key`` from now on."""
    global _current
    _current = get(key).key
    return SEGMENTATIONS[_current]


def product_name(name: str, key: str | None = None) -> str:
    """The file name of one segmentation's own copy of a saved viewer product.

    Vilpellet keeps ``name`` as it is, so every data folder already holding its
    products keeps working; any other segmentation gets ``-<key>`` before the
    extension (``thermal-planes.sqlite3`` becomes ``thermal-planes-refit.sqlite3``).
    """
    key = current().key if key is None else get(key).key
    if key == VILPELLET.key:
        return name
    path = Path(name)
    return f"{path.stem}-{key}{path.suffix}"


def folder(discipline, key: str | None = None) -> Path:
    """The folder of a segmentation inside one discipline's archive."""
    seg = current() if key is None else get(key)
    return discipline.config().derived_dir / seg.folder


def paths(discipline, key: str | None = None) -> dict[str, Path]:
    """The files of a segmentation inside one discipline's archive."""
    root = folder(discipline, key)
    return {name: root / relative for name, relative in FILES.items()}


def missing(disciplines=None, key: str | None = None) -> list[str]:
    """The saved files a segmentation lacks, for a message saying what it needs.

    Args:
        disciplines: The disciplines to check; both when omitted. A discipline whose
            archive folder is absent from the data folder is skipped.
        key: The segmentation; the current one when omitted.

    Returns:
        Paths relative to the data folder, empty when everything is there.
    """
    found = []
    for discipline in DISCIPLINES.values() if disciplines is None else disciplines:
        try:
            root = discipline.config().data_root
        except (FileNotFoundError, KeyError):
            continue
        if not root.is_dir():
            continue
        found += [
            f"{discipline.folder}/{path.relative_to(root).as_posix()}"
            for name, path in paths(discipline, key).items()
            if name != "parameters" and not path.is_file()
        ]
    return found


def needs_text(key: str | None = None, disciplines=None) -> str:
    """One sentence saying which saved files a segmentation lacks, or ``""``."""
    absent = missing(disciplines, key)
    if not absent:
        return ""
    seg = current() if key is None else get(key)
    return f"The {seg.label} segmentation needs {', '.join(absent)} in the data folder."
