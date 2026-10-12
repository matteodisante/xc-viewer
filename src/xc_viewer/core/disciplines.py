"""The two disciplines, and where each one's archive sits inside the data folder.

The viewer reads everything from one data folder chosen at startup. Inside it each
discipline has its own archive root::

    <data folder>/
        paragliders/     raw/igc/<season>/*.igc, catalog/catalog.csv, derived/...
        hang_gliders/    the same layout

A per-discipline environment variable can point one discipline elsewhere.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .style import DISCIPLINE_COLORS

#: Environment variable holding the data folder. The app sets it at startup from the
#: command line, the last folder used, or a folder dialog.
DATA_FOLDER_ENV = "XC_VIEWER_DATA"


@dataclass(frozen=True)
class DataRoot:
    """One discipline's archive root and the fixed layout under it.

    Attributes:
        data_root: The discipline's archive root.
    """

    data_root: Path

    @property
    def igc_dir(self) -> Path:
        """Root directory for `.igc` track files (one subdirectory per season)."""
        return self.data_root / "raw" / "igc"

    @property
    def catalog_path(self) -> Path:
        """Path to the flight catalog CSV."""
        return self.data_root / "catalog" / "catalog.csv"

    @property
    def derived_dir(self) -> Path:
        """Directory of the processed tables and the viewer's saved products."""
        return self.data_root / "derived"


@dataclass(frozen=True)
class Discipline:
    r"""One glider discipline: its names, its data root, and the colour it is drawn in.

    Attributes:
        name: The key the scripts index by, and what a message prints
            (``"paragliders"``, ``"hang gliders"``).
        slug: Filename component of the intermediate arrays (``"para"``, ``"hang"``).
        source: The value of the ``source`` column in the ``fixes`` schema
            (``"paraglider"``, ``"hangglider"``). Deliberately unlike ``name`` and
            unlike the on-disk directory: a new archive is a new value of this column,
            never a new column, so it is the one field the pipeline itself reads.
        tag: Macro-name component (``"Para"``, ``"Hang"``), as in
            ``\\StatMsdParaAlpha``.
        env: Environment variable overriding ``data_root`` for this discipline.
        folder: Name of this discipline's archive root inside the data folder.
        color: The colour every figure uses for this discipline, so that a panel added
            later cannot disagree with the ones already printed.
    """

    name: str
    slug: str
    source: str
    tag: str
    env: str
    folder: str
    color: str

    def config(self) -> DataRoot:
        """This discipline's archive root.

        Returns:
            The :class:`DataRoot` from this discipline's own environment variable if
            set, otherwise ``<data folder>/<folder>``.

        Raises:
            FileNotFoundError: If neither the override nor the data folder is set.
        """
        override = os.environ.get(self.env)
        if override:
            return DataRoot(Path(override).expanduser().resolve())
        folder = os.environ.get(DATA_FOLDER_ENV)
        if folder:
            return DataRoot(Path(folder).expanduser().resolve() / self.folder)
        raise FileNotFoundError("No data folder chosen")

    def derived_dir(self, require: str = "fixes.parquet") -> Path | None:
        """This discipline's ``derived/`` directory, or ``None`` if it is not reachable.

        Unreachable covers both no data folder being chosen and the folder missing,
        and the callers treat them the same way: skip this discipline, report what was
        written for the other. A missing archive is the normal state of a fresh
        checkout, not an error.

        Args:
            require: The table that must exist for the directory to count as usable.
                ``fixes.parquet`` for a pass that streams the fix table,
                ``flights_meta.parquet`` for a reduction that needs the flight rows.

        Returns:
            The directory, or ``None``.
        """
        try:
            cfg = self.config()
        except (FileNotFoundError, KeyError):
            return None
        return cfg.derived_dir if (cfg.derived_dir / require).is_file() else None

    def catalog_path(self) -> Path | None:
        """The acquisition catalogue for this discipline, or ``None`` if unreachable.

        Returns:
            The path to ``catalog.csv``, which is metadata and can be wrong -- a coarse
            pre-filter and a provenance source, never the basis of a cut.
        """
        try:
            return self.config().catalog_path
        except (FileNotFoundError, KeyError):
            return None


PARAGLIDERS = Discipline(
    name="paragliders",
    slug="para",
    source="paraglider",
    tag="Para",
    env="XC_VIEWER_PARA_ROOT",
    folder="paragliders",
    color=DISCIPLINE_COLORS["paragliders"],
)

HANG_GLIDERS = Discipline(
    name="hang gliders",
    slug="hang",
    source="hangglider",
    tag="Hang",
    env="XC_VIEWER_HANG_ROOT",
    folder="hang_gliders",
    color=DISCIPLINE_COLORS["hang gliders"],
)

#: Both disciplines, keyed by name. The iteration order is the order every table, figure
#: legend and console report presents them in, so it is fixed here, not per script.
DISCIPLINES: dict[str, Discipline] = {d.name: d for d in (PARAGLIDERS, HANG_GLIDERS)}
