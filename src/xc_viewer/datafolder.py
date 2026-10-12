"""The data folder: the one local folder every tab reads its data from.

The folder holds one archive root per discipline (see
:mod:`xc_viewer.core.disciplines` for the layout). It is chosen at startup or
with the "Choose data folder" button, and published through an environment variable,
so that every :meth:`~xc_viewer.core.disciplines.Discipline.config` lookup in
this process resolves against it.

No Qt import (see the package docstring): the app and the main window supply the
dialog and the remembered setting.
"""

from __future__ import annotations

import os
from pathlib import Path

from .core.disciplines import DATA_FOLDER_ENV, DISCIPLINES

#: Key of the last data folder used, in the app's ``QSettings``.
SETTINGS_KEY = "data_folder"


def current() -> Path | None:
    """The data folder this process reads from, or ``None`` if none is chosen."""
    value = os.environ.get(DATA_FOLDER_ENV)
    return Path(value).expanduser() if value else None


def missing_parts(folder: str | Path) -> list[str]:
    """Why ``folder`` is not a usable data folder.

    Args:
        folder: A candidate data folder.

    Returns:
        Human-readable problems; empty when at least one discipline's archive root is
        present.
    """
    folder = Path(folder).expanduser()
    if not folder.is_dir():
        return [f"{folder} is not a folder"]
    names = [d.folder for d in DISCIPLINES.values()]
    if any((folder / name).is_dir() for name in names):
        return []
    return [f"{folder} contains neither {' nor '.join(f'{n}/' for n in names)}"]


def use(folder: str | Path) -> Path:
    """Read every discipline from ``folder`` from now on.

    Also drops any per-discipline override from the environment, so that the new
    folder applies to both disciplines.

    Args:
        folder: The data folder.

    Returns:
        The folder as an absolute path.
    """
    folder = Path(folder).expanduser().resolve()
    os.environ[DATA_FOLDER_ENV] = str(folder)
    for discipline in DISCIPLINES.values():
        os.environ.pop(discipline.env, None)
    return folder
