"""English month and weekday names and number formats, whatever the system language."""

from __future__ import annotations

from PyQt6.QtCore import QLocale

ENGLISH = QLocale(QLocale.Language.English, QLocale.Country.UnitedStates)


def use_english() -> None:
    """Make every Qt widget format dates and numbers in English.

    Without it, date fields and calendar popups follow the operating system: an
    Italian system shows "giu" for June. Call it once, after the ``QApplication`` exists
    and before the first widget is built.
    """
    QLocale.setDefault(ENGLISH)
