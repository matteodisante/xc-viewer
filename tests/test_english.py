"""The viewer shows month names in English, whatever language the system uses."""

from __future__ import annotations

import pytest
from PyQt6.QtCore import QDate, QLocale
from PyQt6.QtWidgets import QCalendarWidget, QDateEdit

from xc_viewer.widgets.english import use_english
from xc_viewer.widgets.thermal_plane import _day_month_calendar


@pytest.fixture
def italian_system(qapp):
    """Default locale of an Italian system, restored afterwards."""
    previous = QLocale()
    QLocale.setDefault(QLocale(QLocale.Language.Italian, QLocale.Country.Italy))
    yield
    QLocale.setDefault(previous)


def test_use_english_overrides_an_italian_system(italian_system):
    edit = QDateEdit(QDate(2000, 6, 1))
    edit.setDisplayFormat("dd MMM")
    assert edit.text() == "01 giu"
    use_english()
    assert QDateEdit(QDate(2000, 6, 1)).locale().monthName(6) == "June"
    edit = QDateEdit(QDate(2000, 6, 1))
    edit.setDisplayFormat("dd MMM")
    assert edit.text() == "01 Jun"


def test_day_month_calendar_shows_no_year_or_weekday_names(qapp):
    calendar = _day_month_calendar()
    assert calendar.horizontalHeaderFormat() == (
        QCalendarWidget.HorizontalHeaderFormat.NoHorizontalHeader
    )
    year_widgets = [
        child
        for child in calendar.findChildren(object)
        if child.objectName() in ("qt_calendar_yearbutton", "qt_calendar_yearedit")
    ]
    assert len(year_widgets) == 2
    assert all(child.isHidden() for child in year_widgets)
