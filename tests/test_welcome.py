"""The welcome screen stays operable by keyboard and stops painting when hidden."""

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import Qt
from PyQt6.QtTest import QSignalSpy, QTest

from xc_viewer.widgets.welcome import WelcomeScreen


@pytest.fixture
def welcome(qapp):
    widget = WelcomeScreen()
    widget.show()
    widget.activateWindow()
    qapp.processEvents()
    yield widget
    widget.close()


def test_enter_activates_focused_control_and_loading_prevents_start(welcome, qapp):
    starts = QSignalSpy(welcome.start_requested)
    about = QSignalSpy(welcome.about_requested)
    QTest.keyClick(welcome, Qt.Key.Key_Return)
    assert len(starts) == 1
    welcome._about.setFocus()
    qapp.processEvents()
    QTest.keyClick(welcome, Qt.Key.Key_Return)
    assert len(about) == 1
    assert len(starts) == 1
    welcome._start.setFocus()
    welcome.set_loading(True)
    QTest.keyClick(welcome, Qt.Key.Key_Return)
    assert len(starts) == 1
    welcome.set_loading(False)
    welcome._start.setFocus()
    QTest.keyClick(welcome, Qt.Key.Key_Return)
    assert len(starts) == 2


def test_pause_and_hiding_freeze_animation_without_losing_pause_choice(welcome, qapp):
    assert welcome._timer.isActive()
    welcome._pause.click()
    frozen = welcome._animation_time()
    QTest.qWait(40)
    assert welcome._animation_time() == frozen
    assert not welcome._timer.isActive()
    welcome.hide()
    welcome.show()
    qapp.processEvents()
    assert not welcome._timer.isActive()
    welcome._pause.click()
    assert welcome._timer.isActive()
    welcome.hide()
    frozen = welcome._animation_time()
    QTest.qWait(40)
    assert welcome._animation_time() == frozen
    assert not welcome._timer.isActive()
    welcome.show()
    qapp.processEvents()
    assert welcome._timer.isActive()


@pytest.mark.parametrize("size", [(680, 500), (1024, 600), (1440, 900)])
def test_controls_remain_visible_and_clickable_when_resized(welcome, qapp, size):
    welcome.resize(*size)
    qapp.processEvents()
    buttons = (welcome._start, welcome._about, welcome._pause)
    for button in buttons:
        assert welcome.rect().contains(button.geometry())
        assert welcome.childAt(button.geometry().center()) is button
    for index, button in enumerate(buttons):
        for other in buttons[index + 1 :]:
            assert not button.geometry().intersects(other.geometry())
