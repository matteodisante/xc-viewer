"""Startup configures graphics early and opens the viewer only after Start."""

import os
import subprocess
import sys

import pytest

pytest.importorskip("PyQt6")


def _run_startup(code, *, data_folder=None):
    env = {
        **os.environ,
        "QT_QPA_PLATFORM": "offscreen",
        "PYTHONPATH": os.pathsep.join(sys.path),
    }
    env.pop("XC_VIEWER_DATA", None)
    if data_folder is not None:
        env["XC_VIEWER_DATA"] = str(data_folder)
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_core_profile_and_sharing_are_configured_before_qapplication(tmp_path):
    (tmp_path / "paragliders").mkdir()
    code = """
import sys
from types import ModuleType
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import QCoreApplication, QTimer, Qt
from PyQt6.QtGui import QSurfaceFormat
from xc_viewer import app as entrypoint
from xc_viewer.app import main
from xc_viewer.widgets import welcome as welcome_module

import truststore

original = QtWidgets.QApplication
events = []
windows = []
truststore.inject_into_ssl = lambda: events.append("certificates")

class Settings:
    def value(self, key, default="", type=str):
        raise AssertionError("Environment folder should take priority over settings")

    def setValue(self, key, value):
        raise AssertionError("Environment folder must not be saved")

class CheckedApplication(original):
    def __init__(self, argv):
        assert QCoreApplication.instance() is None
        fmt = QSurfaceFormat.defaultFormat()
        assert (fmt.majorVersion(), fmt.minorVersion()) >= (3, 3)
        assert fmt.profile() == QSurfaceFormat.OpenGLContextProfile.CoreProfile
        assert fmt.depthBufferSize() >= 24
        assert fmt.stencilBufferSize() >= 8
        assert QCoreApplication.testAttribute(
            Qt.ApplicationAttribute.AA_ShareOpenGLContexts
        )
        events.append("application")
        super().__init__(argv)

class Window(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        assert QCoreApplication.instance() is not None
        assert events[-1] == "start"
        events.append("window")
        windows.append(self)

    def showEvent(self, event):
        super().showEvent(event)
        events.append("visible")
        QTimer.singleShot(0, QCoreApplication.instance().quit)

class CheckedWelcome(welcome_module.WelcomeScreen):
    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self.enter)

    def enter(self):
        assert self.isVisible()
        assert events == ["certificates", "application"]
        assert not windows
        events.append("start")
        self.start_requested.emit()
        self.start_requested.emit()
        # Starting is queued so the loading state can paint first.
        assert not windows

    def closeEvent(self, event):
        assert len(windows) == 1
        assert windows[0].isVisible()
        events.append("welcome closed")
        super().closeEvent(event)

def unexpected_picker(*args, **kwargs):
    raise AssertionError("The provided data folder should not need a picker")

QtWidgets.QApplication = CheckedApplication
QtCore.QSettings = Settings
welcome_module.WelcomeScreen = CheckedWelcome
entrypoint.choose_data_folder = unexpected_picker
module = ModuleType("xc_viewer.main_window")
module.MainWindow = Window
sys.modules[module.__name__] = module
assert main() == 0
assert events == [
    "certificates", "application", "start", "window", "visible", "welcome closed"
]
"""
    _run_startup(code, data_folder=tmp_path)


def test_closing_welcome_before_start_never_initializes_the_viewer():
    code = """
import sys
from types import ModuleType
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import QTimer
from xc_viewer import app as entrypoint
from xc_viewer.widgets import welcome as welcome_module

events = []

class Settings:
    def value(self, key, default="", type=str):
        events.append("settings read")
        return default

    def setValue(self, key, value):
        events.append("settings written")

class Window(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        events.append("window")
        QTimer.singleShot(0, QtWidgets.QApplication.instance().quit)

class CheckedWelcome(welcome_module.WelcomeScreen):
    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self.dismiss)

    def dismiss(self):
        assert self.isVisible()
        assert not events
        events.append("welcome closed")
        self.close()

def picker(*args, **kwargs):
    events.append("folder picker")
    return None

QtCore.QSettings = Settings
welcome_module.WelcomeScreen = CheckedWelcome
entrypoint.choose_data_folder = picker
module = ModuleType("xc_viewer.main_window")
module.MainWindow = Window
sys.modules[module.__name__] = module
assert entrypoint.main() == 0
assert events == ["welcome closed"]
"""
    _run_startup(code)


@pytest.mark.parametrize("mode", ["normal", "maximized", "fullscreen"])
def test_start_preserves_window_state_and_restored_geometry(tmp_path, mode):
    (tmp_path / "paragliders").mkdir()
    code = """
import sys
from types import ModuleType
from PyQt6.QtCore import QRect, QTimer, Qt
from PyQt6.QtWidgets import QWidget
from xc_viewer.app import main
from xc_viewer.widgets import welcome as welcome_module

mode = MODE
expected = {}
checked = []
state_mask = Qt.WindowState.WindowMaximized | Qt.WindowState.WindowFullScreen

class CheckedWelcome(welcome_module.WelcomeScreen):
    scheduled = False

    def showEvent(self, event):
        super().showEvent(event)
        if not self.scheduled:
            self.scheduled = True
            QTimer.singleShot(0, self.prepare)

    def prepare(self):
        self.setGeometry(QRect(38, 42, 710, 535))
        if mode == "maximized":
            self.showMaximized()
        elif mode == "fullscreen":
            self.showFullScreen()
        QTimer.singleShot(0, self.enter)

    def enter(self):
        expected["state"] = self.windowState() & state_mask
        expected["normal"] = self.normalGeometry()
        expected["screen"] = self.screen()
        self._start.click()

class Window(QWidget):
    scheduled = False

    def showEvent(self, event):
        super().showEvent(event)
        if not self.scheduled:
            self.scheduled = True
            QTimer.singleShot(0, self.inspect)

    def inspect(self):
        assert self.isVisible()
        assert self.windowState() & state_mask == expected["state"]
        assert self.screen() is expected["screen"]
        # Exiting fullscreen/maximized must return to the size chosen on Home.
        self.showNormal()
        QTimer.singleShot(0, self.inspect_restored)

    def inspect_restored(self):
        assert self.geometry() == expected["normal"], (
            self.geometry(), expected["normal"]
        )
        checked.append(mode)
        self.close()

welcome_module.WelcomeScreen = CheckedWelcome
module = ModuleType("xc_viewer.main_window")
module.MainWindow = Window
sys.modules[module.__name__] = module
assert main([]) == 0
assert checked == [mode]
"""
    _run_startup(code.replace("MODE", repr(mode)), data_folder=tmp_path)
