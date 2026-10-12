"""The Info and How to use buttons every screen carries, and the windows they open.

The texts live in :mod:`.help_text`. A window is created on the first click and then
reused, so it keeps its size and scroll position, and never blocks the viewer.
"""

from __future__ import annotations

from PyQt6.QtCore import QSize
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .help_text import page
from .info_browser import info_browser

LABELS = {
    "info": ("Info", "What this screen shows, and how to read it"),
    "howto": ("How to use", "Step by step: what to click, and in which order"),
}


class HelpWindow(QDialog):
    """A resizable, non-modal reading window."""

    def __init__(self, title: str, html: str, parent: QWidget | None = None):
        """Show ``html`` with a Close button."""
        super().__init__(parent)
        self.setWindowTitle(title)
        size = QSize(760, 760)
        if parent is not None and parent.screen() is not None:
            size = size.boundedTo(
                parent.screen().availableGeometry().size() - QSize(40, 60)
            )
        self.resize(size)
        self.browser = info_browser(self)
        self.browser.setHtml(html)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.close)
        layout = QVBoxLayout(self)
        layout.addWidget(self.browser)
        layout.addWidget(buttons)


class HelpButton(QPushButton):
    """Open, and reuse, one screen's Info or How to use window."""

    def __init__(self, topic: str, kind: str, parent: QWidget | None = None):
        """Build nothing until clicked: no texts, configs or archive reads yet."""
        label, tooltip = LABELS[kind]
        super().__init__(label, parent)
        self.setProperty("emphasis", "quiet")
        self._topic, self._kind = topic, kind
        self.window_ = None
        self.setAutoDefault(False)
        self.setToolTip(tooltip)
        self.clicked.connect(self.show_help)

    def show_help(self) -> None:
        """Bring the window forward, creating it on first use."""
        if self.window_ is None:
            title, html = page(self._topic, self._kind)
            self.window_ = HelpWindow(title, html, self)
        self.window_.show()
        self.window_.raise_()
        self.window_.activateWindow()


def help_buttons(topic: str, parent: QWidget | None = None):
    """The ``(Info, How to use)`` buttons of one screen."""
    return HelpButton(topic, "info", parent), HelpButton(topic, "howto", parent)
