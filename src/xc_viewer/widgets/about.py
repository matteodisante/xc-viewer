"""Project provenance for XC Viewer, separate from its scientific sources."""

from __future__ import annotations

from PyQt6.QtCore import QSize
from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QVBoxLayout, QWidget

from .info_browser import info_browser

_ABOUT_HTML = """
<h2>XC Viewer</h2>
<p class="role">Explore cross-country flights and the thermals that carry them.</p>
<p>An interactive viewer of paraglider and hang-glider flights, their flight
phases, thermal structure and collective soaring patterns.</p>

<h3>Project and development</h3>
<p>Developed by <b>Matteo Di Sante</b> as part of the internship:</p>
<p><i>Anomalous Transport in Soaring Flights: Collective Strategies and
Intermittent Search Models</i></p>
<p>At the <b>Econophysics Lab at CFM (Paris)</b>, under the supervision of
<b>Prof. Michael Benzaquen</b> and <b>Dr. Alexandre Darmon</b>.</p>

<p><a href="https://github.com/matteodisante/xc-viewer">
Project repository</a></p>
<p>Data sources, scientific methods and licences are documented in the viewer's
<b>Sources &amp; methods</b> tab.</p>
"""


class AboutDialog(QDialog):
    """A reusable, non-modal window with the project's author and research context."""

    def __init__(self, parent: QWidget | None = None):
        """Show project credits in a resizable, scrollable reading window."""
        super().__init__(parent)
        self.setWindowTitle("About XC Viewer")
        self.setModal(False)
        size = QSize(650, 610)
        screen = self.screen()
        if screen is not None:
            size = size.boundedTo(screen.availableGeometry().size() - QSize(40, 60))
        self.resize(size)

        self.browser = info_browser(self)
        self.browser.setAccessibleName("About XC Viewer")
        self.browser.setHtml(_ABOUT_HTML)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.close)
        layout = QVBoxLayout(self)
        layout.addWidget(self.browser)
        layout.addWidget(buttons)
