"""Readable typography and link handling shared by the viewer's help pages.

Three kinds of link appear in them: web pages (opened in the browser), anchors in the
same page (scrolled to), and ``methods:<anchor>`` links, which open a section of the
Sources & methods tab from any Info or How to use window.
"""

import weakref

from PyQt6.QtCore import QTimer, QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import QTextBrowser, QWidget

METHODS_SCHEME = "methods"

_methods_opener = None


def set_methods_opener(opener) -> None:
    """Route ``methods:`` links to ``opener(anchor)``, held weakly.

    Args:
        opener: A bound method of the main window; a closed window is never called.
    """
    global _methods_opener
    _methods_opener = weakref.WeakMethod(opener)


def open_methods(anchor: str) -> bool:
    """Show ``anchor`` in the Sources & methods tab, if a main window is listening."""
    opener = _methods_opener() if _methods_opener is not None else None
    if opener is None:
        return False
    try:
        opener(anchor)
    except RuntimeError:  # the window was destroyed by Qt
        return False
    return True


def follow_link(browser: QTextBrowser, url: QUrl) -> None:
    """Open one clicked link the way its kind requires."""
    if url.scheme() == METHODS_SCHEME:
        open_methods(url.path())
    elif url.scheme() in ("http", "https"):
        QDesktopServices.openUrl(url)
    elif url.hasFragment():
        anchor = url.fragment()
        QTimer.singleShot(0, lambda: browser.scrollToAnchor(anchor))


def info_browser(parent: QWidget | None = None) -> QTextBrowser:
    """Create a text browser with larger type, spacing and working links."""
    browser = QTextBrowser(parent)
    font = browser.font()
    font.setPointSizeF(max(15.0, font.pointSizeF() + 2))
    browser.setFont(font)
    browser.document().setDocumentMargin(20)
    browser.document().setDefaultStyleSheet(
        "h2 { margin-top: 0px; margin-bottom: 8px; }"
        "h3 { margin-top: 24px; margin-bottom: 10px; }"
        "p, ul, ol { margin-top: 0px; margin-bottom: 12px; }"
        "li { margin-bottom: 6px; }"
        "td, th { padding: 8px; text-align: left; vertical-align: top; }"
        ".role { color: #6b7280; margin-bottom: 18px; }"
    )
    # Every link goes through follow_link: a methods: link must never be loaded
    # as the page's new source, which would blank it.
    browser.setOpenLinks(False)
    browser.anchorClicked.connect(lambda url: follow_link(browser, url))
    return browser
