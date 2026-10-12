"""Give a plot its existing window without reparenting or recreating its canvas."""

from PyQt6.QtCore import QEvent, QObject, Qt
from PyQt6.QtWidgets import QStackedWidget, QTabWidget, QWidget


class MapFocus(QObject):
    """Temporarily hide plot siblings, preserving their exact hidden states."""

    def __init__(self, root, target, *, hide=(), keep=()):
        """Keep the canvas in place and leave explicit return controls accessible."""
        super().__init__(root)
        self._widgets = {}
        self._layouts = []
        self._tab_frames = []
        for widget in hide:
            self._hide(widget)
        child = target
        while child is not root:
            parent = child.parentWidget()
            if parent is None:
                raise ValueError("The map must belong to the supplied window")
            if isinstance(parent, QTabWidget):
                self._tab_frames.append((parent, parent.documentMode()))
                parent.setDocumentMode(True)
            for sibling in parent.findChildren(
                QWidget, options=Qt.FindChildOption.FindDirectChildrenOnly
            ):
                # The stack owns page visibility, including transitions to another
                # tab. Hiding its inactive pages explicitly can suppress that show.
                if (
                    sibling is not child
                    and sibling not in keep
                    and not isinstance(parent, QStackedWidget)
                ):
                    self._hide(sibling)
            layout = parent.layout()
            if layout is not None:
                self._layouts.append(
                    (layout, layout.contentsMargins(), layout.spacing())
                )
                layout.setContentsMargins(0, 0, 0, 0)
                layout.setSpacing(0)
            child = parent

    def _hide(self, widget):
        self._widgets.setdefault(widget, widget.isHidden())
        widget.hide()
        widget.installEventFilter(self)

    def eventFilter(self, watched, event):  # noqa: N802
        """Keep controls hidden if a pending data load makes them available."""
        if event.type() == QEvent.Type.Show and watched in self._widgets:
            self._widgets[watched] = False
            watched.hide()
        return super().eventFilter(watched, event)

    def restore(self):
        """Return controls, optional panels and layout spacing to their prior state."""
        for layout, margins, spacing in self._layouts:
            layout.setContentsMargins(margins)
            layout.setSpacing(spacing)
        for widget, hidden in self._widgets.items():
            widget.removeEventFilter(self)
            widget.setHidden(hidden)
        for tab, document_mode in self._tab_frames:
            tab.setDocumentMode(document_mode)
        self._widgets.clear()
        self._layouts.clear()
        self._tab_frames.clear()
        self.deleteLater()
