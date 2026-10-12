"""Control rows that wrap as the viewer window narrows."""

from PyQt6.QtCore import QEvent, QRect, QSize, Qt
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QScrollArea,
    QSizePolicy,
    QWidget,
)


def labeled_control(text, widget):
    """Keep a label beside its control when the surrounding row wraps."""
    group = QWidget()
    layout = QHBoxLayout(group)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(QLabel(text))
    layout.addWidget(widget)
    return group


class ControlPanel(QScrollArea):
    """Show every wrapped control row, and scroll them when the plot needs the room.

    Rows normally get their full height. In a short window the plot below keeps its
    minimum height, and the rows scroll instead of pushing the plot out of sight.
    """

    def __init__(self, layout, parent=None):
        """Place ``layout`` in a frameless area that scrolls only vertically."""
        super().__init__(parent)
        content = QWidget()
        content.setLayout(layout)
        content.installEventFilter(self)
        self.setWidget(content)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

    def sizeHint(self):  # noqa: N802
        """Ask for all rows as they wrap at the current width."""
        content = self.widget()
        width = max(self.width(), content.minimumSizeHint().width())
        return QSize(content.minimumSizeHint().width(), content.heightForWidth(width))

    def minimumSizeHint(self):  # noqa: N802
        """Keep about one row of controls in view."""
        height = min(self.sizeHint().height(), 4 * self.fontMetrics().height())
        return QSize(self.widget().minimumSizeHint().width(), height)

    def eventFilter(self, watched, event):  # noqa: N802
        """Ask for a new height when controls appear, hide or change text."""
        if event.type() == QEvent.Type.LayoutRequest:
            self.updateGeometry()
        return super().eventFilter(watched, event)

    def resizeEvent(self, event):  # noqa: N802
        """Rows rewrap at a new width, which changes the height they need."""
        super().resizeEvent(event)
        if event.size().width() != event.oldSize().width():
            self.updateGeometry()


class FlowLayout(QLayout):
    """Keep controls at readable sizes and move overflow onto another row."""

    def __init__(self, parent=None):
        """Use the same compact spacing inside and between rows."""
        super().__init__(parent)
        self._items = []
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(6)

    def addItem(self, item):  # noqa: N802
        """Take ownership of a widget or nested layout item."""
        self._items.append(item)

    def count(self):
        """Return the number of owned items, including hidden controls."""
        return len(self._items)

    def itemAt(self, index):  # noqa: N802
        """Return an item without removing it."""
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index):  # noqa: N802
        """Release an item when Qt reparents or destroys a control."""
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):  # noqa: N802
        """Fill width while reserving only the height needed for the rows."""
        return Qt.Orientation.Horizontal

    def hasHeightForWidth(self):  # noqa: N802
        """Let the enclosing layout account for wrapped rows."""
        return True

    def heightForWidth(self, width):  # noqa: N802
        """Measure the rows without moving widgets."""
        return self._arrange(QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect):  # noqa: N802
        """Reflow controls whenever the parent changes size."""
        super().setGeometry(rect)
        self._arrange(rect, apply=True)

    def minimumSize(self):  # noqa: N802
        """Allow shrinking down to the largest individual control."""
        size = QSize()
        for item in self._items:
            if not item.isEmpty():
                size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        return size + QSize(
            margins.left() + margins.right(), margins.top() + margins.bottom()
        )

    def sizeHint(self):  # noqa: N802
        """Avoid imposing the combined width of every control on the window."""
        return self.minimumSize()

    def _arrange(self, rect, *, apply):
        """Measure or position rows, sharing spare width among expanding items."""
        margins = self.contentsMargins()
        area = rect.marginsRemoved(margins)
        spacing = self.spacing()
        rows, row, used = [], [], 0
        for item in self._items:
            if item.isEmpty():
                continue
            size = item.sizeHint().expandedTo(item.minimumSize())
            width = min(size.width(), max(area.width(), item.minimumSize().width()))
            if row and used + spacing + width > area.width():
                rows.append(row)
                row, used = [], 0
            used += (spacing if row else 0) + width
            row.append((item, width, size.height()))
        if row:
            rows.append(row)
        y = area.y()
        for row in rows:
            height = max(h for _, _, h in row)
            extra = max(
                0,
                area.width() - sum(w for _, w, _ in row) - spacing * (len(row) - 1),
            )
            expanding = sum(
                bool(item.expandingDirections() & Qt.Orientation.Horizontal)
                for item, _, _ in row
            )
            x = area.x()
            for item, width, item_height in row:
                if expanding and item.expandingDirections() & Qt.Orientation.Horizontal:
                    share = extra // expanding
                    width += share
                    extra -= share
                    expanding -= 1
                if apply:
                    item.setGeometry(
                        QRect(x, y + (height - item_height) // 2, width, item_height)
                    )
                x += width + spacing
            y += height + spacing
        return (
            y - area.y() - (spacing if rows else 0) + margins.top() + margins.bottom()
        )
