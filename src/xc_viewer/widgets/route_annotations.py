"""Legible cell callouts and screen-size markers at actual clean endpoints."""

import numpy as np
from OpenGL import GL
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem


class RouteAnnotations(GLGraphicsItem):
    """Circles mark first clean fixes, squares last fixes, and A/B mark cells."""

    def __init__(self, tracks, anchors):
        """Keep original endpoint coordinates and separate cell-centre anchors."""
        super().__init__()
        self.endpoints = np.array([[track[0][0], track[-1][-1]] for track in tracks])
        self.anchors = np.asarray(anchors)
        self.mask = np.ones(len(tracks), dtype=bool)
        self.selected = set()
        self.setGLOptions("translucent")
        self.updateGLOptions({GL.GL_DEPTH_TEST: False, "glDepthMask": (True,)})
        self.setDepthValue(100)

    def set_flights(self, mask, selected):
        """Match marker visibility and emphasis to the trajectory controls."""
        self.mask, self.selected = np.asarray(mask, bool), selected
        self.update()

    def _project(self, xyz):
        """Project clean coordinates to screen pixels, rejecting behind-view points."""
        matrix = np.asarray(self.mvpMatrix().data()).reshape(4, 4, order="F")
        clip = np.column_stack((xyz, np.ones(len(xyz)))) @ matrix.T
        visible = np.isfinite(clip).all(axis=1) & (clip[:, 3] > 0)
        visible &= (abs(clip[:, :3]) <= clip[:, 3:]).all(axis=1)
        ndc = clip[:, :2] / np.where(clip[:, 3:] != 0, clip[:, 3:], 1)
        points = np.column_stack(
            (
                (ndc[:, 0] + 1) * self.view().width() / 2,
                (1 - ndc[:, 1]) * self.view().height() / 2,
            )
        )
        return points, visible

    def paint(self):
        """Draw outlined endpoint symbols and opaque label plates over the scene."""
        self.setupGLState()
        painter = QPainter(self.view())
        painter.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing
        )
        try:
            points, visible = self._project(self.endpoints.reshape(-1, 3))
            points, visible = points.reshape(-1, 2, 2), visible.reshape(-1, 2)
            # Selected flights' symbols are drawn last at a larger screen size.
            for i in sorted(
                np.flatnonzero(self.mask), key=lambda n: n in self.selected
            ):
                for end in (0, 1):
                    if not visible[i, end]:
                        continue
                    x, y = points[i, end]
                    radius = 6 if i in self.selected else 4
                    rect = QRectF(x - radius, y - radius, 2 * radius, 2 * radius)
                    painter.setPen(QPen(QColor("#10202e"), 3))
                    painter.setBrush(QColor("#ffffff" if end == 0 else "#10202e"))
                    if end == 0:
                        painter.drawEllipse(rect)
                    else:
                        painter.drawRect(rect)
                        painter.setPen(QPen(QColor("#ffffff"), 1.2))
                        painter.drawRect(rect)
            anchors, visible = self._project(self.anchors)
            painter.setFont(QFont("Helvetica", 11, QFont.Weight.Bold))
            for i, ((x, y), show) in enumerate(zip(anchors, visible, strict=True)):
                if not show:
                    continue
                text = "A · DEPARTURE" if i == 0 else "B · ARRIVAL"
                width = painter.fontMetrics().horizontalAdvance(text) + 20
                left = x - width - 18 if i == 0 else x + 18
                top = y - 44 if i == 0 else y + 18
                left = np.clip(left, 5, max(5, self.view().width() - width - 5))
                top = np.clip(top, 5, max(5, self.view().height() - 33))
                box = QRectF(float(left), float(top), width, 28)
                line_end = QPointF(
                    box.right() if i == 0 else box.left(), box.center().y()
                )
                painter.setPen(QPen(QColor("#10202e"), 4))
                painter.drawLine(QPointF(x, y), line_end)
                painter.setPen(QPen(QColor("#ffffff"), 1.5))
                painter.drawLine(QPointF(x, y), line_end)
                painter.setBrush(QColor("#10202e"))
                painter.drawRoundedRect(box, 5, 5)
                painter.drawText(box, Qt.AlignmentFlag.AlignCenter, text)
        finally:
            painter.end()
