"""An illustrated, animated entry point for the soaring flight viewer."""

from __future__ import annotations

import math

from PyQt6.QtCore import (
    QElapsedTimer,
    QEvent,
    QPointF,
    QRectF,
    QSize,
    Qt,
    QTimer,
    pyqtSignal,
)
from PyQt6.QtGui import (
    QColor,
    QFont,
    QKeySequence,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
    QRadialGradient,
    QShortcut,
)
from PyQt6.QtWidgets import QLabel, QPushButton, QWidget

_SCENE_WIDTH = 1440.0
_SCENE_HEIGHT = 900.0
_FLIGHT_COLORS = ("#efbd7c", "#c8efdf", "#ef957f", "#d0dfed", "#87d4cb")


def _polygon(points: tuple[tuple[float, float], ...]) -> QPolygonF:
    return QPolygonF([QPointF(x, y) for x, y in points])


def _draw_landscape(painter: QPainter) -> None:
    """Paint a sunlit Alpine illustration in a fixed coordinate system."""
    painter.setPen(Qt.PenStyle.NoPen)
    sky = QLinearGradient(0, 0, 0, _SCENE_HEIGHT)
    sky.setColorAt(0, QColor("#479ed5"))
    sky.setColorAt(0.42, QColor("#aad5e9"))
    sky.setColorAt(0.66, QColor("#e5ecdb"))
    sky.setColorAt(1, QColor("#7fa99b"))
    painter.fillRect(QRectF(0, 0, _SCENE_WIDTH, _SCENE_HEIGHT), sky)

    # A high, radiant sun makes the daytime setting clear above the thermal climbs.
    sun = QPointF(1160, 150)
    glow = QRadialGradient(sun, 280)
    glow.setColorAt(0, QColor(255, 234, 151, 195))
    glow.setColorAt(0.22, QColor(255, 229, 133, 100))
    glow.setColorAt(0.55, QColor(255, 240, 177, 32))
    glow.setColorAt(1, QColor(255, 244, 196, 0))
    painter.fillRect(QRectF(0, 0, _SCENE_WIDTH, _SCENE_HEIGHT), glow)
    # Keep the sun circular when the landscape stretches to fit a compact window.
    painter.save()
    transform = painter.worldTransform()
    painter.translate(sun)
    painter.scale(1, transform.m11() / transform.m22())
    sun = QPointF(0, 0)
    painter.setPen(
        QPen(
            QColor(255, 239, 166, 165),
            2,
            Qt.PenStyle.SolidLine,
            Qt.PenCapStyle.RoundCap,
        )
    )
    for ray in range(12):
        angle = ray * math.tau / 12
        direction = QPointF(math.cos(angle), math.sin(angle))
        painter.drawLine(sun + direction * 48, sun + direction * 61)
    painter.setPen(Qt.PenStyle.NoPen)
    disk = QRadialGradient(sun, 35)
    disk.setColorAt(0, QColor("#fffde4"))
    disk.setColorAt(0.72, QColor("#fff3aa"))
    disk.setColorAt(1, QColor("#ffce60"))
    painter.setBrush(disk)
    painter.drawEllipse(sun, 35, 35)
    painter.restore()

    # Wind-stretched high clouds; deliberately quiet behind the flight paths.
    for x, y, width in ((600, 164, 490), (915, 227, 390), (770, 290, 270)):
        cloud = QPainterPath(QPointF(x, y))
        cloud.cubicTo(x + width * 0.3, y - 10, x + width * 0.7, y + 4, x + width, y)
        cloud.cubicTo(x + width * 0.7, y + 9, x + width * 0.2, y + 2, x, y)
        painter.fillPath(cloud, QColor(255, 255, 244, 75))

    # Each ridge has its own depth and lighting, with asymmetric shaded faces.
    ridges = (
        (
            "#9abcc5",
            (
                (0, 535),
                (90, 490),
                (142, 500),
                (231, 416),
                (280, 461),
                (359, 414),
                (444, 486),
                (520, 461),
                (598, 499),
                (700, 442),
                (779, 469),
                (858, 420),
                (924, 463),
                (1006, 411),
                (1086, 450),
                (1168, 394),
                (1221, 449),
                (1308, 401),
                (1440, 491),
            ),
        ),
        (
            "#6d9696",
            (
                (0, 616),
                (100, 564),
                (197, 587),
                (301, 519),
                (390, 566),
                (492, 511),
                (590, 568),
                (690, 512),
                (786, 430),
                (823, 476),
                (861, 488),
                (942, 538),
                (1000, 506),
                (1099, 384),
                (1129, 425),
                (1164, 459),
                (1194, 451),
                (1245, 520),
                (1313, 475),
                (1440, 567),
            ),
        ),
    )
    for color, ridge in ridges:
        painter.setBrush(QColor(color))
        painter.drawPolygon(_polygon((*ridge, (1440, 900), (0, 900))))

    facets = (
        (
            "#abb798",
            ((786, 430), (746, 509), (724, 546), (670, 590), (814, 534), (823, 476)),
        ),
        (
            "#507b84",
            ((786, 430), (814, 534), (760, 634), (951, 584), (861, 488), (823, 476)),
        ),
        (
            "#c2c6a4",
            (
                (1099, 384),
                (1034, 480),
                (1047, 505),
                (1000, 576),
                (1109, 490),
                (1129, 425),
            ),
        ),
        (
            "#57818a",
            (
                (1099, 384),
                (1109, 490),
                (1080, 572),
                (1176, 546),
                (1245, 520),
                (1194, 451),
                (1164, 459),
                (1129, 425),
            ),
        ),
        (
            "#f0f0d9",
            (
                (1099, 384),
                (1060, 442),
                (1087, 426),
                (1096, 439),
                (1105, 417),
                (1129, 425),
            ),
        ),
        ("#e0e6d3", ((786, 430), (758, 466), (783, 457), (792, 465), (804, 454))),
        ("#a0b28f", ((1313, 475), (1266, 564), (1328, 536), (1380, 580))),
    )
    for color, points in facets:
        painter.setBrush(QColor(color))
        painter.drawPolygon(_polygon(points))

    mist = QLinearGradient(0, 460, 0, 735)
    mist.setColorAt(0, QColor(208, 225, 201, 0))
    mist.setColorAt(0.76, QColor(208, 225, 201, 75))
    mist.setColorAt(1, QColor(208, 225, 201, 0))
    painter.fillRect(QRectF(0, 460, 1440, 275), mist)

    painter.setBrush(QColor("#73977c"))
    painter.drawPolygon(
        _polygon(
            (
                (0, 684),
                (126, 597),
                (221, 630),
                (323, 576),
                (434, 647),
                (547, 614),
                (695, 690),
                (823, 615),
                (933, 642),
                (1035, 577),
                (1145, 631),
                (1297, 589),
                (1440, 675),
                (1440, 900),
                (0, 900),
            )
        )
    )
    painter.setBrush(QColor("#4a7765"))
    painter.drawPolygon(
        _polygon(
            (
                (0, 693),
                (155, 742),
                (319, 654),
                (512, 688),
                (662, 779),
                (830, 744),
                (1011, 692),
                (1133, 711),
                (1270, 656),
                (1440, 729),
                (1440, 900),
                (0, 900),
            )
        )
    )
    painter.setBrush(QColor("#315d50"))
    painter.drawPolygon(
        _polygon(
            (
                (0, 826),
                (169, 787),
                (345, 814),
                (497, 760),
                (703, 832),
                (916, 790),
                (1115, 813),
                (1278, 738),
                (1440, 705),
                (1440, 900),
                (0, 900),
            )
        )
    )

    # A few contour-like valley folds give the foreground scale without clutter.
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.setPen(QPen(QColor(119, 167, 162, 22), 1))
    for offset in range(0, 100, 20):
        fold = QPainterPath(QPointF(750, 830 + offset))
        fold.cubicTo(915, 756 + offset, 1170, 907 + offset, 1480, 738 + offset)
        painter.drawPath(fold)

    shade = QLinearGradient(0, 0, 1060, 0)
    shade.setColorAt(0, QColor(7, 22, 36, 239))
    shade.setColorAt(0.40, QColor(7, 22, 36, 219))
    shade.setColorAt(0.76, QColor(7, 22, 36, 71))
    shade.setColorAt(1, QColor(7, 22, 36, 0))
    painter.fillRect(QRectF(0, 0, 1440, 900), shade)
    bottom = QLinearGradient(0, 720, 0, 900)
    bottom.setColorAt(0, QColor(7, 22, 36, 0))
    bottom.setColorAt(1, QColor(7, 22, 36, 180))
    painter.fillRect(QRectF(0, 720, 1440, 180), bottom)


def _flight_position(seconds: float, index: int) -> QPointF:
    """Return a continuous illustrative trajectory with loops and long glides."""
    phase = seconds * 0.23 + index * 1.23
    orbit = seconds * 1.35 + index * 1.8
    # The secondary orbit opens out when the wider route is moving quickly.
    circle = 0.5 + 0.5 * math.cos(phase)
    x = 1090 + 204 * math.sin(phase) + (24 + 32 * circle) * math.sin(orbit)
    y = 295 + 75 * math.cos(phase) + (10 + 14 * circle) * math.cos(orbit)
    return QPointF(x, y + index * 20)


def _draw_paraglider(
    painter: QPainter, position: QPointF, color: QColor, bank: float, scale: float
) -> None:
    """Draw a crescent canopy, suspension lines, and a suspended pilot."""
    painter.save()
    painter.translate(position)
    painter.scale(scale, scale)
    painter.rotate(bank)

    canopy = QPainterPath(QPointF(-18, 2))
    canopy.cubicTo(-17, -15, 17, -15, 18, 2)
    canopy.cubicTo(9, -5, -9, -5, -18, 2)
    painter.setPen(QPen(color.lighter(120), 0.65))
    painter.setBrush(color)
    painter.drawPath(canopy)
    painter.setPen(QPen(color.darker(140), 0.6))
    for x in (-12, -6, 0, 6, 12):
        rib = QPainterPath(QPointF(x, -10 + abs(x) * 0.24))
        rib.quadTo(x * 1.04, -5, x * 1.06, -3 + abs(x) * 0.2)
        painter.drawPath(rib)
    painter.setPen(QPen(QColor(222, 238, 231, 159), 0.65))
    for x in (-16, -8, 8, 16):
        painter.drawLine(QPointF(x, -1), QPointF(0, 18))
    painter.setPen(
        QPen(QColor("#112c39"), 2.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
    )
    painter.drawLine(QPointF(0, 18), QPointF(1, 23))
    painter.drawLine(QPointF(1, 23), QPointF(5, 24))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor("#142e38"))
    painter.drawEllipse(QPointF(0, 15.5), 1.8, 1.8)
    painter.restore()


class WelcomeScreen(QWidget):
    """Show a calm Alpine landing screen before opening the analysis workspace."""

    start_requested = pyqtSignal()
    about_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create accessible native controls over a lightweight animated scene."""
        super().__init__(parent)
        self.setObjectName("welcomeScreen")
        self.setWindowTitle("XC Viewer")
        self.setMinimumSize(680, 500)
        self.setAccessibleName("XC Viewer welcome screen")
        self._background: QPixmap | None = None
        self._background_key: tuple[int, int, float] | None = None
        self._paused = False
        self._loading = False
        self._seconds = 0.0
        self._clock = QElapsedTimer()
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self.update)

        self._eyebrow = self._label("FLIGHTS · THERMALS · COLLECTIVE MOTION", "#b0cfcb")
        self._title = self._label("XC\nViewer", "#f3f4e9")
        self._subtitle = self._label(
            "Explore soaring flights, thermal climbs\n"
            "and the patterns that connect them.",
            "#c3d3d4",
        )
        self._credit = self._label("Developed by Matteo Di Sante", "#d2dedb")
        self._lab = self._label("Econophysics Lab · CFM, Paris", "#94b1b5")
        self._illustration_note = self._label("Illustrative flight paths", "#a6c0c1")

        self._start = QPushButton("Start exploring  →", self)
        self._start.setObjectName("startExploring")
        self._start.setCursor(Qt.CursorShape.PointingHandCursor)
        self._start.setAccessibleName("Start exploring flights")
        self._start.setToolTip("Open the flight viewer (Enter)")
        self._start.clicked.connect(self.start_requested.emit)
        self._start.setStyleSheet("""
            QPushButton {
                color: #112e39; background: #e7edda; border: 1px solid #e7edda;
                border-radius: 7px; padding: 0 18px; font-weight: 600;
            }
            QPushButton:hover { background: #f5f6e9; border-color: #f5f6e9; }
            QPushButton:pressed { background: #cbdac4; }
            QPushButton:focus { border: 2px solid #78ddcc; }
            QPushButton:disabled { color: #4d666a; background: #bccbc3; }
        """)
        self._about = QPushButton("About the project", self)
        self._about.setObjectName("aboutProject")
        self._about.setCursor(Qt.CursorShape.PointingHandCursor)
        self._about.clicked.connect(self.about_requested.emit)
        self._about.setStyleSheet("""
            QPushButton {
                color: #d2dfdc; background: transparent; border: 1px solid transparent;
                border-radius: 7px; padding: 0 10px;
            }
            QPushButton:hover { color: #ffffff; background: rgba(230, 244, 237, 12); }
            QPushButton:focus { border: 1px solid #89c6c1; }
            QPushButton:pressed { background: rgba(230, 244, 237, 24); }
        """)
        self._pause = QPushButton("Pause animation", self)
        self._pause.setObjectName("pauseAnimation")
        self._pause.setCheckable(True)
        self._pause.setCursor(Qt.CursorShape.PointingHandCursor)
        self._pause.setToolTip("Pause or resume the illustrated flights")
        self._pause.toggled.connect(self._set_paused)
        self._pause.setStyleSheet("""
            QPushButton {
                color: #c5d9d7; background: rgba(8, 30, 41, 90);
                border: 1px solid rgba(160, 195, 191, 50); border-radius: 6px;
                padding: 0 10px;
            }
            QPushButton:hover { background: rgba(43, 77, 84, 180); }
            QPushButton:focus { border: 1px solid #89c6c1; }
            QPushButton:checked { color: #ecf2e8; }
        """)
        for key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(self._activate_focused_control)
        QWidget.setTabOrder(self._start, self._about)
        QWidget.setTabOrder(self._about, self._pause)
        initial_size = self.sizeHint()
        screen = self.screen()
        if screen is not None:
            available = screen.availableGeometry().size() - QSize(40, 60)
            initial_size = initial_size.boundedTo(available)
        self.resize(initial_size)
        self._arrange_controls()

    def _label(self, text: str, color: str) -> QLabel:
        label = QLabel(text, self)
        label.setStyleSheet(f"color: {color}; background: transparent;")
        label.setTextFormat(Qt.TextFormat.PlainText)
        label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        return label

    def sizeHint(self) -> QSize:  # noqa: N802
        """Suggest a spacious opening view without requiring a large display."""
        return QSize(1300, 820)

    def set_loading(self, loading: bool) -> None:
        """Reflect workspace initialization and prevent duplicate launch requests."""
        self._loading = loading
        self._start.setEnabled(not loading)
        self._start.setText("Opening viewer…" if loading else "Start exploring  →")

    def _activate_focused_control(self) -> None:
        if self._about.hasFocus():
            self._about.click()
        elif self._pause.hasFocus():
            self._pause.click()
        elif not self._loading:
            self._start.click()

    def _animation_time(self) -> float:
        return self._seconds + (
            self._clock.elapsed() / 1000 if self._timer.isActive() else 0
        )

    def _stop_animation(self) -> None:
        if self._timer.isActive():
            self._seconds = self._animation_time()
            self._timer.stop()

    def _start_animation(self) -> None:
        if (
            not self._paused
            and self.isVisible()
            and not self.window().isMinimized()
            and not self._timer.isActive()
        ):
            self._clock.start()
            self._timer.start()

    def _set_paused(self, paused: bool) -> None:
        self._paused = paused
        self._pause.setText("Resume animation" if paused else "Pause animation")
        if paused:
            self._stop_animation()
        else:
            self._start_animation()
        self.update()

    def showEvent(self, event) -> None:  # noqa: N802
        """Run animation only while the welcome screen is visible."""
        super().showEvent(event)
        self._start_animation()
        self._start.setFocus(Qt.FocusReason.OtherFocusReason)

    def hideEvent(self, event) -> None:  # noqa: N802
        """Release the animation timer when entering the viewer or hiding it."""
        self._stop_animation()
        super().hideEvent(event)

    def changeEvent(self, event) -> None:  # noqa: N802
        """Suspend illustration work while the opening window is minimized."""
        super().changeEvent(event)
        if event.type() == QEvent.Type.WindowStateChange:
            if self.window().isMinimized():
                self._stop_animation()
            else:
                self._start_animation()

    def resizeEvent(self, event) -> None:  # noqa: N802
        """Reflow the type and controls while invalidating the scenery cache."""
        self._background = None
        self._arrange_controls()
        super().resizeEvent(event)

    def _arrange_controls(self) -> None:
        width, height = self.width(), self.height()
        inset = round(max(32, min(108, width * 0.075)))
        compact = width < 950 or height < 640
        title_size = 48 if compact else min(78, round(width * 0.057))
        title_height = round(title_size * 2.45)
        top = round(height * (0.20 if compact else 0.23))
        text_width = max(360, round(width * 0.47))
        eyebrow_font = QFont(self.font())
        eyebrow_font.setPixelSize(10 if compact else 11)
        eyebrow_font.setWeight(QFont.Weight.Medium)
        eyebrow_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1.6)
        self._eyebrow.setFont(eyebrow_font)
        self._eyebrow.setGeometry(inset, top - 32, text_width, 22)
        title_font = QFont(self.font())
        title_font.setPixelSize(title_size)
        title_font.setWeight(QFont.Weight.DemiBold)
        title_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, -1.8)
        self._title.setFont(title_font)
        self._title.setGeometry(inset - 3, top, text_width, title_height)
        self._title.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        body_font = QFont(self.font())
        body_font.setPixelSize(14 if compact else 17)
        self._subtitle.setFont(body_font)
        subtitle_y = top + title_height + (10 if compact else 16)
        self._subtitle.setGeometry(inset, subtitle_y, text_width, 54)
        button_y = subtitle_y + (70 if compact else 86)
        button_font = QFont(self.font())
        button_font.setPixelSize(13 if compact else 14)
        self._start.setFont(button_font)
        self._about.setFont(button_font)
        self._start.setGeometry(inset, button_y, 182, 46)
        self._about.setGeometry(inset + 194, button_y, 154, 46)

        small_font = QFont(self.font())
        small_font.setPixelSize(11 if compact else 12)
        for label in (self._credit, self._lab, self._illustration_note):
            label.setFont(small_font)
        self._credit.setGeometry(inset, height - 74, 320, 22)
        self._lab.setGeometry(inset, height - 53, 320, 22)
        self._pause.setFont(small_font)
        self._pause.setGeometry(width - inset - 144, height - 67, 144, 32)
        self._illustration_note.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._illustration_note.setGeometry(width - inset - 220, height - 97, 220, 24)

    def _ensure_background(self) -> QPixmap:
        ratio = self.devicePixelRatioF()
        key = (self.width(), self.height(), ratio)
        if self._background is None or self._background_key != key:
            self._background = QPixmap(
                round(self.width() * ratio), round(self.height() * ratio)
            )
            self._background.setDevicePixelRatio(ratio)
            painter = QPainter(self._background)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.scale(self.width() / _SCENE_WIDTH, self.height() / _SCENE_HEIGHT)
            _draw_landscape(painter)
            painter.end()
            self._background_key = key
        return self._background

    def paintEvent(self, event) -> None:  # noqa: N802
        """Composite the cached illustration with a handful of moving gliders."""
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self._ensure_background())
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.scale(self.width() / _SCENE_WIDTH, self.height() / _SCENE_HEIGHT)
        seconds = self._animation_time()
        for index, hex_color in enumerate(_FLIGHT_COLORS):
            color = QColor(hex_color)
            # Short, fading wakes communicate motion without suggesting real data.
            for segment in range(46):
                age = (46 - segment) * 0.085
                point = _flight_position(seconds - age, index)
                next_point = _flight_position(seconds - age + 0.085, index)
                trail_color = QColor(color)
                trail_color.setAlpha(round(72 * (segment / 46) ** 1.5))
                painter.setPen(QPen(trail_color, 1.1))
                painter.drawLine(point + QPointF(0, 12), next_point + QPointF(0, 12))
            position = _flight_position(seconds, index)
            next_position = _flight_position(seconds + 0.12, index)
            bank = max(-16, min(16, (next_position.x() - position.x()) * 2.5))
            scale = (0.82, 0.63, 0.75, 0.48, 0.55)[index]
            _draw_paraglider(painter, position, color, bank, scale)
        painter.end()
