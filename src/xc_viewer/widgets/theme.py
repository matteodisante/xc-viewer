"""A lightweight, shared light theme for the desktop viewer's Qt controls."""

from pathlib import Path

from PyQt6.QtGui import QColor, QFontDatabase, QPalette
from PyQt6.QtWidgets import QApplication

# Static styling only: no effects, animation, custom fonts or rendering workers.
_STYLE = """
QMainWindow { background: #eef3f5; }
QLabel[role="heading"] { color: #193f4a; font-size: 17px; font-weight: 600; }
QLabel[role="section"] { color: #355863; font-weight: 600; }
QLabel[role="muted"] { color: #627781; font-size: 11px; }

QGroupBox {
    background: #ffffff; border: 1px solid #d7e2e7; border-radius: 6px;
    margin-top: 9px; padding: 8px;
}
QGroupBox::title {
    subcontrol-origin: margin; subcontrol-position: top left;
    left: 10px; padding: 0 4px; color: #355863;
}
QTabWidget::pane { background: #ffffff; border: 1px solid #d7e2e7; }
QTabBar::tab {
    background: #eef3f5; color: #536e79;
    padding: 8px 10px; border: none; border-bottom: 2px solid transparent;
}
QTabBar::tab:selected {
    background: #ffffff; color: #155e6b; border-bottom: 2px solid #247c86;
}
QTabBar::tab:hover:!selected { background: #e2ecef; }
QTabBar::tab:disabled { color: #8a9ba3; }

QPushButton {
    background: #ffffff; color: #294650; border: 1px solid #cadbe1;
    border-radius: 5px; padding: 4px 9px; min-height: 16px;
}
QPushButton:hover { background: #edf5f5; border-color: #8eb6be; }
QPushButton:pressed, QPushButton:checked {
    background: #dceeed; color: #155e6b; border-color: #7aabb4;
}
QPushButton:focus { border-color: #247c86; }
QPushButton[emphasis="primary"] {
    background: #216f7b; color: #ffffff; border-color: #216f7b;
    font-weight: 600;
}
QPushButton[emphasis="primary"]:hover { background: #195f6b; }
QPushButton[emphasis="primary"]:pressed { background: #124f5b; }
QPushButton[emphasis="primary"]:focus { border: 2px solid #78b9c3; }
QPushButton[emphasis="quiet"] { background: transparent; border-color: transparent; }
QPushButton[emphasis="quiet"]:hover { background: #e4eff1; }
QPushButton[emphasis="quiet"]:focus { border-color: #247c86; }
QPushButton:disabled, QPushButton[emphasis="primary"]:disabled {
    color: #82939b; background: #edf1f3; border-color: #dce5e9;
}

QComboBox, QLineEdit, QAbstractSpinBox {
    color: #263f49; background: #ffffff;
    border: 1px solid #cadbe1; border-radius: 4px; padding: 3px 6px;
    selection-background-color: #216f7b; selection-color: #ffffff;
}
QComboBox { padding-right: 22px; }
QComboBox::drop-down { border: none; width: 20px; }
QComboBox::down-arrow { image: url("CHEVRON_PATH"); width: 10px; height: 7px; }
QComboBox QAbstractItemView {
    background: #ffffff; color: #263f49; border: 1px solid #cadbe1;
    selection-background-color: #dceeed; selection-color: #164e5a;
}
QComboBox:hover, QLineEdit:hover, QAbstractSpinBox:hover { border-color: #9cbac3; }
QComboBox:focus, QLineEdit:focus, QAbstractSpinBox:focus { border-color: #247c86; }
QComboBox:disabled, QLineEdit:disabled, QAbstractSpinBox:disabled {
    background: #edf1f3; color: #82939b; border-color: #dce5e9;
}
QCheckBox, QRadioButton { spacing: 5px; }
QCheckBox:disabled, QRadioButton:disabled { color: #82939b; }

QToolBar {
    background: #f7f9fa; border: none; spacing: 2px; padding: 2px;
    qproperty-iconSize: 18px 18px;
}
QToolButton {
    background: transparent; border: 1px solid transparent;
    border-radius: 4px; padding: 3px;
}
QToolButton:hover { background: #e4eff1; }
QToolButton:checked { background: #dceeed; border-color: #9cbdc4; }
QToolButton:focus { border-color: #247c86; }
QToolBar::separator { background: #d7e2e7; width: 1px; margin: 5px 3px; }
QSplitter::handle { background: #dce6ea; }
QSplitter::handle:hover { background: #8fb8c0; }

QTableView, QListView, QTreeView, QTextBrowser {
    background: #ffffff; alternate-background-color: #f3f7f8;
    color: #263f49; border: 1px solid #d7e2e7;
    selection-background-color: #dceeed; selection-color: #164e5a;
    gridline-color: #edf2f4;
}
QHeaderView::section {
    background: #f0f5f6; color: #47636e; border: none;
    border-bottom: 1px solid #d7e2e7; padding: 5px 6px;
}
QTableCornerButton::section { background: #f0f5f6; border: none; }
QScrollBar:vertical { background: #edf2f4; width: 10px; margin: 0; }
QScrollBar:horizontal { background: #edf2f4; height: 10px; margin: 0; }
QScrollBar::handle {
    background: #b9ccd3; border: 2px solid #edf2f4; border-radius: 5px;
    min-width: 24px; min-height: 24px;
}
QScrollBar::handle:hover { background: #8caeb9; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }
QProgressBar {
    border: 1px solid #d7e2e7; border-radius: 4px;
    background: #edf2f4; text-align: center; color: #264650;
}
QProgressBar::chunk { background: #76b6ba; border-radius: 3px; }
QMenuBar { background: #eef3f5; color: #35535e; }
QMenu { background: #ffffff; color: #263f49; border: 1px solid #cadbe1; }
QMenu::item { padding: 5px 20px; }
QMenu::item:selected { background: #dceeed; }
QToolTip {
    background: #ffffff; color: #263f49;
    border: 1px solid #cadbe1; padding: 5px;
}
"""


def apply_theme(app: QApplication) -> None:
    """Apply the same compact controls and light palette on every desktop OS."""
    app.setStyle("Fusion")
    palette = QPalette()
    for role, color in (
        (QPalette.ColorRole.Window, "#eef3f5"),
        (QPalette.ColorRole.WindowText, "#263f49"),
        (QPalette.ColorRole.Base, "#ffffff"),
        (QPalette.ColorRole.AlternateBase, "#f3f7f8"),
        (QPalette.ColorRole.Text, "#263f49"),
        (QPalette.ColorRole.Button, "#ffffff"),
        (QPalette.ColorRole.ButtonText, "#294650"),
        (QPalette.ColorRole.Highlight, "#216f7b"),
        (QPalette.ColorRole.HighlightedText, "#ffffff"),
        (QPalette.ColorRole.ToolTipBase, "#ffffff"),
        (QPalette.ColorRole.ToolTipText, "#263f49"),
        (QPalette.ColorRole.Link, "#176d7a"),
        (QPalette.ColorRole.LinkVisited, "#536d89"),
        (QPalette.ColorRole.Light, "#ffffff"),
        (QPalette.ColorRole.Midlight, "#e5edf0"),
        (QPalette.ColorRole.Mid, "#c5d6dd"),
        (QPalette.ColorRole.Dark, "#6c8792"),
        (QPalette.ColorRole.Shadow, "#365560"),
        (QPalette.ColorRole.BrightText, "#ffffff"),
        (QPalette.ColorRole.PlaceholderText, "#6b818b"),
        (QPalette.ColorRole.Accent, "#216f7b"),
    ):
        palette.setColor(role, QColor(color))
    for role in (
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
    ):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor("#82939b"))
    app.setPalette(palette)
    font = QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont)
    font.setPointSizeF(max(10.0, font.pointSizeF()))
    app.setFont(font)
    chevron = Path(__file__).resolve().parents[1] / "assets/ui/chevron-down.svg"
    app.setStyleSheet(_STYLE.replace("CHEVRON_PATH", chevron.as_posix()))
