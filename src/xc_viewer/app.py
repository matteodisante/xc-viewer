"""Entry point for the ``xc-viewer`` console script."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import datafolder


def configure_graphics() -> None:
    """Choose a shared core profile before Qt creates any internal contexts.

    macOS cannot share the terrain widget's core-profile textures with Qt's
    default legacy context. Configure all windows before QApplication exists,
    including the backing-store compositor used by modeless 3D dialogs.
    """
    from PyQt6.QtCore import QCoreApplication, Qt
    from PyQt6.QtGui import QSurfaceFormat

    fmt = QSurfaceFormat()
    fmt.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setDepthBufferSize(24)
    fmt.setStencilBufferSize(8)
    fmt.setSamples(4)
    QSurfaceFormat.setDefaultFormat(fmt)
    QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)


def use_system_certificates() -> None:
    """Verify HTTPS downloads with the operating system's certificates, as a browser.

    Company networks that inspect HTTPS re-sign it with their own certificate, which
    the IT department installs in the system. Python's bundled certificates do not
    know it, so map downloads would fail with an unknown-issuer error.
    """
    import truststore

    truststore.inject_into_ssl()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """The command line: an optional data folder."""
    parser = argparse.ArgumentParser(
        prog="xc-viewer",
        description="Interactive viewer of cross-country soaring flights and thermals.",
    )
    parser.add_argument(
        "data_folder",
        nargs="?",
        type=Path,
        help=(
            "Folder holding paragliders/ and/or hang_gliders/. Remembered for the "
            "next launch; without it the last folder is reused, or a dialog asks."
        ),
    )
    return parser.parse_args(argv)


def choose_data_folder(parent=None, start: str | Path = "") -> Path | None:
    """Ask for a data folder until a usable one is chosen or the user cancels.

    Args:
        parent: The dialog's parent widget, if any.
        start: The folder the dialog opens in.

    Returns:
        The chosen folder, or ``None`` if the user cancelled.
    """
    from PyQt6.QtWidgets import QFileDialog, QMessageBox

    while True:
        chosen = QFileDialog.getExistingDirectory(
            parent, "Choose the XC Viewer data folder", str(start)
        )
        if not chosen:
            return None
        problems = datafolder.missing_parts(chosen)
        if not problems:
            return Path(chosen)
        QMessageBox.warning(parent, "Not a data folder", "\n".join(problems))
        start = chosen


def _initial_data_folder(
    argument: Path | None, settings, parent=None
) -> tuple[Path | None, bool]:
    """The data folder to start with, and whether to remember it.

    The command-line argument wins, then the environment variable (used for this run
    only), then the folder used last time, then a dialog.
    """
    if argument is not None:
        problems = datafolder.missing_parts(argument)
        if problems:
            raise SystemExit("\n".join(problems))
        return argument, True
    if datafolder.current() is not None:
        return datafolder.current(), False
    remembered = settings.value(datafolder.SETTINGS_KEY, "", type=str)
    if remembered and not datafolder.missing_parts(remembered):
        return Path(remembered), False
    return choose_data_folder(parent, start=remembered), True


def main(argv: list[str] | None = None) -> int:
    """Launch the viewer. Returns the process exit code."""
    args = _parse_args(argv)
    if args.data_folder is not None:
        problems = datafolder.missing_parts(args.data_folder)
        if problems:
            raise SystemExit("\n".join(problems))
    use_system_certificates()
    configure_graphics()

    from PyQt6.QtCore import QSettings, QTimer
    from PyQt6.QtWidgets import QApplication, QMessageBox

    from .widgets.about import AboutDialog
    from .widgets.theme import apply_theme
    from .widgets.welcome import WelcomeScreen

    app = QApplication(sys.argv[:1])
    app.setOrganizationName("xc-viewer")
    app.setApplicationName("xc-viewer")
    apply_theme(app)
    settings = QSettings()
    welcome = WelcomeScreen()
    window = None
    about = None
    launching = False

    def show_about() -> None:
        nonlocal about
        if about is None:
            about = AboutDialog(welcome)
        about.show()
        about.raise_()
        about.activateWindow()

    def load_viewer() -> None:
        nonlocal window, launching
        try:
            folder, remember = _initial_data_folder(args.data_folder, settings, welcome)
            if folder is not None:
                folder = datafolder.use(folder)
                if remember:
                    settings.setValue(datafolder.SETTINGS_KEY, str(folder))
            # Keep the welcome screen quick: plots and archive readers load only
            # after Start, with the chosen data folder already in place.
            from .main_window import MainWindow

            window = MainWindow()
            # Carry over the screen, normal size/position and native fullscreen or
            # maximized state before showing the analysis window.
            window.restoreGeometry(welcome.saveGeometry())
            window.show()
        except (Exception, SystemExit) as error:
            # Exceptions must not escape a Qt callback, which would abort the app.
            import traceback

            traceback.print_exc()
            QMessageBox.warning(welcome, "Unable to open the viewer", str(error))
            launching = False
            welcome.set_loading(False)
            return
        if about is not None:
            about.close()
        welcome.close()

    def enter_viewer() -> None:
        nonlocal launching
        if launching:
            return
        launching = True
        welcome.set_loading(True)
        # Give the button's loading state a chance to paint before importing plots.
        QTimer.singleShot(0, load_viewer)

    welcome.start_requested.connect(enter_viewer)
    welcome.about_requested.connect(show_about)
    welcome.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
