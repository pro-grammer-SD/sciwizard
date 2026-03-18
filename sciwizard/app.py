"""Application bootstrap — sets up Qt, Windows AppUserModelID, and runs the event loop."""

from __future__ import annotations

import ctypes
import logging
import sys

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from sciwizard.config import APP_ID, APP_NAME, APP_VERSION, ICON_PATH
from sciwizard.ui.main_window import MainWindow
from sciwizard.ui.theme import apply_dark_theme

logger = logging.getLogger(__name__)


def _set_windows_app_id() -> None:
    """Set Windows AppUserModelID so the taskbar icon is correct."""
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
        logger.debug("AppUserModelID set to %s", APP_ID)
    except AttributeError:
        # Not on Windows
        pass


def main() -> None:
    """Bootstrap and launch SciWizard."""
    _set_windows_app_id()

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setOrganizationName("SciWizard")

    if ICON_PATH.exists():
        app.setWindowIcon(QIcon(str(ICON_PATH)))
        logger.debug("Icon loaded from %s", ICON_PATH)
    else:
        logger.warning("Icon not found at %s — skipping", ICON_PATH)

    apply_dark_theme(app)

    window = MainWindow()
    window.show()

    logger.info("SciWizard %s started", APP_VERSION)
    sys.exit(app.exec())
