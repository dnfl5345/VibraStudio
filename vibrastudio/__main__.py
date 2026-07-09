"""Entry point: ``python -m vibrastudio``."""

from __future__ import annotations

import sys


def main() -> int:
    from PySide6.QtWidgets import QApplication

    from .config import AppConfig
    from .ui import theme
    from .ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("VibraStudio")
    theme.apply_pyqtgraph_theme()
    app.setStyleSheet(theme.STYLESHEET)

    cfg = AppConfig()
    if "--simulate" in sys.argv:
        cfg.acquisition.simulate = True

    win = MainWindow(cfg)
    if "--simulate" in sys.argv:
        win.sidebar.source_box.setCurrentIndex(1)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
