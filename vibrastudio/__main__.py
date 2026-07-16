"""Entry point: ``python -m vibrastudio``."""

from __future__ import annotations

import sys


def _selftest() -> int:
    """Write a bundle sanity report next to the executable and exit.

    Lets us verify a frozen build (windowed exe, no console) actually ships
    nidaqmx and OpenGL: ``VibraStudio.exe --selftest``.
    """
    import json
    from pathlib import Path

    report: dict[str, object] = {}
    try:
        import nidaqmx  # noqa: F401
        report["nidaqmx"] = True
    except Exception as exc:
        report["nidaqmx"] = f"FAIL: {exc}"
    try:
        import pyqtgraph.opengl  # noqa: F401
        report["opengl"] = True
    except Exception as exc:
        report["opengl"] = f"FAIL: {exc}"

    out = Path(sys.executable).parent / "selftest.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))
    return 0 if all(v is True for v in report.values()) else 1


def main() -> int:
    if "--selftest" in sys.argv:
        return _selftest()

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
