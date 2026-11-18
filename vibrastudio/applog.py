"""A small text log next to the program: what happened to the rig and the DAQ connection.

The exe has no console, so without this a failure at start-up leaves no trace.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

MAX_BYTES = 200_000


def log_path() -> Path:
    base = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path.cwd()
    return base / "vibrastudio.log"


def log(msg: str) -> None:
    """Append one line; never raises."""
    try:
        p = log_path()
        if p.exists() and p.stat().st_size > MAX_BYTES:
            p.write_text(p.read_text(encoding="utf-8", errors="replace")[-MAX_BYTES // 2:], encoding="utf-8")
        with p.open("a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}\n")
    except Exception:  # noqa: BLE001 - logging must not break the app
        pass
