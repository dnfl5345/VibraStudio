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


def _livecheck(seconds: float) -> int:
    """Headless real-time ingest test over every detected input.

    Connects (registering/reserving chassis as needed), streams all channels
    for ``seconds`` and writes livecheck.txt / livecheck.json: delivered
    sample rate vs nominal, driver backlog, and a state per input.
    ``VibraStudio.exe --livecheck 30``
    """
    import json
    import time
    from pathlib import Path

    from .config import AppConfig, channels_from_detected
    from .daq.engine import AcquisitionEngine, auto_connect, split_physical
    from .dsp import health as hl

    frozen = getattr(sys, "frozen", False)
    out_dir = Path(sys.executable).parent if frozen else Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    report: dict = {"seconds": seconds}

    def finish(code: int) -> int:
        report["exit_code"] = code
        (out_dir / "livecheck.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (out_dir / "livecheck.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        try:
            print("\n".join(lines))
        except UnicodeEncodeError:      # legacy console code page (cp949)
            print("\n".join(lines).encode("ascii", "replace").decode())
        return code

    res = auto_connect()
    report["connect"] = {"ok": res.ok, "chassis": res.chassis, "added": res.added,
                         "reserved": res.reserved, "messages": res.messages,
                         "channels": len(res.channels)}
    lines.append("== connect ==")
    lines.extend(res.summary().splitlines())
    if not res.ok:
        lines.append("RESULT: FAIL (no AI channels)")
        return finish(2)

    cfg = AppConfig()
    channels = channels_from_detected(res.channels)
    engine = AcquisitionEngine(channels, cfg.acquisition)
    try:
        engine.start()
    except Exception as exc:
        lines.append(f"RESULT: FAIL (start: {str(exc).splitlines()[0]})")
        return finish(2)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds and not engine.error:
        time.sleep(0.25)
    fs = engine.fs
    data = engine.ring.latest(int(fs))
    rate, age = engine.measured_rate(), engine.block_age()
    stream = {"nominal_rate_hz": fs, "measured_rate_hz": rate,
              "backlog_samples": engine.backlog, "max_backlog_samples": engine.max_backlog,
              "max_gap_s": engine.max_gap, "blocks": engine.blocks,
              "samples_per_channel": engine.ring.total, "error": engine.error}
    engine.stop()

    verdict, reason = hl.stream_verdict(fs, rate, stream["backlog_samples"], age,
                                        cfg.acquisition.chunk_size)
    if engine.error:
        verdict, reason = "FAIL", engine.error
    report["stream"] = dict(stream, verdict=verdict, reason=reason)
    lines.append("== stream ==")
    lines.append(f"{len(channels)} channels, {engine.ring.total:,} samples/ch in {seconds:.0f} s")
    lines.append(f"rate {rate:,.1f} S/s (nominal {fs:,.0f}, {100 * (rate / fs - 1):+.2f} %)"
                 f"  backlog peak {engine.max_backlog} samples  longest gap "
                 f"{1e3 * engine.max_gap:.0f} ms")
    lines.append(f"stream: {verdict} ({reason})")

    groups = [split_physical(c.physical_channel)[:2] for c in channels]
    health = hl.assess_channels(data, fs, groups, full_scale=abs(cfg.acquisition.max_val),
                                settling=seconds < 22.0)
    report["channels"] = [
        {"label": c.label, "physical_channel": c.physical_channel, "state": h.state,
         "rms_g": h.rms, "peak_g": h.peak, "f_dom_hz": h.f_dom}
        for c, h in zip(channels, health)]
    counts: dict[str, int] = {}
    lines.append("== inputs ==")
    for c, h in zip(channels, health):
        counts[h.state] = counts.get(h.state, 0) + 1
        lines.append(f"{c.physical_channel:<28} {c.label:<7} {h.state:<9} "
                     f"{1e3 * h.rms:9.3f} mg  {h.f_dom:7.1f} Hz")
    report["counts"] = counts
    lines.append("  ".join(f"{k} {v}" for k, v in counts.items()))
    ok = verdict != "FAIL" and counts.get(hl.NO_DATA, 0) == 0
    if ok:
        lines.append(f"RESULT: PASS — all {len(channels)} inputs are digitized in real time.")
        unsure = counts.get(hl.QUIET, 0) + counts.get(hl.COMMON, 0) + counts.get(hl.HUM, 0)
        if unsure:
            lines.append(f"NOTE: {unsure} inputs show no sensor-like signal "
                         "(QUIET/COMMON/HUM) — confirm with a tap test in the DAQ Health tab.")
    else:
        lines.append("RESULT: FAIL — the stream is not keeping up or an input returns no data.")
    return finish(0 if ok else 1)


def main() -> int:
    if "--selftest" in sys.argv:
        return _selftest()
    if "--livecheck" in sys.argv:
        i = sys.argv.index("--livecheck")
        try:
            secs = float(sys.argv[i + 1])
        except (IndexError, ValueError):
            secs = 30.0
        return _livecheck(secs)

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
