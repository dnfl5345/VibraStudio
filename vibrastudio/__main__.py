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

    try:
        import matplotlib  # noqa: F401
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        report["matplotlib"] = True
    except Exception as exc:
        report["matplotlib"] = f"FAIL: {exc}"
    try:
        import cv2
        report["opencv"] = bool(cv2.__version__)
    except Exception as exc:
        report["opencv"] = f"FAIL: {exc}"
    try:
        import imageio_ffmpeg
        report["ffmpeg"] = Path(imageio_ffmpeg.get_ffmpeg_exe()).exists()
    except Exception as exc:
        report["ffmpeg"] = f"FAIL: {exc}"

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


def _ods_video(argv: list[str]) -> int:
    """Headless 3D ODS video export from a recording.

    VibraStudio.exe --ods-video FILE [--rig RIG.json] [--mode peaks|list|sweep|time]
        [--freqs 7,46.1] [--fmin 3 --fmax 500 --peaks 6] [--t0 S --t1 S --slow 20]
        [--lo 3,3,3 --hi 400] [--size 1920x1080] [--fps 30] [--out FILE.mp4]
    """
    import argparse
    from pathlib import Path

    from .daq.playback import load_recording
    from .dsp.ods_video import model_from
    from .rig import Rig
    from .ui.ods_video_render import VideoOptions, build_frames, resolve_mapping, write_video

    ap = argparse.ArgumentParser(prog="VibraStudio --ods-video")
    ap.add_argument("file")
    ap.add_argument("--rig"); ap.add_argument("--map", default="auto", choices=["auto", "rig", "file"])
    ap.add_argument("--mode", default="peaks", choices=["peaks", "list", "sweep", "time"])
    ap.add_argument("--freqs", default="")
    ap.add_argument("--peaks", type=int, default=6)
    ap.add_argument("--fmin", type=float, default=3.0); ap.add_argument("--fmax", type=float, default=500.0)
    ap.add_argument("--each", type=float, default=3.0); ap.add_argument("--sweep-seconds", type=float, default=20.0)
    ap.add_argument("--t0", type=float, default=0.0); ap.add_argument("--t1", type=float, default=None)
    ap.add_argument("--slow", type=float, default=20.0)
    ap.add_argument("--lo", default="3,3,3"); ap.add_argument("--hi", type=float, default=400.0)
    ap.add_argument("--size", default="1920x1080"); ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--fixed-base", action="store_true",
                    help="keep fixed nodes at rest even next to a sensor (default: they follow it)")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    rec = load_recording(a.file)
    rig = Rig.load(a.rig) if a.rig else None
    chans, geo, how = resolve_mapping(rec, rig, a.map)
    model = model_from(geo, chans, rec.data.shape[0])
    w, h = (int(v) for v in a.size.lower().split("x"))
    o = VideoOptions(mode=a.mode, freqs=tuple(float(v) for v in a.freqs.split(",") if v.strip()),
                     n_peaks=a.peaks, fmin=a.fmin, fmax=a.fmax, seconds_each=a.each,
                     sweep_seconds=a.sweep_seconds, t0=a.t0, t1=a.t1, slow=a.slow,
                     lo_xyz=tuple(float(v) for v in a.lo.split(",")), hi_hz=a.hi, fps=a.fps, size=(w, h),
                     base_follows=not a.fixed_base)
    frames, strip = build_frames(rec.data, rec.fs, model, o, Path(a.file).stem)
    out = Path(a.out or Path(a.file).with_name(f"{Path(a.file).stem}_ods_{a.mode}.mp4"))
    print(f"mapping: {how}; {len(frames)} frames -> {out}")
    res = write_video(frames, model, strip, out, (w, h), a.fps, stills=(len(frames) // 3,))
    print(f"written {res}")
    return 0


def _magnify(argv: list[str]) -> int:
    """Headless motion magnification vs sensors.

    VibraStudio.exe --magnify --video V.mp4 --recording R.npz --rig RIG.json
        [--fps F | --mains 60 [--fps-hint 250]] [--alphas 5,10,20 | auto] [--out DIR]
        [--synth [--synth-freqs 5,18.9,37.8 | auto] [--synth-source auto|sensors|fusion]
                 [--synth-sensors S1,S8] [--synth-target-px 10] [--beam-length-cm 50]]
    """
    import argparse
    from pathlib import Path

    from .daq.playback import load_recording
    from .dsp import magnify as mg
    from .dsp import ods_video as ov
    from .rig import Rig
    from .dsp.synth import SynthOptions
    from .ui.magnify_render import overlay_video, save_results, summary_figure, synthesis_figure, synthesis_video
    from .ui.ods_video_render import resolve_mapping

    ap = argparse.ArgumentParser(prog="VibraStudio --magnify")
    ap.add_argument("--synth", action="store_true", help="add the per-frequency sensor-guided synthesis")
    ap.add_argument("--synth-freqs", default="auto", help="auto, or Hz separated by commas")
    ap.add_argument("--synth-source", default="auto", choices=["auto", "sensors", "fusion"])
    ap.add_argument("--synth-sensors", default="", help="labels the shape may use, e.g. S1,S8 (default: all)")
    ap.add_argument("--synth-target-px", type=float, default=10.0)
    ap.add_argument("--beam-length-cm", type=float)
    ap.add_argument("--video", required=True); ap.add_argument("--recording", required=True)
    ap.add_argument("--rig", required=True)
    ap.add_argument("--fps", type=float); ap.add_argument("--fps-hint", type=float)
    ap.add_argument("--mains", type=float, default=60.0)
    ap.add_argument("--alphas", default="auto"); ap.add_argument("--target-px", type=float, default=12.0)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    out = Path(a.out or Path(a.video).with_name(Path(a.video).stem + "_magnify"))
    out.mkdir(parents=True, exist_ok=True)
    rec = load_recording(a.recording)
    chans, geo, how = resolve_mapping(rec, Rig.load(a.rig), "rig")
    model = ov.model_from(geo, chans, rec.data.shape[0])
    acc = ov.node_acceleration(rec.data, model, base_follows=False)
    print(f"sensor map: {how}")
    alphas = None if a.alphas == "auto" else [float(v) for v in a.alphas.split(",")]
    synth = None
    if a.synth:
        synth = SynthOptions(
            freqs=None if a.synth_freqs == "auto" else [float(v) for v in a.synth_freqs.split(",")],
            target_px=a.synth_target_px, source=a.synth_source,
            use_sensors=[v.strip() for v in a.synth_sensors.split(",") if v.strip()] or None,
            length_cm=a.beam_length_cm)
    r = mg.run(a.video, model, acc, rec.fs, fps=a.fps, mains=a.mains, fps_hint=a.fps_hint,
               alphas=alphas, target_px=a.target_px, synth=synth)
    save_results(r, out, a.video, a.recording)
    summary_figure(r, out, f"{Path(a.video).name}  vs  {Path(a.recording).name}")
    overlay_video(r, out)
    if r.synth:
        synthesis_figure(r, out, f"{Path(a.video).name}  vs  {Path(a.recording).name}")
        synthesis_video(r, out)
    for row in r.per_sensor:
        print("  ", row)
    print(f"written to {out}")
    return 0


def main() -> int:
    if "--magnify" in sys.argv:
        i = sys.argv.index("--magnify")
        return _magnify(sys.argv[i + 1:])
    if "--ods-video" in sys.argv:
        i = sys.argv.index("--ods-video")
        return _ods_video(sys.argv[i + 1:])
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
