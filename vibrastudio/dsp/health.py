"""Channel health: is live, plausible data arriving on every input?

NI-9230 inputs offer no hardware open-sensor detection in a synchronized
multi-device task, so each channel is judged from its signal:

* NO DATA  — constant or non-finite samples (nothing is being digitized)
* SETTLING — IEPE bias still charging after start (large offset / railing)
* CLIP     — peaks at the input range
* COMMON   — the same waveform on several inputs of one module. A triaxial
             sensor never produces identical axes, so this is a wiring or
             ground pickup, not vibration.
* HUM      — dominated by mains frequency and its harmonics
* QUIET    — at the noise floor: an idle sensor *or* an empty input. Only a
             tap on the sensor tells the two apart.
* LIVE     — a signal above the noise floor
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LIVE, QUIET, HUM, COMMON, CLIP, SETTLING, NO_DATA = (
    "LIVE", "QUIET", "HUM", "COMMON", "CLIP", "SETTLING", "NO DATA")

STATE_HELP = {
    LIVE: "Signal above the noise floor.",
    QUIET: "Noise floor only: idle sensor or nothing connected. Tap the sensor to confirm.",
    HUM: "Dominated by mains frequency and harmonics: check shielding/grounding.",
    COMMON: "Identical waveform on several inputs of this module: wiring or ground "
            "pickup, not a triaxial sensor response.",
    CLIP: "Peaks reach the input range: reduce excitation or check the sensor.",
    SETTLING: "IEPE bias still settling after start (about 20 s).",
    NO_DATA: "Constant or invalid samples: no data from this input.",
}


@dataclass
class ChannelHealth:
    state: str
    rms: float          # AC RMS, g
    peak: float         # peak |AC|, g
    mean: float         # DC offset, g
    f_dom: float        # dominant frequency, Hz
    hum_share: float    # fraction of energy at mains harmonics


def assess_channels(
    data: np.ndarray,
    fs: float,
    groups: list,
    full_scale: float = 50.0,
    line_hz: float = 60.0,
    quiet_g: float = 0.003,
    settling: bool = False,
) -> list[ChannelHealth]:
    """Classify every row of ``data`` (n_channels, n_samples, unit g).

    ``groups`` holds one hashable key per channel; channels sharing a key
    sit on the same module and are compared for the COMMON state.
    """
    n_ch, n = data.shape
    if n < 64:
        return [ChannelHealth(NO_DATA, 0.0, 0.0, 0.0, 0.0, 0.0) for _ in range(n_ch)]

    finite = np.all(np.isfinite(data), axis=1)
    safe = np.where(np.isfinite(data), data, 0.0)
    mean = safe.mean(axis=1)
    x = safe - mean[:, None]
    rms = x.std(axis=1)
    peak = np.max(np.abs(x), axis=1)
    absmax = np.max(np.abs(safe), axis=1)
    # Judge "constant" on the raw samples: after mean removal a flat line
    # leaves rounding residue and would pass as noise.
    flat = np.ptp(safe, axis=1) == 0.0

    spec = np.abs(np.fft.rfft(x * np.hanning(n)[None, :], axis=1)) ** 2
    freq = np.fft.rfftfreq(n, 1.0 / fs)
    df = freq[1] if freq.size > 1 else 1.0
    lo = max(1, int(np.ceil(3.0 / df)))            # ignore drift below 3 Hz
    total = np.maximum(spec[:, lo:].sum(axis=1), 1e-30)
    f_dom = freq[lo + np.argmax(spec[:, lo:], axis=1)]
    hum = np.zeros(n_ch)
    half = max(1, int(round(1.5 / df)))
    for h in range(1, 8):
        k = int(round(line_hz * h / df))
        if lo <= k < spec.shape[1]:
            hum += spec[:, max(lo, k - half):k + half + 1].sum(axis=1)
    hum_share = np.clip(hum / total, 0.0, 1.0)

    # Same-module channels carrying one waveform.
    common = np.zeros(n_ch, dtype=bool)
    by_group: dict = {}
    for i, key in enumerate(groups):
        by_group.setdefault(key, []).append(i)
    for idx in by_group.values():
        if len(idx) < 2:
            continue
        sub = x[idx]
        if np.any(sub.std(axis=1) <= 0):
            continue
        corr = np.corrcoef(sub)
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                similar = abs(rms[idx[a]] - rms[idx[b]]) <= 0.05 * max(rms[idx[a]], rms[idx[b]])
                if corr[a, b] >= 0.98 and similar:
                    common[idx[a]] = common[idx[b]] = True

    out: list[ChannelHealth] = []
    for i in range(n_ch):
        clipped = absmax[i] >= 0.95 * full_scale
        if not finite[i] or flat[i]:
            state = NO_DATA
        elif settling and (clipped or abs(mean[i]) > 0.2):
            state = SETTLING
        elif clipped:
            state = CLIP
        elif common[i]:
            state = COMMON
        elif hum_share[i] > 0.5 and rms[i] >= quiet_g:
            state = HUM
        elif rms[i] < quiet_g:
            state = QUIET
        else:
            state = LIVE
        out.append(ChannelHealth(state, float(rms[i]), float(peak[i]), float(mean[i]),
                                 float(f_dom[i]), float(hum_share[i])))
    return out


def stream_verdict(nominal_fs: float, measured_fs: float, backlog: int,
                   block_age: float, chunk: int) -> tuple[str, str]:
    """('OK' | 'WARN' | 'FAIL', reason) for the stream as a whole."""
    if measured_fs <= 0:
        return "FAIL", "no blocks received"
    if block_age > max(1.0, 10.0 * chunk / nominal_fs):
        return "FAIL", f"stream stalled for {block_age:.1f} s"
    err = abs(measured_fs - nominal_fs) / nominal_fs
    if err > 0.05:
        return "FAIL", f"rate off by {100 * err:.1f} %"
    if backlog > 2.0 * nominal_fs:
        return "FAIL", f"reader {backlog / nominal_fs:.1f} s behind the hardware"
    if err > 0.01:
        return "WARN", f"rate off by {100 * err:.1f} %"
    if backlog > 0.5 * nominal_fs:
        return "WARN", f"reader {backlog / nominal_fs:.2f} s behind the hardware"
    return "OK", "real-time"
