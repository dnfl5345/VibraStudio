"""Streaming spectral analysis engine.

Benchmarked against the LabVIEW Sound & Vibration Measurement Suite:
windowed FFT with amplitude/energy correction, linear / exponential /
peak-hold averaging, PSD, single-/double-integration (accel -> velocity
-> displacement), cross-spectrum accumulation for FRF (H1),
transmissibility, coherence, and ODS amplitude/phase extraction.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as spsig
from scipy.stats import kurtosis, skew

G_TO_MS2 = 9.80665

WINDOWS = {
    "rectangular": lambda n: np.ones(n),
    "hann": lambda n: spsig.windows.hann(n, sym=False),
    "hamming": lambda n: spsig.windows.hamming(n, sym=False),
    "blackman": lambda n: spsig.windows.blackman(n, sym=False),
    "blackmanharris": lambda n: spsig.windows.blackmanharris(n, sym=False),
    "flattop": lambda n: spsig.windows.flattop(n, sym=False),
}


class StreamingAnalyzer:
    """Block-wise FFT engine with averaged auto- and cross-spectra.

    Raw averaged quantities (Sxx = <|X|^2>, Sxy = <X_ref* . X_i> where
    X = rfft(w * x)) are kept internally; display scalings (peak amplitude,
    RMS, PSD, dB) are derived on demand so the averaging state survives a
    change of display units.
    """

    def __init__(self, n_channels: int, fs: float) -> None:
        self.n_ch = n_channels
        self.fs = float(fs)
        self.nfft = 8192
        self.window_name = "hann"
        self.averaging = "exponential"
        self.avg_count = 8
        self.overlap = 0.5
        self.ref_index = 0
        self._configure_arrays()

    # ------------------------------------------------------------------
    def configure(
        self,
        nfft: int | None = None,
        window: str | None = None,
        averaging: str | None = None,
        avg_count: int | None = None,
        overlap: float | None = None,
        ref_index: int | None = None,
    ) -> None:
        if nfft is not None:
            self.nfft = int(nfft)
        if window is not None:
            self.window_name = window
        if averaging is not None:
            self.averaging = averaging
        if avg_count is not None:
            self.avg_count = max(1, int(avg_count))
        if overlap is not None:
            self.overlap = min(0.9, max(0.0, float(overlap)))
        if ref_index is not None:
            self.ref_index = int(ref_index)
        self._configure_arrays()

    def _configure_arrays(self) -> None:
        self.win = WINDOWS[self.window_name](self.nfft)
        self.s1 = float(np.sum(self.win))            # amplitude normalization
        self.s2 = float(np.sum(self.win ** 2))       # energy normalization
        self.freq = np.fft.rfftfreq(self.nfft, d=1.0 / self.fs)
        nb = self.freq.size
        self.sxx = np.zeros((self.n_ch, nb))
        self.sxy = np.zeros((self.n_ch, nb), dtype=np.complex128)
        self.navg = 0
        self._acc = np.zeros((self.n_ch, 0))
        self.hop = max(1, int(self.nfft * (1.0 - self.overlap)))

    def reset_averaging(self) -> None:
        self.sxx[:] = 0.0
        self.sxy[:] = 0.0
        self.navg = 0

    # ------------------------------------------------------------------
    def push(self, block: np.ndarray) -> int:
        """Feed new raw samples; process every complete hop. Returns #blocks."""
        if block.size == 0:
            return 0
        self._acc = np.concatenate((self._acc, block), axis=1)
        processed = 0
        while self._acc.shape[1] >= self.nfft:
            frame = self._acc[:, : self.nfft]
            self._process_frame(frame)
            self._acc = self._acc[:, self.hop :]
            processed += 1
            if processed >= 8:  # bound the work per GUI tick
                keep = min(self._acc.shape[1], self.nfft)
                self._acc = self._acc[:, -keep:]
                break
        # Bound accumulation growth if GUI stalls.
        if self._acc.shape[1] > 4 * self.nfft:
            self._acc = self._acc[:, -self.nfft :]
        return processed

    def _process_frame(self, frame: np.ndarray) -> None:
        x = frame - frame.mean(axis=1, keepdims=True)
        spec = np.fft.rfft(x * self.win[None, :], axis=1)
        pxx = np.abs(spec) ** 2
        pxy = np.conj(spec[self.ref_index])[None, :] * spec

        if self.averaging == "none":
            self.sxx, self.sxy = pxx, pxy
            self.navg = 1
        elif self.averaging == "peakhold":
            if self.navg == 0:
                self.sxx, self.sxy = pxx, pxy
            else:
                grow = pxx > self.sxx
                self.sxx = np.where(grow, pxx, self.sxx)
                self.sxy = np.where(grow, pxy, self.sxy)
            self.navg += 1
        elif self.averaging == "linear":
            n = min(self.navg, self.avg_count - 1)
            self.sxx = (self.sxx * n + pxx) / (n + 1)
            self.sxy = (self.sxy * n + pxy) / (n + 1)
            self.navg += 1
        else:  # exponential
            if self.navg == 0:
                self.sxx, self.sxy = pxx, pxy
            else:
                a = 1.0 / self.avg_count
                self.sxx = a * pxx + (1 - a) * self.sxx
                self.sxy = a * pxy + (1 - a) * self.sxy
            self.navg += 1

    # -- derived spectra ------------------------------------------------
    def _single_sided(self, power: np.ndarray) -> np.ndarray:
        out = power.copy()
        out[..., 1:] *= 2.0
        if self.nfft % 2 == 0:
            out[..., -1] /= 2.0
        return out

    def amplitude_spectrum(self, scaling: str = "rms") -> np.ndarray:
        """Amplitude spectrum in signal units. scaling: 'peak' | 'rms'."""
        ss = self._single_sided(self.sxx)
        amp = np.sqrt(np.maximum(ss, 0.0)) / self.s1 * np.sqrt(2.0)
        # sqrt(2*|X|^2)/S1 = peak amplitude of a tone
        if scaling == "rms":
            amp /= np.sqrt(2.0)
        return amp

    def psd(self) -> np.ndarray:
        """One-sided power spectral density, unit^2/Hz."""
        return self._single_sided(self.sxx) / (self.fs * self.s2)

    def integrate_amplitude(self, amp_g: np.ndarray, order: int, f_min: float = 2.0) -> np.ndarray:
        """Frequency-domain integration of an acceleration amplitude spectrum.

        order=1 -> velocity (mm/s), order=2 -> displacement (um).
        """
        w = 2.0 * np.pi * self.freq
        out = np.zeros_like(amp_g)
        mask = self.freq >= f_min
        a_ms2 = amp_g * G_TO_MS2
        if order == 1:
            out[..., mask] = a_ms2[..., mask] / w[mask] * 1e3      # m/s -> mm/s
        else:
            out[..., mask] = a_ms2[..., mask] / (w[mask] ** 2) * 1e6  # m -> um
        return out

    # -- FRF / coherence / ODS -------------------------------------------
    def frf(self) -> tuple[np.ndarray, np.ndarray]:
        """H1 transmissibility w.r.t. reference channel: (H, coherence)."""
        gxx_ref = np.maximum(self.sxx[self.ref_index], 1e-30)
        h1 = self.sxy / gxx_ref[None, :]
        coh = (np.abs(self.sxy) ** 2) / (
            gxx_ref[None, :] * np.maximum(self.sxx, 1e-30)
        )
        return h1, np.clip(coh, 0.0, 1.0)

    def ods_at(self, f_target: float, bins: int = 1) -> tuple[float, np.ndarray, np.ndarray]:
        """ODS vector at a frequency: (actual_freq, amplitude_rms, phase_rad).

        Amplitude comes from the averaged auto-spectrum, phase from the
        averaged cross-spectrum against the reference channel.
        """
        idx = int(np.argmin(np.abs(self.freq - f_target)))
        lo, hi = max(0, idx - bins + 1), min(self.freq.size, idx + bins)
        amp_all = self.amplitude_spectrum("rms")
        amp = np.max(amp_all[:, lo:hi], axis=1) if hi > lo else amp_all[:, idx]
        phase = np.angle(self.sxy[:, idx])
        return float(self.freq[idx]), amp, phase


# ----------------------------------------------------------------------
def compute_spectrogram(
    y: np.ndarray, fs: float, nperseg: int = 1024, overlap: float = 0.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Magnitude (dB) spectrogram of a single channel."""
    nperseg = min(nperseg, y.size)
    if nperseg < 64:
        return np.array([]), np.array([]), np.zeros((0, 0))
    f, t, sxx = spsig.spectrogram(
        y, fs=fs, window="hann", nperseg=nperseg,
        noverlap=int(nperseg * overlap), scaling="spectrum", mode="magnitude",
    )
    return f, t, 20.0 * np.log10(np.maximum(sxx, 1e-12))


def envelope_spectrum(
    y: np.ndarray, fs: float, f_lo: float = 500.0, f_hi: float = 5000.0
) -> tuple[np.ndarray, np.ndarray]:
    """Bearing-fault envelope spectrum: bandpass -> Hilbert envelope -> FFT."""
    n = y.size
    if n < 256:
        return np.array([]), np.array([])
    nyq = 0.5 * fs
    f_lo = max(1.0, min(f_lo, nyq * 0.9))
    f_hi = max(f_lo + 1.0, min(f_hi, nyq * 0.98))
    sos = spsig.butter(4, [f_lo / nyq, f_hi / nyq], btype="band", output="sos")
    env = np.abs(spsig.hilbert(spsig.sosfiltfilt(sos, y)))
    env = env - env.mean()
    win = spsig.windows.hann(n, sym=False)
    spec = np.abs(np.fft.rfft(env * win)) / np.sum(win) * 2.0
    freq = np.fft.rfftfreq(n, d=1.0 / fs)
    return freq, spec


# -- octave analysis ----------------------------------------------------
def octave_bands(fraction: int, f_min: float, f_max: float) -> np.ndarray:
    """IEC 61260 base-10 center frequencies for 1/n octave bands."""
    centers = []
    k = -40
    while k <= 60:
        fc = 1000.0 * 10 ** (3.0 * k / (10.0 * fraction))
        if f_min <= fc <= f_max:
            centers.append(fc)
        k += 1
    return np.asarray(centers)


def octave_band_levels(
    freq: np.ndarray, psd: np.ndarray, fraction: int = 3, f_min: float = 10.0, f_max: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate a PSD into 1/n-octave band RMS values."""
    if f_max is None:
        f_max = freq[-1] if freq.size else 1000.0
    centers = octave_bands(fraction, f_min, f_max)
    factor = 10 ** (3.0 / (20.0 * fraction))
    rms = np.zeros_like(centers)
    df = freq[1] - freq[0] if freq.size > 1 else 1.0
    for i, fc in enumerate(centers):
        mask = (freq >= fc / factor) & (freq < fc * factor)
        rms[i] = np.sqrt(np.sum(psd[mask]) * df) if np.any(mask) else 0.0
    return centers, rms


# -- time-domain metrics -------------------------------------------------
def channel_metrics(y: np.ndarray) -> dict[str, float]:
    if y.size < 8:
        return {k: 0.0 for k in ("rms", "peak", "pkpk", "crest", "kurtosis", "skew", "mean")}
    mean = float(np.mean(y))
    ac = y - mean
    rms = float(np.sqrt(np.mean(ac ** 2)))
    peak = float(np.max(np.abs(ac)))
    return {
        "rms": rms,
        "peak": peak,
        "pkpk": float(np.ptp(y)),
        "crest": peak / max(rms, 1e-12),
        "kurtosis": float(kurtosis(ac, fisher=False, bias=False)),
        "skew": float(skew(ac, bias=False)),
        "mean": mean,
    }
