"""Acquisition engine: NI-DAQmx backend, ring buffer and streaming thread."""

from __future__ import annotations

import re
import socket
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from ..config import AcquisitionConfig, ChannelConfig
from .simulator import SimBackend

# C-series dynamic-signal modules with IEPE excitation (accelerometer inputs).
_IEPE_MODULE_PAT = re.compile(r"\b9(230|231|232|234|250|260)\b")
_PHYS_PAT = re.compile(r"^(?P<chassis>.+?)Mod(?P<mod>\d+)/ai(?P<ai>\d+)$")


def split_physical(name: str) -> tuple[str, int, int]:
    """'cDAQ9189-2196A44Mod3/ai1' -> ('cDAQ9189-2196A44', 3, 1)."""
    m = _PHYS_PAT.match(name)
    if not m:
        return name, 0, 0
    return m.group("chassis"), int(m.group("mod")), int(m.group("ai"))


def discover_network_chassis(timeout: float = 12.0) -> list[str]:
    """Hostnames of NI chassis reachable on wired links (best effort, Windows).

    A chassis that was never added in NI MAX is invisible to DAQmx, so a
    second chassis on the same cable would silently be ignored. NI devices
    are found by their MAC prefix (00-80-2F) in the IPv6 neighbor table and
    named through mDNS reverse lookup.
    """
    if sys.platform != "win32":
        return []
    script = (
        "Get-NetAdapter -Physical | Where-Object { $_.Status -eq 'Up' -and "
        "$_.MediaType -eq '802.3' } | ForEach-Object { $i = $_.ifIndex; "
        "ping -6 -n 1 -w 400 \"ff02::1%$i\" | Out-Null; "
        "Get-NetNeighbor -InterfaceIndex $i -AddressFamily IPv6 -ErrorAction SilentlyContinue | "
        "Where-Object { $_.LinkLayerAddress -match '^00-80-2F' } | ForEach-Object { "
        "try { (Resolve-DnsName -Name $_.IPAddress -Type PTR -ErrorAction Stop | "
        "Select-Object -First 1).NameHost } catch {} } }"
    )
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, text=True, timeout=timeout,
            stdin=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).stdout
    except Exception:
        return []
    hosts = []
    for line in out.splitlines():
        host = line.strip()
        if host and host not in hosts:
            hosts.append(host)
    return hosts


def _network_address(dev) -> tuple[str, str] | None:
    """(ip, hostname) of a network device, or None for local (USB/PCI) ones."""
    try:
        ip = str(dev.tcpip_ethernet_ip or "")
    except Exception:
        return None
    try:
        host = str(dev.tcpip_hostname or "")
    except Exception:
        host = ""
    return (ip, host) if (ip or host) else None


def chassis_reachable(dev, timeout: float = 0.8) -> bool:
    """True if a network chassis answers right now.

    DAQmx keeps a disconnected chassis registered together with the last
    module list it saw, so "registered with modules" says nothing about
    being on the wire. A task that includes such a chassis fails at start
    with -201401. Probe the chassis web/service ports by cached IP, then by
    mDNS hostname (the link-local IP can change between sessions).
    """
    addr = _network_address(dev)
    if addr is None:
        return True
    ip, host = addr

    def answers(target: str) -> bool:
        for port in (80, 31415):
            try:
                with socket.create_connection((target, port), timeout=timeout):
                    return True
            except OSError:
                continue
        return False

    if ip and answers(ip):
        return True
    if host:
        # mDNS lookup of a host that is gone blocks for seconds and takes no
        # timeout, so bound it with a helper thread.
        found: list[str] = []

        def resolve() -> None:
            try:
                found.extend(a[4][0] for a in socket.getaddrinfo(
                    f"{host}.local", 80, socket.AF_INET, socket.SOCK_STREAM))
            except OSError:
                pass

        t = threading.Thread(target=resolve, daemon=True)
        t.start()
        t.join(1.5)
        for addr_ip in dict.fromkeys(found):
            if addr_ip != ip and answers(addr_ip):
                return True
    return False


def unreachable_chassis(physical_channels: list[str]) -> list[str]:
    """Chassis referenced by these channels that do not answer right now."""
    try:
        from nidaqmx.system import Device
    except Exception:
        return []
    names = []
    for ch in physical_channels:
        chassis, mod, _ = split_physical(ch)
        if mod and chassis not in names:
            names.append(chassis)
    dead = []
    for name in names:
        try:
            if not chassis_reachable(Device(name)):
                dead.append(name)
        except Exception:
            dead.append(name)
    return dead


@dataclass
class ConnectResult:
    """Outcome of an auto-connect scan over the NI-DAQmx system."""

    ok: bool = False
    chassis: list[str] = field(default_factory=list)      # "name (product)"
    modules: list[tuple[str, str]] = field(default_factory=list)  # (name, product)
    channels: list[str] = field(default_factory=list)     # preferred AI channels
    all_channels: list[str] = field(default_factory=list)
    reserved: list[str] = field(default_factory=list)     # chassis reserved by us
    added: list[str] = field(default_factory=list)        # chassis newly registered
    offline: list[str] = field(default_factory=list)      # registered but not answering
    messages: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """One line per chassis: 'cDAQ9189-233EBC6 — 8× NI 9230 (BNC) — 24 AI ch'."""
        if not self.ok:
            if self.offline:
                return ("No reachable chassis — offline: " + ", ".join(self.offline)
                        + "\nCheck power and the Ethernet cable.")
            return self.messages[-1] if self.messages else "No DAQ device found"
        lines = []
        for entry in self.chassis:
            name = entry.split(" (")[0]
            kinds = Counter(p for n, p in self.modules if n.startswith(name + "Mod"))
            mods = ", ".join(f"{c}× {p}" for p, c in kinds.items()) or "no modules"
            n_ch = sum(1 for c in self.channels if c.startswith(name + "Mod"))
            tag = " (new)" if name in self.added else ""
            lines.append(f"{name}{tag} — {mods} — {n_ch} AI ch")
        for name in self.offline:
            lines.append(f"{name} — offline (not reachable), skipped")
        if len(self.chassis) > 1:
            lines.append(f"{len(self.chassis)} chassis · {len(self.channels)} AI ch total")
        return "\n".join(lines) if lines else f"{len(self.channels)} AI ch"


def auto_connect(reserve: bool = True, prefer_iepe: bool = True,
                 discover: bool = True) -> ConnectResult:
    """Discover NI-DAQmx devices and make them usable without NI MAX.

    Network cDAQ chassis (e.g. cDAQ-9189) lose their host reservation after
    a reboot or an ownership change; their modules then enumerate empty even
    though the chassis answers on the network. This scans the system, reserves
    any module-less network chassis, and returns the usable AI channels,
    preferring IEPE-capable dynamic modules (NI-923x) for accelerometers.
    """
    res = ConnectResult()
    try:
        import nidaqmx
    except Exception as exc:
        res.messages.append(f"nidaqmx unavailable: {exc}")
        return res

    try:
        devices = list(nidaqmx.system.System.local().devices)
    except Exception as exc:
        res.messages.append(f"DAQmx system scan failed: {exc}")
        return res

    # Register chassis that are on the wire but unknown to DAQmx (e.g. a
    # second chassis added to extend the rig).
    if discover:
        known = {str(getattr(d, "name", "")).lower() for d in devices}
        for host in discover_network_chassis():
            short = host.split(".")[0]
            if short.lower() in known:
                continue
            try:
                from nidaqmx.system import Device

                Device.add_network_device(host, device_name=short,
                                          attempt_reservation=True, timeout=30.0)
                res.added.append(short)
                res.messages.append(f"Registered new network chassis {short}")
            except Exception as exc:
                res.messages.append(f"Could not add {short}: {str(exc).splitlines()[0]}")
        if res.added:
            devices = list(nidaqmx.system.System.local().devices)

    if not devices:
        res.messages.append("No NI devices configured on this system")
        return res

    for dev in devices:
        name = str(getattr(dev, "name", "?"))
        product = str(getattr(dev, "product_type", "?"))
        try:
            modules = [(m.name, str(m.product_type)) for m in dev.chassis_module_devices]
        except Exception:
            modules = []
        is_chassis = "Mod" not in name and (modules or "cDAQ" in product or "cRIO" in product)
        if not is_chassis:
            continue
        if not chassis_reachable(dev):
            res.offline.append(name)
            res.messages.append(f"{name} is not reachable — skipped (cached modules ignored)")
            continue
        res.chassis.append(f"{name} ({product})")

        if not modules and reserve:
            try:
                dev.reserve_network_device(override_reservation=True)
                res.reserved.append(name)
                res.messages.append(f"Reserved network chassis {name}")
                # Module enumeration trickles in after a reservation (seen:
                # 6 of 8 modules at first) — wait until the count is stable.
                last = -1
                for _ in range(8):
                    modules = [(m.name, str(m.product_type))
                               for m in dev.chassis_module_devices]
                    if modules and len(modules) == last:
                        break
                    last = len(modules)
                    time.sleep(0.5)
            except Exception as exc:
                res.messages.append(f"Reserve failed for {name}: {exc}")
        res.modules.extend(modules)

    # Enumerate AI channels (chassis reservation above may have added modules).
    iepe_channels: list[str] = []
    for dev in nidaqmx.system.System.local().devices:
        dev_name = str(getattr(dev, "name", ""))
        if any(dev_name.startswith(off + "Mod") or dev_name == off for off in res.offline):
            continue
        try:
            chans = [ch.name for ch in dev.ai_physical_chans]
        except Exception:
            continue
        res.all_channels.extend(chans)
        if _IEPE_MODULE_PAT.search(str(getattr(dev, "product_type", ""))):
            iepe_channels.extend(chans)

    # Chassis in registration order, then module and input numerically
    # (a plain string sort would put Mod10 before Mod2 and reorder chassis).
    chassis_order = {entry.split(" (")[0]: i for i, entry in enumerate(res.chassis)}

    def natural(name: str) -> tuple[int, str, int, int]:
        chassis, mod, ai = split_physical(name)
        return chassis_order.get(chassis, len(chassis_order)), chassis, mod, ai

    res.all_channels = sorted(set(res.all_channels), key=natural)
    res.channels = (sorted(set(iepe_channels), key=natural)
                    if (prefer_iepe and iepe_channels) else res.all_channels)
    res.ok = len(res.channels) > 0
    if not res.ok and not res.messages:
        res.messages.append("Devices found but no AI channels available")
    return res


# ----------------------------------------------------------------------
def discover_ai_channels() -> list[str]:
    """List all NI-DAQmx AI physical channels visible on this system."""
    try:
        import nidaqmx
    except Exception:
        return []
    channels: list[str] = []
    try:
        for dev in nidaqmx.system.System.local().devices:
            try:
                channels.extend(ch.name for ch in dev.ai_physical_chans)
            except Exception:
                continue
    except Exception:
        return []
    return sorted(set(channels))


def hardware_available() -> bool:
    return len(discover_ai_channels()) > 0


# ----------------------------------------------------------------------
class NidaqBackend:
    """Continuous multi-channel IEPE accelerometer (or voltage) acquisition."""

    def __init__(self, channels: list[ChannelConfig], acq: AcquisitionConfig) -> None:
        self.channels = channels
        self.acq = acq
        self._task = None
        self._reader = None
        self._buf: np.ndarray | None = None

    def start(self) -> None:
        import nidaqmx
        from nidaqmx.constants import AcquisitionType, ExcitationSource, AccelUnits
        from nidaqmx.stream_readers import AnalogMultiChannelReader

        task = nidaqmx.Task()
        try:
            for ch in self.channels:
                if self.acq.coupling == "iepe":
                    task.ai_channels.add_ai_accel_chan(
                        physical_channel=ch.physical_channel,
                        min_val=self.acq.min_val,
                        max_val=self.acq.max_val,
                        units=AccelUnits.G,
                        sensitivity=ch.sensitivity_mv_g,
                        current_excit_source=ExcitationSource.INTERNAL,
                        current_excit_val=0.002,
                    )
                else:
                    task.ai_channels.add_ai_voltage_chan(
                        ch.physical_channel,
                        min_val=self.acq.min_val,
                        max_val=self.acq.max_val,
                    )
            # Channels from several cDAQ-9189 chassis may share this one task:
            # DAQmx synchronizes them over the network (802.1AS), so all
            # channels stay sample-aligned. Keep ~4 s of driver buffer so a
            # busy GUI cannot overflow a 48-channel stream.
            task.timing.cfg_samp_clk_timing(
                rate=self.acq.sample_rate,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=max(8 * self.acq.chunk_size,
                                   int(4 * self.acq.sample_rate)),
            )
            self._reader = AnalogMultiChannelReader(task.in_stream)
            self._buf = np.zeros((len(self.channels), self.acq.chunk_size), dtype=np.float64)
            task.start()
        except Exception:
            task.close()
            raise
        self._task = task

    def read(self) -> np.ndarray:
        assert self._reader is not None and self._buf is not None
        # A multi-chassis task starts on a synchronized future time, so the
        # first block can take several seconds to arrive.
        self._reader.read_many_sample(
            self._buf, number_of_samples_per_channel=self.acq.chunk_size, timeout=15.0
        )
        data = self._buf.copy()
        if self.acq.coupling == "voltage":
            # Convert volts to g using per-channel sensitivity.
            sens = np.array([1000.0 / max(1e-9, c.sensitivity_mv_g) for c in self.channels])
            data *= sens[:, None]
        return data

    def backlog(self) -> int:
        """Samples per channel waiting in the driver buffer (0 = keeping up)."""
        try:
            return int(self._task.in_stream.avail_samp_per_chan) if self._task else 0
        except Exception:
            return 0

    def stop(self) -> None:
        if self._task is not None:
            try:
                self._task.stop()
            finally:
                self._task.close()
                self._task = None


# ----------------------------------------------------------------------
class RingBuffer:
    """Thread-safe multi-channel sample ring buffer."""

    def __init__(self, n_channels: int, capacity: int) -> None:
        self.n_channels = n_channels
        self.capacity = int(capacity)
        self._data = np.zeros((n_channels, self.capacity), dtype=np.float64)
        self._write = 0            # next write position
        self.total = 0             # total samples ever written
        self._lock = threading.Lock()

    def append(self, block: np.ndarray) -> None:
        n = block.shape[1]
        with self._lock:
            if n >= self.capacity:
                self._data[:] = block[:, -self.capacity:]
                self._write = 0
                self.total += n
                return
            end = self._write + n
            if end <= self.capacity:
                self._data[:, self._write:end] = block
            else:
                k = self.capacity - self._write
                self._data[:, self._write:] = block[:, :k]
                self._data[:, : end - self.capacity] = block[:, k:]
            self._write = end % self.capacity
            self.total += n

    def latest(self, n: int) -> np.ndarray:
        """Return the most recent n samples (may be fewer early on)."""
        with self._lock:
            n = min(n, self.total, self.capacity)
            if n <= 0:
                return np.zeros((self.n_channels, 0))
            start = (self._write - n) % self.capacity
            if start + n <= self.capacity:
                return self._data[:, start : start + n].copy()
            k = self.capacity - start
            return np.concatenate((self._data[:, start:], self._data[:, : n - k]), axis=1)

    def read_range(self, start_abs: int, end_abs: int | None = None
                   ) -> tuple[np.ndarray, int, int]:
        """Samples [start_abs, end_abs) by absolute index, read atomically.

        Returns (data, start, end) clipped to what the ring still holds;
        ``end_abs=None`` means "up to the newest sample".
        """
        with self._lock:
            end = self.total if end_abs is None else min(int(end_abs), self.total)
            oldest = self.total - min(self.total, self.capacity)
            start = max(int(start_abs), oldest)
            n = end - start
            if n <= 0:
                return np.zeros((self.n_channels, 0)), start, start
            end_pos = (self._write - (self.total - end)) % self.capacity
            start_pos = (end_pos - n) % self.capacity
            if start_pos + n <= self.capacity:
                data = self._data[:, start_pos:start_pos + n].copy()
            else:
                k = self.capacity - start_pos
                data = np.concatenate((self._data[:, start_pos:], self._data[:, :n - k]), axis=1)
            return data, start, end

    def read_since(self, total_mark: int, max_samples: int) -> tuple[np.ndarray, int]:
        """Return samples written after ``total_mark`` (up to max), and new mark."""
        with self._lock:
            available = self.total - total_mark
            if available <= 0:
                return np.zeros((self.n_channels, 0)), total_mark
            n = int(min(available, max_samples, self.capacity))
        block = self.latest(n)
        return block, self.total


# ----------------------------------------------------------------------
class AcquisitionEngine:
    """Background streaming thread feeding a ring buffer and listeners."""

    def __init__(
        self,
        channels: list[ChannelConfig],
        acq: AcquisitionConfig,
        buffer_seconds: float = 30.0,
        backend=None,
    ) -> None:
        self.channels = channels
        self.acq = acq
        self.fs = acq.sample_rate
        capacity = int(acq.sample_rate * buffer_seconds)
        self.ring = RingBuffer(len(channels), capacity)
        self.simulated = acq.simulate
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._listeners: list = []   # callables(block) e.g. recorder
        self._listener_lock = threading.Lock()

        # Stream health counters (written by the acquisition thread).
        self.blocks = 0
        self.t_first = 0.0
        self.t_last = 0.0
        self.backlog = 0
        self.max_backlog = 0
        self.max_gap = 0.0           # longest wait between two blocks (s)
        self._rate_marks: list[tuple[float, int]] = []

        if backend is not None:
            self.backend = backend   # e.g. PlaybackBackend
        elif acq.simulate:
            self.backend = SimBackend(channels, acq.sample_rate, acq.chunk_size)
        else:
            self.backend = NidaqBackend(channels, acq)

    # -- listeners (recorder tee) --------------------------------------
    def add_listener(self, fn) -> None:
        with self._listener_lock:
            self._listeners.append(fn)

    def remove_listener(self, fn) -> None:
        with self._listener_lock:
            if fn in self._listeners:
                self._listeners.remove(fn)

    # -- lifecycle ------------------------------------------------------
    def start(self) -> None:
        self.backend.start()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="daq-stream")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        try:
            self.backend.stop()
        except Exception:
            pass

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- stream health ----------------------------------------------------
    def measured_rate(self, window: float = 5.0) -> float:
        """Samples/s actually delivered over the last ``window`` seconds."""
        marks = list(self._rate_marks)   # snapshot: the DAQ thread mutates it
        if len(marks) < 2:
            return 0.0
        t_end, n_end = marks[-1]
        t_ref, n_ref = marks[0]
        for t, n in marks:
            if t_end - t <= window:
                t_ref, n_ref = t, n
                break
        dt = t_end - t_ref
        return (n_end - n_ref) / dt if dt > 0 else 0.0

    def iepe_settling(self, settle_s: float = 22.0) -> bool:
        """True while IEPE bias is still charging after a hardware start."""
        if not isinstance(self.backend, NidaqBackend) or self.acq.coupling != "iepe":
            return False
        return self.t_first == 0.0 or time.perf_counter() - self.t_first < settle_s

    def block_age(self) -> float:
        """Seconds since the last block arrived (grows if the stream stalls)."""
        return time.perf_counter() - self.t_last if self.t_last else 0.0

    def _note_block(self, n_samples: int) -> None:
        now = time.perf_counter()
        if self.blocks == 0:
            self.t_first = now
        else:
            self.max_gap = max(self.max_gap, now - self.t_last)
        self.t_last = now
        self.blocks += 1
        backlog = getattr(self.backend, "backlog", None)
        self.backlog = backlog() if callable(backlog) else 0
        self.max_backlog = max(self.max_backlog, self.backlog)
        marks = self._rate_marks
        marks.append((now, self.ring.total))
        while len(marks) > 2 and now - marks[0][0] > 12.0:
            marks.pop(0)

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                block = self.backend.read()
                self.ring.append(block)
                self._note_block(block.shape[1])
                with self._listener_lock:
                    listeners = list(self._listeners)
                for fn in listeners:
                    try:
                        fn(block)
                    except Exception:
                        pass
        except Exception as exc:
            if not self._stop.is_set():
                self.error = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
