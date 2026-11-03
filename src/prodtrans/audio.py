"""Wejścia audio: lista urządzeń, źródło z interfejsu (Dante) i symulacja z plików."""

from __future__ import annotations

import threading
import time
from math import gcd
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import resample_poly

BlockCallback = Callable[[np.ndarray], None]
StatusCallback = Callable[[str, str], None]

TARGET_SR = 16000


def list_devices() -> list[dict]:
    out = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            out.append(
                {
                    "index": i,
                    "name": d["name"],
                    "max_input_channels": d["max_input_channels"],
                    "default_samplerate": d["default_samplerate"],
                }
            )
    return out


def find_device(name: str) -> dict | None:
    """Dopasowuje urządzenie po pełnej nazwie albo jej fragmencie (bez wielkości liter)."""
    devices = list_devices()
    for d in devices:
        if d["name"] == name:
            return d
    for d in devices:
        if name.lower() in d["name"].lower():
            return d
    return None


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return np.asarray(x, dtype=np.float32)
    g = gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g, axis=0).astype(np.float32)


def to_16k(x: np.ndarray, sr: int) -> np.ndarray:
    return resample(x, sr, TARGET_SR)


class AudioSource(Protocol):
    def start(self, on_block: BlockCallback, on_status: StatusCallback) -> None: ...
    def stop(self) -> None: ...


def _reinit_portaudio() -> None:
    """PortAudio zapamiętuje listę urządzeń przy starcie, więc po restarcie DVS trzeba ją odświeżyć."""
    sd._terminate()
    sd._initialize()


def _close_quietly(stream, timeout: float = 1.0) -> None:
    """Zamknięcie strumienia martwego urządzenia potrafi się zawiesić, więc robi to osobny wątek."""
    t = threading.Thread(target=lambda: stream.close(ignore_errors=True), daemon=True)
    t.start()
    t.join(timeout)


class DeviceSource:
    """Wielokanałowe wejście z urządzenia CoreAudio (np. Dante Virtual Soundcard).

    Callback PortAudio tylko kopiuje blok dalej. PortAudio na CoreAudio nie kończy strumienia,
    gdy urządzenie zniknie, więc brak callbacków przez `stall_s` sekund też jest awarią.
    Po awarii zgłasza status "error", odświeża listę urządzeń i próbuje ponownie co `retry_s`.
    """

    def __init__(self, device: str, sample_rate: int, n_channels: int,
                 blocksize: int = 1536, retry_s: float = 2.0, stall_s: float = 1.0):
        self.device = device
        self.sample_rate = sample_rate
        self.n_channels = n_channels
        self.blocksize = blocksize
        self.retry_s = retry_s
        self.stall_s = stall_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, on_block: BlockCallback, on_status: StatusCallback) -> None:
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, args=(on_block, on_status), name="audio-device", daemon=True
        )
        self._thread.start()

    def _run(self, on_block: BlockCallback, on_status: StatusCallback) -> None:
        first = True
        while not self._stop.is_set():
            if not first:
                try:
                    _reinit_portaudio()
                except Exception as e:
                    on_status("error", f"Brak audio: {e}")
            first = False
            self._run_once(on_block, on_status)
            self._stop.wait(self.retry_s)

    def _run_once(self, on_block: BlockCallback, on_status: StatusCallback) -> None:
        last_cb = [time.monotonic()]
        failed = threading.Event()

        def callback(indata, frames, time_info, status):
            last_cb[0] = time.monotonic()
            on_block(indata.copy())

        stream = None
        try:
            dev = find_device(self.device)
            if dev is None:
                raise RuntimeError(f"Nie znaleziono urządzenia '{self.device}'")
            stream = sd.InputStream(
                device=dev["index"],
                channels=self.n_channels,
                samplerate=self.sample_rate,
                blocksize=self.blocksize,
                dtype="float32",
                callback=callback,
                finished_callback=failed.set,
            )
            stream.start()
            last_cb[0] = time.monotonic()
            on_status("ok", f"Audio: {dev['name']}, {self.n_channels} kan., {self.sample_rate} Hz")
            while not self._stop.is_set():
                if failed.is_set():
                    on_status("error", "Strumień audio przerwany, ponawiam…")
                    break
                if time.monotonic() - last_cb[0] > self.stall_s:
                    on_status("error", "Brak sygnału z urządzenia, ponawiam…")
                    break
                self._stop.wait(0.1)
        except Exception as e:  # PortAudioError, RuntimeError
            on_status("error", f"Brak audio: {e}")
        finally:
            if stream is not None:
                _close_quietly(stream)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)


class FileSource:
    """Symulacja wejścia: plik i trafia do kanału i (do testów i demo bez Dante)."""

    def __init__(self, paths: list[Path], sample_rate: int = 48000, realtime: bool = True,
                 blocksize: int = 1536, on_end: Callable[[], None] | None = None,
                 n_channels: int | None = None):
        self.paths = [Path(p) for p in paths]
        self.sample_rate = sample_rate
        self.realtime = realtime
        self.blocksize = blocksize
        self.on_end = on_end
        self.n_channels = max(n_channels or 0, len(self.paths))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _load(self) -> np.ndarray:
        tracks = []
        for p in self.paths:
            data, sr = sf.read(p, dtype="float32", always_2d=True)
            tracks.append(resample(data.mean(axis=1), sr, self.sample_rate))
        n = max(len(t) for t in tracks)
        out = np.zeros((n, self.n_channels), np.float32)
        for i, t in enumerate(tracks):
            out[: len(t), i] = t
        return out

    def start(self, on_block: BlockCallback, on_status: StatusCallback) -> None:
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, args=(on_block, on_status), name="audio-file", daemon=True
        )
        self._thread.start()

    def _run(self, on_block: BlockCallback, on_status: StatusCallback) -> None:
        audio = self._load()
        on_status("ok", f"Symulacja z {len(self.paths)} plików")
        # Dopełnienie do pełnego bloku plus sekunda ciszy, żeby VAD domknął ostatnią wypowiedź.
        tail = (-(len(audio) + self.sample_rate)) % self.blocksize + self.sample_rate
        audio = np.vstack([audio, np.zeros((tail, self.n_channels), np.float32)])
        block_s = self.blocksize / self.sample_rate
        t_next = time.monotonic()
        for i in range(0, len(audio), self.blocksize):
            if self._stop.is_set():
                return
            on_block(audio[i : i + self.blocksize])
            if self.realtime:
                t_next += block_s
                self._stop.wait(max(0.0, t_next - time.monotonic()))
        if self.on_end:
            self.on_end()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)


class StreamDecimator:
    """Ciągła zamiana częstotliwości na 16 kHz blok po bloku, z zachowaniem stanu filtra.

    Dla wielokrotności 16 kHz (48 kHz z Dante) to filtr dolnoprzepustowy FIR i co n-ta
    próbka. Dla innych częstotliwości przelicza każdy blok osobno (gorsza jakość na
    granicach bloków).
    """

    def __init__(self, sr_in: int):
        from scipy.signal import firwin

        self.sr_in = sr_in
        self.factor = sr_in // TARGET_SR if sr_in % TARGET_SR == 0 else 0
        if self.factor > 1:
            self._taps = firwin(96 * self.factor + 1, 7000, fs=sr_in).astype(np.float32)
            self._zi = np.zeros(len(self._taps) - 1, np.float32)
            self._phase = 0

    def process(self, x: np.ndarray) -> np.ndarray:
        if self.factor == 1:
            return np.asarray(x, np.float32)
        if self.factor == 0:
            return to_16k(x, self.sr_in)
        from scipy.signal import lfilter

        y, self._zi = lfilter(self._taps, 1.0, np.asarray(x, np.float32), zi=self._zi)
        out = y[self._phase :: self.factor]
        self._phase = (self._phase - len(x)) % self.factor
        return out.astype(np.float32)
