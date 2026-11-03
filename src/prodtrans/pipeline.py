"""Nasłuch: źródło audio → resampling → VAD i segmentacja na kanał → kolejka transkrypcji.

Model mowy żyje dłużej niż potok (patrz Controller), więc potok tylko wysyła segmenty
przez `submit`. Przy zmianie ustawień potok jest zatrzymywany i budowany od nowa.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Callable

import numpy as np

from prodtrans.audio import AudioSource, StreamDecimator
from prodtrans.config import Settings
from prodtrans.events import EventHub
from prodtrans.segmenter import ChannelSegmenter, SegmentJob
from prodtrans.vad import SileroVad

log = logging.getLogger(__name__)

LEVEL_INTERVAL_S = 0.1
SILENCE_DB = -90.0
RESET = "reset"


def _db(x: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.mean(np.square(x, dtype=np.float32), axis=0))
    return np.where(rms > 1e-9, 20 * np.log10(np.maximum(rms, 1e-9)), SILENCE_DB)


class Pipeline:
    def __init__(self, settings: Settings, sample_rate: int, source: AudioSource,
                 submit: Callable[[SegmentJob], None], hub: EventHub,
                 save_audio: Callable[[str, np.ndarray], None] | None = None,
                 vad_factory: Callable[[], Callable[[np.ndarray], float]] = SileroVad):
        self.settings = settings
        self.sample_rate = sample_rate
        self.source = source
        self.submit = submit
        self.hub = hub
        self.save_audio = save_audio
        self.channels = settings.enabled_channels
        self._decimators = {c.id: StreamDecimator(sample_rate) for c in self.channels}
        self._segmenters = {
            c.id: ChannelSegmenter(c.id, settings.vad, vad_factory(), self._on_segment)
            for c in self.channels
        }
        self._blocks: queue.Queue[np.ndarray | str | None] = queue.Queue()
        self._audio_failed = False
        self._thread: threading.Thread | None = None
        self._input_peak: np.ndarray | None = None
        self._level_samples = 0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="pipeline", daemon=True)
        self._thread.start()
        self.source.start(self._blocks.put, self._on_status)

    def stop(self) -> None:
        self.source.stop()
        self._blocks.put(None)
        if self._thread:
            self._thread.join(timeout=5)
        # Domknij rozpoczęte wypowiedzi, żeby nie zostały wiszące teksty wstępne.
        for seg in self._segmenters.values():
            try:
                seg.reset()
            except Exception:
                log.exception("Błąd domykania wypowiedzi")

    def _on_status(self, state: str, message: str) -> None:
        log.info("audio %s: %s", state, message)
        if state == "error":
            self._audio_failed = True
        elif self._audio_failed:
            self._audio_failed = False
            self._blocks.put(RESET)  # audio wróciło po przerwie
        self.hub.publish({"type": "status", "source": "audio", "state": state, "message": message})

    def _run(self) -> None:
        while (block := self._blocks.get()) is not None:
            try:
                if isinstance(block, str):  # RESET
                    self._reset()
                    continue
                self._process(block)
            except Exception:
                log.exception("Błąd przetwarzania bloku audio")

    def _reset(self) -> None:
        for c in self.channels:
            self._segmenters[c.id].reset()
            self._decimators[c.id] = StreamDecimator(self.sample_rate)

    def _process(self, block: np.ndarray) -> None:
        n_inputs = block.shape[1]
        for c in self.channels:
            if c.input < n_inputs:
                self._segmenters[c.id].feed(self._decimators[c.id].process(block[:, c.input]))

        # Poziomy wszystkich wejść urządzenia: w ustawieniach widać, gdzie jest sygnał.
        peak = _db(block)
        self._input_peak = peak if self._input_peak is None or len(self._input_peak) != n_inputs \
            else np.maximum(self._input_peak, peak)
        self._level_samples += len(block)
        if self._level_samples >= LEVEL_INTERVAL_S * self.sample_rate:
            inputs = [round(float(v), 1) for v in self._input_peak]
            self.hub.publish({
                "type": "level",
                "inputs": inputs,
                "channels": {str(c.id): inputs[c.input] if c.input < n_inputs else SILENCE_DB
                             for c in self.channels},
            })
            self._input_peak = None
            self._level_samples = 0

    def _on_segment(self, job: SegmentJob) -> None:
        self.submit(job)
        if job.final and self.save_audio:
            self.save_audio(job.segment_id, job.audio)
