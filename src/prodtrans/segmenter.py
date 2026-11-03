"""Cięcie mowy jednego kanału na wypowiedzi (SegmentJob) na podstawie VAD."""

from __future__ import annotations

import math
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np

from prodtrans.config import VadConfig
from prodtrans.vad import CHUNK

SR = 16000
HYSTERESIS = 0.15


@dataclass
class SegmentJob:
    channel: int
    segment_id: str
    audio: np.ndarray
    t_start: float
    t_end: float
    final: bool


class ChannelSegmenter:
    def __init__(self, channel: int, cfg: VadConfig, vad: Callable[[np.ndarray], float],
                 emit: Callable[[SegmentJob], None], clock: Callable[[], float] = time.time):
        self.channel = channel
        self.cfg = cfg
        self.vad = vad
        self.emit = emit
        self.clock = clock

        self._t0: float | None = None
        self._pos = 0  # liczba przetworzonych próbek (oś czasu kanału)
        self._pending = np.zeros(0, np.float32)
        self._preroll: deque[np.ndarray] = deque(maxlen=max(1, math.ceil(cfg.pre_roll_ms / 32)))
        self._silence_blocks_needed = max(1, math.ceil(cfg.min_silence_ms / 32))
        self._max_samples = int(cfg.max_segment_s * SR)
        self._min_samples = int(cfg.min_segment_s * SR)
        self._partial_samples = int(cfg.partial_every_s * SR)

        self._in_speech = False
        self._blocks: list[np.ndarray] = []
        self._seg_len = 0
        self._seg_start_pos = 0
        self._seg_id = ""
        self._silence_run = 0
        self._next_partial = 0
        self._speech_samples = 0
        self._partials_sent = 0

    def reset(self) -> None:
        """Po przerwie w audio: domknij otwartą wypowiedź i licz czas od nowa."""
        if self._in_speech:
            self._finish()
        self._t0 = None
        self._pos = 0
        self._pending = np.zeros(0, np.float32)
        self._preroll.clear()
        reset_vad = getattr(self.vad, "reset", None)
        if reset_vad:
            reset_vad()

    def _time(self, pos: int) -> float:
        return self._t0 + pos / SR

    def feed(self, x16k: np.ndarray) -> None:
        if self._t0 is None:
            self._t0 = self.clock()
        self._pending = np.concatenate([self._pending, np.asarray(x16k, np.float32)])
        n = len(self._pending) // CHUNK
        for i in range(n):
            self._process(self._pending[i * CHUNK : (i + 1) * CHUNK])
        self._pending = self._pending[n * CHUNK :]

    def _process(self, block: np.ndarray) -> None:
        p = self.vad(block)
        if not self._in_speech:
            if p >= self.cfg.threshold:
                self._start_segment(list(self._preroll) + [block])
                self._speech_samples = CHUNK
                self._preroll.clear()
            else:
                self._preroll.append(block)
            self._pos += CHUNK
            return

        self._append(block)
        self._pos += CHUNK
        if p >= self.cfg.threshold:
            self._speech_samples += CHUNK
        if p < self.cfg.threshold - HYSTERESIS:
            self._silence_run += 1
        else:
            self._silence_run = 0

        if self._silence_run >= self._silence_blocks_needed:
            self._finish()
        elif self._seg_len >= self._max_samples:
            # Wymuszone cięcie: następna próbka zaczyna nowy segment, nic nie ginie.
            self._finish()
            self._start_segment([])
        elif self._seg_len >= self._next_partial and self._speech_samples >= self._min_samples:
            self._emit(final=False)
            self._partials_sent += 1
            self._next_partial += self._partial_samples

    def _start_segment(self, blocks: list[np.ndarray]) -> None:
        self._in_speech = True
        self._blocks = []
        self._seg_len = 0
        self._seg_start_pos = self._pos - CHUNK * (len(blocks) - 1) if blocks else self._pos
        self._seg_id = f"{self.channel}-{uuid.uuid4().hex[:8]}"
        self._silence_run = 0
        self._next_partial = self._partial_samples
        self._speech_samples = 0
        self._partials_sent = 0
        for b in blocks:
            self._append(b)

    def _append(self, block: np.ndarray) -> None:
        self._blocks.append(block)
        self._seg_len += len(block)

    def _emit(self, final: bool) -> None:
        audio = np.concatenate(self._blocks) if self._blocks else np.zeros(0, np.float32)
        self.emit(
            SegmentJob(
                channel=self.channel,
                segment_id=self._seg_id,
                audio=audio,
                t_start=self._time(self._seg_start_pos),
                t_end=self._time(self._seg_start_pos + len(audio)),
                final=final,
            )
        )

    def _finish(self) -> None:
        # Segment z samym trzaskiem jest odrzucany. Jeśli wyszedł już tekst wstępny,
        # finał zawsze idzie dalej, żeby zastąpić wstępny wpis w widoku.
        if self._speech_samples >= self._min_samples or self._partials_sent:
            self._emit(final=True)
        self._in_speech = False
        self._blocks = []
        self._seg_len = 0
