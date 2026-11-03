"""Wątek transkrypcji: jeden model, finały przed tekstem wstępnym."""

from __future__ import annotations

import heapq
import itertools
import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable

from prodtrans.engines.base import Engine
from prodtrans.segmenter import SegmentJob

log = logging.getLogger(__name__)

FINAL, PARTIAL = 0, 1


@dataclass
class TranscriptResult:
    job: SegmentJob
    text: str
    engine_ms: float


class TranscriptionWorker:
    def __init__(self, engine: Engine, on_result: Callable[[TranscriptResult], None]):
        self.engine = engine
        self.on_result = on_result
        self._heap: list[list] = []  # [priorytet, kolejność, job, aktywny]
        self._partials: dict[str, list] = {}  # segment_id -> wpis czekającego wstępnego
        self._seq = itertools.count()
        self._cv = threading.Condition()
        self._stop = False
        self._thread: threading.Thread | None = None
        self.ready = threading.Event()
        self.error: Exception | None = None

    def submit(self, job: SegmentJob) -> None:
        with self._cv:
            stale = self._partials.pop(job.segment_id, None)
            if stale:
                stale[3] = False
                stale[2] = None  # zwolnij audio od razu  # nowszy wstępny albo finał zastępuje czekający wstępny
            entry = [FINAL if job.final else PARTIAL, next(self._seq), job, True]
            if not job.final:
                self._partials[job.segment_id] = entry
            heapq.heappush(self._heap, entry)
            self._cv.notify()

    def pending(self) -> int:
        with self._cv:
            return sum(1 for e in self._heap if e[3])

    def start(self) -> None:
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="transcriber", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify()
        if self._thread:
            self._thread.join(timeout=10)

    def _next(self) -> SegmentJob | None:
        with self._cv:
            while True:
                while self._heap and not self._heap[0][3]:
                    heapq.heappop(self._heap)
                if self._stop:
                    return None
                if self._heap:
                    entry = heapq.heappop(self._heap)
                    job = entry[2]
                    if not job.final and self._partials.get(job.segment_id) is entry:
                        del self._partials[job.segment_id]
                    return job
                self._cv.wait()

    def _run(self) -> None:
        load = getattr(self.engine, "load", None)
        if load:
            try:
                load()  # w tym wątku, bo MLX wiąże obliczenia z wątkiem
            except Exception as e:
                log.exception("Nie udało się wczytać silnika %s", self.engine.name)
                self.error = e
                self.ready.set()
                return
        self.ready.set()
        while (job := self._next()) is not None:
            t0 = time.perf_counter()
            try:
                text = self.engine.transcribe(job.audio).strip()
            except Exception:
                log.exception("Błąd transkrypcji (kanał %s)", job.channel)
                text = ""
            ms = (time.perf_counter() - t0) * 1000
            # Pusty wstępny pomijamy. Pusty finał idzie dalej, żeby usunąć wiszący tekst wstępny.
            if not text and not job.final:
                continue
            try:
                self.on_result(TranscriptResult(job, text, ms))
            except Exception:
                log.exception("Błąd obsługi wyniku (kanał %s)", job.channel)
