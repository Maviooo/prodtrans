"""Zdarzenia dla przeglądarki i rozsyłanie ich z wątków roboczych do WebSocketów."""

from __future__ import annotations

import asyncio
import threading
from collections import OrderedDict


def transcript_event(channel: int, segment_id: str, text: str, final: bool,
                     t_start: float, t_end: float) -> dict:
    return {
        "type": "transcript",
        "channel": channel,
        "segment_id": segment_id,
        "text": text,
        "final": final,
        "t_start": t_start,
        "t_end": t_end,
    }


RESYNC = {"type": "resync"}


def _deliver(q: asyncio.Queue, event: dict) -> None:
    """Wolny klient: poziomy można zgubić, resztę zastępuje prośba o pełne odświeżenie."""
    if q.full():
        if event["type"] == "level":
            return
        while not q.empty():
            q.get_nowait()
        q.put_nowait(dict(RESYNC))
        return
    q.put_nowait(event)


class EventHub:
    """Bezpieczny wątkowo: `publish` można wołać z dowolnego wątku."""

    def __init__(self, history_size: int = 200):
        self.history_size = history_size
        self._lock = threading.Lock()
        self._history: OrderedDict[str, dict] = OrderedDict()
        self._subs: list[tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = []
        self._status: dict[str, dict] = {}  # ostatni status per źródło ("audio", "engine")

    def publish(self, event: dict) -> None:
        with self._lock:
            if event["type"] == "transcript":
                seg = event["segment_id"]
                old = self._history.get(seg)
                if old is not None and old["final"] and not event["final"]:
                    return  # spóźniony tekst wstępny po finale
                if event["final"] and not event["text"]:
                    self._history.pop(seg, None)  # pusty finał: wypowiedzi nie było
                else:
                    self._history[seg] = event  # nadpisuje w miejscu, kolejność zostaje
                while len(self._history) > self.history_size:
                    self._history.popitem(last=False)
            elif event["type"] == "status":
                self._status[event.get("source", "audio")] = event
            subs = list(self._subs)
        for loop, q in subs:
            try:
                loop.call_soon_threadsafe(_deliver, q, event)
            except RuntimeError:  # pętla już zamknięta
                pass

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=1000)
        with self._lock:
            self._subs.append((asyncio.get_running_loop(), q))
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        with self._lock:
            self._subs = [(l, s) for l, s in self._subs if s is not q]

    def statuses(self) -> list[dict]:
        with self._lock:
            return list(self._status.values())

    def clear_status(self, source: str) -> None:
        with self._lock:
            self._status.pop(source, None)

    def resync(self) -> None:
        """Wszyscy klienci dostaną świeże hello (np. po zmianie nazw kanałów)."""
        self.publish(dict(RESYNC))

    def history(self) -> list[dict]:
        with self._lock:
            return list(self._history.values())
