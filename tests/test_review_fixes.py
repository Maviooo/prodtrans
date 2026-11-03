"""Testy do poprawek z końcowej recenzji (każdy najpierw nie przechodził)."""

import asyncio
import threading
import time
from pathlib import Path

import numpy as np

from prodtrans import audio as audio_mod
from prodtrans.config import VadConfig, default_settings, settings_from_dict
from prodtrans.events import EventHub, transcript_event
from prodtrans.segmenter import ChannelSegmenter, SegmentJob
from prodtrans.store import SessionStore
from prodtrans.worker import TranscriptionWorker

ROOT = Path(__file__).resolve().parent.parent
SR = 16000


def fake_vad(chunk):
    return 1.0 if np.abs(chunk).max() > 0.1 else 0.0


def tone(seconds):
    t = np.arange(int(seconds * SR)) / SR
    return (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


# --- Krytyczne 1 i 2: odświeżanie listy urządzeń i watchdog zamrożonego strumienia ---

class FakeStream:
    instances = []

    def __init__(self, callbacks_before_stall, **kw):
        self.kw = kw
        self.n = callbacks_before_stall
        self.closed = False
        FakeStream.instances.append(self)

    def start(self):
        cb = self.kw["callback"]

        def run():
            for _ in range(self.n):
                cb(np.zeros((1536, 2), np.float32), 1536, None, None)
                time.sleep(0.03)
            # potem cisza: urządzenie zniknęło, PortAudio nie kończy strumienia
        threading.Thread(target=run, daemon=True).start()

    def close(self, ignore_errors=True):
        self.closed = True


def test_device_watchdog_detects_stalled_stream_and_reopens(monkeypatch):
    FakeStream.instances = []
    reinit = []
    monkeypatch.setattr(audio_mod, "find_device", lambda name: {"index": 1, "name": "Dante", "max_input_channels": 2})
    monkeypatch.setattr(audio_mod.sd, "InputStream", lambda **kw: FakeStream(5, **kw))
    monkeypatch.setattr(audio_mod, "_reinit_portaudio", lambda: reinit.append(1))
    statuses = []
    src = audio_mod.DeviceSource("Dante", 48000, 2, retry_s=0.1, stall_s=0.3)
    src.start(lambda b: None, lambda s, m: statuses.append(s))
    deadline = time.time() + 3
    while len(FakeStream.instances) < 2 and time.time() < deadline:
        time.sleep(0.02)
    src.stop()
    assert "error" in statuses
    assert len(FakeStream.instances) >= 2  # ponowne otwarcie
    assert FakeStream.instances[0].closed
    assert reinit  # lista urządzeń odświeżona przed ponowną próbą


def test_device_appearing_later_is_found(monkeypatch):
    calls = {"n": 0}

    def find(name):
        calls["n"] += 1
        return None if calls["n"] < 3 else {"index": 1, "name": "Dante", "max_input_channels": 2}

    FakeStream.instances = []
    reinit = []
    monkeypatch.setattr(audio_mod, "find_device", find)
    monkeypatch.setattr(audio_mod.sd, "InputStream", lambda **kw: FakeStream(1000, **kw))
    monkeypatch.setattr(audio_mod, "_reinit_portaudio", lambda: reinit.append(1))
    statuses = []
    src = audio_mod.DeviceSource("Dante", 48000, 2, retry_s=0.05)
    src.start(lambda b: None, lambda s, m: statuses.append(s))
    deadline = time.time() + 2
    while "ok" not in statuses and time.time() < deadline:
        time.sleep(0.02)
    src.stop()
    assert statuses[:2] == ["error", "error"] and "ok" in statuses
    assert len(reinit) >= 2


# --- Ważne 3: pusty finał usuwa wiszący tekst wstępny ---

class Engine:
    name = "e"

    def __init__(self, texts):
        self.texts = list(texts)

    def transcribe(self, a):
        t = self.texts.pop(0)
        if isinstance(t, Exception):
            raise t
        return t


def _job(final, seg="0-a"):
    return SegmentJob(0, seg, np.zeros(100, np.float32), 0.0, 1.0, final)


def _drain(worker, results, n):
    worker.start()
    deadline = time.time() + 3
    while len(results) < n and time.time() < deadline:
        time.sleep(0.01)
    worker.stop()


def test_empty_or_failed_final_is_still_reported():
    results = []
    w = TranscriptionWorker(Engine(["", RuntimeError("boom")]), results.append)
    w.submit(_job(True, "0-a"))
    w.submit(_job(True, "0-b"))
    _drain(w, results, 2)
    assert [(r.job.segment_id, r.text) for r in results] == [("0-a", ""), ("0-b", "")]


def test_empty_partial_still_skipped():
    results = []
    w = TranscriptionWorker(Engine([" "]), results.append)
    w.submit(_job(False))
    _drain(w, results, 1)
    assert results == []


def test_hub_empty_final_removes_entry(tmp_path):
    hub = EventHub()
    hub.publish(transcript_event(0, "0-a", "Dziękuję.", False, 1, 2))
    hub.publish(transcript_event(0, "0-a", "", True, 1, 2))
    assert hub.history() == []
    store = SessionStore(tmp_path)
    store.append(transcript_event(0, "0-a", "", True, 1, 2), "A")
    assert store.txt_path.read_text() == ""


# --- Ważne 4: wyjątek w on_result nie zabija wątku ---

def test_worker_survives_on_result_exception():
    results = []

    def on_result(r):
        results.append(r)
        if len(results) == 1:
            raise OSError("dysk pełny")

    w = TranscriptionWorker(Engine(["a", "b"]), on_result)
    w.submit(_job(True, "0-a"))
    w.submit(_job(True, "0-b"))
    _drain(w, results, 2)
    assert len(results) == 2


# --- Ważne 5: krótki trzask nie przechodzi przez tekst wstępny ---

def test_short_blip_with_preroll_emits_nothing():
    jobs = []
    seg = ChannelSegmenter(0, VadConfig(), fake_vad, jobs.append, clock=lambda: 0.0)
    for part in (np.zeros(SR, np.float32), tone(0.2), np.zeros(SR, np.float32)):
        seg.feed(part)
    assert jobs == []


# --- Ważne 6: po przerwie w audio czas jest liczony od nowa ---

def test_segmenter_reset_reanchors_time_and_closes_segment():
    now = {"t": 1000.0}
    jobs = []
    seg = ChannelSegmenter(0, VadConfig(), fake_vad, jobs.append, clock=lambda: now["t"])
    seg.feed(tone(2.0))  # mowa w toku
    now["t"] = 2000.0  # przerwa w audio
    seg.reset()
    assert jobs and jobs[-1].final  # otwarta wypowiedź domknięta
    seg.feed(np.zeros(SR // 2, np.float32))
    seg.feed(tone(1.0))
    seg.feed(np.zeros(SR, np.float32))
    assert jobs[-1].t_start >= 2000.0


def test_pipeline_resets_segmenters_after_audio_error(monkeypatch):
    from prodtrans import pipeline as pl

    class Src:
        def start(self, on_block, on_status):
            self.on_block, self.on_status = on_block, on_status

        def stop(self):
            pass

    cfg = default_settings([{"name": "D", "max_input_channels": 8, "default_samplerate": 48000.0}])
    src = Src()
    p = pl.Pipeline(cfg, 48000, src, lambda job: None, EventHub(), vad_factory=lambda: fake_vad)
    resets = []
    for s in p._segmenters.values():
        monkeypatch.setattr(s, "reset", lambda s=s: resets.append(s.channel))
    p.start()
    src.on_status("ok", "a")
    src.on_status("error", "b")
    src.on_status("ok", "c")
    time.sleep(0.2)
    after_outage = sorted(resets)
    p.stop()  # zatrzymanie też domyka wypowiedzi (reset), dlatego liczymy przed nim
    assert after_outage == list(range(8))


# --- Ważne 7: przepełniona kolejka klienta wymusza resynchronizację, nie gubi finałów ---

def test_full_subscriber_queue_gets_resync():
    async def main():
        hub = EventHub()
        q = hub.subscribe()
        for _ in range(q.maxsize):
            q.put_nowait({"type": "level", "levels": []})
        hub.publish(transcript_event(0, "0-a", "ważne", True, 1, 2))
        await asyncio.sleep(0.05)
        items = [q.get_nowait() for _ in range(q.qsize())]
        return items

    items = asyncio.run(main())
    assert items == [{"type": "resync"}]


# --- dodatkowe: Qwen bez zmiany modelu, tryb offline przy niepełnym pobraniu ---

def test_engine_default_model_when_unset():
    s = settings_from_dict({"device": "X", "engine": "qwen",
                            "channels": [{"id": 0, "name": "A", "color": "#ff0000", "input": 0}]})
    assert s.model is None


def test_offline_requires_weights(monkeypatch):
    import os

    from huggingface_hub import constants

    import prodtrans.__main__ as m

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache",
                        lambda repo, f: "/x/config.json" if f == "config.json" else None)
    m._offline_if_cached("mlx-community/parakeet-tdt-0.6b-v3")
    assert "HF_HUB_OFFLINE" not in os.environ
