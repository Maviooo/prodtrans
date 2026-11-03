import threading

import numpy as np

from prodtrans.segmenter import SegmentJob
from prodtrans.worker import TranscriptionWorker


class FakeEngine:
    name = "fake"

    def __init__(self, text="tekst"):
        self.text = text
        self.calls = []

    def transcribe(self, audio16k):
        self.calls.append(len(audio16k))
        return self.text


def job(seg="0-a", final=False, n=100, ch=0):
    return SegmentJob(ch, seg, np.zeros(n, np.float32), 0.0, 1.0, final)


def run_until(worker, results, count, timeout=5):
    done = threading.Event()
    orig = worker.on_result

    def on_result(r):
        orig(r)
        if len(results) >= count:
            done.set()

    worker.on_result = on_result
    worker.start()
    done.wait(timeout)
    worker.stop()


def test_worker_finals_first():
    eng, results = FakeEngine(), []
    w = TranscriptionWorker(eng, results.append)
    for i in range(3):
        w.submit(job(seg=f"1-{i}", n=10 + i))
    w.submit(job(seg="2-x", final=True, n=999))
    run_until(w, results, 4)
    assert eng.calls[0] == 999


def test_worker_drops_stale_partials():
    eng, results = FakeEngine(), []
    w = TranscriptionWorker(eng, results.append)
    for i in range(5):
        w.submit(job(seg="0-a", n=10 + i))
    run_until(w, results, 1)
    assert len(eng.calls) <= 1
    assert eng.calls == [14]  # zostaje najnowszy


def test_final_cancels_pending_partials():
    eng, results = FakeEngine(), []
    w = TranscriptionWorker(eng, results.append)
    w.submit(job(seg="0-a", n=10))
    w.submit(job(seg="0-a", final=True, n=20))
    run_until(w, results, 1)
    assert eng.calls == [20]


def test_worker_never_drops_finals():
    eng, results = FakeEngine(), []
    w = TranscriptionWorker(eng, results.append)
    for ch in range(8):
        for k in range(3):
            w.submit(job(seg=f"{ch}-{k}", final=True, ch=ch))
    run_until(w, results, 24)
    assert len(results) == 24


def test_worker_skips_empty_partial_text():
    eng, results = FakeEngine("  "), []
    w = TranscriptionWorker(eng, results.append)
    w.submit(job(final=False))
    w.start()
    import time

    time.sleep(0.3)
    w.stop()
    assert results == []
    assert len(eng.calls) == 1
