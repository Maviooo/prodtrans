import threading
import time

import numpy as np
import soundfile as sf

from prodtrans.audio import DeviceSource, FileSource, to_16k


def test_to_16k_length():
    y = to_16k(np.zeros(48000, np.float32), 48000)
    assert abs(len(y) - 16000) <= 1
    assert y.dtype == np.float32


def test_to_16k_preserves_tone():
    t = np.arange(48000) / 48000
    y = to_16k(np.sin(2 * np.pi * 440 * t).astype(np.float32), 48000)
    spec = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), 1 / 16000)
    assert abs(freqs[spec.argmax()] - 440) <= 5


def test_file_source_shapes(tmp_path):
    paths = []
    for i in range(2):
        p = tmp_path / f"{i}.wav"
        sf.write(p, np.zeros(16000, np.float32), 16000)
        paths.append(p)
    blocks = []
    done = threading.Event()
    src = FileSource(paths, realtime=False, on_end=done.set)
    src.start(blocks.append, lambda s, m: None)
    assert done.wait(5)
    src.stop()
    assert all(b.shape == (1536, 2) for b in blocks)
    assert sum(len(b) for b in blocks) >= 48000


def test_device_source_reports_missing_device():
    statuses = []
    src = DeviceSource("Nie ma takiego urządzenia", 48000, 8)
    src.start(lambda b: None, lambda s, m: statuses.append((s, m)))
    deadline = time.time() + 0.5
    while not statuses and time.time() < deadline:
        time.sleep(0.01)
    assert statuses and statuses[0][0] == "error"
    t0 = time.time()
    src.stop()
    assert time.time() - t0 < 3


def test_stream_decimator_matches_one_shot():
    from prodtrans.audio import StreamDecimator

    t = np.arange(48000) / 48000
    x = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    d = StreamDecimator(48000)
    y = np.concatenate([d.process(x[i : i + 1536]) for i in range(0, len(x), 1536)])
    assert len(y) == 16000
    # po ustabilizowaniu filtra sygnał jest ciągły: brak skoków na granicach bloków
    diffs = np.abs(np.diff(y[2000:]))
    assert diffs.max() < 0.1
    spec = np.abs(np.fft.rfft(y[2000:]))
    freqs = np.fft.rfftfreq(len(y) - 2000, 1 / 16000)
    assert abs(freqs[spec.argmax()] - 300) <= 5
