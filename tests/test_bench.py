import numpy as np
import soundfile as sf

from prodtrans.bench import bench, normalize, wer


class FakeEngine:
    name = "fake"

    def transcribe(self, audio16k):
        return "światło dwanaście"


def test_wer_normalization():
    assert normalize("Światło, dwanaście!") == ["światło", "dwanaście"]
    assert wer("Światło, dwanaście!", "światło dwanaście") == 0.0
    assert wer("a b c d", "a x c") == 0.5


def test_bench_rows(tmp_path):
    items = []
    for i, ref in enumerate(["Światło, dwanaście!", None]):
        p = tmp_path / f"{i}.wav"
        sf.write(p, np.zeros(16000, np.float32), 16000)
        items.append((p, ref))
    rows = bench([FakeEngine()], items)
    assert len(rows) == 2
    assert all(set(r) == {"engine", "file", "text", "ms", "wer"} for r in rows)
    assert rows[0]["wer"] == 0.0
    assert rows[1]["wer"] is None
