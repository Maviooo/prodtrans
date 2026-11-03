from pathlib import Path

import pytest
import soundfile as sf

FIX = Path(__file__).parent / "fixtures"


@pytest.mark.slow
def test_parakeet_polish():
    from prodtrans.engines.parakeet import ParakeetEngine

    audio, sr = sf.read(FIX / "pl_1.wav", dtype="float32")
    assert sr == 16000
    text = ParakeetEngine("mlx-community/parakeet-tdt-0.6b-v3").transcribe(audio).lower()
    print(text)
    assert "światło" in text
    assert "chór" in text
