"""NVIDIA Parakeet TDT 0.6B v3 na MLX (GPU Apple Silicon)."""

from __future__ import annotations

import numpy as np

DEFAULT_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"


class ParakeetEngine:
    """Model wczytuje się przy pierwszym użyciu, w wątku, który go używa.

    MLX wiąże strumienie obliczeń z wątkiem, więc wczytanie modelu w jednym wątku
    i liczenie w innym kończy się błędem "There is no Stream(cpu, 1) in current thread".
    """

    name = "parakeet"

    def __init__(self, model_id: str = DEFAULT_MODEL):
        self.model_id = model_id
        self.model = None

    def load(self) -> None:
        if self.model is not None:
            return
        from parakeet_mlx import from_pretrained

        self.model = from_pretrained(self.model_id)
        assert self.model.preprocessor_config.sample_rate == 16000
        self.transcribe(np.zeros(16000, np.float32))  # rozgrzanie

    def transcribe(self, audio16k: np.ndarray) -> str:
        self.load()
        if len(audio16k) < 1600:  # < 0,1 s
            return ""
        import mlx.core as mx
        from parakeet_mlx.audio import get_logmel

        mel = get_logmel(mx.array(np.asarray(audio16k, np.float32)), self.model.preprocessor_config)
        # Parakeet zaczyna czasem zdanie od myślnika dialogowego ("- To jest…").
        return self.model.generate(mel)[0].text.strip().lstrip("-–— ").strip()
