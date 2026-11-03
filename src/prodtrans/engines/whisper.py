"""Whisper large-v3-turbo na MLX: wolniejszy, dokładniejszy model do weryfikacji wypowiedzi."""

from __future__ import annotations

import numpy as np

DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"


class WhisperEngine:
    """Model wczytuje się przy pierwszym użyciu, w wątku, który go używa (jak Parakeet)."""

    name = "whisper"

    def __init__(self, model_id: str = DEFAULT_MODEL, language: str | None = "pl"):
        self.model_id = model_id
        self.language = language  # None = Whisper sam wykrywa język
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        self.transcribe(np.zeros(16000, np.float32))  # pobranie i rozgrzanie

    def transcribe(self, audio16k: np.ndarray) -> str:
        if len(audio16k) < 1600:
            return ""
        import mlx_whisper

        result = mlx_whisper.transcribe(
            np.asarray(audio16k, np.float32),
            path_or_hf_repo=self.model_id,
            language=self.language,
            # Bez powtórek z wyższą temperaturą: na krótkich nagraniach potrafiły trwać ~10 s.
            temperature=0.0,
            condition_on_previous_text=False,
            verbose=None,
        )
        return result["text"].strip()
