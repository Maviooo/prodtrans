"""Qwen3-ASR (Alibaba) przez pakiet qwen-asr na GPU Maca (MPS). Instalacja: uv sync --extra qwen"""

from __future__ import annotations

import numpy as np

DEFAULT_MODEL = "Qwen/Qwen3-ASR-0.6B"


class QwenEngine:
    name = "qwen"

    def __init__(self, model_id: str = DEFAULT_MODEL, language: str = "Polish"):
        self.model_id = model_id
        self.language = language
        self.model = None

    def load(self) -> None:
        if self.model is not None:
            return
        try:
            import torch
            from qwen_asr import Qwen3ASRModel
        except ImportError as e:
            raise RuntimeError("Brak pakietu qwen-asr. Zainstaluj: uv sync --extra qwen") from e
        mps = torch.backends.mps.is_available()
        self.model = Qwen3ASRModel.from_pretrained(
            self.model_id,
            dtype=torch.float16 if mps else torch.float32,
            device_map="mps" if mps else "cpu",
            max_new_tokens=256,
        )
        self.transcribe(np.zeros(16000, np.float32))  # rozgrzanie

    def transcribe(self, audio16k: np.ndarray) -> str:
        self.load()
        if len(audio16k) < 1600:
            return ""
        result = self.model.transcribe(audio=(np.asarray(audio16k, np.float32), 16000), language=self.language)
        return result[0].text.strip()
