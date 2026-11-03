"""Silero VAD dla jednego kanału (model trzyma stan, więc instancja na kanał)."""

from __future__ import annotations

import numpy as np
import torch
from silero_vad import load_silero_vad

CHUNK = 512  # 32 ms przy 16 kHz, rozmiar wymagany przez Silero


class SileroVad:
    def __init__(self) -> None:
        self._model = load_silero_vad(onnx=True)

    def prob(self, chunk512: np.ndarray) -> float:
        return float(self._model(torch.from_numpy(np.ascontiguousarray(chunk512, np.float32)), 16000))

    __call__ = prob

    def reset(self) -> None:
        self._model.reset_states()
