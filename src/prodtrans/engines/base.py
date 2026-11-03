"""Wspólny interfejs silników rozpoznawania mowy."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Engine(Protocol):
    name: str

    def transcribe(self, audio16k: np.ndarray) -> str: ...


def make_engine(name: str, model: str | None = None, language: str | None = None) -> Engine:
    if name == "parakeet":
        from prodtrans.engines.parakeet import DEFAULT_MODEL, ParakeetEngine

        return ParakeetEngine(model or DEFAULT_MODEL)
    if name == "qwen":
        from prodtrans.engines.qwen import DEFAULT_MODEL, QwenEngine

        return QwenEngine(model or DEFAULT_MODEL)
    if name == "whisper":
        from prodtrans.engines.whisper import DEFAULT_MODEL, WhisperEngine

        return WhisperEngine(model or DEFAULT_MODEL, language=language)
    raise ValueError(f"Nieznany silnik: {name}")
