"""Porównanie silników na nagraniach: tekst, czas i WER (gdy obok .wav leży .txt ze wzorcem)."""

from __future__ import annotations

import re
import time
from pathlib import Path

import soundfile as sf

from prodtrans.audio import to_16k
from prodtrans.engines.base import Engine


def normalize(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def wer(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize(reference), normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1] / len(ref)


def load_items(paths: list[Path]) -> list[tuple[Path, str | None]]:
    items = []
    for p in paths:
        ref = p.with_suffix(".txt")
        items.append((p, ref.read_text(encoding="utf-8").strip() if ref.exists() else None))
    return items


def bench(engines: list[Engine], items: list[tuple[Path, str | None]]) -> list[dict]:
    rows = []
    for engine in engines:
        for path, ref in items:
            data, sr = sf.read(path, dtype="float32", always_2d=True)
            audio = to_16k(data.mean(axis=1), sr)
            t0 = time.perf_counter()
            text = engine.transcribe(audio)
            ms = (time.perf_counter() - t0) * 1000
            rows.append({
                "engine": engine.name,
                "file": path.name,
                "text": text,
                "ms": round(ms),
                "wer": None if ref is None else round(wer(ref, text), 3),
            })
    return rows
