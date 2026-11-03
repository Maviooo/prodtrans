"""Decyzja, czy przyjąć tekst z modelu weryfikującego, czy zostawić szybki szkic."""

from __future__ import annotations

import re
import unicodedata

import numpy as np

MAX_CHARS_PER_S = 30  # szybka mowa to ~15-20 znaków/s; więcej = model "dopisuje" tekst

# Frazy, które Whisper wymyśla na ciszy i szumie (z napisów filmowych w danych treningowych).
PHANTOMS = {
    "dziękuję za obejrzenie", "dziękuję za uwagę", "dzięki za obejrzenie",
    "napisy stworzone przez społeczność amara org", "subskrybuj", "zostaw łapkę w górę",
    "thank you for watching", "thanks for watching",
}


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def _non_latin(text: str) -> bool:
    for ch in text:
        if ch.isalpha() and not unicodedata.name(ch, "").startswith("LATIN"):
            return True
    return False


def accept_verified(draft: str, verified: str, audio16k: np.ndarray, language: str | None) -> str:
    v = verified.strip()
    if not any(ch.isalnum() for ch in v):  # puste albo sama interpunkcja ("!", "...")
        return draft
    seconds = max(len(audio16k) / 16000, 0.5)
    if len(v) > MAX_CHARS_PER_S * seconds + 20:
        return draft
    if language == "pl" and _non_latin(v):
        return draft
    if _norm(v) in PHANTOMS:
        return draft
    return v
