"""Zapis sesji: transcript.txt (do czytania) i transcript.jsonl (do obróbki)."""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf


class SessionStore:
    def __init__(self, root: Path, debug_record: bool = False):
        self.dir = Path(root) / datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.txt_path = self.dir / "transcript.txt"
        self.jsonl_path = self.dir / "transcript.jsonl"
        self.txt_path.touch()
        self.debug_record = debug_record
        self._lock = threading.Lock()

    def append(self, event: dict, channel_name: str) -> None:
        if event.get("type") != "transcript" or not event["final"] or not event["text"]:
            return
        ts = datetime.fromtimestamp(event["t_start"]).strftime("%H:%M:%S")
        with self._lock:
            with self.txt_path.open("a", encoding="utf-8") as f:
                f.write(f"[{ts}] {channel_name}: {event['text']}\n")
            with self.jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({**event, "channel_name": channel_name}, ensure_ascii=False) + "\n")

    def save_audio(self, segment_id: str, audio16k: np.ndarray) -> None:
        """Tylko przy debug_record: zapis segmentu do późniejszego porównania silników."""
        if not self.debug_record:
            return
        d = self.dir / "audio"
        d.mkdir(exist_ok=True)
        sf.write(d / f"{segment_id}.wav", audio16k, 16000)
