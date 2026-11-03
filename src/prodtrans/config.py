"""Ustawienia aplikacji: model, walidacja i zapis w JSON (edytowane z przeglądarki)."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

MAX_CHANNELS = 16
ENGINES = ("parakeet", "qwen")
VERIFY_LANGUAGES = ("pl", "auto")
PALETTE = [
    "#e4572e", "#29a3e6", "#f3a712", "#a8c256", "#c45bd6", "#3ec7b0", "#ef6fa8", "#d9d9d9",
    "#ff8c42", "#6c8cff", "#ffd23f", "#5fd068", "#b388eb", "#4dd0e1", "#f78fb3", "#a1887f",
]
_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


class ConfigError(ValueError):
    pass


@dataclass
class ChannelConfig:
    id: int        # stały identyfikator kanału w aplikacji (historia, kolory)
    name: str
    color: str
    input: int     # numer wejścia urządzenia liczony od 0
    enabled: bool = True


@dataclass
class VadConfig:
    threshold: float = 0.5
    min_silence_ms: int = 500
    pre_roll_ms: int = 300
    min_segment_s: float = 0.3
    max_segment_s: float = 12.0
    partial_every_s: float = 1.0


# (pole, min, max, etykieta w UI) — komunikaty walidacji mówią językiem użytkownika
_VAD_RANGES = [
    ("threshold", 0.05, 0.95, "Czułość"),
    ("min_silence_ms", 150, 3000, "Cisza kończąca wypowiedź"),
    ("pre_roll_ms", 0, 1000, "Dźwięk przed mową"),
    ("min_segment_s", 0.1, 2.0, "Najkrótsza wypowiedź"),
    ("max_segment_s", 3.0, 60.0, "Najdłuższy fragment"),
    ("partial_every_s", 0.3, 5.0, "Odświeżanie tekstu wstępnego"),
]


@dataclass
class Settings:
    device: str | None
    channels: list[ChannelConfig]
    sample_rate: int | None = None  # None = domyślna częstotliwość urządzenia
    vad: VadConfig = field(default_factory=VadConfig)
    engine: str = "parakeet"
    model: str | None = None  # None = domyślny model wybranego silnika
    debug_record: bool = False
    verify: bool = True             # drugi model sprawdza i poprawia każdą wypowiedź
    verify_language: str = "pl"     # "pl" = wymuszony polski, "auto" = model sam wykrywa

    @property
    def enabled_channels(self) -> list[ChannelConfig]:
        return [c for c in self.channels if c.enabled]


def inputs_word(n: int) -> str:
    if n == 1:
        return "1 wejście"
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return f"{n} wejścia"
    return f"{n} wejść"


def default_channel(i: int, input_: int | None = None) -> ChannelConfig:
    return ChannelConfig(id=i, name=f"Kanał {i + 1}", color=PALETTE[i % len(PALETTE)],
                         input=i if input_ is None else input_)


def default_settings(devices: list[dict]) -> Settings:
    """Pierwsze uruchomienie: urządzenie z największą liczbą wejść, kanał na wejście (max 8)."""
    if not devices:
        return Settings(device=None, channels=[default_channel(0)])
    dev = max(devices, key=lambda d: d["max_input_channels"])
    n = max(1, min(8, dev["max_input_channels"]))
    return Settings(device=dev["name"], channels=[default_channel(i) for i in range(n)])


def settings_to_dict(s: Settings) -> dict:
    return asdict(s)


def _num(d: dict, key: str, kind, where: str):
    v = d.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ConfigError(f"{where}: pole '{key}' musi być liczbą")
    return kind(v)


def settings_from_dict(d: dict, devices: list[dict] | None = None) -> Settings:
    """Buduje i sprawdza ustawienia. `devices` pozwala sprawdzić numery wejść, jeśli urządzenie jest widoczne."""
    if not isinstance(d, dict):
        raise ConfigError("Ustawienia muszą być obiektem")

    raw_channels = d.get("channels") or []
    if not raw_channels:
        raise ConfigError("Dodaj co najmniej jeden kanał")
    if len(raw_channels) > MAX_CHANNELS:
        raise ConfigError(f"Najwyżej {MAX_CHANNELS} kanałów")

    channels = []
    for i, c in enumerate(raw_channels):
        where = f"Kanał {i + 1}"
        name = str(c.get("name", "")).strip()
        if not name:
            raise ConfigError(f"{where}: podaj nazwę")
        color = str(c.get("color", ""))
        if not _COLOR_RE.match(color):
            raise ConfigError(f"{where}: kolor '{color}' musi mieć format #rrggbb")
        input_ = _num(c, "input", int, where)
        if input_ < 0:
            raise ConfigError(f"{where}: numer wejścia musi być dodatni")
        channels.append(ChannelConfig(id=_num(c, "id", int, where), name=name[:40], color=color.lower(),
                                      input=input_, enabled=bool(c.get("enabled", True))))
    ids = [c.id for c in channels]
    if len(set(ids)) != len(ids):
        raise ConfigError("Powtórzony id kanału")

    vad_raw = d.get("vad") or {}
    vad = VadConfig()
    for key, lo, hi, label in _VAD_RANGES:
        if key in vad_raw:
            kind = type(getattr(vad, key))
            v = _num(vad_raw, key, kind, label)
            if not lo <= v <= hi:
                raise ConfigError(f"{label}: wartość {v} poza zakresem {lo}–{hi}")
            setattr(vad, key, v)

    engine = d.get("engine", "parakeet")
    if engine not in ENGINES:
        raise ConfigError(f"Nieznany silnik: {engine}")
    verify_language = d.get("verify_language", "pl")
    if verify_language not in VERIFY_LANGUAGES:
        raise ConfigError(f"Nieznany język weryfikacji: {verify_language}")
    sr = d.get("sample_rate")
    if sr is not None and sr not in (16000, 44100, 48000, 88200, 96000):
        raise ConfigError(f"Nieobsługiwana częstotliwość: {sr}")

    s = Settings(device=d.get("device") or None, channels=channels, sample_rate=sr, vad=vad,
                 engine=engine, model=d.get("model") or None, debug_record=bool(d.get("debug_record")),
                 verify=bool(d.get("verify", True)), verify_language=verify_language)

    dev = next((x for x in devices or [] if x["name"] == s.device), None)
    if dev is not None:
        bad = [c for c in s.channels if c.input >= dev["max_input_channels"]]
        if bad:
            names = ", ".join(f"„{c.name}” (wejście {c.input + 1})" for c in bad)
            who = "kanał" if len(bad) == 1 else "kanały"
            verb = "wskazuje" if len(bad) == 1 else "wskazują"
            raise ConfigError(
                f"Urządzenie „{dev['name']}” ma {inputs_word(dev['max_input_channels'])}, "
                f"a {who} {names} {verb} dalej"
            )
    return s


class SettingsStore:
    """Plik settings.json w katalogu danych aplikacji. Użytkownik go nie edytuje."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def load(self, devices: list[dict]) -> Settings:
        if self.path.exists():
            try:
                return settings_from_dict(json.loads(self.path.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, ConfigError, TypeError, AttributeError) as e:
                broken = self.path.with_name(self.path.name + ".broken")
                log.warning("Uszkodzone ustawienia (%s), zapisuję kopię w %s i zaczynam od domyślnych", e, broken)
                self.path.replace(broken)
        s = default_settings(devices)
        self.save(s)
        return s

    def save(self, s: Settings) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(settings_to_dict(s), ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)
