import time

import numpy as np
import pytest

from prodtrans.config import ConfigError, SettingsStore, settings_to_dict
from prodtrans.controller import Controller
from prodtrans.events import EventHub
from prodtrans.store import SessionStore

DEVICES = [
    {"index": 0, "name": "Mikrofon", "max_input_channels": 1, "default_samplerate": 44100.0},
    {"index": 1, "name": "Interfejs", "max_input_channels": 4, "default_samplerate": 48000.0},
]


class FakeSource:
    def __init__(self, settings, sample_rate, n_inputs):
        self.settings, self.sample_rate, self.n_inputs = settings, sample_rate, n_inputs
        self.started = self.stopped = False

    def start(self, on_block, on_status):
        self.started = True
        self.on_block, self.on_status = on_block, on_status
        on_status("ok", "fake")

    def stop(self):
        self.stopped = True


class FakeEngine:
    def __init__(self, name, model):
        self.name, self.model_id = name, model or f"{name}-default"
        self.loaded = False

    def load(self):
        self.loaded = True

    def transcribe(self, audio):
        return "tekst"


def fake_vad():
    return lambda chunk: 1.0 if np.abs(chunk).max() > 0.1 else 0.0


@pytest.fixture
def ctl(tmp_path):
    sources, engines = [], []

    def make_source(settings, sr, n):
        sources.append(FakeSource(settings, sr, n))
        return sources[-1]

    def make_engine(name, model, language=None):
        e = FakeEngine(name, model)
        if name != "whisper":  # weryfikator liczymy osobno (test_verify.py)
            engines.append(e)
        return e

    c = Controller(
        SettingsStore(tmp_path / "settings.json"), EventHub(), SessionStore(tmp_path / "sessions"),
        list_devices=lambda: DEVICES, make_source=make_source, make_engine=make_engine,
        vad_factory=fake_vad, reinit=lambda: None,
    )
    c.sources, c.engines = sources, engines
    yield c
    c.shutdown()


def test_first_run_defaults_and_start(ctl):
    assert ctl.settings.device == "Interfejs"
    assert len(ctl.settings.channels) == 4
    ctl.start()
    assert ctl.running
    src = ctl.sources[-1]
    assert src.started and src.sample_rate == 48000 and src.n_inputs == 4


def test_apply_renames_without_reloading_engine_and_resyncs(ctl):
    ctl.start()
    d = settings_to_dict(ctl.settings)
    d["channels"][0]["name"] = "Inspicjent"
    s = ctl.apply(d)
    assert s.channels[0].name == "Inspicjent"
    assert len(ctl.engines) == 1  # model nie wczytywany ponownie
    assert ctl.sources[0].stopped and ctl.sources[-1].started  # nasłuch przebudowany
    assert ctl.store.load(devices=[]).channels[0].name == "Inspicjent"


def test_apply_switches_device_and_rate(ctl):
    ctl.start()
    d = settings_to_dict(ctl.settings)
    d["device"] = "Mikrofon"
    d["channels"] = [d["channels"][0]]
    ctl.apply(d)
    src = ctl.sources[-1]
    assert src.settings.device == "Mikrofon" and src.sample_rate == 44100 and src.n_inputs == 1


def test_apply_rejects_input_beyond_device(ctl):
    d = settings_to_dict(ctl.settings)
    d["device"] = "Mikrofon"  # 1 wejście, a kanały wskazują na 2-4
    with pytest.raises(ConfigError):
        ctl.apply(d)
    assert ctl.settings.device == "Interfejs"  # nic nie zmienione


def test_engine_change_loads_new_engine(ctl):
    ctl.start()
    d = settings_to_dict(ctl.settings)
    d["engine"] = "qwen"
    ctl.apply(d)
    assert [e.name for e in ctl.engines] == ["parakeet", "qwen"]


def test_stop_keeps_engine(ctl):
    ctl.start()
    ctl.stop()
    assert not ctl.running and ctl.sources[-1].stopped
    ctl.start()
    assert len(ctl.engines) == 1


def test_results_use_current_channel_names(ctl):
    ctl.start()
    src = ctl.sources[-1]
    t = np.arange(48000) / 48000
    speech = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    block = np.zeros((48000 * 3, 4), np.float32)
    block[: len(speech), 2] = speech
    for i in range(0, len(block), 1536):
        src.on_block(block[i : i + 1536])
    deadline = time.time() + 3
    while time.time() < deadline:
        finals = [e for e in ctl.hub.history() if e["final"]]
        if finals:
            break
        time.sleep(0.05)
    assert finals and finals[0]["channel"] == ctl.settings.channels[2].id
    assert "Kanał 3: tekst" in ctl.sessions.txt_path.read_text(encoding="utf-8")


def test_status_reports_running_and_engine(ctl):
    ctl.start()
    deadline = time.time() + 2
    while not ctl.status()["engine_ready"] and time.time() < deadline:
        time.sleep(0.02)
    st = ctl.status()
    assert st["running"] and st["engine_ready"] and st["engine"] == "parakeet"


def test_stop_finishes_utterance_in_progress(ctl):
    ctl.start()
    src = ctl.sources[-1]
    t = np.arange(48000 * 2) / 48000
    block = np.zeros((48000 * 2, 4), np.float32)
    block[:, 0] = 0.5 * np.sin(2 * np.pi * 220 * t)  # mowa trwa, gdy przychodzi stop
    for i in range(0, len(block), 1536):
        src.on_block(block[i : i + 1536])
    time.sleep(0.3)
    ctl.stop()
    deadline = time.time() + 3
    while not [e for e in ctl.hub.history() if e["final"]] and time.time() < deadline:
        time.sleep(0.05)
    hist = ctl.hub.history()
    assert hist and all(e["final"] for e in hist)  # żadnej wiszącej kursywy
