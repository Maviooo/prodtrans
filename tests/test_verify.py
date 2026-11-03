import time
from pathlib import Path

import numpy as np
import pytest

from prodtrans.config import SettingsStore, settings_to_dict
from prodtrans.controller import Controller
from prodtrans.events import EventHub
from prodtrans.store import SessionStore
from prodtrans.verify import accept_verified
from test_controller import DEVICES, FakeEngine, FakeSource, fake_vad

FIX = Path(__file__).parent / "fixtures"


class FakeVerifier(FakeEngine):
    def __init__(self, name, model, language=None, reply="Zweryfikowany tekst."):
        super().__init__(name, model)
        self.language = language
        self.reply = reply
        self.calls = 0

    def transcribe(self, audio):
        self.calls += 1
        return self.reply


def make(tmp_path, reply="Zweryfikowany tekst.", verify=True):
    verifiers = []

    def make_engine(name, model, language=None):
        if name == "whisper":
            verifiers.append(FakeVerifier(name, model, language, reply))
            return verifiers[-1]
        return FakeEngine(name, model)

    store = SettingsStore(tmp_path / "s.json")
    ctl = Controller(store, EventHub(), SessionStore(tmp_path / "sess"), list_devices=lambda: DEVICES,
                     make_source=lambda s, sr, n: FakeSource(s, sr, n), make_engine=make_engine,
                     vad_factory=fake_vad, reinit=lambda: None)
    if not verify:
        d = settings_to_dict(ctl.settings)
        d["verify"] = False
        ctl.apply(d)
    ctl.verifiers = verifiers
    return ctl


def speak(ctl, input_=0):
    src = ctl._pipeline.source
    t = np.arange(48000) / 48000
    block = np.zeros((48000 * 3, 4), np.float32)
    block[:48000, input_] = 0.5 * np.sin(2 * np.pi * 220 * t)
    for i in range(0, len(block), 1536):
        src.on_block(block[i : i + 1536])


def wait_for(pred, timeout=3):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(0.02)
    return False


def finals(ctl):
    return [e for e in ctl.hub.history() if e["final"]]


def test_verified_text_replaces_draft_and_is_saved_once(tmp_path):
    ctl = make(tmp_path)
    ctl.start()
    speak(ctl)
    assert wait_for(lambda: any(e.get("verify") == "done" for e in finals(ctl)))
    [ev] = finals(ctl)
    assert ev["text"] == "Zweryfikowany tekst." and ev["draft"] == "tekst"
    assert ctl.verifiers[0].language == "pl"
    txt = ctl.sessions.txt_path.read_text(encoding="utf-8")
    assert txt.count("Zweryfikowany tekst.") == 1 and "tekst\n" not in txt.replace("Zweryfikowany tekst.", "")
    ctl.shutdown()


def test_draft_published_first_with_pending_marker(tmp_path):
    ctl = make(tmp_path)
    seen = []
    orig = ctl.hub.publish
    ctl.hub.publish = lambda ev: (seen.append(dict(ev)), orig(ev))
    ctl.start()
    speak(ctl)
    assert wait_for(lambda: any(e.get("verify") == "done" for e in seen))
    kinds = [e.get("verify") for e in seen if e["type"] == "transcript" and e["final"]]
    assert kinds == ["pending", "done"]
    ctl.shutdown()


def test_empty_verification_keeps_draft(tmp_path):
    ctl = make(tmp_path, reply="")
    ctl.start()
    speak(ctl)
    assert wait_for(lambda: any(e.get("verify") == "done" for e in finals(ctl)))
    assert finals(ctl)[0]["text"] == "tekst"
    assert "Kanał 1: tekst" in ctl.sessions.txt_path.read_text(encoding="utf-8")
    ctl.shutdown()


def test_verification_off_saves_draft_directly(tmp_path):
    ctl = make(tmp_path, verify=False)
    ctl.start()
    speak(ctl)
    assert wait_for(lambda: finals(ctl))
    assert "verify" not in finals(ctl)[0]
    assert ctl.verifiers == []
    assert wait_for(lambda: "Kanał 1: tekst" in ctl.sessions.txt_path.read_text(encoding="utf-8"))
    ctl.shutdown()


def test_verify_language_auto_passes_none(tmp_path):
    ctl = make(tmp_path)
    d = settings_to_dict(ctl.settings)
    d["verify_language"] = "auto"
    ctl.apply(d)
    ctl.start()
    assert ctl.verifiers[-1].language is None
    ctl.shutdown()


def test_accept_verified_rejects_hallucinations():
    sec = np.zeros(16000 * 2, np.float32)  # 2 s
    assert accept_verified("tekst", "Inny tekst.", sec, "pl") == "Inny tekst."
    assert accept_verified("tekst", "", sec, "pl") == "tekst"
    assert accept_verified("tekst", "a" * 200, sec, "pl") == "tekst"  # za długi na 2 s mowy
    assert accept_verified("tekst", "Садур.", sec, "pl") == "tekst"  # obce pismo przy polskim
    assert accept_verified("tekst", "Dziękuję za obejrzenie!", sec, "pl") == "tekst"  # znane widmo
    assert accept_verified("tekst", "건조 건조", sec, None) == "건조 건조"  # tryb auto: bez filtra pisma


@pytest.mark.slow
def test_whisper_polish_fixture():
    import soundfile as sf

    from prodtrans.engines.whisper import WhisperEngine

    a, _ = sf.read(FIX / "pl_2.wav", dtype="float32")
    text = WhisperEngine(language="pl").transcribe(a).lower()
    assert "monitor" in text and "scenie" in text


@pytest.mark.slow
def test_parakeet_strips_leading_dialogue_dash(monkeypatch):
    from prodtrans.engines.parakeet import ParakeetEngine

    e = ParakeetEngine()
    e.load()

    class R:
        text = " - To jest test."

    monkeypatch.setattr(e.model, "generate", lambda mel: [R()])
    assert e.transcribe(np.zeros(16000, np.float32)) == "To jest test."


def test_empty_draft_is_still_verified(tmp_path):
    # Parakeet potrafi nie usłyszeć cichej końcówki; Whisper dostaje ją mimo to.
    ctl = make(tmp_path, reply="Druga część zdania.")
    seen = []
    orig = ctl.hub.publish
    ctl.hub.publish = lambda ev: (seen.append(dict(ev)), orig(ev))
    ctl._worker = None
    ctl.start()
    ctl._worker.engine.transcribe = lambda a: ""
    speak(ctl)
    assert wait_for(lambda: any(e.get("verify") == "done" for e in finals(ctl)))
    [ev] = finals(ctl)
    assert ev["text"] == "Druga część zdania." and ev["draft"] == ""
    # pusty szkic nie pojawia się na ekranie jako osobny wpis "weryfikuję…"
    assert not [e for e in seen if e["type"] == "transcript" and e["final"] and e.get("verify") == "pending"]
    assert "Druga część zdania." in ctl.sessions.txt_path.read_text(encoding="utf-8")
    ctl.shutdown()


def test_empty_draft_and_empty_verification_publish_nothing_saved(tmp_path):
    ctl = make(tmp_path, reply="")
    ctl.start()
    ctl._worker.engine.transcribe = lambda a: ""
    speak(ctl)
    time.sleep(1.0)
    assert finals(ctl) == []
    assert ctl.sessions.txt_path.read_text(encoding="utf-8") == ""
    ctl.shutdown()


def test_verified_without_letters_rejected():
    sec = np.zeros(16000 * 2, np.float32)
    assert accept_verified("Ha ha ha!", "!", sec, "pl") == "Ha ha ha!"
    assert accept_verified("raz", "1, 2, 3", sec, "pl") == "1, 2, 3"  # same cyfry są w porządku
