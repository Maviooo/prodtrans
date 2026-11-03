import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from prodtrans.config import SettingsStore
from prodtrans.events import EventHub, transcript_event
from prodtrans.server import create_app
from prodtrans.store import SessionStore
from test_controller import DEVICES, FakeEngine, FakeSource, fake_vad

FIX = Path(__file__).parent / "fixtures"


def make_ctl(tmp_path, **kw):
    from prodtrans.controller import Controller

    opts = dict(list_devices=lambda: DEVICES, make_source=lambda s, sr, n: FakeSource(s, sr, n),
                make_engine=lambda n, m, language=None: FakeEngine(n, m), vad_factory=fake_vad, reinit=lambda: None)
    opts.update(kw)
    return Controller(SettingsStore(tmp_path / "s.json"), EventHub(), SessionStore(tmp_path / "sess"), **opts)


def test_ws_hello_contains_channels_and_history(tmp_path):
    ctl = make_ctl(tmp_path)
    ctl.hub.publish(transcript_event(0, "0-a", "cześć", True, 1.0, 2.0))
    ctl.sessions.append(ctl.hub.history()[0], "Kanał 1")
    client = TestClient(create_app(ctl))
    with client.websocket_connect("/ws") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "hello"
        assert len(hello["channels"]) == 4
        assert hello["channels"][0] == {"id": 0, "name": "Kanał 1", "color": "#e4572e"}
        assert len(hello["history"]) == 1
        ctl.hub.publish(transcript_event(1, "1-b", "nowy", True, 3.0, 4.0))
        assert ws.receive_json()["text"] == "nowy"
    r = client.get("/api/export")
    assert r.status_code == 200 and "Kanał 1: cześć" in r.text
    assert client.get("/").status_code == 200
    ctl.shutdown()


def test_settings_api_roundtrip_and_resync(tmp_path):
    ctl = make_ctl(tmp_path)
    client = TestClient(create_app(ctl))
    data = client.get("/api/settings").json()
    assert data["settings"]["device"] == "Interfejs"
    assert {e["id"] for e in data["engines"]} == {"parakeet", "qwen"}
    s = data["settings"]
    s["channels"][0]["name"] = "Inspicjent"
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # hello
        r = client.put("/api/settings", json=s)
        assert r.status_code == 200
        hello = ws.receive_json()
        assert hello["type"] == "hello" and hello["channels"][0]["name"] == "Inspicjent"
    ctl.shutdown()


def test_settings_api_rejects_with_polish_message(tmp_path):
    ctl = make_ctl(tmp_path)
    client = TestClient(create_app(ctl))
    s = client.get("/api/settings").json()["settings"]
    s["channels"][0]["name"] = ""
    r = client.put("/api/settings", json=s)
    assert r.status_code == 422 and "nazwę" in r.json()["detail"]
    ctl.shutdown()


def test_devices_start_stop(tmp_path):
    ctl = make_ctl(tmp_path)
    client = TestClient(create_app(ctl))
    assert [d["name"] for d in client.get("/api/devices").json()["devices"]] == ["Mikrofon", "Interfejs"]
    assert client.post("/api/start").json()["running"] is True
    assert client.get("/api/devices?refresh=1").status_code == 200
    assert ctl.running  # odświeżenie nie wyłącza nasłuchu na stałe
    assert client.post("/api/stop").json()["running"] is False
    ctl.shutdown()


@pytest.mark.slow
def test_pipeline_end_to_end_simulated(tmp_path):
    from prodtrans.audio import FileSource
    from prodtrans.engines import make_engine
    from prodtrans.vad import SileroVad

    ctl = make_ctl(
        tmp_path,
        make_source=lambda s, sr, n: FileSource([FIX / "pl_1.wav", FIX / "pl_2.wav"], sr,
                                                realtime=False, n_channels=n),
        make_engine=make_engine, vad_factory=SileroVad,
    )
    ctl.start()
    try:
        deadline = time.time() + 30
        while time.time() < deadline:
            finals = {e["channel"]: e["text"].lower() for e in ctl.hub.history() if e["final"]}
            if 0 in finals and 1 in finals:
                break
            time.sleep(0.2)
    finally:
        ctl.shutdown()
    assert "światło" in finals.get(0, "")
    assert "monitor" in finals.get(1, "")
