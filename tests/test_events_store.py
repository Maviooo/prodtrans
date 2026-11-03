import json

from prodtrans.events import EventHub, transcript_event
from prodtrans.store import SessionStore


def ev(seg, text="t", final=True, ch=0):
    return transcript_event(ch, seg, text, final, 1730000000.0, 1730000001.0)


def test_history_keeps_last_200():
    hub = EventHub()
    for i in range(250):
        hub.publish(ev(f"0-{i}", text=str(i)))
    h = hub.history()
    assert len(h) == 200
    assert h[-1]["text"] == "249"


def test_history_replaces_partial_with_final():
    hub = EventHub()
    hub.publish(ev("0-a", "wstępny", final=False))
    hub.publish(ev("0-b", "inny"))
    hub.publish(ev("0-a", "ostateczny"))
    h = hub.history()
    assert [e["text"] for e in h] == ["ostateczny", "inny"]
    assert all(e["final"] for e in h)


def test_level_events_not_in_history():
    hub = EventHub()
    hub.publish({"type": "level", "levels": [-60.0]})
    assert hub.history() == []


def test_store_writes_txt_line(tmp_path):
    store = SessionStore(tmp_path)
    store.append(ev("0-a", "test"), "Inspicjent")
    store.append(ev("0-b", "wstępny", final=False), "Inspicjent")
    txt = store.txt_path.read_text(encoding="utf-8")
    assert "Inspicjent: test" in txt
    assert "wstępny" not in txt
    lines = store.jsonl_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["channel_name"] == "Inspicjent"
