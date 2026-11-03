import json

import pytest

from prodtrans.config import (
    ConfigError,
    Settings,
    SettingsStore,
    default_settings,
    settings_from_dict,
    settings_to_dict,
)

DEVICES = [
    {"index": 0, "name": "MacBook Pro Microphone", "max_input_channels": 1, "default_samplerate": 48000.0},
    {"index": 3, "name": "Dante Virtual Soundcard", "max_input_channels": 64, "default_samplerate": 48000.0},
]


def base(**over):
    d = settings_to_dict(default_settings(DEVICES[1:]))
    d.update(over)
    return d


def test_default_settings_use_device_inputs_up_to_8():
    s = default_settings(DEVICES[1:])
    assert s.device == "Dante Virtual Soundcard"
    assert [c.input for c in s.channels] == list(range(8))
    assert [c.id for c in s.channels] == list(range(8))
    assert s.channels[0].name == "Kanał 1"
    assert len({c.color for c in s.channels}) == 8


def test_default_settings_single_mic():
    s = default_settings(DEVICES[:1])
    assert len(s.channels) == 1 and s.channels[0].input == 0


def test_default_settings_no_devices():
    s = default_settings([])
    assert s.device is None and len(s.channels) == 1


def test_roundtrip():
    s = default_settings(DEVICES[1:])
    assert settings_from_dict(json.loads(json.dumps(settings_to_dict(s)))) == s


def test_duplicate_channel_id_rejected():
    d = base()
    d["channels"][1]["id"] = d["channels"][0]["id"]
    with pytest.raises(ConfigError, match="id"):
        settings_from_dict(d)


def test_bad_color_rejected():
    d = base()
    d["channels"][0]["color"] = "red"
    with pytest.raises(ConfigError, match="kolor"):
        settings_from_dict(d)


def test_empty_name_rejected():
    d = base()
    d["channels"][0]["name"] = "   "
    with pytest.raises(ConfigError, match="nazw"):
        settings_from_dict(d)


def test_input_beyond_device_rejected():
    d = base()
    d["channels"][0]["input"] = 64
    with pytest.raises(ConfigError, match="64"):
        settings_from_dict(d, devices=DEVICES)


def test_unknown_device_allowed():
    # urządzenie może być chwilowo wyłączone (np. Dante), ustawienia muszą się zapisać
    s = settings_from_dict(base(device="Coś wyłączonego"), devices=DEVICES)
    assert s.device == "Coś wyłączonego"


def test_vad_ranges_validated():
    d = base()
    d["vad"]["threshold"] = 1.5
    with pytest.raises(ConfigError, match="Czułość"):
        settings_from_dict(d)


def test_too_many_channels_rejected():
    d = base()
    d["channels"] = [dict(d["channels"][0], id=i) for i in range(17)]
    with pytest.raises(ConfigError, match="16"):
        settings_from_dict(d)


def test_store_creates_defaults_and_persists(tmp_path):
    store = SettingsStore(tmp_path / "settings.json")
    s = store.load(devices=DEVICES[1:])
    assert isinstance(s, Settings) and s.device == "Dante Virtual Soundcard"
    s.channels[0].name = "Inspicjent"
    store.save(s)
    assert SettingsStore(tmp_path / "settings.json").load(devices=[]).channels[0].name == "Inspicjent"


def test_store_recovers_from_corrupt_file(tmp_path):
    p = tmp_path / "settings.json"
    p.write_text("{nie json", encoding="utf-8")
    s = SettingsStore(p).load(devices=DEVICES[:1])
    assert s.device == "MacBook Pro Microphone"
    assert (tmp_path / "settings.json.broken").exists()


def test_input_error_message_grammar():
    from prodtrans.config import inputs_word

    assert [inputs_word(n) for n in (1, 2, 5, 12, 22, 64)] == [
        "1 wejście", "2 wejścia", "5 wejść", "12 wejść", "22 wejścia", "64 wejścia"]
    d = base(device="MacBook Pro Microphone")
    d["channels"] = [dict(d["channels"][0], input=5)]
    with pytest.raises(ConfigError) as e:
        settings_from_dict(d, devices=DEVICES)
    assert str(e.value) == ("Urządzenie „MacBook Pro Microphone” ma 1 wejście, "
                            "a kanał „Kanał 1” (wejście 6) wskazuje dalej")


def test_verify_settings_defaults_and_validation():
    s = default_settings(DEVICES[1:])
    assert s.verify is True and s.verify_language == "pl"
    d = base(verify_language="xx")
    with pytest.raises(ConfigError, match="weryfikacji"):
        settings_from_dict(d)
    assert settings_from_dict(base(verify=False, verify_language="auto")).verify_language == "auto"


def test_old_settings_file_without_verify_fields_loads():
    d = base()
    del d["verify"], d["verify_language"]
    assert settings_from_dict(d).verify is True

