"""Stan aplikacji sterowany z przeglądarki: ustawienia, nasłuch (start/stop) i model mowy.

Model jest drogi w wczytaniu, więc żyje tak długo, jak wybrany silnik się nie zmienia.
Zmiana urządzenia, kanałów czy czułości przebudowuje tylko potok nasłuchu.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable

from prodtrans import audio
from prodtrans.config import ConfigError, Settings, SettingsStore, settings_from_dict
from prodtrans.engines import make_engine as default_make_engine
from prodtrans.events import EventHub, transcript_event
from prodtrans.pipeline import Pipeline
from prodtrans.store import SessionStore
from prodtrans.vad import SileroVad
from prodtrans.verify import accept_verified
from prodtrans.worker import TranscriptionWorker, TranscriptResult

log = logging.getLogger(__name__)

MAX_INPUTS = 64  # DVS potrafi mieć ich więcej; tyle mierzymy i pokazujemy


def default_make_source(settings: Settings, sample_rate: int, n_inputs: int) -> audio.AudioSource:
    return audio.DeviceSource(settings.device or "", sample_rate, n_inputs)


class Controller:
    def __init__(self, store: SettingsStore, hub: EventHub, sessions: SessionStore,
                 list_devices: Callable[[], list[dict]] = audio.list_devices,
                 make_source: Callable[[Settings, int, int], audio.AudioSource] = default_make_source,
                 make_engine=default_make_engine,
                 vad_factory=SileroVad,
                 reinit: Callable[[], None] = audio._reinit_portaudio):
        self.store = store
        self.hub = hub
        self.sessions = sessions
        self._list_devices = list_devices
        self._make_source = make_source
        self._make_engine = make_engine
        self._vad_factory = vad_factory
        self._reinit = reinit
        self._lock = threading.RLock()
        self._pipeline: Pipeline | None = None
        self._worker: TranscriptionWorker | None = None
        self._engine_key: tuple[str, str | None] | None = None
        self._verifier: TranscriptionWorker | None = None
        self._verifier_lang: str | None = None
        self._pending: dict[str, dict] = {}  # segment_id -> szkic czekający na weryfikację
        self._pending_lock = threading.Lock()
        self.settings: Settings = store.load(self.devices())

    # --- urządzenia ---

    def devices(self, refresh: bool = False) -> list[dict]:
        with self._lock:
            if refresh:
                # Odświeżenie listy PortAudio zamyka otwarte strumienie, więc nasłuch idzie na chwilę w dół.
                was_running = self.running
                self._stop_pipeline()
                try:
                    self._reinit()
                except Exception:
                    log.exception("Nie udało się odświeżyć listy urządzeń")
                if was_running:
                    self._start_pipeline()
            try:
                return self._list_devices()
            except Exception:
                log.exception("Nie udało się pobrać listy urządzeń")
                return []

    def _device_info(self, settings: Settings) -> dict | None:
        return next((d for d in self.devices() if d["name"] == settings.device), None)

    # --- nasłuch ---

    @property
    def running(self) -> bool:
        return self._pipeline is not None

    def start(self) -> None:
        with self._lock:
            if not self.running:
                self._start_pipeline()

    def stop(self) -> None:
        with self._lock:
            self._stop_pipeline()

    def _start_pipeline(self) -> None:
        s = self.settings
        self._ensure_worker()
        self._ensure_verifier()
        dev = self._device_info(s)
        sample_rate = s.sample_rate or (int(dev["default_samplerate"]) if dev else 48000)
        if dev:
            n_inputs = min(dev["max_input_channels"], MAX_INPUTS)
        else:
            n_inputs = max(c.input for c in s.channels) + 1
        source = self._make_source(s, sample_rate, n_inputs)
        self._pipeline = Pipeline(s, sample_rate, source, self._worker.submit, self.hub,
                                  save_audio=self.sessions.save_audio if s.debug_record else None,
                                  vad_factory=self._vad_factory)
        self._pipeline.start()

    def _stop_pipeline(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None
            self.hub.publish({"type": "status", "source": "audio", "state": "stopped",
                              "message": "Nasłuch zatrzymany"})

    # --- model mowy ---

    def _ensure_worker(self) -> None:
        key = (self.settings.engine, self.settings.model)
        if self._worker is not None and self._engine_key == key:
            return
        if self._worker is not None:
            self._worker.stop()
        engine = self._make_engine(*key)
        self._engine_key = key
        self._worker = TranscriptionWorker(engine, self._on_result)
        self.hub.publish({"type": "status", "source": "engine", "state": "loading",
                          "message": f"Wczytuję model {engine.name}…"})
        self._worker.start()
        threading.Thread(target=self._report_engine, args=(self._worker,), daemon=True).start()

    def _report_engine(self, worker: TranscriptionWorker, source: str = "engine") -> None:
        worker.ready.wait()
        if worker is not (self._worker if source == "engine" else self._verifier):
            return
        label = "Weryfikacja" if source == "verifier" else "Model"
        if worker.error:
            self.hub.publish({"type": "status", "source": source, "state": "error",
                              "message": f"{label} nie wystartowała: {worker.error}" if source == "verifier"
                              else f"Model nie wystartował: {worker.error}"})
        else:
            self.hub.publish({"type": "status", "source": source, "state": "ok",
                              "message": f"{label} gotowa" if source == "verifier"
                              else f"Model gotowy ({worker.engine.name})"})

    # --- weryfikacja drugim modelem ---

    def _ensure_verifier(self) -> None:
        lang = self.settings.verify_language if self.settings.verify else None
        if not self.settings.verify:
            if self._verifier is not None:
                self._verifier.stop()
                self._verifier = None
                self.hub.clear_status("verifier")
            self._flush_pending()
            return
        if self._verifier is not None and self._verifier_lang == lang:
            return
        if self._verifier is not None:
            self._verifier.stop()
        self._flush_pending()
        engine = self._make_engine("whisper", None, language=None if lang == "auto" else lang)
        self._verifier_lang = lang
        self._verifier = TranscriptionWorker(engine, self._on_verified)
        self.hub.publish({"type": "status", "source": "verifier", "state": "loading",
                          "message": "Wczytuję model weryfikacji (pierwszy raz pobiera ~1,6 GB)…"})
        self._verifier.start()
        threading.Thread(target=self._report_engine, args=(self._verifier, "verifier"), daemon=True).start()

    def _verifying(self) -> TranscriptionWorker | None:
        v = self._verifier
        return v if v is not None and not v.error else None

    def _channel_name(self, channel: int) -> str:
        return next((c.name for c in self.settings.channels if c.id == channel), f"Kanał {channel + 1}")

    def _on_result(self, r: TranscriptResult) -> None:
        j = r.job
        ev = transcript_event(j.channel, j.segment_id, r.text, j.final, j.t_start, j.t_end)
        # Weryfikujemy też puste szkice: Parakeet potrafi nie usłyszeć cichej końcówki zdania.
        verifier = self._verifying() if j.final else None
        if verifier is not None:
            ev["verify"] = "pending"
            with self._pending_lock:
                self._pending[j.segment_id] = ev
            if r.text:
                self.hub.publish(ev)
            verifier.submit(j)
            return
        self.hub.publish(ev)
        self.sessions.append(ev, self._channel_name(j.channel))

    def _on_verified(self, r: TranscriptResult) -> None:
        with self._pending_lock:
            draft = self._pending.pop(r.job.segment_id, None)
        if draft is None:
            return
        lang = None if self._verifier_lang == "auto" else self._verifier_lang
        text = accept_verified(draft["text"], r.text, r.job.audio, lang)
        ev = {**draft, "text": text, "verify": "done", "draft": draft["text"]}
        self.hub.publish(ev)
        self.sessions.append(ev, self._channel_name(ev["channel"]))

    def _flush_pending(self) -> None:
        """Szkice bez weryfikacji (wyłączenie, zamknięcie) trafiają do zapisu w obecnej postaci."""
        with self._pending_lock:
            pending, self._pending = self._pending, {}
        for ev in pending.values():
            ev = {**ev, "verify": "skipped"}
            self.hub.publish(ev)
            self.sessions.append(ev, self._channel_name(ev["channel"]))

    # --- ustawienia ---

    def apply(self, data: dict) -> Settings:
        """Sprawdza, zapisuje i stosuje nowe ustawienia. Przy błędzie nic się nie zmienia."""
        new = settings_from_dict(data, devices=self.devices())
        with self._lock:
            self.store.save(new)
            was_running = self.running
            self._stop_pipeline()
            self.settings = new
            self.sessions.debug_record = new.debug_record
            if was_running:
                self._start_pipeline()
            else:
                if (new.engine, new.model) != self._engine_key and self._worker is not None:
                    self._ensure_worker()
                if self._verifier is not None or not new.verify:
                    self._ensure_verifier()
        self.hub.resync()
        return new

    def status(self) -> dict:
        w = self._worker
        return {
            "running": self.running,
            "device": self.settings.device,
            "engine": self.settings.engine,
            "engine_ready": bool(w and w.ready.is_set() and not w.error),
            "verify": self.settings.verify,
            "verifier_ready": bool(self._verifier and self._verifier.ready.is_set() and not self._verifier.error),
        }

    def shutdown(self) -> None:
        with self._lock:
            self._stop_pipeline()
            if self._worker is not None:
                self._worker.stop()
                self._worker = None
            if self._verifier is not None:
                self._verifier.stop()
                self._verifier = None
            self._flush_pending()


__all__ = ["Controller", "ConfigError"]
