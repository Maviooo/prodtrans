# ProdTrans MVP — plan implementacji

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Cel:** Lokalna aplikacja na Maca, która słucha 8 kanałów z Dante Virtual Soundcard i pokazuje w przeglądarce transkrypcję po polsku na żywo, z kolorami kanałów. Działa bez internetu.

**Architektura:** Proces Pythona otwiera urządzenie DVS (CoreAudio) przez `sounddevice`. Dla każdego kanału Silero VAD tnie mowę na wypowiedzi. Jeden wątek roboczy przepisuje je modelem Parakeet TDT 0.6B v3 (MLX, GPU Apple Silicon). Wyniki trafiają przez WebSocket (FastAPI) do prostej strony HTML/JS i do plików sesji. Silnik jest wymienny (`Engine`), więc Qwen3-ASR dochodzi jako drugi silnik do porównań.

**Tech stack:** Python 3.12 (uv), `parakeet-mlx`, `mlx`, `sounddevice`, `numpy`, `scipy` (resampling), `silero-vad` (+ `torch`), `fastapi`, `uvicorn[standard]`, `pytest`. Frontend: czysty HTML/CSS/JS, bez builda.

**Spec:** wymagania z rozmowy, spisane tutaj (brak osobnego dokumentu):
1. 8 kanałów wejściowych z Dante (DVS widoczny jako urządzenie CoreAudio), 48 kHz.
2. Transkrypcja po polsku, w pełni lokalnie. Domyślny silnik: Parakeet TDT 0.6B v3.
3. Widok w przeglądarce: jeden chronologiczny strumień, każdy kanał ma nazwę i kolor, można filtrować kanały, „skocz do teraz”.
4. Tekst wstępny (kursywa) w trakcie mówienia, potem tekst ostateczny po zakończeniu wypowiedzi.
5. Eksport transkrypcji z sesji (txt + jsonl ze znacznikami czasu).
6. Poza MVP: komendy, OSC, tłumaczenie.
7. Opcjonalny drugi silnik Qwen3-ASR i skrypt porównujący silniki na nagraniach.

## Global Constraints

- Python **3.12** (`uv python pin 3.12`). Systemowy 3.14 nie ma pewnych kół MLX/torch.
- Tylko macOS na Apple Silicon (MLX).
- Częstotliwość próbkowania wejścia: **48000 Hz**. Model i VAD: **16000 Hz**.
- Blok VAD: **512 próbek przy 16 kHz** (32 ms). Tego wymaga Silero.
- Domyślny model: `mlx-community/parakeet-tdt-0.6b-v3`.
- Serwer słucha na `127.0.0.1:8765` (opcja `--host 0.0.0.0`, żeby inni w sieci LAN mogli podglądać).
- Żadne audio nie jest zapisywane na dysk, chyba że włączy się `debug_record = true`.
- Teksty w UI po polsku.

## Review Focus

1. **Ciągła mowa bez pauz** (inspicjent mówi 30 s): wymuszone cięcie co `max_segment_s` (12 s), bez utraty próbek na granicy. → test w Task 3.
2. **Gorący mikrofon z szumem lub buczeniem**: brak pustych i wymyślonych wpisów. Segmenty krótsze niż `min_segment_s` (0,3 s) i puste wyniki są odrzucane. → testy w Task 3 i Task 4.
3. **Wszystkie 8 kanałów mówi naraz**: wyniki ostateczne nigdy nie giną, przestarzałe wstępne są pomijane. → test w Task 4.
4. **Restart DVS / odłączenie urządzenia**: status „brak audio” w UI i automatyczna próba ponownego otwarcia co 2 s. → test w Task 2.
5. **Przeglądarka dołącza później albo się przeładowuje**: dostaje ostatnie 200 zdarzeń historii. → test w Task 5.

---

## Struktura plików

```
prodtrans/
  pyproject.toml
  config.example.toml
  src/prodtrans/
    __main__.py        # CLI: devices | run | bench
    config.py          # AppConfig, ChannelConfig, load_config()
    audio.py           # list_devices(), DeviceSource, FileSource, to_16k()
    vad.py             # SileroVad (jedna instancja na kanał)
    segmenter.py       # ChannelSegmenter -> SegmentJob
    worker.py          # TranscriptionWorker (kolejka priorytetowa)
    engines/
      base.py          # Engine Protocol, make_engine()
      parakeet.py      # ParakeetEngine
      qwen.py          # QwenEngine (Task 7)
    events.py          # TranscriptEvent, LevelEvent, StatusEvent, EventHub
    store.py           # SessionStore (txt + jsonl)
    pipeline.py        # Pipeline: spina źródło, VAD, segmenter, worker, hub
    server.py          # FastAPI: /, /ws, /api/export
    web/index.html, web/app.js, web/style.css
  tests/
    fixtures/          # wav po polsku generowane przez `say -v Zosia`
    test_config.py, test_audio.py, test_segmenter.py, test_worker.py,
    test_engine_parakeet.py, test_events_store.py, test_server.py
  scripts/make_fixtures.sh
```

Wątki: callback `sounddevice` tylko kopiuje blok do `queue.Queue`. Wątek `pipeline` dzieli kanały, robi resampling, VAD i segmentację. Wątek `worker` uruchamia MLX (jeden model, wywołania po kolei). Zdarzenia przechodzą do pętli asyncio przez `loop.call_soon_threadsafe`.

---

### Task 1: Szkielet projektu, konfiguracja, lista urządzeń

**Files:**
- Create: `pyproject.toml`, `config.example.toml`, `src/prodtrans/__init__.py`, `src/prodtrans/config.py`, `src/prodtrans/__main__.py`, `src/prodtrans/audio.py` (tylko `list_devices`)
- Test: `tests/test_config.py`

**Interfaces:**
- Produces:
  - `@dataclass ChannelConfig(index: int, name: str, color: str, enabled: bool = True)`. `index` liczony od 0 w obrębie urządzenia.
  - `@dataclass VadConfig(threshold: float = 0.5, min_silence_ms: int = 500, pre_roll_ms: int = 300, min_segment_s: float = 0.3, max_segment_s: float = 12.0, partial_every_s: float = 1.0)`
  - `@dataclass AppConfig(device: str, sample_rate: int, channels: list[ChannelConfig], vad: VadConfig, engine: str = "parakeet", model: str = "mlx-community/parakeet-tdt-0.6b-v3", sessions_dir: str = "sessions", debug_record: bool = False)`
  - `load_config(path: Path) -> AppConfig`. Rzuca `ConfigError(str)`.
  - `validate_against_device(cfg: AppConfig, max_input_channels: int) -> None`. Rzuca `ConfigError`.
  - `list_devices() -> list[dict]` z kluczami `index, name, max_input_channels, default_samplerate`.
  - CLI: `uv run python -m prodtrans devices`

- [ ] **Step 1:** `uv init --package prodtrans`, `uv python pin 3.12`, `uv add parakeet-mlx sounddevice numpy scipy silero-vad fastapi "uvicorn[standard]"`, `uv add --dev pytest httpx`, `git init`.
- [ ] **Step 2: Napisz testy, które na razie nie przechodzą** w `tests/test_config.py`:
  - `test_loads_example_config`: `load_config(Path("config.example.toml"))` daje 8 kanałów, `device == "Dante Virtual Soundcard"`, `sample_rate == 48000`, `vad.max_segment_s == 12.0`.
  - `test_duplicate_channel_index_rejected`: dwa kanały z `index=0` dają `ConfigError` z „index” w treści.
  - `test_bad_color_rejected`: `color="red"` daje `ConfigError` (wymagany format `#rrggbb`).
  - `test_channel_beyond_device`: `validate_against_device(cfg, max_input_channels=4)` przy kanale `index=7` daje `ConfigError`, którego treść zawiera „7” i „4”.
- [ ] **Step 3:** `uv run pytest tests/test_config.py -v`. Oczekiwane: FAIL (import error).
- [ ] **Step 4:** Zaimplementuj `config.py` (`tomllib`) i `config.example.toml`: kanały 0–7 z nazwami „Inspicjent”, „FOH”, „Monitory”, „Światło”, „Scena L”, „Scena P”, „Kamera 1”, „Reżyser” i 8 rozróżnialnymi kolorami. Dodaj `list_devices()` (z `sd.query_devices()`, tylko urządzenia z `max_input_channels > 0`) i podkomendę `devices`, która je wypisuje.
- [ ] **Step 5:** `uv run pytest tests/test_config.py -v` daje PASS. `uv run python -m prodtrans devices` wypisuje listę urządzeń (DVS, jeśli jest uruchomiony).
- [ ] **Step 6:** Commit: `feat: project skeleton, config and device listing`.

### Task 2: Źródła audio (Dante i pliki) oraz resampling

**Files:**
- Modify: `src/prodtrans/audio.py`
- Test: `tests/test_audio.py`

**Interfaces:**
- Consumes: `AppConfig` (Task 1)
- Produces:
  - `to_16k(x: np.ndarray, sr: int) -> np.ndarray`: float32, mono. `scipy.signal.resample_poly(x, 1, 3)` dla 48 kHz, ogólnie `resample_poly(x, 16000//g, sr//g)`.
  - `class AudioSource(Protocol)`: `start(on_block: Callable[[np.ndarray], None], on_status: Callable[[str, str], None]) -> None`, `stop() -> None`. Blok to `np.ndarray` o kształcie `(frames, n_device_channels)`, float32, 48 kHz. `on_status(state, message)`, gdzie `state ∈ {"ok", "error"}`.
  - `DeviceSource(device: str, sample_rate: int, n_channels: int, blocksize: int = 1536)`: `sd.InputStream`. Callback tylko wykonuje `on_block(indata.copy())`. Przy wyjątku otwarcia lub `CallbackAbort` wysyła `on_status("error", ...)` i próbuje ponownie co 2 s w wątku.
  - `FileSource(paths: list[Path], sample_rate: int = 48000, realtime: bool = True)`: plik *i* trafia do kanału *i*, krótsze pliki są dopełniane ciszą. Bloki po 1536 klatek, z `time.sleep` przy `realtime=True`. Służy do testów i demo bez Dante.

- [ ] **Step 1: Napisz testy, które na razie nie przechodzą:**
  - `test_to_16k_length`: 48000 próbek daje 16000 (±1).
  - `test_to_16k_preserves_tone`: sinus 440 Hz po resamplingu ma szczyt FFT przy 440 Hz (±5 Hz).
  - `test_file_source_shapes`: dwa pliki wav 16 kHz 1 s (wygenerowane w teście przez `soundfile`/`scipy.io.wavfile`) dają bloki `(1536, 2)` przy `realtime=False`, łącznie ≥ 48000 klatek.
  - `test_device_source_reports_missing_device`: `DeviceSource("Nie ma takiego", 48000, 8)` → `on_status` jest wywołane z `"error"` w ciągu 0,5 s. Potem `stop()` kończy wątek ponawiania w ciągu 3 s.
- [ ] **Step 2:** `uv run pytest tests/test_audio.py -v` daje FAIL.
- [ ] **Step 3:** Zaimplementuj według interfejsów. `FileSource` przelicza plik na 48 kHz przez `resample_poly`.
- [ ] **Step 4:** `uv run pytest tests/test_audio.py -v` daje PASS.
- [ ] **Step 5:** Commit: `feat: audio sources (Dante device, file simulation) and resampling`.

### Task 3: VAD i segmentacja na kanał

**Files:**
- Create: `src/prodtrans/vad.py`, `src/prodtrans/segmenter.py`
- Test: `tests/test_segmenter.py`

**Interfaces:**
- Consumes: `VadConfig` (Task 1)
- Produces:
  - `class SileroVad`: `__init__()` (`silero_vad.load_silero_vad(onnx=True)`, osobna instancja na kanał, bo model ma stan), `prob(chunk512: np.ndarray) -> float`, `reset() -> None`.
  - `@dataclass SegmentJob(channel: int, segment_id: str, audio: np.ndarray, t_start: float, t_end: float, final: bool)`. `audio` to 16 kHz float32, `t_*` to `time.time()`.
  - `class ChannelSegmenter(channel: int, cfg: VadConfig, vad: Callable[[np.ndarray], float], emit: Callable[[SegmentJob], None], clock: Callable[[], float] = time.time)`, `feed(x16k: np.ndarray) -> None`. Bufor wewnętrzny dzieli wejście na bloki po 512 próbek.

Logika (deterministyczna, dlatego opisana):
- Bufor pre-roll trzyma ostatnie `pre_roll_ms` ciszy. Gdy `prob ≥ threshold`, zaczyna się mowa: segment = pre-roll + bieżący blok, nowe `segment_id = f"{channel}-{uuid4().hex[:8]}"`.
- W trakcie mowy, co `partial_every_s`, emituj `SegmentJob(final=False)` z całym dotychczasowym audio segmentu.
- Koniec mowy: `min_silence_ms` kolejnych bloków z `prob < threshold - 0.15` (histereza). Wtedy emituj `final=True`, o ile długość ≥ `min_segment_s`. Krótsze segmenty odrzucaj bez emitowania.
- Gdy długość ≥ `max_segment_s`: emituj `final=True`, a następny segment zaczyna się od kolejnej próbki z **nowym** `segment_id`. Nic nie ginie.

- [ ] **Step 1: Napisz testy, które na razie nie przechodzą** (fałszywy VAD `lambda chunk: 1.0 if chunk.max() > 0.1 else 0.0` i fałszywy zegar przesuwany o 0,032 s na blok):
  - `test_single_utterance`: 0,5 s ciszy + 2 s „mowy” (sinus amp. 0,5) + 1 s ciszy daje dokładnie jeden `final=True`. Długość audio ≈ 2,0 s + 0,3 s pre-roll + 0,5 s ciszy końcowej (±0,1 s).
  - `test_partials_emitted`: przy tych samych danych wpada ≥ 1 `final=False` z tym samym `segment_id` co final.
  - `test_short_blip_dropped`: 0,1 s mowy nie daje żadnego `final`.
  - `test_long_speech_split_without_loss` (Review Focus 1): 30 s ciągłej mowy daje 3 finały o różnych `segment_id`. Suma długości audio finałów ≥ 30 s.
  - `test_noise_below_threshold_ignored` (Review Focus 2): 10 s szumu o amp. 0,05 nie daje żadnego zdarzenia.
- [ ] **Step 2:** `uv run pytest tests/test_segmenter.py -v` daje FAIL.
- [ ] **Step 3:** Zaimplementuj `segmenter.py` według powyższej logiki i `vad.py`.
- [ ] **Step 4:** `uv run pytest tests/test_segmenter.py -v` daje PASS. Dodatkowo test dymny `test_silero_loads`: `SileroVad().prob(np.zeros(512, np.float32)) < 0.5`.
- [ ] **Step 5:** Commit: `feat: per-channel VAD segmentation with partials and forced splits`.

### Task 4: Silnik Parakeet i kolejka transkrypcji

**Files:**
- Create: `src/prodtrans/engines/base.py`, `src/prodtrans/engines/parakeet.py`, `src/prodtrans/worker.py`, `scripts/make_fixtures.sh`
- Test: `tests/test_engine_parakeet.py`, `tests/test_worker.py`

**Interfaces:**
- Consumes: `SegmentJob` (Task 3)
- Produces:
  - `class Engine(Protocol)`: `name: str`, `transcribe(audio16k: np.ndarray) -> str`.
  - `make_engine(name: str, model: str) -> Engine`, gdzie `name ∈ {"parakeet", "qwen"}`.
  - `ParakeetEngine(model_id: str)`: `from_pretrained(model_id)`, potem `get_logmel(mx.array(audio), model.preprocessor_config)`, potem `model.generate(mel)[0].text.strip()`. Przy pierwszym uruchomieniu rozgrzewa model 1 s ciszy.
  - `@dataclass TranscriptResult(job: SegmentJob, text: str, engine_ms: float)`
  - `class TranscriptionWorker(engine: Engine, on_result: Callable[[TranscriptResult], None])`: `submit(job) -> None`, `start()`, `stop()`. Kolejka priorytetowa: finały przed wstępnymi, wewnątrz grupy FIFO. Przy `submit` wstępnego wyniku starszy wstępny wynik tego samego `segment_id`, który jeszcze czeka w kolejce, jest usuwany. Gdy przyjdzie finał danego `segment_id`, czekające wstępne wyniki tego segmentu też są usuwane. Puste teksty (`""` po `strip`) nie są zgłaszane.

- [ ] **Step 1:** `scripts/make_fixtures.sh`: `say -v Zosia -o /tmp/x.aiff "<zdanie>"` i `afconvert -f WAVE -d LEI16@16000`. Tworzy `tests/fixtures/pl_1.wav` („Uwaga, stand by światło dwanaście, za chwilę wchodzi chór.”) i `pl_2.wav` („Mikrofon numer trzy, proszę podgłośnić monitor na scenie.”).
- [ ] **Step 2: Napisz testy, które na razie nie przechodzą:**
  - `test_parakeet_polish` (oznaczony `@pytest.mark.slow`, pobiera model ~1,2 GB): transkrypcja `pl_1.wav` po zmianie na małe litery zawiera „światło” i „chór”.
  - `test_worker_finals_first`: fałszywy silnik, który zapisuje kolejność wywołań. 3 wstępne i 1 finał wysłane przed `start()` → pierwsze wywołanie dotyczy finału.
  - `test_worker_drops_stale_partials` (Review Focus 3): 5 wstępnych dla tego samego `segment_id` przed `start()` daje najwyżej 1 wywołanie dla wstępnych.
  - `test_worker_never_drops_finals`: 8 kanałów × 3 finały → `on_result` dostaje dokładnie 24 wyniki.
  - `test_worker_skips_empty_text` (Review Focus 2): silnik zwraca `"  "`, więc `on_result` nie jest wywołane.
- [ ] **Step 3:** `uv run pytest tests/test_worker.py tests/test_engine_parakeet.py -v` daje FAIL.
- [ ] **Step 4:** Zaimplementuj. Worker to `threading.Thread` + `heapq` pod `threading.Condition`.
- [ ] **Step 5:** Testy dają PASS. Zanotuj w commicie `engine_ms` dla `pl_1.wav` (oczekiwane < 300 ms na M1+).
- [ ] **Step 6:** Commit: `feat: parakeet engine and prioritized transcription worker`.

### Task 5: Zdarzenia, zapis sesji, serwer WebSocket i potok

**Files:**
- Create: `src/prodtrans/events.py`, `src/prodtrans/store.py`, `src/prodtrans/pipeline.py`, `src/prodtrans/server.py`
- Modify: `src/prodtrans/__main__.py` (podkomenda `run`)
- Test: `tests/test_events_store.py`, `tests/test_server.py`

**Interfaces:**
- Consumes: wszystko z Task 1–4
- Produces:
  - JSON zdarzeń (kontrakt z frontendem):
    - `{"type":"transcript","channel":0,"segment_id":"0-ab12cd34","text":"...","final":true,"t_start":1730000000.1,"t_end":1730000002.4}`
    - `{"type":"level","levels":[-42.1, ... 8 wartości dBFS]}` (10 Hz)
    - `{"type":"status","state":"ok"|"error","message":"..."}`
    - `{"type":"hello","channels":[{"index":0,"name":"Inspicjent","color":"#e4572e"}...],"history":[...transcript events...]}`
  - `class EventHub(history_size: int = 200)`: `publish(event: dict)` (bezpieczne wątkowo przez `loop.call_soon_threadsafe`), `subscribe() -> asyncio.Queue`, `unsubscribe(q)`, `history() -> list[dict]` (tylko finały, plus ostatni wstępny dla segmentów jeszcze bez finału).
  - `class SessionStore(dir: Path)`: tworzy `sessions/YYYY-MM-DD_HHMMSS/` z `transcript.jsonl` i `transcript.txt`. Linia txt: `[HH:MM:SS] Nazwa: tekst`. `append(event: dict, channel_name: str)` zapisuje tylko finały i robi `flush` po każdym zapisie.
  - `class Pipeline(cfg: AppConfig, source: AudioSource, engine: Engine, hub: EventHub, store: SessionStore)`: `start()`, `stop()`. Dzieli blok na kanały z `cfg.channels` (włączone), wywołuje `to_16k`, potem `ChannelSegmenter.feed`, liczy poziom RMS w dBFS i wysyła go co 100 ms.
  - `create_app(cfg, hub) -> FastAPI`: `GET /` (pliki z `web/`), `WS /ws` (najpierw `hello`, potem strumień zdarzeń), `GET /api/export` (aktualny `transcript.txt` jako `text/plain`).
  - CLI: `uv run python -m prodtrans run --config config.toml [--simulate a.wav b.wav ...] [--host 127.0.0.1] [--port 8765]`

- [ ] **Step 1: Napisz testy, które na razie nie przechodzą:**
  - `test_history_keeps_last_200` (Review Focus 5): 250 finałów daje `history()` o długości 200 kończącą się na ostatnim.
  - `test_history_replaces_partial_with_final`: wstępny, a potem finał tego samego `segment_id` → w historii jest tylko finał.
  - `test_store_writes_txt_line`: plik txt zawiera `"Inspicjent: test"`, a jsonl ma 1 linię. Wstępne wyniki nie są zapisywane.
  - `test_ws_hello_contains_channels_and_history` (`fastapi.testclient`): po `hub.publish(final)` nowy klient dostaje `hello` z 8 kanałami i 1 wpisem w historii.
  - `test_pipeline_end_to_end_simulated` (`@pytest.mark.slow`): `FileSource([pl_1.wav, pl_2.wav], realtime=False)` z prawdziwym `ParakeetEngine` → w ciągu 15 s hub dostaje finał z kanału 0 zawierający „światło” i finał z kanału 1 zawierający „monitor”.
- [ ] **Step 2:** Testy dają FAIL.
- [ ] **Step 3:** Zaimplementuj. `run` buduje `DeviceSource` albo `FileSource` (gdy podano `--simulate`), woła `validate_against_device` dla DVS, startuje `Pipeline` i `uvicorn`, a przy Ctrl+C zatrzymuje wszystko w kolejności: źródło, potok, worker.
- [ ] **Step 4:** Testy dają PASS.
- [ ] **Step 5:** Commit: `feat: event hub, session store, websocket server and pipeline`.

### Task 6: Interfejs w przeglądarce

**Files:**
- Create: `src/prodtrans/web/index.html`, `src/prodtrans/web/app.js`, `src/prodtrans/web/style.css`

**Interfaces:**
- Consumes: kontrakt JSON z Task 5

Zachowanie:
- Ciemny motyw (praca przy konsoli). Pasek górny: status połączenia i audio („Połączono”, „Brak audio: …”), przyciski „Eksport” (`/api/export`) i „Wyczyść widok”.
- Rząd 8 „kart kanałów”: nazwa w kolorze kanału, miernik poziomu z `level`, kliknięcie przełącza widoczność kanału. Filtr zapamiętuje `localStorage`, owinięty w try/catch.
- Strumień: wpis = `[HH:MM:SS] Nazwa  tekst`, z kolorowym paskiem po lewej. Wstępny wpis jest kursywą i wyszarzony. Finał o tym samym `segment_id` zastępuje go w miejscu (mapa `segment_id → element`).
- Autoprzewijanie, gdy użytkownik jest na dole. Gdy przewinie w górę, pojawia się przycisk „Skocz do teraz (N nowych)”.
- Ponowne łączenie WebSocket co 1 s. Po `hello` strumień jest budowany od nowa z historii.
- Duża czcionka (domyślnie 20 px), przyciski A− / A+.

- [ ] **Step 1:** Zaimplementuj trzy pliki.
- [ ] **Step 2:** Weryfikacja: `uv run python -m prodtrans run --config config.example.toml --simulate tests/fixtures/pl_1.wav tests/fixtures/pl_2.wav`, potem otwórz `http://127.0.0.1:8765` w przeglądarce w aplikacji. Oczekiwane: wpisy w kolorach 2 kanałów, kursywa przechodzi w zwykły tekst, mierniki się ruszają, przeładowanie strony odtwarza historię, ukrycie kanału filtruje wpisy, eksport pobiera txt.
- [ ] **Step 3:** Commit: `feat: live transcript web UI`.

### Task 7: Drugi silnik (Qwen3-ASR) i skrypt porównawczy

**Files:**
- Create: `src/prodtrans/engines/qwen.py`
- Modify: `src/prodtrans/engines/base.py` (`make_engine`), `src/prodtrans/__main__.py` (podkomenda `bench`)
- Test: `tests/test_bench.py`

**Interfaces:**
- Consumes: `Engine`, `make_engine` (Task 4)
- Produces:
  - `QwenEngine(model_id: str = "Qwen/Qwen3-ASR-0.6B")` → `transcribe(audio16k) -> str`, z językiem wymuszonym na polski.
  - `bench(engines: list[Engine], items: list[tuple[Path, str | None]]) -> list[dict]`. Wiersz: `engine, file, text, ms, wer` (WER liczony, gdy jest plik `.txt` z wzorcem obok `.wav`; normalizacja: małe litery, bez interpunkcji). Klucze wiersza są stałe.
  - CLI: `uv run python -m prodtrans bench --engines parakeet,qwen nagrania/*.wav` wypisuje tabelę i średnie WER/ms na silnik.

- [ ] **Step 1 (spike, limit 2 h):** sprawdź, którą drogą Qwen3-ASR działa na Macu: najpierw pakiet `qwen-asr` (transformers na MPS/CPU), a jeśli nie zadziała, GGUF `handy-computer/Qwen3-ASR-1.7B-gguf` przez transcribe.cpp. Zapisz w `docs/qwen-spike.md`, co zadziałało, i czas dla `pl_1.wav`. Jeśli żadna droga nie działa, pomiń Steps 3–4 dla Qwen. `bench` i tak powstaje dla Parakeeta.
- [ ] **Step 2: Napisz testy, które na razie nie przechodzą:** `test_wer_normalization`: wzorzec „Światło, dwanaście!” i hipoteza „światło dwanaście” dają `wer == 0.0`. `test_bench_rows`: fałszywy silnik i 2 pliki dają 2 wiersze z kluczami `engine,file,text,ms,wer`.
- [ ] **Step 3:** Zaimplementuj `QwenEngine` według wyniku spike'a oraz `bench` (WER: odległość Levenshteina na słowach, bez dodatkowej zależności).
- [ ] **Step 4:** Testy dają PASS. Uruchom `bench` na fixture'ach.
- [ ] **Step 5:** Commit: `feat: qwen3-asr engine and engine benchmark`.

### Task 8: Uruchomienie z Dante i README

**Files:**
- Create: `README.md`

- [ ] **Step 1:** Napisz w README: ustawienia DVS (48 kHz, co najmniej 8 kanałów wejściowych, routing w Dante Controller na wejścia 1–8 DVS), `prodtrans devices`, skopiowanie `config.example.toml` do `config.toml` i ustawienie `device` na nazwę z listy, `run`, podgląd z innych komputerów (`--host 0.0.0.0`), plus znane ograniczenia.
- [ ] **Step 2:** Test na żywym sprzęcie (razem z Tobą): 8 kanałów z Dante, 15 minut próby. Zanotuj w README: obciążenie CPU/GPU (Monitor aktywności), średnie opóźnienie od końca zdania do finału i odczucie co do jakości po polsku.
- [ ] **Step 3:** Commit: `docs: setup and Dante routing guide`.

---

## Po MVP (poza zakresem)
Słownik/podpowiedzi żargonu, komendy po słowach kluczowych + OSC, tłumaczenie, aplikacja .app (py2app) zamiast terminala.
