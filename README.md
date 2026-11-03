# ProdTrans

Transkrypcja po polsku na żywo z wielu kanałów audio (Dante), w pełni lokalnie na Macu.
Podgląd w przeglądarce: każdy kanał ma swój kolor, tekst wstępny (kursywa) zamienia się
w ostateczny po zakończeniu wypowiedzi. Sesje zapisują się do `sessions/`.

Silnik: NVIDIA Parakeet TDT 0.6B v3 na MLX (GPU Apple Silicon). Na M1 Max zdanie
4-sekundowe przepisuje w ~0,1–0,25 s. Opcjonalnie Qwen3-ASR do porównań.

## Wymagania

- Mac z Apple Silicon (M1 lub nowszy), macOS 14+.
- [uv](https://docs.astral.sh/uv/).
- Dante Virtual Soundcard (DVS) albo inny interfejs audio.
- Internet tylko przy pierwszym uruchomieniu (pobranie modelu, ~1,2 GB). Potem działa offline.

## Instalacja i uruchomienie

```bash
cd prodtrans
uv sync
uv run prodtrans
```

Aplikacja otworzy się w przeglądarce pod http://127.0.0.1:8765. Pierwsze uruchomienie
pobiera model (~1,2 GB). Całą konfigurację robisz w zakładce **Ustawienia**. Nie ma
plików do edytowania: ustawienia i zapisy sesji trafiają do
`~/Library/Application Support/ProdTrans/`.

Przy pierwszym starcie aplikacja sama wybiera urządzenie z największą liczbą wejść i
tworzy kanał na każde wejście (najwyżej 8).

## Ustawienia (w przeglądarce)

- **Źródło dźwięku:** wybór urządzenia (Dante Virtual Soundcard, mikrofon Maca, karta USB…),
  częstotliwość i podgląd sygnału na każdym wejściu. Kafelek z dźwiękiem się zapala, więc
  od razu widać, gdzie jest sygnał. Po włączeniu Dante kliknij „Odśwież listę”.
- **Kanały:** nazwa, kolor, wejście urządzenia, włącz/wyłącz, dodawanie i usuwanie
  (do 16 kanałów). Dwa kanały mogą słuchać tego samego wejścia, aplikacja to podpowie.
- **Rozpoznawanie mowy:** wybór silnika, a w „Ustawieniach zaawansowanych” czułość,
  cisza kończąca wypowiedź, odświeżanie tekstu na żywo, długości fragmentów i zapis
  nagrań do porównań.

Zmiany zatwierdzasz przyciskiem „Zapisz i zastosuj”. Nasłuch przebudowuje się w ułamku
sekundy, a model zostaje w pamięci. Zmiany nazw od razu widać na wszystkich podłączonych
ekranach. Przycisk w prawym górnym rogu włącza i wyłącza nasłuch.

## Dante

1. **DVS**: 48 kHz, liczba kanałów co najmniej tyle, ilu potrzebujesz (np. 8×8). Uruchom DVS.
2. **Dante Controller**: w zakładce *Routing* skieruj sygnały (interkom, FOH, inspicjent…)
   na wejścia DVS tego Maca.
3. W ProdTrans: **Ustawienia → Odśwież listę → wybierz Dante Virtual Soundcard**, potem
   przypisz kanały do wejść.

## Opcje uruchomienia

- `--host 0.0.0.0`: podgląd z innych urządzeń w sieci (np. iPad inspicjenta) pod
  `http://<ip-maca>:8765`. Serwer nie ma hasła i działa po zwykłym HTTP, więc używaj go
  tylko w zamkniętej sieci produkcyjnej.
- `--no-browser`: nie otwieraj przeglądarki.
- `--data-dir ŚCIEŻKA`: inny folder na ustawienia i sesje.
- `--simulate a.wav b.wav`: próba bez urządzenia (plik *i* = wejście *i*).

Eksport transkrypcji: przycisk „Eksport .txt” w widoku na żywo albo plik
`~/Library/Application Support/ProdTrans/sessions/<data>/transcript.txt`.

## Porównanie silników na własnych nagraniach

1. W Ustawieniach → Zaawansowane włącz „Zapisuj nagrania wypowiedzi”. Każda wypowiedź
   zapisze się do `~/Library/Application Support/ProdTrans/sessions/<data>/audio/*.wav`.
2. Obok wybranych plików `.wav` połóż `.txt` z poprawnym tekstem (do wyliczenia WER).
3. Uruchom:
   ```bash
   uv sync --extra qwen
   uv run --extra qwen python -m prodtrans bench --engines parakeet,qwen ~/Library/Application\ Support/ProdTrans/sessions/*/audio/*.wav
   ```

Aby używać Qwen na żywo: `uv sync --extra qwen`, uruchom aplikację przez
`uv run --extra qwen prodtrans` i wybierz Qwen3-ASR w Ustawieniach. Wyniki wstępne opisuje
`docs/qwen-spike.md`.

## Testy

```bash
uv run pytest -m "not slow"   # szybkie
uv run pytest                 # wszystkie (pobierają model)
```

Nagrania testowe po polsku tworzy `scripts/make_fixtures.sh` (głos „Zosia” z macOS).

## Znane ograniczenia

- Tekst pojawia się w kawałkach (co ~1 s wstępny, finał ~0,5 s po końcu zdania),
  a nie słowo po słowie.
- WER w `bench` liczy „12” i „dwanaście” jako różne słowa.
- Wyniki na syntetycznym głosie nie mówią wiele o prawdziwym interkomie. Do oceny
  jakości potrzebne są nagrania z próby.
- To narzędzie pomocnicze, nie zastępuje interkomu ani człowieka.

## Pomiary na żywym sprzęcie

_Do uzupełnienia po pierwszej próbie z Dante: obciążenie CPU/GPU, opóźnienie, jakość._
