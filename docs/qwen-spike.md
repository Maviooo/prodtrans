# Spike: Qwen3-ASR na Macu (2026-11-03, M1 Max)

Droga, która działa: pakiet `qwen-asr` 0.0.6 (backend transformers 4.57) na GPU przez MPS,
`dtype=float16`, `device_map="mps"`, `language="Polish"`. Instalacja: `uv sync --extra qwen`.
vLLM i GGUF nie były potrzebne.

| Plik | Parakeet v3 (MLX) | Qwen3-ASR 0.6B (MPS) |
|---|---|---|
| pl_1 (4,1 s) | 102 ms, „Uwaga, stądby światło 12. Za chwilę wchodzi chór.” | ~510 ms, „Uwaga: Stand by światło 12 za chwilę wchodzi furg.” |
| pl_2 (3,8 s) | 228 ms, „Mikrofon numer 3. Proszę podgłośnić monitor na scenie.” | ~460 ms, „Mikrofon numer trzy, proszę podłączyć monitor na scenę.” |

Pierwsze wczytanie (pobranie ~1,8 GB) trwało 77 s, pierwsza transkrypcja ~4 s (rozgrzewanie).

Wniosek: Qwen działa, ale jest wolniejszy, a na syntetycznym głosie mniej dokładny. Rozstrzygnąć
mogą tylko prawdziwe nagrania z interkomu (`debug_record = true` + `prodtrans bench`).
Wersja 1.7B nie była testowana.
