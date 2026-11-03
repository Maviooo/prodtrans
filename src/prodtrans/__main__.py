"""CLI: prodtrans devices | run | bench"""

from __future__ import annotations

import argparse
from pathlib import Path


def cmd_devices(_args) -> None:
    from prodtrans.audio import list_devices

    devices = list_devices()
    if not devices:
        print("Brak urządzeń wejściowych.")
    for d in devices:
        print(
            f"[{d['index']:>2}] {d['name']}  "
            f"(wejścia: {d['max_input_channels']}, {int(d['default_samplerate'])} Hz)"
        )


def _offline_if_cached(model_id: str) -> None:
    """Gdy model jest już pobrany, nie pytaj Hugging Face (start bez sieci na miejscu)."""
    import os

    import huggingface_hub
    from huggingface_hub import constants

    try_to_load_from_cache = huggingface_hub.try_to_load_from_cache

    cached = all(isinstance(try_to_load_from_cache(model_id, f), str)
                 for f in ("config.json", "model.safetensors"))
    if "HF_HUB_OFFLINE" not in os.environ and cached:
        os.environ["HF_HUB_OFFLINE"] = "1"
        constants.HF_HUB_OFFLINE = True  # biblioteka czyta zmienną przy imporcie


def default_data_dir() -> Path:
    return Path.home() / "Library" / "Application Support" / "ProdTrans"


def cmd_run(args) -> None:
    import logging
    import threading
    import webbrowser

    import uvicorn

    from prodtrans.config import SettingsStore
    from prodtrans.controller import Controller
    from prodtrans.events import EventHub
    from prodtrans.server import create_app
    from prodtrans.store import SessionStore

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    data = Path(args.data_dir).expanduser()
    _offline_if_cached("mlx-community/parakeet-tdt-0.6b-v3")

    kwargs = {}
    if args.simulate:
        from prodtrans.audio import FileSource

        files = [Path(p) for p in args.simulate]
        kwargs["make_source"] = lambda s, sr, n: FileSource(files, sr, n_channels=max(n, len(files)))
        kwargs["list_devices"] = lambda: [{"index": 0, "name": "Symulacja (pliki)",
                                           "max_input_channels": len(files), "default_samplerate": 48000.0}]

    hub = EventHub()
    ctl = Controller(SettingsStore(data / "settings.json"), hub, SessionStore(data / "sessions"), **kwargs)
    if ctl.settings.device:
        ctl.start()  # model wczytuje się w tle, podgląd działa od razu

    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '::') else args.host}:{args.port}"
    print(f"ProdTrans: {url}   (ustawienia i nagrania: {data})", flush=True)
    if not args.no_browser:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    try:
        uvicorn.run(create_app(ctl), host=args.host, port=args.port, log_level="warning",
                    timeout_graceful_shutdown=2)
    finally:
        ctl.shutdown()


def cmd_bench(args) -> None:
    from pathlib import Path

    from prodtrans.bench import bench, load_items
    from prodtrans.engines import make_engine

    engines = [make_engine(name.strip()) for name in args.engines.split(",")]
    for e in engines:
        e.load()
    rows = bench(engines, load_items([Path(f) for f in args.files]))
    for r in rows:
        w = "  –  " if r["wer"] is None else f"{r['wer']:.3f}"
        print(f"{r['engine']:<9} {r['file']:<28} {r['ms']:>6} ms  WER {w}  {r['text']}")
    print()
    for e in engines:
        mine = [r for r in rows if r["engine"] == e.name]
        wers = [r["wer"] for r in mine if r["wer"] is not None]
        avg_wer = f"{sum(wers) / len(wers):.3f}" if wers else "–"
        print(f"{e.name:<9} średnio {sum(r['ms'] for r in mine) / len(mine):.0f} ms, WER {avg_wer}")


def _add_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--host", default="127.0.0.1", help="0.0.0.0 = dostęp z innych urządzeń w sieci")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--data-dir", default=str(default_data_dir()), help="gdzie trzymać ustawienia i sesje")
    p.add_argument("--no-browser", action="store_true", help="nie otwieraj przeglądarki")
    p.add_argument("--simulate", nargs="+", metavar="WAV", help="pliki zamiast urządzenia (plik i = wejście i)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="prodtrans", description="Transkrypcja kanałów na żywo po polsku")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("devices", help="wypisz urządzenia wejściowe").set_defaults(func=cmd_devices)
    r = sub.add_parser("run", help="uruchom aplikację (domyślne polecenie)")
    _add_run_args(r)
    r.set_defaults(func=cmd_run)
    b = sub.add_parser("bench", help="porównaj silniki na plikach wav (wzorzec w pliku .txt obok)")
    b.add_argument("--engines", default="parakeet")
    b.add_argument("files", nargs="+")
    b.set_defaults(func=cmd_bench)
    return p


def main(argv: list[str] | None = None) -> None:
    import sys

    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0].startswith("-") and argv[0] not in ("-h", "--help"):
        argv = ["run", *argv]  # samo `prodtrans` uruchamia aplikację
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
