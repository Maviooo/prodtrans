"""Serwer HTTP/WebSocket: podgląd na żywo i konfiguracja z przeglądarki."""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles

from prodtrans.config import MAX_CHANNELS, PALETTE, ConfigError, settings_to_dict
from prodtrans.controller import Controller

WEB_DIR = Path(__file__).parent / "web"


def available_engines() -> list[dict]:
    return [
        {"id": "parakeet", "label": "Parakeet v3", "hint": "Szybki, zalecany", "available": True},
        {"id": "qwen", "label": "Qwen3-ASR", "hint": "Wolniejszy, do porównań",
         "available": importlib.util.find_spec("qwen_asr") is not None},
    ]


def create_app(ctl: Controller) -> FastAPI:
    app = FastAPI(title="ProdTrans")
    hub = ctl.hub

    def hello() -> dict:
        return {
            "type": "hello",
            "channels": [{"id": c.id, "name": c.name, "color": c.color}
                         for c in ctl.settings.enabled_channels],
            "history": hub.history(),
            "status": hub.statuses(),
            "running": ctl.running,
        }

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        q = hub.subscribe()
        try:
            await ws.send_json(hello())
            while True:
                event = await q.get()
                await ws.send_json(hello() if event["type"] == "resync" else event)
        except (WebSocketDisconnect, asyncio.CancelledError, RuntimeError):
            pass
        finally:
            hub.unsubscribe(q)

    @app.get("/api/settings")
    def get_settings():
        return {
            "settings": settings_to_dict(ctl.settings),
            "engines": available_engines(),
            "palette": PALETTE,
            "max_channels": MAX_CHANNELS,
        }

    @app.put("/api/settings")
    def put_settings(data: dict = Body(...)):
        try:
            return {"settings": settings_to_dict(ctl.apply(data))}
        except ConfigError as e:
            raise HTTPException(status_code=422, detail=str(e))

    @app.get("/api/devices")
    def get_devices(refresh: bool = False):
        return {"devices": ctl.devices(refresh=refresh)}

    @app.get("/api/status")
    def get_status():
        return ctl.status()

    @app.post("/api/start")
    def start():
        ctl.start()
        hub.resync()
        return ctl.status()

    @app.post("/api/stop")
    def stop():
        ctl.stop()
        hub.resync()
        return ctl.status()

    @app.get("/api/export")
    def export():
        name = f"transkrypcja_{ctl.sessions.dir.name}.txt"
        return PlainTextResponse(
            ctl.sessions.txt_path.read_text(encoding="utf-8"),
            headers={"Content-Disposition": f'attachment; filename="{name}"'},
        )

    @app.middleware("http")
    async def no_cache(request, call_next):
        # Po aktualizacji aplikacji przeglądarka ma od razu dostać nowy interfejs.
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-cache"
        return response

    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app
