import asyncio
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from artifacts import ArtifactStore
from engine import EidosLifeEngine
from schemas import CommandRequest


app = FastAPI(title="Eidos Life Lab", version="0.1.0-local")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:8787", "http://127.0.0.1:8787"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = EidosLifeEngine()
artifacts = ArtifactStore()
engine_lock = asyncio.Lock()


class ConnectionManager:
    def __init__(self) -> None:
        self._clients: Dict[WebSocket, asyncio.Lock] = {}

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._clients[websocket] = asyncio.Lock()

    def disconnect(self, websocket: WebSocket) -> None:
        self._clients.pop(websocket, None)

    async def send_json(self, websocket: WebSocket, payload: Dict[str, Any]) -> None:
        lock = self._clients.get(websocket)
        if lock is None:
            return
        async with lock:
            await websocket.send_json(payload)

    async def broadcast(self, payload: Dict[str, Any]) -> None:
        stale = []
        for websocket in list(self._clients.keys()):
            try:
                await self.send_json(websocket, payload)
            except Exception:
                stale.append(websocket)
        for websocket in stale:
            self.disconnect(websocket)


manager = ConnectionManager()
_simulation_task: asyncio.Task | None = None
_last_auto_checkpoint_generation = 0


@app.on_event("startup")
async def startup() -> None:
    global _simulation_task
    _simulation_task = asyncio.create_task(simulation_loop())


@app.on_event("shutdown")
async def shutdown() -> None:
    if _simulation_task:
        _simulation_task.cancel()


async def simulation_loop() -> None:
    global _last_auto_checkpoint_generation
    while True:
        fps = int(engine.settings.get("broadcastFps", 12))
        interval = 1.0 / max(1, fps)
        snapshot = None
        async with engine_lock:
            if engine.settings.get("running"):
                speed = int(engine.settings.get("speed", 1))
                engine.step(speed)
                checkpoint_interval = int(engine.settings.get("checkpointInterval", 5000))
                if checkpoint_interval > 0 and engine.generation and engine.generation % checkpoint_interval == 0:
                    if engine.generation != _last_auto_checkpoint_generation:
                        artifacts.save_checkpoint(engine.full_state())
                        _last_auto_checkpoint_generation = engine.generation
                snapshot = engine.snapshot()
        if snapshot is not None:
            await manager.broadcast({"type": "snapshot", "snapshot": snapshot})
        await asyncio.sleep(interval)


def handle_artifact_command(command: str) -> Dict[str, Any]:
    if command == "export":
        return artifacts.save_export(engine.full_state())
    if command == "checkpoint":
        return artifacts.save_checkpoint(engine.full_state())
    raise ValueError(f"Not an artifact command: {command}")


@app.get("/api/state")
async def state() -> Dict[str, Any]:
    async with engine_lock:
        return engine.snapshot()


@app.post("/api/command")
async def command(request: CommandRequest) -> Dict[str, Any]:
    payload = request.to_payload()
    command_name = payload.get("command")
    try:
        async with engine_lock:
            if command_name in {"export", "checkpoint"}:
                result = handle_artifact_command(command_name)
                snapshot = engine.snapshot()
            else:
                snapshot = engine.apply_command(payload)
                result = {"command": command_name}
        await manager.broadcast({"type": "snapshot", "snapshot": snapshot})
        return {"ok": True, "result": result, "snapshot": snapshot}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/export")
async def export_state() -> Dict[str, Any]:
    async with engine_lock:
        result = artifacts.save_export(engine.full_state())
    return {"ok": True, "result": result}


@app.post("/api/checkpoint")
async def checkpoint_state() -> Dict[str, Any]:
    async with engine_lock:
        result = artifacts.save_checkpoint(engine.full_state())
    return {"ok": True, "result": result}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await manager.connect(websocket)
    try:
        async with engine_lock:
            await manager.send_json(websocket, {"type": "snapshot", "snapshot": engine.snapshot()})
        while True:
            payload = await websocket.receive_json()
            command_name = payload.get("command")
            async with engine_lock:
                if command_name in {"export", "checkpoint"}:
                    result = handle_artifact_command(command_name)
                    snapshot = engine.snapshot()
                else:
                    snapshot = engine.apply_command(payload)
                    result = {"command": command_name}
            await manager.send_json(websocket, {"type": "ack", "ok": True, "result": result})
            await manager.broadcast({"type": "snapshot", "snapshot": snapshot})
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as exc:
        await manager.send_json(websocket, {"type": "error", "ok": False, "detail": str(exc)})
        manager.disconnect(websocket)
