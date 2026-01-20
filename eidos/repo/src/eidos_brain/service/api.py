"""
api.py

FastAPI Interface for Eidos Brain.
Provides:
- /ingest (push frames)
- /status (health)
- /stream (monitor)
"""

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional
import asyncio
import json
import logging
from datetime import datetime

# Import engine adapter
from eidos_brain.engine.adapters import run_session
from eidos_brain.utils.config import load_config
from eidos_brain.io.hive_event import HiveEventV1

app = FastAPI(title="Eidos Brain API", version="0.4.7")
logger = logging.getLogger("eidos-api")

# Global State
class ServiceState:
    is_running = False
    last_run_status = {}
    current_session = None

state = ServiceState()

@app.get("/healthz")
async def healthz():
    return {"ok": True, "ts": datetime.utcnow().isoformat()}

@app.get("/status")
async def status():
    return {
        "running": state.is_running,
        "last_status": state.last_run_status,
        "current_session": state.current_session
    }

class IngestRequest(BaseModel):
    event_type: str
    payload: Dict[str, Any]

@app.post("/ingest")
async def ingest(item: IngestRequest):
    """
    Accept an event/frame.
    In a real implementation, this would push to a queue that the engine reads from.
    For this 'minimal edit' version, we log it or push to a file that LOCAL engine watches?
    Or if engine is STREAM mode, we might push to the socket it listens to.
    """
    # TODO: Implement queue logic
    return {"status": "accepted", "id": "mock_id"}

def _background_run(config: dict):
    state.is_running = True
    try:
        res = run_session(config)
        state.last_run_status = res
    finally:
        state.is_running = False

@app.post("/run")
async def trigger_run(background_tasks: BackgroundTasks):
    if state.is_running:
        return JSONResponse(status_code=409, content={"error": "Busy"})
    
    config = load_config()
    background_tasks.add_task(_background_run, config)
    return {"status": "started"}

async def _event_generator():
    """Mock stream of events for demo"""
    while True:
        # In real world, subscribe to engine output queue
        await asyncio.sleep(1)
        evt = HiveEventV1.create("eidos.heartbeat", "system", {"status": "ok"})
        yield f"data: {evt.to_jsonl_line()}\n\n"

@app.get("/stream")
async def stream():
    return StreamingResponse(_event_generator(), media_type="text/event-stream")

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
