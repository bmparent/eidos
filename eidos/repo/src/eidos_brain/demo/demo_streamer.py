"""
demo_streamer.py

Generates a live NDJSON stream (via HTTP) or WebSocket events for testing.
"""

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio
import json
import time
import random
import math

app = FastAPI()

def generate_sine_wave(t):
    return [math.sin(t) + random.gauss(0, 0.1) for _ in range(64)]

def generate_anomaly(t):
    return [10.0 + random.random() for _ in range(64)]

async def data_stream():
    t = 0
    while True:
        await asyncio.sleep(0.1) # 10Hz
        t += 0.1
        
        # Inject anomaly every 10 seconds
        if int(t) % 10 == 0:
            vec = generate_anomaly(t)
        else:
            vec = generate_sine_wave(t)
            
        payload = {"ts": time.time(), "frame": vec}
        yield json.dumps(payload) + "\n"

@app.get("/stream")
async def stream():
    return StreamingResponse(data_stream(), media_type="application/x-ndjson")

def main():
    import uvicorn
    print("Starting Demo Streamer on port 8001...")
    uvicorn.run(app, host="0.0.0.0", port=8001)

if __name__ == "__main__":
    main()
