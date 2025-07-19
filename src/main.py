from __future__ import annotations

import json
import sys
from pathlib import Path
from asyncio import sleep

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Ensure the src directory (this file's parent) is on sys.path so that
# `import utils.*` works regardless of PYTHONPATH environment variable.
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from utils.logger import init_logger
from api.project import router as project_router
from api.analyze import router as analyze_router
from api.threads import router as threads_router

# Initialise logging at import time
logger = init_logger()

app = FastAPI(title="Code Analyzer Demo")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Register route groups
app.include_router(project_router)
app.include_router(analyze_router)
app.include_router(threads_router)


@app.get("/health")
async def health():
    return {"status": "ok"} 

async def waypoints_generator():
    waypoints = open('waypoints.json')
    waypoints = json.load(waypoints)
    for waypoint in waypoints[0: 10]:
        data = json.dumps(waypoint)
        yield f"event: locationUpdate\ndata: {data}\n\n"
        await sleep(1)

@app.get("/get-waypoints")
async def root():
    return StreamingResponse(waypoints_generator(), media_type="text/event-stream")
