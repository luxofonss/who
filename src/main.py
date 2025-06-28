from __future__ import annotations

import sys
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Ensure the src directory (this file's parent) is on sys.path so that
# `import utils.*` works regardless of PYTHONPATH environment variable.
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from fastapi import FastAPI

from utils.logger import init_logger
from api.project import router as project_router
from api.analyze import router as analyze_router
from api.chat import router as chat_router

# Initialise logging at import time
logger = init_logger()

app = FastAPI(title="Code Analyzer Demo")

# Register route groups
app.include_router(project_router)
app.include_router(analyze_router)
app.include_router(chat_router)


@app.get("/health")
async def health():
    return {"status": "ok"} 