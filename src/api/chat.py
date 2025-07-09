from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from loguru import logger

from services.chat_chain import ChatChain
from utils.file import read_json, write_json, ensure_dir

router = APIRouter(tags=["chat"])

STORAGE_DIR = Path("storage")


@router.post("/chat")
async def chat(project_id: str, message: str):
    """Conversational endpoint that is aware of the codebase using LangGraph."""
    # Validate project exists
    if not (STORAGE_DIR / "metadata" / f"{project_id}.json").exists():
        raise HTTPException(status_code=404, detail="Project not found")

    logger.info(f"💬 Chat request for project {project_id}: {message[:100]}...")

    # Load chat history
    history_path = STORAGE_DIR / "chat_memory" / f"{project_id}.json"
    history = read_json(history_path, default=[]) or []

    # Use ChatChain with LangGraph
    chat_chain = ChatChain(project_id)
    result = await chat_chain.chat(message, history)

    # Update chat history
    history.extend([f"User: {message}", f"AI: {result.response}"])
    ensure_dir(history_path.parent)
    write_json(history_path, history)

    logger.info(f"✅ Chat completed using {result.method} method with {result.iteration_count} iterations")

    return {
        "ai_response": result.response,
        "method": result.method,
        "iterations": result.iteration_count,
        "symbols_retrieved": result.symbols_retrieved
    } 