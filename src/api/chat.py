from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from services.retriever import LangChainRetriever
from services.prompt_builder import PromptBuilder
from adapters.gemini import Gemini
from utils.file import read_json, write_json, ensure_dir

router = APIRouter(tags=["chat"])

STORAGE_DIR = Path("storage")


@router.post("/chat")
async def chat(project_id: str, message: str):
    """Conversational endpoint that is aware of the codebase."""
    # Validate project exists
    if not (STORAGE_DIR / "metadata" / f"{project_id}.json").exists():
        raise HTTPException(status_code=404, detail="Project not found")

    history_path = STORAGE_DIR / "chat_memory" / f"{project_id}.json"
    history: list[str] = read_json(history_path, default=[])

    retriever = LangChainRetriever(project_id)
    docs = retriever.retrieve(message)
    context = "\n".join(d.page_content for d in docs)

    prompt = PromptBuilder.build_chat_prompt(history="\n".join(history), context=context, message=message)
    llm = Gemini()
    ai_response = llm.invoke(prompt)

    # Update chat history
    history.extend([f"User: {message}", f"AI: {ai_response}"])
    ensure_dir(history_path.parent)
    write_json(history_path, history)

    return {"ai_response": ai_response} 