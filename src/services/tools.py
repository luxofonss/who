from __future__ import annotations

from functools import lru_cache
from langchain_core.tools import tool
from services.retriever import LangChainRetriever


@lru_cache(maxsize=4)
def _get_retriever(project_id: str) -> LangChainRetriever:
    return LangChainRetriever(project_id)


@tool
def get_code_context(symbol: str, project_id: str) -> str:  # type: ignore[override]
    """Return code (summary + content) related to *symbol* (Class.method) from the given project.

    Usage example inside the agent:
        get_code_context{"symbol": "UserService.getUserById", "project_id": "demo"}
    """
    retriever = _get_retriever(project_id)
    docs = retriever.find_by_symbol_name(symbol)
    return "\n\n".join(d.page_content for d in docs) if docs else "No code found for symbol"
