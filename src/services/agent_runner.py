from __future__ import annotations

from functools import lru_cache
from typing import List

from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage

from adapters.gemini import LangChainGemini
from services.retriever import LangChainRetriever
from services.tools import get_code_context


@lru_cache(maxsize=4)
def _agent_for_project(project_id: str):
    """Create (and cache) a LangChain agent wired with initial RAG context."""
    retriever = LangChainRetriever(project_id)
    top_docs = retriever.retrieve_sync("", top=5, user_text="initial")  # blank query returns nothing meaningful but shows flow
    context_text = "\n\n".join(d.page_content for d in top_docs)

    prefix = (
        "You are a senior Java engineer. You have access to a `get_code_context` tool "
        "that can fetch classes/methods on demand.  START with the context below, "
        "but feel free to call the tool for missing pieces.\n\n" + context_text
    )

    llm = LangChainGemini(temperature=0)

    tools = [get_code_context]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        memory=ConversationBufferMemory(memory_key="chat_history"),
        agent_kwargs={"prefix": prefix},
    )
    return agent


def run_agent(project_id: str, user_message: str) -> str:
    agent = _agent_for_project(project_id)
    return agent.run(user_message)
