from __future__ import annotations

import json
from typing import Dict

from loguru import logger

from adapters.gemini import Gemini
from services.retriever import LangChainRetriever
from services.prompt_builder import PromptBuilder


class AnalyzerChain:
    """High-level orchestrator for endpoint analysis."""

    def __init__(self, project_id: str):
        self.retriever = LangChainRetriever(project_id)
        self.llm = Gemini()

    async def run(
        self,
        *,
        endpoint: str,
        requirements_txt: str,
        testcases_txt: str,
        user_text: str,
    ) -> Dict:
        # Retrieve code context
        logger.info(f"Retrieving context for endpoint: {endpoint}")
        logger.info(f"User text: {user_text}")

        # TODO: This is for test only
        # docs = await self.retriever.retrieve(endpoint, user_text, top=3)
        # docs = await self.retriever.retrieve(docs, user_text, top=7)
        # docs = await self.retriever.retrieve(docs, user_text, top=13)

        docs = await self.retriever.retrieve(endpoint, user_text, top=3)

        # Extract text from documents
        doc_text = "\n".join(doc.page_content for doc in docs)
        logger.info(f"Retrieved: {doc_text}")

        docs = docs + await self.retriever.retrieve(doc_text, user_text, top=30, hyde=True)
        logger.info(f"Retrieved: {doc_text}")

        context = "\n".join(d.page_content for d in docs)

        logger.info(f"Retrieved context: {context}")

        prompt = PromptBuilder.build_analysis_prompt(
            endpoint=endpoint,
            context=context,
            requirements=requirements_txt,
            testcases=testcases_txt,
            user_text=user_text,
        )
        logger.info(f"Prompt: {prompt}")
        return {}

        # Call LLM (synchronously)
        # Offload to thread to avoid blocking event loop
        # import asyncio
        # resp = await asyncio.to_thread(self.llm.invoke, prompt)
        # try:
        #     data = json.loads(resp)
        # except json.JSONDecodeError:
        #     logger.error(f"LLM returned non-JSON; wrapping into error structure")
        #     data = {
        #         "documentation": "LLM response was not valid JSON",
        #         "raw": resp,
        #     }
        # return data 