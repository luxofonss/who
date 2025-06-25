from __future__ import annotations

import json
import asyncio
import re
from typing import Dict, List

from loguru import logger

from adapters.gemini import Gemini
from services.retriever import LangChainRetriever
from services.prompt_builder import PromptBuilder


class AnalyzerChain:
    """High-level orchestrator for endpoint analysis with multi-hop retrieval."""

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
        logger.info(f"🔍 Starting AnalyzerChain for: {endpoint}")
        retrieved_docs = []
        seen_symbols = set()
        max_iter = 3
        query = endpoint

        for iteration in range(max_iter):
            logger.info(f"🔁 Iteration {iteration + 1} – retrieving context")

            # Step 1: Retrieve relevant chunks
            docs = await self.retriever.retrieve(query, user_text, top=6, hyde=(iteration > 0))
            new_docs = [doc for doc in docs if doc.page_content not in {d.page_content for d in retrieved_docs}]
            retrieved_docs.extend(new_docs)

            # Step 2: Build context & prompt
            context = "\n\n".join(doc.page_content for doc in retrieved_docs)
            prompt = PromptBuilder.build_analysis_prompt(
                endpoint=endpoint,
                context=context,
                requirements=requirements_txt,
                testcases=testcases_txt,
                user_text=user_text,
            )
            logger.debug(f"📨 Prompt sent to LLM (len={len(prompt)}):\n{prompt[:1000]}...")

            # Step 3: Call LLM in thread to avoid blocking
            resp = await asyncio.to_thread(self.llm.invoke, prompt)

            # Step 4: Try parse JSON result
            try:
                result = json.loads(resp)
                logger.info("✅ LLM returned valid JSON – finishing.")
                return result
            except json.JSONDecodeError:
                logger.warning("⚠️ LLM did not return JSON, checking for missing context...")

                if not self._needs_more_context(resp):
                    logger.info("🛑 LLM appears confident. Returning raw response.")
                    return {"raw": resp, "note": "LLM returned non-JSON but appears confident."}

                # Step 5: Try extract symbols from LLM output
                new_symbols = self._extract_symbols(resp)
                new_symbols = [s for s in new_symbols if s not in seen_symbols]
                if not new_symbols:
                    logger.warning("🔍 No new symbols extracted. Stopping.")
                    return {"raw": resp, "note": "LLM asked for more info but no symbols found."}

                logger.info(f"🔁 LLM requested more info: {new_symbols}")
                seen_symbols.update(new_symbols)

                # Step 6: Retrieve by symbol name
                for symbol in new_symbols:
                    docs = self.retriever.find_by_symbol_name(symbol)
                    new_docs = [doc for doc in docs if doc.page_content not in {d.page_content for d in retrieved_docs}]
                    retrieved_docs.extend(new_docs)

        logger.warning("⛔ Max iteration reached. Returning partial context.")
        return {
            "note": "Max iteration reached. Returning partial result.",
            "raw": context,
        }

    def _needs_more_context(self, llm_output: str) -> bool:
        phrases = [
            "not enough context",
            "i need to see",
            "missing information",
            "not defined",
            "i cannot determine",
            "i don't know",
            "unknown method",
        ]
        llm_output_lower = llm_output.lower()
        return any(p in llm_output_lower for p in phrases)

    def _extract_symbols(self, llm_output: str) -> List[str]:
        # Heuristic: Look for things like ClassName.methodName or variable.method()
        matches = re.findall(r"([A-Z][a-zA-Z0-9_]+\.[a-zA-Z0-9_]+)", llm_output)
        # Also support getXyz(), doSomething()
        matches += re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]{2,})\s*\(", llm_output)
        return list(set(matches))
