from __future__ import annotations

import re
import asyncio
from pathlib import Path
from typing import List, Dict, Set

from langchain_core.documents import Document
from loguru import logger

from services.indexer import Indexer
from services.embedder import embed
from utils.file import read_json
from adapters.gemini import Gemini
from services.bm25 import bm25_search
from services.reranker import rerank_chunks
from utils.prompt import HYDE_ENHANCE_CODE_DEPENDENCIES

STORAGE_DIR = Path("storage")


class LangChainRetriever:
    def __init__(self, project_id: str, *, k: int = 100):
        self.project_id = project_id
        self.k = k
        self.indexer = Indexer(project_id)
        self._loaded = False

        meta_path = STORAGE_DIR / "metadata" / f"{project_id}.json"
        data = read_json(meta_path) or {}
        self.dep_graph: Dict[str, Dict[str, List[str]]] = data.get("dependency_graph", {})

    async def _ensure_loaded(self):
        if self._loaded:
            return
        await self.indexer.load()
        self._chunk_lookup: Dict[str, Dict] = {
            f"{c['class_name']}.{c.get('method_name')}": c
            for c in self.indexer.metadata
            if c.get("method_name")
        }
        self._loaded = True

    def _key(self, chunk: Dict) -> str:
        return f"{chunk.get('class_name')}.{chunk.get('method_name')}"

    async def retrieve(self, query: str, user_text: str, top: int, hyde: bool = False) -> List[Document]:
        await self._ensure_loaded()

        # 1. HyDE prompt enhancement (optional)
        hyde_prompt = query
        if hyde:
            hyde_prompt = HYDE_ENHANCE_CODE_DEPENDENCIES.format(query=query)
            hyde_answer = Gemini().invoke(hyde_prompt)
            logger.info(f"HyDE answer: {hyde_answer}")
            hyde_prompt += hyde_answer

        # 2. Embedding
        query_emb = embed(hyde_prompt + query)

        # 3. Hybrid retrieval
        bm25_task = asyncio.to_thread(bm25_search, query, self.indexer.metadata, top_k=top)
        faiss_task = asyncio.to_thread(self.indexer.search, query_emb, top)
        bm25_results, faiss_results = await asyncio.gather(bm25_task, faiss_task)
        faiss_results = faiss_results[1]  # Just chunks

        # 4. Combine & rerank
        all_candidates = bm25_results + faiss_results
        top_chunks = rerank_chunks(query, all_candidates, top_k=top)

        docs: List[Document] = []
        seen: Set[str] = set()

        for c in top_chunks:
            if "content" not in c:
                continue
            key = self._key(c)
            full_text = f"# Summary: {c.get('summary', '')}\n\n{c['content']}"
            docs.append(Document(page_content=full_text, metadata=c))
            seen.add(key)

            # ✅ Traverse call graph to pull transitive callee dependencies
            self._traverse_call_graph(c, docs, seen)

        trimmed_docs = self._trim_overlaps(docs)
        logger.info(f"Retriever finished – returned {len(trimmed_docs)} trimmed docs")
        return trimmed_docs

    def _traverse_call_graph(self, seed: Dict, docs: List[Document], seen: Set[str]):
        """Recursively traverse transitive calls to include full logic."""
        stack = [seed]
        while stack:
            c = stack.pop()
            key = self._key(c)
            for callee in self.dep_graph.get(key, {}).get("calls", []):
                if callee in seen:
                    continue
                callee_chunk = self._chunk_lookup.get(callee)
                if not callee_chunk or "content" not in callee_chunk:
                    continue
                full_text = f"# Summary: {callee_chunk.get('summary', '')}\n\n{callee_chunk['content']}"
                docs.append(Document(page_content=full_text, metadata=callee_chunk))
                seen.add(callee)
                stack.append(callee_chunk)

    def _trim_overlaps(self, docs: List[Document]) -> List[Document]:
        seen_lines = set()
        trimmed = []
        for doc in docs:
            lines = doc.page_content.splitlines()
            unique = [l for l in lines if l.strip() not in seen_lines]
            seen_lines.update(l.strip() for l in unique)
            if unique:
                doc.page_content = "\n".join(unique)
                trimmed.append(doc)
        return trimmed

    def find_by_symbol_name(self, symbol: str) -> List[Document]:
        """Optional external call: retrieve chunk(s) from method/class name"""
        found = []
        logger.info(f"🔍 Finding by symbol name: {symbol}")
        logger.info(f"🔍 Chunk lookup: {self._chunk_lookup}")
        for key, chunk in self._chunk_lookup.items():
            if symbol.lower() in key.lower():
                full_text = f"# Summary: {chunk.get('summary', '')}\n\n{chunk['content']}"
                found.append(Document(page_content=full_text, metadata=chunk))
        return found

    # ------------------------------------------------------------------
    # Convenience sync wrapper for LangChain tools / agents
    # ------------------------------------------------------------------

    def retrieve_sync(self, query: str, user_text: str = "", top: int = 5, hyde: bool = False) -> List[Document]:
        """Blocking wrapper around *retrieve* for non-async callers.

        If called from within a running event loop it schedules the task and
        waits; otherwise it spins up a new loop with ``asyncio.run``.
        """
        import asyncio

        coro = self.retrieve(query, user_text, top, hyde)

        try:
            return asyncio.run(coro)
        except RuntimeError:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(coro)
