from __future__ import annotations

import re
import asyncio

from pathlib import Path
from typing import List, Dict, Set

from langchain.schema import Document
from loguru import logger

from services.indexer import Indexer
from services.embedder import embed
from utils.file import read_json
from adapters.gemini import Gemini

from services.bm25 import bm25_search
from services.reranker import rerank_chunks

from utils.prompt import HYDE_V2_SYSTEM_PROMPT

STORAGE_DIR = Path("storage")


class LangChainRetriever:
    def __init__(self, project_id: str, *, k: int = 12):
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

    async def retrieve(self, query: str, user_text: str) -> List[Document]:
        await self._ensure_loaded()
        processed = self._preprocess_query(query)

        logger.debug(f"Retriever query='{query}' processed='{processed}'")

        # HyDE: Use LLM to generate a hypothetical answer
        hyde_prompt = HYDE_V2_SYSTEM_PROMPT.format(query=query, temp_context=user_text)
        hyde_answer = Gemini().invoke(hyde_prompt)
        logger.info(f"HyDE answer: {hyde_answer}")  

        query_emb = embed(hyde_answer)

        # Hybrid search: BM25 and FAISS
        bm25_task = asyncio.to_thread(bm25_search, processed, self.indexer.metadata, top_k=self.k)
        faiss_task = asyncio.to_thread(self.indexer.search, query_emb, self.k)
        bm25_results, faiss_results = await asyncio.gather(bm25_task, faiss_task)
        faiss_results = faiss_results[1]

        all_candidates = bm25_results + faiss_results
        logger.debug(f"Total candidates before reranking: {len(all_candidates)}")

        # Re-rank using cross-encoder (you implement rerank_chunks)
        top_chunks = rerank_chunks(query, all_candidates, top_k=self.k)

        docs: List[Document] = []
        seen: Set[str] = set()

        for c in top_chunks:
            logger.info(f"Retriever candidate: {c}")
            if "content" not in c:
                continue
            full_text = f"# Summary: {c.get('summary', '')}\n\n{c['content']}"
            docs.append(Document(page_content=full_text, metadata=c))
            seen.add(self._key(c))

        for c in list(top_chunks):
            self._traverse_iterative(c, docs, seen)

        trimmed_docs = self._trim_overlaps(docs)
        logger.info(f"Retriever finished – returned {len(trimmed_docs)} trimmed docs")
        return trimmed_docs

    def _traverse_iterative(self, seed: Dict, docs: List[Document], seen: Set[str]):
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

    def _preprocess_query(self, query: str) -> str:
        method_match = re.search(r"@method=(GET|POST|PUT|DELETE|PATCH)", query, re.IGNORECASE)
        endpoint_match = re.search(r"@endpoint=([^\s]+)", query)

        if not method_match or not endpoint_match:
            raise ValueError(f"Invalid query format: {query}")

        method = method_match.group(1).lower()
        endpoint = endpoint_match.group(1).strip().lower()

        endpoint = endpoint.lstrip("/")
        if endpoint.startswith("api/"):
            endpoint = endpoint[4:]

        return f"{method} {endpoint}"

