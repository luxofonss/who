from __future__ import annotations

"""Hybrid Retriever: FAISS similarity + dependency graph traversal."""

from pathlib import Path
from typing import List, Dict, Set

from langchain.schema import Document
from loguru import logger

from services.indexer import Indexer
from services.embedder import embed
from utils.file import read_json

STORAGE_DIR = Path("storage")


class LangChainRetriever:
    """Hybrid Retriever: metadata + FAISS + dependency traversal."""

    def __init__(self, project_id: str, *, k: int = 12):
        self.project_id = project_id
        self.k = k
        self.indexer = Indexer(project_id)
        self._loaded = False

        meta_path = STORAGE_DIR / "metadata" / f"{project_id}.json"
        data = read_json(meta_path) or {}
        self.dep_graph: Dict[str, Dict[str, List[str]]] = data.get("dependency_graph", {})

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Query preprocessing & metadata heuristics
    # ------------------------------------------------------------------

    def _preprocess_query(self, query: str) -> str:
        q = query.lower().lstrip("/")
        if q.startswith("api/"):
            q = q[4:]
        parts = q.split("/")
        resource = parts[-1].replace("-", "_")
        return f"get_{resource}"

    def _metadata_candidates(self, resource: str) -> List[Dict]:
        res_lower = resource.lower()
        return [c for c in self.indexer.metadata if res_lower in (c.get("class_name", "").lower()+c.get("method_name", "").lower())]

    # ------------------------------------------------------------------
    async def retrieve(self, query: str) -> List[Document]:
        await self._ensure_loaded()

        processed = self._preprocess_query(query)
        resource = processed.replace("get_", "")

        # try metadata first
        chunks = self._metadata_candidates(resource)

        if not chunks:
            # dynamic k relative to code base size
            dyn_k = min(30, max(self.k, len(self.indexer.metadata)//200))
            query_emb = embed(processed)
            _, chunks = self.indexer.search(query_emb, dyn_k)

        docs: List[Document] = []
        seen: Set[str] = set()

        for c in chunks:
            if "content" not in c:
                continue
            docs.append(Document(page_content=c["content"], metadata=c))
            seen.add(self._key(c))

        for c in list(chunks):
            self._traverse_iterative(c, docs, seen)

        logger.debug("Retriever returned %d docs (seed %d)", len(docs), len(chunks))
        return docs

    # ------------------------------------------------------------------
    def _key(self, chunk: Dict) -> str:
        return f"{chunk.get('class_name')}.{chunk.get('method_name')}"

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
                docs.append(Document(page_content=callee_chunk["content"], metadata=callee_chunk))
                seen.add(callee)
                stack.append(callee_chunk) 