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

    def _ensure_loaded_sync(self):
        if self._loaded:
            return

        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                logger.warning("Cannot load index synchronously from within async context")
                return
            else:
                asyncio.run(self._ensure_loaded())
        except RuntimeError:
            # No event loop, create one
            asyncio.run(self._ensure_loaded())

    def _key(self, chunk: Dict) -> str:
        return f"{chunk.get('class_name')}.{chunk.get('method_name')}"

    def _chunk_id(self, chunk: Dict) -> str:
        class_name = chunk.get('class_name', '')
        method_name = chunk.get('method_name', '')
        file_path = chunk.get('file_path', '')
        start_line = chunk.get('start_line', 0)
        
        # Create a unique ID combining these fields
        return f"{file_path}::{class_name}::{method_name}::{start_line}"

    def _deduplicate_chunks(self, chunks: List[Dict]) -> List[Dict]:
        if not chunks:
            return chunks
            
        seen_ids = set()
        unique_chunks = []
        duplicates_count = 0
        
        for chunk in chunks:
            chunk_id = self._chunk_id(chunk)
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                unique_chunks.append(chunk)
            else:
                duplicates_count += 1
        
        if duplicates_count > 0:
            logger.debug(f"🔄 Removed {duplicates_count} duplicate chunks")
            
        return unique_chunks

    def _deduplicate_documents(self, docs: List[Document]) -> List[Document]:
        if not docs:
            return docs
            
        seen_ids = set()
        unique_docs = []
        duplicates_count = 0
        
        for doc in docs:
            # Use chunk metadata to create document ID
            chunk_id = self._chunk_id(doc.metadata)
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                unique_docs.append(doc)
            else:
                duplicates_count += 1
        
        if duplicates_count > 0:
            logger.debug(f"🔄 Removed {duplicates_count} duplicate documents")
            
        return unique_docs

    async def retrieve(self, query: str, user_text: str, top: int, hyde: bool = False) -> List[Document]:
        await self._ensure_loaded()
        # 2. Embedding
        query_emb = embed(query)

        # 3. Hybrid retrieval
        bm25_task = asyncio.to_thread(bm25_search, query, self.indexer.metadata, top_k=top)
        faiss_task = asyncio.to_thread(self.indexer.search, query_emb, top)
        bm25_results, faiss_results_tuple = await asyncio.gather(bm25_task, faiss_task)

        # Extract chunks from FAISS results tuple (distances, chunks)
        faiss_results = faiss_results_tuple[1]  # Just chunks
        logger.debug(f"BM25 search returned {len(bm25_results)} chunks")
        logger.debug(f"FAISS search returned {len(faiss_results)} chunks")

        # 4. Deduplicate and combine results
        combined_results = bm25_results + faiss_results
        logger.debug(f"Combined total: {len(combined_results)} chunks")
        
        all_candidates = self._deduplicate_chunks(combined_results)
        logger.debug(f"After deduplication: {len(all_candidates)} unique chunks")
        
        # 5. Rerank deduplicated results
        top_chunks = rerank_chunks(query, all_candidates, top_k=top)

        docs: List[Document] = []
        seen: Set[str] = set()

        for c in top_chunks:
            if "content" not in c:
                continue
            key = self._key(c)
            if key in seen:
                continue  # Skip if we've already processed this chunk
                
            full_text = f"# Summary: {c.get('summary', '')}\n\n{c['content']}"
            docs.append(Document(page_content=full_text, metadata=c))
            seen.add(key)

            # ✅ Traverse call graph to pull transitive callee dependencies
            self._traverse_call_graph(c, docs, seen)

        # Final deduplication and cleanup
        logger.debug(f"Before final deduplication: {len(docs)} documents")
        deduplicated_docs = self._deduplicate_documents(docs)
        logger.debug(f"After document deduplication: {len(deduplicated_docs)} documents")
        
        trimmed_docs = self._trim_overlaps(deduplicated_docs)
        logger.debug(f"After overlap trimming: {len(trimmed_docs)} documents")
        
        logger.info(f"✅ Retrieval complete: {len(trimmed_docs)} unique documents (removed duplicates and overlaps)")
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
        """Remove overlapping lines between documents to reduce redundancy."""
        if not docs:
            return docs
            
        seen_lines = set()
        trimmed = []
        total_lines_before = 0
        total_lines_after = 0
        
        for doc in docs:
            lines = doc.page_content.splitlines()
            total_lines_before += len(lines)
            
            # Keep lines that haven't been seen before
            unique = [l for l in lines if l.strip() and l.strip() not in seen_lines]
            seen_lines.update(l.strip() for l in unique if l.strip())
            
            if unique:
                doc.page_content = "\n".join(unique)
                trimmed.append(doc)
                total_lines_after += len(unique)
        
        lines_removed = total_lines_before - total_lines_after
        if lines_removed > 0:
            logger.debug(f"🧹 Trimmed {lines_removed} overlapping lines from {len(docs)} documents")
            
        return trimmed

    def find_by_symbol_name(self, symbol: str) -> List[Document]:
        """Optional external call: retrieve chunk(s) from method/class name"""
        # Ensure the index and chunk lookup are loaded
        if not self._loaded:
            logger.debug(f"Loading index for symbol search: {symbol}")
            self._ensure_loaded_sync()
            
        if not self._loaded:
            logger.warning(f"Failed to load index for find_by_symbol_name, returning empty results")
            return []
            
        found = []
        logger.debug(f"🔍 Finding by symbol name: {symbol}")
        
        if not hasattr(self, '_chunk_lookup'):
            logger.warning(f"Chunk lookup not available, returning empty results")
            return []
            
        for key, chunk in self._chunk_lookup.items():
            if symbol.lower() in key.lower():
                full_text = f"# Summary: {chunk.get('summary', '')}\n\n{chunk['content']}"
                found.append(Document(page_content=full_text, metadata=chunk))
                
        logger.debug(f"Found {len(found)} matches for symbol: {symbol}")
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
