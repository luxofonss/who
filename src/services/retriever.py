from __future__ import annotations

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
from utils.prompt import HYDE_ENHANCE_CODE_DEPENDENCIES
from rank_bm25 import BM25Okapi

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

    async def retrieve(self, query: str, top: int, hyde: bool = False, method: str | None = None) -> List[Document]:
        await self._ensure_loaded()
        # turn raw query to embedded query
        logger.info(f"query: {query}")
        
        # Apply HyDE if enabled
        if hyde:
            logger.info("Applying HyDE enhancement")
            gemini = Gemini()
            hyde_prompt = HYDE_ENHANCE_CODE_DEPENDENCIES.format(query=query)
            hyde_response = await asyncio.to_thread(gemini.invoke, hyde_prompt)
            enhanced_query = f"{query}\n\n{hyde_response}"
            logger.info(f"HyDE enhanced query: {enhanced_query}")
            query_emb = embed(enhanced_query)
        else:
            query_emb = embed(query)

        # hybrid retrieval using bm25 and faiss (get scores for both)
        bm25_task = asyncio.to_thread(self._bm25_with_scores, query, self.indexer.metadata)
        faiss_task = asyncio.to_thread(self.indexer.search, query_emb, self.k)
        
        bm25_scored, faiss_results_tuple = await asyncio.gather(bm25_task, faiss_task) # async
        
        faiss_distances, faiss_chunks = faiss_results_tuple
        logger.info(f"first index 0 faiss_results_tuple")
        # Convert FAISS distances to similarity scores (lower distance = higher score)
        logger.info(f"faiss_distances: {faiss_distances}")
        if len(faiss_distances) > 0:
            faiss_sim = 1 / (1 + faiss_distances)
        else:
            faiss_sim = []
        faiss_dict = {}
        for chunk, score in zip(faiss_chunks, faiss_sim):
            faiss_dict[chunk.get("id")] = score

        # BM25 scores
        bm25_dict = {}
        for chunk, score in bm25_scored:
            bm25_dict[chunk.get("id")] = score

        # Normalize scores
        def normalize(scores):
            if not scores:
                return {}
            values = list(scores.values())
            min_s, max_s = min(values), max(values)
            diff = max_s - min_s
            if diff == 0:  # All scores are the same
                return {k: 1.0 for k in scores}
            return {k: (v - min_s) / diff for k, v in scores.items()}

        norm_faiss = normalize(faiss_dict)
        norm_bm25 = normalize(bm25_dict)

        # Combine all unique chunk_ids
        all_chunk_ids = set(norm_faiss) | set(norm_bm25)
        hybrid_scores = {}
        for cid in all_chunk_ids:
            s_faiss = norm_faiss.get(cid, 0)
            s_bm25 = norm_bm25.get(cid, 0)
            hybrid_scores[cid] = 0.7 * s_faiss + 0.3 * s_bm25

        # Get chunk lookup
        chunk_lookup = {c.get("id"): c for c in faiss_chunks + [c for c, _ in bm25_scored]}
        # Sort by hybrid score
        sorted_chunk_ids = sorted(hybrid_scores, key=lambda cid: -hybrid_scores[cid])[:top]
        selected_chunks = [chunk_lookup[cid] for cid in sorted_chunk_ids if cid in chunk_lookup]

        logger.debug(f"Hybrid (0.7 FAISS + 0.3 BM25) selected {len(selected_chunks)} chunks")
        logger.info(f"selected_chunks: {selected_chunks}")
        all_candidates = self._deduplicate_chunks(selected_chunks)
        logger.debug(f"After deduplication: {len(all_candidates)} unique chunks")

        # 5. Rerank deduplicated results
        top_chunks = all_candidates

        docs: List[Document] = []
        seen: Set[str] = set()

        for c in top_chunks:
            if "content" not in c:
                continue
            key = self._key(c)
            if key in seen:
                continue  # Skip if we've already processed this chunk
            full_text = self._get_full_text(c)
            docs.append(Document(page_content=full_text, metadata=c))
            seen.add(key)
            self._traverse_call_graph(c, docs, seen)

        # Final deduplication and cleanup
        logger.debug(f"Before final deduplication: {len(docs)} documents")
        deduplicated_docs = self._deduplicate_documents(docs)
        logger.debug(f"After document deduplication: {len(deduplicated_docs)} documents")
        return deduplicated_docs

    async def retrieve_endpoints(self, symbols: List[str]) -> List[str]:
        await self._ensure_loaded()
        endpoints = []
        for symbol in symbols:
            endpoints.extend(self.find_endpoint_by_symbol_name(symbol))
        # Remove duplicates based on endpoint path + method combination
        unique_endpoints = []
        seen_combinations = set()
        for endpoint in endpoints:
            combination = (endpoint.get('path', ''), endpoint.get('method', ''))
            if combination not in seen_combinations:
                seen_combinations.add(combination)
                unique_endpoints.append(endpoint)
        
        return unique_endpoints

    def _get_full_text(self, c: Dict) -> str:
        if (c.get("chunk_type") == "method"):
            return f"# Summary: {c.get('class_name', '')}.{c.get('method_name', '')} {c.get('summary', '')}\n\n{c['content']}"
        else:
            return f"# Summary: {c.get('summary', '')}\n\n{c['content']}"

    def _traverse_call_graph(self, seed: Dict, docs: List[Document], seen: Set[str]):
        """Recursively traverse transitive dependencies including calls, inheritance, and interface relations."""
        logger.info(f"Traversing call graph for {seed.get('class_name')}.{seed.get('method_name')}")
        stack = [seed]
        RELATION_TYPES = [
            "calls", 
            "implemented_by", 
            "extended_by",
            "vars"
        ]

        while stack:
            c = stack.pop()

            for rel in RELATION_TYPES:
                relates = c.get(rel, [])
                for related in relates:
                    if related in seen:
                        continue
                    seen.add(related)
                    if seed.get("method_name") and "." not in related:
                        seen.add(related + "." + seed.get("method_name", ""))
                    d = self.find_chunk_by_symbol_name(related)
                    if seed.get("method_name") and "." not in related :
                        d = d + self.find_chunk_by_symbol_name(related + "." + seed.get("method_name", ""))
                    for dx in d:
                        full_text = self._get_full_text(dx)
                        docs.append(Document(page_content=full_text, metadata=dx))
                        stack.append(dx)

    def find_chunk_by_symbol_name(self, symbol: str, method_name: str = '') -> List[Dict]:
        # Ensure the index and chunk lookup are loaded
        if not self._loaded:
            logger.debug(f"Loading index for symbol search: {symbol}")
            self._ensure_loaded_sync()
            
        if not self._loaded:
            logger.warning(f"Failed to load index for find_by_symbol_name, returning empty results")
            return []
            
        found = []
        # logger.debug(f"🔍 Finding by symbol name: {symbol}")
        
        if not hasattr(self, '_chunk_lookup'):
            logger.warning(f"Chunk lookup not available, returning empty results")
            return []
            
        for key, chunk in self._chunk_lookup.items():
            # logger.info(f"key: {key}")
            if symbol.lower() in key.lower():
                found.append(chunk)
                if chunk.get("chunk_type") == "interface":
                    interfaces = chunk.get("implemented_by", [])
                    for interface in interfaces:
                        if method_name:
                            itfs = self.find_chunk_by_symbol_name(interface + "." + method_name)
                        else:
                            itfs = self.find_chunk_by_symbol_name(interface)
                        for itf in itfs:
                            found.append(itf)
                if chunk.get("chunk_type") == "abstract_class":
                    sub_classes = chunk.get("extended_by", [])
                    for sub_class in sub_classes:
                        if method_name:
                            sb_classes = self.find_chunk_by_symbol_name(sub_class + "." + method_name)
                        else:
                            sb_classes = self.find_chunk_by_symbol_name(sub_class)
                        for sb_class in sb_classes:
                            found.append(sb_class)
        # logger.info(f"Found {len(found)} matches for symbol: {symbol}")
        return found

    def find_endpoint_by_symbol_name(self, symbol: str) -> List[Dict[str, str]]:
        if not self._loaded:
            logger.debug(f"Loading index for endpoint search: {symbol}")
            self._ensure_loaded_sync()
            
        if not self._loaded:
            logger.warning(f"Failed to load index for find_endpoint_by_symbol_name, returning empty results")
            return []
            
        if not hasattr(self, '_chunk_lookup'):
            logger.warning(f"Chunk lookup not available, returning empty results")
            return []

        result_endpoints = []
        visited = set()
        
        # Start by finding all chunks that call the target symbol
        calling_chunks = []
        for key, chunk in self._chunk_lookup.items():
            calls = chunk.get('calls', [])
            # Check if any of the calls match our symbol
            for call in calls:
                if symbol in call or call in symbol:
                    calling_chunks.append(chunk)
                    break
        
        logger.debug(f"Found {len(calling_chunks)} chunks calling symbol: {symbol}")
        
        # For each calling chunk, traverse backward to find endpoints
        for chunk in calling_chunks:
            self._traverse_to_endpoints(chunk, symbol, visited, result_endpoints)
        
        return result_endpoints
    
    def _traverse_to_endpoints(self, chunk: Dict, original_symbol: str, visited: set, result_endpoints: List[Dict]):
        """
        Recursively traverse backward through the call graph to find endpoints.
        """
        chunk_id = chunk.get('id', '')
        if chunk_id in visited:
            return
        visited.add(chunk_id)
        
        # Check if this chunk has endpoints
        endpoints = chunk.get('endpoints', [])
        if endpoints:
            for endpoint in endpoints:
                result_endpoints.append({
                    'path': endpoint.get('path', ''),
                    'method': endpoint.get('method', ''),
                    'calling_method': chunk.get('method_name', ''),
                    'calling_class': chunk.get('class_name', ''),
                    'original_symbol': original_symbol
                })
            return  # Found endpoints, no need to traverse further
        
        # If no endpoints, find chunks that call this chunk
        chunk_symbol = self._get_chunk_symbol(chunk)
        if chunk_symbol:
            for key, other_chunk in self._chunk_lookup.items():
                calls = other_chunk.get('calls', [])
                for call in calls:
                    if chunk_symbol in call or call in chunk_symbol:
                        self._traverse_to_endpoints(other_chunk, original_symbol, visited, result_endpoints)
                        break
    
    def _get_chunk_symbol(self, chunk: Dict) -> str:
        """
        Get the symbol representation of a chunk (class.method or just class)
        """
        class_name = chunk.get('class_name', '')
        method_name = chunk.get('method_name', '')
        
        if class_name and method_name:
            return f"{class_name}.{method_name}"
        elif class_name:
            return class_name
        else:
            return ''

    def find_by_symbol_name(self, symbol: str) -> List[Document]:
        # Ensure the index and chunk lookup are loaded
        if not self._loaded:
            logger.debug(f"Loading index for symbol search: {symbol}")
            self._ensure_loaded_sync()
            
        if not self._loaded:
            logger.warning(f"Failed to load index for find_by_symbol_name, returning empty results")
            return []
            
        found = []
        # logger.debug(f"🔍 Finding by symbol name: {symbol}")
        
        if not hasattr(self, '_chunk_lookup'):
            logger.warning(f"Chunk lookup not available, returning empty results")
            return []
            
        for key, chunk in self._chunk_lookup.items():
            # logger.info(f"key: {key}")
            if symbol.lower() in key.lower():
                full_text = self._get_full_text(chunk)
                found.append(Document(page_content=full_text, metadata=chunk))
                
        # logger.debug(f"Found {len(found)} matches for symbol: {symbol}")
        return found

    def retrieve_sync(self, query: str, top: int = 5, hyde: bool = False) -> List[Document]:

        coro = self.retrieve(query, top, hyde)

        try:
            return asyncio.run(coro)
        except RuntimeError:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(coro)

    def _bm25_with_scores(self, query: str, metadata: List[Dict], top_k: int = 100):
        texts = [f"{c.get('summary','')}\n{c.get('content','')}" for c in metadata]
        tokenised = [t.split() for t in texts]
        bm25 = BM25Okapi(tokenised)
        scores = bm25.get_scores(query.split())
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [(metadata[i], scores[i]) for i in ranked]
    
    def _deduplicate_chunks(self, chunks: List[Dict]) -> List[Dict]:
        if not chunks:
            return chunks
            
        seen_ids = set()
        unique_chunks = []
        duplicates_count = 0
        
        for chunk in chunks:
            chunk_id = chunk.get("id")
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
            chunk_id = doc.metadata.get("id")
            if chunk_id not in seen_ids:
                # logger.info(f"Adding document: {doc.metadata.get('class_name')}.{doc.metadata.get('method_name')}")
                seen_ids.add(chunk_id)
                unique_docs.append(doc)
            else:
                # logger.info(f"Duplicate document: {doc.metadata.get('class_name')}.{doc.metadata.get('method_name')}")
                duplicates_count += 1
        
        if duplicates_count > 0:
            logger.debug(f"🔄 Removed {duplicates_count} duplicate documents")
            
        return unique_docs