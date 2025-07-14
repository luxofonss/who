from __future__ import annotations

import asyncio
from pathlib import Path
from typing import List, Dict, Set

from langchain_core.documents import Document
from loguru import logger

from utils.file import read_json

STORAGE_DIR = Path("storage")

class LangChainRetriever:
    def __init__(self, project_id: str):
        self.project_id = project_id
        self._loaded = False

        meta_path = STORAGE_DIR / "metadata" / f"{project_id}.json"
        data = read_json(meta_path) or {}
        self.metadata = data.get("chunks", [])
        self.dep_graph: Dict[str, Dict[str, List[str]]] = data.get("dependency_graph", {})
        self._chunk_lookup: Dict[str, Dict] = {
            f"{c['class_name']}.{c.get('method_name')}": c
            for c in self.metadata
        }
        self._loaded = True

    def _key(self, chunk: Dict) -> str:
        return f"{chunk.get('class_name')}.{chunk.get('method_name')}"

    def _chunk_id(self, chunk: Dict) -> str:
        class_name = chunk.get('class_name', '')
        method_name = chunk.get('method_name', '')
        file_path = chunk.get('file_path', '')
        start_line = chunk.get('start_line', 0)
        return f"{file_path}::{class_name}::{method_name}::{start_line}"

    async def retrieve(self, query: str, top: int, hyde: bool = False, method: str | None = None) -> List[Document]:
        # Parse query to extract path and method from dict format
        try:
            import ast
            query_dict = ast.literal_eval(query)
            target_path = query_dict.get('path', '')
            target_method = query_dict.get('method', '').upper()
        except (ValueError, SyntaxError) as e:
            logger.warning(f"Invalid query format. Expected dict string, got: {query}. Error: {e}")
            return []
        
        logger.info(f"Searching for endpoint: path='{target_path}', method='{target_method}'")
        
        # Search through all chunks
        for chunk in self.metadata:
            endpoints = chunk.get('endpoints', [])
            
            # Check if this chunk has the matching endpoint
            for endpoint in endpoints:
                endpoint_path = endpoint.get('path', '')
                endpoint_method = endpoint.get('method', '').upper()
                
                if endpoint_path == target_path and endpoint_method == target_method:
                    logger.info(f"Found matching chunk: {chunk.get('class_name', '')}.{chunk.get('method_name', '')}")
                    full_text = self._get_full_text(chunk)
                    doc = Document(page_content=full_text, metadata=chunk)
                    docs = [doc]
                    seen = {self._key(chunk)}
                    self._traverse_call_graph(chunk, docs, seen)
                    return self._deduplicate_documents(docs)
        
        logger.warning(f"No chunk found with endpoint: {target_path} {target_method}")
        return []

    def _get_full_text(self, c: Dict) -> str:
        if (c.get("chunk_type") == "method"):
            return f"# Summary: {c.get('class_name', '')}.{c.get('method_name', '')} {c.get('summary', '')}\n\n{c['content']}"
        else:
            return f"# Summary: {c.get('summary', '')}\n\n{c['content']}"

    def _traverse_call_graph(self, seed: Dict, docs: List[Document], seen: Set[str]):
        """Recursively traverse transitive dependencies including calls, inheritance, and interface relations."""
        logger.info(f"Traversing call graph for {seed.get('class_name')}.{seed.get('method_name')}")
        stack = [seed]
        RELATION_TYPES = ["calls", "implemented_by", "extended_by", "vars"]

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
                    if seed.get("method_name") and "." not in related:
                        d = d + self.find_chunk_by_symbol_name(related + "." + seed.get("method_name", ""))
                    for dx in d:
                        full_text = self._get_full_text(dx)
                        if dx.get("chunk_type") == "controller":
                            continue
                        docs.append(Document(page_content=full_text, metadata=dx))
                        stack.append(dx)

    def find_chunk_by_symbol_name(self, symbol: str, method_name: str = '') -> List[Dict]:
        found = []
        for key, chunk in self._chunk_lookup.items():
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
        return found

    async def retrieve_endpoints(self, symbols: List[str]) -> List[str]:
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

    def find_endpoint_by_symbol_name(self, symbol: str) -> List[Dict[str, str]]:
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
        """Recursively traverse backward through the call graph to find endpoints."""
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
        """Get the symbol representation of a chunk (class.method or just class)"""
        class_name = chunk.get('class_name', '')
        method_name = chunk.get('method_name', '')
        
        if class_name and method_name:
            return f"{class_name}.{method_name}"
        elif class_name:
            return class_name
        else:
            return ''


    def _deduplicate_documents(self, docs: List[Document]) -> List[Document]:
        if not docs:
            return docs
            
        seen_ids = set()
        unique_docs = []
        duplicates_count = 0
        
        for doc in docs:
            chunk_id = doc.metadata.get("id")
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                unique_docs.append(doc)
            else:
                duplicates_count += 1
        
        if duplicates_count > 0:
            logger.debug(f"🔄 Removed {duplicates_count} duplicate documents")
            
        return unique_docs