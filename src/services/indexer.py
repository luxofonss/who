"""FAISS index management with metadata & async persistence.

This module defines the ``Indexer`` class, responsible for building, updating,
searching, and serialising FAISS indices together with chunk-level metadata.

Key features
------------
* Supports IndexFlatL2 (exact) and IndexHNSWFlat (approximate) indexes.
* Associates every vector with a metadata dict (e.g. Tree-sitter chunk info).
* Persists ``.faiss`` file alongside a JSON metadata file.
* Async save/load using ``aiofiles`` + ``asyncio.to_thread`` so FastAPI isn't
  blocked by disk I/O or FAISS serialisation.
* Allows incremental additions via :py:meth:`add_vectors`.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Optional, Sequence, Tuple

import asyncio
import json

import aiofiles
import faiss
import numpy as np
from loguru import logger

from utils.file import ensure_dir

__all__ = ["Indexer"]


class Indexer:
    """Manage a FAISS index (+ metadata) for a single project."""

    def __init__(
        self,
        project_id: str,
        *,
        index_type: str = "flat",
        dim: int = 768,
    ) -> None:
        self.project_id = project_id
        self.dim = dim
        self.index_type = index_type.lower()

        self.index_path = Path(f"storage/indexes/{project_id}.faiss")
        self.metadata_path = Path(f"storage/indexes/{project_id}_metadata.json")

        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict] = []  # 1-to-1 with vectors – order matters

    # ---------------------------------------------------------------------
    # Index construction / update
    # ---------------------------------------------------------------------

    def _create_index(self) -> faiss.Index:
        if self.index_type == "hnsw":
            logger.debug(f"Creating IndexHNSWFlat (M=32)")
            index = faiss.IndexHNSWFlat(self.dim, 32)
        else:
            logger.debug(f"Creating IndexFlatL2")
            index = faiss.IndexFlatL2(self.dim)
        return index

    def build_index(self, embeddings: np.ndarray, metadata: Sequence[Dict] | None = None):
        """Create a fresh index from *embeddings* and optional *metadata*."""
        self._validate_embeddings(embeddings)
        index = self._create_index()
        if embeddings.shape[0] > 0:
            index.add(embeddings)
        self.index = index
        self.metadata = list(metadata) if metadata is not None else []
        logger.info(f"Built index with {embeddings.shape[0]} vectors (dim={self.dim})")

    def add_vectors(self, embeddings: np.ndarray, metadata: Sequence[Dict]):
        """Add new vectors + metadata without rebuilding the index."""
        if self.index is None:
            self.build_index(np.empty((0, self.dim), dtype=np.float32))

        self._validate_embeddings(embeddings)
        if embeddings.shape[0] != len(metadata):
            raise ValueError("Embeddings count and metadata length mismatch")

        self.index.add(embeddings)
        self.metadata.extend(metadata)
        logger.debug(f"Added {embeddings.shape[0]} vectors – index now {self.index.ntotal} vectors")

    # ------------------------------------------------------------------
    # Persistence (async)
    # ------------------------------------------------------------------

    async def save(self):
        """Persist FAISS index (.faiss) and metadata (.json)."""
        if self.index is None:
            logger.warning(f"No index to save for project {self.project_id}")
            return

        ensure_dir(self.index_path.parent)

        # Save index in a thread – FAISS is CPU-bound & blocking.
        await asyncio.to_thread(faiss.write_index, self.index, str(self.index_path))

        async with aiofiles.open(self.metadata_path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(self.metadata, indent=2, ensure_ascii=False))

        logger.info(f"Saved index + metadata for {self.project_id} (vectors={self.index.ntotal})")

    async def load(self) -> Optional[faiss.Index]:
        """Load index + metadata from disk.  Returns the index or *None* if missing."""
        if not self.index_path.exists():
            logger.warning(f"Index file {self.index_path} not found")
            return None

        index = await asyncio.to_thread(faiss.read_index, str(self.index_path))
        if index.d != self.dim:
            raise ValueError(f"Loaded index dim {index.d} != expected {self.dim}")

        if self.metadata_path.exists():
            async with aiofiles.open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.loads(await f.read())
        else:
            logger.warning(f"Metadata file {self.metadata_path} missing – proceeding with empty list")
            self.metadata = []

        self.index = index
        logger.info(f"Loaded index for {self.project_id} (vectors={self.index.ntotal})")
        return index

    # ------------------------------------------------------------------
    # Search helpers
    # ------------------------------------------------------------------

    def search(self, query: np.ndarray, k: int = 5) -> Tuple[np.ndarray, List[Dict]]:
        """Return (distances, chunk_metadata) for top-*k*.

        *query* must be 1-D embedding.
        """
        logger.info(f"Searching index for project {self.project_id} with query shape {query.shape} and k={k}")
        if self.index is None:
            raise ValueError("Index not initialised – call build_index/load() first")

        query = query.astype("float32").reshape(1, -1)
        if query.shape[1] != self.dim:
            raise ValueError(f"Query dim {query.shape[1]} != index dim {self.dim}")

        distances, indices = self.index.search(query, k)
        idxs = indices[0]
        chunks = [self.metadata[i] for i in idxs if 0 <= i < len(self.metadata)]
        return distances[0], chunks

    def _validate_embeddings(self, embs: np.ndarray):
        if embs.ndim != 2 or embs.shape[1] != self.dim:
            raise ValueError(f"Embeddings must have shape (n, {self.dim})")
        if np.isnan(embs).any() or np.isinf(embs).any():
            raise ValueError("Embeddings contain NaN/Inf values")
