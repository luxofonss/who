from __future__ import annotations

from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, HttpUrl

from utils.logger import init_logger
from utils.file import write_json, ensure_dir
from services.fetcher import clone_repo
from services.parser import parse_project
from services.embedder import embed_texts
from services.indexer import Indexer
from services.merkle import compute_merkle_tree, diff_trees, save_merkle, load_merkle

logger = init_logger()

router = APIRouter(tags=["project"])

STORAGE_DIR = Path("storage")


class CreateProjectRequest(BaseModel):
    project_id: str = Field(..., pattern=r"^[a-zA-Z0-9_\-]+$")
    github_url: HttpUrl
    branch: str = "main"


@router.post("/create-project")
async def create_project(body: CreateProjectRequest):
    # Clone or update repo
    repo_path, sha = clone_repo(body.project_id, str(body.github_url), body.branch)

    # Parse files and obtain dependency graph
    chunks, dep_graph = parse_project(repo_path)
    if not chunks:
        raise HTTPException(status_code=400, detail="No Java files found in repository")

    # Embeddings & FAISS index
    texts = [f"{c.get('summary','')}\n\n{c['content']}" for c in chunks]
    vectors = embed_texts(texts)

    # Build index with metadata (without content to save space)
    meta_for_index = [{k: v for k, v in ch.items()} for ch in chunks]
    indexer = Indexer(body.project_id, index_type="hnsw")
    indexer.build_index(vectors, metadata=meta_for_index)
    await indexer.save()

    # Write metadata
    meta = {
        "commit": sha,
        "chunks": [{k: v for k, v in c.items()} for c in chunks],
        "dependency_graph": dep_graph,
    }
    ensure_dir(STORAGE_DIR / "metadata")
    write_json(STORAGE_DIR / "metadata" / f"{body.project_id}.json", meta)

    # Merkle tree
    merkle_tree = compute_merkle_tree(repo_path)
    save_merkle(body.project_id, merkle_tree)

    return {"status": "created", "indexed_files": len(chunks)}


class ReindexRequest(BaseModel):
    project_id: str


@router.post("/reindex")
async def reindex(body: ReindexRequest):
    repo_path = STORAGE_DIR / "repos" / body.project_id
    if not repo_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    # Compute diff
    old_tree = load_merkle(body.project_id)
    new_tree = compute_merkle_tree(repo_path)
    changed_files: List[str] = diff_trees(old_tree, new_tree)

    if not changed_files:
        return {"status": "reindexed", "changed_files": []}

    # Re-parse & re-index – could be incremental, here we rebuild for clarity
    chunks, dep_graph = parse_project(repo_path)
    texts = [f"{c.get('summary','')}\n\n{c['content']}" for c in chunks]
    vectors = embed_texts(texts)

    meta_for_index = [{k: v for k, v in ch.items()} for ch in chunks]
    indexer = Indexer(body.project_id, index_type="hnsw")
    indexer.build_index(vectors, metadata=meta_for_index)
    await indexer.save()

    meta = {
        "chunks": [{k: v for k, v in c.items()} for c in chunks],
        "dependency_graph": dep_graph,
    }
    write_json(STORAGE_DIR / "metadata" / f"{body.project_id}.json", meta)
    save_merkle(body.project_id, new_tree)

    return {"status": "reindexed", "changed_files": changed_files} 