from __future__ import annotations

from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field, HttpUrl

from utils.logger import init_logger
from utils.file import write_json, ensure_dir
from services.bitbucket_mcp_service import BitbucketMCPService, BitbucketMCPConfigBuilder
from services.parser import parse_project
from services.merkle import compute_merkle_tree, diff_trees, save_merkle, load_merkle
from services.neo4j import get_neo4j_connection
from models import get_db_session, Project

logger = init_logger()

router = APIRouter(tags=["project"])

STORAGE_DIR = Path("storage")


class CreateProjectRequest(BaseModel):
    project_id: str = Field(..., pattern=r"^[a-zA-Z0-9_\-]+$")
    bitbucket_url: HttpUrl
    branch: str = "main"


@router.post("/api/v1/create-project")
async def create_project(body: CreateProjectRequest, db=Depends(get_db_session)):
    # Use Bitbucket MCP service to pull repo
    config = BitbucketMCPConfigBuilder.from_env()
    async with BitbucketMCPService(config) as bitbucket:
        repo_name = body.bitbucket_url.path.strip("/").split("/")[-1].replace('.git', '')
        repo_path = STORAGE_DIR / "repos" / body.project_id
        # clone_result = await bitbucket.clone_repository(
        #     session_id=f"create_project_{body.project_id}",
        #     repository=repo_name,
        #     branch=body.branch,
        #     target_path=repo_path
        # )
        # if clone_result["status"] != "success":
        #     raise HTTPException(status_code=400, detail=f"Failed to clone Bitbucket repo: {clone_result.get('error', 'Unknown error')}")
        # sha = clone_result["data"]["commit_hash"]

    # Save project info to database
    project = Project(
        project_id=body.project_id,
        name=repo_name,
        description=f"Imported from {body.bitbucket_url}",
        bitbucket_url=str(body.bitbucket_url),
        workspace=repo_name.split('/')[0] if '/' in repo_name else '',
        repository=repo_name,
        default_branch=body.branch,
        commit_hash="todo",
        indexed_files=0,
        extracted_files=0,
        status="active"
    )
    db.add(project)
    db.commit()
    db.refresh(project)

    # Parse files and obtain dependency graph
    # chunks, dep_graph = parse_project(repo_path, body.project_id)
    # if not chunks:
    #     raise HTTPException(status_code=400, detail="No Java files found in repository")

    # # Use Neo4j connection to import chunks
    # neo4j_conn = get_neo4j_connection()
    # neo4j_conn.delete_project_data(body.project_id)
    # neo4j_conn.import_code_chunks(chunks, 50)

    # Write metadata
    meta = {
        "commit": sha,
        # "chunks": [{k: v for k, v in c.items()} for c in chunks],
        # "dependency_graph": dep_graph,
    }
    ensure_dir(STORAGE_DIR / "metadata")
    write_json(STORAGE_DIR / "metadata" / f"{body.project_id}.json", meta)

    # Merkle tree
    merkle_tree = compute_merkle_tree(repo_path)
    save_merkle(body.project_id, merkle_tree)

    # Update project with file counts
    # project.indexed_files = len(chunks)
    # project.extracted_files = len(chunks)
    db.commit()
    db.refresh(project)

    return {"status": "created", "indexed_files": 1, "project": project.to_dict()}


class ReindexRequest(BaseModel):
    project_id: str


@router.post("/api/v1/reindex")
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

    # Re-parse project
    chunks, dep_graph = parse_project(repo_path, body.project_id, True)
    # Use Neo4j connection to import chunks
    neo4j_conn = get_neo4j_connection()
    neo4j_conn.delete_project_data(body.project_id)
    neo4j_conn.import_code_chunks(chunks, 50)
    
    # Update metadata
    meta = {
        "chunks": [{k: v for k, v in c.items()} for c in chunks],
        "dependency_graph": dep_graph,
    }
    write_json(STORAGE_DIR / "metadata" / f"{body.project_id}.json", meta)
    save_merkle(body.project_id, new_tree)

    return {"status": "reindexed", "changed_files": changed_files}


@router.get("/api/v1/projects")
async def get_projects(db=Depends(get_db_session)):
    projects = db.query(Project).all()
    return {"projects": [p.to_dict() for p in projects]} 