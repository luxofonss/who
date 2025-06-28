from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from loguru import logger

from utils.file import read_json, write_json, ensure_dir
from services.analyzer_chain import AnalyzerChain

router = APIRouter(tags=["analyze"])

STORAGE_DIR = Path("storage")


@router.post("/analyze")
async def analyze(
    project_id: str = Form(...),
    endpoint: str = Form(...),
    user_text: str = Form(""),
    requirements: UploadFile = File(...),
    testcases: UploadFile = File(...),
):
    metadata_path = STORAGE_DIR / "metadata" / f"{project_id}.json"
    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    # Read uploaded files
    requirements_txt = (await requirements.read()).decode("utf-8", errors="ignore")
    testcases_txt = (await testcases.read()).decode("utf-8", errors="ignore")

    logger.info(f"Analyzing project {project_id} at endpoint {endpoint}")

    analyzer = AnalyzerChain(project_id)
    logger.info(f"Retrieving context for endpoint: {endpoint}")
    logger.info(f"User text: {user_text}")

    result = await analyzer.run(
        endpoint=endpoint,
        requirements_txt=requirements_txt,
        testcases_txt=testcases_txt,
        user_text=user_text,
    )

    # Persist analysis
    analyze_dir = STORAGE_DIR / "analyze"
    ensure_dir(analyze_dir)
    key = endpoint.strip("/").replace("/", "_")
    write_json(analyze_dir / f"{project_id}_{key}.json", result)

    return result 