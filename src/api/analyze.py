from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from loguru import logger

from utils.file import read_json, write_json, ensure_dir
from services.analyzer_chain import AnalyzerChain

router = APIRouter(tags=["analyze"])

STORAGE_DIR = Path("storage")


def extract_endpoint_and_user_text(raw: str) -> tuple[str, str]:
    """
    Extracts endpoint path and user description from a string like:
    '@endpoint=/api/v1/test @method=POST some text that user describe'
    Returns (endpoint, user_text)
    """
    import re
    endpoint_match = re.search(r'@endpoint=([^\s]+)', raw)
    endpoint = endpoint_match.group(1) if endpoint_match else ''
    # Remove the tags from the string để lấy phần mô tả user
    user_desc = re.sub(r'@endpoint=[^\s]+', '', raw)
    user_desc = re.sub(r'@method=[^\s]+', '', user_desc).strip()
    return endpoint, user_desc


@router.post("/analyze")
async def analyze(
    project_id: str = Form(...),
    endpoint: str = Form(...),
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

    # Tách endpoint và mô tả user từ form endpoint
    endpoint_path, endpoint_user_desc = extract_endpoint_and_user_text(endpoint)
    combined_user_text = endpoint_user_desc.strip()

    analyzer = AnalyzerChain(project_id)
    logger.info(f"Retrieving context for endpoint: {endpoint_path}")
    logger.info(f"User text: {combined_user_text}")

    result = await analyzer.run(
        endpoint=endpoint_path,
        requirements_txt=requirements_txt,
        testcases_txt=testcases_txt,
        user_text=combined_user_text,
    )

    # Persist analysis
    analyze_dir = STORAGE_DIR / "analyze"
    ensure_dir(analyze_dir)
    key = endpoint_path.strip("/").replace("/", "_")
    write_json(analyze_dir / f"{project_id}_{key}.json", result)

    return result 