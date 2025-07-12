from __future__ import annotations

from pathlib import Path
import os
import asyncio
import uuid
import datetime

from fastapi import APIRouter, HTTPException, Depends
from services import jira_mcp_service
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from loguru import logger

from utils.file import read_json, write_json, ensure_dir
from utils.query_parser import validate_parsed_query, format_query_help, extract_confluence_page_info, extract_jira_issue_key
from services.analyzer_chain import AnalyzerChain
from services.confluence_mcp_service import ConfluenceMCPService, ConfluenceMCPConfigBuilder
from services.jira_mcp_service import JiraMCPService, JiraMCPConfigBuilder
from services.bitbucket_mcp_service import BitbucketMCPService, BitbucketMCPConfigBuilder
from models import get_db_session, Project, ProjectThread, ChatHistory

router = APIRouter(tags=["analyze"])

STORAGE_DIR = Path("storage")


class AnalyzeRequest(BaseModel):
    thread_id: str = Field(..., description="Thread ID to analyze")
    user_query: str = Field(..., description="User query for analysis")
