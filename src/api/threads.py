"""
Project Thread API endpoints for conversation context management (flat version)
"""

import uuid
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from models import get_db_session, Project, ProjectThread, ChatHistory
from utils.logger import init_logger

logger = init_logger()

router = APIRouter(tags=["threads"])

# Request/Response Models
class CreateThreadRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    project_id: str = Field(..., description="Project ID this thread belongs to")
    branch: str = Field(default="main", description="Code branch to focus on")
    api_method: Optional[str] = Field(None, description="API method (GET, POST, etc)")
    api_path: Optional[str] = Field(None, description="API path (e.g. /api/users)")
    documents: Optional[List[str]] = Field(default_factory=list, description="List of business document URLs (comma-separated in DB)")
    api_documents: Optional[List[str]] = Field(default_factory=list, description="List of API document URLs (comma-separated in DB)")
    jira_links: Optional[List[str]] = Field(default_factory=list, description="List of Jira URLs (comma-separated in DB)")
    references: Optional[List[str]] = Field(default_factory=list, description="List of class/method symbols (comma-separated in DB)")

class UpdateThreadRequest(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    branch: Optional[str] = None
    is_active: Optional[bool] = None
    api_method: Optional[str] = None
    api_path: Optional[str] = None
    documents: Optional[List[str]] = None
    api_documents: Optional[List[str]] = None
    jira_links: Optional[List[str]] = None
    references: Optional[List[str]] = None

@router.post("/threads")
async def create_thread(
    request: CreateThreadRequest,
    db: Session = Depends(get_db_session)
):
    """Create a new project thread (flat version)."""
    try:
        # Verify project exists
        project = db.query(Project).filter(Project.project_id == request.project_id).first()
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Generate unique thread ID
        thread_id = f"thread_{uuid.uuid4().hex[:8]}"
        
        # Store documents, api_documents, jira_links, references as comma-separated strings
        documents_str = ",".join(request.documents) if request.documents else None
        api_documents_str = ",".join(request.api_documents) if request.api_documents else None
        jira_links_str = ",".join(request.jira_links) if request.jira_links else None
        references_str = ",".join(request.references) if request.references else None
        
        # Create thread
        thread = ProjectThread(
            thread_id=thread_id,
            name=request.name,
            description=request.description,
            project_id=request.project_id,
            branch=request.branch,
            api_method=request.api_method,
            api_path=request.api_path,
            documents=documents_str,
            api_documents=api_documents_str,
            jira_links=jira_links_str,
            references=references_str
        )
        db.add(thread)
        db.commit()
        db.refresh(thread)
        return {
            "status": "created",
            "thread": thread.to_dict()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating thread: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@router.put("/threads/{thread_id}")
async def update_thread(
    thread_id: str,
    request: UpdateThreadRequest,
    db: Session = Depends(get_db_session)
):
    """Update a thread (flat version)."""
    try:
        thread = db.query(ProjectThread).filter(ProjectThread.thread_id == thread_id).first()
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")
        
        # Update fields if provided
        if request.name is not None:
            thread.name = request.name
        if request.description is not None:
            thread.description = request.description
        if request.branch is not None:
            thread.branch = request.branch
        if request.is_active is not None:
            thread.is_active = request.is_active
        if request.api_method is not None:
            thread.api_method = request.api_method
        if request.api_path is not None:
            thread.api_path = request.api_path
        if request.documents is not None:
            thread.documents = ",".join(request.documents)
        if request.api_documents is not None:
            thread.api_documents = ",".join(request.api_documents)
        if request.jira_links is not None:
            thread.jira_links = ",".join(request.jira_links)
        if request.references is not None:
            thread.references = ",".join(request.references)
        thread.updated_at = datetime.now(datetime.timezone.utc)
        db.commit()
        db.refresh(thread)
        logger.info(f"Updated thread {thread_id}")
        return {
            "status": "updated",
            "thread": thread.to_dict()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating thread {thread_id}: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@router.get("/threads")
async def list_threads(
    project_id: Optional[str] = None,
    is_active: Optional[bool] = True,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db_session)
):
    """List project threads with optional filtering"""
    try:
        query = db.query(ProjectThread)
        
        if project_id:
            query = query.filter(ProjectThread.project_id == project_id)
        
        if is_active is not None:
            query = query.filter(ProjectThread.is_active == is_active)
        
        query = query.order_by(ProjectThread.last_activity.desc())
        threads = query.offset(offset).limit(limit).all()
        
        total = query.count()
        
        return {
            "threads": [thread.to_dict() for thread in threads],
            "total": total,
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Error listing threads: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@router.get("/threads/{thread_id}")
async def get_thread(
    thread_id: str,
    db: Session = Depends(get_db_session)
):
    """Get a specific thread with all details"""
    try:
        thread = db.query(ProjectThread).filter(ProjectThread.thread_id == thread_id).first()
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")
        
        return {
            "thread": thread.to_dict(include_details=True)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting thread {thread_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@router.delete("/threads/{thread_id}")
async def delete_thread(
    thread_id: str,
    db: Session = Depends(get_db_session)
):
    """Delete a thread (soft delete by setting is_active=False)"""
    try:
        thread = db.query(ProjectThread).filter(ProjectThread.thread_id == thread_id).first()
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")
        
        thread.is_active = False
        thread.updated_at = datetime.now(datetime.timezone.utc)
        
        db.commit()
        
        logger.info(f"Deleted thread {thread_id}")
        
        return {"status": "deleted"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting thread {thread_id}: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@router.get("/projects/{project_id}/threads")
async def get_threads_by_project(
    project_id: str,
    db: Session = Depends(get_db_session)
):
    """Get all threads for a given project (active and inactive)."""
    try:
        threads = db.query(ProjectThread).filter(ProjectThread.project_id == project_id).order_by(ProjectThread.last_activity.desc()).all()
        return {"threads": [thread.to_dict() for thread in threads]}
    except Exception as e:
        logger.error(f"Error getting threads for project {project_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@router.get("/threads/{thread_id}/chat-history")
async def get_thread_chat_history(
    thread_id: str,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db_session)
):
    """Get chat history for a specific thread"""
    try:
        # Verify thread exists
        thread = db.query(ProjectThread).filter(ProjectThread.thread_id == thread_id).first()
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")
        
        # Get chat history
        messages = db.query(ChatHistory).filter(
            ChatHistory.thread_id == thread_id
        ).order_by(ChatHistory.created_at.desc()).offset(offset).limit(limit).all()
        
        total = db.query(ChatHistory).filter(ChatHistory.thread_id == thread_id).count()
        
        return {
            "thread_id": thread_id,
            "messages": [message.to_dict() for message in messages],
            "total": total,
            "limit": limit,
            "offset": offset
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chat history for thread {thread_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@router.get("/threads/{thread_id}/messages")
async def get_latest_messages(
    thread_id: str,
    limit: int = 10,
    offset: int = 0,
    db: Session = Depends(get_db_session)
):
    """Get latest messages for a specific thread with pagination"""
    try:
        # Verify thread exists
        thread = db.query(ProjectThread).filter(ProjectThread.thread_id == thread_id).first()
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")
        
        # Get latest messages ordered by creation time (newest first)
        messages = db.query(ChatHistory).filter(
            ChatHistory.thread_id == thread_id
        ).order_by(ChatHistory.created_at.desc()).offset(offset).limit(limit).all()
        
        # Get total count for pagination
        total = db.query(ChatHistory).filter(ChatHistory.thread_id == thread_id).count()
        
        # Calculate pagination info
        has_next = (offset + limit) < total
        has_previous = offset > 0
        total_pages = (total + limit - 1) // limit  # Ceiling division
        current_page = (offset // limit) + 1
        
        return {
            "thread_id": thread_id,
            "messages": [message.to_dict() for message in messages],
            "pagination": {
                "total": total,
                "limit": limit,
                "offset": offset,
                "current_page": current_page,
                "total_pages": total_pages,
                "has_next": has_next,
                "has_previous": has_previous
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting latest messages for thread {thread_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}") 