"""
Project Thread API endpoints for conversation context management (flat version)
"""

import uuid
from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
import uuid
import asyncio

from services.chat_chain import ChatChain
from utils.logger import init_logger
from utils.file import read_json, write_json, ensure_dir, _split_csv_field
from utils.query_parser import validate_parsed_query, format_query_help, extract_confluence_page_info, extract_jira_issue_key
from services.analyzer_chain import AnalyzerChain
from services.confluence_mcp_service import ConfluenceMCPService, ConfluenceMCPConfigBuilder
from services.jira_mcp_service import JiraMCPService, JiraMCPConfigBuilder
from services.bitbucket_mcp_service import BitbucketMCPService, BitbucketMCPConfigBuilder
from models import get_db_session, Project, ProjectThread, ChatHistory
from .analyze import AnalyzeRequest

logger = init_logger()

def _split_csv_field(field):
    if not field:
        return []
    return [x.strip() for x in field.split(",") if x.strip()]

async def get_thread_context_and_requirements(thread_id: str, db: Session):
    """Extract reusable function to get thread context, requirements, and code commits"""
    # Retrieve thread context from DB
    thread = db.query(ProjectThread).filter(ProjectThread.thread_id == thread_id).first()
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    project = db.query(Project).filter(Project.project_id == thread.project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    api_method = thread.api_method or "GET"
    api_path = thread.api_path or "/"
    jira_links = _split_csv_field(thread.jira_links)
    confluence_business_docs = _split_csv_field(thread.documents)
    confluence_api_docs = _split_csv_field(getattr(thread, 'api_documents', None))
    references = _split_csv_field(getattr(thread, 'references', None))
    branch = thread.branch

    # Combine all Confluence docs
    confluence_urls = confluence_business_docs + confluence_api_docs
    requirements_txt = ""
    code_commit = ""
    try:
        # Retrieve Confluence content
        if confluence_urls:
            confluence_content = await retrieve_confluence_content(confluence_urls, thread_id)
            requirements_txt += confluence_content
        # Retrieve Jira content
        jira_content = ""
        commits = []
        if jira_links:
            jira_content = await retrieve_jira_content(jira_links, thread_id)
            requirements_txt += jira_content
            
            # Extract commits from Jira issues using JiraMCPService
            config = JiraMCPConfigBuilder.from_env()
            async with JiraMCPService(config) as jira_service:
                for url in jira_links:
                    issue_key = extract_jira_issue_key(url)
                    if issue_key:
                        logger.info(f"Extracting commits for Jira issue: {issue_key}")
                        try:
                            # Extract commits using the _extract_commits_from_issue method
                            issue_commits = await jira_service._extract_commits_from_issue(issue_key)
                            if issue_commits:
                                logger.info(f"Found {len(issue_commits)} commits for {issue_key}")
                                for commit in issue_commits:
                                    logger.debug(f"Commit: {commit.get('commit_hash', 'Unknown')} in {commit.get('repository', 'Unknown')}")
                                commits.extend(issue_commits)
                            else:
                                logger.info(f"No commits found for {issue_key}")
                        except Exception as e:
                            logger.error(f"Error extracting commits for {issue_key}: {str(e)}")
        
        # Get diffs for commits using Bitbucket MCP service
        if commits:
            try:
                # Initialize Bitbucket MCP service
                bitbucket_config = BitbucketMCPConfigBuilder.from_env()
                
                # Check if Bitbucket configuration is valid
                if not bitbucket_config.email or not bitbucket_config.workspace:
                    logger.warning("Bitbucket configuration incomplete - skipping diff retrieval")
                    logger.warning(f"Email: {bitbucket_config.email}, Workspace: {bitbucket_config.workspace}")
                elif not bitbucket_config.app_password and not bitbucket_config.api_token:
                    logger.warning("Bitbucket authentication not configured - skipping diff retrieval")
                else:
                    async with BitbucketMCPService(bitbucket_config) as bitbucket_service:
                        logger.info(f"Getting diffs for {len(commits)} commits using Bitbucket MCP service")
                        
                        for commit in commits:
                            repository = commit.get('repository', '')
                            commit_hash = commit.get('commit_hash', '')
                            
                            if repository and commit_hash:
                                logger.info(f"Getting diff for commit {commit_hash} in repository {repository}")
                                try:
                                    # Get diff for this commit
                                    diff_result = await bitbucket_service._get_commit_diff(repository, commit_hash)
                                    if diff_result['status'] == 'success':
                                        commit['diff'] = diff_result
                                        logger.info(f"Successfully retrieved diff for {commit_hash}")
                                    else:
                                        logger.warning(f"Failed to get diff for {commit_hash}: {diff_result.get('error', 'Unknown error')}")
                                        commit['diff'] = {'status': 'error', 'error': diff_result.get('error', 'Unknown error')}
                                except Exception as e:
                                    logger.error(f"Error getting diff for commit {commit_hash}: {str(e)}")
                                    commit['diff'] = {'status': 'error', 'error': str(e)}
                            else:
                                logger.warning(f"Missing repository or commit hash for commit: {commit}")
                                
            except Exception as e:
                logger.error(f"Error initializing Bitbucket MCP service: {str(e)}")
                # Continue without diffs if Bitbucket service fails
        
        # Format commit content for analysis
        code_commit = ""
        changed_methods = []  # List to store changed methods
        if commits:
            commit_details = []
            for commit in commits:
                logger.info(f"Original commit: {commit}")
                commit_detail = f"Commit: {commit.get('commit_hash', 'Unknown')}\n"
                commit_detail += f"Display ID: {commit.get('display_id', commit.get('commit_hash', '')[:7])}\n"
                commit_detail += f"Repository: {commit.get('repository', 'Unknown')}\n"
                commit_detail += f"Repository ID: {commit.get('repository_id', 'Unknown')}\n"
                commit_detail += f"Author: {commit.get('author', 'Unknown')}\n"
                commit_detail += f"Message: {commit.get('message', 'No message')}\n"
                commit_detail += f"Date: {commit.get('date', 'Unknown')}\n"
                commit_detail += f"Files Changed: {commit.get('files_changed', 0)}\n"
                commit_detail += f"Merge Commit: {commit.get('merge', False)}\n"
                if commit.get('url'):
                    commit_detail += f"URL: {commit.get('url')}\n"
                
                # Add diff information if available
                if commit.get('diff') and commit['diff'].get('status') == 'success':
                    diff_data = commit['diff']
                    commit_detail += f"\n=== DIFF INFORMATION ===\n"
                    commit_detail += f"Total Files Changed: {diff_data.get('total_files', 0)}\n"
                    
                    # Add file change details
                    files_changed = diff_data.get('files_changed', [])
                    if files_changed:
                        commit_detail += f"\nFiles Changed:\n"
                        for file_info in files_changed:
                            commit_detail += f"- {file_info.get('new_path', file_info.get('old_path', 'Unknown'))}\n"
                            commit_detail += f"  Status: {file_info.get('status', 'Unknown')}\n"
                            commit_detail += f"  Additions: {file_info.get('additions', 0)}, Deletions: {file_info.get('deletions', 0)}\n"
                    
                    # Add diff text (truncated if too long)
                    diff_text = diff_data.get('diff_text', '')
                    logger.info(f"Diff text: {diff_text}")
                    if diff_text:
                        # Truncate diff text if it's too long to avoid overwhelming the analysis
                        max_diff_length = 50000  # Limit to 50000 characters (increased from 5000)
                        if len(diff_text) > max_diff_length:
                            diff_text = diff_text[:max_diff_length] + f"\n... (truncated, total length: {len(diff_data.get('diff_text', ''))} characters)"
                        commit_detail += f"\nDiff:\n{diff_text}\n"
                    
                    # Extract changed methods from diff text
                    changed_file_paths = [file.get('new_path', file.get('old_path', '')) for file in diff_data.get('files_changed', [])]
                    for file_path in changed_file_paths:
                        if file_path:
                            # Extract methods from diff text for this file
                            file_changed_methods = bitbucket_service._extract_changed_methods(diff_text, file_path)
                            changed_methods.extend(file_changed_methods)
                elif commit.get('diff') and commit['diff'].get('status') == 'error':
                    commit_detail += f"\nDiff Error: {commit['diff'].get('error', 'Unknown error')}\n"
                else:
                    commit_detail += f"\nNo diff information available\n"
                
                commit_details.append(commit_detail)

            
            code_commit = "\n\n".join(commit_details)
            logger.info(f"Code commit: {code_commit}")
            logger.info(f"Formatted {len(commits)} commits with diffs for analysis")
            
            # Remove duplicates from changed_methods based on class and method combination
            seen_methods = set()
            deduplicated_methods = []
            for method_info in changed_methods:
                method_key = f"{method_info.get('class', '')}.{method_info.get('method', '')}"
                if method_key not in seen_methods:
                    seen_methods.add(method_key)
                    deduplicated_methods.append(method_info)
            changed_methods = deduplicated_methods
            
            logger.info(f"Changed methods (after deduplication): {changed_methods}")
    except Exception as e:
        logger.error(f"Error retrieving MCP content: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving content from external sources: {str(e)}")

    return {
        "thread": thread,
        "project": project,
        "api_method": api_method,
        "api_path": api_path,
        "requirements_txt": requirements_txt,
        "code_commit": code_commit,
        "changed_methods": changed_methods
    }

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

class ChatMessageRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000, description="User message to send to the AI")

@router.post("/api/v1/threads")
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

@router.put("/api/v1/threads/{thread_id}")
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
        thread.updated_at = datetime.now(timezone.utc)
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

@router.get("/api/v1/threads")
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

@router.get("/api/v1/threads/{thread_id}")
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
        thread.updated_at = datetime.now(timezone.utc)
        
        db.commit()
        
        logger.info(f"Deleted thread {thread_id}")
        
        return {"status": "deleted"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting thread {thread_id}: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@router.get("/api/v1/projects/{project_id}/threads")
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

@router.get("/api/v1/threads/{thread_id}/messages")
async def get_messages(
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

@router.post("/api/v1/threads/{thread_id}/messages")
async def send_message(
    thread_id: str,
    request: ChatMessageRequest,
    db: Session = Depends(get_db_session)
):
    """Send a chat message to a thread and get AI response using ChatChain or AnalyzerChain"""
    try:
        # Verify thread exists and is active
        thread = db.query(ProjectThread).filter(ProjectThread.thread_id == thread_id).first()
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")
        
        if not thread.is_active:
            raise HTTPException(status_code=400, detail="Thread is not active")
        
        # Verify project exists
        project = db.query(Project).filter(Project.project_id == thread.project_id).first()
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        logger.info(f"💬 Chat message request for thread {thread_id}: {request.message[:100]}...")
        
        # Check if message starts with @analyze
        is_analyze_request = request.message.strip().lower().startswith("@analyze")
        
        if is_analyze_request:
            # Extract the actual query by removing @analyze prefix
            user_query = request.message.strip()[8:].strip()  # Remove "@analyze" and trim
            logger.info(f"🔍 Detected analyze request: {user_query[:100]}...")
            
            # Use analyze logic
            result = await handle_analyze_request(thread_id, user_query, db)
            
        else:
            # Use regular chat logic
            result = await handle_chat_request(thread_id, request.message, db)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing message for thread {thread_id}: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


async def handle_analyze_request(thread_id: str, user_query: str, db: Session):
    """Handle analyze request logic"""
    # Get thread context and requirements using reusable function
    context_data = await get_thread_context_and_requirements(thread_id, db)
    
    # Run the analyzer
    analyzer = AnalyzerChain(context_data["project"].project_id)
    logger.info(f"Running analysis for endpoint: {context_data['api_path']}")
    logger.info(f"Method: {context_data['api_method']}")
    logger.info(f"User query: {user_query}")
    
    endpoint = {
        "path": context_data["api_path"],
        "method": context_data["api_method"]
    }
    
    result = await analyzer.run(
        endpoint=str(endpoint),
        requirements_txt=context_data["requirements_txt"],
        user_text=user_query,
        code_commit=context_data["code_commit"],
        changed_methods=context_data["changed_methods"]
    )

    # Save chat history
    try:
        # Save user message (with original @analyze prefix)
        user_message_id = f"msg_{uuid.uuid4().hex[:8]}"
        user_chat = ChatHistory(
            message_id=user_message_id,
            thread_id=thread_id,
            role="user",
            content=f"@analyze {user_query}",  # Keep original format
            analysis_result=None
        )
        db.add(user_chat)
        
        # Save assistant message with analysis result
        assistant_message_id = f"msg_{uuid.uuid4().hex[:8]}"
        assistant_chat = ChatHistory(
            message_id=assistant_message_id,
            thread_id=thread_id,
            role="assistant",
            content=result.get("markdown_response", ""),
            analysis_result=result.get("json_response", "")
        )
        db.add(assistant_chat)
        
        # Update thread's message count and last activity
        context_data["thread"].message_count += 2  # User + Assistant messages
        context_data["thread"].last_activity = datetime.now(timezone.utc)
        
        db.commit()
        logger.info(f"Saved analyze chat history for thread {thread_id}")
        
    except Exception as e:
        logger.error(f"Error saving analyze chat history: {str(e)}")
        db.rollback()

    return result


async def handle_chat_request(thread_id: str, message: str, db: Session):
    """Handle regular chat request logic"""
    # Get chat history for this thread
    history_messages = db.query(ChatHistory).filter(
        ChatHistory.thread_id == thread_id
    ).order_by(ChatHistory.created_at.asc()).limit(20).all()
    
    # Convert to ChatChain format (list of strings)
    history = []
    for msg in history_messages:
        if msg.role == "assistant" and msg.analysis_result:
            history.append(f"AI: {msg.analysis_result}")
        else:
            history.append(f"{msg.role}: {msg.content}")
        
    
    # Use ChatChain to get AI response
    context_data = await get_thread_context_and_requirements(thread_id, db)
    requirements = context_data["requirements_txt"]
    code_commit = context_data["code_commit"]
    api_path = context_data["api_path"]
    api_method = context_data["api_method"]
    endpoint = {
        "path": api_path,
        "method": api_method
    }

    chat_chain = ChatChain(context_data["project"].project_id)
    result = await chat_chain.chat(
        message,
        history,
        str(endpoint),
        requirements, 
        code_commit, 
        changed_methods=context_data["changed_methods"]
    )

    # Generate unique message IDs
    user_message_id = f"msg_{uuid.uuid4().hex[:8]}"
    ai_message_id = f"msg_{uuid.uuid4().hex[:8]}"
    
    # Save user message to database
    user_message = ChatHistory(
        message_id=user_message_id,
        thread_id=thread_id,
        role="user",
        content=message
    )
    db.add(user_message)
    
    # Save AI response to database
    ai_message = ChatHistory(
        message_id=ai_message_id,
        thread_id=thread_id,
        role="assistant",
        content=result.response,
        analysis_result=""
    )
    db.add(ai_message)
    
    # Update thread's last activity
    thread = db.query(ProjectThread).filter(ProjectThread.thread_id == thread_id).first()
    thread.last_activity = datetime.now(timezone.utc)
    
    db.commit()
    
    logger.info(f"✅ Chat message processed using {result.method} method with {result.iteration_count} iterations")
    final_response = {
                "markdown_response": result.response,
                "json_response": "",
            }
    return final_response


# Keep the original analyze endpoint as a backup or for direct API calls
@router.post("/api/v1/threads/{thread_id}/analyze")
async def analyze(
    thread_id: str,
    request: AnalyzeRequest,
    db: Session = Depends(get_db_session)
):
    """Direct analyze endpoint (can be deprecated if not needed)"""
    return await handle_analyze_request(thread_id, request.user_query, db)


# Alternative implementation with more flexible prefix detection
@router.post("/api/v1/threads/{thread_id}/messages")
async def send_message_alternative(
    thread_id: str,
    request: ChatMessageRequest,
    db: Session = Depends(get_db_session)
):
    """Alternative implementation with flexible command detection"""
    try:
        # Verify thread exists and is active
        thread = db.query(ProjectThread).filter(ProjectThread.thread_id == thread_id).first()
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")
        
        if not thread.is_active:
            raise HTTPException(status_code=400, detail="Thread is not active")
        
        # Verify project exists
        project = db.query(Project).filter(Project.project_id == thread.project_id).first()
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        logger.info(f"💬 Chat message request for thread {thread_id}: {request.message[:100]}...")
        
        # More flexible command detection
        message_lower = request.message.strip().lower()
        
        # Check for various analyze patterns
        analyze_patterns = ["@analyze", "/analyze", "analyze:", "!analyze"]
        is_analyze_request = any(message_lower.startswith(pattern) for pattern in analyze_patterns)
        
        if is_analyze_request:
            # Extract the actual query by removing command prefix
            for pattern in analyze_patterns:
                if message_lower.startswith(pattern):
                    user_query = request.message.strip()[len(pattern):].strip()
                    break
            
            logger.info(f"🔍 Detected analyze request: {user_query[:100]}...")
            
            # Validate that there's actually a query after the command
            if not user_query:
                raise HTTPException(status_code=400, detail="Please provide a query after @analyze command")
            
            # Use analyze logic
            result = await handle_analyze_request(thread_id, user_query, db)
            
        else:
            # Use regular chat logic
            result = await handle_chat_request(thread_id, request.message, db)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing message for thread {thread_id}: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
        

async def retrieve_confluence_content(confluence_urls: list[str], session_id: str) -> str:
    """Retrieve content from Confluence URLs using MCP service"""
    try:
        config = ConfluenceMCPConfigBuilder.from_env()
        
        async with ConfluenceMCPService(config) as confluence:
            all_content = []
            
            for url in confluence_urls:
                logger.info(f"Retrieving Confluence content from: {url}")
                page_info = extract_confluence_page_info(url)
                logger.info(f"Extracted page info: {page_info}")
                
                result = None
                
                if page_info['page_id']:
                    # Get by page ID
                    logger.info(f"Attempting to get page by ID: {page_info['page_id']}")
                    result = await confluence.get_page_by_id(session_id, page_info['page_id'])
                
                # If page ID retrieval failed or no page ID, try by title and space
                if (not result or result["status"] != "success") and page_info['page_title'] and page_info['space_key']:
                    logger.info(f"Attempting to get page by title '{page_info['page_title']}' in space '{page_info['space_key']}'")
                    result = await confluence.get_page_by_title(session_id, page_info['page_title'], page_info['space_key'])
                # If still no success, try searching by title (only if space_key is not None)
                if (not result or result["status"] != "success") and page_info['page_title'] and page_info['space_key']:
                    logger.info(f"Attempting to search for page with title: {page_info['page_title']}")
                    result = await confluence.search_pages(session_id, page_info['page_title'], page_info['space_key'], limit=1)
                # Last resort: search using URL segments
                if not result or result["status"] != "success":
                    search_query = url.split('/')[-1].replace('-', ' ').replace('_', ' ').replace('+', ' ')
                    logger.info(f"Last resort search with query: {search_query}")
                    result = await confluence.search_pages(session_id, search_query, "", limit=1)
                
                if result["status"] == "success":
                    if "page" in result["data"]:
                        page = result["data"]["page"]
                        content = page.get('body', {}).get('storage', {}).get('value', '')
                        title = page.get('title', '')
                        content = f"=== {title} ===\n{content}\n\n"
                        all_content.append(content)
                        logger.info(f"Retrieved Confluence page: {title}")
                    elif "pages" in result["data"] and result["data"]["pages"]:
                        page = result["data"]["pages"][0]
                        content = page.get('body', {}).get('storage', {}).get('value', '')
                        title = page.get('title', '')
                        content = f"=== {title} ===\n{content}\n\n"
                        all_content.append(content)
                        logger.info(f"Retrieved Confluence page: {title}")
                else:
                    logger.warning(f"Failed to retrieve Confluence content from {url}: {result.get('error', 'Unknown error')}")
            
            return "\n".join(all_content)
            
    except Exception as e:
        logger.error(f"Error retrieving Confluence content: {str(e)}")
        return f"Error retrieving Confluence content: {str(e)}"


async def retrieve_jira_content(jira_urls: list[str], session_id: str) -> str:
    """Retrieve content from Jira URLs using MCP service"""
    try:
        config = JiraMCPConfigBuilder.from_env()
        
        async with JiraMCPService(config) as jira:
            all_content = []
            
            for url in jira_urls:
                logger.info(f"Retrieving Jira content from: {url}")
                issue_key = extract_jira_issue_key(url)
                
                if issue_key:
                    result = await asyncio.to_thread(jira.get_issue_by_key, session_id, issue_key)
                    logger.info(f"Jira issue result: {result}")
                    if result["status"] == "success":
                        issue = result["data"]["issue"]
                        logger.info(f"Jira issue: {issue}")
                        fields = issue.get('fields', {})
                        content = f"=== {issue.get('key', '')}: {fields.get('summary', '')} ===\n"
                        content += f"URL: {issue.get('self', '')}\n\n"
                        
                        content += f"Description:\n{fields.get('description', '')}\n\n"
                        
                        # Fix: get comments from fields
                        comments = fields.get('comment', {}).get('comments', [])
                        if comments:
                            content += f"Comments ({len(comments)} total):\n"
                            for i, comment in enumerate(comments, 1):
                                content += f"Content: {comment.get('body', '')}\n"
                        else:
                            content += "No comments.\n"
                        # Fix: get attachments from fields
                        attachments = fields.get('attachment', [])
                        if attachments:
                            content += f"\nAttachments ({len(attachments)} total):\n"
                            for attachment in attachments:
                                content += f"- {attachment.get('filename', '')} ({attachment.get('size', 0)} bytes, {attachment.get('mimeType', '')})\n"
                        
                        # Note: Commits will be extracted separately using _extract_commits_from_issue
                        content += "\n" + "="*50 + "\n\n"
                        all_content.append(content)
                        logger.info(f"Retrieved Jira issue: {issue.get('key', '')} with comments")
                    else:
                        logger.warning(f"Failed to retrieve Jira issue {issue_key}: {result.get('error', 'Unknown error')}")
                else:
                    logger.warning(f"Could not extract issue key from URL: {url}")
            
            return "\n".join(all_content)
            
    except Exception as e:
        logger.error(f"Error retrieving Jira content: {str(e)}")
        return f"Error retrieving Jira content: {str(e)}"

