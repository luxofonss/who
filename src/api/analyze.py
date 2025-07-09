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


def _split_csv_field(field):
    if not field:
        return []
    return [x.strip() for x in field.split(",") if x.strip()]


@router.post("/analyze")
async def analyze(
    request: AnalyzeRequest,
    db: Session = Depends(get_db_session)
):
    # Retrieve thread context from DB
    thread = db.query(ProjectThread).filter(ProjectThread.thread_id == request.thread_id).first()
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
            confluence_content = await retrieve_confluence_content(confluence_urls, request.thread_id)
            requirements_txt += confluence_content
        # Retrieve Jira content
        jira_content = ""
        commits = []
        if jira_links:
            jira_content = await retrieve_jira_content(jira_links, request.thread_id)
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
                        max_diff_length = 5000  # Limit to 5000 characters
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

    # Run the analyzer
    analyzer = AnalyzerChain(project.project_id)
    logger.info(f"Running analysis for endpoint: {api_path}")
    logger.info(f"Method: {api_method}")
    logger.info(f"User query: {request.user_query}")
    endpoint = {
        "path": api_path,
        "method": api_method
    }
    result = await analyzer.run(
        endpoint=str,
        requirements_txt=requirements_txt,
        user_text=request.user_query,
        code_commit=code_commit,
        changed_methods=changed_methods
    )

    # Save chat history
    try:
        # Save user message
        user_message_id = f"msg_{uuid.uuid4().hex[:8]}"
        user_chat = ChatHistory(
            message_id=user_message_id,
            thread_id=request.thread_id,
            role="user",
            content=request.user_query,
            analysis_result=None
        )
        db.add(user_chat)
        
        # Save assistant message with analysis result
        assistant_message_id = f"msg_{uuid.uuid4().hex[:8]}"
        assistant_chat = ChatHistory(
            message_id=assistant_message_id,
            thread_id=request.thread_id,
            role="assistant",
            content=result.get("document", "Analysis completed"),
            analysis_result=result
        )
        db.add(assistant_chat)
        
        # Update thread's message count and last activity
        thread.message_count += 2  # User + Assistant messages
        thread.last_activity = datetime.datetime.now(datetime.timezone.utc)
        
        db.commit()
        logger.info(f"Saved chat history for thread {request.thread_id}")
        
    except Exception as e:
        logger.error(f"Error saving chat history: {str(e)}")
        db.rollback()
        # Continue without failing the analysis

    return result


@router.get("/analyze/help")
async def get_analyze_help():
    """Get help information for the analyze API query format"""
    return {
        "help": format_query_help(),
        "examples": [
            {
                "description": "Basic API analysis with Confluence and Jira",
                "query": "analyze api @endpoint=/api/v1/quizzes @method=POST @jira=https://company.atlassian.net/browse/QUIZ-123 @confluence=https://company.atlassian.net/wiki/spaces/API/pages/123 This API creates new quizzes"
            },
            {
                "description": "Multiple sources analysis",
                "query": "analyze api @endpoint=/api/v1/users @method=GET @jira=https://company.atlassian.net/browse/USER-456,https://company.atlassian.net/browse/USER-789 @confluence=https://company.atlassian.net/wiki/spaces/API/pages/456,https://company.atlassian.net/wiki/spaces/DOC/pages/789 User management API"
            }
        ],
        "environment_setup": {
            "confluence": {
                "required_vars": ["CONFLUENCE_BASE_URL", "CONFLUENCE_USERNAME", "CONFLUENCE_API_TOKEN"],
                "optional_vars": ["CONFLUENCE_SPACE_KEYS", "CONFLUENCE_MAX_RESULTS", "CONFLUENCE_EXPAND_CONTENT"]
            },
            "jira": {
                "required_vars": ["JIRA_BASE_URL", "JIRA_USERNAME", "JIRA_API_TOKEN"],
                "optional_vars": ["JIRA_PROJECT_KEYS", "JIRA_MAX_RESULTS", "JIRA_EXPAND_FIELDS", "JIRA_GIT_REPOS", "JIRA_USE_BITBUCKET"]
            },
            "bitbucket": {
                "required_vars": ["BITBUCKET_USERNAME", "BITBUCKET_WORKSPACE", "BITBUCKET_APP_PASSWORD or BITBUCKET_ACCESS_TOKEN"],
                "optional_vars": ["BITBUCKET_BASE_URL", "BITBUCKET_REPOSITORIES", "BITBUCKET_MAX_RESULTS"]
            }
        }
    } 