from __future__ import annotations

from pathlib import Path
import os

from fastapi import APIRouter, Form, HTTPException

from loguru import logger

from utils.file import read_json, write_json, ensure_dir
from utils.query_parser import parse_analyze_query, validate_parsed_query, format_query_help, extract_confluence_page_info, extract_jira_issue_key
from services.analyzer_chain import AnalyzerChain
from services.confluence_mcp_service import ConfluenceMCPService, ConfluenceMCPConfigBuilder
from services.jira_mcp_service import JiraMCPService, JiraMCPConfigBuilder

router = APIRouter(tags=["analyze"])

STORAGE_DIR = Path("storage")


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
                
                # If still no success, try searching by title
                if (not result or result["status"] != "success") and page_info['page_title']:
                    logger.info(f"Attempting to search for page with title: {page_info['page_title']}")
                    result = await confluence.search_pages(session_id, page_info['page_title'], page_info['space_key'], limit=1)
                
                # Last resort: search using URL segments
                if not result or result["status"] != "success":
                    search_query = url.split('/')[-1].replace('-', ' ').replace('_', ' ').replace('+', ' ')
                    logger.info(f"Last resort search with query: {search_query}")
                    result = await confluence.search_pages(session_id, search_query, limit=1)
                
                if result["status"] == "success":
                    if "page" in result["data"]:
                        page = result["data"]["page"]
                        content = f"=== {page['title']} ===\n{page['content']}\n\n"
                        all_content.append(content)
                        logger.info(f"Retrieved Confluence page: {page['title']}")
                    elif "pages" in result["data"] and result["data"]["pages"]:
                        page = result["data"]["pages"][0]
                        content = f"=== {page['title']} ===\n{page['content']}\n\n"
                        all_content.append(content)
                        logger.info(f"Retrieved Confluence page: {page['title']}")
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
                    result = await jira.get_issue_by_key(session_id, issue_key)
                    
                    if result["status"] == "success":
                        issue = result["data"]["issue"]
                        content = f"=== {issue['key']}: {issue['summary']} ===\n"
                        content += f"Type: {issue['issue_type']}\n"
                        content += f"Status: {issue['status']}\n"
                        content += f"Priority: {issue['priority']}\n"
                        content += f"Assignee: {issue['assignee']}\n"
                        content += f"Reporter: {issue['reporter']}\n"
                        content += f"Created: {issue['created']}\n"
                        content += f"Updated: {issue['updated']}\n"
                        content += f"URL: {issue['url']}\n\n"
                        
                        content += f"Description:\n{issue['description']}\n\n"
                        
                        if issue['comments']:
                            content += f"Comments ({len(issue['comments'])} total):\n"
                            for i, comment in enumerate(issue['comments'], 1):
                                content += f"\n--- Comment {i} ---\n"
                                content += f"Author: {comment['author']}"
                                if comment.get('author_email'):
                                    content += f" ({comment['author_email']})"
                                content += f"\n"
                                content += f"Created: {comment['created']}\n"
                                if comment['updated'] != comment['created']:
                                    content += f"Updated: {comment['updated']}"
                                    if comment.get('update_author') and comment['update_author'] != comment['author']:
                                        content += f" by {comment['update_author']}"
                                    content += f"\n"
                                if comment.get('is_internal'):
                                    content += f"[INTERNAL COMMENT]\n"
                                content += f"Content: {comment['body']}\n"
                        else:
                            content += "No comments found.\n"
                        
                        if issue['attachments']:
                            content += f"\nAttachments ({len(issue['attachments'])} total):\n"
                            for attachment in issue['attachments']:
                                content += f"- {attachment['filename']} ({attachment['size']} bytes, {attachment['mimeType']})\n"
                        
                        content += "\n" + "="*50 + "\n\n"
                        all_content.append(content)
                        logger.info(f"Retrieved Jira issue: {issue['key']} with {len(issue['comments'])} comments")
                    else:
                        logger.warning(f"Failed to retrieve Jira issue {issue_key}: {result.get('error', 'Unknown error')}")
                else:
                    logger.warning(f"Could not extract issue key from URL: {url}")
            
            return "\n".join(all_content)
            
    except Exception as e:
        logger.error(f"Error retrieving Jira content: {str(e)}")
        return f"Error retrieving Jira content: {str(e)}"


@router.post("/analyze")
async def analyze(
    project_id: str = Form(...),
    query: str = Form(...),
):
    metadata_path = STORAGE_DIR / "metadata" / f"{project_id}.json"
    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    # Parse the query
    logger.info(f"Analyzing project {project_id} with query: {query}")
    parsed_query = parse_analyze_query(query)
    logger.info(f"Parsed query: {parsed_query}")
    
    # Validate the parsed query
    is_valid, errors = validate_parsed_query(parsed_query)
    if not is_valid:
        error_message = "Invalid query format:\n" + "\n".join(errors) + "\n\n" + format_query_help()
        raise HTTPException(status_code=400, detail=error_message)

    # Create session ID for MCP services
    session_id = f"analyze_{project_id}_{hash(query)}"

    # Retrieve content from Confluence and Jira
    requirements_txt = ""
    testcases_txt = "" #actually this is jira tickets
    
    try:
        # Retrieve Confluence content (for requirements and documentation)
        if parsed_query.confluence_urls:
            logger.info(f"Retrieving content from {len(parsed_query.confluence_urls)} Confluence URLs")
            confluence_content = await retrieve_confluence_content(parsed_query.confluence_urls, session_id)
            requirements_txt = confluence_content
        
        # Retrieve Jira content (for test cases and acceptance criteria)
        if parsed_query.jira_urls:
            logger.info(f"Retrieving content from {len(parsed_query.jira_urls)} Jira URLs")
            jira_content = await retrieve_jira_content(parsed_query.jira_urls, session_id)
            testcases_txt = jira_content
            
    except Exception as e:
        logger.error(f"Error retrieving MCP content: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving content from external sources: {str(e)}")

    # If no content retrieved, provide helpful error
    if not requirements_txt and not testcases_txt:
        raise HTTPException(
            status_code=400, 
            detail="No content could be retrieved from the provided URLs. Please check the URLs and ensure they are accessible."
        )
    logger.info(f"Requirements text: {requirements_txt}")
    logger.info(f"Jira tickets text: {testcases_txt}")
    # Run the analyzer
    analyzer = AnalyzerChain(project_id)
    logger.info(f"Running analysis for endpoint: {parsed_query.endpoint}")
    logger.info(f"Method: {parsed_query.method}")
    logger.info(f"User description: {parsed_query.user_description}")

    # Combine user description with method info
    combined_user_text = f"HTTP {parsed_query.method} {parsed_query.endpoint}"
    if parsed_query.user_description:
        combined_user_text += f" - {parsed_query.user_description}"

    result = await analyzer.run(
        endpoint=parsed_query.endpoint,
        requirements_txt=requirements_txt,
        testcases_txt=testcases_txt,
        user_text=combined_user_text,
    )

    # Add metadata about the sources
    result["analysis_metadata"] = {
        "endpoint": parsed_query.endpoint,
        "method": parsed_query.method,
        "user_description": parsed_query.user_description,
        "confluence_urls": parsed_query.confluence_urls,
        "jira_urls": parsed_query.jira_urls,
        "session_id": session_id,
        "content_sources": {
            "confluence_pages": len(parsed_query.confluence_urls),
            "jira_issues": len(parsed_query.jira_urls),
            "requirements_length": len(requirements_txt),
            "testcases_length": len(testcases_txt)
        }
    }

    # Persist analysis
    analyze_dir = STORAGE_DIR / "analyze"
    ensure_dir(analyze_dir)
    key = parsed_query.endpoint.strip("/").replace("/", "_")
    write_json(analyze_dir / f"{project_id}_{key}.json", result)

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
                "optional_vars": ["JIRA_PROJECT_KEYS", "JIRA_MAX_RESULTS", "JIRA_EXPAND_FIELDS"]
            }
        }
    } 