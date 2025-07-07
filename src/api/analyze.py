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
from services.bitbucket_mcp_service import BitbucketMCPService, BitbucketMCPConfigBuilder

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
                        
                        if issue['commits']:
                            content += f"\nCommits ({len(issue['commits'])} total):\n"
                            for i, commit in enumerate(issue['commits'], 1):
                                content += f"\n--- Commit {i} ---\n"
                                content += f"Repository: {commit['repository']}\n"
                                content += f"Hash: {commit['commit_hash']}\n"
                                content += f"Author: {commit['author']}"
                                if commit.get('author_email'):
                                    content += f" ({commit['author_email']})"
                                content += f"\n"
                                content += f"Date: {commit['date']}\n"
                                content += f"Message: {commit['message']}\n"
                                content += f"Files Changed: {commit['files_changed']}\n"
                                
                                # Include code changes if available
                                code_changes = commit.get('code_changes', {})
                                if code_changes.get('status') == 'success':
                                    source = code_changes.get('source', 'unknown')
                                    content += f"\nCode Changes (via {source}):\n"
                                    
                                    # Show file summary if available
                                    if 'total_files' in code_changes:
                                        content += f"Files changed: {code_changes['total_files']}\n"
                                    
                                    content += f"```diff\n{code_changes.get('diff', '')}\n```\n"
                                elif code_changes.get('status') == 'no_repo_config':
                                    content += f"[Code changes not available - no repository configuration]\n"
                                elif code_changes.get('status') == 'repo_not_found':
                                    content += f"[Repository '{commit['repository']}' not found in configuration]\n"
                                elif code_changes.get('status') == 'no_commit_hash':
                                    content += f"[No commit hash available]\n"
                                else:
                                    content += f"[Code changes not available: {code_changes.get('status', 'unknown error')}]\n"
                        
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


async def retrieve_commit_code_changes(commits: list, session_id: str) -> list:
    """Retrieve code changes for commits using Bitbucket MCP service"""
    if not commits:
        return commits
    
    # Check if Bitbucket is enabled
    use_bitbucket = os.getenv("JIRA_USE_BITBUCKET", "false").lower() == "true"
    if not use_bitbucket:
        logger.info("Bitbucket integration disabled, skipping commit code retrieval")
        return commits
    
    try:
        bitbucket_config = BitbucketMCPConfigBuilder.from_env()
        
        async with BitbucketMCPService(bitbucket_config) as bitbucket:
            enhanced_commits = []
            
            for commit in commits:
                commit_hash = commit.get('commit_hash', '')
                repository = commit.get('repository', '')
                
                if commit_hash and repository:
                    logger.info(f"🔗 Retrieving code changes for commit {commit_hash} from repository {repository}")
                    
                    # Get commit details from Bitbucket
                    result = await bitbucket.get_commit_by_hash(session_id, repository, commit_hash)
                    
                    if result["status"] == "success":
                        commit_data = result["data"]["commit"]
                        diff_data = commit_data.get("diff", {})
                        
                        # Add code changes to commit info
                        enhanced_commit = commit.copy()
                        if diff_data.get("status") == "success":
                            enhanced_commit['code_changes'] = {
                                'status': 'success',
                                'source': 'bitbucket_api',
                                'diff': diff_data['diff_text'],
                                'files_changed': diff_data['files_changed'],
                                'total_files': diff_data['total_files']
                            }
                        else:
                            enhanced_commit['code_changes'] = {
                                'status': 'failed',
                                'source': 'bitbucket_api',
                                'error': diff_data.get('error', 'Failed to get diff')
                            }
                        
                        enhanced_commits.append(enhanced_commit)
                    else:
                        # Failed to get commit from Bitbucket
                        enhanced_commit = commit.copy()
                        enhanced_commit['code_changes'] = {
                            'status': 'failed',
                            'source': 'bitbucket_api', 
                            'error': result.get('error', 'Failed to retrieve commit')
                        }
                        enhanced_commits.append(enhanced_commit)
                else:
                    # No commit hash or repository
                    enhanced_commit = commit.copy()
                    enhanced_commit['code_changes'] = {
                        'status': 'no_commit_info',
                        'error': 'Missing commit hash or repository information'
                    }
                    enhanced_commits.append(enhanced_commit)
            
            return enhanced_commits
            
    except Exception as e:
        logger.error(f"Error retrieving commit code changes: {str(e)}")
        # Return original commits if Bitbucket fails
        for commit in commits:
            commit['code_changes'] = {
                'status': 'error',
                'error': f'Bitbucket service error: {str(e)}'
            }
        return commits


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
            
            # Extract commits from Jira content and enhance with code changes
            if "Commits (" in jira_content:
                logger.info("Found commits in Jira content, retrieving code changes...")
                
                # Re-fetch Jira issues to get commit data
                config = JiraMCPConfigBuilder.from_env()
                async with JiraMCPService(config) as jira:
                    all_commits = []
                    
                    for url in parsed_query.jira_urls:
                        issue_key = extract_jira_issue_key(url)
                        if issue_key:
                            result = await jira.get_issue_by_key(session_id, issue_key)
                            if result["status"] == "success":
                                issue = result["data"]["issue"]
                                commits = issue.get('commits', [])
                                if commits:
                                    all_commits.extend(commits)
                    
                    # Enhance commits with code changes using Bitbucket
                    if all_commits:
                        enhanced_commits = await retrieve_commit_code_changes(all_commits, session_id)
                        
                        # Update the testcases_txt with enhanced commit information
                        enhanced_content_parts = []
                        for url in parsed_query.jira_urls:
                            issue_key = extract_jira_issue_key(url)
                            if issue_key:
                                result = await jira.get_issue_by_key(session_id, issue_key)
                                if result["status"] == "success":
                                    issue = result["data"]["issue"]
                                    
                                    # Build enhanced content for this issue
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
                                    
                                    # Add comments
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
                                    
                                    # Add enhanced commits
                                    issue_commits = [c for c in enhanced_commits if any(ic['commit_hash'] == c['commit_hash'] for ic in issue.get('commits', []))]
                                    if issue_commits:
                                        content += f"\nCommits ({len(issue_commits)} total):\n"
                                        for i, commit in enumerate(issue_commits, 1):
                                            content += f"\n--- Commit {i} ---\n"
                                            content += f"Repository: {commit['repository']}\n"
                                            content += f"Hash: {commit['commit_hash']}\n"
                                            content += f"Author: {commit['author']}"
                                            if commit.get('author_email'):
                                                content += f" ({commit['author_email']})"
                                            content += f"\n"
                                            content += f"Date: {commit['date']}\n"
                                            content += f"Message: {commit['message']}\n"
                                            content += f"Files Changed: {commit['files_changed']}\n"
                                            
                                            # Include code changes
                                            code_changes = commit.get('code_changes', {})
                                            if code_changes.get('status') == 'success':
                                                source = code_changes.get('source', 'unknown')
                                                content += f"\nCode Changes (via {source}):\n"
                                                if 'total_files' in code_changes:
                                                    content += f"Files changed: {code_changes['total_files']}\n"
                                                content += f"```diff\n{code_changes.get('diff', '')}\n```\n"
                                            else:
                                                error = code_changes.get('error', 'Unknown error')
                                                content += f"[Code changes not available: {error}]\n"
                                    
                                    content += "\n" + "="*50 + "\n\n"
                                    enhanced_content_parts.append(content)
                        
                        # Update testcases_txt with enhanced content
                        if enhanced_content_parts:
                            testcases_txt = "\n".join(enhanced_content_parts)
            
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
                "optional_vars": ["JIRA_PROJECT_KEYS", "JIRA_MAX_RESULTS", "JIRA_EXPAND_FIELDS", "JIRA_GIT_REPOS", "JIRA_USE_BITBUCKET"]
            },
            "bitbucket": {
                "required_vars": ["BITBUCKET_USERNAME", "BITBUCKET_WORKSPACE", "BITBUCKET_APP_PASSWORD or BITBUCKET_ACCESS_TOKEN"],
                "optional_vars": ["BITBUCKET_BASE_URL", "BITBUCKET_REPOSITORIES", "BITBUCKET_MAX_RESULTS"]
            }
        }
    } 