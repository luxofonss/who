"""
Jira MCP Service

A Model Context Protocol (MCP) service for retrieving issue information from Atlassian Jira.
This service implements the MCP standard to provide structured, context-aware access to Jira issues.
"""

import asyncio
import aiohttp
import json
import os
import re
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import urljoin, urlparse
from loguru import logger
from pydantic import BaseModel, Field


class MCPJiraConfig(BaseModel):
    """Configuration for Jira MCP service"""
    base_url: str = Field(..., description="Jira base URL")
    username: str = Field(..., description="Username for authentication")
    api_token: str = Field(..., description="API token for authentication")
    project_keys: Optional[List[str]] = Field(None, description="Limit to specific projects")
    max_results: int = Field(50, description="Max results per request")
    expand_fields: List[str] = Field(["description", "comments", "attachment"], description="Fields to expand")
    include_commits: bool = Field(True, description="Whether to retrieve commit information")
    cache_duration: int = Field(3600, description="Cache duration in seconds")


@dataclass
class JiraMCPContext:
    """MCP session context for Jira caching and state management"""
    session_id: str
    cached_issues: Dict[str, Any] = field(default_factory=dict)
    recent_searches: List[str] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    last_accessed: datetime = field(default_factory=datetime.now)


class JiraMCPService:
    """Main MCP service for Jira integration"""
    
    def __init__(self, config: MCPJiraConfig):
        self.config = config
        self.session = None
        self.contexts: Dict[str, JiraMCPContext] = {}
        
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize_session()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close_session()
        
    async def initialize_session(self):
        """Initialize HTTP session"""
        auth = aiohttp.BasicAuth(self.config.username, self.config.api_token)
        self.session = aiohttp.ClientSession(auth=auth)
        
    async def close_session(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
            
    def get_or_create_context(self, session_id: str) -> JiraMCPContext:
        """Get or create MCP context for session"""
        if session_id not in self.contexts:
            self.contexts[session_id] = JiraMCPContext(session_id=session_id)
        return self.contexts[session_id]
        
    def build_mcp_metadata(self, session_id: str) -> Dict[str, Any]:
        """Build MCP metadata for responses"""
        return {
            "protocol_version": "1.0",
            "server_name": "jira-mcp",
            "session_id": session_id,
            "base_url": self.config.base_url,
            "timestamp": datetime.now().isoformat(),
            "capabilities": ["search", "retrieve", "resources", "prompts"]
        }
        
    def _extract_issue_key_from_url(self, url: str) -> Optional[str]:
        """Extract issue key from Jira URL"""
        patterns = [
            r'/browse/([A-Z]+-\d+)',
            r'/([A-Z]+-\d+)$',
            r'selectedIssue=([A-Z]+-\d+)'
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
        
    def _build_jql_query(self, query: str, project_key: str = None, issue_type: str = None) -> str:
        """Build JQL (Jira Query Language) query"""
        jql_parts = []
        
        # Add text search
        if query:
            jql_parts.append(f'(summary ~ "{query}" OR description ~ "{query}")')
        
        # Add project filter
        if project_key:
            jql_parts.append(f'project = "{project_key}"')
        elif self.config.project_keys:
            project_list = ', '.join([f'"{pk}"' for pk in self.config.project_keys])
            jql_parts.append(f'project in ({project_list})')
            
        # Add issue type filter
        if issue_type:
            jql_parts.append(f'issuetype = "{issue_type}"')
            
        return " AND ".join(jql_parts) if jql_parts else "project is not empty"
        
    async def search_issues(self, session_id: str, query: str, project_key: str = None, 
                           issue_type: str = None, limit: int = None) -> Dict[str, Any]:
        """Search Jira issues using JQL"""
        try:
            context = self.get_or_create_context(session_id)
            metadata = self.build_mcp_metadata(session_id)
            
            # Build JQL query
            jql = self._build_jql_query(query, project_key, issue_type)
            
            # API parameters
            params = {
                'jql': jql,
                'maxResults': limit or self.config.max_results,
                'expand': ','.join(self.config.expand_fields),
                'fields': 'summary,description,issuetype,priority,status,assignee,reporter,created,updated,project,comment,attachment'
            }
            
            # Make API request
            url = urljoin(self.config.base_url, '/rest/api/2/search')
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Process results
                    issues = []
                    for issue in data.get('issues', []):
                        fields = issue.get('fields', {})
                        issue_data = {
                            'key': issue['key'],
                            'id': issue['id'],
                            'summary': fields.get('summary', ''),
                            'description': fields.get('description', ''),
                            'issue_type': fields.get('issuetype', {}).get('name', ''),
                            'priority': fields.get('priority', {}).get('name', ''),
                            'status': fields.get('status', {}).get('name', ''),
                            'assignee': fields.get('assignee', {}).get('displayName', 'Unassigned'),
                            'reporter': fields.get('reporter', {}).get('displayName', ''),
                            'created': fields.get('created', ''),
                            'updated': fields.get('updated', ''),
                            'project_key': fields.get('project', {}).get('key', ''),
                            'project_name': fields.get('project', {}).get('name', ''),
                            'url': f"{self.config.base_url}/browse/{issue['key']}",
                            'comments': self._extract_comments(fields.get('comment', {})),
                            'attachments': self._extract_attachments(fields.get('attachment', [])),
                            'commits': await self._extract_commits_from_issue(issue['key'], issue['id']) if self.config.include_commits else []
                        }
                        issues.append(issue_data)
                        
                        # Cache the issue
                        context.cached_issues[issue['key']] = issue_data
                    
                    # Update context
                    context.recent_searches.append(query)
                    context.last_accessed = datetime.now()
                    
                    return {
                        "metadata": metadata,
                        "tool": "search_issues",
                        "status": "success",
                        "data": {
                            "query": query,
                            "jql": jql,
                            "issues": issues,
                            "total": data.get('total', len(issues))
                        }
                    }
                else:
                    error_text = await response.text()
                    return {
                        "metadata": metadata,
                        "tool": "search_issues",
                        "status": "error",
                        "error": f"HTTP {response.status}: {error_text}"
                    }
                    
        except Exception as e:
            logger.error(f"Search issues error: {str(e)}")
            return {
                "metadata": self.build_mcp_metadata(session_id),
                "tool": "search_issues",
                "status": "error",
                "error": str(e)
            }
            
    async def get_issue_by_key(self, session_id: str, issue_key: str) -> Dict[str, Any]:
        """Get specific issue by key"""
        try:
            context = self.get_or_create_context(session_id)
            metadata = self.build_mcp_metadata(session_id)
            
            # Check cache first
            if issue_key in context.cached_issues:
                return {
                    "metadata": metadata,
                    "tool": "get_issue_by_key",
                    "status": "success",
                    "data": {"issue": context.cached_issues[issue_key]}
                }
            
            # API parameters
            params = {
                'expand': ','.join(self.config.expand_fields),
                'fields': 'summary,description,issuetype,priority,status,assignee,reporter,created,updated,project,comment,attachment'
            }
            
            # Make API request
            url = urljoin(self.config.base_url, f'/rest/api/2/issue/{issue_key}')
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    issue = await response.json()
                    fields = issue.get('fields', {})
                    logger.debug(f"Available fields in Jira response: {list(fields.keys())}")
                    # logger.debug(f"Comment field content: {fields.get('comment', 'NOT FOUND')}")
                    
                    issue_data = {
                        'key': issue['key'],
                        'id': issue['id'],
                        'summary': fields.get('summary', ''),
                        'description': fields.get('description', ''),
                        'issue_type': fields.get('issuetype', {}).get('name', ''),
                        'priority': fields.get('priority', {}).get('name', ''),
                        'status': fields.get('status', {}).get('name', ''),
                        'assignee': fields.get('assignee', {}).get('displayName', 'Unassigned'),
                        'reporter': fields.get('reporter', {}).get('displayName', ''),
                        'created': fields.get('created', ''),
                        'updated': fields.get('updated', ''),
                        'project_key': fields.get('project', {}).get('key', ''),
                        'project_name': fields.get('project', {}).get('name', ''),
                        'url': f"{self.config.base_url}/browse/{issue['key']}",
                        'comments': self._extract_comments(fields.get('comment', {})),
                        'attachments': self._extract_attachments(fields.get('attachment', [])),
                        'commits': await self._extract_commits_from_issue(issue['key'], issue['id']) if self.config.include_commits else []
                    }
                    
                    # Cache the issue
                    context.cached_issues[issue_key] = issue_data
                    context.last_accessed = datetime.now()
                    
                    return {
                        "metadata": metadata,
                        "tool": "get_issue_by_key",
                        "status": "success",
                        "data": {"issue": issue_data}
                    }
                else:
                    error_text = await response.text()
                    return {
                        "metadata": metadata,
                        "tool": "get_issue_by_key",
                        "status": "error",
                        "error": f"HTTP {response.status}: {error_text}"
                    }
                    
        except Exception as e:
            logger.error(f"Get issue by key error: {str(e)}")
            return {
                "metadata": self.build_mcp_metadata(session_id),
                "tool": "get_issue_by_key",
                "status": "error",
                "error": str(e)
            }
            
    async def get_issues_by_urls(self, session_id: str, urls: List[str]) -> Dict[str, Any]:
        """Get multiple issues by their URLs"""
        try:
            metadata = self.build_mcp_metadata(session_id)
            issues = []
            errors = []
            
            for url in urls:
                issue_key = self._extract_issue_key_from_url(url)
                if issue_key:
                    result = await self.get_issue_by_key(session_id, issue_key)
                    if result["status"] == "success":
                        issues.append(result["data"]["issue"])
                    else:
                        errors.append(f"Failed to get {issue_key}: {result.get('error', 'Unknown error')}")
                else:
                    errors.append(f"Could not extract issue key from URL: {url}")
            
            return {
                "metadata": metadata,
                "tool": "get_issues_by_urls",
                "status": "success",
                "data": {
                    "issues": issues,
                    "total": len(issues),
                    "errors": errors
                }
            }
            
        except Exception as e:
            logger.error(f"Get issues by URLs error: {str(e)}")
            return {
                "metadata": self.build_mcp_metadata(session_id),
                "tool": "get_issues_by_urls",
                "status": "error",
                "error": str(e)
            }
            
    def _extract_comments(self, comment_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract comments from Jira issue with enhanced details"""
        comments = []
        if comment_data and 'comments' in comment_data:
            for comment in comment_data['comments']:
                # Extract author information
                author_info = comment.get('author', {})
                
                # Extract comment body (handle both text and ADF format)
                body = ''
                if 'body' in comment:
                    if isinstance(comment['body'], str):
                        body = comment['body']
                    elif isinstance(comment['body'], dict):
                        # Handle Atlassian Document Format (ADF)
                        body = self._extract_text_from_adf(comment['body'])
                
                # Extract update author if different from original author
                update_author_info = comment.get('updateAuthor', {})
                
                comment_info = {
                    'id': comment.get('id', ''),
                    'author': author_info.get('displayName', 'Unknown'),
                    'author_email': author_info.get('emailAddress', ''),
                    'author_account_id': author_info.get('accountId', ''),
                    'body': body,
                    'created': comment.get('created', ''),
                    'updated': comment.get('updated', ''),
                    'update_author': update_author_info.get('displayName', '') if update_author_info else '',
                    'visibility': comment.get('visibility', {}),  # For restricted comments
                    'is_internal': bool(comment.get('visibility')),  # True if comment has visibility restrictions
                }
                comments.append(comment_info)
        
        # Sort comments by creation date (oldest first)
        comments.sort(key=lambda x: x['created'])
        return comments
    
    def _extract_text_from_adf(self, adf_content: Dict[str, Any]) -> str:
        """Extract plain text from Atlassian Document Format (ADF)"""
        if not isinstance(adf_content, dict):
            return str(adf_content)
        
        text_parts = []
        
        def extract_text_recursive(node):
            if isinstance(node, dict):
                # Handle text nodes
                if node.get('type') == 'text':
                    text_parts.append(node.get('text', ''))
                # Handle other node types with content
                elif 'content' in node:
                    for child in node['content']:
                        extract_text_recursive(child)
                # Handle nodes with text directly
                elif 'text' in node:
                    text_parts.append(node['text'])
            elif isinstance(node, list):
                for item in node:
                    extract_text_recursive(item)
        
        extract_text_recursive(adf_content)
        return ' '.join(text_parts).strip()
        
    async def _get_issue_numeric_id(self, issue_key: str) -> Optional[str]:
        """Get numeric issue ID from issue key"""
        try:
            url = urljoin(self.config.base_url, f'/rest/api/2/issue/{issue_key}')
            params = {'fields': 'id'}  # Only get the ID field for efficiency
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    issue_data = await response.json()
                    issue_id = issue_data.get('id')
                    logger.debug(f"Got numeric ID for {issue_key}: {issue_id}")
                    return issue_id
                else:
                    logger.warning(f"Failed to get numeric ID for {issue_key}: HTTP {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting numeric ID for {issue_key}: {str(e)}")
            return None
        
    async def _extract_commits_from_issue(self, issue_key: str, issue_id: str = None) -> List[Dict[str, Any]]:
        """Extract commit information from Jira issue using development panel"""
        try:
            commits = []
            
            # Get numeric issue ID if not provided
            if not issue_id:
                issue_id = await self._get_issue_numeric_id(issue_key)
                if not issue_id:
                    logger.warning(f"Could not get numeric ID for {issue_key} - skipping commit extraction")
                    return []
            
            # Try different development panel API endpoints with numeric issue ID
            # Some Jira instances use different application types
            api_endpoints = [
                f'/rest/dev-status/1.0/issue/detail?issueId={issue_id}&applicationType=bitbucket&dataType=repository',
                f'/rest/dev-status/1.0/issue/detail?issueId={issue_id}&applicationType=stash&dataType=repository',
                f'/rest/dev-status/1.0/issue/detail?issueId={issue_id}&dataType=repository'
            ]
            
            for endpoint in api_endpoints:
                url = urljoin(self.config.base_url, endpoint)
                logger.debug(f"Trying development panel API: {endpoint}")
                
                async with self.session.get(url) as response:
                    if response.status == 200:
                        dev_data = await response.json()
                        logger.debug(f"Development data for {issue_key} (ID: {issue_id}): {dev_data}")
                        
                        # Extract commit information from development panel
                        details = dev_data.get('detail', [])
                        for detail in details:
                            repositories = detail.get('repositories', [])
                            for repo in repositories:
                                repo_name = repo.get('name', '')
                                repo_commits = repo.get('commits', [])
                                
                                for commit in repo_commits:
                                    commit_info = {
                                        'repository': repo_name,
                                        'commit_hash': commit.get('id', ''),
                                        'message': commit.get('message', ''),
                                        'author': commit.get('author', {}).get('name', ''),
                                        'author_email': commit.get('author', {}).get('emailAddress', ''),
                                        'date': commit.get('authorTimestamp', ''),
                                        'url': commit.get('url', ''),
                                        'files_changed': commit.get('fileCount', 0)
                                    }
                                    commits.append(commit_info)
                        
                        # If we found commits, break from trying other endpoints
                        if commits:
                            logger.info(f"Found {len(commits)} commits for {issue_key} (ID: {issue_id}) using endpoint: {endpoint}")
                            break
                            
                    elif response.status == 404:
                        logger.debug(f"Endpoint {endpoint} returned 404 for {issue_key} (ID: {issue_id})")
                    elif response.status == 400:
                        logger.debug(f"Endpoint {endpoint} returned 400 for {issue_key} (ID: {issue_id})")
                    elif response.status == 403:
                        logger.debug(f"Endpoint {endpoint} returned 403 for {issue_key} (ID: {issue_id}) - insufficient permissions")
                    else:
                        logger.debug(f"Endpoint {endpoint} returned {response.status} for {issue_key} (ID: {issue_id})")
            
            if not commits:
                logger.debug(f"No commits found for {issue_key} (ID: {issue_id}) via development panel - issue may not have linked commits or development integrations may not be configured")
                    
            return commits
            
        except Exception as e:
            logger.error(f"Error extracting commits for {issue_key}: {str(e)}")
            return []
    
    async def _extract_commits_from_comments_and_links(self, issue_key: str) -> List[Dict[str, Any]]:
        """This method is no longer used - commits are in development panel, not comments"""
        logger.debug(f"Commit extraction from comments disabled for {issue_key} - commits are linked via development panel")
        return []
    
    def _extract_attachments(self, attachment_data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Extract attachments from Jira issue"""
        attachments = []
        for attachment in attachment_data:
            attachments.append({
                'filename': attachment.get('filename', ''),
                'size': attachment.get('size', 0),
                'mimeType': attachment.get('mimeType', ''),
                'author': attachment.get('author', {}).get('displayName', 'Unknown'),
                'created': attachment.get('created', ''),
                'content': attachment.get('content', '')
            })
        return attachments
        
    def get_mcp_resources(self, session_id: str) -> Dict[str, Any]:
        """Get MCP resources for session"""
        context = self.get_or_create_context(session_id)
        resources = []
        
        for issue_key, issue_data in context.cached_issues.items():
            resources.append({
                "uri": f"jira://issue/{issue_key}",
                "name": f"{issue_key}: {issue_data.get('summary', 'No summary')}",
                "description": f"Jira {issue_data.get('issue_type', 'Issue')} in {issue_data.get('project_name', 'Unknown project')}",
                "mimeType": "application/json"
            })
            
        return {
            "resources": resources,
            "total": len(resources)
        }
        
    def get_mcp_prompts(self, session_id: str) -> Dict[str, Any]:
        """Get MCP prompts for session"""
        prompts = [
            {
                "name": "summarize_jira_issue",
                "description": "Generate a summary of a Jira issue",
                "arguments": [
                    {"name": "issue_key", "description": "Jira issue key", "required": True},
                    {"name": "focus_areas", "description": "Areas to focus on", "required": False}
                ]
            },
            {
                "name": "extract_requirements_from_jira",
                "description": "Extract requirements from Jira issues",
                "arguments": [
                    {"name": "project_key", "description": "Project containing requirements", "required": True},
                    {"name": "issue_type", "description": "Issue type (Story, Epic, etc.)", "required": False}
                ]
            },
            {
                "name": "generate_test_cases_from_jira",
                "description": "Generate test cases from Jira stories/requirements",
                "arguments": [
                    {"name": "issue_key", "description": "Issue key with requirements", "required": True},
                    {"name": "endpoint", "description": "API endpoint to generate tests for", "required": True}
                ]
            }
        ]
        
        return {
            "prompts": prompts,
            "total": len(prompts)
        }
        
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get session information"""
        context = self.get_or_create_context(session_id)
        
        return {
            "session_info": {
                "session_id": session_id,
                "cached_issues_count": len(context.cached_issues),
                "recent_searches": context.recent_searches[-10:],  # Last 10 searches
                "user_preferences": context.user_preferences,
                "last_accessed": context.last_accessed.isoformat()
            }
        }


class JiraMCPConfigBuilder:
    """Helper class to build Jira MCP configuration"""
    
    @staticmethod
    def from_env() -> MCPJiraConfig:
        """Create config from environment variables"""
        project_keys = None
        if os.getenv("JIRA_PROJECT_KEYS"):
            project_keys = [s.strip() for s in os.getenv("JIRA_PROJECT_KEYS").split(",")]
            
        expand_fields = ["description", "comments", "attachment"]
        if os.getenv("JIRA_EXPAND_FIELDS"):
            expand_fields = [s.strip() for s in os.getenv("JIRA_EXPAND_FIELDS").split(",")]
        
        return MCPJiraConfig(
            base_url=os.getenv("JIRA_BASE_URL", ""),
            username=os.getenv("JIRA_USERNAME", ""),
            api_token=os.getenv("JIRA_API_TOKEN", ""),
            project_keys=project_keys,
            max_results=int(os.getenv("JIRA_MAX_RESULTS", "50")),
            expand_fields=expand_fields,
            include_commits=os.getenv("JIRA_INCLUDE_COMMITS", "true").lower() == "true",
            cache_duration=int(os.getenv("JIRA_CACHE_DURATION", "3600"))
        )
        
    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> MCPJiraConfig:
        """Create config from dictionary"""
        return MCPJiraConfig(**config_dict)


def print_jira_environment_status():
    """Print current Jira environment setup status"""
    required_vars = ["JIRA_BASE_URL", "JIRA_USERNAME", "JIRA_API_TOKEN"]
    
    print("🔧 Jira MCP Configuration Status")
    print("=" * 50)
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: {'*' * (len(value) - 4) + value[-4:]}")
        else:
            print(f"❌ {var}: Not set")
    
    optional_vars = ["JIRA_PROJECT_KEYS", "JIRA_MAX_RESULTS", "JIRA_EXPAND_FIELDS"]
    print("\nOptional Configuration:")
    for var in optional_vars:
        value = os.getenv(var)
        print(f"   {var}: {value or 'Default'}") 