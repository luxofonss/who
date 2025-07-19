import asyncio
import aiohttp
import json
import os
import re
from typing import Dict, List, Optional, Any, Union, Set    
from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import urljoin, urlparse
from loguru import logger
from pydantic import BaseModel, Field
from git import Repo, GitCommandError, InvalidGitRepositoryError
import shutil
from pathlib import Path


class MCPBitbucketConfig(BaseModel):
    """Configuration for Bitbucket MCP service"""
    base_url: str = Field(..., description="Bitbucket base URL (e.g., https://api.bitbucket.org/2.0)")
    email: str = Field(..., description="Atlassian account email for authentication")
    username: str = Field(..., description="Bitbucket username for authentication")
    app_password: Optional[str] = Field(None, description="App password for authentication")
    workspace: str = Field(..., description="Bitbucket workspace name")
    repositories: Optional[List[str]] = Field(None, description="Limit to specific repositories")
    max_results: int = Field(50, description="Max results per request")
    cache_duration: int = Field(3600, description="Cache duration in seconds")
    
    @classmethod
    def model_validate(cls, values):
        """Validate that app_password is provided"""
        if isinstance(values, dict):
            app_password = values.get('app_password')
            
            if not app_password:
                raise ValueError("app_password must be provided")
            
        return super().model_validate(values)


@dataclass
class BitbucketMCPContext:
    """MCP session context for Bitbucket caching and state management"""
    session_id: str
    cached_commits: Dict[str, Any] = field(default_factory=dict)
    cached_diffs: Dict[str, Any] = field(default_factory=dict)
    recent_searches: List[str] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    last_accessed: datetime = field(default_factory=datetime.now)


class BitbucketMCPService:
    """Main MCP service for Bitbucket integration"""
    
    def __init__(self, config: MCPBitbucketConfig):
        self.config = config
        self.session = None
        self.contexts: Dict[str, BitbucketMCPContext] = {}
        
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize_session()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close_session()
        
    async def initialize_session(self):
        """Initialize HTTP session with Basic Auth (email:token or email:app_password)"""
        auth = None
        
        if self.config.app_password:
            # Use Basic authentication with email and app password
            auth = aiohttp.BasicAuth(self.config.username, self.config.app_password)
            logger.info("🔑 Using Bitbucket app password authentication (Basic Auth)")
        
        self.session = aiohttp.ClientSession(auth=auth)
        
    async def close_session(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
            
    def get_or_create_context(self, session_id: str) -> BitbucketMCPContext:
        """Get or create MCP context for session"""
        if session_id not in self.contexts:
            self.contexts[session_id] = BitbucketMCPContext(session_id=session_id)
        return self.contexts[session_id]
        
    def build_mcp_metadata(self, session_id: str) -> Dict[str, Any]:
        """Build MCP metadata for responses"""
        return {
            "protocol_version": "1.0",
            "server_name": "bitbucket-mcp",
            "session_id": session_id,
            "base_url": self.config.base_url,
            "workspace": self.config.workspace,
            "timestamp": datetime.now().isoformat(),
            "capabilities": ["commits", "diffs", "repositories", "branches"]
        }
        
    def _extract_repo_and_commit_from_url(self, url: str) -> Optional[Dict[str, str]]:
        """Extract repository and commit hash from Bitbucket URL"""
        patterns = [
            r'/([^/]+)/([^/]+)/commits/([a-f0-9]+)',
            r'/([^/]+)/([^/]+)/src/([a-f0-9]+)',
            r'/([^/]+)/([^/]+)/commit/([a-f0-9]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return {
                    'workspace': match.group(1),
                    'repository': match.group(2),
                    'commit_hash': match.group(3)
                }
        return None
        
    async def get_commit_by_hash(self, session_id: str, repository: str, commit_hash: str) -> Dict[str, Any]:
        """Get specific commit by hash"""
        try:
            context = self.get_or_create_context(session_id)
            metadata = self.build_mcp_metadata(session_id)
            
            cache_key = f"{repository}:{commit_hash}"
            
            # Check cache first
            if cache_key in context.cached_commits:
                return {
                    "metadata": metadata,
                    "tool": "get_commit_by_hash",
                    "status": "success",
                    "data": {"commit": context.cached_commits[cache_key]}
                }
            
            # Make API request
            url = f"{self.config.base_url}/repositories/{self.config.workspace}/{repository}/commit/{commit_hash}"
            logger.info(f"Bitbucket URL: {url}")

            async with self.session.get(url) as response:
                if response.status == 200:
                    commit_data = await response.json()
                    
                    # Extract commit information
                    commit_info = {
                        'hash': commit_data.get('hash', ''),
                        'short_hash': commit_data.get('hash', '')[:7],
                        'message': commit_data.get('message', ''),
                        'summary': commit_data.get('summary', {}).get('raw', ''),
                        'author': {
                            'name': commit_data.get('author', {}).get('user', {}).get('display_name', ''),
                            'email': commit_data.get('author', {}).get('user', {}).get('email', ''),
                            'username': commit_data.get('author', {}).get('user', {}).get('username', ''),
                            'raw': commit_data.get('author', {}).get('raw', '')
                        },
                        'date': commit_data.get('date', ''),
                        'repository': repository,
                        'workspace': self.config.workspace,
                        'parents': [parent.get('hash', '') for parent in commit_data.get('parents', [])],
                        'links': commit_data.get('links', {}),
                        'type': commit_data.get('type', 'commit')
                    }
                    
                    # Get diff information
                    diff_data = await self._get_commit_diff(repository, commit_hash)
                    commit_info['diff'] = diff_data
                    
                    # Cache the commit
                    context.cached_commits[cache_key] = commit_info
                    context.last_accessed = datetime.now()
                    
                    return {
                        "metadata": metadata,
                        "tool": "get_commit_by_hash",
                        "status": "success",
                        "data": {"commit": commit_info}
                    }
                else:
                    error_text = await response.text()
                    return {
                        "metadata": metadata,
                        "tool": "get_commit_by_hash",
                        "status": "error",
                        "error": f"HTTP {response.status}: {error_text}"
                    }
                    
        except Exception as e:
            logger.error(f"Get commit by hash error: {str(e)}")
            return {
                "metadata": self.build_mcp_metadata(session_id),
                "tool": "get_commit_by_hash",
                "status": "error",
                "error": str(e)
            }
            
    async def _get_commit_diff(self, repository: str, commit_hash: str) -> Dict[str, Any]:
        """Get diff for a specific commit"""
        try:
            url = f"{self.config.base_url}/repositories/{self.config.workspace}/{repository}/diff/{commit_hash}"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    diff_text = await response.text()
                    
                    # Parse diff to extract file information
                    files_changed = self._parse_diff_files(diff_text)
                    
                    return {
                        'status': 'success',
                        'diff_text': diff_text,
                        'files_changed': files_changed,
                        'total_files': len(files_changed)
                    }
                else:
                    return {
                        'status': 'error',
                        'error': f"HTTP {response.status}"
                    }
                    
        except Exception as e:
            logger.error(f"Error getting commit diff: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
            
    def _parse_diff_files(self, diff_text: str) -> List[Dict[str, Any]]:
        """Parse diff text to extract file information"""
        files = []
        current_file = None
        
        for line in diff_text.split('\n'):
            if line.startswith('diff --git'):
                # New file diff
                if current_file:
                    files.append(current_file)
                
                # Extract file paths
                parts = line.split()
                if len(parts) >= 4:
                    old_path = parts[2][2:]  # Remove 'a/' prefix
                    new_path = parts[3][2:]  # Remove 'b/' prefix
                    
                    current_file = {
                        'old_path': old_path,
                        'new_path': new_path,
                        'status': 'modified',
                        'additions': 0,
                        'deletions': 0,
                        'chunks': []
                    }
            elif line.startswith('new file mode'):
                if current_file:
                    current_file['status'] = 'added'
            elif line.startswith('deleted file mode'):
                if current_file:
                    current_file['status'] = 'deleted'
            elif line.startswith('@@') and current_file:
                # Diff chunk header
                current_file['chunks'].append(line)
            elif line.startswith('+') and not line.startswith('+++') and current_file:
                current_file['additions'] += 1
            elif line.startswith('-') and not line.startswith('---') and current_file:
                current_file['deletions'] += 1
        
        # Add the last file
        if current_file:
            files.append(current_file)
            
        return files
        
    async def get_commits_by_urls(self, session_id: str, urls: List[str]) -> Dict[str, Any]:
        """Get multiple commits by their URLs"""
        try:
            metadata = self.build_mcp_metadata(session_id)
            commits = []
            errors = []
            
            for url in urls:
                extracted = self._extract_repo_and_commit_from_url(url)
                if extracted:
                    repository = extracted['repository']
                    commit_hash = extracted['commit_hash']
                    
                    result = await self.get_commit_by_hash(session_id, repository, commit_hash)
                    if result["status"] == "success":
                        commits.append(result["data"]["commit"])
                    else:
                        errors.append(f"Failed to get {commit_hash}: {result.get('error', 'Unknown error')}")
                else:
                    errors.append(f"Could not extract repository and commit from URL: {url}")
            
            return {
                "metadata": metadata,
                "tool": "get_commits_by_urls",
                "status": "success",
                "data": {
                    "commits": commits,
                    "total": len(commits),
                    "errors": errors
                }
            }
            
        except Exception as e:
            logger.error(f"Get commits by URLs error: {str(e)}")
            return {
                "metadata": self.build_mcp_metadata(session_id),
                "tool": "get_commits_by_urls",
                "status": "error",
                "error": str(e)
            }
            
    async def search_commits(self, session_id: str, repository: str, query: str = None, 
                           author: str = None, branch: str = None, limit: int = None) -> Dict[str, Any]:
        """Search commits in a repository"""
        try:
            context = self.get_or_create_context(session_id)
            metadata = self.build_mcp_metadata(session_id)
            
            # Build API parameters
            params = {
                'pagelen': limit or self.config.max_results
            }
            
            if query:
                params['q'] = query
            if author:
                params['author'] = author
            if branch:
                params['include'] = branch
                
            # Make API request
            url = f"{self.config.base_url}/repositories/{self.config.workspace}/{repository}/commits"
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    commits = []
                    for commit_data in data.get('values', []):
                        commit_info = {
                            'hash': commit_data.get('hash', ''),
                            'short_hash': commit_data.get('hash', '')[:7],
                            'message': commit_data.get('message', ''),
                            'summary': commit_data.get('summary', {}).get('raw', ''),
                            'author': {
                                'name': commit_data.get('author', {}).get('user', {}).get('display_name', ''),
                                'email': commit_data.get('author', {}).get('user', {}).get('email', ''),
                                'username': commit_data.get('author', {}).get('user', {}).get('username', ''),
                                'raw': commit_data.get('author', {}).get('raw', '')
                            },
                            'date': commit_data.get('date', ''),
                            'repository': repository,
                            'workspace': self.config.workspace,
                            'parents': [parent.get('hash', '') for parent in commit_data.get('parents', [])],
                            'links': commit_data.get('links', {}),
                            'type': commit_data.get('type', 'commit')
                        }
                        commits.append(commit_info)
                        
                        # Cache the commit
                        cache_key = f"{repository}:{commit_info['hash']}"
                        context.cached_commits[cache_key] = commit_info
                    
                    # Update context
                    if query:
                        context.recent_searches.append(query)
                    context.last_accessed = datetime.now()
                    
                    return {
                        "metadata": metadata,
                        "tool": "search_commits",
                        "status": "success",
                        "data": {
                            "query": query,
                            "repository": repository,
                            "commits": commits,
                            "total": len(commits),
                            "has_more": len(data.get('values', [])) >= (limit or self.config.max_results)
                        }
                    }
                else:
                    error_text = await response.text()
                    return {
                        "metadata": metadata,
                        "tool": "search_commits",
                        "status": "error",
                        "error": f"HTTP {response.status}: {error_text}"
                    }
                    
        except Exception as e:
            logger.error(f"Search commits error: {str(e)}")
            return {
                "metadata": self.build_mcp_metadata(session_id),
                "tool": "search_commits",
                "status": "error",
                "error": str(e)
            }
            
    async def get_repository_info(self, session_id: str, repository: str) -> Dict[str, Any]:
        """Get repository information"""
        try:
            metadata = self.build_mcp_metadata(session_id)
            
            url = f"{self.config.base_url}/repositories/{self.config.workspace}/{repository}"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    repo_data = await response.json()
                    
                    repo_info = {
                        'name': repo_data.get('name', ''),
                        'full_name': repo_data.get('full_name', ''),
                        'description': repo_data.get('description', ''),
                        'language': repo_data.get('language', ''),
                        'size': repo_data.get('size', 0),
                        'created_on': repo_data.get('created_on', ''),
                        'updated_on': repo_data.get('updated_on', ''),
                        'is_private': repo_data.get('is_private', False),
                        'fork_policy': repo_data.get('fork_policy', ''),
                        'main_branch': repo_data.get('mainbranch', {}).get('name', 'main'),
                        'links': repo_data.get('links', {}),
                        'workspace': self.config.workspace
                    }
                    
                    return {
                        "metadata": metadata,
                        "tool": "get_repository_info",
                        "status": "success",
                        "data": {"repository": repo_info}
                    }
                else:
                    error_text = await response.text()
                    return {
                        "metadata": metadata,
                        "tool": "get_repository_info",
                        "status": "error",
                        "error": f"HTTP {response.status}: {error_text}"
                    }
                    
        except Exception as e:
            logger.error(f"Get repository info error: {str(e)}")
            return {
                "metadata": self.build_mcp_metadata(session_id),
                "tool": "get_repository_info",
                "status": "error",
                "error": str(e)
            }
            
    def get_mcp_resources(self, session_id: str) -> Dict[str, Any]:
        """Get available MCP resources"""
        metadata = self.build_mcp_metadata(session_id)
        
        resources = [
            {
                "uri": f"bitbucket://{self.config.workspace}/commits",
                "name": "Commits",
                "description": "Access to repository commits",
                "mimeType": "application/json"
            },
            {
                "uri": f"bitbucket://{self.config.workspace}/diffs",
                "name": "Diffs",
                "description": "Access to commit diffs and code changes",
                "mimeType": "text/plain"
            },
            {
                "uri": f"bitbucket://{self.config.workspace}/repositories",
                "name": "Repositories",
                "description": "Access to repository information",
                "mimeType": "application/json"
            }
        ]
        
        return {
            "metadata": metadata,
            "tool": "get_mcp_resources",
            "status": "success",
            "data": {"resources": resources}
        }
        
    def get_mcp_prompts(self, session_id: str) -> Dict[str, Any]:
        """Get available MCP prompts"""
        metadata = self.build_mcp_metadata(session_id)
        
        prompts = [
            {
                "name": "analyze_commit",
                "description": "Analyze a specific commit and its changes",
                "arguments": [
                    {"name": "repository", "description": "Repository name", "required": True},
                    {"name": "commit_hash", "description": "Commit hash", "required": True}
                ]
            },
            {
                "name": "compare_commits",
                "description": "Compare changes between two commits",
                "arguments": [
                    {"name": "repository", "description": "Repository name", "required": True},
                    {"name": "from_commit", "description": "From commit hash", "required": True},
                    {"name": "to_commit", "description": "To commit hash", "required": True}
                ]
            },
            {
                "name": "search_code_changes",
                "description": "Search for specific code changes across commits",
                "arguments": [
                    {"name": "repository", "description": "Repository name", "required": True},
                    {"name": "query", "description": "Search query", "required": True},
                    {"name": "author", "description": "Filter by author", "required": False}
                ]
            }
        ]
        
        return {
            "metadata": metadata,
            "tool": "get_mcp_prompts",
            "status": "success",
            "data": {"prompts": prompts}
        }
        
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get session information"""
        context = self.get_or_create_context(session_id)
        metadata = self.build_mcp_metadata(session_id)
        
        return {
            "metadata": metadata,
            "tool": "get_session_info",
            "status": "success",
            "data": {
                "session_id": session_id,
                "cached_commits": len(context.cached_commits),
                "cached_diffs": len(context.cached_diffs),
                "recent_searches": context.recent_searches[-10:],  # Last 10 searches
                "last_accessed": context.last_accessed.isoformat(),
                "user_preferences": context.user_preferences
            }
        }

    async def clone_repository(self, session_id: str, repository: str, branch: str = "main", target_path: Path = None):
        if target_path is None:
            target_path = Path("storage/repos") / repository
        
        # URL encode the username and password/token to handle special characters
        from urllib.parse import quote
        encoded_username = quote(self.config.username, safe='')
        encoded_auth = quote(self.config.app_password or '', safe='')
        repo_url = f"https://{encoded_username}:{encoded_auth}@bitbucket.org/{self.config.workspace}/{repository}.git"
        try:
            if target_path.exists():
                try:
                    repo = Repo(target_path)
                    repo.git.fetch()
                    repo.git.checkout(branch)
                    repo.git.pull()
                except (GitCommandError, InvalidGitRepositoryError):
                    shutil.rmtree(target_path)
                    repo = Repo.clone_from(repo_url, target_path, branch=branch)
            else:
                repo = Repo.clone_from(repo_url, target_path, branch=branch)
            sha = repo.head.commit.hexsha
            return {"status": "success", "data": {"commit_hash": sha}}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def extract_changed_methods(self, diff_text: str) -> List[Dict[str, str]]:
        """
        Extract all methods that have changes in a git diff.
        Returns a list of strings with both class names and class.element format
        """
        logger.info(f"Extracting changed methods from diff text: {diff_text}")
        changed_methods = []
        
        # Split diff into file sections
        file_sections = re.split(r'^diff --git', diff_text, flags=re.MULTILINE)
        
        for section in file_sections:
            if not section.strip():
                continue
                
            # Extract file path and class name
            file_path = self._extract_file_path(section)
            if not file_path or not file_path.endswith('.java'):
                continue
                
            class_name = self._extract_class_name(section, file_path)
            methods = self._find_changed_methods_in_section(section, class_name)
            # logger.info(f"Methods: {methods}")
            
            changed_methods.append({"class": class_name, "method": None})
            
            # Add all the specific method/field changes
            changed_methods.extend(methods)
        
        return sorted(list(changed_methods), key=lambda x: x["class"])

    def _extract_file_path(self, section: str) -> str:
        """Extract file path from diff section header"""
        # Look for the +++ line which contains the new file path
        for line in section.split('\n'):
            if line.startswith('+++'):
                # Remove +++ and any a/ or b/ prefix
                path = line[4:].strip()
                if path.startswith('a/') or path.startswith('b/'):
                    path = path[2:]
                return path
        return ""

    def _extract_class_name(self,section: str, file_path: str) -> str:
        """Extract class name from diff section or file path"""
        # First try to find class declaration in the diff
        class_pattern = re.compile(r'^\s*(?:public\s+)?(?:final\s+)?(?:class|interface|record)\s+(\w+)', re.MULTILINE)
        
        for line in section.split('\n'):
            # Look in both added and removed lines, and context lines
            clean_line = line[1:] if line.startswith(('+', '-', ' ')) else line
            match = class_pattern.match(clean_line.strip())
            if match:
                return match.group(1)
        
        # Fallback to filename
        return file_path.split('/')[-1].replace('.java', '')

    def _find_changed_methods_in_section(self,section: str, class_name: str) -> List[Dict[str, str]]:
        """Find all methods that have changes in this file section"""
        changed_methods = []
        
        # Pattern to match method signatures
        method_pattern = re.compile(
            r'^\s*(?:public|private|protected)?\s*(?:static\s+)?(?:final\s+)?'
            r'(?:synchronized\s+)?(?:abstract\s+)?'
            r'(?:<[^>]+>\s+)?'  # Generic type parameters
            r'(?:[\w\[\]<>.,\s]+\s+)'  # Return type
            r'(\w+)\s*\('  # Method name
        )
        
        lines = section.split('\n')
        current_context = None
        current_method = None
        brace_depth = 0
        
        for i, line in enumerate(lines):
            # Skip file header lines
            if line.startswith(('diff --git', 'index', '---', '+++')):
                continue
                
            # Check if this is a hunk header
            if line.startswith('@@'):
                current_method = None
                current_context = None
                brace_depth = 0
                continue
            
            # Only process lines that are part of the diff (added, removed, or context)
            if not line.startswith(('+', '-', ' ')):
                continue
                
            clean_line = line[1:] if line.startswith(('+', '-', ' ')) else line
            
            # Update brace depth to track context
            brace_depth += clean_line.count('{') - clean_line.count('}')
            
            # Check if this line contains a method signature
            method_match = method_pattern.match(clean_line.strip())
            if method_match:
                method_name = method_match.group(1)
                current_method = method_name
                current_context = 'method'
                
                # If this is a changed line (method signature changed), mark it
                if line.startswith(('+', '-')):
                    changed_methods.append({
                        "class": class_name,
                        "method": method_name
                    })
            
            # If we see a change line, determine what changed
            elif line.startswith(('+', '-')):
                clean_stripped = clean_line.strip()
                
                # Skip empty lines and pure structural changes
                if not clean_stripped or clean_stripped in ['{', '}']:
                    continue
                
                # If we're in a method, attribute change to that method
                if current_method and brace_depth > 0:
                    changed_methods.append({
                        "class": class_name,
                        "method": current_method
                    })
                else:
                    # This is a class-level change (field, annotation, etc.)
                    # Look for the nearest field or method to attribute it to
                    nearest_element = self._find_nearest_element(lines, i, class_name)
                    if nearest_element:
                        changed_methods.append({
                            "class": class_name,
                            "method": nearest_element
                        })
                    else:
                        # If no specific element found, mark as class-level change
                        changed_methods.append({
                            "class": class_name,
                            "method": None
                        })
            
            # Update current method context based on brace depth
            if current_method and brace_depth == 0:
                current_method = None
                current_context = None

        return changed_methods

    def _find_nearest_element(self, lines: List[str], change_idx: int, class_name: str) -> str:
        """Find the nearest field or method to attribute a class-level change to"""
        
        # Patterns to match fields and methods
        field_pattern = re.compile(r'^\s*(?:private|public|protected)?\s*(?:static\s+)?(?:final\s+)?[\w\[\]<>.,\s]+\s+(\w+)\s*[;=]')
        method_pattern = re.compile(r'^\s*(?:public|private|protected)?\s*(?:static\s+)?(?:final\s+)?(?:synchronized\s+)?(?:abstract\s+)?(?:<[^>]+>\s+)?(?:[\w\[\]<>.,\s]+\s+)(\w+)\s*\(')
        
        # Look forward and backward from the change
        for offset in range(1, min(10, len(lines) - change_idx)):
            # Check lines after the change
            if change_idx + offset < len(lines):
                line = lines[change_idx + offset]
                if line.startswith(('+', '-', ' ')):
                    clean_line = line[1:] if line.startswith(('+', '-', ' ')) else line
                    
                    # Try to match field
                    field_match = field_pattern.match(clean_line.strip())
                    if field_match:
                        return field_match.group(1)  # Return only the field name
                    
                    # Try to match method
                    method_match = method_pattern.match(clean_line.strip())
                    if method_match:
                        return method_match.group(1)  # Return only the method name
            
            # Check lines before the change
            if change_idx - offset >= 0:
                line = lines[change_idx - offset]
                if line.startswith(('+', '-', ' ')):
                    clean_line = line[1:] if line.startswith(('+', '-', ' ')) else line
                    
                    # Try to match field
                    field_match = field_pattern.match(clean_line.strip())
                    if field_match:
                        return field_match.group(1)  # Return only the field name
                    
                    # Try to match method
                    method_match = method_pattern.match(clean_line.strip())
                    if method_match:
                        return method_match.group(1)  # Return only the method name
        
        return ""

class BitbucketMCPConfigBuilder:
    """Builder for Bitbucket MCP configuration"""
    
    @staticmethod
    def from_env() -> MCPBitbucketConfig:
        """Build configuration from environment variables"""
        base_url = os.getenv("BITBUCKET_BASE_URL", "https://api.bitbucket.org/2.0")
        
        # Parse repositories configuration
        repositories = None
        if os.getenv("BITBUCKET_REPOSITORIES"):
            repositories = [repo.strip() for repo in os.getenv("BITBUCKET_REPOSITORIES").split(",")]
        
        # Get authentication credentials
        email = os.getenv("BITBUCKET_EMAIL", "")
        username = os.getenv("BITBUCKET_USERNAME", "")
        app_password = os.getenv("BITBUCKET_APP_PASSWORD")
        
        return MCPBitbucketConfig(
            base_url=base_url,
            email=email,
            username=username,
            app_password=app_password,
            workspace=os.getenv("BITBUCKET_WORKSPACE", ""),
            repositories=repositories,
            max_results=int(os.getenv("BITBUCKET_MAX_RESULTS", "50")),
            cache_duration=int(os.getenv("BITBUCKET_CACHE_DURATION", "3600"))
        )
    
    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> MCPBitbucketConfig:
        """Build configuration from dictionary"""
        return MCPBitbucketConfig(**config_dict)


def print_bitbucket_environment_status():
    """Print current Bitbucket environment configuration status"""
    required_vars = ["BITBUCKET_EMAIL", "BITBUCKET_USERNAME", "BITBUCKET_WORKSPACE"]
    auth_vars = ["BITBUCKET_APP_PASSWORD"]
    optional_vars = ["BITBUCKET_BASE_URL", "BITBUCKET_REPOSITORIES", "BITBUCKET_MAX_RESULTS", "BITBUCKET_CACHE_DURATION"]
    
    print("🔧 Bitbucket MCP Service Environment Status")
    print("=" * 50)
    
    # Check required variables
    for var in required_vars:
        value = os.getenv(var)
        status = "✅" if value else "❌"
        print(f"{status} {var}: {value or 'Not set'}")
    
    # Check authentication (either app password or API token)
    print("\nAuthentication:")
    app_password = os.getenv("BITBUCKET_APP_PASSWORD")
    
    if app_password:
        print("✅ BITBUCKET_APP_PASSWORD: ***")
    else:
        print("❌ BITBUCKET_APP_PASSWORD: Not set")
        print("   -> App password is required for authentication")
    
    print("\nOptional variables:")
    for var in optional_vars:
        value = os.getenv(var)
        status = "✅" if value else "⚪"
        print(f"{status} {var}: {value or 'Not set'}")
    
    print("\n📝 Authentication Setup (choose one):")
    print("\n🔑 Option 1: App Password (recommended for personal use)")
    print("1. Go to https://bitbucket.org/account/settings/app-passwords/")
    print("2. Create app password with 'Repositories: Read' permission")
    print("3. Set BITBUCKET_EMAIL and BITBUCKET_APP_PASSWORD")
    
    print("\n🔑 Option 2: API Token (for scripting and automation)")
    print("1. Go to https://bitbucket.org/account/settings/api")
    print("2. Create API token with 'Repositories: Read' permission")
    print("3. Set BITBUCKET_EMAIL")
    
    print("\n💡 Important: Use your Atlassian account email AND username")
    print("4. Set BITBUCKET_USERNAME to your Bitbucket username")
    print("5. Set BITBUCKET_WORKSPACE to your workspace name")


if __name__ == "__main__":
    print_bitbucket_environment_status() 