import os
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger
from pydantic import BaseModel, Field
from atlassian import Confluence


class MCPConfluenceConfig(BaseModel):
    """Configuration for Confluence MCP service"""
    base_url: str = Field(..., description="Confluence base URL")
    username: str = Field(..., description="Username for authentication")
    api_token: str = Field(..., description="API token for authentication")
    space_keys: Optional[List[str]] = Field(None, description="Limit to specific spaces")
    max_results: int = Field(50, description="Max results per request")
    content_format: str = Field("storage", description="Content format")
    expand_content: bool = Field(True, description="Include full content")
    cache_duration: int = Field(3600, description="Cache duration in seconds")


@dataclass
class MCPContext:
    """MCP session context for caching and state management"""
    session_id: str
    cached_resources: Dict[str, Any] = field(default_factory=dict)
    recent_searches: List[str] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    last_accessed: datetime = field(default_factory=datetime.now)


class ConfluenceMCPService:
    """Main MCP service for Confluence integration (using atlassian-python-api)"""
    
    def __init__(self, config: MCPConfluenceConfig):
        self.config = config
        self.client = Confluence(
            url=self.config.base_url,
            username=self.config.username,
            password=self.config.api_token,
            cloud=True
        )
        self.contexts: Dict[str, MCPContext] = {}
        
    async def __aenter__(self):
        """Async context manager entry"""
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        pass
        
    def get_or_create_context(self, session_id: str) -> MCPContext:
        """Get or create MCP context for session"""
        if session_id not in self.contexts:
            self.contexts[session_id] = MCPContext(session_id=session_id)
        return self.contexts[session_id]
        
    def build_mcp_metadata(self, session_id: str) -> Dict[str, Any]:
        """Build MCP metadata for responses"""
        return {
            "protocol_version": "1.0",
            "server_name": "confluence-mcp",
            "session_id": session_id,
            "base_url": self.config.base_url,
            "timestamp": datetime.now().isoformat(),
            "capabilities": ["search", "retrieve", "resources", "prompts"]
        }
        
    def _extract_page_id_from_url(self, url: str) -> Optional[str]:
        """Extract page ID from Confluence URL"""
        patterns = [
            r'pageId=(\d+)',
            r'/pages/(\d+)(?:/|$)',  # Handle pages/ID/ or pages/ID at end
            r'/pages/viewpage\.action\?pageId=(\d+)',
            r'/pages/(\d+)/[^/]*$'  # Handle pages/ID/title format
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
        
    def _build_cql_query(self, query: str, space_key: str = None, content_type: str = "page") -> str:
        """Build Confluence Query Language (CQL) query"""
        cql_parts = [f'text ~ "{query}"']
        
        if space_key:
            cql_parts.append(f'space = "{space_key}"')
            
        if content_type:
            cql_parts.append(f'type = "{content_type}"')
            
        return " AND ".join(cql_parts)
        
    async def search_pages(self, session_id: str, query: str, space_key: str = None, 
                          limit: int = None, content_type: str = "page") -> Dict[str, Any]:
        """Search Confluence pages using CQL"""
        try:
            cql = self._build_cql_query(query, space_key, content_type)
            results = self.client.cql(cql, limit=limit or self.config.max_results, expand='body.storage,space,version,metadata.labels')
            pages = results.get('results', [])
            return {
                "status": "success",
                "data": {
                    "pages": pages,
                    "total": len(pages)
                }
            }
        except Exception as e:
            logger.error(f"Search pages error: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
            
    async def get_page_by_id(self, session_id: str, page_id: str) -> Dict[str, Any]:
        """Get specific page by ID"""
        try:
            page = self.client.get_page_by_id(page_id, expand='body.storage,space,version,metadata.labels')
            if page:
                return {
                    "status": "success",
                    "data": {"page": page}
                }
            else:
                return {
                    "status": "error",
                    "error": f"Page {page_id} not found"
                }
        except Exception as e:
            logger.error(f"Get page by ID error: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
            
    async def get_page_by_title(self, session_id: str, title: str, space_key: str) -> Dict[str, Any]:
        """Get page by title and space"""
        try:
            page = self.client.get_page_by_title(space_key, title)
            if page:
                return {
                    "status": "success",
                    "data": {"page": page}
                }
            else:
                return {
                    "status": "error",
                    "error": f"Page '{title}' not found in space '{space_key}'"
                }
        except Exception as e:
            logger.error(f"Get page by title error: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
            
    async def list_spaces(self, session_id: str, limit: int = None) -> Dict[str, Any]:
        """List available Confluence spaces"""
        try:
            metadata = self.build_mcp_metadata(session_id)
            
            # API parameters
            params = {
                'limit': limit or self.config.max_results,
                'expand': 'description'
            }
            
            # Make API request
            url = urljoin(self.config.base_url, '/rest/api/space')
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    spaces = []
                    for space in data.get('results', []):
                        space_data = {
                            'key': space['key'],
                            'name': space['name'],
                            'description': space.get('description', {}).get('plain', {}).get('value', ''),
                            'url': urljoin(self.config.base_url, space['_links']['webui'])
                        }
                        spaces.append(space_data)
                    
                    return {
                        "metadata": metadata,
                        "tool": "list_spaces",
                        "status": "success",
                        "data": {
                            "spaces": spaces,
                            "total": len(spaces)
                        }
                    }
                else:
                    error_text = await response.text()
                    return {
                        "metadata": metadata,
                        "tool": "list_spaces",
                        "status": "error",
                        "error": f"HTTP {response.status}: {error_text}"
                    }
                    
        except Exception as e:
            logger.error(f"List spaces error: {str(e)}")
            return {
                "metadata": self.build_mcp_metadata(session_id),
                "tool": "list_spaces",
                "status": "error",
                "error": str(e)
            }
            
    async def get_document_info_from_confluence(self, session_id: str, confluence_url: str = None,
                                               space_key: str = None, page_title: str = None,
                                               page_id: str = None, search_query: str = None) -> Dict[str, Any]:
        """Main method to get document info from Confluence with different approaches"""
        try:
            # If URL provided, extract page ID
            if confluence_url:
                page_id = self._extract_page_id_from_url(confluence_url)
                if not page_id:
                    return {
                        "metadata": self.build_mcp_metadata(session_id),
                        "tool": "get_document_info_from_confluence",
                        "status": "error",
                        "error": "Could not extract page ID from URL"
                    }
            
            # Get page by ID
            if page_id:
                return await self.get_page_by_id(session_id, page_id)
            
            # Get page by title and space
            if page_title and space_key:
                return await self.get_page_by_title(session_id, page_title, space_key)
            
            # Search for pages
            if search_query:
                return await self.search_pages(session_id, search_query, space_key)
            
            return {
                "metadata": self.build_mcp_metadata(session_id),
                "tool": "get_document_info_from_confluence",
                "status": "error",
                "error": "No valid parameters provided"
            }
            
        except Exception as e:
            logger.error(f"Get document info error: {str(e)}")
            return {
                "metadata": self.build_mcp_metadata(session_id),
                "tool": "get_document_info_from_confluence",
                "status": "error",
                "error": str(e)
            }
            
    def get_mcp_resources(self, session_id: str) -> Dict[str, Any]:
        """Get MCP resources for session"""
        context = self.get_or_create_context(session_id)
        resources = []
        
        for page_id, page_data in context.cached_resources.items():
            resources.append({
                "uri": f"confluence://page/{page_id}",
                "name": page_data.get('title', f"Page {page_id}"),
                "description": f"Confluence page in {page_data.get('space_name', 'Unknown space')}",
                "mimeType": "text/html"
            })
            
        return {
            "resources": resources,
            "total": len(resources)
        }
        
    def get_mcp_prompts(self, session_id: str) -> Dict[str, Any]:
        """Get MCP prompts for session"""
        prompts = [
            {
                "name": "summarize_confluence_page",
                "description": "Generate a summary of a Confluence page",
                "arguments": [
                    {"name": "page_id", "description": "Confluence page ID", "required": True},
                    {"name": "focus_areas", "description": "Areas to focus on", "required": False}
                ]
            },
            {
                "name": "extract_requirements_from_confluence",
                "description": "Extract requirements from Confluence documentation",
                "arguments": [
                    {"name": "space_key", "description": "Space containing requirements", "required": True},
                    {"name": "search_query", "description": "Query to find requirements", "required": True}
                ]
            },
            {
                "name": "generate_test_cases_from_confluence",
                "description": "Generate test cases from Confluence specifications",
                "arguments": [
                    {"name": "page_id", "description": "Page ID with specifications", "required": True},
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
                "cached_resources_count": len(context.cached_resources),
                "recent_searches": context.recent_searches[-10:],  # Last 10 searches
                "user_preferences": context.user_preferences,
                "last_accessed": context.last_accessed.isoformat()
            }
        }

    async def update_page_content(self, session_id: str, page_id: str, new_content: str) -> dict:
        try:
            page = self.client.get_page_by_id(page_id, expand='body.storage,version')
            if not page:
                return {"status": "error", "error": "Page not found"}
            title = page['title']
            space = page['space']['key']
            version = int(page['version']['number']) + 1
            updated = self.client.update_page(
                page_id=page_id,
                title=title,
                body=new_content,
                parent_id=None,
                type='page',
                representation='storage',
                minor_edit=False,
                version=version
            )
            return {"status": "success", "data": updated}
        except Exception as e:
            logger.error(f"Error updating Confluence page {page_id}: {str(e)}")
            return {"status": "error", "error": str(e)}


class ConfluenceMCPConfigBuilder:
    """Helper class to build Confluence MCP configuration"""
    
    @staticmethod
    def from_env() -> MCPConfluenceConfig:
        """Create config from environment variables"""
        space_keys = None
        if os.getenv("CONFLUENCE_SPACE_KEYS"):
            space_keys = [s.strip() for s in os.getenv("CONFLUENCE_SPACE_KEYS").split(",")]
            
        return MCPConfluenceConfig(
            base_url=os.getenv("CONFLUENCE_BASE_URL", ""),
            username=os.getenv("CONFLUENCE_USERNAME", ""),
            api_token=os.getenv("CONFLUENCE_API_TOKEN", ""),
            space_keys=space_keys,
            max_results=int(os.getenv("CONFLUENCE_MAX_RESULTS", "50")),
            expand_content=os.getenv("CONFLUENCE_EXPAND_CONTENT", "true").lower() == "true",
            cache_duration=int(os.getenv("CONFLUENCE_CACHE_DURATION", "3600"))
        )
        
    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> MCPConfluenceConfig:
        """Create config from dictionary"""
        return MCPConfluenceConfig(**config_dict)


def print_environment_status():
    """Print current environment setup status"""
    required_vars = ["CONFLUENCE_BASE_URL", "CONFLUENCE_USERNAME", "CONFLUENCE_API_TOKEN"]
    
    print("🔧 Confluence MCP Configuration Status")
    print("=" * 50)
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: {'*' * (len(value) - 4) + value[-4:]}")
        else:
            print(f"❌ {var}: Not set")
    
    optional_vars = ["CONFLUENCE_SPACE_KEYS", "CONFLUENCE_MAX_RESULTS", "CONFLUENCE_EXPAND_CONTENT"]
    print("\nOptional Configuration:")
    for var in optional_vars:
        value = os.getenv(var)
        print(f"   {var}: {value or 'Default'}") 