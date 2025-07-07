# Confluence MCP Service

A Model Context Protocol (MCP) service for retrieving document information from Atlassian Confluence. This service implements the MCP standard to provide structured, context-aware access to Confluence pages and spaces.

## Overview

The Confluence MCP Service enables your AI applications to:

- **Retrieve documents** from Confluence pages automatically
- **Search content** using Confluence Query Language (CQL)
- **Manage context** with session-based caching
- **Follow MCP standards** for consistent tool integration
- **Replace file uploads** with direct Confluence integration

## Features

### 🔧 MCP Protocol Compliance
- **Resources**: Structured access to Confluence pages and spaces
- **Tools**: Search, retrieve, list operations
- **Prompts**: Reusable templates for document processing
- **Context Management**: Session-based caching and state management

### 📄 Document Operations
- Get page by ID
- Get page by title and space
- Search pages using CQL
- List pages in a space
- List available spaces

### 🚀 Performance Features
- Async/await support
- HTTP connection pooling
- Intelligent caching
- Session persistence

### 🔒 Security
- Basic authentication with API tokens
- Space-based access control
- Input validation and sanitization

## Installation

### Prerequisites

1. **Python Dependencies** (already included in requirements.txt):
   ```
   aiohttp==3.8.4
   pydantic==2.6.4
   loguru==0.7.2
   ```

2. **Confluence Access**:
   - Confluence Cloud or Server instance
   - Valid user account with read permissions
   - API token for authentication

### Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set environment variables**:
   ```bash
   export CONFLUENCE_BASE_URL="https://your-company.atlassian.net/wiki"
   export CONFLUENCE_USERNAME="your-email@company.com"
   export CONFLUENCE_API_TOKEN="your-api-token"
   ```

3. **Optional configuration**:
   ```bash
   export CONFLUENCE_SPACE_KEYS="DOC,API,SPEC"  # Limit to specific spaces
   export CONFLUENCE_MAX_RESULTS="50"           # Max results per request
   export CONFLUENCE_EXPAND_CONTENT="true"      # Include full content
   export CONFLUENCE_CACHE_DURATION="3600"     # Cache duration in seconds
   ```

## Quick Start

### Basic Usage

```python
import asyncio
from services.confluence_mcp_service import ConfluenceMCPService
from services.confluence_mcp_config import ConfluenceMCPConfigBuilder

async def main():
    # Create configuration from environment variables
    config = ConfluenceMCPConfigBuilder.from_env()
    
    # Initialize service
    async with ConfluenceMCPService(config) as confluence:
        session_id = "my_session_001"
        
        # Search for documents
        result = await confluence.search_pages(
            session_id=session_id,
            query="API documentation",
            space_key="DOC",
            limit=5
        )
        
        if result["status"] == "success":
            for page in result["data"]["pages"]:
                print(f"Found: {page['title']}")
                print(f"Content: {page['content'][:200]}...")

asyncio.run(main())
```

### Integration with Analyze API

Replace file uploads with Confluence document retrieval:

```python
from services.confluence_mcp_service import ConfluenceMCPService
from services.confluence_mcp_config import ConfluenceMCPConfigBuilder

async def get_requirements_from_confluence(endpoint_info: str):
    """
    Get requirements and test cases from Confluence instead of file uploads
    """
    config = ConfluenceMCPConfigBuilder.from_env()
    
    async with ConfluenceMCPService(config) as confluence:
        session_id = f"analyze_{hash(endpoint_info)}"
        
        # Extract endpoint and user description
        endpoint_path, user_desc = extract_endpoint_and_user_text(endpoint_info)
        
        # Search for requirements
        req_result = await confluence.search_pages(
            session_id=session_id,
            query=f"{endpoint_path} requirements {user_desc}",
            space_key="REQ",
            limit=3
        )
        
        # Search for test cases  
        test_result = await confluence.search_pages(
            session_id=session_id,
            query=f"{endpoint_path} test cases {user_desc}",
            space_key="TEST",
            limit=3
        )
        
        # Extract content
        requirements_txt = ""
        testcases_txt = ""
        
        if req_result["status"] == "success" and req_result["data"]["pages"]:
            requirements_txt = req_result["data"]["pages"][0].get("content", "")
        
        if test_result["status"] == "success" and test_result["data"]["pages"]:
            testcases_txt = test_result["data"]["pages"][0].get("content", "")
        
        return requirements_txt, testcases_txt
```

## API Reference

### Core Classes

#### `ConfluenceMCPService`

Main service class implementing MCP protocol for Confluence integration.

```python
class ConfluenceMCPService:
    def __init__(self, config: MCPConfluenceConfig)
    
    async def search_pages(self, session_id: str, query: str, 
                          space_key: str = None, limit: int = None) -> Dict[str, Any]
    
    async def get_page_by_id(self, session_id: str, page_id: str) -> Dict[str, Any]
    
    async def get_page_by_title(self, session_id: str, title: str, 
                               space_key: str) -> Dict[str, Any]
    
    async def list_spaces(self, session_id: str, limit: int = None) -> Dict[str, Any]
    
    async def get_document_info_from_confluence(self, session_id: str, 
                                               confluence_url: str = None,
                                               space_key: str = None,
                                               page_title: str = None,
                                               page_id: str = None,
                                               search_query: str = None) -> Dict[str, Any]
```

#### `MCPConfluenceConfig`

Configuration model for the service.

```python
class MCPConfluenceConfig(BaseModel):
    base_url: str                    # Confluence base URL
    username: str                    # Username for authentication
    api_token: str                   # API token for authentication
    space_keys: Optional[List[str]]  # Limit to specific spaces
    max_results: int = 50            # Max results per request
    content_format: str = "storage"  # Content format
    expand_content: bool = True      # Include full content
    cache_duration: int = 3600       # Cache duration in seconds
```

### MCP Tools

#### `search_pages`
Search Confluence pages using CQL (Confluence Query Language).

**Parameters:**
- `session_id`: MCP session identifier
- `query`: Search query text
- `space_key`: Optional space to limit search
- `limit`: Maximum number of results
- `content_type`: Type of content ("page", "blogpost", etc.)

**Returns:**
```json
{
  "metadata": { "protocol_version": "1.0", "session_id": "...", ... },
  "tool": "search_pages",
  "status": "success",
  "data": {
    "query": "API documentation",
    "cql": "text ~ \"API documentation\" AND space = \"DOC\" AND type = \"page\"",
    "pages": [
      {
        "id": "12345",
        "title": "API Documentation",
        "space_key": "DOC",
        "space_name": "Documentation",
        "url": "https://company.atlassian.net/wiki/spaces/DOC/pages/12345",
        "content": "<h1>API Documentation</h1>...",
        "excerpt": "This page contains API documentation...",
        "labels": ["api", "documentation"],
        "version": 5
      }
    ],
    "total": 1
  }
}
```

#### `get_page_by_id`
Retrieve a specific page by its Confluence ID.

**Parameters:**
- `session_id`: MCP session identifier
- `page_id`: Confluence page ID

#### `get_page_by_title`
Retrieve a page by title and space.

**Parameters:**
- `session_id`: MCP session identifier
- `title`: Page title
- `space_key`: Space key where page is located

#### `list_spaces`
List available Confluence spaces.

**Parameters:**
- `session_id`: MCP session identifier
- `limit`: Maximum number of spaces to return

### MCP Resources

The service exposes resources following MCP URI patterns:

- `confluence://page/{PAGE_ID}` - Individual Confluence pages
- `confluence://space/{SPACE_KEY}` - Confluence spaces

### MCP Prompts

Pre-defined prompt templates for common operations:

#### `summarize_confluence_page`
Generate a summary of a Confluence page.

**Arguments:**
- `page_id`: Confluence page ID
- `focus_areas`: Optional areas to focus on

#### `extract_requirements_from_confluence`
Extract requirements from Confluence documentation.

**Arguments:**
- `space_key`: Space containing requirements
- `search_query`: Query to find requirements

#### `generate_test_cases_from_confluence`
Generate test cases from Confluence specifications.

**Arguments:**
- `page_id`: Page ID with specifications
- `endpoint`: API endpoint to generate tests for

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `CONFLUENCE_BASE_URL` | ✅ | - | Base URL of Confluence instance |
| `CONFLUENCE_USERNAME` | ✅ | - | Username for authentication |
| `CONFLUENCE_API_TOKEN` | ✅ | - | API token for authentication |
| `CONFLUENCE_SPACE_KEYS` | ❌ | None | Comma-separated space keys to limit access |
| `CONFLUENCE_MAX_RESULTS` | ❌ | 50 | Maximum results per request |
| `CONFLUENCE_EXPAND_CONTENT` | ❌ | true | Whether to expand page content |
| `CONFLUENCE_CACHE_DURATION` | ❌ | 3600 | Cache duration in seconds |

### Configuration Helpers

```python
from services.confluence_mcp_config import ConfluenceMCPConfigBuilder, print_environment_status

# Check environment setup
print_environment_status()

# Create config from environment
config = ConfluenceMCPConfigBuilder.from_env()

# Create config from dictionary
config = ConfluenceMCPConfigBuilder.from_dict({
    "base_url": "https://company.atlassian.net/wiki",
    "username": "user@company.com",
    "api_token": "token123",
    "space_keys": ["DOC", "API"],
    "max_results": 25
})
```

## Error Handling

The service provides comprehensive error handling:

### Common Errors

- **Authentication Failed (401)**: Check username and API token
- **Access Forbidden (403)**: Check permissions for spaces/pages
- **Resource Not Found (404)**: Page or space doesn't exist
- **Network Error**: Connection issues

### Error Response Format

```json
{
  "metadata": { "protocol_version": "1.0", "session_id": "...", ... },
  "tool": "search_pages",
  "status": "error",
  "error": "Authentication failed - check credentials",
  "query": "API documentation"
}
```

## Caching and Performance

### Session-Based Caching
- Each session maintains its own cache
- Automatic cache invalidation based on duration
- Persistent cache storage to files

### Performance Optimizations
- HTTP connection pooling with aiohttp
- Async/await for non-blocking operations
- Intelligent content expansion (only when needed)
- CQL query optimization

### Cache Management

```python
# Save cache to file
await confluence.save_cache_to_file(session_id)

# Load cache from file
await confluence.load_cache_from_file(session_id)

# Get session info
session_info = confluence.get_session_info(session_id)
print(f"Cached resources: {session_info['session_info']['cached_resources_count']}")
```

## Examples

### Example 1: Replace File Upload in Analyze API

```python
# Before: File upload required
@router.post("/analyze")
async def analyze(
    project_id: str = Form(...),
    endpoint: str = Form(...),
    requirements: UploadFile = File(...),  # File upload
    testcases: UploadFile = File(...),     # File upload
):
    requirements_txt = (await requirements.read()).decode("utf-8")
    testcases_txt = (await testcases.read()).decode("utf-8")
    # ... rest of the code

# After: Confluence integration
@router.post("/analyze-confluence")
async def analyze_confluence(
    project_id: str = Form(...),
    endpoint: str = Form(...),
    confluence_requirements_query: str = Form(...),  # Search query
    confluence_testcases_query: str = Form(...),     # Search query
):
    config = ConfluenceMCPConfigBuilder.from_env()
    
    async with ConfluenceMCPService(config) as confluence:
        session_id = f"analyze_{project_id}_{hash(endpoint)}"
        
        # Get requirements from Confluence
        req_result = await confluence.search_pages(
            session_id=session_id,
            query=confluence_requirements_query,
            limit=1
        )
        
        # Get test cases from Confluence
        test_result = await confluence.search_pages(
            session_id=session_id,
            query=confluence_testcases_query,
            limit=1
        )
        
        requirements_txt = ""
        testcases_txt = ""
        
        if req_result["status"] == "success" and req_result["data"]["pages"]:
            requirements_txt = req_result["data"]["pages"][0].get("content", "")
        
        if test_result["status"] == "success" and test_result["data"]["pages"]:
            testcases_txt = test_result["data"]["pages"][0].get("content", "")
        
        # Continue with existing analyzer logic
        analyzer = AnalyzerChain(project_id)
        result = await analyzer.run(
            endpoint=endpoint,
            requirements_txt=requirements_txt,
            testcases_txt=testcases_txt,
            user_text="Retrieved from Confluence via MCP"
        )
        
        return result
```

### Example 2: Automated Document Discovery

```python
async def discover_api_documentation(endpoint_path: str):
    """
    Automatically discover relevant documentation for an API endpoint
    """
    config = ConfluenceMCPConfigBuilder.from_env()
    
    async with ConfluenceMCPService(config) as confluence:
        session_id = f"discovery_{hash(endpoint_path)}"
        
        # Search for multiple types of documentation
        searches = [
            ("requirements", f"{endpoint_path} requirements specification"),
            ("design", f"{endpoint_path} design architecture"),
            ("tests", f"{endpoint_path} test cases scenarios"),
            ("examples", f"{endpoint_path} examples usage")
        ]
        
        documents = {}
        
        for doc_type, query in searches:
            result = await confluence.search_pages(
                session_id=session_id,
                query=query,
                limit=3
            )
            
            if result["status"] == "success" and result["data"]["pages"]:
                documents[doc_type] = result["data"]["pages"]
        
        return documents
```

## Troubleshooting

### Common Issues

1. **Authentication Errors**
   ```bash
   # Check credentials
   python -m services.confluence_mcp_config
   ```

2. **No Results Found**
   - Verify space keys are correct
   - Check search query syntax
   - Ensure user has read permissions

3. **Connection Timeouts**
   - Check network connectivity
   - Verify Confluence URL is accessible
   - Consider increasing timeout values

4. **Cache Issues**
   - Clear cache directory: `rm -rf storage/confluence_cache/`
   - Restart service to reset in-memory cache

### Debug Mode

Enable detailed logging:

```python
import logging
logging.getLogger("aiohttp").setLevel(logging.DEBUG)

# Or use loguru
from loguru import logger
logger.add("confluence_mcp.log", level="DEBUG")
```

## Security Considerations

1. **API Token Management**
   - Store tokens in environment variables
   - Use least-privilege access
   - Rotate tokens regularly

2. **Space Access Control**
   - Limit `space_keys` to required spaces only
   - Validate user permissions
   - Monitor access logs

3. **Content Filtering**
   - Sanitize search queries
   - Validate page IDs
   - Filter sensitive content

## Future Enhancements

- [ ] Support for Confluence Server (non-cloud)
- [ ] Content creation and editing capabilities
- [ ] Advanced CQL query builder
- [ ] Webhook integration for real-time updates
- [ ] Multi-language content support
- [ ] Integration with other Atlassian products (Jira, Bitbucket)

## Contributing

When contributing to the Confluence MCP service:

1. Follow existing code patterns
2. Add comprehensive tests
3. Update documentation
4. Ensure MCP protocol compliance
5. Test with different Confluence versions

## License

This service is part of the AI-TCBS5 project and follows the same license terms. 