# Analyze API with MCP Integration

The analyze API has been updated to automatically retrieve requirements and test cases from Confluence and Jira using Model Context Protocol (MCP) services instead of requiring file uploads.

## Overview

The new analyze API allows users to:
- Specify an API endpoint and HTTP method
- Provide Confluence page URLs for requirements and documentation
- Provide Jira issue URLs for test cases and acceptance criteria
- Get automated analysis without manual file uploads

## Query Format

```
analyze api @endpoint=/api/v1/endpoint @method=POST
@jira=https://company.atlassian.net/browse/PROJ-123,https://company.atlassian.net/browse/PROJ-124
@confluence=https://company.atlassian.net/wiki/spaces/SPACE/pages/123,https://company.atlassian.net/wiki/spaces/SPACE/pages/456
Additional description of the API functionality
```

### Required Parameters

- `@endpoint`: API endpoint path (e.g., `/api/v1/users`)
- `@method`: HTTP method (GET, POST, PUT, DELETE, etc.)
- `@jira`: Comma-separated list of Jira issue URLs
- `@confluence`: Comma-separated list of Confluence page URLs

### Optional Parameters

- Additional text description after the structured parameters

## Environment Setup

### Confluence Configuration

Set these environment variables:

```bash
# Try these URL formats (use only one):
export CONFLUENCE_BASE_URL="https://luxofons.atlassian.net/wiki"
# OR if above doesn't work:
# export CONFLUENCE_BASE_URL="https://luxofons.atlassian.net"
export CONFLUENCE_USERNAME="your-email@luxofons.com"
export CONFLUENCE_API_TOKEN="your-api-token"
```

Optional configuration:
```bash
export CONFLUENCE_SPACE_KEYS="DOC,API,SPEC"  # Limit to specific spaces
export CONFLUENCE_MAX_RESULTS="50"           # Max results per request
export CONFLUENCE_EXPAND_CONTENT="true"      # Include full content
```

### Jira Configuration

Set these environment variables:

```bash
export JIRA_BASE_URL="https://your-company.atlassian.net"
export JIRA_USERNAME="your-email@company.com"
export JIRA_API_TOKEN="your-api-token"
```

Optional configuration:
```bash
export JIRA_PROJECT_KEYS="PROJ,API,TEST"     # Limit to specific projects
export JIRA_MAX_RESULTS="50"                 # Max results per request
export JIRA_EXPAND_FIELDS="description,comments,attachment"  # Fields to expand (comments included by default)
```

## API Usage

### Endpoint

```
POST /analyze
```

### Parameters

- `project_id` (form): The project ID to analyze
- `query` (form): The structured query string

### Example Request

```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "project_id=demo10" \
  -F "query=analyze api @endpoint=/api/v1/quizzes @method=POST @jira=https://mycompany.atlassian.net/browse/QUIZ-123,https://mycompany.atlassian.net/browse/QUIZ-124 @confluence=https://mycompany.atlassian.net/wiki/spaces/API/pages/123,https://mycompany.atlassian.net/wiki/spaces/API/pages/456 This API creates new quizzes for the learning platform"
```

### Example Response

```json
{
  "document": "Detailed explanation of what the endpoint does...",
  "requirement_coverage": [
    {
      "requirement": "API must validate quiz data",
      "coverage_score": "85",
      "explain": "The implementation includes validation logic..."
    }
  ],
  "test_cases": [
    {
      "test_case": "Should create quiz with valid data",
      "coverage_score": "90",
      "explain": "Test case is covered by the implementation..."
    }
  ],
  "improvements": [
    {
      "type": "security",
      "reason": "Missing input validation",
      "solution": "Add comprehensive input validation"
    }
  ],
  "endpoint": "/api/v1/quizzes",
  "analysis_method": "langgraph",
  "analysis_metadata": {
    "endpoint": "/api/v1/quizzes",
    "method": "POST",
    "user_description": "This API creates new quizzes for the learning platform",
    "confluence_urls": [
      "https://mycompany.atlassian.net/wiki/spaces/API/pages/123",
      "https://mycompany.atlassian.net/wiki/spaces/API/pages/456"
    ],
    "jira_urls": [
      "https://mycompany.atlassian.net/browse/QUIZ-123",
      "https://mycompany.atlassian.net/browse/QUIZ-124"
    ],
    "session_id": "analyze_demo10_1234567890",
    "content_sources": {
      "confluence_pages": 2,
      "jira_issues": 2,
      "requirements_length": 1500,
      "testcases_length": 800
    }
  }
}
```

## Getting Help

### Help Endpoint

```
GET /analyze/help
```

This endpoint returns:
- Query format documentation
- Example queries
- Environment setup instructions

### Example Help Request

```bash
curl -X GET "http://localhost:8000/analyze/help"
```

## Examples

### Basic API Analysis

```
analyze api @endpoint=/api/v1/users @method=GET
@jira=https://company.atlassian.net/browse/USER-123
@confluence=https://company.atlassian.net/wiki/spaces/API/pages/456
User management API for retrieving user information
```

### Complex Analysis with Multiple Sources

```
analyze api @endpoint=/api/v1/orders @method=POST
@jira=https://company.atlassian.net/browse/ORDER-123,https://company.atlassian.net/browse/ORDER-124,https://company.atlassian.net/browse/ORDER-125
@confluence=https://company.atlassian.net/wiki/spaces/API/pages/789,https://company.atlassian.net/wiki/spaces/DOC/pages/101,https://company.atlassian.net/wiki/spaces/SPEC/pages/202
Order creation API with payment processing and inventory management
```

### Authentication API Analysis

```
analyze api @endpoint=/api/v1/auth/login @method=POST
@jira=https://company.atlassian.net/browse/AUTH-456
@confluence=https://company.atlassian.net/wiki/spaces/SECURITY/pages/303
Authentication endpoint for user login with JWT token generation
```

## How It Works

1. **Query Parsing**: The API parses the structured query to extract endpoint, method, URLs, and description
2. **Content Retrieval**: 
   - Confluence MCP service retrieves documentation and requirements from specified pages
   - Jira MCP service retrieves issue details, descriptions, acceptance criteria, and all comments with full conversation history
3. **Analysis**: The analyzer chain processes the retrieved content along with the codebase context
4. **Response**: Returns structured analysis with requirement coverage, test case validation, and improvement suggestions

## Error Handling

### Common Errors

1. **Invalid Query Format**: Returns 400 with format help
2. **Missing Environment Variables**: Returns 500 with configuration error
3. **URL Access Issues**: Returns 400 with specific URL error details
4. **No Content Retrieved**: Returns 400 with helpful message

### Example Error Response

```json
{
  "detail": "Invalid query format:\nEndpoint is required (@endpoint=...)\nAt least one Jira URL (@jira=...) or Confluence URL (@confluence=...) is required\n\nQuery Format:\nanalyze api @endpoint=/api/v1/endpoint @method=POST\n..."
}
```

## Migration from File-based API

### Old Format (Deprecated)
```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "project_id=demo10" \
  -F "endpoint=@endpoint=/api/v1/test @method=POST" \
  -F "requirements=@requirements.txt" \
  -F "testcases=@testcases.txt"
```

### New Format
```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "project_id=demo10" \
  -F "query=analyze api @endpoint=/api/v1/test @method=POST @jira=https://company.atlassian.net/browse/TEST-123 @confluence=https://company.atlassian.net/wiki/spaces/API/pages/456 Test API functionality"
```

## Jira Comment Extraction

The system automatically extracts and formats all comments from Jira issues, providing:

### Comment Details Included
- **Author Information**: Name, email, and account ID
- **Timestamps**: Creation and last update times
- **Content**: Full comment text (supports both plain text and Atlassian Document Format)
- **Edit History**: Shows if comment was updated and by whom
- **Visibility**: Identifies internal/restricted comments
- **Chronological Order**: Comments sorted by creation date

### Comment Format in Analysis
```
Comments (3 total):

--- Comment 1 ---
Author: John Doe (john.doe@company.com)
Created: 2024-01-15T10:30:00.000Z
Content: This API should validate input parameters before processing...

--- Comment 2 ---
Author: Jane Smith (jane.smith@company.com)
Created: 2024-01-16T14:20:00.000Z
Updated: 2024-01-16T15:45:00.000Z by Jane Smith
[INTERNAL COMMENT]
Content: Updated requirements based on security review...
```

### Configuration
Comments are included by default. To customize:
```bash
# Include comments (default)
export JIRA_EXPAND_FIELDS="description,comments,attachment"

# Exclude comments if needed
export JIRA_EXPAND_FIELDS="description,attachment"
```

## Benefits

1. **No Manual File Management**: Automatically retrieves content from source systems
2. **Real-time Data**: Always uses the latest information from Confluence and Jira
3. **Comprehensive Analysis**: Combines requirements, test cases, code context, and full conversation history
4. **Audit Trail**: Tracks which sources were used for analysis
5. **Scalable**: Can handle multiple sources and large documents
6. **Complete Context**: Includes all comments and discussions for thorough understanding

## Troubleshooting

### Authentication Issues
- Verify API tokens are valid and not expired
- Check that the user has read permissions for the specified pages/issues
- Ensure base URLs are correct (include `/wiki` for Confluence)

### No Content Retrieved
- Verify URLs are accessible and public or user has permissions
- Check that page IDs or issue keys are correct
- Test URLs manually in browser first

### Performance Issues
- Limit the number of URLs per request (recommended: max 5 each)
- Use specific page IDs when possible instead of search queries
- Consider breaking large analyses into smaller chunks

## Advanced Configuration

### Custom Space/Project Filtering
```bash
export CONFLUENCE_SPACE_KEYS="API,DOC,SPEC"
export JIRA_PROJECT_KEYS="PROJ,API,TEST"
```

### Content Expansion Control
```bash
export CONFLUENCE_EXPAND_CONTENT="true"
export JIRA_EXPAND_FIELDS="description,comments,attachment"
```

### Caching and Performance
```bash
export CONFLUENCE_CACHE_DURATION="3600"  # 1 hour
export JIRA_CACHE_DURATION="3600"        # 1 hour
export CONFLUENCE_MAX_RESULTS="25"
export JIRA_MAX_RESULTS="25"
``` 