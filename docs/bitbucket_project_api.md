# Bitbucket Project API

The project API has been updated to use Bitbucket repositories instead of GitHub, leveraging the Bitbucket MCP (Model Context Protocol) service for enhanced repository management.

## Features

- **Direct Bitbucket Integration**: Clone and manage repositories directly from Bitbucket
- **Automatic URL Parsing**: Extract workspace and repository from Bitbucket URLs
- **MCP Protocol**: Uses standardized MCP service for robust API communication
- **Real-time Updates**: Pull latest changes during reindexing
- **Enhanced Metadata**: Store repository information including commit hashes

## Environment Setup

### Required Environment Variables

```bash
# Required
BITBUCKET_EMAIL=your-email@domain.com
BITBUCKET_WORKSPACE=your-workspace-name

# Authentication (choose one)
BITBUCKET_APP_PASSWORD=your-app-password    # For personal use
BITBUCKET_API_TOKEN=your-api-token          # For automation

# Optional
BITBUCKET_BASE_URL=https://api.bitbucket.org/2.0
BITBUCKET_REPOSITORIES=repo1,repo2           # Limit to specific repos
```

### Authentication Setup

#### Option 1: App Password (Recommended for personal use)
1. Go to [Bitbucket App Passwords](https://bitbucket.org/account/settings/app-passwords/)
2. Create new app password with "Repositories: Read" permission
3. Set `BITBUCKET_EMAIL` and `BITBUCKET_APP_PASSWORD`

#### Option 2: API Token (For scripting)
1. Go to [Bitbucket API Tokens](https://bitbucket.org/account/settings/api)
2. Create new API token with "Repositories: Read" permission
3. Set `BITBUCKET_EMAIL` and `BITBUCKET_API_TOKEN`

## API Endpoints

### Create Project

**POST** `/create-project`

Create a new project from a Bitbucket repository.

#### Request Body

```json
{
  "project_id": "my-project",
  "bitbucket_url": "https://bitbucket.org/workspace/repository",
  "branch": "main",
  "workspace": "workspace-name",  // Optional if URL contains it
  "repository": "repository-name" // Optional if URL contains it
}
```

#### Response

```json
{
  "status": "created",
  "indexed_files": 156,
  "commit_hash": "abc123...",
  "extracted_files": 342,
  "workspace": "workspace-name",
  "repository": "repository-name"
}
```

### Reindex Project

**POST** `/reindex`

Update an existing project by pulling latest changes from Bitbucket.

#### Request Body

```json
{
  "project_id": "my-project"
}
```

#### Response

```json
{
  "status": "reindexed",
  "changed_files": ["src/main/java/App.java", "pom.xml"],
  "indexed_files": 158,
  "commit_hash": "def456..."
}
```

## URL Format Support

The API supports various Bitbucket URL formats:

```
https://bitbucket.org/workspace/repository
https://bitbucket.org/workspace/repository.git
https://bitbucket.org/workspace/repository/src/main/
```

The `workspace` and `repository` fields are automatically extracted from the URL if not explicitly provided.

## Example Usage

### Python Client

```python
import aiohttp
import asyncio

async def create_project():
    async with aiohttp.ClientSession() as session:
        data = {
            "project_id": "my-java-project",
            "bitbucket_url": "https://bitbucket.org/funji-tcbs/onestudy-server",
            "branch": "main"
        }
        
        async with session.post(
            "http://localhost:8000/create-project",
            json=data
        ) as response:
            result = await response.json()
            print(f"Created project with {result['indexed_files']} files")

asyncio.run(create_project())
```

### cURL

```bash
# Create project
curl -X POST http://localhost:8000/create-project \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "my-project",
    "bitbucket_url": "https://bitbucket.org/workspace/repository",
    "branch": "main"
  }'

# Reindex project
curl -X POST http://localhost:8000/reindex \
  -H "Content-Type: application/json" \
  -d '{"project_id": "my-project"}'
```

## How It Works

1. **Repository Cloning**: Uses Bitbucket API to download repository as ZIP archive
2. **File Extraction**: Extracts source files to local storage (`storage/repos/`)
3. **Code Parsing**: Analyzes Java files for classes, methods, and endpoints
4. **Indexing**: Creates FAISS vector index for semantic search
5. **Metadata Storage**: Saves project metadata including Bitbucket information

## Advantages Over Git Clone

- **No Git Dependencies**: Works without local Git installation
- **Lightweight**: Downloads only the current branch state
- **API-Based**: Leverages Bitbucket's robust API infrastructure
- **MCP Integration**: Standardized protocol for external service integration
- **Caching**: Built-in caching and session management

## File Structure

After project creation:

```
storage/
├── repos/
│   └── my-project/          # Extracted repository files
├── metadata/
│   └── my-project.json      # Project metadata
├── indexes/
│   ├── my-project.faiss     # Vector index
│   └── my-project_metadata.json
└── merkle/
    └── my-project.json      # File change tracking
```

## Error Handling

Common error scenarios:

- **Authentication Failed**: Check `BITBUCKET_EMAIL` and credentials
- **Repository Not Found**: Verify workspace and repository names
- **No Java Files**: Repository must contain Java source files
- **Network Issues**: Bitbucket API connectivity problems

## Integration with Analysis API

Created projects can be analyzed using the existing analyze API:

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "query": "@endpoint GET /api/users @project my-project",
    "project_id": "my-project"
  }'
```

This combines:
- **Local Code**: From the Bitbucket repository
- **Jira Issues**: Related to the repository
- **Confluence Docs**: Associated documentation

## Troubleshooting

### Authentication Issues
```bash
# Test your credentials
python -c "
from services.bitbucket_mcp_service import print_bitbucket_environment_status
print_bitbucket_environment_status()
"
```

### Repository Access
- Ensure repository is accessible with your credentials
- Check workspace and repository names are correct
- Verify the repository contains Java files

### Environment Check
Run the example script to validate your setup:
```bash
python examples/create_project_bitbucket.py
``` 