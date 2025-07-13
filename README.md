# AI TCBS - Code Analysis System

A FastAPI-based application for analyzing code and generating test cases using AI.

## Setup

### 1. Environment Configuration

Copy the example environment file and configure your settings:

```bash
cp env.example .env
```

Edit the `.env` file with your actual values:

```env
# Required: Google Gemini API Key
GOOGLE_API_KEY=your_actual_gemini_api_key_here

# Database (defaults work for local development)
DATABASE_URL=postgresql://postgres:admin@db:5432/ai_tcbs
POSTGRES_DB=ai_tcbs
POSTGRES_USER=postgres
POSTGRES_PASSWORD=admin

# Optional: Confluence Integration
CONFLUENCE_BASE_URL=https://your-company.atlassian.net/wiki
CONFLUENCE_USERNAME=your_confluence_username
CONFLUENCE_API_TOKEN=your_confluence_api_token

# Optional: Jira Integration
JIRA_BASE_URL=https://your-company.atlassian.net
JIRA_USERNAME=your_jira_username
JIRA_API_TOKEN=your_jira_api_token

# Optional: Bitbucket Integration
BITBUCKET_USERNAME=your_bitbucket_username
BITBUCKET_WORKSPACE=your_workspace
BITBUCKET_APP_PASSWORD=your_bitbucket_app_password
```

### 2. Running with Docker Compose

```bash
# Build and start the application
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the application
docker-compose down
```

### 3. API Endpoints

- **Health Check**: `GET /health`
- **Create Project**: `POST /create-project`
- **Analyze**: `POST /analyze`
- **Get Projects**: `GET /projects`

## Environment Variables

### Required
- `GOOGLE_API_KEY`: Your Google Gemini API key

### Optional (for integrations)
- **Confluence**: `CONFLUENCE_*` variables for Confluence integration
- **Jira**: `JIRA_*` variables for Jira integration  
- **Bitbucket**: `BITBUCKET_*` variables for Bitbucket integration

### Database
- `DATABASE_URL`: PostgreSQL connection string
- `POSTGRES_DB`: Database name
- `POSTGRES_USER`: Database user
- `POSTGRES_PASSWORD`: Database password 