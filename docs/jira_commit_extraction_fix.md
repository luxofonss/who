# Jira Commit Extraction Fix

## Problem

The Jira MCP service was failing to extract commits from Jira issues because it was using the issue **key** (e.g., "SCRUM-1") in the development panel API calls, but the API expects the numeric **issue ID** (e.g., "10001").

### Error Details

❌ **Incorrect Request Format:**
```http
GET /rest/dev-status/1.0/issue/detail?issueId=SCRUM-1&applicationType=bitbucket&dataType=repository
```

The development panel API expects:
- `issueId=10001` ← **numeric ID**
- `applicationType=bitbucket` or `stash` ← depending on integration
- `dataType=repository` ← for commits

## Solution

### 1. Added Helper Method to Get Numeric Issue ID

```python
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
```

### 2. Updated Commit Extraction Method

```python
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
        api_endpoints = [
            f'/rest/dev-status/1.0/issue/detail?issueId={issue_id}&applicationType=bitbucket&dataType=repository',
            f'/rest/dev-status/1.0/issue/detail?issueId={issue_id}&applicationType=stash&dataType=repository',
            f'/rest/dev-status/1.0/issue/detail?issueId={issue_id}&dataType=repository'
        ]
        
        # ... rest of the extraction logic
```

### 3. Updated Method Calls

Modified calls to pass the numeric ID when already available:

```python
# In search_issues and get_issue_by_key methods:
'commits': await self._extract_commits_from_issue(issue['key'], issue['id']) if self.config.include_commits else []
```

### 4. Fixed Configuration Issue

```python
# Fixed boolean configuration in from_env():
include_commits=os.getenv("JIRA_INCLUDE_COMMITS", "true").lower() == "true",
```

## API Workflow

✅ **Correct Request Flow:**

1. **Get Issue Details:**
   ```http
   GET /rest/api/2/issue/SCRUM-1
   ```
   
   **Response:**
   ```json
   {
     "id": "10001",  ← numeric ID
     "key": "SCRUM-1"
   }
   ```

2. **Get Development Data:**
   ```http
   GET /rest/dev-status/1.0/issue/detail?issueId=10001&applicationType=bitbucket&dataType=repository
   ```

## Testing

The fix should now properly:

1. ✅ Get the numeric issue ID from the issue key
2. ✅ Use the numeric ID in development panel API calls
3. ✅ Try multiple application types (bitbucket, stash)
4. ✅ Extract commit information from the development panel
5. ✅ Handle cases where commits are not available or integrations are not configured

## Environment Variables

Make sure these are set in your `.env` file:

```bash
JIRA_BASE_URL=https://company.atlassian.net
JIRA_USERNAME=email@company.com
JIRA_API_TOKEN=your_api_token
JIRA_INCLUDE_COMMITS=true
```

## Notes

- The method now tries different `applicationType` values (`bitbucket`, `stash`) as different Jira instances may use different integrations
- If no numeric ID can be retrieved, commit extraction is skipped gracefully
- The fix maintains backward compatibility and doesn't break existing functionality
- Enhanced logging shows both issue key and numeric ID for better debugging

## Benefits

1. **Fixes 400/404 errors** from development panel API
2. **Properly extracts commits** linked to Jira issues
3. **Better error handling** and logging
4. **Performance optimization** by passing numeric ID when already available
5. **Graceful degradation** when commits are not available 