# Jira MCP Service Test Scripts

This directory contains test scripts for testing the Jira MCP service functionality, specifically commit and branch extraction from Jira issues.

## Prerequisites

Before running the tests, make sure you have the following environment variables set:

```bash
export JIRA_BASE_URL="https://your-jira-instance.atlassian.net"
export JIRA_EMAIL="your-email@company.com"
export JIRA_API_TOKEN="your-jira-api-token"
```

## Test Scripts

### 1. `get_branch.py` - Comprehensive Test Script

This script provides comprehensive testing of:
- Basic issue retrieval
- Commit extraction from issues
- Branch extraction from issues

**Usage:**
```bash
cd test
python get_branch.py
```

**Features:**
- Tests multiple issue keys
- Detailed output for each test
- Error handling and reporting
- Environment variable validation

### 2. `run_jira_test.py` - Simple Test Script

This script provides a simpler interface for testing specific issues.

**Usage:**
```bash
cd test
python run_jira_test.py ISSUE-KEY1 ISSUE-KEY2
```

**Example:**
```bash
python run_jira_test.py ONESTUDY-123 ONESTUDY-456
```

## Configuration

### Environment Variables

The scripts require the following environment variables:

- `JIRA_BASE_URL`: Your Jira instance URL (e.g., `https://company.atlassian.net`)
- `JIRA_EMAIL`: Your Jira account email
- `JIRA_API_TOKEN`: Your Jira API token (not your password)

### Issue Keys

Replace the placeholder issue keys in the scripts with actual issue keys from your Jira instance:

```python
# In get_branch.py
test_issue_keys = [
    "YOUR-PROJECT-123",  # Replace with actual issue key
    "YOUR-PROJECT-456",  # Replace with actual issue key
    "YOUR-PROJECT-789"   # Replace with actual issue key
]

# In run_jira_test.py
issue_keys = [
    "YOUR-PROJECT-123",  # Replace with actual issue key
    "YOUR-PROJECT-456",  # Replace with actual issue key
]
```

## Expected Output

### Successful Test Output

```
Jira MCP Service Test Script
Started at: 2024-01-15 10:30:00

✅ All required environment variables are set

============================================================
Testing Jira Issue Retrieval
============================================================

--- Testing Issue Retrieval for YOUR-PROJECT-123 ---
✅ Successfully retrieved issue YOUR-PROJECT-123
  Summary: Implement new feature
  Status: In Progress
  Assignee: John Doe
  Reporter: Jane Smith
  Created: 2024-01-10T09:00:00.000+0000
  Updated: 2024-01-15T08:30:00.000+0000

============================================================
Testing Jira Commit Extraction
============================================================

--- Testing Commit Extraction for YOUR-PROJECT-123 ---
✅ Found 3 commits for YOUR-PROJECT-123
  Commit 1:
    Repository: my-project
    Hash: abc123def456
    Display ID: abc123d
    Author: John Doe
    Message: Implement new feature
    Date: 2024-01-15T08:30:00.000+0000
    Files Changed: 5
    URL: https://bitbucket.org/company/my-project/commits/abc123def456

============================================================
Testing Jira Branch Extraction
============================================================

--- Testing Branch Extraction for YOUR-PROJECT-123 ---
✅ Found 2 branches for YOUR-PROJECT-123
  Branch 1: feature/YOUR-PROJECT-123-new-feature
  Branch 2: bugfix/YOUR-PROJECT-123-fix

============================================================
Test completed!
Finished at: 2024-01-15 10:35:00
```

### Error Output

```
❌ Missing required environment variables:
  - JIRA_API_TOKEN

Please set these environment variables before running the test.
```

## Troubleshooting

### Common Issues

1. **Authentication Errors**
   - Verify your Jira API token is correct
   - Ensure your email matches your Jira account
   - Check that your API token has the necessary permissions

2. **Issue Not Found**
   - Verify the issue key exists in your Jira instance
   - Check that you have access to the issue
   - Ensure the issue key format is correct (e.g., `PROJECT-123`)

3. **No Commits/Branches Found**
   - The issue may not have linked commits or branches
   - Development panel integration may not be configured
   - The issue may not be linked to any repositories

4. **Network Errors**
   - Check your internet connection
   - Verify the Jira base URL is correct
   - Check if your Jira instance is accessible

### Debug Mode

To enable debug logging, modify the logger configuration in the scripts:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## API Endpoints Tested

The scripts test the following Jira API endpoints:

1. **Issue Retrieval**: `/rest/api/3/issue/{issueKey}`
2. **Commit Extraction**: `/rest/dev-status/1.0/issue/detail?issueId={id}&dataType=repository`
3. **Branch Extraction**: `/rest/dev-status/1.0/issue/detail?issueId={id}&dataType=pullrequest`

## Notes

- The scripts use the development panel API which requires proper integration between Jira and your development tools (Bitbucket, GitHub, etc.)
- Some Jira instances may have different API endpoints or configurations
- The scripts include error handling and will continue testing other issues even if one fails
- All sensitive information (API tokens, emails) is masked in the output for security 