# Bitbucket Authentication Fix

## Problem

The Bitbucket MCP service was using **Bearer token authentication** which is incorrect for Bitbucket API tokens. According to Atlassian documentation, Bitbucket API tokens must use **Basic Authentication** with the user's email and token.

### Error Details

❌ **Incorrect Authentication:**
```python
headers['Authorization'] = f'Bearer {api_token}'
```

✅ **Correct Authentication:**
```python
auth = aiohttp.BasicAuth(email, api_token)
```

## Solution

### 1. Updated Configuration Model

**Before:**
```python
class MCPBitbucketConfig(BaseModel):
    username: str = Field(..., description="Username for authentication")
    access_token: Optional[str] = Field(None, description="Repository access token")
```

**After:**
```python
class MCPBitbucketConfig(BaseModel):
    email: str = Field(..., description="Atlassian account email for authentication") 
    api_token: Optional[str] = Field(None, description="API token for authentication")
```

### 2. Fixed Authentication Method

**Before (Incorrect):**
```python
if self.config.access_token:
    headers['Authorization'] = f'Bearer {self.config.access_token}'
    
self.session = aiohttp.ClientSession(auth=auth, headers=headers)
```

**After (Correct):**
```python
if self.config.api_token:
    auth = aiohttp.BasicAuth(self.config.email, self.config.api_token)
elif self.config.app_password:
    auth = aiohttp.BasicAuth(self.config.email, self.config.app_password)
    
self.session = aiohttp.ClientSession(auth=auth)
```

### 3. Updated Environment Variables

**Before:**
```bash
BITBUCKET_USERNAME=your_username
BITBUCKET_ACCESS_TOKEN=your_token
```

**After:**
```bash
BITBUCKET_EMAIL=your_email@domain.com
BITBUCKET_API_TOKEN=your_api_token
```

## Atlassian Documentation Reference

According to the official Atlassian Bitbucket documentation:

### Using API tokens with Bitbucket APIs

> The API token, along with the user's Atlassian account email, can be sent as login credentials.

**Example with curl:**
```bash
curl --request POST \
 --url 'https://api.bitbucket.org/2.0/repositories/{workspace}/{repository}/commits' \
 --user '{atlassian_account_email}:{api_token}' \
 --header 'Accept: application/json'
```

**Alternative with Basic Auth header:**
```bash
my_credentials_after_base64_encoding=`echo -n '{atlassian_account_email}:{api_token}' | base64`
curl --request POST \
 --url 'https://api.bitbucket.org/2.0/repositories/{workspace}/{repository}/commits' \
 --header "Authorization: Basic $my_credentials_after_base64_encoding" \
 --header 'Accept: application/json'
```

## Environment Setup

### Required Variables

```bash
BITBUCKET_EMAIL=your_email@domain.com      # Your Atlassian account email
BITBUCKET_WORKSPACE=your_workspace          # Bitbucket workspace name
```

### Authentication (Choose One)

**Option 1: API Token (Recommended for automation)**
```bash
BITBUCKET_API_TOKEN=your_api_token
```

**Option 2: App Password (For personal use)**
```bash
BITBUCKET_APP_PASSWORD=your_app_password
```

### Optional Variables

```bash
BITBUCKET_BASE_URL=https://api.bitbucket.org/2.0  # Default API URL
BITBUCKET_REPOSITORIES=repo1,repo2                # Limit to specific repos
BITBUCKET_MAX_RESULTS=50                          # Max results per request
BITBUCKET_CACHE_DURATION=3600                     # Cache duration in seconds
```

## Authentication Setup Instructions

### Option 1: API Token (Recommended)

1. Go to [Bitbucket API Tokens](https://bitbucket.org/account/settings/api)
2. Create new API token with **Repositories: Read** permission
3. Set environment variables:
   ```bash
   BITBUCKET_EMAIL=your_email@domain.com
   BITBUCKET_API_TOKEN=your_generated_token
   BITBUCKET_WORKSPACE=your_workspace
   ```

### Option 2: App Password

1. Go to [Bitbucket App Passwords](https://bitbucket.org/account/settings/app-passwords/)
2. Create app password with **Repositories: Read** permission  
3. Set environment variables:
   ```bash
   BITBUCKET_EMAIL=your_email@domain.com
   BITBUCKET_APP_PASSWORD=your_generated_password
   BITBUCKET_WORKSPACE=your_workspace
   ```

## Important Notes

- ✅ **Use Atlassian account email**, not Bitbucket username
- ✅ **Basic Authentication** is the correct method for Bitbucket APIs
- ✅ **Bearer tokens are NOT supported** by Bitbucket API
- ✅ Email is required for both API tokens and app passwords
- ✅ The service now follows official Atlassian documentation

## Benefits

1. **Correct authentication** according to Atlassian standards
2. **Fixes 401 authentication errors** from Bitbucket API
3. **Proper credential handling** with email + token/password
4. **Better error messages** and environment validation
5. **Consistent with official documentation** and examples 