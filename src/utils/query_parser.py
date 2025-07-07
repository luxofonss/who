"""
Query Parser Utility

Utility functions to parse complex query formats for the analyze API.
Handles extraction of endpoint, method, Jira URLs, and Confluence URLs from user queries.
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class ParsedQuery:
    """Parsed query components"""
    endpoint: str
    method: str
    jira_urls: List[str]
    confluence_urls: List[str]
    user_description: str
    raw_query: str


def parse_analyze_query(query: str) -> ParsedQuery:
    """
    Parse analyze query with format:
    analyze api @endpoint=/api/v1/quizzes @method=POST
    @jira=https://yourcompany.atlassian.net/browse/PROJ-123,https://yourcompany.atlassian.net/browse/PROJ-124
    @confluence=https://yourcompany.atlassian.net/wiki/1,https://yourcompany.atlassian.net/wiki/2
    this is api to create quiz
    
    Returns ParsedQuery object with extracted components
    """
    logger.info(f"Parsing analyze query: {query[:100]}...")
    
    # Extract endpoint
    endpoint_match = re.search(r'@endpoint=([^\s]+)', query)
    endpoint = endpoint_match.group(1) if endpoint_match else ''
    
    # Extract method
    method_match = re.search(r'@method=([^\s]+)', query)
    method = method_match.group(1) if method_match else 'GET'
    
    # Extract Jira URLs
    jira_urls = []
    jira_match = re.search(r'@jira=([^\s@]+)', query)
    if jira_match:
        jira_urls_str = jira_match.group(1)
        # Split by comma and clean up
        jira_urls = [url.strip() for url in jira_urls_str.split(',') if url.strip()]
    
    # Extract Confluence URLs
    confluence_urls = []
    confluence_match = re.search(r'@confluence=([^\s@]+)', query)
    if confluence_match:
        confluence_urls_str = confluence_match.group(1)
        # Split by comma and clean up
        confluence_urls = [url.strip() for url in confluence_urls_str.split(',') if url.strip()]
    
    # Extract user description by removing all the @ tags
    user_description = query
    # Remove all @ tags
    user_description = re.sub(r'@endpoint=[^\s]+', '', user_description)
    user_description = re.sub(r'@method=[^\s]+', '', user_description)
    user_description = re.sub(r'@jira=[^\s@]+', '', user_description)
    user_description = re.sub(r'@confluence=[^\s@]+', '', user_description)
    # Remove "analyze api" from the beginning
    user_description = re.sub(r'^analyze\s+api\s*', '', user_description, flags=re.IGNORECASE)
    user_description = user_description.strip()
    
    parsed = ParsedQuery(
        endpoint=endpoint,
        method=method,
        jira_urls=jira_urls,
        confluence_urls=confluence_urls,
        user_description=user_description,
        raw_query=query
    )
    
    logger.info(f"Parsed query - Endpoint: {endpoint}, Method: {method}, "
                f"Jira URLs: {len(jira_urls)}, Confluence URLs: {len(confluence_urls)}")
    
    return parsed


def extract_confluence_page_info(url: str) -> Dict[str, Optional[str]]:
    """
    Extract page information from Confluence URL
    Returns dict with page_id, space_key, and page_title if available
    """
    page_info = {
        'page_id': None,
        'space_key': None,
        'page_title': None,
        'search_query': None
    }
    
    # Pattern for page ID extraction - updated to handle more formats
    page_id_patterns = [
        r'pageId=(\d+)',
        r'/pages/(\d+)(?:/|$)',  # Handle pages/ID/ or pages/ID at end
        r'/pages/viewpage\.action\?pageId=(\d+)',
        r'/pages/(\d+)/[^/]*$'  # Handle pages/ID/title format
    ]
    
    for pattern in page_id_patterns:
        match = re.search(pattern, url)
        if match:
            page_info['page_id'] = match.group(1)
            break
    
    # Pattern for space key extraction - updated to handle encoded spaces and user spaces
    space_patterns = [
        r'/spaces/([A-Z0-9~][A-Z0-9a-f~_-]+)/',  # Handle user spaces with ~ and encoded characters
        r'/spaces/([A-Z0-9]+)/',  # Regular space keys
        r'/display/([A-Z0-9]+)/',
        r'spaceKey=([A-Z0-9~][A-Z0-9a-f~_-]+)'  # Handle encoded space keys in params
    ]
    
    for pattern in space_patterns:
        match = re.search(pattern, url)
        if match:
            page_info['space_key'] = match.group(1)
            break
    
    # Extract title from URL - improved to handle various formats
    if not page_info['page_title']:
        # Try to extract title from the end of URL
        title_patterns = [
            r'/pages/\d+/([^/?]+)',  # pages/ID/title format
            r'/([^/]+)$'  # Last segment of URL
        ]
        
        for pattern in title_patterns:
            title_match = re.search(pattern, url)
            if title_match:
                title = title_match.group(1)
                # Clean up the title
                title = title.replace('+', ' ').replace('%20', ' ').replace('%2B', '+')
                # URL decode other common characters
                title = title.replace('%21', '!').replace('%40', '@').replace('%23', '#')
                page_info['page_title'] = title
                break
    
    return page_info


def extract_jira_issue_key(url: str) -> Optional[str]:
    """
    Extract issue key from Jira URL
    Returns issue key like PROJ-123
    """
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


def build_confluence_search_query(endpoint: str, method: str, user_description: str) -> str:
    """
    Build a search query for Confluence based on endpoint and description
    """
    query_parts = []
    
    # Add endpoint parts
    if endpoint:
        # Extract meaningful parts from endpoint
        endpoint_parts = endpoint.strip('/').split('/')
        # Remove version indicators
        endpoint_parts = [part for part in endpoint_parts if not re.match(r'^v\d+$', part)]
        query_parts.extend(endpoint_parts)
    
    # Add method
    if method:
        query_parts.append(method)
    
    # Add user description words
    if user_description:
        # Extract meaningful words (remove common words)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', user_description.lower())
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use'}
        meaningful_words = [word for word in words if word not in stop_words]
        query_parts.extend(meaningful_words[:5])  # Limit to 5 words
    
    return ' '.join(query_parts)


def build_jira_search_query(endpoint: str, method: str, user_description: str) -> str:
    """
    Build a search query for Jira based on endpoint and description
    """
    query_parts = []
    
    # Add endpoint parts
    if endpoint:
        # Extract meaningful parts from endpoint
        endpoint_parts = endpoint.strip('/').split('/')
        # Remove version indicators
        endpoint_parts = [part for part in endpoint_parts if not re.match(r'^v\d+$', part)]
        query_parts.extend(endpoint_parts)
    
    # Add method
    if method:
        query_parts.append(method)
    
    # Add user description words
    if user_description:
        # Extract meaningful words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', user_description.lower())
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use'}
        meaningful_words = [word for word in words if word not in stop_words]
        query_parts.extend(meaningful_words[:5])  # Limit to 5 words
    
    return ' '.join(query_parts)


def validate_parsed_query(parsed_query: ParsedQuery) -> Tuple[bool, List[str]]:
    """
    Validate parsed query and return validation status and error messages
    """
    errors = []
    
    if not parsed_query.endpoint:
        errors.append("Endpoint is required (@endpoint=...)")
    
    if not parsed_query.method:
        errors.append("HTTP method is required (@method=...)")
    
    if not parsed_query.jira_urls and not parsed_query.confluence_urls:
        errors.append("At least one Jira URL (@jira=...) or Confluence URL (@confluence=...) is required")
    
    # Validate Jira URLs
    for url in parsed_query.jira_urls:
        if not extract_jira_issue_key(url):
            errors.append(f"Invalid Jira URL format: {url}")
    
    # Validate Confluence URLs
    for url in parsed_query.confluence_urls:
        page_info = extract_confluence_page_info(url)
        if not any(page_info.values()):
            errors.append(f"Invalid Confluence URL format: {url}")
    
    return len(errors) == 0, errors


def format_query_help() -> str:
    """
    Return help text for query format
    """
    return """
Query Format:
analyze api @endpoint=/api/v1/endpoint @method=POST
@jira=https://company.atlassian.net/browse/PROJ-123,https://company.atlassian.net/browse/PROJ-124
@confluence=https://company.atlassian.net/wiki/spaces/SPACE/pages/123,https://company.atlassian.net/wiki/spaces/SPACE/pages/456
Additional description of the API functionality

Required Parameters:
- @endpoint: API endpoint path (e.g., /api/v1/users)
- @method: HTTP method (GET, POST, PUT, DELETE, etc.)
- @jira: Comma-separated list of Jira issue URLs
- @confluence: Comma-separated list of Confluence page URLs

Optional:
- Additional text description after the parameters

Example:
analyze api @endpoint=/api/v1/quizzes @method=POST
@jira=https://mycompany.atlassian.net/browse/QUIZ-123,https://mycompany.atlassian.net/browse/QUIZ-124
@confluence=https://mycompany.atlassian.net/wiki/spaces/API/pages/123,https://mycompany.atlassian.net/wiki/spaces/API/pages/456
This API creates new quizzes for the learning platform
""" 