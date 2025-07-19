#!/usr/bin/env python3
"""
Test script for Jira MCP service commit and branch extraction
"""

import asyncio
import os
import sys
from datetime import datetime

# Add src to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from services.jira_mcp_service import JiraMCPService, JiraMCPConfigBuilder

async def test_commit_extraction():
    """Test commit extraction from Jira issues"""
    print("=" * 60)
    print("Testing Jira Commit Extraction")
    print("=" * 60)
    
    try:
        # Initialize Jira MCP service
        config = JiraMCPConfigBuilder.from_env()
        print(f"Jira Config - Base URL: {config.base_url}")
        print(f"Jira Config - Username: {config.username}")
        print(f"Jira Config - API Token: {'*' * len(config.api_token) if config.api_token else 'None'}")
        
        async with JiraMCPService(config) as jira_service:
            # Test issue keys - replace with actual issue keys from your Jira instance
            test_issue_keys = [
                "SCRUM-1",  # Replace with actual issue key
                "SCRUM-1",  # Replace with actual issue key
                "SCRUM-12"    # Replace with actual issue key
            ]
            
            for issue_key in test_issue_keys:
                print(f"\n--- Testing Commit Extraction for {issue_key} ---")
                
                try:
                    # Extract commits
                    commits = await jira_service._extract_commits_from_issue(issue_key)
                    
                    if commits:
                        print(f"✅ Found {len(commits)} commits for {issue_key}")
                        for i, commit in enumerate(commits, 1):
                            print(f"  Commit {i}:")
                            print(f"    Repository: {commit.get('repository', 'N/A')}")
                            print(f"    Hash: {commit.get('commit_hash', 'N/A')}")
                            print(f"    Display ID: {commit.get('display_id', 'N/A')}")
                            print(f"    Author: {commit.get('author', 'N/A')}")
                            print(f"    Message: {commit.get('message', 'N/A')[:100]}...")
                            print(f"    Date: {commit.get('date', 'N/A')}")
                            print(f"    Files Changed: {commit.get('files_changed', 'N/A')}")
                            print(f"    URL: {commit.get('url', 'N/A')}")
                            print()
                    else:
                        print(f"❌ No commits found for {issue_key}")
                        
                except Exception as e:
                    print(f"❌ Error extracting commits for {issue_key}: {str(e)}")
                    
    except Exception as e:
        print(f"❌ Error initializing Jira service: {str(e)}")

async def test_branch_extraction():
    """Test branch extraction from Jira issues"""
    print("\n" + "=" * 60)
    print("Testing Jira Branch Extraction")
    print("=" * 60)
    
    try:
        # Initialize Jira MCP service
        config = JiraMCPConfigBuilder.from_env()
        
        async with JiraMCPService(config) as jira_service:
            # Test issue keys - replace with actual issue keys from your Jira instance
            test_issue_keys = [
                "SCRUM-1",  # Replace with actual issue key
                "SCRUM-10",  # Replace with actual issue key
                "SCRUM-6"    # Replace with actual issue key
            ]
            
            for issue_key in test_issue_keys:
                print(f"\n--- Testing Branch Extraction for {issue_key} ---")
                
                try:
                    # Extract branches
                    branches = await jira_service._extract_branches_from_issue(issue_key)
                    
                    if branches:
                        print(f"✅ Found {len(branches)} branches for {issue_key}")
                        for i, branch in enumerate(branches, 1):
                            print(f"  Branch {i}: {branch}")
                    else:
                        print(f"❌ No branches found for {issue_key}")
                        
                except Exception as e:
                    print(f"❌ Error extracting branches for {issue_key}: {str(e)}")
                    
    except Exception as e:
        print(f"❌ Error initializing Jira service: {str(e)}")

async def test_issue_retrieval():
    """Test basic issue retrieval"""
    print("\n" + "=" * 60)
    print("Testing Jira Issue Retrieval")
    print("=" * 60)
    
    try:
        # Initialize Jira MCP service
        config = JiraMCPConfigBuilder.from_env()
        
        async with JiraMCPService(config) as jira_service:
            # Test issue key - replace with actual issue key from your Jira instance
            test_issue_key = "TEST-123"  # Replace with actual issue key
            
            print(f"\n--- Testing Issue Retrieval for {test_issue_key} ---")
            
            try:
                # Get issue details
                result = await jira_service.get_issue_by_key("test_session", test_issue_key)
                
                if result["status"] == "success":
                    issue = result["data"]["issue"]
                    print(f"✅ Successfully retrieved issue {test_issue_key}")
                    print(f"  Summary: {issue.get('fields', {}).get('summary', 'N/A')}")
                    print(f"  Status: {issue.get('fields', {}).get('status', {}).get('name', 'N/A')}")
                    print(f"  Assignee: {issue.get('fields', {}).get('assignee', {}).get('displayName', 'N/A')}")
                    print(f"  Reporter: {issue.get('fields', {}).get('reporter', {}).get('displayName', 'N/A')}")
                    print(f"  Created: {issue.get('fields', {}).get('created', 'N/A')}")
                    print(f"  Updated: {issue.get('fields', {}).get('updated', 'N/A')}")
                else:
                    print(f"❌ Failed to retrieve issue {test_issue_key}: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"❌ Error retrieving issue {test_issue_key}: {str(e)}")
                    
    except Exception as e:
        print(f"❌ Error initializing Jira service: {str(e)}")

async def main():
    """Main test function"""
    print("Jira MCP Service Test Script")
    print(f"Started at: {datetime.now()}")
    print()
    
    # Check environment variables
    required_env_vars = [
        "JIRA_BASE_URL",
        "JIRA_USERNAME", 
        "JIRA_API_TOKEN"
    ]
    
    missing_vars = []
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        print("\nPlease set these environment variables before running the test.")
        return
    
    print("✅ All required environment variables are set")
    print()
    
    # Run tests
#     await test_issue_retrieval()
#     await test_commit_extraction()
    await test_branch_extraction()
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print(f"Finished at: {datetime.now()}")

if __name__ == "__main__":
    asyncio.run(main())
