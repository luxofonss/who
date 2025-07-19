#!/usr/bin/env python3
"""
Simple script to run Jira MCP service tests with specific issue keys
"""

import asyncio
import os
import sys
from datetime import datetime

# Add src to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from services.jira_mcp_service import JiraMCPService, JiraMCPConfigBuilder
from utils.logger import init_logger

logger = init_logger()

async def test_specific_issues():
    """Test specific Jira issues"""
    print("Jira MCP Service Test - Specific Issues")
    print(f"Started at: {datetime.now()}")
    print()
    
    # Replace these with actual issue keys from your Jira instance
    issue_keys = [
        "ONESTUDY-123",  # Replace with actual issue key
        "ONESTUDY-456",  # Replace with actual issue key
        # Add more issue keys as needed
    ]
    
    try:
        config = JiraMCPConfigBuilder.from_env()
        print(f"Jira Base URL: {config.base_url}")
        print(f"Jira Email: {config.email}")
        print()
        
        async with JiraMCPService(config) as jira_service:
            for issue_key in issue_keys:
                print(f"\n{'='*50}")
                print(f"Testing Issue: {issue_key}")
                print(f"{'='*50}")
                
                # Test issue retrieval
                try:
                    result = await jira_service.get_issue_by_key("test_session", issue_key)
                    if result["status"] == "success":
                        issue = result["data"]["issue"]
                        print(f"✅ Issue retrieved successfully")
                        print(f"   Summary: {issue.get('fields', {}).get('summary', 'N/A')}")
                        print(f"   Status: {issue.get('fields', {}).get('status', {}).get('name', 'N/A')}")
                    else:
                        print(f"❌ Failed to retrieve issue: {result.get('error', 'Unknown error')}")
                        continue
                except Exception as e:
                    print(f"❌ Error retrieving issue: {str(e)}")
                    continue
                
                # Test commit extraction
                try:
                    commits = await jira_service._extract_commits_from_issue(issue_key)
                    if commits:
                        print(f"✅ Found {len(commits)} commits")
                        for i, commit in enumerate(commits[:3], 1):  # Show first 3 commits
                            print(f"   Commit {i}: {commit.get('commit_hash', 'N/A')} - {commit.get('message', 'N/A')[:50]}...")
                        if len(commits) > 3:
                            print(f"   ... and {len(commits) - 3} more commits")
                    else:
                        print(f"❌ No commits found")
                except Exception as e:
                    print(f"❌ Error extracting commits: {str(e)}")
                
                # Test branch extraction
                try:
                    branches = await jira_service._extract_branches_from_issue(issue_key)
                    if branches:
                        print(f"✅ Found {len(branches)} branches")
                        for branch in branches:
                            print(f"   Branch: {branch}")
                    else:
                        print(f"❌ No branches found")
                except Exception as e:
                    print(f"❌ Error extracting branches: {str(e)}")
                
                print()
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    # Check if issue keys are provided as command line arguments
    if len(sys.argv) > 1:
        # Use command line arguments as issue keys
        issue_keys = sys.argv[1:]
        print(f"Testing issue keys: {issue_keys}")
        # You can modify the script to use these issue keys
    else:
        print("No issue keys provided. Please edit the script to add your issue keys.")
        print("Usage: python run_jira_test.py ISSUE-KEY1 ISSUE-KEY2 ...")
    
    asyncio.run(test_specific_issues()) 