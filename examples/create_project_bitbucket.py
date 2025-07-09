#!/usr/bin/env python3
"""
Example: Create Project from Bitbucket Repository

This script demonstrates how to use the updated project API to create
a project from a Bitbucket repository using the Bitbucket MCP service.
"""

import asyncio
import aiohttp
import json
import os
from pathlib import Path

# API endpoint
API_BASE = "http://localhost:8000"

async def create_project_from_bitbucket():
    """Create a project from a Bitbucket repository"""
    
    # Example project configuration
    project_config = {
        "project_id": "onestudy-server-demo",
        "bitbucket_url": "https://bitbucket.org/funji-tcbs/onestudy-server",
        "branch": "main",
        # workspace and repository will be auto-extracted from URL
        # but you can also specify them explicitly:
        # "workspace": "funji-tcbs", 
        # "repository": "onestudy-server"
    }
    
    print("🚀 Creating project from Bitbucket repository...")
    print(f"Project ID: {project_config['project_id']}")
    print(f"Repository: {project_config['bitbucket_url']}")
    print(f"Branch: {project_config['branch']}")
    print()
    
    async with aiohttp.ClientSession() as session:
        try:
            # Create project
            async with session.post(
                f"{API_BASE}/create-project",
                json=project_config,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    print("✅ Project created successfully!")
                    print(f"📊 Indexed files: {result['indexed_files']}")
                    print(f"📁 Extracted files: {result['extracted_files']}")
                    print(f"🔍 Commit hash: {result['commit_hash']}")
                    print(f"🏢 Workspace: {result['workspace']}")
                    print(f"📦 Repository: {result['repository']}")
                    return result
                else:
                    error_text = await response.text()
                    print(f"❌ Error creating project: {response.status}")
                    print(f"Response: {error_text}")
                    return None
                    
        except Exception as e:
            print(f"❌ Exception occurred: {str(e)}")
            return None

async def reindex_project(project_id: str):
    """Reindex an existing project (pulls latest changes from Bitbucket)"""
    
    print(f"\n🔄 Reindexing project: {project_id}")
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                f"{API_BASE}/reindex",
                json={"project_id": project_id},
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    print("✅ Project reindexed successfully!")
                    print(f"📊 Indexed files: {result['indexed_files']}")
                    print(f"📝 Changed files: {len(result['changed_files'])}")
                    print(f"🔍 Commit hash: {result['commit_hash']}")
                    if result['changed_files']:
                        print("📋 Changed files:")
                        for file in result['changed_files'][:5]:  # Show first 5
                            print(f"  - {file}")
                        if len(result['changed_files']) > 5:
                            print(f"  ... and {len(result['changed_files']) - 5} more")
                    return result
                else:
                    error_text = await response.text()
                    print(f"❌ Error reindexing project: {response.status}")
                    print(f"Response: {error_text}")
                    return None
                    
        except Exception as e:
            print(f"❌ Exception occurred: {str(e)}")
            return None

def check_environment():
    """Check if required environment variables are set"""
    required_vars = [
        "BITBUCKET_EMAIL",
        "BITBUCKET_WORKSPACE"
    ]
    
    auth_vars = [
        "BITBUCKET_APP_PASSWORD", 
        "BITBUCKET_API_TOKEN"
    ]
    
    print("🔧 Checking environment configuration...")
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
            print(f"❌ {var}: Not set")
        else:
            print(f"✅ {var}: Set")
    
    # Check authentication
    auth_set = any(os.getenv(var) for var in auth_vars)
    if auth_set:
        for var in auth_vars:
            if os.getenv(var):
                print(f"✅ {var}: Set")
            else:
                print(f"⚪ {var}: Not set")
    else:
        print("❌ Authentication: No BITBUCKET_APP_PASSWORD or BITBUCKET_API_TOKEN set")
        missing_vars.extend(auth_vars)
    
    if missing_vars:
        print(f"\n❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("\n📝 Setup instructions:")
        print("1. Set BITBUCKET_EMAIL to your Atlassian account email")
        print("2. Set BITBUCKET_WORKSPACE to your workspace name")
        print("3. Set either BITBUCKET_APP_PASSWORD or BITBUCKET_API_TOKEN")
        print("\n🔑 To create credentials:")
        print("- App Password: https://bitbucket.org/account/settings/app-passwords/")
        print("- API Token: https://bitbucket.org/account/settings/api")
        return False
    
    print("✅ Environment configuration looks good!")
    return True

async def main():
    """Main example function"""
    print("🎯 Bitbucket Project API Example")
    print("=" * 40)
    
    # Check environment
    if not check_environment():
        return
    
    print()
    
    # Create project
    result = await create_project_from_bitbucket()
    if not result:
        return
    
    # Ask user if they want to reindex
    project_id = result.get('project_id', 'onestudy-server-demo')
    
    print(f"\n📋 Project '{project_id}' has been created successfully!")
    print("\nYou can now:")
    print(f"1. Query the project using the analyze API")
    print(f"2. Reindex the project to pull latest changes")
    
    # Optionally reindex to demonstrate the functionality
    user_input = input("\n🔄 Would you like to reindex the project now? (y/N): ")
    if user_input.lower().startswith('y'):
        await reindex_project(project_id)
    
    print("\n🎉 Example completed!")

if __name__ == "__main__":
    asyncio.run(main()) 