#!/usr/bin/env python3
"""
Debug script for Bitbucket authentication issues
"""

import os
import sys
from pathlib import Path
from urllib.parse import quote

# Add src to path so we can import the service
sys.path.insert(0, str(Path(__file__).parent / "src"))

from services.bitbucket_mcp_service import MCPBitbucketConfig, BitbucketMCPService

def check_environment():
    """Check environment variables and configuration"""
    print("🔧 Bitbucket Environment Check")
    print("=" * 50)
    
    # Check required variables
    email = os.getenv("BITBUCKET_EMAIL")
    workspace = os.getenv("BITBUCKET_WORKSPACE")
    app_password = os.getenv("BITBUCKET_APP_PASSWORD")
    api_token = os.getenv("BITBUCKET_API_TOKEN")
    
    print(f"📧 Email: {email}")
    print(f"🏢 Workspace: {workspace}")
    print(f"🔑 App Password: {'Set' if app_password else 'Not set'}")
    print(f"🔑 API Token: {'Set' if api_token else 'Not set'}")
    
    if not email or not workspace:
        print("❌ Missing required environment variables")
        return None
    
    if not app_password and not api_token:
        print("❌ No authentication method configured")
        return None
    
    print("✅ Environment variables look good")
    return {
        "email": email,
        "workspace": workspace,
        "app_password": app_password,
        "api_token": api_token
    }

def test_url_construction(env_vars):
    """Test URL construction with different encoding scenarios"""
    print("\n🔗 Testing URL Construction")
    print("=" * 50)
    
    email = env_vars["email"]
    password = env_vars["app_password"] or env_vars["api_token"]
    workspace = env_vars["workspace"]
    repository = "onestudy-server"
    
    # Test different encoding scenarios
    scenarios = [
        ("No encoding", email, password),
        ("Email encoded", quote(email, safe=''), password),
        ("Password encoded", email, quote(password, safe='')),
        ("Both encoded", quote(email, safe=''), quote(password, safe='')),
    ]
    
    for name, test_email, test_password in scenarios:
        url = f"https://{test_email}:{test_password}@bitbucket.org/{workspace}/{repository}.git"
        print(f"\n📋 {name}:")
        print(f"   Email: {test_email}")
        print(f"   URL: {url}")
        
        # Check for problematic characters
        if ':' in email and not test_email.startswith('%'):
            print("   ⚠️  Email contains colon - should be encoded")
        if ':' in password and not test_password.startswith('%'):
            print("   ⚠️  Password contains colon - should be encoded")

def test_config_creation(env_vars):
    """Test configuration creation"""
    print("\n⚙️ Testing Configuration Creation")
    print("=" * 50)
    
    try:
        config = MCPBitbucketConfig(
            base_url="https://api.bitbucket.org/2.0",
            email=env_vars["email"],
            app_password=env_vars["app_password"],
            api_token=env_vars["api_token"],
            workspace=env_vars["workspace"]
        )
        print("✅ Configuration created successfully")
        return config
    except Exception as e:
        print(f"❌ Configuration creation failed: {e}")
        return None

async def test_api_connection(config):
    """Test API connection"""
    print("\n🌐 Testing API Connection")
    print("=" * 50)
    
    try:
        async with BitbucketMCPService(config) as service:
            # Test repository info
            result = await service.get_repository_info("test-session", "onestudy-server")
            
            if result["status"] == "success":
                print("✅ API connection successful")
                repo_info = result["data"]["repository"]
                print(f"📦 Repository: {repo_info['name']}")
                print(f"📝 Description: {repo_info['description']}")
                print(f"🔒 Private: {repo_info['is_private']}")
                return True
            else:
                print(f"❌ API connection failed: {result.get('error', 'Unknown error')}")
                return False
    except Exception as e:
        print(f"❌ API connection error: {e}")
        return False

async def test_clone_operation(config):
    """Test clone operation"""
    print("\n📥 Testing Clone Operation")
    print("=" * 50)
    
    try:
        async with BitbucketMCPService(config) as service:
            # Test clone with a test path
            test_path = Path("storage/repos/test-clone")
            if test_path.exists():
                import shutil
                shutil.rmtree(test_path)
            
            result = await service.clone_repository(
                "test-session", 
                "onestudy-server", 
                "main", 
                test_path
            )
            
            if result["status"] == "success":
                print("✅ Clone operation successful")
                print(f"🔍 Commit hash: {result['data']['commit_hash'][:7]}")
                return True
            else:
                print(f"❌ Clone operation failed: {result.get('error', 'Unknown error')}")
                return False
    except Exception as e:
        print(f"❌ Clone operation error: {e}")
        return False

async def main():
    """Main debug function"""
    print("🐛 Bitbucket Authentication Debug Tool")
    print("=" * 60)
    
    # Check environment
    env_vars = check_environment()
    if not env_vars:
        print("\n❌ Environment check failed. Please fix the issues above.")
        return
    
    # Test URL construction
    test_url_construction(env_vars)
    
    # Test configuration
    config = test_config_creation(env_vars)
    if not config:
        print("\n❌ Configuration test failed.")
        return
    
    # Test API connection
    api_success = await test_api_connection(config)
    
    # Test clone operation
    clone_success = await test_clone_operation(config)
    
    # Summary
    print("\n📊 Summary")
    print("=" * 50)
    print(f"🔧 Environment: ✅")
    print(f"⚙️  Configuration: ✅")
    print(f"🌐 API Connection: {'✅' if api_success else '❌'}")
    print(f"📥 Clone Operation: {'✅' if clone_success else '❌'}")
    
    if not api_success:
        print("\n💡 API Connection Issues:")
        print("- Check your credentials are correct")
        print("- Verify the repository exists and you have access")
        print("- Ensure your app password/token has 'Repositories: Read' permission")
    
    if not clone_success:
        print("\n💡 Clone Operation Issues:")
        print("- Check the repository URL is correct")
        print("- Verify the branch exists")
        print("- Ensure you have clone permissions")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 