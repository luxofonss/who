"""
Example usage of the Confluence MCP Service

This example demonstrates how to use the Model Context Protocol service
to retrieve document information from Confluence.
"""

import asyncio
import os
from pathlib import Path
import sys

# Add the src directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

from services.confluence_mcp_service import ConfluenceMCPService, MCPConfluenceConfig


async def main():
    """
    Example usage of Confluence MCP Service
    """
    
    # Configuration - these should be set as environment variables
    config = MCPConfluenceConfig(
        base_url=os.getenv("CONFLUENCE_BASE_URL", "https://your-company.atlassian.net/wiki"),
        username=os.getenv("CONFLUENCE_USERNAME", "your-email@company.com"),
        api_token=os.getenv("CONFLUENCE_API_TOKEN", "your-api-token"),
        space_keys=["DOC", "API", "TECH"],  # Optional: limit to specific spaces
        max_results=20,
        expand_content=True
    )
    
    # Create session ID (in real usage, this would come from your application)
    session_id = "demo_session_001"
    
    # Initialize the MCP service
    async with ConfluenceMCPService(config) as confluence_service:
        
        print("🚀 Confluence MCP Service Demo")
        print("=" * 50)
        
        # Example 1: List available spaces
        print("\n1. 📋 Listing Confluence Spaces:")
        spaces_result = await confluence_service.list_spaces(session_id, limit=5)
        
        if spaces_result["status"] == "success":
            for space in spaces_result["data"]["spaces"]:
                print(f"   • {space['name']} ({space['key']})")
        else:
            print(f"   ❌ Error: {spaces_result['error']}")
        
        # Example 2: Search for pages
        print("\n2. 🔍 Searching for API documentation:")
        search_result = await confluence_service.search_pages(
            session_id=session_id,
            query="API documentation",
            space_key="DOC",  # Optional: limit to specific space
            limit=3
        )
        
        if search_result["status"] == "success":
            for page in search_result["data"]["pages"]:
                print(f"   • {page['title']} (ID: {page['id']})")
                print(f"     Space: {page['space_name']}")
                print(f"     URL: {page['url']}")
                if page.get("excerpt"):
                    print(f"     Excerpt: {page['excerpt'][:100]}...")
                print()
        else:
            print(f"   ❌ Error: {search_result['error']}")
        
        # Example 3: Get a specific page by ID (using first result from search)
        if search_result["status"] == "success" and search_result["data"]["pages"]:
            page_id = search_result["data"]["pages"][0]["id"]
            
            print(f"\n3. 📄 Getting page details for ID: {page_id}")
            page_result = await confluence_service.get_page_by_id(session_id, page_id)
            
            if page_result["status"] == "success":
                page = page_result["data"]["page"]
                print(f"   Title: {page['title']}")
                print(f"   Author: {page.get('author', 'Unknown')}")
                print(f"   Last Modified: {page.get('modified_date', 'Unknown')}")
                print(f"   Labels: {', '.join(page.get('labels', []))}")
                
                if page.get("content"):
                    content_preview = page["content"][:200].replace('\n', ' ')
                    print(f"   Content Preview: {content_preview}...")
            else:
                print(f"   ❌ Error: {page_result['error']}")
        
        # Example 4: Get page by title and space
        print("\n4. 📄 Getting page by title:")
        title_result = await confluence_service.get_page_by_title(
            session_id=session_id,
            title="API Guidelines",  # Example title
            space_key="DOC"
        )
        
        if title_result["status"] == "success":
            page = title_result["data"]["page"]
            print(f"   Found: {page['title']} in {page['space_name']}")
        else:
            print(f"   ❌ Error: {title_result['error']}")
        
        # Example 5: Get all pages in a space
        print("\n5. 📋 Getting pages in DOC space:")
        space_pages_result = await confluence_service.get_pages_by_space(
            session_id=session_id,
            space_key="DOC",
            limit=5
        )
        
        if space_pages_result["status"] == "success":
            for page in space_pages_result["data"]["pages"]:
                print(f"   • {page['title']} (Version: {page.get('version', 'Unknown')})")
        else:
            print(f"   ❌ Error: {space_pages_result['error']}")
        
        # Example 6: Main method - get document info with different approaches
        print("\n6. 🎯 Using main method - get_document_info_from_confluence:")
        
        # Method 1: By search query
        doc_info_result = await confluence_service.get_document_info_from_confluence(
            session_id=session_id,
            search_query="REST API"
        )
        
        if doc_info_result["status"] == "success":
            if "pages" in doc_info_result["data"]:
                print(f"   Found {len(doc_info_result['data']['pages'])} pages matching 'REST API'")
            elif "page" in doc_info_result["data"]:
                print(f"   Found page: {doc_info_result['data']['page']['title']}")
        else:
            print(f"   ❌ Error: {doc_info_result['error']}")
        
        # Example 7: Get MCP resources and prompts
        print("\n7. 🔧 MCP Protocol Information:")
        
        # Get available resources
        resources = confluence_service.get_mcp_resources(session_id)
        print(f"   Available Resources: {resources['total']}")
        
        # Get available prompts
        prompts = confluence_service.get_mcp_prompts(session_id)
        print(f"   Available Prompts: {prompts['total']}")
        for prompt in prompts["prompts"]:
            print(f"     • {prompt['name']}: {prompt['description']}")
        
        # Get session info
        session_info = confluence_service.get_session_info(session_id)
        print(f"   Cached Resources: {session_info['session_info']['cached_resources_count']}")
        print(f"   Recent Searches: {session_info['session_info']['recent_searches']}")
        
        # Example 8: Save cache for persistence
        print("\n8. 💾 Saving session cache:")
        await confluence_service.save_cache_to_file(session_id)
        print("   Cache saved successfully!")
        
        print("\n✅ Demo completed successfully!")


async def integration_example():
    """
    Example showing how to integrate with the existing analyze.py workflow
    """
    
    print("\n" + "=" * 60)
    print("🔗 Integration Example with Analyze Workflow")
    print("=" * 60)
    
    # This is how you would integrate with your existing analyze.py
    config = MCPConfluenceConfig(
        base_url=os.getenv("CONFLUENCE_BASE_URL", "https://your-company.atlassian.net/wiki"),
        username=os.getenv("CONFLUENCE_USERNAME", "your-email@company.com"),
        api_token=os.getenv("CONFLUENCE_API_TOKEN", "your-api-token"),
        space_keys=["REQ", "SPEC", "API"],  # Requirements and specifications spaces
        max_results=10,
        expand_content=True
    )
    
    session_id = "analyze_session_001"
    
    async with ConfluenceMCPService(config) as confluence_service:
        
        # Simulate getting requirements from Confluence instead of file upload
        print("\n1. 📋 Getting requirements from Confluence:")
        
        # Search for requirements document
        requirements_result = await confluence_service.search_pages(
            session_id=session_id,
            query="API requirements authentication",
            space_key="REQ",
            limit=1
        )
        
        requirements_content = ""
        if requirements_result["status"] == "success" and requirements_result["data"]["pages"]:
            page = requirements_result["data"]["pages"][0]
            requirements_content = page.get("content", "")
            print(f"   Found requirements: {page['title']}")
            print(f"   Content length: {len(requirements_content)} characters")
        
        # Search for test cases document
        print("\n2. 🧪 Getting test cases from Confluence:")
        
        testcases_result = await confluence_service.search_pages(
            session_id=session_id,
            query="test cases authentication API",
            space_key="SPEC",
            limit=1
        )
        
        testcases_content = ""
        if testcases_result["status"] == "success" and testcases_result["data"]["pages"]:
            page = testcases_result["data"]["pages"][0]
            testcases_content = page.get("content", "")
            print(f"   Found test cases: {page['title']}")
            print(f"   Content length: {len(testcases_content)} characters")
        
        # Now you would pass these to your analyzer
        print("\n3. 🔄 Ready to pass to AnalyzerChain:")
        print(f"   Requirements: {'✅ Retrieved' if requirements_content else '❌ Not found'}")
        print(f"   Test Cases: {'✅ Retrieved' if testcases_content else '❌ Not found'}")
        
        # Example of what the analyzer call would look like:
        print("\n   Example analyzer call:")
        print("   ```python")
        print("   analyzer = AnalyzerChain(project_id)")
        print("   result = await analyzer.run(")
        print("       endpoint='/api/v1/auth/login',")
        print("       requirements_txt=requirements_content,")
        print("       testcases_txt=testcases_content,")
        print("       user_text='Retrieved from Confluence via MCP'")
        print("   )")
        print("   ```")


if __name__ == "__main__":
    print("🌟 Confluence MCP Service Examples")
    print("=" * 60)
    print()
    print("⚠️  SETUP REQUIRED:")
    print("   Set these environment variables:")
    print("   - CONFLUENCE_BASE_URL (e.g., https://company.atlassian.net/wiki)")
    print("   - CONFLUENCE_USERNAME (your email)")
    print("   - CONFLUENCE_API_TOKEN (your API token)")
    print()
    print("   To get an API token:")
    print("   1. Go to https://id.atlassian.com/manage-profile/security/api-tokens")
    print("   2. Create a new token")
    print("   3. Copy the token value")
    print()
    
    # Check if environment variables are set
    required_vars = ["CONFLUENCE_BASE_URL", "CONFLUENCE_USERNAME", "CONFLUENCE_API_TOKEN"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("   Please set them before running this example.")
        sys.exit(1)
    
    # Run the examples
    asyncio.run(main())
    asyncio.run(integration_example()) 