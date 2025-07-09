"""
Simple test suite for Confluence MCP Service

This module provides basic tests to validate the Confluence MCP service
functionality without requiring actual Confluence credentials.
"""

import asyncio
import unittest
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from confluence_mcp_service import (
    ConfluenceMCPService, 
    MCPConfluenceConfig, 
    ConfluencePageResource,
    ConfluenceSpaceResource,
    MCPConfluenceError
)


class TestConfluenceMCPService(unittest.TestCase):
    """Test cases for Confluence MCP Service"""
    
    def setUp(self):
        """Set up test configuration"""
        self.config = MCPConfluenceConfig(
            base_url="https://test.atlassian.net/wiki",
            username="test@example.com",
            api_token="test-token",
            space_keys=["TEST"],
            max_results=10,
            expand_content=True
        )
        
        self.service = ConfluenceMCPService(self.config)
        self.session_id = "test_session_001"
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Valid config
        config = MCPConfluenceConfig(
            base_url="https://test.atlassian.net/wiki",
            username="test@example.com",
            api_token="test-token"
        )
        self.assertIsInstance(config, MCPConfluenceConfig)
        
        # Test default values
        self.assertEqual(config.max_results, 50)
        self.assertEqual(config.content_format, "storage")
        self.assertTrue(config.expand_content)
    
    def test_context_management(self):
        """Test MCP context creation and management"""
        context = self.service.get_or_create_context(self.session_id)
        
        self.assertEqual(context.session_id, self.session_id)
        self.assertIsInstance(context.cached_resources, dict)
        self.assertIsInstance(context.recent_searches, list)
        self.assertIsNotNone(context.last_accessed)
    
    def test_metadata_building(self):
        """Test MCP metadata generation"""
        metadata = self.service.build_mcp_metadata(self.session_id)
        
        self.assertEqual(metadata["protocol_version"], "1.0")
        self.assertEqual(metadata["server_name"], "confluence-mcp")
        self.assertEqual(metadata["session_id"], self.session_id)
        self.assertEqual(metadata["base_url"], self.config.base_url)
        self.assertIn("timestamp", metadata)
    
    def test_url_extraction(self):
        """Test URL parsing for page ID extraction"""
        test_urls = [
            ("https://test.atlassian.net/wiki/pages/viewpage.action?pageId=12345", "12345"),
            ("https://test.atlassian.net/wiki/spaces/TEST/pages/67890", "67890"),
            ("https://test.atlassian.net/wiki/display/TEST/Page?pageId=11111", "11111"),
            ("https://invalid-url.com", None)
        ]
        
        for url, expected_id in test_urls:
            result = self.service._extract_page_id_from_url(url)
            self.assertEqual(result, expected_id)
    
    @patch('aiohttp.ClientSession.get')
    async def test_search_pages_success(self, mock_get):
        """Test successful page search"""
        # Mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "results": [
                {
                    "id": "12345",
                    "title": "Test Page",
                    "space": {"key": "TEST", "name": "Test Space"},
                    "_links": {"webui": "/spaces/TEST/pages/12345"},
                    "body": {"storage": {"value": "<h1>Test Content</h1>"}},
                    "excerpt": "Test excerpt",
                    "version": {"number": 1, "when": "2024-01-01", "by": {"displayName": "Test User"}},
                    "metadata": {"labels": {"results": [{"name": "test"}]}}
                }
            ]
        })
        mock_get.return_value.__aenter__.return_value = mock_response
        
        # Initialize session
        await self.service.initialize_session()
        
        # Test search
        result = await self.service.search_pages(
            session_id=self.session_id,
            query="test query",
            space_key="TEST",
            limit=5
        )
        
        # Validate result
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["tool"], "search_pages")
        self.assertIn("data", result)
        self.assertIn("pages", result["data"])
        self.assertEqual(len(result["data"]["pages"]), 1)
        
        page = result["data"]["pages"][0]
        self.assertEqual(page["id"], "12345")
        self.assertEqual(page["title"], "Test Page")
        self.assertEqual(page["space_key"], "TEST")
        
        # Clean up
        await self.service.close_session()
    
    @patch('aiohttp.ClientSession.get')
    async def test_search_pages_error(self, mock_get):
        """Test search with API error"""
        # Mock error response
        mock_response = AsyncMock()
        mock_response.status = 401
        mock_response.text = AsyncMock(return_value="Unauthorized")
        mock_get.return_value.__aenter__.return_value = mock_response
        
        # Initialize session
        await self.service.initialize_session()
        
        # Test search
        result = await self.service.search_pages(
            session_id=self.session_id,
            query="test query"
        )
        
        # Validate error result
        self.assertEqual(result["status"], "error")
        self.assertIn("error", result)
        
        # Clean up
        await self.service.close_session()
    
    def test_mcp_resources(self):
        """Test MCP resources generation"""
        # Add some mock cached resources
        context = self.service.get_or_create_context(self.session_id)
        context.cached_resources["test_page"] = {
            "id": "12345",
            "title": "Test Page",
            "space_name": "Test Space"
        }
        
        resources = self.service.get_mcp_resources(self.session_id)
        
        self.assertIn("resources", resources)
        self.assertIn("total", resources)
        self.assertEqual(resources["total"], 1)
        
        resource = resources["resources"][0]
        self.assertEqual(resource["uri"], "confluence://page/12345")
        self.assertEqual(resource["name"], "Test Page")
    
    def test_mcp_prompts(self):
        """Test MCP prompts generation"""
        prompts = self.service.get_mcp_prompts(self.session_id)
        
        self.assertIn("prompts", prompts)
        self.assertIn("total", prompts)
        self.assertGreater(prompts["total"], 0)
        
        # Check for expected prompts
        prompt_names = [p["name"] for p in prompts["prompts"]]
        expected_prompts = [
            "summarize_confluence_page",
            "extract_requirements_from_confluence", 
            "generate_test_cases_from_confluence"
        ]
        
        for expected in expected_prompts:
            self.assertIn(expected, prompt_names)
    
    def test_session_info(self):
        """Test session information retrieval"""
        session_info = self.service.get_session_info(self.session_id)
        
        self.assertIn("session_info", session_info)
        info = session_info["session_info"]
        
        self.assertEqual(info["session_id"], self.session_id)
        self.assertIn("cached_resources_count", info)
        self.assertIn("recent_searches", info)
        self.assertIn("user_preferences", info)


class TestMCPDataModels(unittest.TestCase):
    """Test MCP data models"""
    
    def test_confluence_page_resource(self):
        """Test ConfluencePageResource model"""
        page = ConfluencePageResource(
            id="12345",
            title="Test Page",
            space_key="TEST",
            space_name="Test Space",
            url="https://test.atlassian.net/wiki/spaces/TEST/pages/12345",
            content="<h1>Test</h1>",
            labels=["test", "api"]
        )
        
        self.assertEqual(page.id, "12345")
        self.assertEqual(page.type, "page")
        self.assertEqual(len(page.labels), 2)
        
        # Test model dump
        data = page.model_dump()
        self.assertIsInstance(data, dict)
        self.assertEqual(data["id"], "12345")
    
    def test_confluence_space_resource(self):
        """Test ConfluenceSpaceResource model"""
        space = ConfluenceSpaceResource(
            key="TEST",
            name="Test Space",
            url="https://test.atlassian.net/wiki/spaces/TEST",
            description="Test space description"
        )
        
        self.assertEqual(space.key, "TEST")
        self.assertEqual(space.type, "space")
        self.assertEqual(space.description, "Test space description")


async def run_async_tests():
    """Run async test methods"""
    test_instance = TestConfluenceMCPService()
    test_instance.setUp()
    
    print("🧪 Running async tests...")
    
    try:
        # Test search pages success (mocked)
        await test_instance.test_search_pages_success()
        print("✅ test_search_pages_success passed")
        
        # Test search pages error (mocked)
        await test_instance.test_search_pages_error()
        print("✅ test_search_pages_error passed")
        
    except Exception as e:
        print(f"❌ Async test failed: {str(e)}")


def run_sync_tests():
    """Run synchronous test methods"""
    print("🧪 Running sync tests...")
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestConfluenceMCPService)
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestMCPDataModels))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def main():
    """Main test runner"""
    print("🚀 Confluence MCP Service Test Suite")
    print("=" * 50)
    
    # Run sync tests
    sync_success = run_sync_tests()
    
    # Run async tests
    print("\n" + "=" * 50)
    asyncio.run(run_async_tests())
    
    print("\n" + "=" * 50)
    if sync_success:
        print("✅ All tests completed!")
    else:
        print("❌ Some tests failed!")
    
    print("\n💡 To test with real Confluence:")
    print("   1. Set environment variables (see confluence_mcp_config.py)")
    print("   2. Run: python examples/confluence_mcp_example.py")


if __name__ == "__main__":
    main() 