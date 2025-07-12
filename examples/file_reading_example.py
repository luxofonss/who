#!/usr/bin/env python3
"""
Example demonstrating the use of the read_file utility function
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.file import read_file
from loguru import logger

def main():
    """Demonstrate the read_file utility function"""
    
    # Example 1: Read a file that exists
    logger.info("Example 1: Reading an existing file")
    content = read_file("env.example")
    if content:
        logger.info(f"Successfully read env.example ({len(content)} characters)")
        logger.info(f"First 100 characters: {content[:100]}...")
    else:
        logger.warning("Could not read env.example")
    
    # Example 2: Read a file that doesn't exist (returns default)
    logger.info("\nExample 2: Reading a non-existent file")
    content = read_file("non_existent_file.txt", default="File not found")
    logger.info(f"Result: {content}")
    
    # Example 3: Read with custom encoding
    logger.info("\nExample 3: Reading with custom encoding")
    content = read_file("env.example", encoding="utf-8")
    logger.info(f"Read with UTF-8 encoding: {len(content)} characters")
    
    # Example 4: Read markdown files (like in analyzer_chain.py)
    logger.info("\nExample 4: Reading markdown files")
    files_to_read = [
        "api_docs_example.md",
        "software_testing_guide.md", 
        "response_ac_guide.md"
    ]
    
    for filename in files_to_read:
        content = read_file(filename)
        if content:
            logger.info(f"✅ {filename}: {len(content)} characters")
        else:
            logger.warning(f"❌ {filename}: File not found or empty")
    
    # Example 5: Reading with different default values
    logger.info("\nExample 5: Reading with custom default")
    content = read_file("missing_file.txt", default="Custom default message")
    logger.info(f"Custom default result: {content}")

if __name__ == "__main__":
    main() 