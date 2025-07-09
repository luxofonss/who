#!/usr/bin/env python3
"""
Script to create all database tables
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models.database import create_tables, test_connection
from loguru import logger

def main():
    """Create all database tables"""
    logger.info("Testing database connection...")
    
    if not test_connection():
        logger.error("Database connection failed. Please check your DATABASE_URL environment variable.")
        sys.exit(1)
    
    logger.info("Creating database tables...")
    try:
        create_tables()
        logger.info("✅ All tables created successfully!")
        logger.info("Tables created:")
        logger.info("  - projects")
        logger.info("  - project_threads") 
        logger.info("  - chat_history")
    except Exception as e:
        logger.error(f"❌ Error creating tables: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 