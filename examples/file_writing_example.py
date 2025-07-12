#!/usr/bin/env python3
"""
Example demonstrating the write_analysis_results utility function
"""

import sys
import os
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.file import write_analysis_results
from loguru import logger

def create_sample_state_data():
    """Create sample state data for testing"""
    return {
        "endpoint": "/api/v1/users/login",
        "context": "Sample context for user login endpoint",
        "requirements": "User should be able to login with valid credentials",
        "phase_complete": {
            "phase1_extract_testcases": True,
            "phase1_generate_missing_testcases": True,
            "phase1_improve_testcases": True,
            "phase2_generate_current_ac": True,
            "phase2_generate_missing_ac": True,
            "phase2_improve_and_finalize_ac": True,
            "phase3_generate_additional_coverage": True,
            "format_output": True
        },
        "existing_testcases": [
            {
                "test_case": "Test login with valid credentials",
                "test_type": "positive",
                "coverage_area": "authentication",
                "priority": "high"
            }
        ],
        "generated_missing_testcases": [
            {
                "test_case": "Test login with invalid credentials",
                "test_type": "negative",
                "category": "functional",
                "priority": "high",
                "rationale": "Security testing"
            }
        ],
        "final_testcases": [
            {
                "test_case": "Test login with valid credentials",
                "test_type": "positive",
                "coverage_area": "authentication",
                "priority": "high"
            },
            {
                "test_case": "Test login with invalid credentials",
                "test_type": "negative",
                "category": "functional",
                "priority": "high"
            }
        ],
        "current_ac": [
            {
                "ac": "User can login with valid credentials",
                "type": "positive",
                "priority": "high"
            }
        ],
        "generated_missing_ac": [
            {
                "ac": "System should reject invalid credentials",
                "type": "negative",
                "priority": "high"
            }
        ],
        "final_ac": [
            {
                "ac": "User can login with valid credentials",
                "type": "positive",
                "priority": "high"
            },
            {
                "ac": "System should reject invalid credentials",
                "type": "negative",
                "priority": "high"
            }
        ],
        "additional_coverage": {
            "coverage_summary": "Good coverage of authentication scenarios",
            "coverage_metrics": {
                "test_coverage": 85.5,
                "ac_coverage": 90.0,
                "overall_coverage": 87.75
            }
        },
        "final_analysis_result": {
            "overall_coverage": 87.75,
            "test_coverage": 85.5,
            "ac_coverage": 90.0,
            "improvements": [
                "Add more edge case testing",
                "Include performance testing"
            ]
        },
        "html_response": """
        <html>
        <head><title>Analysis Report</title></head>
        <body>
            <h1>API Analysis Report</h1>
            <p>This is a sample HTML report for the login endpoint.</p>
        </body>
        </html>
        """
    }

def main():
    """Demonstrate the write_analysis_results function"""
    
    # Create sample state data
    sample_state = create_sample_state_data()
    
    logger.info("📝 Example: Writing analysis results to files")
    logger.info("=" * 60)
    
    # Test writing to default directory
    logger.info("\n1. Writing to default directory (storage/analyze):")
    written_files = write_analysis_results(
        state_data=sample_state,
        project_id="test-project",
        endpoint="/api/v1/users/login"
    )
    
    if written_files:
        logger.info("✅ Files written successfully:")
        for file_type, file_path in written_files.items():
            logger.info(f"   📄 {file_type.upper()}: {file_path}")
    else:
        logger.warning("⚠️ No files were written")
    
    # Test writing to custom directory
    logger.info("\n2. Writing to custom directory (examples/output):")
    custom_files = write_analysis_results(
        state_data=sample_state,
        project_id="test-project",
        endpoint="/api/v1/users/register",
        base_dir="examples/output"
    )
    
    if custom_files:
        logger.info("✅ Custom files written successfully:")
        for file_type, file_path in custom_files.items():
            logger.info(f"   📄 {file_type.upper()}: {file_path}")
    else:
        logger.warning("⚠️ No custom files were written")
    
    # Test with different endpoint (special characters)
    logger.info("\n3. Writing with special characters in endpoint:")
    special_files = write_analysis_results(
        state_data=sample_state,
        project_id="test-project",
        endpoint="/api/v1/users/search?query=test&page=1&limit=10"
    )
    
    if special_files:
        logger.info("✅ Special character files written successfully:")
        for file_type, file_path in special_files.items():
            logger.info(f"   📄 {file_type.upper()}: {file_path}")
    else:
        logger.warning("⚠️ No special character files were written")
    
    logger.info("\n" + "=" * 60)
    logger.info("🎉 File writing example completed!")
    logger.info("Check the generated files to see the analysis results.")

if __name__ == "__main__":
    main() 