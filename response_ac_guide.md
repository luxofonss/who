{
  "analysis_metadata": {
    "jira_task_id": "Jira task ID related to the analysis, e.g., SCRUM-1",
    "git_commit_id": "List of Git commit IDs analyzed, e.g., a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6",
    "analysis_date": "Date and time of analysis (ISO 8601), e.g., 2025-07-12T07:45:00+07:00"
  },
  "detailed_mapping": [ // DO NOT assume, always return all items existed in <final_ac/>.
    {
      "id": "Unique identifier for the acceptance criteria, e.g., AC1, AC2, AC_AI_Suggest_1",
      "ac_description": "data from item in <final_ac/>",
      "status": "Status from <final_ac/>",
      "testcase_name": "Must be exactly field "test_case" of the test case from <final_testcases/> linked to this acceptance criteria. If none related then left empty",
      "code_location": "Codebase location (e.g., ClassName.java:LineNumber or class/method) in <current_context/> or <code_diff_commit/> implementing this acceptance criteria. Blank if not fully covered.",
      "assessment": "After extracting testcase_name and code_location, calculate assessment:
        - 'Chưa có code': No implementation in <current_context/> or <code_diff_commit/> (Dev action needed).
        - 'Chưa có testcase': No test case in {existing_test_cases} or {generated_missing_test_cases} (Tester action needed).
        - 'Testcase chưa đủ': Test cases in {existing_test_cases} only partially cover the AC (Tester must expand).
        - 'Code chưa đủ': Code in <current_context/> or <code_diff_commit/> partially implements the AC (Dev must complete).
        - 'BA cần xem xét': This status is assigned to an Acceptance Criterion (AC) or scenario when its description is empty (or null), but it is linked to {existing_test_cases} or <code_diff_commit/>. This indicates the presence of code or a test case without a clear corresponding business requirement, necessitating clarification from the Business Analyst",
        - 'Đạt yêu cầu': AC fully implemented in <current_context/> or <code_diff_commit/> with complete coverage in above testcase_name".
    }
  ],
  "coverage_overview": {
    "total_acceptance_criteria": "Total number of acceptance criteria identified",
    "fully_covered": "Number of acceptance criteria with 'Đạt yêu cầu' assessment",
    "partially_covered": "Number of acceptance criteria with 'Testcase chưa đủ' or 'Code chưa đủ' assessment",
    "not_covered": "Number of acceptance criteria with 'Chưa có code' or 'Chưa có testcase' assessment",
    "requirement_coverage": "Percentage of acceptance criteria with 'Đã định nghĩa' status. eg. "20%" ",
    "code_coverage": "Percentage of 'Đã định nghĩa' acceptance criteria with 'Đạt yêu cầu' or 'Code chưa đủ' assessment",
    "test_case_coverage": "Percentage of 'Đã định nghĩa' acceptance criteria with 'Đạt yêu cầu' or 'Testcase chưa đủ' assessment",
    "assessment": "Overall quality evaluation: 'Satisfactory' or 'Not Satisfactory'",
    "quality_score": "Quality score (0-100), based on coverage metrics",
    "visual_summary_data": {
      "coverage_distribution": {
        "fully_covered": "Number of fully covered acceptance criteria",
        "partially_covered": "Number of partially covered acceptance criteria",
        "not_covered": "Number of uncovered acceptance criteria"
      },
      "progress_metrics": {
        "requirement": "Percentage of requirement coverage",
        "code": "Percentage of code coverage",
        "test_case": "Percentage of test case coverage"
      }
    }
  }
}