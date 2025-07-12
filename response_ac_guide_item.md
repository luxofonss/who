{
      "id": "Unique identifier for the acceptance criteria, e.g., AC1, AC2, AC_AI_Suggest1",
      "ac_description": "Description of the acceptance criteria (AC) or requirement, derived from:
        - Business requirements: Explicit requirements from {requirements} or Jira documentation.
        - AI-suggested ACs: System-generated criteria to enhance feature completeness, stability, and user experience, based on:
          - Industry standards and API design best practices (e.g., RESTful principles, HTTP status codes, error handling, input validation, pagination, filtering, sorting, authentication, authorization).
          - Patterns from similar APIs in the ecosystem or related projects, identifying common use cases, error scenarios, and features like search, logging, or monitoring.
          - Edge cases and usage scenarios inferred from {requirements} and {current_context}, ensuring robustness and security (e.g., empty data, oversized inputs, invalid values, injection attacks).",
      "status": "Status of the acceptance criteria, indicating origin and clarity:
        - 'Đã định nghĩa': AC explicitly defined in {requirements} or Jira.
        - 'Đề xuất': AI-suggested AC to address gaps or enhancements, requiring BA review.
        - 'Cần làm rõ': AC needs stakeholder clarification (e.g., BA/PO) due to ambiguous requirements in {requirements}, test cases in {existing_test_cases} without clear ACs, or code in {code_diff_commit} without defined requirements.",
      "testcase_name": "Name or description of the test case from {existing_test_cases} or {generated_missing_test_cases} linked to this acceptance criteria.",
      "code_location": "Codebase location (e.g., ClassName.java:LineNumber or class/method) in {current_context} or {code_diff_commit} implementing this acceptance criteria. Blank if not fully covered.",
      "assessment": "Evaluation of acceptance criteria fulfillment, guiding actions for BA, Dev, and Tester:
        - 'Đạt yêu cầu': AC fully implemented in {current_context} or {code_diff_commit} with complete coverage in {existing_test_cases}.
        - 'Chưa có code': No implementation in {current_context} or {code_diff_commit} (Dev action needed).
        - 'Chưa có testcase': No test case in {existing_test_cases} or {generated_missing_test_cases} (Tester action needed).
        - 'Testcase chưa đủ': Test cases in {existing_test_cases} only partially cover the AC (Tester must expand).
        - 'Code chưa đủ': Code in {current_context} or {code_diff_commit} partially implements the AC (Dev must complete).
        - 'BA cần xem xét': AC or scenarios need BA clarification due to gaps in {requirements}, {existing_test_cases}, or {code_diff_commit}.",
      "priority": "Action priority: 'High', 'Medium', 'Low', or 'N/A' (if 'Đạt yêu cầu'), e.g., N/A"
    }