{
      "id": "Unique identifier for the acceptance criteria, e.g., AC1, AC2, AC_AI_Suggest1",
      "ac_description": "Description of the acceptance criteria (AC) or requirement, derived from the following sources, with the condition of excluding ACs already defined in Jira documentation:
      - Business requirements: Explicit requirements from {requirements} or Jira documentation, but only including requirements not yet defined as ACs in Jira.
      - AI-suggested ACs: System-generated criteria to enhance feature completeness, stability, and user experience, based on:
         - Industry standards and API design best practices: Applying RESTful API design principles (e.g., HTTP status codes, error handling, input validation, pagination, filtering, sorting, authentication, and authorization).
         - Patterns from similar APIs: Analysis of APIs within the same ecosystem or related projects to identify common use cases, error scenarios, or features like search, logging, or monitoring, excluding those already covered in Jira ACs.
         - Edge cases and usage scenarios: Inferred from {requirements} and {current_context}, ensuring robustness and security (e.g., handling empty data, oversized inputs, invalid values, or injection attacks), including only cases not addressed in Jira.
      - Auto-generated additional ACs: The system will automatically generate new acceptance criteria, non-duplicative with those already defined in Jira, based on analysis of context, requirements, and best practice patterns to ensure feature comprehensiveness.",
      "status": "Status of the acceptance criteria, indicating origin and clarity:
        - 'Đã định nghĩa': AC explicitly defined in {requirements} or Jira.
        - 'Đề xuất': AI-suggested AC to address gaps or enhancements, requiring BA review.
        - 'Cần làm rõ': AC needs stakeholder clarification (e.g., BA/PO) due to ambiguous requirements in {requirements}, test cases in {existing_test_cases} without clear ACs, or code in {code_diff_commit} without defined requirements.",
      "priority": "Action priority: 'High', 'Medium', 'Low', or 'N/A' (if 'Đạt yêu cầu'), e.g., N/A"
    }