import json
from typing import List, Dict, Any, Optional

def safe_string_format(value: Any, default: str = "") -> str:
    """Safely convert value to string and escape markdown table characters"""
    if value is None:
        return default
    
    try:
        # Convert to string and handle newlines and pipe characters
        str_value = str(value).replace("\n", " ").replace("|", "\\|")
        return str_value.strip()
    except Exception:
        return default

def convert_json_testcase_to_markdown(test_cases: List[Dict]) -> str:
    """Convert test cases JSON to markdown table with error handling"""
    try:
        if not test_cases:
            return "No test cases found"
        
        if not isinstance(test_cases, list):
            return str(test_cases)
        
        # Header
        markdown_lines = [
            "| Test Case | Type | Category | Priority | Rationale |",
            "|-----------|------|----------|----------|-----------|"
        ]

        # Rows
        for i, case in enumerate(test_cases):
            if not isinstance(case, dict):
                print(f"Warning: Test case {i} is not a dictionary, skipping...")
                continue
                
            test_case = safe_string_format(case.get("test_case"))
            test_type = safe_string_format(case.get("test_type"))
            category = safe_string_format(case.get("category"))
            priority = safe_string_format(case.get("priority"))
            rationale = safe_string_format(case.get("rationale") or case.get("explain_coverage"))

            markdown_lines.append(f"| {test_case} | {test_type} | {category} | {priority} | {rationale} |")

        return "\n".join(markdown_lines)
    
    except Exception as e:
        return f"Error converting test cases to markdown: {str(e)}"

def convert_api_doc_json_to_markdown(doc: Dict) -> str:
    """Convert API documentation JSON to markdown with error handling"""
    try:
        if not isinstance(doc, dict):
            return str(doc)
        
        def bool_to_checkmark(val):
            if isinstance(val, bool):
                return "✅" if val else "❌"
            return "❌"  # Default to false for non-boolean values

        lines = []

        # Title and description
        title = safe_string_format(doc.get('title', 'API Documentation'))
        description = safe_string_format(doc.get('description', ''))
        
        lines.append(f"# {title}\n")
        lines.append(f"{description}\n")

        sections = doc.get('sections', [])
        if not isinstance(sections, list):
            return str(sections)

        for section_idx, section in enumerate(sections):
            if not isinstance(section, dict):
                print(f"Warning: Section {section_idx} is not a dictionary, skipping...")
                continue
                
            section_title = safe_string_format(section.get('section', 'Section'))
            section_desc = safe_string_format(section.get('description', ''))
            
            lines.append(f"## {section_title}\n")
            lines.append(f"{section_desc}\n")

            # If section has fields, make a markdown table
            if 'fields' in section and isinstance(section['fields'], list):
                lines.append("| Name | Type | Required | Description |")
                lines.append("|------|------|----------|-------------|")

                for field in section['fields']:
                    if not isinstance(field, dict):
                        continue
                        
                    name = safe_string_format(field.get("name"))
                    type_ = safe_string_format(field.get("type"))
                    required = bool_to_checkmark(field.get("required", False))
                    description = safe_string_format(field.get("description"))
                    
                    lines.append(f"| {name} | {type_} | {required} | {description} |")

                lines.append("")  # spacing

            # If section has example format (e.g., example response), render as code block
            if 'format' in section:
                lines.append("### Ví dụ response:")
                format_content = safe_string_format(section["format"])
                lines.append(f"```\n{format_content}\n```")
                lines.append("")

        return "\n".join(lines)
    
    except Exception as e:
        return f"Error converting API documentation to markdown: {str(e)}"

def convert_acceptance_criteria_to_markdown(data: List[Dict]) -> str:
    """Convert acceptance criteria JSON to markdown with error handling"""
    try:
        if not data or not isinstance(data, list):
            return str(data)
        
        lines = []
        
        lines.append("## Acceptance Criteria\n")
        lines.append("| ID | Priority | Status | Description | Test Case | Assessment |")
        lines.append("|----|----------|--------|-------------|-----------|------------|")

        for i, item in enumerate(data):
            if not isinstance(item, dict):
                print(f"Warning: Acceptance criteria item {i} is not a dictionary, skipping...")
                continue
                
            id_ = safe_string_format(item.get("id"))
            priority = safe_string_format(item.get("priority"))
            status = safe_string_format(item.get("status"))
            description = safe_string_format(item.get("ac_description"))
            testcase = safe_string_format(item.get("testcase_name"))
            assessment = safe_string_format(item.get("assessment"))

            lines.append(f"| {id_} | {priority} | {status} | {description} | {testcase} | {assessment} |")
        
        return "\n".join(lines)
    
    except Exception as e:
        return f"Error converting acceptance criteria to markdown: {str(e)}"

def convert_final_result_to_markdown(data: Dict) -> str:
    """Convert final result JSON to markdown with comprehensive error handling"""
    try:
        if not isinstance(data, dict):
            return str(data)
        
        markdown = []

        # Section 1: Test Cases
        markdown.append("## 🧪 Test Cases\n")
        markdown.append("| Test Case | Type | Category | Priority | Code Coverage | Explanation |")
        markdown.append("|-----------|------|----------|----------|----------------|-------------|")
        
        test_cases = data.get("test_cases_coverage", [])
        if isinstance(test_cases, list):
            for tc in test_cases:
                if not isinstance(tc, dict):
                    continue
                    
                test_case = safe_string_format(tc.get('test_case'))
                test_type = safe_string_format(tc.get('test_type'))
                category = safe_string_format(tc.get('category'))
                priority = safe_string_format(tc.get('priority'))
                code_coverage = safe_string_format(tc.get('code_coverage_score'))
                explanation = safe_string_format(tc.get('explain_coverage'))
                
                markdown.append(
                    f"| {test_case} | {test_type} | {category} | {priority} | {code_coverage} | {explanation} |"
                )

        # Section 2: AC Analysis
        markdown.append("\n\n## 📋 Acceptance Criteria Analysis\n")
        markdown.append("| ID | Priority | Status | Description | Test Case | Code Location | Assessment |")
        markdown.append("|----|----------|--------|-------------|-----------|----------------|-------------|")
        
        ac_analysis = data.get("ac_analysis", {})
        if isinstance(ac_analysis, dict):
            detailed_mapping = ac_analysis.get("detailed_mapping", [])
            if isinstance(detailed_mapping, list):
                for ac in detailed_mapping:
                    if not isinstance(ac, dict):
                        continue
                        
                    id_ = safe_string_format(ac.get('id'))
                    priority = safe_string_format(ac.get('priority'))
                    status = safe_string_format(ac.get('status'))
                    description = safe_string_format(ac.get('ac_description'))
                    testcase = safe_string_format(ac.get('testcase_name'))
                    code_location = safe_string_format(ac.get('code_location'))
                    assessment = safe_string_format(ac.get('assessment'))
                    
                    markdown.append(
                        f"| {id_} | {priority} | {status} | {description} | {testcase} | {code_location} | {assessment} |"
                    )

        # Section 3: Coverage Overview
        coverage = ac_analysis.get("coverage_overview", {}) if isinstance(ac_analysis, dict) else {}
        if isinstance(coverage, dict):
            markdown.append("\n\n## 📊 Coverage Overview\n")
            markdown.append(f"- **Total ACs**: {safe_string_format(coverage.get('total_acceptance_criteria', 'N/A'))}")
            markdown.append(f"- **Fully Covered**: {safe_string_format(coverage.get('fully_covered', 'N/A'))}")
            markdown.append(f"- **Partially Covered**: {safe_string_format(coverage.get('partially_covered', 'N/A'))}")
            markdown.append(f"- **Not Covered**: {safe_string_format(coverage.get('not_covered', 'N/A'))}")
            markdown.append(f"- **Requirement Coverage**: {safe_string_format(coverage.get('requirement_coverage', 'N/A'))}%")
            markdown.append(f"- **Code Coverage**: {safe_string_format(coverage.get('code_coverage', 'N/A'))}%")
            markdown.append(f"- **Test Case Coverage**: {safe_string_format(coverage.get('test_case_coverage', 'N/A'))}%")
            markdown.append(f"- **Assessment**: {safe_string_format(coverage.get('assessment', 'N/A'))}")
            markdown.append(f"- **Quality Score**: {safe_string_format(coverage.get('quality_score', 'N/A'))}")

        # Section 4: Test Case CSV Export
        markdown.append("\n\n## 📦 Test Case CSV Export\n")
        markdown.append("|Key   | Name | Method | Expected Code | Param | Param Value | Code Coverage | Explanation |")
        markdown.append("|------|------|--------|----------------|--------|--------------|----------------|--------------|")
        
        testcase_csv = data.get("testcase_csv", [])
        if isinstance(testcase_csv, list):
            for tc in testcase_csv:
                if not isinstance(tc, dict):
                    continue
                key = safe_string_format(tc.get('testCaseKey'))
                name = safe_string_format(tc.get('testCaseName'))
                method = safe_string_format(tc.get('httpMethod'))
                expected_code = safe_string_format(tc.get('expectedHttpCode'))
                param_name = safe_string_format(tc.get('requestParamName'))
                param_value = safe_string_format(tc.get('requestParamValue'))
                request_body = safe_string_format(tc.get('requestBody'))
                expect_body = safe_string_format(tc.get('expectBody'))
                code_coverage = safe_string_format(tc.get('code_coverage_score'))
                explanation = safe_string_format(tc.get('explain_coverage'))

                markdown.append(
                    f"| {key} | {name} | {method} | {expected_code} | {param_name} | {param_value} | {request_body} | {expect_body} | {code_coverage} | {explanation} |"
                )

        # Section 5: Unrelated Changes
        markdown.append("\n\n## 🔍 Unrelated Changes\n")
        markdown.append("| File Path | Change Type | Code Snippet | Reason | Severity | Category |")
        markdown.append("|-----------|------|----------|----------|----------------|-------------|")
        
        unrelated_changes = data.get("unrelated_changes", [])
        if isinstance(unrelated_changes, list):
            for uc in unrelated_changes:
                if not isinstance(uc, dict):
                    continue
                    
                file_path = safe_string_format(uc.get('file_path'))
                change_type = safe_string_format(uc.get('change_type'))
                code_snippet = safe_string_format(uc.get('code_snippet'))
                reason = safe_string_format(uc.get('reason'))
                severity = safe_string_format(uc.get('severity'))
                category = safe_string_format(uc.get('category'))
                
                markdown.append(
                    f"| {file_path} | {change_type} | {code_snippet} | {reason} | {severity} | {category} |"
                )

        return "\n".join(markdown)
    
    except Exception as e:
        return f"Error converting final result to markdown: {str(e)}"