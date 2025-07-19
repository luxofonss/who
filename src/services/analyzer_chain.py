from __future__ import annotations

import json
import asyncio
import re
from typing import AsyncGenerator, Dict, List, Optional, Any, TypedDict, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from loguru import logger
from langgraph.graph import StateGraph, END

from adapters.model_factory import ModelFactory
from services.neo4j import get_neo4j_connection
from services.retriever import LangChainRetriever
from services.prompt_builder import PromptBuilder
from utils.file import read_file
from utils.markdown_helper import convert_json_testcase_to_markdown, convert_api_doc_json_to_markdown, convert_acceptance_criteria_to_markdown, convert_final_result_to_markdown

class AgentState(TypedDict):
    question: str
    context: str
    endpoint: str
    requirements: str
    user_text: str
    history: List[str]
    retrieved_symbols: List[str]
    final_response: Optional[str]
    html_response: Optional[str]
    iteration_count: int
    seen_context: List[str]
    last_tool_call_symbols: List[str]
    new_retrieved_symbols: List[str]
    node_call_count: Dict[str, int]
    code_commit: str

    api_docs: Dict[str, Any]
    
    # Multi-phase analysis fields
    current_phase: str
    phase_complete: Dict[str, bool]
    needs_more_context: bool
    
    # Phase 1: Test Cases Analysis
    existing_testcases: List[Dict[str, Any]]
    generated_missing_testcases: List[Dict[str, Any]]
    final_testcases: List[Dict[str, Any]]
    
    # Phase 2: Acceptance Criteria Analysis
    current_ac: List[Dict[str, Any]]
    generated_missing_ac: List[Dict[str, Any]]
    final_ac: List[Dict[str, Any]]
    
    # Phase 3: Coverage Analysis
    additional_coverage: Dict[str, Any]
    final_analysis_result: Dict[str, Any]


@dataclass
class AnalysisResult:
    document: str
    requirement_coverage: List[Dict[str, Any]]
    improvements: List[Dict[str, str]]
    endpoint: str
    existed_test_cases: List[Dict[str, Any]] = field(default_factory=list)
    additional_test_cases: List[Dict[str, Any]] = field(default_factory=list)
    response_ac: str = ""
    curl_command: str = ""
    html_response: str = ""
    raw_response: Optional[str] = None
    analysis_method: str = "langgraph"  # "langgraph" or "fallback"


class AnalysisError(Exception):
    pass


class AnalyzerChain:

    def __init__(self, project_id: str, model_name: str, api_key: str):
        if not project_id or not isinstance(project_id, str):
            raise ValueError("project_id must be a non-empty string")
        
        self.project_id = project_id
        self.retriever = LangChainRetriever(project_id)
        
        if model_name and api_key:
            self._setup_custom_model(model_name, api_key)
        else:
            raise ValueError("model_name and api_key must be provided")
        
        # Create tool and graph
        self._setup_langgraph()
        
        logger.info(f" AnalyzerChain initialized with LangGraph for project: {project_id}")

    def _setup_custom_model(self, model_name: str, api_key: str):
        """Setup custom model with provided API key using ModelFactory."""
        from adapters.model_factory import ModelFactory
        self.llm = ModelFactory.create_llm(model_name=model_name, api_key=api_key, temperature=0.1)
        self.langchain_llm = ModelFactory.create_langchain_llm(model_name=model_name, api_key=api_key, temperature=0.1)
        logger.info(f"🔧 Using custom model: {model_name}")

    def _setup_langgraph(self):
        logger.info(" Setting up LangGraph components...")
        
        self._build_graph()
        logger.info(" LangGraph setup complete")

    def _build_graph(self):
        graph = StateGraph(AgentState)

        # Phase 1: Test Cases Analysis (3 nodes) - Use synchronous methods
        graph.add_node("extract_existing_testcases", self._extract_existing_testcases_node)
        graph.add_node("generate_missing_testcases", self._generate_missing_testcases_node)
        graph.add_node("generate_api_docs", self._generate_api_docs_node)
        graph.add_node("improve_and_finalize_testcases", self._improve_and_finalize_testcases_node)
        
        # Phase 2: Acceptance Criteria Analysis (3 nodes)
        graph.add_node("generate_current_ac", self._generate_current_ac_node)
        graph.add_node("generate_missing_ac", self._generate_missing_ac_node)
        graph.add_node("improve_and_finalize_ac", self._improve_and_finalize_ac_node)
        
        # Phase 3: Coverage Analysis (1 node)
        graph.add_node("generate_final_response", self._generate_final_response)
        
        # Final output formatting
        graph.add_node("format_output", self._format_output_node)

        # Phase 1 flow: Test Cases Analysis
        graph.add_edge("extract_existing_testcases", "generate_missing_testcases")
        graph.add_edge("generate_missing_testcases", "generate_api_docs")
        graph.add_edge("generate_api_docs", "improve_and_finalize_testcases")
        
        # Phase 1 to Phase 2 transition
        graph.add_edge("improve_and_finalize_testcases", "generate_current_ac")
        
        # Phase 2 flow: Acceptance Criteria Analysis
        graph.add_edge("generate_current_ac", "generate_missing_ac")
        graph.add_edge("generate_missing_ac", "improve_and_finalize_ac")
        
        # Phase 2 to Phase 3 transition
        graph.add_edge("improve_and_finalize_ac", "generate_final_response")
        
        # Phase 3 to final output
        graph.add_edge("generate_final_response", "format_output")
        graph.add_edge("format_output", END)

        # Set entry point to Phase 1 Node 1
        graph.set_entry_point("extract_existing_testcases")

        # Compile the graph
        self.graph = graph.compile()
        logger.info(" 3-Phase LangGraph workflow compiled with 7 nodes")

    def _read_api_docs_example(self) -> str:
        """Read API documentation example file"""
        return read_file("api_docs.json")

    def _read_software_testing_guide(self) -> str:
        """Read software testing guide file"""
        return read_file("software_testing_guide.md")

    def _read_response_ac_guide(self) -> str:
        """Read response acceptance criteria guide file"""
        return read_file("response_ac_guide.md")

    def _read_response_ac_guide_item(self) -> str:
        """Read response acceptance criteria guide item file"""
        return read_file("response_ac_guide_item.md")

    def _read_final_response_ac(self) -> str:
        """Read final response template file"""
        return read_file("final_response_ac.txt")
    
    def _read_final_test_response(self) -> str:
        """Read final test response template file"""
        return read_file("final_test_response.txt")

    def _read_testcase_guide_item_json(self) -> str:
        """Read testcase guide item file"""
        return read_file("testcase_guide_item_json.json")

    def _read_json_format_rules(self) -> str:
        """Read json format response file"""
        return read_file("json_format_rules.txt")

    def _read_testcase_guide_item_csv(self) -> str:
        """Read testcase guide item file"""
        return read_file("testcase_guide_item_csv.md")

    async def _extract_existing_testcases_node(self, state: AgentState) -> AgentState:
        """Phase 1 Node 1: Extract existing test cases from requirements"""
        node_name = "extract_existing_testcases"
        state["node_call_count"][node_name] = state["node_call_count"].get(node_name, 0) + 1
        state["current_phase"] = "phase1_extract_testcases"
        
        logger.info(" 🧪 Phase 1.1: Extracting existing test cases from requirements")
        
        prompt = f"""Expert Quality Engineer - Test Case Extraction Prompt
Role Definition
You are an Expert Quality Engineer with specialized expertise in test case extraction and analysis.
Primary Objective
Extract existing test cases from Jira requirements documents with precision and accuracy.
Task Requirements
Data Source Location

Target Section: Comments section of Jira requirements issue
Identification Keywords: Look for entries starting with:

"TestCase" (case-sensitive)
"testcase" (case-sensitive)
"testcases" (case-sensitive)

Critical Instructions

Extract Only Explicitly Mentioned Test Cases: Do not generate, assume, or infer test cases from business logic
Source Verification: Only extract test cases that are clearly documented in the comments section
Exact Text Preservation: Maintain the original wording of test cases as written in the requirements
Complete Data Output: Include all found test cases without summarization, ellipsis (...), or "etc."

Input Data
<requirements>
{state['requirements']}
</requirements>
Analysis Process

Scan the requirements document thoroughly
Identify the comments section within the Jira issue
Locate entries beginning with the specified keywords
Extract the complete test case content
Classify each test case according to the output schema

Output Format
Provide response in Vietnamese language with the following JSON structure:
{{
    "existing_test_cases": [
        {{
            "test_case": "exact test case text from requirements comments section starting with 'testcases', 'testcase', or 'TestCase'",
            "test_type": "positive/negative/edge",
            "coverage_area": "specific aspect or functionality being tested",
            "priority": "high/medium/low"
        }}
    ]
}}
Language Requirements
Response Language: Vietnamese
Preservation Rule: Do not translate English keywords, technical terms, field names, parameters, request body elements, or API endpoints
Mixed Language: Maintain original English technical terminology within Vietnamese explanations

Quality Assurance

JSON Validation: Verify JSON format validity before submission
Completeness Check: Ensure all identified test cases are included
Accuracy Verification: Confirm extracted content matches source material exactly

Edge Cases

If no test cases are found in the comments section, return: {{"existing_test_cases": []}}
If comments section is not present, return empty array
If keywords are found but no valid test cases follow, return empty array

Final Validation Checklist

 All test cases from comments section extracted
 JSON format is valid and parseable
 Vietnamese language used for descriptions
 English technical terms preserved
 No summarization or truncation applied
 Priority and classification assigned to each test case
"""
        try:
            response = await asyncio.to_thread(self.langchain_llm._call, prompt)
            response_clean = self._parse_json_response(response)
            
            try:
                result = json.loads(response_clean)
                state["existing_testcases"] = result.get("existing_test_cases", [])
                state["phase_complete"]["phase1_extract_testcases"] = True
                
                count = len(state["existing_testcases"])
                logger.info(f" ✅ Phase 1.1 completed: Extracted {count} existing test cases")
                
            except json.JSONDecodeError as e:
                logger.error(f" Failed to parse existing test cases JSON: {e}")
                state["existing_testcases"] = []
                state["phase_complete"]["phase1_extract_testcases"] = True
                
        except Exception as e:
            logger.error(f" Error in extracting existing test cases: {e}")
            state["existing_testcases"] = []
            state["phase_complete"]["phase1_extract_testcases"] = True
            
        return state

    async def _generate_missing_testcases_node(self, state: AgentState) -> AgentState:
        """Phase 1 Node 2: Generate missing test cases based on existing test cases + requirements + software_testing_guide"""
        node_name = "generate_missing_testcases"
        state["node_call_count"][node_name] = state["node_call_count"].get(node_name, 0) + 1
        state["current_phase"] = "phase1_generate_missing_testcases"
        
        logger.info(" 🔬 Phase 1.2: Generating missing test cases")
        
        software_testing_guide = self._read_software_testing_guide()
        
        prompt = f"""You are an Expert Quality Engineer specializing in comprehensive test case analysis and generation.

Your objective is to conduct a thorough analysis of existing test cases against the provided requirements and generate missing test cases that ensure complete test coverage. Use the software testing guide as your framework for test case design and naming conventions.

<requirements>
{state['requirements']}
</requirements>

<existing_test_cases>
{json.dumps(state.get('existing_testcases', []), indent=2)}
</existing_test_cases>

<software_testing_guide>
{software_testing_guide}
</software_testing_guide>


LANGUAGE REQUIREMENTS:
- Response in Vietnamese
- Do not translate English keywords, technical terms, field names in request bodies, parameters, API endpoints, or system-specific terminology

ANALYSIS INSTRUCTIONS:
1. Systematically review all existing test cases against the requirements
2. Identify gaps in test coverage across all testing dimensions
3. Ensure comprehensive coverage of both functional and non-functional requirements
4. Consider the complete software testing lifecycle and quality assurance best practices

FOCUS AREAS FOR MISSING TEST CASES:
- Edge cases and boundary value analysis not addressed in existing test cases
- Error handling scenarios and exception conditions
- Negative test cases for invalid inputs and error states
- Integration points between system components
- Performance benchmarks and load testing scenarios
- Security vulnerabilities and authentication/authorization testing
- Data validation and integrity checks
- User experience and accessibility considerations
- Regression testing for critical functionality
- Configuration and environment-specific testing

QUALITY CRITERIA:
- Each test case must be specific, measurable, and actionable
- Test case names must follow the conventions specified in the software testing guide
- Rationale must clearly explain the testing gap being addressed
- Priority assignment should reflect business impact and risk assessment

Provide your analysis in the following JSON response format:

{{
    "missing_test_cases": [
        {{
            "test_case": "detailed test case description using terminology from requirements. test_case name follow instruction in software_testing_guide. ",
            "test_type": "positive/negative/edge/performance/security",
            "category": "functional/integration/unit/performance",
            "priority": "high/medium/low",
            "rationale": "why this test case is important and what gap it fills"
        }}
    ]
}}

JSON FORMAT REQUIREMENTS: {self._read_json_format_rules()}

COMPLETENESS REQUIREMENT:
Do not summarize or skip any data. Please output the full content without using ellipsis (...) or 'etc.' Provide comprehensive and complete test case description
"""
        
        try:
            response = await asyncio.to_thread(self.langchain_llm._call, prompt)
            response_clean = self._parse_json_response(response)
            
            try:
                result = json.loads(response_clean)
                state["generated_missing_testcases"] = result.get("missing_test_cases", [])
                state["phase_complete"]["phase1_generate_missing_testcases"] = True
                
                count = len(state["generated_missing_testcases"])
                logger.info(f" ✅ Phase 1.2 completed: Generated {count} missing test cases")
                
            except json.JSONDecodeError as e:
                logger.error(f" Failed to parse missing test cases JSON: {e}")
                state["generated_missing_testcases"] = []
                state["phase_complete"]["phase1_generate_missing_testcases"] = True
                
        except Exception as e:
            logger.error(f" Error in generating missing test cases: {e}")
            state["generated_missing_testcases"] = []
            state["phase_complete"]["phase1_generate_missing_testcases"] = True
            
        return state

    async def _generate_api_docs_node(self, state: AgentState) -> AgentState:
        """Phase 1 Node 3: Generate API docs based on requirements"""
        node_name = "generate_api_docs"
        state["node_call_count"][node_name] = state["node_call_count"].get(node_name, 0) + 1
        state["current_phase"] = "phase1_generate_api_docs"
        
        logger.info(" 📚 Phase 1.3: Generating API docs")

        prompt = f"""
You are an Expert Quality Engineer and Software Engineer specializing in API documentation generation.

Your task is to analyze the provided code context and generate comprehensive API documentation following the specified guidelines.

<current_context>
{state['context']}
</current_context>

## Instructions:
1. Carefully analyze the current_context code to understand the API's functionality, endpoints, parameters, and responses
2. Generate complete API documentation based on the analysis
3. Follow the API documentation guidelines provided below
4. Ensure all sections are thoroughly documented with realistic examples
5. Use Vietnamese language for descriptions while keeping technical terms, field names, and code examples in English

## API Documentation Guidelines:
{self._read_api_docs_example()}

## Quality Requirements:
- **Completeness**: Document all endpoints, parameters, and responses found in the code
- **Accuracy**: Ensure all technical details match the actual implementation
- **Clarity**: Use clear, concise Vietnamese descriptions while keeping technical terms in English
- **Examples**: Provide realistic, working examples for requests and responses
- **Error Handling**: Include comprehensive error scenarios with appropriate status codes
- **Consistency**: Use consistent naming conventions and formatting throughout

## Response Format Rules:
- Respond in Vietnamese for all descriptions and explanations
- Keep all technical terms, field names, parameter names, and code examples in English
- Ensure the JSON response is valid and properly formatted
- Do not use ellipses (...) or truncate any part of the response
- Include complete examples for request/response bodies
- Provide the full response without shortening

## Final Checklist:
Before returning the response, verify:
- [ ] JSON format is valid and properly structured
- [ ] All required sections are included and complete
- [ ] Examples are realistic and match the API specification
- [ ] Vietnamese descriptions are clear and professional
- [ ] Technical terms remain in English
- [ ] No truncation or ellipses used
- [ ] Logic flow accurately reflects the code implementation

Generate the complete API documentation as a valid JSON response.
"""

        response = await asyncio.to_thread(self.langchain_llm._call, prompt)
        response_clean = self._parse_json_response(response)

        try:
            state["api_docs"] = json.loads(response_clean)
            state["phase_complete"]["phase1_generate_api_docs"] = True

            logger.info(f" ✅ Phase 1.3 completed: Generated API docs")

        except json.JSONDecodeError as e:
            logger.error(f" Failed to parse API docs JSON: {e}")
            state["api_docs"] = {}
            state["phase_complete"]["phase1_generate_api_docs"] = True
        
        return state

    async def _improve_and_finalize_testcases_node(self, state: AgentState) -> AgentState:
        """Phase 1 Node 3: Improve additional test cases and return final test cases with type field"""
        node_name = "improve_and_finalize_testcases"
        state["node_call_count"][node_name] = state["node_call_count"].get(node_name, 0) + 1
        state["current_phase"] = "phase1_improve_testcases"
        
        logger.info(" 🔧 Phase 1.3: Improving and finalizing test cases")
        
        prompt = f"""
You are an Expert Quality Engineer, Software Engineer, and Business Analyst with deep expertise in test case design and software quality assurance.
Your task is to analyze, improve, and consolidate test cases to create a comprehensive final test suite that ensures thorough coverage of all requirements and code changes.
Input Context
<api_docs> {state['api_docs']} </api_docs>
<requirements>
 {state['requirements']}
</requirements>
<code_diff_commit> {state['code_commit']} </code_diff_commit>
<existing_test_cases> {json.dumps(state.get('existing_testcases', []), indent=2)} </existing_test_cases>
<generated_missing_test_cases> {json.dumps(state.get('generated_missing_testcases', []), indent=2)} </generated_missing_test_cases>
<software_testing_guide>
{self._read_software_testing_guide()}
</software_testing_guide>
Detailed Instructions
1. Test Case Improvement and Enhancement
Improve generated_missing_test_cases by:

Language and Clarity: Rewrite descriptions in natural, clear, and professional Vietnamese that focuses on user scenarios, business functionality, and expected behavior
Technical Terms: Preserve all technical terms (keyWord, HTTP 200, HTTP 400, API endpoints, parameter names, etc.) in their original English form without translation
Specificity: Make test cases more specific and actionable based on the API documentation and code changes, ensuring each test case can be implemented without ambiguity
Implementation Independence: Remove code-specific details (class names, method names, enum values) from descriptions; focus on observable behavior and functionality
Completeness: Add new test cases for any scenarios found in requirements or code changes that aren't covered by existing test cases
Structure: Remove "rationale" fields and ensure consistent formatting

2. Gap Analysis and New Test Case Identification
Systematically identify missing coverage for:

Boundary conditions and edge cases not covered in existing tests
Error scenarios and exception handling paths
Integration points between different system components
Performance considerations for critical operations
Security aspects including authentication, authorization, and data validation
Data flow scenarios across different input/output combinations
State transitions and workflow validations

3. Test Case Consolidation Strategy
Create the final comprehensive test suite by:

Preserving existing test cases without modification (mark as "type": "existing")
Including all improved test cases from the generated missing test cases
Adding newly identified test cases based on requirements and code analysis
Ensuring uniqueness by eliminating redundant or overly similar test cases
Maintaining logical grouping of related test scenarios
Balancing coverage across different test types and categories

4. Quality Standards and Classifications
Ensure each test case has:

Clear, actionable descriptions that specify what to test and expected outcomes
Appropriate test_type classification:

positive: Happy path scenarios with valid inputs
negative: Error handling with invalid inputs or conditions
edge: Boundary conditions and limit testing
performance: Response time, throughput, and resource usage
security: Authentication, authorization, and data protection


Correct category assignment:

functional: Business logic and feature validation
integration: Component interaction and data flow
unit: Individual component behavior
performance: Speed, scalability, and resource efficiency


Priority levels based on business impact:

high: Critical functionality, security, or frequently used features
medium: Important but non-critical features
low: Nice-to-have features or edge cases with minimal impact
5. Critical Guidelines
DO NOT:

Delete or modify test cases simply because current code doesn't implement the feature
Change expected behavior to match current implementation when it conflicts with requirements
Overlook or ignore documented requirements that differ from code implementation
Assume current code logic is always correct
Translate technical terms, API parameters, HTTP status codes, or system identifiers into Vietnamese
Use ellipsis (...) or "etc." - provide complete, detailed content

DO:

Prioritize requirements over current code implementation
Identify discrepancies between requirements and code for additional test coverage
Focus on user-facing behavior and business value
Ensure test cases are implementation-agnostic and maintainable
Provide comprehensive coverage without redundancy

Expected Output Format
{{
    "final_test_cases": [
        {{
            "test_case": "Detailed Vietnamese description focusing on user scenario, business functionality, and expected behavior. Technical terms like 'keyWord', 'HTTP 200', parameter names remain in English.",
            "test_type": "positive/negative/edge/performance/security",
            "category": "functional/integration/unit/performance",
            "priority": "high/medium/low",
            "type": "existing/new"
        }}
    ]
}}
JSON Format Requirements
{self._read_json_format_rules()}
Final Checklist

 All test cases are written in clear, professional Vietnamese
 Technical terms remain in original English
 No code-specific implementation details in descriptions
 Each test case focuses on single, testable behavior
 Complete coverage of requirements and code changes
 Valid JSON format with proper escaping
 No truncation or summarization of content
 Appropriate classifications for all test cases

Response Language: Vietnamese (with English technical terms preserved)
Output: Complete JSON response with full content - no ellipsis or abbreviations allowed
"""
        
        try:
            response = await asyncio.to_thread(self.langchain_llm._call, prompt)
            response_clean = self._parse_json_response(response)
            
            try:
                result = json.loads(response_clean)
                state["final_testcases"] = result.get("final_test_cases", [])
                state["phase_complete"]["phase1_improve_testcases"] = True
                
                # Count existing vs new
                existing_count = len([tc for tc in state["final_testcases"] if tc.get("type") == "existing"])
                new_count = len([tc for tc in state["final_testcases"] if tc.get("type") == "new"])
                total_count = len(state["final_testcases"])
                
                logger.info(f" ✅ Phase 1.3 completed: Finalized {total_count} test cases ({existing_count} existing + {new_count} new)")
                
            except json.JSONDecodeError as e:
                logger.error(f" Failed to parse final test cases JSON: {e}")
                state["final_testcases"] = []
                state["phase_complete"]["phase1_improve_testcases"] = True
                
        except Exception as e:
            logger.error(f" Error in improving test cases: {e}")
            state["final_testcases"] = []
            state["phase_complete"]["phase1_improve_testcases"] = True
            
        return state

    async def _generate_current_ac_node(self, state: AgentState) -> AgentState:
        """Phase 2 Node 1: Generate current AC (acceptance criteria) based on requirements"""
        node_name = "generate_current_ac"
        state["node_call_count"][node_name] = state["node_call_count"].get(node_name, 0) + 1
        state["current_phase"] = "phase2_generate_current_ac"
        
        logger.info(" 📋 Phase 2.1: Generating current acceptance criteria from requirements")
        response_ac_guide_item = self._read_response_ac_guide_item()
        prompt = f"""

You are an Expert Quality Engineer and Business Analyst specializing in requirements analysis and test case development.

## Task
Generate comprehensive acceptance criteria (AC) based on the provided requirements document. Extract both explicitly stated and implicitly derived criteria to ensure complete test coverage.

## Input Requirements
```
{state['requirements']}
```

## Analysis Instructions
1. **Thoroughly analyze** the requirements document to identify:
   - Functional requirements and expected behaviors
   - Non-functional requirements (performance, security, usability)
   - Business rules and constraints
   - User interactions and system responses
   - Error handling and edge cases
   - Data validation requirements

2. **Extract and generate** acceptance criteria that cover:
   - Primary user flows and scenarios
   - Alternative paths and exceptions
   - System integrations and dependencies
   - Boundary conditions and limits
   - Security and authorization requirements

## Acceptance Criteria Standards
Each acceptance criterion must be:
- **Specific and measurable**: Include quantifiable metrics where applicable
- **Testable**: Can be verified through manual or automated testing
- **Clear and unambiguous**: Written in plain language with no ambiguity
- **Behavior-focused**: Describe what the system should do, not how it should do it
- **Complete**: Cover all scenarios including happy path, alternative flows, and error cases

## Response Format
Provide a valid JSON response with the following structure:

{{
    "current_ac": [
        // List of acceptance criteria following the specified format
        // Use format: {response_ac_guide_item}
    ]
}}

## Important Guidelines
- **Language**: Respond in Vietnamese while preserving English technical terms, keywords, field names, API parameters, and system terminology
- **Completeness**: Include all relevant acceptance criteria without summarization, omission, or truncation
- **JSON Validity**: Ensure the response is properly formatted, valid JSON before returning
- **Format Rules**: Follow the specified JSON format rules: {self._read_json_format_rules()}
- **No Placeholders**: Do not use ellipsis (...), "etc.", or similar abbreviations

## Quality Checklist
Before finalizing your response, verify:
- [ ] All requirements have been analyzed and covered
- [ ] Each AC is specific, testable, and unambiguous
- [ ] JSON format is valid and follows the specified structure
- [ ] Response is in Vietnamese with preserved English technical terms
- [ ] No content has been summarized or omitted
"""
        
        try:
            response = await asyncio.to_thread(self.langchain_llm._call, prompt)
            response_clean = self._parse_json_response(response)
            
            try:
                result = json.loads(response_clean)
                state["current_ac"] = result.get("current_ac", [])
                state["phase_complete"]["phase2_generate_current_ac"] = True
                
                count = len(state["current_ac"])
                logger.info(f" ✅ Phase 2.1 completed: Generated {count} current acceptance criteria")
                
            except json.JSONDecodeError as e:
                logger.error(f" Failed to parse current AC JSON: {e}")
                state["current_ac"] = []
                state["phase_complete"]["phase2_generate_current_ac"] = True
                
        except Exception as e:
            logger.error(f" Error in generating current AC: {e}")
            state["current_ac"] = []
            state["phase_complete"]["phase2_generate_current_ac"] = True
            
        return state

    async def _generate_missing_ac_node(self, state: AgentState) -> AgentState:
        """Phase 2 Node 2: Analyze current requirements and current AC, generate missing AC"""
        node_name = "generate_missing_ac"
        state["node_call_count"][node_name] = state["node_call_count"].get(node_name, 0) + 1
        state["current_phase"] = "phase2_generate_missing_ac"
        
        logger.info(" 🔍 Phase 2.2: Analyzing and generating missing acceptance criteria")
        response_ac_guide_item = self._read_response_ac_guide_item()
        prompt = f"""
        You are an Expert Quality Engineer specializing in comprehensive requirements analysis and acceptance criteria development.

Your primary objective is to conduct a thorough analysis of the provided requirements and existing acceptance criteria to systematically identify gaps and generate missing acceptance criteria that ensure complete test coverage and requirement validation.

<requirements>
{state['requirements']}
</requirements>

<current_ac>
{json.dumps(state.get('current_ac', []), indent=2)}
</current_ac>

## Analysis Framework

Perform a comprehensive gap analysis by examining the requirements against current acceptance criteria. Your analysis should be systematic and cover all critical aspects of software quality assurance.

### Critical Areas to Evaluate:

**Functional Coverage:**
- Core business logic and workflows not addressed in current AC
- User interaction scenarios and use cases
- Data processing and manipulation requirements
- Feature completeness and functional boundaries

**Non-Functional Requirements:**
- Performance benchmarks and load handling
- Security protocols and access controls
- Usability standards and user experience metrics
- Scalability and reliability requirements
- Compatibility and browser/device support

**Edge Cases and Error Handling:**
- Boundary conditions and limit testing
- Invalid input scenarios and validation rules
- System failure recovery mechanisms
- Timeout and connection error handling
- Resource exhaustion scenarios

**Integration and System Requirements:**
- API endpoints and data exchange protocols
- Third-party service integrations
- Database interactions and data consistency
- Cross-system communication requirements
- Dependency management and version compatibility

**Data Validation and Business Rules:**
- Input validation criteria and formats
- Business logic constraints and rules
- Data integrity and consistency checks
- Authorization and permission requirements
- Audit trail and logging requirements

**User Experience and Accessibility:**
- Interface responsiveness and navigation
- Accessibility compliance (WCAG guidelines)
- Error message clarity and user guidance
- Mobile and responsive design requirements
- Internationalization and localization needs

## Output Requirements

Generate a comprehensive JSON response that includes all identified missing acceptance criteria. Each acceptance criterion should be detailed, testable, and aligned with industry best practices.

Response Format:
{{
    "missing_ac": [
        // Format Rules: {self._read_json_format_rules()}
    ]
}}

## Quality Standards

- **Language**: Respond in Vietnamese while preserving English technical keywords (e.g., API endpoints, request/response parameters, field names, HTTP methods, status codes)
- **JSON Validity**: Ensure the response is properly formatted, valid JSON before submission
- **Completeness**: Provide comprehensive coverage without summarization, ellipsis (...), or truncation
- **Specificity**: Each acceptance criterion should be specific, measurable, and testable
- **Traceability**: Ensure all acceptance criteria can be traced back to specific requirements

## Validation Checklist

Before submitting your response, verify:
1. JSON syntax is valid and properly formatted
2. All identified gaps are covered with specific acceptance criteria
3. Technical terms and field names remain in English
4. Vietnamese explanations are clear and professional
5. No content has been abbreviated or summarized
6. Response format matches the specified structure exactly

Generate missing acceptance criteria that provide complete test coverage and ensure robust quality validation of the specified requirements.
"""
        
        try:
            response = await asyncio.to_thread(self.langchain_llm._call, prompt)
            response_clean = self._parse_json_response(response)
            
            try:
                result = json.loads(response_clean)
                state["generated_missing_ac"] = result.get("missing_ac", [])
                state["phase_complete"]["phase2_generate_missing_ac"] = True
                
                count = len(state["generated_missing_ac"])
                logger.info(f" ✅ Phase 2.2 completed: Generated {count} missing acceptance criteria")
                
            except json.JSONDecodeError as e:
                logger.error(f" Failed to parse missing AC JSON: {e}")
                state["generated_missing_ac"] = []
                state["phase_complete"]["phase2_generate_missing_ac"] = True
                
        except Exception as e:
            logger.error(f" Error in generating missing AC: {e}")
            state["generated_missing_ac"] = []
            state["phase_complete"]["phase2_generate_missing_ac"] = True
            
        return state

    async def _improve_and_finalize_ac_node(self, state: AgentState) -> AgentState:
        """Phase 2 Node 3: Improve additional AC and create final AC list"""
        node_name = "improve_and_finalize_ac"
        state["node_call_count"][node_name] = state["node_call_count"].get(node_name, 0) + 1
        state["current_phase"] = "phase2_improve_ac"
        
        logger.info(" 🔧 Phase 2.3: Improving and finalizing acceptance criteria")
        
        response_ac_guide_item = self._read_response_ac_guide_item()
        
        prompt = f"""
        Quality Engineer & Business Analyst - Acceptance Criteria Improvement

Role
You are an Expert Quality Engineer and Business Analyst specializing in creating comprehensive, testable acceptance criteria.

Task
Analyze and improve the provided acceptance criteria to create a final, comprehensive AC list that aligns with technical implementation and business requirements.

Context Data

API Documentation
{state['api_docs']}

Code Changes/Commits
{state['code_commit']}

Current Acceptance Criteria
{json.dumps(state.get('current_ac', []), indent=2)}

Generated Missing Acceptance Criteria
{json.dumps(state.get('generated_missing_ac', []), indent=2)}

Final Test Cases (Phase 1)
{json.dumps(state.get('final_testcases', []), indent=2)}

Improvement Guidelines

1. Technical Alignment
- Reference API documentation to ensure AC specificity and implementability
- Consider code differences/commits for implementation-specific requirements
- Align with actual system architecture and data structures

2. Quality Standards
- Follow software AC best practices (GIVEN-WHEN-THEN format where applicable)
- Ensure each AC is testable, measurable, and unambiguous
- Remove code dependencies - AC should focus on behavior, not implementation

3. Completeness Check
- Verify alignment with final test cases from Phase 1
- Ensure coverage of all functional and non-functional requirements
- Include edge cases and error scenarios

4. Business Value
- Maintain focus on user outcomes and business objectives
- Ensure AC supports the overall feature goals
- Include validation criteria for success metrics

Output Requirements

Response Format
{{
    "final_ac": "response in format {response_ac_guide_item}. AC MUST NOT depend on code, it must not contain any code"
}}

Content Standards
- Language: Respond in Vietnamese, but do not translate English keywords (e.g., field names, parameters, technical terms)
- Completeness: Output full content without ellipsis (...) or 'etc.'
- JSON Validation: Verify JSON format validity before response
- No Code Dependencies: Focus on behavior and outcomes, not implementation details

Quality Checklist
Before finalizing, ensure:
- Each AC is independently testable
- No code snippets or implementation details in AC
- Clear acceptance criteria using business language
- Alignment with API documentation requirements
- Coverage of both happy path and error scenarios
- Vietnamese language with preserved English technical terms
- Valid JSON format

Additional Instructions
- Apply JSON format rules: {self._read_json_format_rules()}
- Combine existing and improved additional acceptance criteria
- Prioritize clarity and testability over technical complexity
- Ensure traceability between AC and test cases
"""
        
        try:
            response = await asyncio.to_thread(self.langchain_llm._call, prompt)
            response_clean = self._parse_json_response(response)
            
            try:
                result = json.loads(response_clean)
                state["final_ac"] = result.get("final_ac", [])
                state["phase_complete"]["phase2_improve_ac"] = True
                
                # Count existing vs new
                existing_count = len([ac for ac in state["final_ac"] if ac.get("type") == "existing"])
                new_count = len([ac for ac in state["final_ac"] if ac.get("type") == "new"])
                total_count = len(state["final_ac"])
                
                logger.info(f" ✅ Phase 2.3 completed: Finalized {total_count} acceptance criteria ({existing_count} existing + {new_count} new)")
                
            except json.JSONDecodeError as e:
                logger.error(f" Failed to parse final AC JSON: {e}")
                state["final_ac"] = []
                state["phase_complete"]["phase2_improve_ac"] = True
                
        except Exception as e:
            logger.error(f" Error in improving AC: {e}")
            state["final_ac"] = []
            state["phase_complete"]["phase2_improve_ac"] = True
            
        return state

    async def _generate_final_response(self, state: AgentState) -> AgentState:
        """Phase 3: Analyze code coverage of test cases and AC from previous phases"""
        node_name = "generate_final_response"
        state["node_call_count"][node_name] = state["node_call_count"].get(node_name, 0) + 1
        state["current_phase"] = "phase3_additional_coverage"
        
        logger.info(" 📊 Phase 3: Analyzing code coverage of test cases and AC")
        
        response_ac_guide = self._read_response_ac_guide()
        
        prompt = f"""You are an Expert Quality Engineer and Java Software Engineer with extensive experience in code analysis and testing.

Your task is to analyze whether the test cases and acceptance criteria from previous phases are covered by the current code implementation.

<current_context>
{state['context']}
<current_context/>

<requirements>
{state['requirements']}
<requirements/>

<code_diff_commit>
{state['code_commit']}
<code_diff_commit/>

<final_testcases>
{json.dumps(state.get('final_testcases', []), indent=2)}
<final_testcases/

<final_ac>
{json.dumps(state.get('final_ac', []), indent=2)}
<final_ac/>

=== ANALYSIS INSTRUCTIONS ===

Analyze the current code implementation to determine:

1. TEST CASE COVERAGE ANALYSIS:
   - Can each test case be executed against the current code?
   - Does the code have the required methods/endpoints?
   - Does the code meet the exact requirements specified?
   - First analyze testcase_csv, then analyze ac_analysis

2. ACCEPTANCE CRITERIA COVERAGE:
   - Is each acceptance criteria supported by the current code implementation?
   - Classify coverage level: fully covered, partially covered, or not covered at all

3. CODE CHANGE ASSESSMENT:
   - What code changes or additions are needed to support the test cases and AC?
   - Analyze each code change in the <code_diff_commit/> section
   - Read the full code in <current_context/> and determine if changes are related to requirements

4. UNRELATED CHANGES IDENTIFICATION:
   Read new code or deleted code in <code_diff_commit/>. Consider a change "unrelated" if it:
   - Doesn't support any stated requirement
   - Appears to be refactoring unrelated to the feature
   - Introduces new functionality not mentioned in requirements
   - Modifies code that doesn't impact the required behavior

=== RESPONSE FORMAT ===

Provide a valid JSON response in the following format:
testcase_csv: Analyze all test cases from <final_testcases/> and return in new json format. Based on all test cases from <final_testcases/>. Use <current_context/> and <requirements/>. 
{{
    "testcase_csv": [
            {{        
        "testCaseKey": "(String) Should be an empty string. This field will be populated later based on specific test case naming conventions.",
        "testCaseName": "(String) The exact test case text from requirements, typically found in the `comment` section of a Jira issue, starting with `testcases` or `testcase`. This field must be prefixed with [ServiceName], where ServiceName is the name of the service for which the test case is being written (e.g., '[QuizLibraryService] Kiểm tra API tìm kiếm trả status_code = 400 khi không truyền param keyWord').",
        "prepareFileName": "(String) The meaningful filename of the data preparation script or file associated with this test case (e.g., 'prepareData_1.csv').",
        "httpMethod": "(String) The HTTP method used for the request (e.g., 'GET', 'POST', 'PUT', 'DELETE').",
        "expectedHttpCode": "(String) The expected HTTP status code of the response, represented as a string (e.g., '200', '400', '500').",
        "requestParamName": "(String) The name of the request parameter if present; otherwise, an empty string ('').",
        "requestParamValue": "(String or Number) The literal value of the request parameter, formatted as a valid JSON string (enclosed in double quotes) or a plain number (if numeric). Do not use code expressions, variables, functions, concatenation, or repetition methods (e.g., no '.repeat(...)', '+', or '${{}}' syntax). Examples of valid values: '\\"hello\\"', '\\"user123\\"', 42, '\\"password\\"'. For long strings (e.g., exceeding 100 characters), store the value in the file specified by 'prepareFileName' and reference it with a placeholder string (e.g., '\\"see_prepareData_1\\"'). If the parameter is missing, use an empty string: '\\"\\"'. The result must be a valid JSON-compatible literal value.",
        "requestBody": "(String) The JSON string representation of the request body. If no request body is present, return empty.",
        "expectBody": "(String) The JSON string representation of the expected response body. For specific error messages, include the full JSON stringg If the JSON string exceeds 100 characters, store it in a separate file named 'expectBody_N.json' (where N is an incremental number for each test case) and put the filename (e.g., 'expectBody_1.json') in this field instead of the full JSON string. Use '{{}}' for an empty expected body, or 'null' if the body content is not relevant to the assertion.",
        "code_coverage_score": "(String) A string indicating the code coverage percentage for this test case, along with a brief note (e.g., '0%: No logic for test case.').",
        "explain_coverage": "(String) A detailed explanation of the current code coverage status, outlining what is missing or needs improvement in the code to meet the test case's requirements."
        }}
    ],
    "ac_analysis": "{response_ac_guide} analysis from all <final_ac/>. Only analyze 'Code Location', 'Assessment', 'Priority'. Test case and other information must be exactly same as field 'testcase' in <final_testcases/>. AC name and other information must be exactly same as final_ac. DO NOT assume, always return all items.",
    "unrelated_changes": [
        {{
            "file_path": "path/to/file.java", // file change in <code_diff_commit/>, do not use code in <current_context>
            "change_type": "addition|deletion|modification",
            "code_snippet": "actual code that was changed in <code_diff_commit/>, do not mention comments, just active codes",
            "reason": "detailed explanation why this change is unrelated",
            "severity": "low|medium|high",
            "category": "refactoring|new_feature|bug_fix|cleanup|configuration|other"
        }}
    ]
}}

=== QUALITY GUIDELINES ===

1. ACCURACY: Always check your response before returning
2. COMPLETENESS: DO NOT assume, always return exactly the full response
3. CONSISTENCY: Maintain exact field names and structure as specified
4. LANGUAGE: Response MUST be in Vietnamese. If existing test cases and AC are in English, translate to Vietnamese with full context and information. Do not translate English keywords (e.g., fields in request body or params)

=== JSON FORMAT RULES ===
{self._read_json_format_rules()}

=== CRITICAL REMINDERS ===
- Analyze ALL test cases from final_testcases without exception
- Analyze ALL acceptance criteria from final_ac without exception
- Provide detailed code location analysis for each AC
- Maintain traceability between test cases and acceptance criteria
- Ensure all unrelated changes are properly categorized and justified
"""
        
        try:
            # Call LLM and get response
            response = await asyncio.to_thread(self.langchain_llm._call, prompt)
            response_clean = self._parse_json_response(response)
            
            try:
                # Parse the cleaned response
                result = json.loads(response_clean)
                # logger.info(f"Final analysis result: {result}")
                
                # Validate required fields
                if "ac_analysis" not in result:
                    raise ValueError("Missing required fields in LLM response")
                
                
                # Process AC analysis
                ac_analysis = result.get("ac_analysis", {})
 
                testcase_csv = result.get("testcase_csv", [])

                unrelated_changes = result.get("unrelated_changes", [])
                
                # Update state with validated data
                state["final_analysis_result"] = {
                    "ac_analysis": ac_analysis,
                    "testcase_csv": testcase_csv,
                    "unrelated_changes": unrelated_changes
                }
                
                # Mark phase as complete
                state["phase_complete"]["phase3_additional_coverage"] = True

                return state
                
            except json.JSONDecodeError as e:
                logger.error(f" Failed to parse additional coverage JSON: {e}")
                state["additional_coverage"] = {
                    "error": f"JSON parsing failed: {str(e)}",
                    "raw_response": response_clean
                }
                state["phase_complete"]["phase3_additional_coverage"] = True
                
            except ValueError as e:
                logger.error(f" Invalid response structure: {e}")
                state["additional_coverage"] = {
                    "error": f"Invalid response structure: {str(e)}",
                    "raw_response": response_clean
                }
                state["phase_complete"]["phase3_additional_coverage"] = True
                
        except Exception as e:
            logger.error(f" Error in generating additional coverage: {e}")
            state["additional_coverage"] = {
                "error": f"Analysis failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
            state["phase_complete"]["phase3_additional_coverage"] = True
            
        return state

    async def _format_output_node(self, state: AgentState) -> AgentState:
        """Final Phase: Format final output into JSON + HTML"""
        node_name = "format_output"
        state["node_call_count"][node_name] = state["node_call_count"].get(node_name, 0) + 1
        state["current_phase"] = "format_output"
        
        logger.info(" 🎨 Final Phase: Formatting comprehensive output")
        
        # Collect all analysis results from 3 phases
        analysis_results = {
            "endpoint": state["endpoint"],
            "result": state["final_analysis_result"]
        }
        
        # Convert to JSON string
        json_response = json.dumps(analysis_results, indent=2, ensure_ascii=False)
        
        # Generate HTML response
        html_content = await self._generate_html_with_llm(json_response)
        
        state["final_response"] = json_response
        state["html_response"] = html_content
        state["phase_complete"]["format_output"] = True
        
        logger.info(" ✅ Final Phase completed: Comprehensive output formatted")
        
        return state

    def _build_comprehensive_document(self, state: AgentState) -> str:
        """Build comprehensive document from all analysis phases"""
        coverage_analysis = state.get("coverage_analysis", {})
        testcases_analysis = state.get("current_testcases_analysis", {})
        
        document_parts = []
        
        # Add coverage summary
        if coverage_analysis.get("coverage_summary"):
            document_parts.append(f"## Phân tích Coverage\n{coverage_analysis['coverage_summary']}")
        
        # Add test cases quality assessment
        if testcases_analysis.get("test_quality_assessment"):
            document_parts.append(f"## Đánh giá chất lượng Test Cases\n{testcases_analysis['test_quality_assessment']}")
        
        # Add detailed context information
        document_parts.append(f"## Mô tả chi tiết\n")
        document_parts.append("Endpoint này được phân tích với các thành phần sau:")
        document_parts.append(f"- Endpoint: {state['endpoint']}")
        document_parts.append(f"- Context length: {len(state['context'])} characters")
        
        # Add requirements summary if available
        if state.get('requirements'):
            requirements_summary = state['requirements'][:2000] + "..." if len(state['requirements']) > 2000 else state['requirements']
            document_parts.append(f"- Requirements: {requirements_summary}")
        
        return "\n\n".join(document_parts)

    async def _generate_html_with_llm(self, response: str) -> str:
        """Generate HTML from JSON analysis result using LLM."""
        try:
            logger.info(" 🤖 Calling LLM for HTML generation...")
            
            # Create a prompt for HTML generation
            prompt = f"""
You are an Data Analyst. Convert the following JSON into a beautiful, well-structured Markdown report.

JSON Analysis Data:
{response}

Test case must be analyze first, then analyze Acceptance Criteria (AC)
Here is response structure:
{self._read_final_test_response()}

{self._read_final_response_ac()}

{self._read_testcase_guide_item_csv()}

Recheck json response format before return to make sure it is valid json.
"""
            
            logger.info(f" Sending prompt to LLM ({len(prompt)} characters)")
            
            # Use the LLM to generate HTML
            html_response = await asyncio.to_thread(self.langchain_llm._call, prompt)
            
            logger.info(f" Received LLM response ({len(html_response)} characters)")
            
            return html_response
            
        except Exception as e:
            logger.error(f" Error generating HTML with LLM: {str(e)}")
            # Fallback to basic HTML if LLM fails
            return ""

    async def run(
            self,
            *,
            endpoint: str,
            requirements_txt: str,
            user_text: str,
            code_commit: str = "",
            changed_methods: List[Dict[str, str]] = []
    ) -> Dict[str, Any]:
        """Run the LangGraph analysis chain and return structured results."""
        try:
            self._validate_inputs(requirements_txt, user_text)
        except ValueError as e:
            logger.error(f" Input validation failed: {str(e)}")
            raise AnalysisError(f"Invalid input: {str(e)}")
        
        logger.info(f" Starting LangGraph AnalyzerChain for endpoint: {endpoint}")
    
        try:
            neo4j_conn = get_neo4j_connection()
            endpointNodes = []
            for symbol in changed_methods:
                endpointNode = neo4j_conn.find_endpoint_node(symbol["class"], symbol["method"], self.project_id)
                endpointNodes.extend(endpointNode)

            if len(endpointNodes) > 0:
                logger.info(f"endpointNodes: {endpointNodes[0]}")

            relatedNodes = []
            for endpoint in endpointNodes:
                class_name = endpoint.get("class_name")
                method_name = endpoint.get("method_name")
                relatedNode = neo4j_conn.find_related_nodes(class_name, method_name, self.project_id)
                relatedNodes.extend(relatedNode)
            
            configurationNode = neo4j_conn.find_configuration_node(self.project_id)
            if len(configurationNode) > 0:
                logger.info(f"configurationNode: {configurationNode[0]}")
                relatedNodes.extend(configurationNode)

            # Deduplicate relatedNodes based on a unique identifier
            seen_related = set()
            unique_relatedNodes = []
            for node in relatedNodes:
                identifier = node.get("id") or hash(node.get("content", ""))
                if identifier not in seen_related:
                    seen_related.add(identifier)
                    unique_relatedNodes.append(node)

            relatedNodes = unique_relatedNodes
            if len(relatedNodes) > 0:
                logger.info(f"relatedNodes: {relatedNodes[0]}")

            initial_context = "\n\n".join([node.get("content") for node in relatedNodes])
            endpoint_str = "\n\n".join([node.get("endpoint") for node in endpointNodes])

            initial_state: AgentState = {
                "question": f"Analyze the REST endpoint '{endpoint}' according to the requirements and test cases.",
                "context": initial_context,
                "endpoint": endpoint_str,
                "requirements": requirements_txt,
                "user_text": user_text,
                "code_commit": code_commit,
                "history": [],
                "retrieved_symbols": [],
                "seen_context": [],
                "final_response": None,
                "html_response": None,
                "iteration_count": 0,
                "last_tool_call_symbols": [],
                "new_retrieved_symbols": [],
                "node_call_count": {},
                "current_phase": "phase1_extract_testcases",
                "phase_complete": {
                    "phase1_extract_testcases": False,
                    "phase1_generate_missing_testcases": False,
                    "phase1_improve_testcases": False,
                    "phase2_generate_current_ac": False,
                    "phase2_generate_missing_ac": False,
                    "phase2_improve_ac": False,
                    "phase3_additional_coverage": False,
                    "format_output": False
                },
                "existing_testcases": [],
                "generated_missing_testcases": [],
                "final_testcases": [],
                "current_ac": [],
                "generated_missing_ac": [],
                "final_ac": [],
                "additional_coverage": {},
                "final_analysis_result": {},
                "needs_more_context": False,
                "api_docs": {}
            }
            
            logger.info(" Step 3: Starting LangGraph analysis workflow...")
            final_state = self.graph.invoke(initial_state)
                    
            logger.info(" Step 4: Parsing and structuring final response...")

            result = {
                "markdown_response": final_state.get("html_response", ""),
                "json_response": final_state.get("final_analysis_result", ""),
            }
            logger.info(f" Analysis complete - returning final response")
            return result
            
        except AnalysisError:
            raise
        except Exception as e:
            logger.error(f" LangGraph analysis failed: {str(e)}")
            return {}

    async def _fallback_analysis(
            self,
            *,
            endpoint: str,
            requirements_txt: str,
            user_text: str,
            initial_context: str
    ) -> Dict[str, Any]:
        """Fallback to original analysis approach if agent fails."""
        logger.info(" Using fallback analysis method")
        try:
            prompt = PromptBuilder.build_analysis_prompt(
                endpoint=endpoint,
                context=initial_context,
                requirements=requirements_txt,
                testcases="",  # Empty test cases for fallback
                user_text=user_text,
            )
            resp = await asyncio.to_thread(self.llm.invoke, prompt)
            try:
                result_dict = json.loads(resp)
                logger.info(" Fallback analysis returned valid JSON")
                return AnalysisResult(
                    document=result_dict.get("document", ""),
                    requirement_coverage=result_dict.get("requirement_coverage", []),
                    improvements=result_dict.get("improvements", []),
                    endpoint=endpoint,
                    existed_test_cases=result_dict.get("existed_test_cases", []),
                    additional_test_cases=result_dict.get("additional_test_cases", []),
                    curl_command=result_dict.get("curl_command", ""),
                    response_ac=result_dict.get("response_ac", ""),
                    html_response=result_dict.get("html_response", ""),
                    raw_response=resp,
                    analysis_method="fallback"
                ).__dict__
            except json.JSONDecodeError:
                logger.warning(" Fallback analysis failed to return JSON")
                return AnalysisResult(
                    document="Fallback analysis completed but not in JSON format",
                    requirement_coverage=[],
                    improvements=[],
                    endpoint=endpoint,
                    existed_test_cases=[],
                    additional_test_cases=[],
                    curl_command="",
                    response_ac="",
                    html_response="<div class='error'>Fallback analysis failed to return JSON</div>",
                    raw_response=resp,
                    analysis_method="fallback"
                ).__dict__
        except Exception as e:
            logger.error(f" Fallback analysis execution failed: {str(e)}")
            raise AnalysisError(f"Fallback analysis failed: {str(e)}")

    def _validate_inputs(self, requirements_txt: str, user_text: str) -> None:
        """Validate input parameters."""
        for param_name, param_value in [
            ("requirements_txt", requirements_txt),
            ("user_text", user_text)
        ]:
            if not isinstance(param_value, str):
                raise ValueError(f"{param_name} must be a string")

    def _parse_json_response(self, response_text: str) -> str:
        import re
        match = re.search(r"```json\s*([\s\S]+?)\s*```[\s\n]*", response_text)
        if match:
            return match.group(1).strip()
        match = re.search(r"```json\s*([\s\S]+?)\s*```", response_text)
        if match:
            return match.group(1).strip()
        
        match = re.search(r"```\s*([\s\S]+?)\s*```", response_text)
        if match:
            return match.group(1).strip()
        match = re.search(r"\{[\s\S]+\}", response_text)
        if match:
            return match.group(0).strip()
        return response_text.strip()

    def _parse_graph_response(self, graph_response: str, endpoint: str) -> AnalysisResult:
        """Parse LangGraph response and create structured result."""
        response_text = graph_response.strip()
        response_text = self._parse_json_response(response_text)
        
        # Use greedy matching to capture the full JSON response
        json_match = re.search(r'\{.*"document".*\}', response_text, re.DOTALL)
        if json_match and not response_text.strip().startswith('{'):
            response_text = json_match.group(0)
        
        try:
            result_dict = json.loads(response_text)
            logger.info(" LangGraph returned valid JSON analysis")
            return AnalysisResult(
                document=result_dict.get("document", ""),
                requirement_coverage=result_dict.get("requirement_coverage", []),
                improvements=result_dict.get("improvements", []),
                existed_test_cases=result_dict.get("existed_test_cases", []),
                additional_test_cases=result_dict.get("additional_test_cases", []),
                curl_command=result_dict.get("curl_command", ""),
                response_ac=result_dict.get("response_ac", ""),
                html_response=result_dict.get("html_response", ""),
                raw_response=graph_response,
                endpoint=endpoint,
                analysis_method="langgraph"
            )
        except json.JSONDecodeError as e:
            logger.warning(f" LangGraph response is not valid JSON: {str(e)[:100]}...")
            return AnalysisResult(
                document="Analysis completed but not in JSON format",
                requirement_coverage=[],
                improvements=[],
                existed_test_cases=[],
                additional_test_cases=[],
                curl_command="",
                response_ac="",
                html_response="<div class='error'>Failed to parse JSON response</div>",
                raw_response=graph_response,
                endpoint=endpoint,
                analysis_method="langgraph"
            )

class StreamingAnalyzerChain(AnalyzerChain):
    """Extended AnalyzerChain with streaming capabilities"""
    
    def __init__(self, project_id: str, model_name: str, api_key: str, streaming_queue: asyncio.Queue):
        super().__init__(project_id, model_name, api_key)
        self.streaming_queue = streaming_queue
        
    async def _emit_status(self, event_type: str, data: Dict[str, Any]):
        """Emit status update to streaming queue"""
        event = {
            "type": event_type,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        await self.streaming_queue.put(event)
    
    async def _stream_generator(self) -> AsyncGenerator[str, None]:
        """Generate SSE stream from queue"""
        while True:
            try:
                # Wait for next event with timeout
                event = await asyncio.wait_for(self.streaming_queue.get(), timeout=1.0)
                
                # Format as SSE
                sse_data = f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                yield sse_data
                
                # Mark task as done
                self.streaming_queue.task_done()
                
                # Check if this is the final event
                if event["type"] == "analysis_complete":
                    break
                    
            except asyncio.TimeoutError:
                # Send heartbeat to keep connection alive
                heartbeat = f"data: {json.dumps({'type': 'heartbeat', 'timestamp': datetime.now().isoformat()})}\n\n"
                yield heartbeat
                continue
            except Exception as e:
                error_event = {
                    "type": "error",
                    "timestamp": datetime.now().isoformat(),
                    "data": {"error": str(e)}
                }
                yield f"data: {json.dumps(error_event)}\n\n"
                break
    
    async def _extract_existing_testcases_node_streaming(self, state: AgentState) -> AgentState:
        """Phase 1 Node 1: Extract existing test cases with streaming"""
        await self._emit_status("node_start", {
            "node": "extract_existing_testcases",
            "phase": "1.1",
            "result": {
                "status": "start",
                "html_response": "### 🔧 Starting to extract existing test cases from requirements"
            }
        })
        
        # Call original synchronous method
        state = await self._extract_existing_testcases_node(state)

        markdown = ""
        try:
            markdown = convert_json_testcase_to_markdown(state.get("existing_testcases", []))
        except Exception as e:
            logger.error(f"Error converting existing test cases to markdown: {str(e)}")
        
        header = "### 🔅 Existing test cases \n"
        await self._emit_status("node_complete", {
            "node": "extract_existing_testcases",
            "phase": "1.1",
            "result": {
                "html_response": header + markdown,
                "status": "completed"
            }
        })
        
        return state
    
    async def _generate_missing_testcases_node_streaming(self, state: AgentState) -> AgentState:
        """Phase 1 Node 2: Generate missing test cases with streaming"""
        await self._emit_status("node_start", {
            "node": "generate_missing_testcases",
            "phase": "1.2",
            "result": {
                "status": "start",
                "html_response": "### 🔦 Starting to generate missing test cases"
            }
        })
        
        state = await self._generate_missing_testcases_node(state)

        markdown = ""
        try:
            markdown = convert_json_testcase_to_markdown(state.get("generated_missing_testcases", []))
        except Exception as e:
            logger.error(f"Error converting generated missing test cases to markdown: {str(e)}")

        header = "### 🔦 Generated missing test cases \n"
        await self._emit_status("node_complete", {
            "node": "generate_missing_testcases",
            "phase": "1.2",
            "result": {
                "html_response": header + markdown,
                "status": "completed"
            }
        })
        
        return state

    async def _generate_api_docs_node_streaming(self, state: AgentState) -> AgentState:
        """Phase 1 Node 3: Generate API docs with streaming"""
        await self._emit_status("node_start", {
            "node": "generate_api_docs",
            "phase": "1.3",
            "result": {
                "status": "start",
                "html_response": "### 📄 Starting to read code base to extract information"
            }
        })

        state = await self._generate_api_docs_node(state)

        markdown = ""
        # try:
            # markdown = convert_api_doc_json_to_markdown(state.get("api_docs", {}))
        # except Exception as e:
            # logger.error(f"Error converting API docs to markdown: {str(e)}")
        
        # header = "### API docs \n"
        # await self._emit_status("node_complete", {
        #     "node": "generate_api_docs",
        #     "phase": "1.3",
        #     "result": {
        #         "html_response": header + markdown,
        #         "status": "completed"
        #     }
        # })
        try:
            header = "### 🔎 Read current codebase to extract information successfully \n"
            await self._emit_status("node_complete", {
                "node": "generate_api_docs",
                "phase": "1.3",
                "result": {
                    "html_response": header,
                    "status": "completed"
                }
            })
        except Exception as e:
            logger.error(f"Error generating API docs: {str(e)}")
        
        return state
    
    async def _improve_and_finalize_testcases_node_streaming(self, state: AgentState) -> AgentState:
        """Phase 1 Node 3: Improve and finalize test cases with streaming"""
        await self._emit_status("node_start", {
            "node": "improve_and_finalize_testcases", 
            "phase": "1.3",
            "result": {
                "status": "start",
                "html_response": "### 📋 Starting to improve and finalize test cases"
            }
        })
        
        state = await self._improve_and_finalize_testcases_node(state)
        
        markdown = ""
        try:
            markdown = convert_json_testcase_to_markdown(state.get("final_testcases", []))
        except Exception as e:
            logger.error(f"Error converting final test cases to markdown: {str(e)}")
        
        header = "### 📋 Final test cases \n"
        await self._emit_status("node_complete", {
            "node": "improve_and_finalize_testcases",
            "phase": "1.3",
            "result": {
                "html_response": header + markdown,
                "status": "completed"
            }
        })
        
        return state
    
    async def _generate_current_ac_node_streaming(self, state: AgentState) -> AgentState:
        """Phase 2 Node 1: Generate current AC with streaming"""
        await self._emit_status("node_start", {
            "node": "generate_current_ac",
            "phase": "2.1", 
            "result": {
                "status": "start",
                "html_response": "### 📋 Starting to generate current acceptance criteria"
            }
        })
        
        state = await self._generate_current_ac_node(state)

        markdown = ""
        try:
            markdown = convert_acceptance_criteria_to_markdown(state.get("current_ac", []))
        except Exception as e:
            logger.error(f"Error converting current acceptance criteria to markdown: {str(e)}")

        header = "### 📜 Current acceptance criteria \n"
        await self._emit_status("node_complete", {
            "node": "generate_current_ac",
            "phase": "2.1",
            "result": {
                "html_response": header + markdown,
                "status": "completed"
            }
        })
        
        return state
    
    async def _generate_missing_ac_node_streaming(self, state: AgentState) -> AgentState:
        """Phase 2 Node 2: Generate missing AC with streaming"""
        await self._emit_status("node_start", {
            "node": "generate_missing_ac",
            "phase": "2.2",
            "result": {
                "status": "start",
                "html_response": "### 📋 Starting to generate missing acceptance criteria"
            }
        })
        
        state = await self._generate_missing_ac_node(state)

        markdown = ""
        try:
            markdown = convert_acceptance_criteria_to_markdown(state.get("generated_missing_ac", []))
        except Exception as e:
            logger.error(f"Error converting generated missing acceptance criteria to markdown: {str(e)}")

        header = "### 📜 Generated missing acceptance criteria \n"
        await self._emit_status("node_complete", {
            "node": "generate_missing_ac", 
            "phase": "2.2",
            "result": {
                "html_response": header + markdown,
                "status": "completed"
            }
        })
        
        return state
    
    async def _improve_and_finalize_ac_node_streaming(self, state: AgentState) -> AgentState:
        """Phase 2 Node 3: Improve and finalize AC with streaming"""
        await self._emit_status("node_start", {
            "node": "improve_and_finalize_ac",
            "phase": "2.3",
            "result": {
                "status": "start",
                "html_response": "### 📏 Starting to improve and finalize acceptance criteria"
            }
        })
        
        state = await self._improve_and_finalize_ac_node(state)

        markdown = ""
        try:
            markdown = convert_acceptance_criteria_to_markdown(state.get("final_ac", []))
        except Exception as e:
            logger.error(f"Error converting final acceptance criteria to markdown: {str(e)}")

        header = "### 📜 Final acceptance criteria \n"
        await self._emit_status("node_complete", {
            "node": "improve_and_finalize_ac",
            "phase": "2.3",
            "result": {
                "html_response": markdown,
                "status": "completed"
            }
        })
        
        return state
    
    async def _generate_final_response_streaming(self, state: AgentState) -> AgentState:
        """Phase 3: Generate final response with streaming"""
        await self._emit_status("node_start", {
            "node": "generate_final_response",
            "phase": "3",
            "result": {
                "status": "start",
                "html_response": "### 📋 Starting to analyze code coverage and generate final response"
            }
        })
        
        state = await self._generate_final_response(state)

        markdown = ""
        try:
            logger.info(f"Final analysis result: {state.get('final_analysis_result', {})}")
            markdown = convert_final_result_to_markdown(state.get("final_analysis_result", {}))
        except Exception as e:
            logger.error(f"Error converting final result to markdown: {str(e)}")
        
        header = "### 📋 Final response \n"
        await self._emit_status("node_complete", {
            "node": "generate_final_response",
            "phase": "final",
            "result": {
                "html_response": header + markdown,
                "status": "completed"
            }
        })
        
        return state
    
    async def _format_output_node_streaming(self, state: AgentState) -> AgentState:
        """Final Phase: Format output with streaming"""
        # await self._emit_status("node_start", {
        #     "node": "format_output",
        #     "phase": "final",
        #     "description": "Formatting final output"
        # })
        
        # state = self._format_output_node(state)
        
        # await self._emit_status("node_complete", {
        #     "node": "format_output", 
        #     "phase": "final",
        #     "result": {
        #         "status": "completed"
        #     }
        # })
        
        return state
    
    def _build_streaming_graph(self):
        """Build LangGraph with streaming node methods"""
        
        graph = StateGraph(AgentState)

        # Add streaming nodes
        graph.add_node("extract_existing_testcases", self._extract_existing_testcases_node_streaming)
        graph.add_node("generate_missing_testcases", self._generate_missing_testcases_node_streaming)
        graph.add_node("generate_api_docs", self._generate_api_docs_node_streaming)
        graph.add_node("improve_and_finalize_testcases", self._improve_and_finalize_testcases_node_streaming)
        graph.add_node("generate_current_ac", self._generate_current_ac_node_streaming)
        # graph.add_node("generate_missing_ac", self._generate_missing_ac_node_streaming)
        graph.add_node("improve_and_finalize_ac", self._improve_and_finalize_ac_node_streaming)
        graph.add_node("generate_final_response", self._generate_final_response_streaming)
        # graph.add_node("format_output", self._format_output_node_streaming)

        # Same edges as original
        graph.add_edge("extract_existing_testcases", "generate_missing_testcases")
        graph.add_edge("generate_missing_testcases", "generate_api_docs")
        graph.add_edge("generate_api_docs", "improve_and_finalize_testcases")
        graph.add_edge("improve_and_finalize_testcases", "generate_current_ac")
        graph.add_edge("generate_current_ac", "improve_and_finalize_ac")
        # graph.add_edge("generate_current_ac", "generate_missing_ac")
        # graph.add_edge("generate_missing_ac", "improve_and_finalize_ac")
        graph.add_edge("improve_and_finalize_ac", "generate_final_response")
        # graph.add_edge("generate_final_response", "format_output")
        # graph.add_edge("format_output", END)
        graph.add_edge("generate_final_response", END)

        graph.set_entry_point("extract_existing_testcases")
        self.streaming_graph = graph.compile()
        
        logger.info("🔄 Streaming LangGraph workflow compiled")
    
    async def run_streaming(
        self,
        *,
        endpoint: str,
        requirements_txt: str,
        user_text: str,
        code_commit: str = "",
        changed_methods: List[Dict[str, str]] = []
    ) -> AsyncGenerator[str, None]:
        """Run analysis with streaming updates"""
        
        # Build streaming graph
        self._build_streaming_graph()
        
        try:
            # Initialize state (same as original)
            initial_state = await self._prepare_initial_state(
                endpoint, requirements_txt, user_text, code_commit, changed_methods
            )
            
            # Run streaming graph in background
            async def run_graph():
                try:
                    final_state = await self.streaming_graph.ainvoke(initial_state)
                    logger.info(f"Final state: {final_state}")
                    json_result = final_state.get('final_analysis_result', {})
                    
                    # Emit completion event\
                    await self._emit_status("analysis_complete", {
                        "status": "success",
                        "html_response": "### Analysis completed successfully",
                        "result": json_result
                    })
                    
                except Exception as e:
                    await self._emit_status("analysis_error", {
                        "error": str(e),
                        "status": "error",
                        "html_response": "### Error: " + str(e)
                    })
            
            # Start analysis task
            analysis_task = asyncio.create_task(run_graph())
            
            # Yield events from stream
            async for event in self._stream_generator():
                yield event
                
            # Wait for analysis to complete
            await analysis_task
            
        except Exception as e:
            await self._emit_status("analysis_error", {
                "error": str(e),
                "status": "error"
            })
            yield f"data: {json.dumps({'type': 'error', 'data': {'error': str(e)}})}\n\n"
    
    async def _prepare_initial_state(
        self, 
        endpoint: str,
        requirements_txt: str, 
        user_text: str,
        code_commit: str,
        changed_methods: List[Dict[str, str]]
    ) -> AgentState:
        """Prepare initial state for analysis"""
        
        # Same logic as original run method for context retrieval
        initial_context = ""
        endpoint_str = ""
        try:
            neo4j_conn = get_neo4j_connection()
            endpointNodes = []
            logger.info(f"changed_methods: {changed_methods}")
            endpoint_set = []
            for symbol in changed_methods:
                endpointNode = neo4j_conn.find_endpoint_node(symbol["class"], symbol["method"], self.project_id)
                for node in endpointNode:
                    if (node.get("element_id") not in endpoint_set):
                        endpoint_set.append(node.get("element_id"))
                        endpointNodes.append(node)

            logger.info(f"endpointNodes: {endpointNodes}")

            relatedNodes = []
            for endpoint_node in endpointNodes:
                try:
                    class_name = endpoint_node.get("class_name")
                    method_name = endpoint_node.get("method_name")
                    relatedNode = neo4j_conn.find_related_nodes(class_name, method_name, self.project_id)
                    logger.info(f"relatedNode: {relatedNode}")
                    relatedNodes.extend(relatedNode)
                except Exception as e:
                    logger.error(f"Error finding related nodes: {e}")
            
            logger.info(f"relatedNodes: {relatedNodes}")
            configurationNode = neo4j_conn.find_configuration_node(self.project_id)
            if len(configurationNode) > 0:
                logger.info(f"configurationNode: {configurationNode[0]}")
                relatedNodes.extend(configurationNode)
                
            # Deduplicate
            seen_related = set()
            unique_relatedNodes = []
            for node in relatedNodes:
                identifier = node.get("id") or hash(node.get("content", ""))
                if identifier not in seen_related:
                    seen_related.add(identifier)
                    unique_relatedNodes.append(node)

            initial_context = "\n\n".join([node.get("content") for node in unique_relatedNodes])
            endpoint_str = "\n\n".join([node.get("endpoint") for node in endpointNodes])
        except Exception as e:
            logger.error(f"Error preparing initial state: {str(e)}")

        return {
            "question": f"Analyze the REST endpoint '{endpoint}' according to the requirements and test cases.",
            "context": initial_context,
            "endpoint": endpoint_str,
            "requirements": requirements_txt,
            "user_text": user_text,
            "code_commit": code_commit,
            "history": [],
            "retrieved_symbols": [],
            "seen_context": [],
            "final_response": None,
            "html_response": None,
            "iteration_count": 0,
            "last_tool_call_symbols": [],
            "new_retrieved_symbols": [],
            "node_call_count": {},
            "current_phase": "phase1_extract_testcases",
            "phase_complete": {
                "phase1_extract_testcases": False,
                "phase1_generate_missing_testcases": False,
                "phase1_improve_testcases": False,
                "phase2_generate_current_ac": False,
                "phase2_generate_missing_ac": False,
                "phase2_improve_ac": False,
                "phase3_additional_coverage": False,
                "format_output": False
            },
            "existing_testcases": [],
            "generated_missing_testcases": [],
            "final_testcases": [],
            "current_ac": [],
            "generated_missing_ac": [],
            "final_ac": [],
            "additional_coverage": {},
            "final_analysis_result": {},
            "needs_more_context": False,
            "api_docs": {}
        }