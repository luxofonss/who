from __future__ import annotations

import json
import asyncio
import re
from typing import Dict, List, Optional, Any, TypedDict, Tuple
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

        # Phase 1: Test Cases Analysis (3 nodes)
        graph.add_node("extract_existing_testcases", self._extract_existing_testcases_node)
        graph.add_node("generate_missing_testcases", self._generate_missing_testcases_node)
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
        graph.add_edge("generate_missing_testcases", "improve_and_finalize_testcases")
        
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
        return read_file("api_docs_example.md")

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

    def _read_testcase_guide_item_csv(self) -> str:
        """Read testcase guide item file"""
        return read_file("testcase_guide_item_csv.md")

    def _extract_existing_testcases_node(self, state: AgentState) -> AgentState:
        """Phase 1 Node 1: Extract existing test cases from requirements"""
        node_name = "extract_existing_testcases"
        state["node_call_count"][node_name] = state["node_call_count"].get(node_name, 0) + 1
        state["current_phase"] = "phase1_extract_testcases"
        
        logger.info(" 🧪 Phase 1.1: Extracting existing test cases from requirements")
        
        prompt = f"""You are Expert Quality Engineer.

Your task is to extract existing test cases from the requirements document.
Test cases are in comment section in requirements jira issue with format start with "TestCase", "testcase", "testcases".
Do not assume or generate testcase from business logic, only extract the ones that are explicitly mentioned in the comments section in jira requirements.

<requirements>
{state['requirements']}
</requirements>

Analyze the requirements and extract any existing test cases that are explicitly mentioned or implied.

Provide a JSON response:
{{
    "existing_test_cases": [
        {{
            "test_case": "exact test case text from requirements. the testcase are in "comment" section in jira issue and start with "testcases" or "testcase",
            "test_type": "positive/negative/edge",
            "coverage_area": "what aspect it tests",
            "priority": "high/medium/low"
        }}
    ]
}}

If no existing test cases are found, return an empty array.
Response in Vietnamese. Do not translate English keyword. for example fields in request body or params, etc
"""
        
        try:
            response = self.langchain_llm._call(prompt)
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

    def _generate_missing_testcases_node(self, state: AgentState) -> AgentState:
        """Phase 1 Node 2: Generate missing test cases based on existing test cases + requirements + software_testing_guide"""
        node_name = "generate_missing_testcases"
        state["node_call_count"][node_name] = state["node_call_count"].get(node_name, 0) + 1
        state["current_phase"] = "phase1_generate_missing_testcases"
        
        logger.info(" 🔬 Phase 1.2: Generating missing test cases")
        
        software_testing_guide = self._read_software_testing_guide()
        
        prompt = f"""You are Expert Quality Engineer.

Your task is to generate missing test cases based on existing_test_cases, requirements, software_testing_guide bellow.

<requirements>
{state['requirements']}
</requirements>

<existing_test_cases>
{json.dumps(state.get('existing_testcases', []), indent=2)}
</existing_test_cases>

<software_testing_guide>
{software_testing_guide}
</software_testing_guide>

Analyze the existing_test_cases and requirements to identify what test cases are missing. Generate additional test cases to fill the gaps.

Focus on:
- Edge cases not covered by existing_test_cases
- Error scenarios and exception handling
- Boundary conditions
- Integration scenarios
- Performance considerations
- Security aspects

Provide a JSON response:
{{
    "missing_test_cases": [
        {{
            "test_case": "detailed test case description using terminology from requirements. test_case name follow instruction in software_testing_guide. ",
            "test_type": "positive/negative/edge/performance/security",
            "category": "functional/integration/unit/performance",
            "priority": "high/medium/low",
            "rationale": "why this test case is important and what gap it fills",
        }}
    ]
}}
Response in Vietnamese.  Do not translate English keyword. for example fields in request body or params, etc
"""
        
        try:
            response = self.langchain_llm._call(prompt)
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

    def _improve_and_finalize_testcases_node(self, state: AgentState) -> AgentState:
        """Phase 1 Node 3: Improve additional test cases and return final test cases with type field"""
        node_name = "improve_and_finalize_testcases"
        state["node_call_count"][node_name] = state["node_call_count"].get(node_name, 0) + 1
        state["current_phase"] = "phase1_improve_testcases"
        
        logger.info(" 🔧 Phase 1.3: Improving and finalizing test cases")
        
        prompt = f"""You are Expert Quality Engineer.
Your task is to improve the additional test cases from the previous step and create a final comprehensive test cases list.

<current_context> {state['context']} </current_context>

 {state['requirements']}

<code_diff_commit> {state['code_commit']} </code_diff_commit>

<existing_test_cases> {json.dumps(state.get('existing_testcases', []), indent=2)} </existing_test_cases>

<generated_missing_test_cases> {json.dumps(state.get('generated_missing_testcases', []), indent=2)} </generated_missing_test_cases>

<software_testing_guide>
{self._read_software_testing_guide()}
</software_testing_guide>

### Instructions for Improving Test Cases
1. **Improve the generated_missing_test_cases**:
   - Rewrite each test case description in **natural, clear, and professional Vietnamese**, focusing on functionality, behavior, or user scenarios.
   - Retain technical terms like "keyWord", "HTTP 200", "HTTP 400", etc., as they are, without translating them into Vietnamese.
   - Make test cases more specific and implementable based on the <current_context> and <code_diff_commit>, without including code snippets, class names, enum names, or implementation-specific details in descriptions.
   - Identify and add new test cases if new scenarios are found in the <requirements> or <code_diff_commit> that are not covered by existing or generated test cases.
   - Follow the <software_testing_guide> for clear, concise, and independent test case design, ensuring each test case focuses on a single behavior or scenario.
   - Remove the "rationale" field from improved test cases, as it is not required in the final output.

2. **Combine Test Cases**:
   - Create a final test cases list that includes:
     - All <existing_test_cases> (do not modify these).
     - Improved versions of <generated_missing_test_cases>.
     - Any new test cases identified from the <requirements> or <code_diff_commit>.
   - Ensure test case names and descriptions:
     - Are written in **natural Vietnamese**, avoiding English phrases unless they are technical terms (e.g., "keyWord", "HTTP 200").
     - Focus on functionality or behavior, independent of code structure.
     - Are unique, clear, and avoid redundancy (e.g., avoid repeating similar test cases for case sensitivity unless necessary).
   - Assign appropriate values for "test_type" (positive/negative/edge/performance/security), "category" (functional/integration/unit/performance), "priority" (high/medium/low), and "type" (existing/new).

3. **DO NOT**:
   - Delete test cases just because the code does not implement those features.
   - Modify expected behavior to align with current code implementation.
   - Overlook requirements when they differ from the code.
   - Assume the existing code logic is correct.
   - Translate technical terms like "keyWord", "HTTP 200", or "HTTP 400" into Vietnamese.

4. **Output Format**:
```json
{{
    "final_test_cases": [
        {{
            "test_case": "Detailed description of the test case focusing on functionality or behavior, based on software_testing_guide, without referencing code, class names, or enums.",
            "test_type": "positive/negative/edge/performance/security",
            "category": "functional/integration/unit/performance",
            "priority": "high/medium/low",
            "type": "existing/new"
        }}
    ]
}}
Response in Vietnamese. Do not translate English technical terms and params, keywords.
"""
        
        try:
            response = self.langchain_llm._call(prompt)
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

    def _generate_current_ac_node(self, state: AgentState) -> AgentState:
        """Phase 2 Node 1: Generate current AC (acceptance criteria) based on requirements"""
        node_name = "generate_current_ac"
        state["node_call_count"][node_name] = state["node_call_count"].get(node_name, 0) + 1
        state["current_phase"] = "phase2_generate_current_ac"
        
        logger.info(" 📋 Phase 2.1: Generating current acceptance criteria from requirements")
        response_ac_guide_item = self._read_response_ac_guide_item()
        prompt = f"""You are Expert Quality Engineer and Business Analyst.

Your task is to generate current acceptance criteria (AC) based on the requirements document.

<requirements>
{state['requirements']}
</requirements>

Extract and generate acceptance criteria that are explicitly mentioned or implied in the requirements.

Acceptance criteria should be:
- Specific and measurable
- Testable
- Clear and unambiguous
- Focused on behavior and outcomes

Provide a JSON response:
{{
    "current_ac": list of AC in format {response_ac_guide_item}
}}

Response in Vietnamese  Do not translate English keyword. for example fields in request body or params, etc

"""
        
        try:
            response = self.langchain_llm._call(prompt)
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

    def _generate_missing_ac_node(self, state: AgentState) -> AgentState:
        """Phase 2 Node 2: Analyze current requirements and current AC, generate missing AC"""
        node_name = "generate_missing_ac"
        state["node_call_count"][node_name] = state["node_call_count"].get(node_name, 0) + 1
        state["current_phase"] = "phase2_generate_missing_ac"
        
        logger.info(" 🔍 Phase 2.2: Analyzing and generating missing acceptance criteria")
        response_ac_guide_item = self._read_response_ac_guide_item()
        prompt = f"""You are Expert Quality Engineer.

Your task is to analyze current requirements and existing acceptance criteria to identify missing acceptance criteria.

<requirements>
{state['requirements']}
</requirements>

<current_ac>
{json.dumps(state.get('current_ac', []), indent=2)}
</current_ac>

Analyze the requirements and current AC to identify gaps. Generate missing acceptance criteria that should be added.

Focus on:
- Functional requirements not covered by current AC
- Non-functional requirements (performance, security, usability)
- Edge cases and error scenarios
- Integration and system requirements
- Data validation and business rules
- User experience requirements

Provide a JSON response:
{{
    "missing_ac": list of AC in format {response_ac_guide_item}
}}

Response in Vietnamese  Do not translate English keyword. for example fields in request body or params, etc

"""
        
        try:
            response = self.langchain_llm._call(prompt)
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

    def _improve_and_finalize_ac_node(self, state: AgentState) -> AgentState:
        """Phase 2 Node 3: Improve additional AC and create final AC list"""
        node_name = "improve_and_finalize_ac"
        state["node_call_count"][node_name] = state["node_call_count"].get(node_name, 0) + 1
        state["current_phase"] = "phase2_improve_ac"
        
        logger.info(" 🔧 Phase 2.3: Improving and finalizing acceptance criteria")
        
        response_ac_guide_item = self._read_response_ac_guide_item()
        
        prompt = f"""You are Expert Quality Engineer and Business Analyst

Your task is to improve the additional acceptance criteria and create a final comprehensive AC list.

<current_context>
{state['context']}
</current_context>

<code_diff_commit>
{state['code_commit']}
</code_diff_commit>

<current_ac>
{json.dumps(state.get('current_ac', []), indent=2)}
</current_ac>

<generated_missing_ac>
{json.dumps(state.get('generated_missing_ac', []), indent=2)}
</generated_missing_ac>

<final_testcases>
{json.dumps(state.get('final_testcases', []), indent=2)}
</final_testcases>

Improve the additional acceptance criteria by:
1. Using the current code context to make them more specific and implementable
2. Considering code differences/commits for any specific changes
3. Following the software AC guide best practices
4. Ensuring alignment with the final test cases from Phase 1
5. Adding technical implementation details based on actual code structure

Create a final AC list that combines existing and improved additional acceptance criteria.

Provide a JSON response:
{{
    "final_ac": response in format {response_ac_guide_item}. AC MUST NOT depend on code, it must not contain any code
}}

Response in Vietnamese  Do not translate English keyword. for example fields in request body or params, etc

"""
        
        try:
            response = self.langchain_llm._call(prompt)
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

    def _generate_final_response(self, state: AgentState) -> AgentState:
        """Phase 3: Analyze code coverage of test cases and AC from previous phases"""
        node_name = "generate_final_response"
        state["node_call_count"][node_name] = state["node_call_count"].get(node_name, 0) + 1
        state["current_phase"] = "phase3_additional_coverage"
        
        logger.info(" 📊 Phase 3: Analyzing code coverage of test cases and AC")
        
        response_ac_guide = self._read_response_ac_guide()
        
        prompt = f"""You are Expert Quality Engineer.

Your task is to analyze whether the test cases and acceptance criteria from previous phases are covered by the current code implementation.

<current_context>
{state['context']}
</current_context>

<requirements>
{state['requirements']}
</requirements>

<code_diff_commit>
{state['code_commit']}
</code_diff_commit>

<final_testcases>
{json.dumps(state.get('final_testcases', []), indent=2)}
</final_testcases>

<final_ac>
{json.dumps(state.get('final_ac', []), indent=2)}
</final_ac>

<software_testing_guide>
{self._read_software_testing_guide()}
</software_testing_guide>

Analyze the current code implementation to determine:
1. Whether each test case can be executed against the current code
2. Whether each acceptance criteria is supported by the current code implementation
3. What code changes or additions are needed to support the test cases and AC
4. Which test cases and AC are fully covered, partially covered, or not covered at all

For each test case, check:
- Does the code have the required methods/endpoints?
- Does the code meet exactly the requirements
- Are the input parameters supported?
- Are the expected behaviors implemented?
- Are error scenarios handled?
- Analyze test_cases first, then analyze ac_anlysis

Provide a JSON response in format of {{
        "test_cases": base on final_testcases and code context, commit diff, response in format:  {{
            "test_case": "exactly test case from final_testcases",
            "test_type": "positive/negative/edge/performance/security",
            "category": "functional/integration/unit/performance", 
            "priority": "high/medium/low",
            "type": "existing/new",
            "code_coverage_score": "0%: No logic for test case. 1 to 49%: Some logic, lacks key checks. 50 to 79%: Handles most requirements, misses critical checks. 80 to 99%: Meets most requirements, minor gaps. 100%: Fully meets all requirements with checks and error handling. ",
            "explain_coverage": "explain how the test case is covered by the code",
        }},
        "ac_analysis": analysis from final_ac. Only analyze "Code Location",	"Assessment", "Priority", testcase and other information must be exactly same as field "testcase" inin final_testcases. AC name and other information must be exactly same as final_ac. Return in format of: {response_ac_guide}.
        "testcase_csv": analyze exactly all testcases from above "final_testcases". use "current_context" and "requirements". response all data in format list of {self._read_testcase_guide_item_json()}.  Please provide the complete response without shortening or using ellipses (...).
    }}  

DO NOT assume , always return full response.  Please provide the complete response without shortening or using ellipses (...).
IMPORTANT: Response MUST be in Vietnamese. if existed testcases and ac are in English, you can translate it to Vietnamese with full context and information.  Do not translate English keyword. for example fields in request body or params, etc"""
        
        try:
            # Call LLM and get response
            response = self.langchain_llm._call(prompt)
            response_clean = self._parse_json_response(response)
            
            try:
                # Parse the cleaned response
                result = json.loads(response_clean)
                
                # Validate required fields
                if "test_cases" not in result or "ac_analysis" not in result:
                    raise ValueError("Missing required fields in LLM response")
                
                # Process test cases coverage
                test_cases = result.get("test_cases", [])
                if isinstance(test_cases, dict):
                    # Convert single test case to list
                    test_cases = [test_cases]
                
                # Process AC analysis
                ac_analysis = result.get("ac_analysis", {})
                if isinstance(ac_analysis, list):
                    for ac in ac_analysis:
                        code_location = ac.get("code_location")
                        testcase_name = ac.get("testcase_name")
                        is_code_sufficient = ac.get("is_code_sufficient", False)
                        is_testcase_sufficient = ac.get("is_testcase_sufficient", False)
                        if code_location and testcase_name:
                            ac["assessment"] = "Đạt yêu cầu"
                        elif code_location and not testcase_name:
                            if not is_code_sufficient:
                                ac["assessment"] = "Code chưa đủ; Chưa có testcase"
                            else:
                                ac["assessment"] = "Chưa có testcase"
                        elif testcase_name and not code_location:
                            if not is_testcase_sufficient:
                                ac["assessment"] = "Testcase chưa đủ; Chưa có code"
                            else:
                                ac["assessment"] = "Chưa có code"
                elif isinstance(ac_analysis, dict):
                    code_location = ac_analysis.get("code_location")
                    testcase_name = ac_analysis.get("testcase_name")
                    is_code_sufficient = ac_analysis.get("is_code_sufficient", False)
                    is_testcase_sufficient = ac_analysis.get("is_testcase_sufficient", False)
                    if code_location and testcase_name:
                        ac_analysis["assessment"] = "Đạt yêu cầu"
                    elif code_location and not testcase_name:
                        if not is_code_sufficient:
                            ac_analysis["assessment"] = "Code chưa đủ; Chưa có testcase"
                        else:
                            ac_analysis["assessment"] = "Chưa có testcase"
                    elif testcase_name and not code_location:
                        if not is_testcase_sufficient:
                            ac_analysis["assessment"] = "Testcase chưa đủ; Chưa có code"
                        else:
                            ac_analysis["assessment"] = "Chưa có code"

                # Sinh trường overall_assessment
                overall_assessment = "Not Satisfactory"
                ac_list = ac_analysis if isinstance(ac_analysis, list) else [ac_analysis] if isinstance(ac_analysis, dict) else []
                if ac_list:
                    all_satisfactory = all(
                        (ac.get("status") == "Đã định nghĩa" and ac.get("assessment") == "Đạt yêu cầu")
                        or ac.get("status") != "Đã định nghĩa"
                        for ac in ac_list
                    )
                    if all_satisfactory:
                        overall_assessment = "Satisfactory"
                # Gán vào kết quả trả về
                result["overall_assessment"] = overall_assessment

                testcase_csv = result.get("testcase_csv", [])
                
                # Update state with validated data
                state["final_analysis_result"] = {
                    "test_cases_coverage": test_cases,
                    "ac_analysis": ac_analysis,
                    "testcase_csv": testcase_csv,
                }
                
                # Mark phase as complete
                state["phase_complete"]["phase3_additional_coverage"] = True
                
                # Log success with metrics
                coverage_metrics = self._calculate_coverage_metrics(test_cases)
                logger.info(
                    f" ✅ Phase 3 completed: Analyzed {len(test_cases)} test cases with "
                    f"avg coverage score: {coverage_metrics['avg_coverage']:.1f}%, "
                )
                
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

    def _calculate_coverage_metrics(self, test_cases: List[Dict]) -> Dict[str, float]:
        """Calculate coverage metrics from test cases."""
        if not test_cases:
            return {"avg_coverage": 0.0, "min_coverage": 0.0, "max_coverage": 0.0}
            
        coverage_scores = []
        for tc in test_cases:
            try:
                score = int(tc.get("code_coverage_score", "0").replace("%", ""))
                coverage_scores.append(score)
            except (ValueError, TypeError):
                continue
                
        if not coverage_scores:
            return {"avg_coverage": 0.0, "min_coverage": 0.0, "max_coverage": 0.0}
            
        return {
            "avg_coverage": sum(coverage_scores) / len(coverage_scores),
            "min_coverage": min(coverage_scores),
            "max_coverage": max(coverage_scores)
        }

    def _format_output_node(self, state: AgentState) -> AgentState:
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
        
        # Format comprehensive document
        # comprehensive_doc = self._build_comprehensive_document(state)
        # analysis_results["comprehensive_document"] = comprehensive_doc
        
        # Convert to JSON string
        json_response = json.dumps(analysis_results, indent=2, ensure_ascii=False)
        
        # Generate HTML response
        html_content = self._generate_html_with_llm(json_response)
        
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

    def _generate_html_with_llm(self, response: str) -> str:
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

"""
            
            logger.info(f" Sending prompt to LLM ({len(prompt)} characters)")
            
            # Use the LLM to generate HTML
            html_response = self.langchain_llm._call(prompt)
            
            logger.info(f" Received LLM response ({len(html_response)} characters)")
            
            # Clean up the response - extract HTML if it's wrapped in markdown
            # html_content = self._extract_html_from_response(html_response)
            
            # logger.info(f" Generated HTML response ({len(html_content)} characters)")
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
        
        # if not endpoint:
        # symbols = [method["class"] + "." + method["method"] for method in changed_methods]
        # endpoints = await self.retriever.retrieve_endpoints(symbols)
        # logger.info(f"Endpoints: {endpoints}")

        # if endpoint and not endpoints:
        #     endpoints.append(endpoint)
        
        logger.info(f" Starting LangGraph AnalyzerChain for endpoint: {endpoint}")
    
        try:
            # TODO: update to retrieve docs of all endpoints
            # TODO: use langchain memory to store chat memory so that user can iterate with ai
            # docs = []
            # endpoint_strs = []
            # for endpt in endpoints:
            #     doc = await self.retriever.retrieve(str(endpt), 1 , hyde=False)
            #     logger.info(f"endpoint {str(endpt)} docs {len(doc)}")
            #     contents = [doc.page_content for doc in doc]
            #     docs.extend(doc)
            #     endpoint_strs.append(str(endpt))
            
            
            
            # endpoint_str = str(endpoint_strs)
            # logger.info(f"endpoint_str: {endpoint_str}")
            
            # logger.info(f"len of docs before deduplicate: {len(docs)}")
            # docs = self.retriever._deduplicate_documents(docs)
            # logger.info(f"len of docs after deduplicate: {len(docs)}")
            # initial_context = "\n\n".join(doc.page_content for doc in docs)
            # initial_chunk_ids = [doc.metadata.get("id", str(hash(doc.page_content))) for doc in docs]

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
                "needs_more_context": False
            }
            
            logger.info(" Step 3: Starting LangGraph analysis workflow...")
            final_state = await asyncio.to_thread(self.graph.invoke, initial_state)
                    
            logger.info(" Step 4: Parsing and structuring final response...")

            result = {
                "markdown_response": final_state.get("html_response", ""),
                "json_response": final_state.get("final_analysis_result", ""),
            }
            # final_analysis_result = final_state.get("final_analysis_result", "")
            logger.info(f" Analysis complete - returning final response")
            return result
            
        except AnalysisError:
            raise
        except Exception as e:
            logger.error(f" LangGraph analysis failed: {str(e)}")
            return {}
            # try:
            #     return await self._fallback_analysis(
            #         endpoint=endpoint,
            #         requirements_txt=requirements_txt,
            #         user_text=user_text,
            #         initial_context=initial_context if 'initial_context' in locals() else ""
            #     )
            # except Exception as fallback_error:
            #     logger.error(f" Fallback analysis also failed: {str(fallback_error)}")
            #     raise AnalysisError(f"Both LangGraph and fallback analysis failed. LangGraph error: {str(e)}, Fallback error: {str(fallback_error)}")

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