from __future__ import annotations

import json
import asyncio
import re
from typing import Dict, List, Optional, Any, TypedDict, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from loguru import logger
from langchain_core.tools import Tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor

from adapters.gemini import Gemini, LangChainGemini
from services.retriever import LangChainRetriever
from services.prompt_builder import PromptBuilder


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

    def __init__(self, project_id: str):
        if not project_id or not isinstance(project_id, str):
            raise ValueError("project_id must be a non-empty string")
        
        self.project_id = project_id
        self.retriever = LangChainRetriever(project_id)
        self.llm = Gemini(temperature=0)  # For fallback analysis
        self.langchain_llm = LangChainGemini(temperature=0)  # For LangGraph agent
        
        # Create tool and graph
        self._setup_langgraph()
        
        logger.info(f" AnalyzerChain initialized with LangGraph for project: {project_id}")

    def _setup_langgraph(self):
        logger.info(" Setting up LangGraph components...")
        
        # Create the tool
        self.get_context_tool = Tool(
            name="get_project_code_context",
            func=self._find_symbol_context,
            description=(
                "Return code content related to any symbol (class/method/DTO/service/etc) from the project. "
                "Use it when you need more details about classes, methods, or components mentioned in the code. "
                "Pass the exact name of the class, interface, method, or component you want to understand better."
            )
        )
        logger.info(" Created get_project_code_context tool")
        
        # Create tool executor
        self.tool_executor = ToolExecutor([self.get_context_tool])
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
        graph.add_node("generate_additional_coverage", self._generate_additional_coverage_node)
        
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
        graph.add_edge("improve_and_finalize_ac", "generate_additional_coverage")
        
        # Phase 3 to final output
        graph.add_edge("generate_additional_coverage", "format_output")
        graph.add_edge("format_output", END)

        # Set entry point to Phase 1 Node 1
        graph.set_entry_point("extract_existing_testcases")

        # Compile the graph
        self.graph = graph.compile()
        logger.info(" 3-Phase LangGraph workflow compiled with 7 nodes")

    def _find_symbol_context(self, symbol: str, seen_chunks: List[str]) -> Tuple[str, List[str]]:
        logger.info(f" Searching for symbol: '{symbol}'")
        try:
            # Try direct symbol search first
            logger.debug(f" Attempting direct symbol lookup for: {symbol}")
            docs = self.retriever.find_by_symbol_name(symbol)
            
            # If no direct match, try a broader search
            if not docs:
                logger.debug(f" No direct match for '{symbol}', trying semantic search...")
                docs = self.retriever.retrieve_sync(
                    symbol,
                    top=1,
                    hyde=False
                )
            
            if docs:
                new_chunks = []
                new_chunk_ids = []
                for doc in docs:
                    chunk_id = doc.metadata.get("id", str(hash(doc.page_content)))  # Fallback to hash if no chunk_id
                    if chunk_id not in seen_chunks:
                        new_chunks.append(doc.page_content)
                        new_chunk_ids.append(chunk_id)
                    else:
                        # logger.debug(f" Skipped already seen chunk {chunk_id} for symbol '{symbol}'")
                        pass
                
                if new_chunks:
                    result = "\n\n".join(new_chunks)
                    logger.info(f" Found {len(new_chunks)} new documents for '{symbol}' ({len(result)} chars)")
                    logger.debug(f" Context preview for '{symbol}': {result[:150]}...")
                    return f"'{symbol}':\n{result}", new_chunk_ids
                else:
                    # logger.warning(f" No new chunks found for symbol: '{symbol}'")
                    return f"No new code found for symbol: {symbol}.", []
            else:
                logger.warning(f" No code found for symbol: '{symbol}'")
                return f"No code found for symbol: {symbol}. Try a different class or method name.", []
        except Exception as e:
            logger.error(f" Error retrieving context for symbol '{symbol}': {str(e)}")
            return f"Error retrieving code for symbol: {symbol} - {str(e)}", []

    def _read_api_docs_example(self) -> str:
        try:
            with open("api_docs_example.md", "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.warning(f"Could not read api_docs_example.md: {e}")
            return ""

    def _read_software_testing_guide(self) -> str:
        try:
            with open("software_testing_guide.md", "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.warning(f"Could not read software_testing_guide.md: {e}")
            return ""
    def _read_response_ac_guide(self) -> str:
        try:
            with open("response_ac_guide.md", "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.warning(f"Could not read response_ac_guide.md: {e}")
            return ""
    
    def _read_response_(self) -> str:
        try:
            with open("response_ac_guide.md", "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.warning(f"Could not read response_ac_guide.md: {e}")
            return ""



    def _extract_existing_testcases_node(self, state: AgentState) -> AgentState:
        """Phase 1 Node 1: Extract existing test cases from requirements"""
        node_name = "extract_existing_testcases"
        state["node_call_count"][node_name] = state["node_call_count"].get(node_name, 0) + 1
        state["current_phase"] = "phase1_extract_testcases"
        
        logger.info(" 🧪 Phase 1.1: Extracting existing test cases from requirements")
        
        prompt = f"""You are Expert Quality Engineer.

Your task is to extract existing test cases from the requirements document.

<requirements>
{state['requirements']}
</requirements>

Analyze the requirements and extract any existing test cases that are explicitly mentioned or implied.

Provide a JSON response:
{{
    "existing_test_cases": [
        {{
            "test_case": "exact test case text from requirements",
            "test_type": "positive/negative/edge",
            "coverage_area": "what aspect it tests",
            "source": "where in requirements this test case is mentioned",
            "priority": "high/medium/low"
        }}
    ]
}}

If no existing test cases are found, return an empty array.

Response in Vietnamese:"""
        
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

Your task is to generate missing test cases based on existing test cases, requirements, and software testing guide.

<requirements>
{state['requirements']}
</requirements>

<existing_test_cases>
{json.dumps(state.get('existing_testcases', []), indent=2)}
</existing_test_cases>

<software_testing_guide>
{software_testing_guide}
</software_testing_guide>

Analyze the existing test cases and requirements to identify what test cases are missing. Generate additional test cases to fill the gaps.

Focus on:
- Edge cases not covered by existing tests
- Error scenarios and exception handling
- Boundary conditions
- Integration scenarios
- Performance considerations
- Security aspects

Provide a JSON response:
{{
    "missing_test_cases": [
        {{
            "test_case": "detailed test case description using terminology from requirements",
            "test_type": "positive/negative/edge/performance/security",
            "category": "functional/integration/unit/performance",
            "priority": "high/medium/low",
            "rationale": "why this test case is important and what gap it fills",
            "expected_behavior": "what should happen",
            "relates_to_existing": "which existing test case this relates to or 'none'"
        }}
    ]
}}

Response in Vietnamese:"""
        
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
        
        software_testing_guide = self._read_software_testing_guide()
        
        prompt = f"""You are Expert Quality Engineer.

Your task is to improve the additional test cases from the previous step and create a final comprehensive test cases list.

<current_context>
{state['context']}
</current_context>

<code_diff_commit>
{state['code_commit']}
</code_diff_commit>

<existing_test_cases>
{json.dumps(state.get('existing_testcases', []), indent=2)}
</existing_test_cases>

<generated_missing_test_cases>
{json.dumps(state.get('generated_missing_testcases', []), indent=2)}
</generated_missing_test_cases>

<software_testing_guide>
{software_testing_guide}
</software_testing_guide>

Improve the additional test cases by:
1. Using the current code context to make them more specific and implementable
2. Considering code differences/commits for any specific changes that need testing
3. Following the software testing guide best practices
4. Adding implementation details based on actual code structure

Create a final test cases list that combines existing and improved additional test cases.

Provide a JSON response:
{{
    "final_test_cases": [
        {{
            "test_case": "detailed test case description",
            "test_type": "positive/negative/edge/performance/security",
            "category": "functional/integration/unit/performance", 
            "priority": "high/medium/low",
            "type": "existing/new",
            "rationale": "why this test case is important",
            "expected_behavior": "what should happen",
        }}
    ]
}}

Response in Vietnamese:"""
        
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
        
        prompt = f"""You are Expert Quality Engineer.

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
    "current_ac": [
        {{
            "ac_id": "unique identifier for this AC",
            "title": "short title of the acceptance criteria",
            "description": "detailed description of the acceptance criteria",
            "category": "functional/non-functional/security/performance/usability",
            "priority": "high/medium/low",
            "source": "where in requirements this AC is derived from",
            "testable": true/false,
            "measurable_criteria": "specific measurable criteria if applicable"
        }}
    ]
}}

Response in Vietnamese:"""
        
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
    "missing_ac": [
        {{
            "ac_id": "unique identifier for this missing AC",
            "title": "short title of the missing acceptance criteria",
            "description": "detailed description of the missing acceptance criteria",
            "category": "functional/non-functional/security/performance/usability",
            "priority": "high/medium/low",
            "rationale": "why this AC is missing and important",
            "gap_analysis": "what gap this AC fills compared to current AC",
            "related_requirement": "which requirement this AC relates to",
            "testable": true/false,
            "measurable_criteria": "specific measurable criteria if applicable"
        }}
    ],
    "gap_summary": "summary of gaps identified in current AC coverage"
}}

Response in Vietnamese:"""
        
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
        
        software_ac_guide = self._read_response_()
        
        prompt = f"""You are Expert Quality Engineer.

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

<software_ac_guide>
{software_ac_guide}
</software_ac_guide>

Improve the additional acceptance criteria by:
1. Using the current code context to make them more specific and implementable
2. Considering code differences/commits for any specific changes
3. Following the software AC guide best practices
4. Ensuring alignment with the final test cases from Phase 1
5. Adding technical implementation details based on actual code structure

Create a final AC list that combines existing and improved additional acceptance criteria.

Provide a JSON response:
{{
    "final_ac": [
        {{
            "ac_id": "unique identifier for this AC",
            "title": "short title of the acceptance criteria",
            "description": "detailed description of the acceptance criteria",
            "category": "functional/non-functional/security/performance/usability",
            "priority": "high/medium/low",
            "type": "existing/new",
            "implementation_notes": "specific implementation details based on code",
            "testable": true/false,
            "measurable_criteria": "specific measurable criteria",
            "related_testcases": ["list of related test case IDs from Phase 1"],
            "technical_requirements": "technical implementation requirements"
        }}
    ],
    "ac_summary": {{
        "total_ac": "number",
        "existing_ac": "number", 
        "new_ac": "number",
        "coverage_assessment": "overall AC coverage assessment"
    }}
}}

Response in Vietnamese:"""
        
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

    def _generate_additional_coverage_node(self, state: AgentState) -> AgentState:
        """Phase 3: Analyze code coverage of test cases and AC from previous phases"""
        node_name = "generate_additional_coverage"
        state["node_call_count"][node_name] = state["node_call_count"].get(node_name, 0) + 1
        state["current_phase"] = "phase3_additional_coverage"
        
        logger.info(" 📊 Phase 3: Analyzing code coverage of test cases and AC")
        
        response_ac_guide = self._read_response_ac_guide()
        
        prompt = f"""You are Expert Quality Engineer.

Your task is to analyze whether the test cases and acceptance criteria from previous phases are covered by the current code implementation.

<current_context>
{state['context']}
</current_context>

<code_diff_commit>
{state['code_commit']}
</code_diff_commit>

<final_testcases>
{json.dumps(state.get('final_testcases', []), indent=2)}
</final_testcases>

<final_ac>
{json.dumps(state.get('final_ac', []), indent=2)}
</final_ac>

<response_ac_guide>
{response_ac_guide}
</response_ac_guide>

Analyze the current code implementation to determine:
1. Whether each test case can be executed against the current code
2. Whether each acceptance criteria is supported by the current code implementation
3. What code changes or additions are needed to support the test cases and AC
4. Which test cases and AC are fully covered, partially covered, or not covered at all

For each test case, check:
- Does the code have the required methods/endpoints?
- Are the input parameters supported?
- Are the expected behaviors implemented?
- Are error scenarios handled?

Provide a JSON response in format of {{
        "existed_testcase": "existed testcase analyze",
        "additional_testcase": "additional testcase analyze",
        "ac_analysis": <response_ac_guide>
    }}  

Response in Vietnamese:"""
        
        try:
            response = self.langchain_llm._call(prompt)
            response_clean = self._parse_json_response(response)
            
            try:
                result = json.loads(response_clean)
                state["additional_coverage"] = result.get("additional_coverage", {})
                state["phase_complete"]["phase3_additional_coverage"] = True
                
                gap_count = len(state["additional_coverage"].get("coverage_gaps", []))
                logger.info(f" ✅ Phase 3 completed: Generated additional coverage analysis with {gap_count} coverage gaps identified")
                
            except json.JSONDecodeError as e:
                logger.error(f" Failed to parse additional coverage JSON: {e}")
                state["additional_coverage"] = {"error": f"JSON parsing failed: {e}"}
                state["phase_complete"]["phase3_additional_coverage"] = True
                
        except Exception as e:
            logger.error(f" Error in generating additional coverage: {e}")
            state["additional_coverage"] = {"error": f"Analysis failed: {e}"}
            state["phase_complete"]["phase3_additional_coverage"] = True
            
        return state

    def _analyze_coverage_node(self, state: AgentState) -> AgentState:
        """Phase 3: Analyze requirements and test case coverage"""
        node_name = "analyze_coverage"
        state["node_call_count"][node_name] = state["node_call_count"].get(node_name, 0) + 1
        state["current_phase"] = "coverage_analysis"
        
        logger.info(" 📊 Phase 3: Analyzing requirements and test case coverage")

        software_testing_guide = self._read_software_testing_guide()
        
        prompt = f"""You are analyzing how well the code implementation covers the requirements and test cases.

<current_context>
{state['context']}
</current_context>

<requirements>
{state['requirements']}
</requirements>

<current_testcases_analysis>
{json.dumps(state.get('current_testcases_analysis', {}), indent=2)}
</current_testcases_analysis>

<requirements_based_testcases>
{json.dumps(state.get('generated_testcases', []), indent=2)}
</requirements_based_testcases>

<code_based_testcases>
{json.dumps(state.get('generated_code_testcases', []), indent=2)}
</code_based_testcases>

<software_testing_guide>
{software_testing_guide}
</software_testing_guide>

Phase 3 Task: Analyze requirements coverage and test case coverage against the actual implementation.

Provide a JSON response:
{{
    "requirement_coverage": [
        {{
            "requirement": "exact requirement text",
            "coverage_score": "0-100",
            "implementation_status": "fully_implemented/partially_implemented/not_implemented",
            "code_evidence": "specific code that implements this requirement",
            "gaps": ["what's missing in implementation"],
            "explain": "detailed explanation of coverage"
        }}
    ],
    "test_case_coverage": [
        {{
            "test_case": "test case description",
            "coverage_score": "0-100",
            "implementation_support": "how well code supports this test",
            "explain": "explanation of test coverage by implementation"
        }}
    ],
    "overall_coverage_score": "0-100",
    "coverage_summary": "summary of overall coverage analysis"
}}

Response in Vietnamese:"""
        
        try:
            response = self.langchain_llm._call(prompt)
            response_clean = self._parse_json_response(response)
            
            try:
                coverage_analysis = json.loads(response_clean)
                state["coverage_analysis"] = coverage_analysis
                state["phase_complete"]["coverage_analysis"] = True
                logger.info(" ✅ Phase 3 completed: Coverage analysis")
                
            except json.JSONDecodeError as e:
                logger.error(f" Failed to parse coverage analysis JSON: {e}")
                state["coverage_analysis"] = {"error": f"JSON parsing failed: {e}"}
                state["phase_complete"]["coverage_analysis"] = True
                
        except Exception as e:
            logger.error(f" Error in coverage analysis: {e}")
            state["coverage_analysis"] = {"error": f"Analysis failed: {e}"}
            state["phase_complete"]["coverage_analysis"] = True
            
        return state



    def _generate_missing_testcase_base_on_code_node(self, state: AgentState) -> AgentState:
        """Phase 2: Generate missing test cases based on code logic"""
        node_name = "generate_missing_testcase_base_on_code"
        state["node_call_count"][node_name] = state["node_call_count"].get(node_name, 0) + 1
        state["current_phase"] = "generate_code_testcases"
        
        logger.info(" 🔬 Phase 2: Generating test cases based on code logic")
        
        software_testing_guide = self._read_software_testing_guide()
        
        prompt = f"""You are generating test cases specifically based on code logic and implementation details.

<current_context>
{state['context']}
</current_context>

<requirements>
{state['requirements']}
</requirements>

<current_testcases_analysis>
{json.dumps(state.get('current_testcases_analysis', {}), indent=2)}
</current_testcases_analysis>

<existing_generated_testcases>
{json.dumps(state.get('generated_testcases', []), indent=2)}
</existing_generated_testcases>

<software_testing_guide>
{software_testing_guide}
</software_testing_guide>

Phase 2 Task: Generate additional test cases that focus specifically on code logic, edge cases, and implementation-specific scenarios.

Focus on:
- Code paths and branching logic
- Exception handling and error conditions
- Input validation and boundary conditions
- Null/empty value handling
- Data type conversions and validations
- Security vulnerabilities in the code
- Performance edge cases
- Integration points and dependencies
- Database transaction scenarios
- Concurrency and threading issues

Analyze the actual code implementation to identify:
- Conditional statements (if/else, switch)
- Loop conditions and iterations
- Try-catch blocks and exception scenarios
- Method parameters and return types
- Database queries and potential failures
- External service calls and timeouts
- Authentication and authorization checks

Provide a JSON response:
{{
    "code_based_test_cases": [
        {{
            "test_case": "detailed test case based on specific code logic or implementation detail",
            "test_type": "positive/negative/edge/boundary/exception/security/performance",
            "code_trigger": "specific code path, method, or condition that triggers this scenario",
            "implementation_detail": "specific implementation aspect being tested",
            "priority": "high/medium/low",
            "rationale": "why this test case is important based on the code",
            "expected_behavior": "what should happen based on code logic",
            "test_data": "specific test data or conditions needed",
            "coverage_score": "0-100",
            "explain": "how current implementation handles this scenario"
        }}
    ],
    "code_complexity_areas": ["areas of code that are complex and need thorough testing"],
    "potential_bugs": ["potential issues identified in the code that need test coverage"],
    "missing_validations": ["input validations or checks that should be tested"]
}}

Response in Vietnamese:"""
        
        try:
            response = self.langchain_llm._call(prompt)
            response_clean = self._parse_json_response(response)
            
            try:
                code_testcases_result = json.loads(response_clean)
                state["generated_code_testcases"] = code_testcases_result.get("code_based_test_cases", [])
                state["phase_complete"]["generate_code_testcases"] = True
                logger.info(f" ✅ Phase 2 completed: Generated {len(state['generated_code_testcases'])} code-based test cases")
                
            except json.JSONDecodeError as e:
                logger.error(f" Failed to parse code test cases JSON: {e}")
                state["generated_code_testcases"] = []
                state["phase_complete"]["generate_code_testcases"] = True
                
        except Exception as e:
            logger.error(f" Error in generating code test cases: {e}")
            state["generated_code_testcases"] = []
            state["phase_complete"]["generate_code_testcases"] = True
            
        return state

    def _generate_improvements_node(self, state: AgentState) -> AgentState:
        """Phase 4: Generate improvement suggestions and ACs"""
        node_name = "generate_improvements"
        state["node_call_count"][node_name] = state["node_call_count"].get(node_name, 0) + 1
        state["current_phase"] = "generate_improvements"
        
        logger.info(" 🔧 Phase 4: Generating improvements and ACs")
        
        response_ac_guide = self._read_response_ac_guide()
        
        prompt = f"""You are generating improvement suggestions and acceptance criteria based on the comprehensive analysis.

<current_context>
{state['context']}
</current_context>

<requirements>
{state['requirements']}
</requirements>

<coverage_analysis>
{json.dumps(state.get('coverage_analysis', {}), indent=2)}
</coverage_analysis>

<response_ac_guide>
{response_ac_guide}
</response_ac_guide>

Phase 4 Task: Generate improvement suggestions and acceptance criteria.

Provide a JSON response:
{{
    "improvements": [
        {{
            "type": "security/performance/maintainability/functionality/testing",
            "priority": "high/medium/low",
            "current_issue": "what's wrong or missing",
            "reason": "why this improvement is needed",
            "solution": "recommended fix or enhancement",
            "impact": "expected impact of the improvement"
        }}
    ],
    "response_ac": "acceptance criteria based on requirements and analysis following the AC guide format",
    "curl_command": "curl command to test the endpoint",
    "implementation_recommendations": ["high-level recommendations for implementation"]
}}

Response in Vietnamese:"""
        
        try:
            response = self.langchain_llm._call(prompt)
            response_clean = self._parse_json_response(response)
            
            try:
                improvements_result = json.loads(response_clean)
                state["improvements_analysis"] = improvements_result
                state["phase_complete"]["generate_improvements"] = True
                logger.info(" ✅ Phase 4 completed: Generated improvements and ACs")
                
            except json.JSONDecodeError as e:
                logger.error(f" Failed to parse improvements JSON: {e}")
                state["improvements_analysis"] = {"error": f"JSON parsing failed: {e}"}
                state["phase_complete"]["generate_improvements"] = True
                
        except Exception as e:
            logger.error(f" Error in generating improvements: {e}")
            state["improvements_analysis"] = {"error": f"Analysis failed: {e}"}
            state["phase_complete"]["generate_improvements"] = True
            
        return state

    def _agent_node(self, state: AgentState) -> AgentState:
        node_name = "agent"
        state["node_call_count"][node_name] = state["node_call_count"].get(node_name, 0) + 1

        # Read API docs example
        api_docs_example = self._read_api_docs_example()
        code_commit = state.get("code_commit", "")
        software_testing_guide = self._read_software_testing_guide()
        response_ac_guide = self._read_response_ac_guide()
        # Build the analysis  
        prompt = f"""You are an expert software architect analyzing the REST endpoint: {state['endpoint']}
            ANALYSIS STRATEGY:
            1. Review the <current_context> for the endpoint implementation
            2. Get all classes, services, repositories, DTOs, methods and check if they are fully implemented
            3. If you see references to any classes, services, DTOs, or methods that are not fully shown, request more context using the tool
            4. When there are no classes, services, DTOs, or methods that are not fully shown, you have sufficient implementation details, provide your final analysis as JSON
            5. Do not assume that you have enough context, always check the code and use get_project_code_context if any part of the code is not fully implemented.

            WHEN TO USE get_project_code_context TOOL:
            - When you see class/interface names without their implementation
            - When service methods are referenced but not shown
            - When DTO/model classes are mentioned but structure is unclear
            - When exception handling classes are referenced
            - When you need to understand dependencies or business logic

            To use the tool, respond with: "I need to get context for [exact_class_or_method_name]"
            Examples:
            - "I need to get context for UserService"
            - "I need to get context for ValidationException"
            - "I need to get context for OrderDto"

            <current_context> start:
            {state['context']}
            </current_context> end

            <requirements> (it may contains test cases, acceptance criteria, etc.):
            {state['requirements']}
            <requirements/> end

            <additional_instruction> start:
            {state['user_text']}
            <additional_instruction/> end

            <code_diff_commit> start:
            {code_commit}
            <code_diff_commit> end

            <software_testing_guide> start:
            {software_testing_guide}
            <software_testing_guide/> end


            When generating the "document" field, use the format and style shown in the <api_docs_example/> above as a reference.
            When generating the "test_case" field, use the format and style shown in the <software_testing_guide/> above as a reference.

            If you don't have enough context, call the get_project_code_context tool to get more context, don't assume.
            If you have enough context, provide your final analysis as valid JSON with this structure without any other text or symbols like ```json or ``` or "\n":
            {{
                "document": "very detailed step by step explanation of what the endpoint does, including all the business logic and the configuration logic.",
                "requirement_coverage": [
                    {{
                        "requirement": "exact requirement text",
                        "coverage_score": "0-100",
                        "explain": "base on <requirements> and code context in <current_context>, explain how the code meets or fails this requirement. code must match the requirements, all logic, params, request body, etc"
                    }}
                ],
                "existed_test_cases": [
                    {{
                        "test_case": "exact test from <requirements> if included. give the original text, otherwise empty",
                        "coverage_score": "0-100", 
                        "explain": "base on testcase, <requirements> and code context in <current_context>, explain whether this test case is covered by the implementation"
                    }}
                ],
                "additional_test_cases": [
                {{
                    "test_case": "Generate test cases primarily based on the *acceptance criteria and functional requirements*, not solely on the source code. If inconsistencies exist between the requirements and the implementation (e.g., parameter name in requirement is `keyWord`, but the code uses `query`), use the terminology from the *requirement* in test case descriptions. Only reference code to uncover edge cases or unhandled behaviors. Avoid copying implementation-specific terms unless they align with requirements. Apply the formatting rules and test design principles from the SOFTWARE_TESTING_GUIDE above.",
                    "coverage_score": "0-100",
                    "explain": "Evaluate whether the generated test case is currently covered by the source code implementation. Clearly explain how the behavior (expected from the test) matches or differs from the actual code logic."
                }}
                ]

                "improvements": [
                    {{
                        "type": "category",
                        "reason": "base on <requirements> and code context, tell what needs improvement, why?",
                        "solution": "recommended fix"
                    }}
                ],
                "curl_command": "curl command to test the endpoint",
                "response_ac": base on <requirements>, <current_context>, <software_testing_guide>, produce response AC in the following format: {response_ac_guide}
            }}

            Do not assume any code logic, always check the code and use get_project_code_context if any part of the code is not fully implemented.
            Response output in Vietnamese
            Your response:"""
        
        try:
            prompt_file = self._write_prompt_to_file(prompt, f"agent_iteration_{state['iteration_count']}")
            response = self.langchain_llm._call(prompt)
            response_file = self._write_response_to_file(response, state['iteration_count'])
            
            return {
                **state,
                "final_response": response,  
                "history": state["history"] + [response],
                "iteration_count": state["iteration_count"] + 1,
                "node_call_count": state["node_call_count"]
            }
        except Exception as e:
            logger.error(f" Agent node error: {str(e)}")
            return {
                **state,
                "final_response": f"Error in agent reasoning: {str(e)}",
                "iteration_count": state["iteration_count"] + 1,
                "node_call_count": state["node_call_count"]
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
            "phase1_testcases": {
                "existing_testcases": state.get("existing_testcases", []),
                "generated_missing_testcases": state.get("generated_missing_testcases", []),
                "final_testcases": state.get("final_testcases", [])
            },
            "phase2_acceptance_criteria": {
                "current_ac": state.get("current_ac", []),
                "generated_missing_ac": state.get("generated_missing_ac", []),
                "final_ac": state.get("final_ac", [])
            },
            "phase3_coverage_analysis": state.get("additional_coverage", {}),
            "summary": {
                "total_testcases": len(state.get("final_testcases", [])),
                "total_ac": len(state.get("final_ac", [])),
                "analysis_method": "3-phase-langgraph"
            }
        }
        
        # Format comprehensive document
        comprehensive_doc = self._build_comprehensive_document(state)
        analysis_results["comprehensive_document"] = comprehensive_doc
        
        # Convert to JSON string
        json_response = json.dumps(analysis_results, indent=2, ensure_ascii=False)
        
        # Generate HTML response
        html_content = self._generate_html_with_llm(analysis_results)
        
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
            requirements_summary = state['requirements'][:200] + "..." if len(state['requirements']) > 200 else state['requirements']
            document_parts.append(f"- Requirements: {requirements_summary}")
        
        return "\n\n".join(document_parts)

    def _generate_html_with_llm(self, result_dict: Dict[str, Any]) -> str:
        """Generate HTML from JSON analysis result using LLM."""
        try:
            logger.info(" 🤖 Calling LLM for HTML generation...")
            
            # Create a prompt for HTML generation
            prompt = f"""
You are an expert web developer. Convert the following API analysis JSON into a beautiful, well-structured HTML report.

JSON Analysis Data:
{json.dumps(result_dict, indent=2)}

Requirements:
1. Create semantic HTML5 structure
2. Use CSS classes for styling (analysis-report, document-section, requirement-coverage-section, etc.)
3. Color-code coverage scores: green for 80%+, orange for 60-79%, red for <60%
4. Make it responsive and modern looking
5. Include all sections: document, requirement coverage, test cases, improvements, curl command
6. Use proper HTML escaping for special characters
7. Add appropriate CSS classes for easy styling

Generate only the HTML content (no CSS or JavaScript). The HTML should be ready to be embedded in any web application.
HTML Output:
"""
            
            logger.info(f" Sending prompt to LLM ({len(prompt)} characters)")
            
            # Use the LLM to generate HTML
            html_response = self.langchain_llm._call(prompt)
            
            logger.info(f" Received LLM response ({len(html_response)} characters)")
            
            # Clean up the response - extract HTML if it's wrapped in markdown
            html_content = self._extract_html_from_response(html_response)
            
            logger.info(f" Generated HTML response ({len(html_content)} characters)")
            return html_content
            
        except Exception as e:
            logger.error(f" Error generating HTML with LLM: {str(e)}")
            # Fallback to basic HTML if LLM fails
            return ""

    def _extract_html_from_response(self, response: str) -> str:
        """Extract HTML content from LLM response, handling markdown code blocks."""
        import re
        
        logger.info(f" Extracting HTML from response ({len(response)} characters)")
        
        # Try to extract HTML from markdown code blocks
        html_match = re.search(r"```html\s*([\s\S]+?)\s*```", response)
        if html_match:
            logger.info(" Found HTML in ```html code block")
            return html_match.group(1).strip()
        
        # Try without html language specifier
        html_match = re.search(r"```\s*([\s\S]+?)\s*```", response)
        if html_match:
            logger.info(" Found HTML in ``` code block")
            return html_match.group(1).strip()
        
        # If no code blocks, check if response starts with HTML
        if response.strip().startswith('<'):
            logger.info(" Response starts with HTML tag")
            return response.strip()
        
        # If all else fails, return the response as-is
        logger.info(" No HTML code blocks found, returning response as-is")
        return response.strip()

    def _should_use_tool(self, state: AgentState) -> str:
        response = state["final_response"] or ""
        iteration = state["iteration_count"]
        
        # First try to extract JSON from markdown code blocks
        response_text = self._parse_json_response(response)
        response_clean = response_text.strip()
        
        # Check if response is valid JSON (final answer)
        if response_clean.startswith("{") and response_clean.endswith("}"):
            try:
                json.loads(response_clean)
                logger.info(" Valid JSON detected - routing to format_html")
                return "format_html"  # Go to HTML formatting instead of end
            except json.JSONDecodeError:
                logger.warning(" Response looks like JSON but is invalid")
        
        # Also check the original response for JSON
        original_clean = response.strip()
        if original_clean.startswith("{") and original_clean.endswith("}"):
            try:
                json.loads(original_clean)
                logger.info(" Valid JSON detected in original response - routing to format_html")
                return "format_html"
            except json.JSONDecodeError:
                logger.warning(" Original response looks like JSON but is invalid")

        # Check for explicit tool request
        if "I need to get context for" in response or "get_project_code_context" in response:
            logger.info(" Agent explicitly requested tool usage")
            return "use_tool"

        # Stop if max iterations reached
        if iteration >= 5:
            logger.warning(f" Max iterations ({iteration}) reached, forcing end")
            return "end"

        # Check if no new symbols or chunks were retrieved in the last tool call
        if state.get("last_tool_call_symbols") and not state.get("new_retrieved_symbols") and not state.get("new_retrieved_chunks"):
            logger.info(" No new context retrieved in last tool call - ending workflow")
            return "end"

        # Narrow down keyword-based tool triggering
        specific_patterns = [
            r"\b(?:need to inspect|examine|check|see)\s+([A-Z][A-Za-z0-9]*(?:Dto|Service|Controller|Repository|Entity|Exception))\b",
            r"\b(?:implementation of)\s+([A-Z][A-Za-z0-9]*(?:Dto|Service|Controller|Repository|Entity|Exception))\b",
        ]
        for pattern in specific_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                return "use_tool"

        logger.info(" No clear need for tool - proceeding to verification")
        return "end"

    def _get_context_for_symbols(self, symbols: List[str], already_retrieved: List[str], seen_chunks: List[str]) -> Tuple[str, List[str], List[str]]:
        """Fetch and return new context, retrieved symbols, and new chunk IDs."""
        new_context_parts = []
        new_retrieved = []
        new_chunk_ids = []
        for symbol in symbols:
            if symbol not in already_retrieved:
                logger.info(f" Fetching context for: {symbol}")
                context, chunk_ids = self._find_symbol_context(symbol, seen_chunks)
                if "No code found" not in context and "Error retrieving code" not in context:
                    new_context_parts.append(context)
                    new_retrieved.append(symbol)
                    new_chunk_ids.extend(chunk_ids)
                    logger.info(f" Successfully retrieved context for: {symbol} (chunks: {chunk_ids})")
                else:
                    logger.warning(f" No context found for: {symbol}")
            else:
                logger.debug(f" Skipping already retrieved symbol: {symbol}")
        return "\n\n".join(new_context_parts), new_retrieved, new_chunk_ids





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
        symbols = [method["class"] + "." + method["method"] for method in changed_methods]
        endpoints = await self.retriever.retrieve_endpoints(symbols)
        logger.info(f"Endpoints: {endpoints}")

        if endpoint and not endpoints:
            endpoints.append(endpoint)
        
        logger.info(f" Starting LangGraph AnalyzerChain for endpoint: {endpoint}")
    
        try:
            # TODO: update to retrieve docs of all endpoints
            # TODO: use langchain memory to store chat memory so that user can iterate with ai
            docs = []
            endpoint_strs = []
            for endpt in endpoints:
                doc = await self.retriever.retrieve(str(endpt), 1 , hyde=False)
                logger.info(f"endpoint {str(endpt)} docs {len(doc)}")
                docs.extend(doc)
                endpoint_strs.append(str(endpt))
            
            endpoint_str = str(endpoint_strs)
            logger.info(f"endpoint_str: {endpoint_str}")
            
            logger.info(f"len of docs before deduplicate: {len(docs)}")
            docs = self.retriever._deduplicate_documents(docs)
            logger.info(f"len of docs after deduplicate: {len(docs)}")
            initial_context = "\n\n".join(doc.page_content for doc in docs)
            initial_chunk_ids = [doc.metadata.get("id", str(hash(doc.page_content))) for doc in docs]

            initial_state: AgentState = {
                "question": f"Analyze the REST endpoint '{endpoint}' according to the requirements and test cases.",
                "context": initial_context,
                "endpoint": endpoint_str,
                "requirements": requirements_txt,
                "user_text": user_text,
                "code_commit": code_commit,
                "history": [],
                "retrieved_symbols": [],
                "seen_context": initial_chunk_ids,
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
                "needs_more_context": False
            }
            
            logger.info(" Step 3: Starting LangGraph analysis workflow...")
            final_state = await asyncio.to_thread(self.graph.invoke, initial_state)
                    
            logger.info(" Step 4: Parsing and structuring final response...")
            final_response = final_state.get("final_response", "")
            html_response = final_state.get("html_response", "")
            result = self._parse_graph_response(final_response, endpoint)
            result.html_response = html_response  # Add HTML response to the result
            logger.info(f" Analysis complete - method: {result.analysis_method}")
            return result.__dict__
            
        except AnalysisError:
            raise
        except Exception as e:
            logger.error(f" LangGraph analysis failed: {str(e)}")
            try:
                return await self._fallback_analysis(
                    endpoint=endpoint,
                    requirements_txt=requirements_txt,
                    user_text=user_text,
                    initial_context=initial_context if 'initial_context' in locals() else ""
                )
            except Exception as fallback_error:
                logger.error(f" Fallback analysis also failed: {str(fallback_error)}")
                raise AnalysisError(f"Both LangGraph and fallback analysis failed. LangGraph error: {str(e)}, Fallback error: {str(fallback_error)}")

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
            fallback_prompt_file = self._write_prompt_to_file(prompt, "fallback_analysis")
            logger.info(f" Fallback prompt saved to: {fallback_prompt_file}")
            resp = await asyncio.to_thread(self.llm.invoke, prompt)
            fallback_response_file = self._write_response_to_file(resp, 0)
            logger.info(f" Fallback response saved to: {fallback_response_file}")
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

    def clear_cache(self) -> None:
        """Clear any cached resources (LangGraph doesn't require caching)."""
        logger.info("🧹 LangGraph resources cleared (no caching needed)")

    def _write_prompt_to_file(self, prompt: str, prefix: str = "prompt") -> str:
        try:
            prompts_dir = Path("logs/prompts")
            prompts_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{prefix}_{self.project_id}_{timestamp}.txt"
            filepath = prompts_dir / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Project: {self.project_id}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Type: {prefix}\n")
                f.write("=" * 80 + "\n\n")
                f.write(prompt)
            logger.info(f" Prompt saved to: {filepath}")
            return str(filepath)
        except Exception as e:
            logger.warning(f" Failed to save prompt to file: {e}")
            return ""

    def _write_response_to_file(self, response: str, iteration: int) -> str:
        try:
            responses_dir = Path("logs/responses")
            responses_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"response_iteration_{iteration}_{self.project_id}_{timestamp}.txt"
            filepath = responses_dir / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Project: {self.project_id}\n")
                f.write(f"Iteration: {iteration}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write("=" * 80 + "\n\n")
                f.write(response)
            logger.info(f" Response saved to: {filepath}")
            return str(filepath)
        except Exception as e:
            logger.warning(f" Failed to save response to file: {e}")
            return ""

    def _extract_missing_symbols(self, context: str, already_retrieved: List[str]) -> List[str]:
        """Extract symbols that are referenced but not fully implemented in context."""
        import re
        symbols = []
        patterns = [
            r"(\w+(?:Service|Repository|Controller|Dto|Entity|Exception))\b(?!\s*\{)",
            r"new\s+(\w+(?:Service|Repository|Controller|Dto|Entity|Exception))\b",
            r"\b(\w+(?:Service|Repository|Controller|Dto|Entity|Exception))\.\w+\b",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, context)
            for match in matches:
                symbol = match if isinstance(match, str) else match[0]
                if (symbol and symbol not in already_retrieved and len(symbol) > 3 and
                    symbol not in ['String', 'List', 'Map', 'Set', 'Boolean', 'Integer', 'Long', 'Date', 'Time']):
                    symbols.append(symbol)
        unique_symbols = list(dict.fromkeys(symbols))[:3]
        logger.debug(f" Extracted potential symbols: {unique_symbols}")
        return unique_symbols

    def _find_any_symbols(self, context: str, already_retrieved: List[str]) -> List[str]:
        """Find any class names that might be worth investigating."""
        import re
        symbols = []
        class_pattern = r'\b([A-Z][A-Za-z0-9]*(?:Service|Repository|Controller|Dto|Entity|Exception))\b(?!\s*\{)'
        matches = re.findall(class_pattern, context, re.IGNORECASE)
        for match in matches:
            if (len(match) > 0 and
                match not in already_retrieved and
                match not in ['String', 'Object', 'List', 'Map', 'Set', 'Boolean', 'Integer', 'Long', 'Date', 'Time']):
                symbols.append(match)
        unique_symbols = list(dict.fromkeys(symbols))[:2]
        logger.debug(f" Found general class references: {unique_symbols}")
        return unique_symbols

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
        json_match = re.search(r'\{.*?"document".*?\}', response_text, re.DOTALL)
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