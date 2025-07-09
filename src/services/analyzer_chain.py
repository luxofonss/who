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


@dataclass
class AnalysisResult:
    document: str
    requirement_coverage: List[Dict[str, Any]]
    improvements: List[Dict[str, str]]
    endpoint: str
    existed_test_cases: List[Dict[str, Any]] = field(default_factory=list)
    additional_test_cases: List[Dict[str, Any]] = field(default_factory=list)
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

        # Add nodes
        graph.add_node("agent", self._agent_node)
        graph.add_node("use_tool", self._call_tool_node)
        graph.add_node("format_html", self._format_html_node)

        # Agent: decide whether to use tool or end
        graph.add_conditional_edges(
            "agent",
            self._should_use_tool,
            {
                "use_tool": "use_tool",
                "format_html": "format_html",
                "end": END
            }
        )

        # Tool: after tool call, go back to agent
        graph.add_edge("use_tool", "agent")
        
        # Format HTML: after formatting, end
        graph.add_edge("format_html", END)

        # Set entry point
        graph.set_entry_point("agent")

        # Compile the graph
        self.graph = graph.compile()
        logger.info(" LangGraph workflow compiled")

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

    def _agent_node(self, state: AgentState) -> AgentState:
        node_name = "agent"
        state["node_call_count"][node_name] = state["node_call_count"].get(node_name, 0) + 1

        # Read API docs example
        api_docs_example = self._read_api_docs_example()
        code_commit = state.get("code_commit", "")
        software_testing_guide = self._read_software_testing_guide()
        # Build the analysis  
        prompt = f"""You are an expert software architect analyzing the REST endpoint: {state['endpoint']}
            ANALYSIS STRATEGY:
            1. Review the current context for the endpoint implementation
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

            CURRENT CONTEXT:
            {state['context']}

            REQUIREMENTS TO ANALYZE (it may contains test cases, acceptance criteria, etc.):
            {state['requirements']}

            ADDITIONAL INSTRUCTIONS:
            {state['user_text']}

            API_DOCS_EXAMPLE:
            {api_docs_example}

            CODE_COMMIT:
            {code_commit}

            SOFTWARE_TESTING_GUIDE:
            {software_testing_guide}

            When generating the "document" field, use the format and style shown in the API_DOCS_EXAMPLE above as a reference.
            When generating the "test_case" field, use the format and style shown in the SOFTWARE_TESTING_GUIDE above as a reference.

            If you don't have enough context, call the get_project_code_context tool to get more context, don't assume.
            If you have enough context, provide your final analysis as valid JSON with this structure without any other text or symbols like ```json or ``` or "\n":
            {{
                "document": "very detailed step by step explanation of what the endpoint does, including all the business logic and the configuration logic. Then apply the the template in SOFTWARE_TESTING_GUIDE",
                "requirement_coverage": [
                    {{
                        "requirement": "exact requirement text",
                        "coverage_score": "0-100",
                        "explain": "how the code meets or fails this requirement"
                    }}
                ],
                "existed_test_cases": [
                    {{
                        "test_case": "exact test from requirements if included. give the original text, otherwise empty",
                        "coverage_score": "0-100", 
                        "explain": "whether this test case is covered by the implementation"
                    }}
                ],
                "additional_test_cases": [
                    {{
                        "test_case": "base on requirements and implementation, generate a test case",
                        "coverage_score": "0-100", 
                        "explain": "whether this test case is covered by the implementation"
                    }}
                ],
                "improvements": [
                    {{
                        "type": "category",
                        "reason": "what needs improvement",
                        "solution": "recommended fix"
                    }}
                ],
                "curl_command": "curl command to test the endpoint"
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

    def _call_tool_node(self, state: AgentState) -> AgentState:
        """Tool calling node - intelligently extract symbols from context and agent response."""
        logger.info(" Tool node activated - extracting symbols to fetch")
        response = state["final_response"] or ""
        current_context = state["context"]
        already_retrieved = state["retrieved_symbols"]
        seen_chunks = state.get("seen_context", [])
        symbols = []

        explicit_patterns = [
            r"I need to get context for ([A-Za-z][A-Za-z0-9_]*)",
            r"get_project_code_context\([\"']([^\"']+)[\"']\)",
            r"(?:context for|details about|information on) ([A-Za-z][A-Za-z0-9_]*)",
        ]
        for pattern in explicit_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                flattened_matches = []
                for match in matches:
                    if isinstance(match, tuple):
                        flattened_matches.extend([m for m in match if m])
                    else:
                        flattened_matches.append(match)
                symbols.extend(flattened_matches)
                logger.info(f" Found explicit symbol requests: {flattened_matches}")

        symbols = [s for s in symbols if s not in already_retrieved]
        if not symbols:
            logger.info(" Looking for symbols mentioned in 'need more context' statements...")
            inspection_patterns = [
                r"(?:need to inspect|we need to inspect|inspect)\s+`([A-Za-z][A-Za-z0-9_]*)`",
                r"(?:need to examine|we need to examine|examine)\s+`([A-Za-z][A-Za-z0-9_]*)`", 
                r"(?:implementation of|depends on the implementation of)\s+`([A-Za-z][A-Za-z0-9_]*)`",
                r"(?:structure of|details of)\s+`([A-Za-z][A-Za-z0-9_]*)`",
                r"(?:need to check|we need to check|check)\s+`([A-Za-z][A-Za-z0-9_]*)`",
                r"(?:need to see|we need to see|see)\s+`([A-Za-z][A-Za-z0-9_]*)`",
                r"(?:requires inspection of|inspection of)\s+`([A-Za-z][A-Za-z0-9_]*)`",
                r"(?:need to inspect|we need to inspect|inspect)\s+([A-Z][A-Za-z0-9]*(?:Dto|Service|Controller|Repository|Response|Request|Entity|Exception))",
                r"(?:implementation of|depends on the implementation of)\s+([A-Z][A-Za-z0-9]*(?:Dto|Service|Controller|Repository|Response|Request|Entity|Exception))",
                r"(?:structure of|details of)\s+([A-Z][A-Za-z0-9]*(?:Dto|Service|Controller|Repository|Response|Request|Entity|Exception))",
            ]
            for pattern in inspection_patterns:
                matches = re.findall(pattern, response, re.IGNORECASE)
                if matches:
                    symbols.extend(matches)
                    logger.info(f" Found symbols in inspection context: {matches}")
        if not symbols:
            logger.info(" No explicit requests - analyzing context for missing implementations...")
            symbols = self._extract_missing_symbols(current_context, already_retrieved)
        if not symbols:
            logger.info(" Scanning for any class/service references...")
            symbols = self._find_any_symbols(current_context, already_retrieved)

        if not symbols:
            logger.warning(" No symbols identified - ending tool usage")
            return {
                **state,
                "final_response": None,
                "last_tool_call_symbols": [],
                "new_retrieved_symbols": []
            }

        symbols = [s for s in symbols if s not in already_retrieved][:10]
        logger.info(f" Found {len(symbols)} symbols to investigate: {symbols}")
        new_context, new_retrieved, new_chunk_ids = self._get_context_for_symbols(symbols, already_retrieved, seen_chunks)
        updated_context = state["context"]
        if new_context:
            updated_context += "\n\n" + new_context
            logger.info(f" Added {len(new_retrieved)} new context sections, {len(new_chunk_ids)} new chunks")
        
        return {
            **state,
            "context": updated_context,
            "retrieved_symbols": state["retrieved_symbols"] + new_retrieved,
            "seen_context": state["seen_context"] + new_chunk_ids,
            "final_response": None,
            "last_tool_call_symbols": symbols,
            "new_retrieved_symbols": new_retrieved
        }

    def _format_html_node(self, state: AgentState) -> AgentState:
        """Convert JSON response to HTML format using LLM."""
        node_name = "format_html"
        state["node_call_count"][node_name] = state["node_call_count"].get(node_name, 0) + 1
        
        logger.info(f" 🎨 Format HTML node activated (iteration {state['iteration_count']})")
        response = state["final_response"] or ""
        logger.info(f" Processing response: {response[:200]}...")
        
        try:
            # Parse the JSON response
            response_clean = response.strip()
            if response_clean.startswith("{") and response_clean.endswith("}"):
                result_dict = json.loads(response_clean)
                logger.info(" Parsed JSON directly from response")
            else:
                # Try to extract JSON from markdown
                response_text = self._parse_json_response(response)
                result_dict = json.loads(response_text)
                logger.info(" Parsed JSON from markdown code blocks")
            
            # Generate HTML using LLM
            logger.info(f" Generating HTML from JSON with {len(result_dict)} fields")
            html_content = self._generate_html_with_llm(result_dict)
            
            logger.info(f" ✅ HTML generation completed ({len(html_content)} characters)")
            return {
                **state,
                "final_response": response,
                "html_response": html_content,
                "iteration_count": state["iteration_count"] + 1,
                "node_call_count": state["node_call_count"]
            }
            
        except Exception as e:
            logger.error(f" ❌ Error formatting HTML: {str(e)}")
            # Return original response with empty HTML if formatting fails
            return {
                **state,
                "final_response": response,
                "html_response": f"<div class='error'>Error formatting HTML: {str(e)}</div>",
                "iteration_count": state["iteration_count"] + 1,
                "node_call_count": state["node_call_count"]
            }

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

    async def run(
            self,
            *,
            endpoint: str,
            requirements_txt: str,
            user_text: str,
            code_commit: str = ""
    ) -> Dict[str, Any]:
        """Run the LangGraph analysis chain and return structured results."""
        try:
            self._validate_inputs(endpoint, requirements_txt, user_text)
        except ValueError as e:
            logger.error(f" Input validation failed: {str(e)}")
            raise AnalysisError(f"Invalid input: {str(e)}")
        
        logger.info(f" Starting LangGraph AnalyzerChain for endpoint: {endpoint}")
    
        try:
            # TODO: limit only 1 docs
            docs = await self.retriever.retrieve(endpoint, 1, hyde=False)
            logger.info(f"len of docs: {len(docs)}")
            initial_context = "\n\n".join(doc.page_content for doc in docs)
            initial_chunk_ids = [doc.metadata.get("id", str(hash(doc.page_content))) for doc in docs]

            initial_state: AgentState = {
                "question": f"Analyze the REST endpoint '{endpoint}' according to the requirements and test cases.",
                "context": initial_context,
                "endpoint": endpoint,
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
                "node_call_count": {}
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
                    html_response=result_dict.get("html_response", ""),
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

    def _validate_inputs(self, endpoint: str, requirements_txt: str, user_text: str) -> None:
        """Validate input parameters."""
        if not endpoint or not isinstance(endpoint, str):
            raise ValueError("endpoint must be a non-empty string")
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
                html_response="<div class='error'>Failed to parse JSON response</div>",
                raw_response=graph_response,
                endpoint=endpoint,
                analysis_method="langgraph"
            )