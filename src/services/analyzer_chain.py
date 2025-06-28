from __future__ import annotations

import json
import asyncio
import re
from typing import Dict, List, Optional, Any, TypedDict
from dataclasses import dataclass

from loguru import logger
from langchain_core.tools import Tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor

from adapters.gemini import Gemini, LangChainGemini
from services.retriever import LangChainRetriever
from services.prompt_builder import PromptBuilder


class AgentState(TypedDict):
    """State schema for LangGraph agent."""
    question: str
    context: str
    endpoint: str
    requirements: str
    testcases: str
    user_text: str
    history: List[str]
    retrieved_symbols: List[str]
    final_response: Optional[str]
    iteration_count: int


@dataclass
class AnalysisResult:
    """Structured result for endpoint analysis matching PromptBuilder schema."""
    document: str
    requirement_coverage: List[Dict[str, Any]]
    test_cases: List[Dict[str, Any]]
    improvements: List[Dict[str, str]]
    endpoint: str
    raw_response: Optional[str] = None
    analysis_method: str = "langgraph"  # "langgraph" or "fallback"


class AnalysisError(Exception):
    """Custom exception for analysis failures."""
    pass


class AnalyzerChain:
    """High-level orchestrator for endpoint analysis using LangGraph with multi-hop retrieval."""

    def __init__(self, project_id: str):
        if not project_id or not isinstance(project_id, str):
            raise ValueError("project_id must be a non-empty string")
        
        self.project_id = project_id
        self.retriever = LangChainRetriever(project_id)
        self.llm = Gemini(temperature=0)  # For fallback analysis
        self.langchain_llm = LangChainGemini(temperature=0)  # For LangGraph agent
        
        # Create tool and graph
        self._setup_langgraph()
        
        logger.info(f"🚀 AnalyzerChain initialized with LangGraph for project: {project_id}")

    def _setup_langgraph(self):
        """Initialize LangGraph components."""
        logger.info("🔧 Setting up LangGraph components...")
        
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
        logger.info("✅ Created get_project_code_context tool")
        
        # Create tool executor
        self.tool_executor = ToolExecutor([self.get_context_tool])
        logger.info("✅ Created tool executor")
        
        # Build the graph
        logger.info("🏗️ Building LangGraph workflow...")
        self._build_graph()
        logger.info("✅ LangGraph setup complete")

    def _find_symbol_context(self, symbol: str) -> str:
        """Helper method to find code context for a given symbol."""
        logger.info(f"🔍 Searching for symbol: '{symbol}'")
        try:
            # Try direct symbol search first
            logger.debug(f"🎯 Attempting direct symbol lookup for: {symbol}")
            docs = self.retriever.find_by_symbol_name(symbol)

            logger.info(f"🔍 Docs: {docs}")
            
            # If no direct match, try a broader search
            if not docs:
                logger.debug(f"🔄 No direct match for '{symbol}', trying semantic search...")
                docs = self.retriever.retrieve_sync(
                    symbol, 
                    user_text="", 
                    top=3, 
                    hyde=False
                )
            
            if docs:
                result = "\n\n".join(d.page_content for d in docs)
                logger.info(f"✅ Found {len(docs)} documents for '{symbol}' ({len(result)} chars)")
                logger.debug(f"📄 Context preview for '{symbol}': {result[:150]}...")
                return f"Context for '{symbol}':\n{result}"
            else:
                logger.warning(f"❌ No code found for symbol: '{symbol}'")
                return f"No code found for symbol: {symbol}. Try a different class or method name."
        except Exception as e:
            logger.error(f"💥 Error retrieving context for symbol '{symbol}': {str(e)}")
            return f"Error retrieving code for symbol: {symbol} - {str(e)}"

    def _build_graph(self):
        """Build the LangGraph workflow."""
        graph = StateGraph(AgentState)
        
        # Add nodes
        graph.add_node("agent", self._agent_node)
        graph.add_node("use_tool", self._call_tool_node)
        
        # Add conditional edges
        graph.add_conditional_edges(
            "agent",
            self._should_use_tool,
            {
                "use_tool": "use_tool",
                "end": END
            }
        )
        
        # Add edge from tool back to agent
        graph.add_edge("use_tool", "agent")
        
        # Set entry point
        graph.set_entry_point("agent")
        
        # Compile the graph
        self.graph = graph.compile()

    def _agent_node(self, state: AgentState) -> AgentState:
        """Agent reasoning node."""
        logger.info(f"🤖 Agent iteration {state['iteration_count']} - analyzing endpoint: {state['endpoint']}")
        logger.debug(f"📊 Current context length: {len(state['context'])} chars")
        logger.debug(f"🔍 Retrieved symbols so far: {state['retrieved_symbols']}")
        
        # Build the analysis prompt
        prompt = f"""You are an expert software architect analyzing the REST endpoint: {state['endpoint']}

            CURRENT CONTEXT:
            {state['context']}

            REQUIREMENTS TO ANALYZE:
            {state['requirements']}

            TEST CASES TO VERIFY:
            {state['testcases']}

            ADDITIONAL INSTRUCTIONS:
            {state['user_text']}

            ANALYSIS STRATEGY:
            1. Review the current context for the endpoint implementation
            2. If you see references to classes, services, DTOs, or methods that are not fully shown, request more context using the tool
            3. When you have sufficient implementation details, provide your final analysis as JSON

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

            If you have enough context, provide your final analysis as valid JSON with this structure:
            {{
                "document": "detailed explanation of what the endpoint does",
                "requirement_coverage": [
                    {{
                        "requirement": "exact requirement text",
                        "coverage_score": "0-100",
                        "explain": "how the code meets or fails this requirement"
                    }}
                ],
                "test_cases": [
                    {{
                        "test_case": "exact test case text",
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
                ]
            }}

            Your response:"""
        
        try:
            logger.debug("🔄 Calling LLM for agent reasoning...")
            response = self.langchain_llm._call(prompt)
            logger.info(f"✅ Agent response received (length: {len(response)} chars)")
            logger.debug(f"📝 Agent response preview: {response[:300]}...")
            
            return {
                **state,
                "final_response": response,
                "history": state["history"] + [response],
                "iteration_count": state["iteration_count"] + 1
            }
        except Exception as e:
            logger.error(f"❌ Agent node error: {str(e)}")
            return {
                **state,
                "final_response": f"Error in agent reasoning: {str(e)}",
                "iteration_count": state["iteration_count"] + 1
            }

    def _should_use_tool(self, state: AgentState) -> str:
        """Determine if the agent needs to use a tool."""
        response = state["final_response"] or ""
        iteration = state["iteration_count"]
        
        logger.debug(f"🤔 Decision time - iteration {iteration}, response length: {len(response)}")
        
        # Check if agent is requesting tool usage
        if "I need to get context for" in response or "get_project_code_context" in response:
            logger.info("🔧 Agent explicitly requested tool usage")
            return "use_tool"
        
        # Check if response looks like JSON (final answer)
        response_clean = response.strip()
        if (response_clean.startswith("{") and response_clean.endswith("}")) or \
           ("document" in response and "requirement_coverage" in response):
            logger.info("✅ Agent provided final JSON response - workflow complete")
            return "end"
        
        # If too many iterations, force end
        if iteration >= 5:
            logger.warning(f"⚠️ Max iterations ({iteration}) reached, forcing end")
            return "end"
        
        # If agent seems to be describing what it sees but not asking for tools, encourage tool usage
        if iteration < 3 and any(keyword in response.lower() for keyword in 
                                ["service", "class", "method", "dto", "controller", "repository"]):
            logger.info("🔧 Detected class/service references, encouraging tool usage")
            return "use_tool"
        
        # Default to end if we can't determine clearly
        logger.info("🏁 Uncertain response, defaulting to end workflow")
        return "end"

    def _call_tool_node(self, state: AgentState) -> AgentState:
        """Tool calling node - intelligently extract symbols from context and agent response."""
        logger.info("🔧 Tool node activated - extracting symbols to fetch")
        
        response = state["final_response"] or ""
        current_context = state["context"]
        already_retrieved = state["retrieved_symbols"]
        
        # Extract symbol names from the response
        symbols = []
        
        # 1. Look for explicit tool requests from agent
        import re
        explicit_patterns = [
            r"I need to get context for ([A-Za-z][A-Za-z0-9_]*)",
            r"get_project_code_context\([\"']([^\"']+)[\"']\)",
            r"(?:context for|details about|information on) ([A-Za-z][A-Za-z0-9_]*)",
        ]
        
        for pattern in explicit_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            symbols.extend(matches)
            if matches:
                logger.info(f"🎯 Found explicit symbol requests: {matches}")
        
        # 2. If no explicit requests, intelligently extract from current context
        if not symbols:
            logger.info("🔍 No explicit requests - analyzing context for missing implementations...")
            symbols = self._extract_missing_symbols(current_context, already_retrieved)
        
        # 3. If still no symbols, try to find any referenced classes/services
        if not symbols:
            logger.info("🔎 Scanning for any class/service references...")
            symbols = self._find_any_symbols(current_context, already_retrieved)
        
        # Log what we found
        if symbols:
            logger.info(f"📋 Found {len(symbols)} symbols to investigate: {symbols}")
            logger.info(f"symbols: {symbols}")
        else:
            logger.warning("⚠️ No symbols identified - ending tool usage")
            return {
                **state,
                "final_response": None  # Reset for next agent iteration
            }
        
        # Get context for symbols (limit to 2 per iteration for performance)
        new_context_parts = []
        new_retrieved = []
        
        for symbol in symbols[:2]:
            if symbol not in already_retrieved:
                logger.info(f"🔍 Fetching context for: {symbol}")
                context = self._find_symbol_context(symbol)
                if "No code found" not in context:
                    new_context_parts.append(context)
                    new_retrieved.append(symbol)
                    logger.info(f"✅ Successfully retrieved context for: {symbol}")
                else:
                    logger.warning(f"❌ No context found for: {symbol}")
            else:
                logger.debug(f"⏭️ Skipping already retrieved symbol: {symbol}")
        
        # Update state
        updated_context = state["context"]
        if new_context_parts:
            updated_context += "\n\n" + "\n\n".join(new_context_parts)
            logger.info(f"📝 Added {len(new_context_parts)} new context sections")
        
        return {
            **state,
            "context": updated_context,
            "retrieved_symbols": state["retrieved_symbols"] + new_retrieved,
            "final_response": None  # Reset for next agent iteration
        }

    def _extract_missing_symbols(self, context: str, already_retrieved: List[str]) -> List[str]:
        """Extract symbols that are referenced but not fully implemented in context."""
        import re
        symbols = []
        
        # Look for class instantiations and method calls that might need more context
        patterns = [
            r"(\w+Service)(?:\s+\w+\s*=|\.\w+|\()",  # Services
            r"(\w+Repository)(?:\s+\w+\s*=|\.\w+|\()",  # Repositories  
            r"(\w+Controller)(?:\s+\w+\s*=|\.\w+|\()",  # Controllers
            r"(\w+Exception)(?:\s+\w+\s*=|\(|\s+\w+)",  # Exceptions
            r"(\w+Dto)(?:\s+\w+\s*=|\(|\s+\w+)",  # DTOs
            r"(\w+Entity)(?:\s+\w+\s*=|\.\w+|\()",  # Entities
            r"(\w+Config)(?:\s+\w+\s*=|\.\w+|\()",  # Config classes
            r"(\w+Utils?)(?:\s+\w+\s*=|\.\w+|\()",  # Utility classes
            r"(\w+Response)(?:\s+\w+\s*=|\<|\()",  # Response classes
            r"(\w+Request)(?:\s+\w+\s*=|\<|\()",  # Request classes
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, context)
            for match in matches:
                if match not in already_retrieved and len(match) > 3:  # Filter out too short names
                    symbols.append(match)
        
        # Remove duplicates while preserving order
        unique_symbols = list(dict.fromkeys(symbols))
        logger.debug(f"🔍 Extracted potential symbols: {unique_symbols}")
        return unique_symbols[:5]  # Limit to 5 most relevant
    
    def _find_any_symbols(self, context: str, already_retrieved: List[str]) -> List[str]:
        """Find any class names that might be worth investigating."""
        import re
        symbols = []
        
        # Look for any capitalized words that look like class names
        class_pattern = r'\b([A-Z][a-z]+(?:[A-Z][a-z]*)*)\b'
        matches = re.findall(class_pattern, context)
        
        for match in matches:
            # Filter for likely class names (avoid common words)
            if (len(match) > 4 and 
                match not in already_retrieved and
                match not in ['String', 'Object', 'List', 'Map', 'Set', 'Boolean', 'Integer', 'Long', 'Date', 'Time']):
                symbols.append(match)
        
        # Remove duplicates and limit
        unique_symbols = list(dict.fromkeys(symbols))[:3]
        logger.debug(f"🔎 Found general class references: {unique_symbols}")
        return unique_symbols

    def _validate_inputs(self, endpoint: str, requirements_txt: str, testcases_txt: str, user_text: str) -> None:
        """Validate input parameters."""
        if not endpoint or not isinstance(endpoint, str):
            raise ValueError("endpoint must be a non-empty string")
        
        # Other parameters can be empty strings, but should be strings
        for param_name, param_value in [
            ("requirements_txt", requirements_txt),
            ("testcases_txt", testcases_txt), 
            ("user_text", user_text)
        ]:
            if not isinstance(param_value, str):
                raise ValueError(f"{param_name} must be a string")

    def _parse_graph_response(self, graph_response: str, endpoint: str) -> AnalysisResult:
        """Parse LangGraph response and create structured result."""
        response_text = graph_response.strip()
        
        logger.info(f"🔍 Graph response: {response_text[:200]}...")
        
        # Try to extract JSON from markdown code blocks if present
        if response_text.startswith("````json") and response_text.endswith("````"):
            response_text = response_text[8:-4].strip()
        elif response_text.startswith("```json") and response_text.endswith("```"):
            response_text = response_text[7:-3].strip()
        elif response_text.startswith("````") and response_text.endswith("````"):
            # Handle generic 4-backtick code blocks
            lines = response_text.split('\n')
            if len(lines) > 2:
                response_text = '\n'.join(lines[1:-1]).strip()
        elif response_text.startswith("```") and response_text.endswith("```"):
            # Handle generic 3-backtick code blocks
            lines = response_text.split('\n')
            if len(lines) > 2:
                response_text = '\n'.join(lines[1:-1]).strip()
        
        # Find JSON within the response if it's mixed with other text
        json_match = re.search(r'\{.*?"document".*?\}', response_text, re.DOTALL)
        if json_match and not response_text.strip().startswith('{'):
            response_text = json_match.group(0)
        
        try:
            result_dict = json.loads(response_text)
            logger.info("✅ LangGraph returned valid JSON analysis")
            
            return AnalysisResult(
                document=result_dict.get("document", ""),
                requirement_coverage=result_dict.get("requirement_coverage", []),
                test_cases=result_dict.get("test_cases", []),
                improvements=result_dict.get("improvements", []),
                endpoint=endpoint,
                analysis_method="langgraph"
            )
        except json.JSONDecodeError as e:
            logger.warning(f"⚠️ LangGraph response is not valid JSON: {str(e)[:100]}...")
            logger.debug(f"Raw graph response: {response_text[:500]}...")
            return AnalysisResult(
                document="Analysis completed but not in JSON format",
                requirement_coverage=[],
                test_cases=[],
                improvements=[],
                endpoint=endpoint,
                raw_response=graph_response,
                analysis_method="langgraph"
            )

    async def run(
            self,
            *,
            endpoint: str,
            requirements_txt: str,
            testcases_txt: str,
            user_text: str,
    ) -> Dict[str, Any]:
        """Run the LangGraph analysis chain and return structured results."""
        
        # Validate inputs
        try:
            self._validate_inputs(endpoint, requirements_txt, testcases_txt, user_text)
        except ValueError as e:
            logger.error(f"❌ Input validation failed: {str(e)}")
            raise AnalysisError(f"Invalid input: {str(e)}")
        
        logger.info(f"🚀 Starting LangGraph AnalyzerChain for endpoint: {endpoint}")
        logger.info(f"📋 Requirements: {len(requirements_txt)} chars")
        logger.info(f"🧪 Test cases: {len(testcases_txt)} chars")
        logger.info(f"💬 User instructions: {len(user_text)} chars")
        
        try:
            # Step 1: Initial retrieval to get starting context
            logger.info("🔍 Step 1: Retrieving initial context from vector database...")
            docs = await self.retriever.retrieve(endpoint, user_text, top=6, hyde=False)
            initial_context = "\n\n".join(doc.page_content for doc in docs)
            
            logger.info(f"✅ Retrieved {len(docs)} initial documents ({len(initial_context)} chars total)")
            logger.debug(f"📄 Initial context preview: {initial_context[:200]}...")
            
            # Step 2: Create initial state for LangGraph
            logger.info("🏗️ Step 2: Creating initial state for LangGraph workflow...")
            initial_state: AgentState = {
                "question": f"Analyze the REST endpoint '{endpoint}' according to the requirements and test cases.",
                "context": initial_context,
                "endpoint": endpoint,
                "requirements": requirements_txt,
                "testcases": testcases_txt,
                "user_text": user_text,
                "history": [],
                "retrieved_symbols": [],
                "final_response": None,
                "iteration_count": 0
            }
            logger.debug("✅ Initial state created successfully")
            
            # Step 3: Run LangGraph workflow
            logger.info("🚀 Step 3: Starting LangGraph analysis workflow...")
            final_state = await asyncio.to_thread(self.graph.invoke, initial_state)
            
            iterations = final_state['iteration_count']
            retrieved_symbols = final_state['retrieved_symbols']
            final_response_length = len(final_state.get("final_response", ""))
            
            logger.info(f"🎉 LangGraph workflow completed successfully!")
            logger.info(f"📊 Statistics: {iterations} iterations, {len(retrieved_symbols)} symbols retrieved")
            logger.info(f"🔍 Retrieved context for symbols: {retrieved_symbols}")
            logger.info(f"📝 Final response length: {final_response_length} chars")
            
            # Step 4: Parse and structure the response
            logger.info("🔧 Step 4: Parsing and structuring final response...")
            final_response = final_state.get("final_response", "")
            result = self._parse_graph_response(final_response, endpoint)
            logger.info(f"✅ Analysis complete - method: {result.analysis_method}")
            return result.__dict__
            
        except AnalysisError:
            # Re-raise AnalysisError as-is
            raise
        except Exception as e:
            logger.error(f"❌ LangGraph analysis failed: {str(e)}")
            
            # Fallback to original approach if LangGraph fails
            logger.info("🔄 Falling back to original analysis approach")
            try:
                return await self._fallback_analysis(
                    endpoint=endpoint,
                    requirements_txt=requirements_txt,
                    testcases_txt=testcases_txt,
                    user_text=user_text,
                    initial_context=initial_context if 'initial_context' in locals() else ""
                )
            except Exception as fallback_error:
                logger.error(f"❌ Fallback analysis also failed: {str(fallback_error)}")
                raise AnalysisError(f"Both LangGraph and fallback analysis failed. LangGraph error: {str(e)}, Fallback error: {str(fallback_error)}")

    async def _fallback_analysis(
            self,
            *,
            endpoint: str,
            requirements_txt: str,
            testcases_txt: str,
            user_text: str,
            initial_context: str
    ) -> Dict[str, Any]:
        """Fallback to original analysis approach if agent fails."""
        logger.info("🔄 Using fallback analysis method")
        
        try:
            # Use the original prompt-based approach
            prompt = PromptBuilder.build_analysis_prompt(
                endpoint=endpoint,
                context=initial_context,
                requirements=requirements_txt,
                testcases=testcases_txt,
                user_text=user_text,
            )
            
            resp = await asyncio.to_thread(self.llm.invoke, prompt)
            
            try:
                result_dict = json.loads(resp)
                logger.info("✅ Fallback analysis returned valid JSON")
                
                result = AnalysisResult(
                    document=result_dict.get("document", ""),
                    requirement_coverage=result_dict.get("requirement_coverage", []),
                    test_cases=result_dict.get("test_cases", []),
                    improvements=result_dict.get("improvements", []),
                    endpoint=endpoint,
                    analysis_method="fallback"
                )
                return result.__dict__
                
            except json.JSONDecodeError:
                logger.warning("⚠️ Fallback analysis also failed to return JSON")
                result = AnalysisResult(
                    document="Fallback analysis completed but not in JSON format",
                    requirement_coverage=[],
                    test_cases=[],
                    improvements=[],
                    endpoint=endpoint,
                    raw_response=resp,
                    analysis_method="fallback"
                )
                return result.__dict__
                
        except Exception as e:
            logger.error(f"❌ Fallback analysis execution failed: {str(e)}")
            raise AnalysisError(f"Fallback analysis failed: {str(e)}")

    def clear_cache(self) -> None:
        """Clear any cached resources (LangGraph doesn't require caching)."""
        logger.info("🧹 LangGraph resources cleared (no caching needed)")
