from __future__ import annotations

import json
import asyncio
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from loguru import logger
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.tools import tool

from adapters.gemini import Gemini
from services.retriever import LangChainRetriever
from services.prompt_builder import PromptBuilder


@dataclass
class AnalysisResult:
    """Structured result for endpoint analysis matching PromptBuilder schema."""
    document: str
    requirement_coverage: List[Dict[str, Any]]
    test_cases: List[Dict[str, Any]]
    improvements: List[Dict[str, str]]
    endpoint: str
    raw_response: Optional[str] = None
    analysis_method: str = "agent"  # "agent" or "fallback"


class AnalysisError(Exception):
    """Custom exception for analysis failures."""
    pass


class AnalyzerChain:
    """High-level orchestrator for endpoint analysis with multi-hop retrieval using agent-based approach."""

    def __init__(self, project_id: str):
        if not project_id or not isinstance(project_id, str):
            raise ValueError("project_id must be a non-empty string")
        
        self.project_id = project_id
        self.retriever = LangChainRetriever(project_id)
        self.llm = Gemini(temperature=0)
        self._agent = None
        self._agent_context_hash = None
        
        logger.info(f"🚀 AnalyzerChain initialized for project: {project_id}")

    def _get_context_hash(self, initial_context: str, endpoint: str, requirements: str, testcases: str, user_text: str) -> str:
        """Generate a hash for the context to determine if agent needs recreation."""
        content = f"{endpoint}|{requirements}|{testcases}|{user_text}|{len(initial_context)}"
        return str(hash(content))

    def _create_agent(self, initial_context: str, endpoint: str, requirements: str, testcases: str, user_text: str):
        """Create a LangChain agent with initial context and analysis instructions."""
        
        # Check if we can reuse existing agent
        context_hash = self._get_context_hash(initial_context, endpoint, requirements, testcases, user_text)
        if self._agent and self._agent_context_hash == context_hash:
            logger.debug("♻️ Reusing existing agent with same context")
            return self._agent
        
        logger.info("🤖 Creating new analysis agent")
        
        # Use the existing PromptBuilder for consistent prompt structure
        analysis_prompt = PromptBuilder.build_analysis_prompt(
            endpoint=endpoint,
            context=initial_context,
            requirements=requirements,
            testcases=testcases,
            user_text=user_text,
        )
        
        # Add agent-specific instructions to work with tools
        agent_instructions = f"""
{analysis_prompt}

ADDITIONAL AGENT INSTRUCTIONS:
- If you need additional classes, methods, or code blocks to complete your analysis, use the `get_project_code_context` tool with the specific symbol name (e.g., "ClassName.methodName")
- You can call the tool multiple times to gather all necessary context
- Always provide the final response in the JSON format specified above
"""

        @tool
        def get_project_code_context(symbol: str) -> str:
            """Return code (summary + content) related to *symbol* (Class.method) from the current project.
            
            Usage example:
                get_project_code_context("UserService.getUserById")
            """
            try:
                docs = self.retriever.find_by_symbol_name(symbol)
                if docs:
                    result = "\n\n".join(d.page_content for d in docs)
                    logger.info(f"🔍 Retrieved additional context for symbol: {symbol}")
                    return result
                else:
                    logger.warning(f"⚠️ No code found for symbol: {symbol}")
                    return f"No code found for symbol: {symbol}"
            except Exception as e:
                logger.error(f"❌ Error retrieving context for symbol {symbol}: {str(e)}")
                return f"Error retrieving code for symbol: {symbol} - {str(e)}"

        tools = [get_project_code_context]

        try:
            agent = initialize_agent(
                tools=tools,
                llm=self.llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                memory=ConversationBufferMemory(memory_key="chat_history"),
                agent_kwargs={"prefix": agent_instructions},
                max_iterations=3,  # Limit iterations to prevent infinite loops
                early_stopping_method="generate"  # Stop early if no progress
            )
            
            # Cache agent and context hash for reuse
            self._agent = agent
            self._agent_context_hash = context_hash
            
            return agent
        except Exception as e:
            logger.error(f"❌ Failed to create agent: {str(e)}")
            raise AnalysisError(f"Agent creation failed: {str(e)}")

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

    def _parse_agent_response(self, agent_response: str, endpoint: str) -> AnalysisResult:
        """Parse agent response and create structured result."""
        try:
            result_dict = json.loads(agent_response)
            logger.info("✅ Agent returned valid JSON analysis")
            
            return AnalysisResult(
                document=result_dict.get("document", ""),
                requirement_coverage=result_dict.get("requirement_coverage", []),
                test_cases=result_dict.get("test_cases", []),
                improvements=result_dict.get("improvements", []),
                endpoint=endpoint,
                analysis_method="agent"
            )
        except json.JSONDecodeError:
            logger.warning("⚠️ Agent response is not valid JSON, returning raw response")
            return AnalysisResult(
                document="Analysis completed but not in JSON format",
                requirement_coverage=[],
                test_cases=[],
                improvements=[],
                endpoint=endpoint,
                raw_response=agent_response,
                analysis_method="agent"
            )

    async def run(
            self,
            *,
            endpoint: str,
            requirements_txt: str,
            testcases_txt: str,
            user_text: str,
    ) -> Dict[str, Any]:
        """Run the analysis chain and return structured results."""
        
        # Validate inputs
        try:
            self._validate_inputs(endpoint, requirements_txt, testcases_txt, user_text)
        except ValueError as e:
            logger.error(f"❌ Input validation failed: {str(e)}")
            raise AnalysisError(f"Invalid input: {str(e)}")
        
        logger.info(f"🔍 Starting AnalyzerChain for: {endpoint}")
        
        try:
            # Step 1: Initial retrieval to get starting context
            logger.info("🔁 Retrieving initial context from vector database")
            docs = await self.retriever.retrieve(endpoint, user_text, top=6, hyde=False)
            initial_context = "\n\n".join(doc.page_content for doc in docs)
            
            logger.info(f"📄 Retrieved {len(docs)} initial documents")
            
            # Step 2: Create agent with initial context and analysis instructions
            logger.info("🤖 Creating analysis agent with initial context")
            agent = self._create_agent(
                initial_context=initial_context,
                endpoint=endpoint,
                requirements=requirements_txt,
                testcases=testcases_txt,
                user_text=user_text
            )
            
            # Step 3: Run agent analysis (it can request more context as needed)
            logger.info("🚀 Starting agent-based analysis")
            agent_response = await asyncio.to_thread(agent.run, "Please analyze this endpoint.")
            
            # Step 4: Parse and structure the response
            result = self._parse_agent_response(agent_response, endpoint)
            return result.__dict__
            
        except AnalysisError:
            # Re-raise AnalysisError as-is
            raise
        except Exception as e:
            logger.error(f"❌ Agent analysis failed: {str(e)}")
            
            # Fallback to original approach if agent fails
            logger.info("🔄 Falling back to original analysis approach")
            try:
                return await self._fallback_analysis(
                    endpoint=endpoint,
                    requirements_txt=requirements_txt,
                    testcases_txt=testcases_txt,
                    user_text=user_text,
                    initial_context=initial_context
                )
            except Exception as fallback_error:
                logger.error(f"❌ Fallback analysis also failed: {str(fallback_error)}")
                raise AnalysisError(f"Both agent and fallback analysis failed. Agent error: {str(e)}, Fallback error: {str(fallback_error)}")

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
        """Clear cached agent to force recreation on next analysis."""
        self._agent = None
        self._agent_context_hash = None
        logger.info("🧹 Agent cache cleared")
