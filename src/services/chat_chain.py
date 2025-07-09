from __future__ import annotations

import json
import asyncio
import re
from typing import Dict, List, Optional, Any, TypedDict, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from loguru import logger
from langchain_core.tools import Tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor

from adapters.gemini import Gemini, LangChainGemini
from services.retriever import LangChainRetriever
from services.prompt_builder import PromptBuilder


class ChatState(TypedDict):
    question: str
    context: str
    history: List[str]
    retrieved_symbols: List[str]
    final_response: Optional[str]
    iteration_count: int
    seen_context: List[str]
    last_tool_call_symbols: List[str]
    new_retrieved_symbols: List[str]
    node_call_count: Dict[str, int]
    user_message: str


@dataclass
class ChatResult:
    response: str
    context_used: str
    symbols_retrieved: List[str]
    iteration_count: int
    method: str = "langgraph"


class ChatError(Exception):
    pass


class ChatChain:
    """LangGraph-based chat chain for conversational code assistance."""

    def __init__(self, project_id: str):
        if not project_id or not isinstance(project_id, str):
            raise ValueError("project_id must be a non-empty string")
        
        self.project_id = project_id
        self.retriever = LangChainRetriever(project_id)
        self.llm = Gemini(temperature=0.1)  # Slightly higher temperature for chat
        self.langchain_llm = LangChainGemini(temperature=0.1)
        
        # Create tool and graph
        self._setup_langgraph()
        
        logger.info(f"🤖 ChatChain initialized with LangGraph for project: {project_id}")

    def _setup_langgraph(self):
        logger.info("🔧 Setting up LangGraph components for chat...")
        
        # Create the tool
        self.get_context_tool = Tool(
            name="get_project_code_context",
            func=self._find_symbol_context,
            description=(
                "Retrieve code content related to any symbol (class/method/DTO/service/etc) from the project. "
                "Use this when you need more details about classes, methods, or components mentioned in the conversation. "
                "Pass the exact name of the class, interface, method, or component you want to understand better."
            )
        )
        logger.info("✅ Created get_project_code_context tool for chat")
        
        # Create tool executor
        self.tool_executor = ToolExecutor([self.get_context_tool])
        self._build_graph()
        logger.info("✅ LangGraph chat setup complete")

    def _build_graph(self):
        graph = StateGraph(ChatState)

        # Add nodes
        graph.add_node("chat_agent", self._chat_agent_node)
        graph.add_node("use_tool", self._call_tool_node)

        # Agent: decide whether to use tool or end
        graph.add_conditional_edges(
            "chat_agent",
            self._should_use_tool,
            {
                "use_tool": "use_tool",
                "end": END
            }
        )

        # Tool: after tool call, go back to agent
        graph.add_edge("use_tool", "chat_agent")

        # Set entry point
        graph.set_entry_point("chat_agent")

        # Compile the graph
        self.graph = graph.compile()
        logger.info("✅ LangGraph chat workflow compiled")

    def _find_symbol_context(self, symbol: str, seen_chunks: List[str]) -> Tuple[str, List[str]]:
        logger.info(f"🔍 Searching for symbol: '{symbol}' in chat context")
        try:
            # Try direct symbol search first
            logger.debug(f"🎯 Attempting direct symbol lookup for: {symbol}")
            docs = self.retriever.find_by_symbol_name(symbol)
            
            # If no direct match, try a broader search
            if not docs:
                logger.debug(f"🔍 No direct match for '{symbol}', trying semantic search...")
                docs = self.retriever.retrieve_sync(
                    symbol,
                    top=3,  # Get more context for chat
                    hyde=True
                )
            
            if docs:
                new_chunks = []
                new_chunk_ids = []
                for doc in docs:
                    chunk_id = doc.metadata.get("id", str(hash(doc.page_content)))
                    if chunk_id not in seen_chunks:
                        new_chunks.append(doc.page_content)
                        new_chunk_ids.append(chunk_id)
                
                if new_chunks:
                    result = "\n\n".join(new_chunks)
                    logger.info(f"✅ Found {len(new_chunks)} new documents for '{symbol}' ({len(result)} chars)")
                    return f"Code for '{symbol}':\n{result}", new_chunk_ids
                else:
                    return f"No new code found for symbol: {symbol}.", []
            else:
                logger.warning(f"⚠️ No code found for symbol: '{symbol}'")
                return f"No code found for symbol: {symbol}. Try a different class or method name.", []
        except Exception as e:
            logger.error(f"❌ Error retrieving context for symbol '{symbol}': {str(e)}")
            return f"Error retrieving code for symbol: {symbol} - {str(e)}", []

    def _chat_agent_node(self, state: ChatState) -> ChatState:
        node_name = "chat_agent"
        state["node_call_count"][node_name] = state["node_call_count"].get(node_name, 0) + 1

        # Build the chat prompt
        history_text = "\n".join(state["history"]) if state["history"] else "No previous conversation."
        
        prompt = f"""You are an expert software architect and code assistant helping with a Java codebase.

CONVERSATION STRATEGY:
1. Review the current code context and conversation history
2. If the user asks about specific classes, methods, or components that aren't fully shown in the context, use the get_project_code_context tool to get more details
3. Answer the user's question based on the available context
4. Be conversational and helpful
5. If you need more context about any code elements mentioned, request it using the tool

WHEN TO USE get_project_code_context TOOL:
- When the user asks about specific classes, methods, or services not fully shown
- When you need to understand implementation details to answer properly
- When discussing code relationships or dependencies
- When the user wants to see how something works internally
- When you need more context about any topic mentioned by the user

To use the tool, respond with: "I need to get context for `[topic/class/method/api/feature]`"
Examples:
- "I need to get context for `login API`"
- "I need to get context for `UserService`"
- "I need to get context for `authentication flow`"

CURRENT CODE CONTEXT:
{state['context']}

CONVERSATION HISTORY:
{history_text}

USER QUESTION:
{state['user_message']}

If you need more context about any code elements to properly answer the question, use the get_project_code_context tool.
Otherwise, provide a helpful response based on the available context.

Your response:"""
        
        try:
            prompt_file = self._write_prompt_to_file(prompt, f"chat_iteration_{state['iteration_count']}")
            response = self.langchain_llm._call(prompt)
            response_file = self._write_response_to_file(response, state['iteration_count'])
            
            return {
                **state,
                "final_response": response,
                "iteration_count": state["iteration_count"] + 1,
                "node_call_count": state["node_call_count"]
            }
        except Exception as e:
            logger.error(f"❌ Chat agent node error: {str(e)}")
            return {
                **state,
                "final_response": f"I encountered an error while processing your question: {str(e)}",
                "iteration_count": state["iteration_count"] + 1,
                "node_call_count": state["node_call_count"]
            }

    def _should_use_tool(self, state: ChatState) -> str:
        response = state["final_response"] or ""
        iteration = state["iteration_count"]
        
        # Check for explicit tool request
        if "I need to get context for" in response or "get_project_code_context" in response:
            logger.info("🔧 Chat agent explicitly requested tool usage")
            return "use_tool"

        # Stop if max iterations reached
        if iteration >= 4:  # Fewer iterations for chat
            logger.warning(f"⚠️ Max chat iterations ({iteration}) reached, ending")
            return "end"

        # Check if no new symbols were retrieved in the last tool call
        if state.get("last_tool_call_symbols") and not state.get("new_retrieved_symbols"):
            logger.info("✅ No new context retrieved in last tool call - ending chat workflow")
            return "end"

        # Look for patterns indicating need for more context
        context_patterns = [
            r"\b(?:need to see|need to check|let me get|should look at)\s+([A-Z][A-Za-z0-9]*(?:Service|Controller|Repository|Dto|Entity|Exception))\b",
            r"\b(?:examine|inspect|investigate)\s+([A-Z][A-Za-z0-9]*(?:Service|Controller|Repository|Dto|Entity|Exception))\b",
        ]
        
        for pattern in context_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                logger.info(f"🔍 Found context request pattern: {matches}")
                return "use_tool"

        logger.info("✅ Chat response ready - no additional context needed")
        return "end"

    def _call_tool_node(self, state: ChatState) -> ChatState:
        """Tool calling node for chat - use retriever directly with context requests."""
        logger.info("🔧 Chat tool node activated - extracting context requests")
        response = state["final_response"] or ""
        seen_chunks = state.get("seen_context", [])
        
        # Extract what the LLM is asking for context about
        context_requests = []
        
        # Pattern to extract content between quotes/backticks after "context for"
        context_patterns = [
            r"I need to get context for [`\"']([^`\"']+)[`\"']",
            r"I need to get context for ([^\n.!?]+)",
            r"get_project_code_context\([\"']([^\"']+)[\"']\)",
            r"(?:context for|details about|information on) [`\"']([^`\"']+)[`\"']",
            r"(?:context for|details about|information on) ([^\n.!?]+)",
        ]
        
        for pattern in context_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                context_requests.extend(matches)
                logger.info(f"🎯 Found context requests: {matches}")
                break  # Use first matching pattern
        
        if not context_requests:
            logger.info("ℹ️ No context requests identified")
            return {
                **state,
                "final_response": None,
                "last_tool_call_symbols": [],
                "new_retrieved_symbols": []
            }

        # Use retriever directly with the context request
        query = context_requests[0].strip()  # Use first request
        logger.info(f"🔍 Retrieving context for: '{query}'")
        
        try:
            # Use retriever.retrieve directly with the query
            docs = self.retriever.retrieve_sync(query, top=3, hyde=True)
            
            new_chunks = []
            new_chunk_ids = []
            for doc in docs:
                chunk_id = doc.metadata.get("id", str(hash(doc.page_content)))
                if chunk_id not in seen_chunks:
                    new_chunks.append(doc.page_content)
                    new_chunk_ids.append(chunk_id)
            
            if new_chunks:
                new_context = f"Additional context for '{query}':\n\n" + "\n\n".join(new_chunks)
                updated_context = state["context"] + "\n\n" + new_context
                logger.info(f"✅ Added {len(new_chunks)} new context sections for: '{query}'")
            else:
                updated_context = state["context"]
                logger.info(f"ℹ️ No new context found for: '{query}'")
            
            return {
                **state,
                "context": updated_context,
                "retrieved_symbols": state["retrieved_symbols"] + [query],
                "seen_context": state["seen_context"] + new_chunk_ids,
                "final_response": None,
                "last_tool_call_symbols": [query],
                "new_retrieved_symbols": [query] if new_chunks else []
            }
            
        except Exception as e:
            logger.error(f"❌ Error retrieving context for '{query}': {str(e)}")
            return {
                **state,
                "final_response": None,
                "last_tool_call_symbols": [query],
                "new_retrieved_symbols": []
            }

    def _get_context_for_symbols(self, symbols: List[str], already_retrieved: List[str], seen_chunks: List[str]) -> Tuple[str, List[str], List[str]]:
        """Fetch and return new context, retrieved symbols, and new chunk IDs."""
        new_context_parts = []
        new_retrieved = []
        new_chunk_ids = []
        
        for symbol in symbols:
            if symbol not in already_retrieved:
                logger.info(f"🔍 Fetching context for chat: {symbol}")
                context, chunk_ids = self._find_symbol_context(symbol, seen_chunks)
                if "No code found" not in context and "Error retrieving code" not in context:
                    new_context_parts.append(context)
                    new_retrieved.append(symbol)
                    new_chunk_ids.extend(chunk_ids)
                    logger.info(f"✅ Successfully retrieved context for chat: {symbol}")
                else:
                    logger.warning(f"⚠️ No context found for: {symbol}")
            else:
                logger.debug(f"⏭️ Skipping already retrieved symbol: {symbol}")
        
        return "\n\n".join(new_context_parts), new_retrieved, new_chunk_ids

    async def chat(self, message: str, history: List[str]) -> ChatResult:
        """Run the chat chain with LangGraph."""
        logger.info(f"💬 Starting ChatChain for message: {message[:100]}...")
        
        try:
            # Get initial context based on the message
            docs = await self.retriever.retrieve(message, top=3, hyde=True)
            initial_context = "\n\n".join(doc.page_content for doc in docs)
            initial_chunk_ids = [doc.metadata.get("id", str(hash(doc.page_content))) for doc in docs]
            logger.info(f"initial_context: {initial_context}")

            initial_state: ChatState = {
                "question": message,
                "context": initial_context,
                "history": history,
                "retrieved_symbols": [],
                "seen_context": initial_chunk_ids,
                "final_response": None,
                "iteration_count": 0,
                "last_tool_call_symbols": [],
                "new_retrieved_symbols": [],
                "node_call_count": {},
                "user_message": message
            }
            
            logger.info("🚀 Starting LangGraph chat workflow...")
            final_state = await asyncio.to_thread(self.graph.invoke, initial_state)
            
            final_response = final_state.get("final_response", "I couldn't generate a response.")
            
            result = ChatResult(
                response=final_response,
                context_used=final_state.get("context", ""),
                symbols_retrieved=final_state.get("retrieved_symbols", []),
                iteration_count=final_state.get("iteration_count", 0),
                method="langgraph"
            )
            
            logger.info(f"✅ Chat complete - iterations: {result.iteration_count}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Chat chain failed: {str(e)}")
            # Fallback to simple response
            try:
                return await self._fallback_chat(message, history, initial_context if 'initial_context' in locals() else "")
            except Exception as fallback_error:
                logger.error(f"❌ Fallback chat also failed: {str(fallback_error)}")
                raise ChatError(f"Both LangGraph and fallback chat failed. Error: {str(e)}")

    async def _fallback_chat(self, message: str, history: List[str], initial_context: str) -> ChatResult:
        """Fallback to simple chat if LangGraph fails."""
        logger.info("🔄 Using fallback chat method")
        try:
            history_text = "\n".join(history) if history else ""
            prompt = PromptBuilder.build_chat_prompt(
                history=history_text,
                context=initial_context,
                message=message
            )
            
            response = await asyncio.to_thread(self.llm.invoke, prompt)
            
            return ChatResult(
                response=response,
                context_used=initial_context,
                symbols_retrieved=[],
                iteration_count=1,
                method="fallback"
            )
        except Exception as e:
            logger.error(f"❌ Fallback chat execution failed: {str(e)}")
            raise ChatError(f"Fallback chat failed: {str(e)}")

    def _write_prompt_to_file(self, prompt: str, prefix: str = "chat_prompt") -> str:
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
            logger.debug(f"📝 Chat prompt saved to: {filepath}")
            return str(filepath)
        except Exception as e:
            logger.warning(f"⚠️ Failed to save chat prompt to file: {e}")
            return ""

    def _write_response_to_file(self, response: str, iteration: int) -> str:
        try:
            responses_dir = Path("logs/responses")
            responses_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chat_response_iteration_{iteration}_{self.project_id}_{timestamp}.txt"
            filepath = responses_dir / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Project: {self.project_id}\n")
                f.write(f"Iteration: {iteration}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write("=" * 80 + "\n\n")
                f.write(response)
            logger.debug(f"📝 Chat response saved to: {filepath}")
            return str(filepath)
        except Exception as e:
            logger.warning(f"⚠️ Failed to save chat response to file: {e}")
            return "" 