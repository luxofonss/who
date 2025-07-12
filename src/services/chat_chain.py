from __future__ import annotations

import json
import asyncio
import re
from typing import Dict, List, Optional, TypedDict
from dataclasses import dataclass

from loguru import logger
from langgraph.graph import StateGraph

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
    code_commit: str
    requirements: str


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
        
        # Create tool executor
        self._build_graph()
        logger.info("✅ LangGraph chat setup complete")

    def _build_graph(self):
        graph = StateGraph(ChatState)

        # Add nodes
        graph.add_node("chat_agent", self._chat_agent_node)

        # Set entry point
        graph.set_entry_point("chat_agent")

        # Compile the graph
        self.graph = graph.compile()
        logger.info("✅ LangGraph chat workflow compiled")

    def _chat_agent_node(self, state: ChatState) -> ChatState:
        node_name = "chat_agent"
        state["node_call_count"][node_name] = state["node_call_count"].get(node_name, 0) + 1

        # Build the chat prompt
        history_text = "\n".join(state["history"]) if state["history"] else "No previous conversation."
        
        prompt = f"""
        You are Expert Software Engineer and Quantity Engineer
        You are helping a user to answer question: {state['question']}

        Instructions:
        - If additional information or context about code elements, requirements, or other details is needed to answer the question accurately,  ask user to provide more information.
        - Ensure the response is clear, concise, focused, and adheres to the provided requirements.
        - Respond entirely in Vietnamese, using professional, clear, and technical language suitable for a software engineering context.
        - Prioritize accuracy and alignment with the provided Interaction history and project requirements.

        Provided Information:
        - Interaction history: {history_text}
        - Project requirements: {state['requirements']}
        - Code context: {state['context']}
        - Code commit history: {state['code_commit']}
        
    """
            
        try:
            response = self.langchain_llm._call(prompt)
            
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



    async def chat(self,
            message: str, 
            history: List[str], 
            endpoint: str,
            requirements: str,
            code_commit: str = "",
            changed_methods: List[Dict[str, str]] = []) -> ChatResult:
        """Run the chat chain with LangGraph."""
        logger.info(f"💬 Starting ChatChain for message: {message[:100]}...")
        
        try:
            # Get initial context based on the message
            
            symbols = [method["class"] + "." + method["method"] for method in changed_methods]
            endpoints = await self.retriever.retrieve_endpoints(symbols)
            logger.info(f"Endpoints: {endpoints}")

            if endpoint and not endpoints:
                endpoints.append(endpoint)
            
            logger.info(f" Starting LangGraph ChatChain for endpoint: {endpoint}")

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

            initial_state: ChatState = {
                "question": message,
                "context": initial_context,
                "history": history,
                "code_commit": code_commit,
                "requirements": requirements,
                "retrieved_symbols": [],
                "final_response": None,
                "iteration_count": 0,
                "seen_context": initial_chunk_ids,
                "last_tool_call_symbols": [],
                "new_retrieved_symbols": [],
                "node_call_count": {},
                "user_message": message
            }

            logger.info(" Step 3: Starting LangGraph analysis workflow...")
            final_state = await asyncio.to_thread(self.graph.invoke, initial_state)
                    
            logger.info(f" Analysis complete - returning final response")
            return ChatResult(
                response=final_state["final_response"],
                context_used=final_state["context"],
                symbols_retrieved=final_state["retrieved_symbols"],
                iteration_count=final_state["iteration_count"],
                method="langgraph"
            )
            
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