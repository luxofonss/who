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
from adapters.model_factory import ModelFactory

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
    # New fields for context detection
    needs_requirements: bool
    needs_code_context: bool
    needs_commit_history: bool
    needs_conversation_history: bool
    context_analysis: str


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

    def __init__(self, project_id: str, model_name: Optional[str] = None, api_key: Optional[str] = None):
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
        
        logger.info(f"🤖 ChatChain initialized with LangGraph for project: {project_id}")

    def _setup_custom_model(self, model_name: str, api_key: str):
        """Setup custom model with provided API key using ModelFactory."""
        self.llm = ModelFactory.create_llm(model_name=model_name, api_key=api_key, temperature=0.1)
        self.langchain_llm = ModelFactory.create_langchain_llm(model_name=model_name, api_key=api_key, temperature=0.1)
        logger.info(f"🔧 Using custom model: {model_name}")

    def _setup_langgraph(self):
        logger.info("🔧 Setting up LangGraph components for chat...")
        
        # Create tool and graph
        self._build_graph()
        logger.info("✅ LangGraph chat setup complete")

    def _build_graph(self):
        graph = StateGraph(ChatState)

        # Add nodes
        graph.add_node("context_detector", self._context_detector_node)
        graph.add_node("chat_agent", self._chat_agent_node)

        # Set entry point
        graph.set_entry_point("context_detector")
        
        # Add edges
        graph.add_edge("context_detector", "chat_agent")

        # Compile the graph
        self.graph = graph.compile()
        logger.info("✅ LangGraph chat workflow compiled")

    def _context_detector_node(self, state: ChatState) -> ChatState:
        """Analyze user question to determine which context elements are needed."""
        node_name = "context_detector"
        state["node_call_count"][node_name] = state["node_call_count"].get(node_name, 0) + 1
        
        logger.info(f"🔍 Analyzing question for context requirements: {state['question'][:100]}...")
        
        detection_prompt = f"""
        Analyze the following user question to determine which types of context information are needed to provide a comprehensive answer.

        User Question: "{state['question']}"

        Available Context Types:
        1. Business Requirements - Contains project specifications, business rules, functional requirements
        2. Code Context - Contains current codebase, implementation details, API documentation
        3. Commit History - Contains recent code changes, git diffs, what was modified
        4. Conversation History - Contains previous messages and context from this chat session

        Instructions:
        - Analyze the question carefully to understand what information is needed
        - Determine which context types are ESSENTIAL for answering the question
        - Be selective - only include context that directly helps answer the question
        - Provide a brief explanation for your decision

        Respond in the following JSON format:
        {{
            "needs_requirements": true/false,
            "needs_code_context": true/false, 
            "needs_commit_history": true/false,
            "needs_conversation_history": true/false,
            "reasoning": "Brief explanation of why each context type is or isn't needed"
        }}

        Examples:
        - "What are the business rules for user authentication?" → needs_requirements: true, others: false
        - "How does the login function work?" → needs_code_context: true, others: false  
        - "What changed in the last commit?" → needs_commit_history: true, others: false
        - "Continue our previous discussion about the API" → needs_conversation_history: true, others: false
        - "Does the current implementation match the requirements?" → needs_requirements: true, needs_code_context: true
        """
        
        try:
            response = self.langchain_llm._call(detection_prompt)
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
                
                logger.info(f"📊 Context analysis result: {analysis.get('reasoning', 'No reasoning provided')}")
                
                return {
                    **state,
                    "needs_requirements": analysis.get("needs_requirements", False),
                    "needs_code_context": analysis.get("needs_code_context", False),
                    "needs_commit_history": analysis.get("needs_commit_history", False),
                    "needs_conversation_history": analysis.get("needs_conversation_history", False),
                    "context_analysis": analysis.get("reasoning", ""),
                    "iteration_count": state["iteration_count"] + 1,
                    "node_call_count": state["node_call_count"]
                }
            else:
                logger.warning("⚠️ Could not parse context analysis JSON, using fallback")
                # Fallback: include all context if analysis fails
                return {
                    **state,
                    "needs_requirements": True,
                    "needs_code_context": True,
                    "needs_commit_history": True,
                    "needs_conversation_history": True,
                    "context_analysis": "Failed to parse analysis, including all context as fallback",
                    "iteration_count": state["iteration_count"] + 1,
                    "node_call_count": state["node_call_count"]
                }
                
        except Exception as e:
            logger.error(f"❌ Context detector node error: {str(e)}")
            # Fallback: include all context if detection fails
            return {
                **state,
                "needs_requirements": True,
                "needs_code_context": True,
                "needs_commit_history": True,
                "needs_conversation_history": True,
                "context_analysis": f"Error in analysis: {str(e)}, including all context as fallback",
                "iteration_count": state["iteration_count"] + 1,
                "node_call_count": state["node_call_count"]
            }

    def _chat_agent_node(self, state: ChatState) -> ChatState:
        """Generate response using only the selected context elements."""
        node_name = "chat_agent"
        state["node_call_count"][node_name] = state["node_call_count"].get(node_name, 0) + 1

        # Build context sections based on detection results
        context_sections = []
        
        if state.get("needs_requirements", False):
            logger.info(f"Need requirements")
            context_sections.append(f"Project Requirements:\n{state['requirements']}")
            
        if state.get("needs_code_context", False):
            logger.info(f"Need code context")
            context_sections.append(f"Code Context:\n{state['context']}")
            
        if state.get("needs_commit_history", False):
            logger.info(f"Need commit history")
            context_sections.append(f"Recent Code Changes:\n{state['code_commit']}")
            
        logger.info(f"Need conversation history")
        history_text = "\n".join(state["history"]) if state["history"] else "No previous conversation."
        context_sections.append(f"Conversation History:\n{history_text}")
        
        # Build the optimized prompt
        selected_context = "\n\n".join(context_sections) if context_sections else "No additional context selected for this question."
        
        prompt = f"""
        You are Expert Software Engineer and Quantity Engineer
        You are helping a user to answer question: {state['question']}

        Instructions:
        - If additional information or context about code elements, requirements, or other details is needed to answer the question accurately, ask user to provide more information.
        - Ensure the response is clear, concise, focused, and adheres to the provided requirements.
        - Respond entirely in Vietnamese, using professional, clear, and technical language suitable for a software engineering context.
        - Prioritize accuracy and alignment with the provided context.

        Context Analysis: {state.get('context_analysis', 'No analysis available')}

        Relevant Information:
        {selected_context}
        """
            
        try:
            response = self.langchain_llm._call(prompt)
            
            # Log which context was used
            used_context = []
            if state.get("needs_requirements", False):
                used_context.append("requirements")
            if state.get("needs_code_context", False):
                used_context.append("code_context")
            if state.get("needs_commit_history", False):
                used_context.append("commit_history")
            if state.get("needs_conversation_history", False):
                used_context.append("conversation_history")
                
            logger.info(f"📝 Response generated using context: {', '.join(used_context) if used_context else 'none'}")
            
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
                "user_message": message,
                # Initialize new context detection fields
                "needs_requirements": False,
                "needs_code_context": False,
                "needs_commit_history": False,
                "needs_conversation_history": False,
                "context_analysis": ""
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
            raise ChatError(f"LangGraph chat failed. Error: {str(e)}")