from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import Gemini  # hoặc adapter nếu bạn dùng Gemini
from langchain.memory import ConversationBufferMemory

from tools import get_code_context

# ✅ LLM có thể là Gemini, hoặc GeminiAdapter nếu bạn tự viết
llm = Gemini(temperature=0)  # hoặc Gemini()

tools = [get_code_context]

agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=ConversationBufferMemory(memory_key="chat_history"),
)
