from services.agent_runner import run_agent

async def run_agent_analysis(
        endpoint: str,
        context: str,
        requirements: str,
        testcases: str,
        user_text: str,
) -> str:
    system_instruction = f"""
You're a senior Java code analyst. Based on the context below, analyze the logic behind this endpoint: `{endpoint}`.

- If the context is incomplete and you need additional classes or methods, call the `get_code_context` tool using the class/method name.
- Otherwise, return a JSON-formatted analysis including fields like: documentation, requirements_covered, tests_covered, logic_description.
- Do not make assumptions without code.

Requirements:\n{requirements}
Testcases:\n{testcases}
User Question:\n{user_text}
"""

    return run_agent(project_id, system_instruction)
