from __future__ import annotations

from langchain.prompts import PromptTemplate

__all__ = ["PromptBuilder"]


class PromptBuilder:
    """Factory for prompts used by analysis and chat."""

    ANALYSIS_TEMPLATE = PromptTemplate(
        input_variables=[
            "endpoint",
            "context",
            "requirements",
            "testcases",
            "user_text",
        ],
        template=(
            """
You are an expert Java architect. Analyse the REST endpoint {endpoint} using the full code below.

<code>
{context}
</code>

Requirements:
{requirements}

Test cases:
{testcases}

Additional instructions:
{user_text}

Respond ONLY with valid JSON matching this schema:
{{
  "documentation": str,
  "requirement_coverage": list,
  "testcases_coverage": list,
  "issues": list,
  "improvements": list
}}
            """
        ),
    )

    CHAT_TEMPLATE = PromptTemplate(
        input_variables=["history", "context", "message"],
        template=(
            """
You are an AI assistant helping with a Java codebase. Use the following code context and previous conversation to answer the question.

<code>
{context}
</code>

Conversation history:
{history}

User question:
{message}
            """
        ),
    )

    # ------------------------------------------------------------------
    @classmethod
    def build_analysis_prompt(
        cls,
        *,
        endpoint: str,
        context: str,
        requirements: str,
        testcases: str,
        user_text: str,
    ) -> str:
        return cls.ANALYSIS_TEMPLATE.format(
            endpoint=endpoint,
            context=context,
            requirements=requirements,
            testcases=testcases,
            user_text=user_text,
        )

    @classmethod
    def build_chat_prompt(
        cls, *, history: str, context: str, message: str
    ) -> str:
        return cls.CHAT_TEMPLATE.format(history=history, context=context, message=message) 