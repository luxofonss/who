from __future__ import annotations

from langchain_core.prompts import PromptTemplate

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

Your response MUST be valid JSON and follow **exactly** this schema:

{{
  "document": "<A detailed explanation of what the endpoint does, how it works, and any relevant observations about the code design or functionality. Mention request structure, flow, validation, and response if applicable>",

  "requirement_coverage": [
    {{
      "requirement": "<Exact text of one requirement>",
      "coverage_score": "Score from 0 to 100",
      "explain": "<Explain how the source code meets or fails to meet this requirement. Mention flow of logic, validation, branching, service calls, and anything relevant>"
    }}
  ],

  "test_cases": [
    {{
      "test_case": "<Exact test case text>",
      "coverage_score": "Score from 0 to 100",
      "explain": "<Explain why this test case is (or is not) covered by the code. Mention error handling, branching, edge cases, etc.>"
    }}
  ],

  "improvements": [
    {{
      "type": "<code_convention | issue | unhandled_exception | naming | structure | other>",
      "reason": "<Why it's a problem or could be improved>",
      "solution": "<Recommended fix or improvement>"
    }}
  ]
}}

Do not include any explanation outside the JSON.
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

User question or guidance:
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