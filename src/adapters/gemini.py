from __future__ import annotations

"""Google Gemini adapter using the official *google-generativeai* SDK.

Set the environment variable `GOOGLE_API_KEY` with your Gemini API key before
running the server:

    export GOOGLE_API_KEY="your-secret-key"

The model used is *gemini-1.5-flash* (fast & cost-effective).  Calls are
synchronous – they run in a background thread when used inside FastAPI so the
event-loop is not blocked.
"""

import os
import json
from typing import Any, Dict

import google.generativeai as genai  # type: ignore
from loguru import logger


class Gemini:  # pylint: disable=too-few-public-methods
    """Thin wrapper around the Gemini SDK for a single shot prompt."""

    def __init__(
        self,
        *,
        model_name: str = "gemini-1.5-flash",
        temperature: float = 0.1,
    ) -> None:
        api_key = os.getenv("GOOGLE_API_KEY") or "AIzaSyAEV2gHMQpwA3IUznSHucUofFXjIgu80jk"
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY environment variable is not set")

        # Configure the client once.  Idempotent.
        genai.configure(api_key=api_key)

        self._model = genai.GenerativeModel(model_name)
        self._generation_cfg: Dict[str, Any] = {
            "temperature": temperature,
        }
        self.model_name = model_name
        logger.info("Gemini adapter initialised (model=%s, temperature=%.2f)", model_name, temperature)

    # ------------------------------------------------------------------
    def invoke(self, prompt: str) -> str:
        """Send *prompt* to Gemini and return the text response.

        The prompt is logged at DEBUG level (up to 2 000 chars) to make tracing
        easy but avoid overwhelming the logs.
        """
        logger.debug(f"Gemini prompt (first 100000 chars): {prompt[:100000]}")

        try:
            response = self._model.generate_content(
                [prompt],
                generation_config=self._generation_cfg,
            )
            text: str = response.text or ""
            logger.debug(f"Gemini raw response: {text[:10000]}")
            return text
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Gemini API call failed: {exc}")
            # Bubble up so caller can handle / return error JSON.
            raise 