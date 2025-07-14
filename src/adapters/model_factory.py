"""
Model factory for creating LLM instances based on configuration
"""

import os
from typing import Optional
from loguru import logger

from .gemini import Gemini, LangChainGemini
from .claude import Claude, LangChainClaude
from .grok import Grok, LangChainGrok


class ModelFactory:
    """Factory for creating LLM instances based on environment configuration."""
    
    @staticmethod
    def get_llm_provider() -> str:
        """Get the configured LLM provider from environment variables."""
        return os.getenv("LLM_PROVIDER", "gemini").lower()
    
    @staticmethod
    def get_model_name() -> str:
        """Get the configured model name from environment variables."""
        provider = ModelFactory.get_llm_provider()
        
        if provider == "claude":
            return os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022")
        elif provider == "grok":
            return os.getenv("GROK_MODEL", "grok-beta")
        else:  # gemini
            return os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    
    @staticmethod
    def get_temperature() -> float:
        """Get the configured temperature from environment variables."""
        return float(os.getenv("LLM_TEMPERATURE", "0.1"))
    
    @staticmethod
    def create_llm(model_name: str, api_key: str, temperature: Optional[float] = 0.1):
        """Create a single-shot LLM instance based on configuration or provided args."""
        if not model_name:
            model_name = ModelFactory.get_model_name()
        if temperature is None:
            temperature = ModelFactory.get_temperature()
        if model_name.startswith("claude"):
            return Claude(model_name=model_name, temperature=temperature, api_key=api_key)
        elif model_name.startswith("grok"):
            return Grok(model_name=model_name, temperature=temperature, api_key=api_key)
        elif model_name.startswith("gemini"):
            return Gemini(model_name=model_name, temperature=temperature, api_key=api_key)
        else:  # gemini (default)
            raise ValueError("Invalid model name")
    
    @staticmethod
    def create_langchain_llm(model_name: str, api_key: str, temperature: Optional[float] = 0.1):
        """Create a LangChain-compatible LLM instance based on configuration or provided args."""
        if not model_name:
            model_name = ModelFactory.get_model_name()
        if temperature is None:
            temperature = ModelFactory.get_temperature()
        if model_name.startswith("claude"):
            return LangChainClaude(model_name=model_name, temperature=temperature, api_key=api_key)
        elif model_name.startswith("grok"):
            return LangChainGrok(model_name=model_name, temperature=temperature, api_key=api_key)
        elif model_name.startswith("gemini"):
            return LangChainGemini(model_name=model_name, temperature=temperature, api_key=api_key)
        else:  # gemini (default)
            raise ValueError("Invalid model name")