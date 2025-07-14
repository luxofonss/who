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
            return os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    
    @staticmethod
    def get_temperature() -> float:
        """Get the configured temperature from environment variables."""
        return float(os.getenv("LLM_TEMPERATURE", "0.1"))
    
    @staticmethod
    def create_llm() -> Gemini | Claude | Grok:
        """Create a single-shot LLM instance based on configuration."""
        provider = ModelFactory.get_llm_provider()
        model_name = ModelFactory.get_model_name()
        temperature = ModelFactory.get_temperature()
        
        logger.info(f"Creating LLM instance: provider={provider}, model={model_name}, temperature={temperature}")
        
        if provider == "claude":
            return Claude(model_name=model_name, temperature=temperature)
        elif provider == "grok":
            return Grok(model_name=model_name, temperature=temperature)
        else:  # gemini (default)
            return Gemini(model_name=model_name, temperature=temperature)
    
    @staticmethod
    def create_langchain_llm() -> LangChainGemini | LangChainClaude | LangChainGrok:
        """Create a LangChain-compatible LLM instance based on configuration."""
        provider = ModelFactory.get_llm_provider()
        model_name = ModelFactory.get_model_name()
        temperature = ModelFactory.get_temperature()
        
        logger.info(f"Creating LangChain LLM instance: provider={provider}, model={model_name}, temperature={temperature}")
        
        if provider == "claude":
            return LangChainClaude(model_name=model_name, temperature=temperature)
        elif provider == "grok":
            return LangChainGrok(model_name=model_name, temperature=temperature)
        else:  # gemini (default)
            return LangChainGemini(model_name=model_name, temperature=temperature) 