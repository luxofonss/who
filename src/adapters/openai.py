import os
import time
from typing import List, Optional, Dict, Any
from loguru import logger

try:
    from openai import OpenAI
    from openai.types.chat import ChatCompletion
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from langchain_openai import ChatOpenAI
    from langchain.schema import HumanMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


class OpenAIAdapter:
    """Direct OpenAI API adapter"""
    
    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.1, api_key: Optional[str] = None):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not installed. Run: pip install openai")
        
        self.model_name = model_name
        self.temperature = temperature
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter")
        
        self.client = OpenAI(api_key=self.api_key)
        logger.info(f"Initialized OpenAI adapter with model: {model_name}")
    
    def invoke(self, prompt: str, system_message: Optional[str] = None) -> str:
        """Invoke OpenAI model with a prompt"""
        try:
            messages = []
            
            if system_message:
                messages.append({"role": "system", "content": system_message})
            
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=4000
            )
            
            # Add null check for response content
            content = response.choices[0].message.content
            if content is None:
                logger.warning("OpenAI API returned empty response")
                return ""
            
            return content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def invoke_with_retry(self, prompt: str, system_message: Optional[str] = None, max_retries: int = 3) -> str:
        """Invoke OpenAI model with retry logic"""
        for attempt in range(max_retries):
            try:
                return self.invoke(prompt, system_message)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                logger.warning(f"OpenAI API attempt {attempt + 1} failed: {e}, retrying...")
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            "provider": "openai",
            "model": self.model_name,
            "temperature": self.temperature
        }


class LangChainOpenAI:
    """LangChain-compatible OpenAI adapter"""
    
    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.1, api_key: Optional[str] = None):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain OpenAI library not installed. Run: pip install langchain-openai")
        
        self.model_name = model_name
        self.temperature = temperature
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter")
        
        # Fix: Use 'model' parameter instead of deprecated 'model_name'
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            api_key=self.api_key  # Use 'api_key' instead of 'openai_api_key'
        )
        logger.info(f"Initialized LangChain OpenAI adapter with model: {model_name}")
    
    def invoke(self, prompt: str, system_message: Optional[str] = None) -> str:
        """Invoke LangChain OpenAI model with a prompt"""
        try:
            messages = []
            
            if system_message:
                messages.append(SystemMessage(content=system_message))
            
            messages.append(HumanMessage(content=prompt))
            
            response = self.llm.invoke(messages)
            
            # Add null check for response content
            if response.content is None:
                logger.warning("LangChain OpenAI returned empty response")
                return ""
            
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"LangChain OpenAI error: {e}")
            raise
    
    def invoke_with_retry(self, prompt: str, system_message: Optional[str] = None, max_retries: int = 3) -> str:
        """Invoke LangChain OpenAI model with retry logic"""
        for attempt in range(max_retries):
            try:
                return self.invoke(prompt, system_message)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                logger.warning(f"LangChain OpenAI attempt {attempt + 1} failed: {e}, retrying...")
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            "provider": "openai",
            "model": self.model_name,
            "temperature": self.temperature
        }
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """LangChain compatibility method - required for some LangChain integrations"""
        return self.invoke(prompt)


# Factory function to create the appropriate adapter
def create_openai_adapter(use_langchain: bool = False, **kwargs) -> Any:
    """Factory function to create OpenAI adapter based on preference"""
    if use_langchain:
        return LangChainOpenAI(**kwargs)
    else:
        return OpenAIAdapter(**kwargs)
