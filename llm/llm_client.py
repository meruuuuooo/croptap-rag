"""
LLM Client Module

Client for Ollama local LLM using OpenAI-compatible API.
"""

import sys
from pathlib import Path
from typing import Optional

from openai import OpenAI
from loguru import logger

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings


class LLMClient:
    """
    Ollama LLM client using OpenAI-compatible API.
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        """
        Initialize the LLM client.
        
        Args:
            base_url: Ollama API base URL.
            model: Model name to use.
        """
        self.base_url = base_url or settings.ollama_base_url
        self.model = model or settings.llm_model
        
        # Ollama doesn't need an API key, but OpenAI client requires one
        self.client = OpenAI(
            base_url=self.base_url,
            api_key="ollama"  # Dummy key, Ollama ignores this
        )
        
        logger.info(f"Initialized Ollama client: {self.base_url}, model: {self.model}")
    
    def generate(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate a response using the LLM.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
            
        Returns:
            Generated response text.
        """
        temperature = temperature if temperature is not None else settings.llm_temperature
        max_tokens = max_tokens or settings.llm_max_tokens
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            content = response.choices[0].message.content
            
            logger.debug(
                f"Generated response: {len(content)} chars"
            )
            
            return content
            
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            
            # Check if Ollama is running
            if "Connection refused" in str(e) or "connect" in str(e).lower():
                return (
                    "Error: Cannot connect to Ollama. "
                    "Please ensure Ollama is running with: `ollama serve`"
                )
            
            return f"Error generating response: {str(e)}"
    
    def generate_with_context(
        self,
        system_prompt: str,
        user_message: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate a response with system and user messages.
        
        Args:
            system_prompt: System instruction.
            user_message: User query.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
            
        Returns:
            Generated response text.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        return self.generate(messages, temperature, max_tokens)
    
    def is_configured(self) -> bool:
        """Check if the client is properly configured."""
        try:
            # Quick test to see if Ollama is reachable
            self.client.models.list()
            return True
        except Exception:
            return False
    
    def list_models(self) -> list[str]:
        """List available Ollama models."""
        try:
            response = self.client.models.list()
            return [model.id for model in response.data]
        except Exception as e:
            logger.warning(f"Could not list models: {e}")
            return []


# Singleton instance
_llm_client_instance: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Get or create singleton LLM client instance."""
    global _llm_client_instance
    if _llm_client_instance is None:
        _llm_client_instance = LLMClient()
    return _llm_client_instance
