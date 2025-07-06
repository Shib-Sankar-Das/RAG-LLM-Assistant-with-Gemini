"""
Custom Gemini LLM wrapper for LangChain integration.
"""

from typing import List, Optional, Any
from langchain.llms.base import LLM
from langchain_google_genai import GoogleGenerativeAI

class GeminiLLM(LLM):
    """Custom Gemini LLM wrapper for LangChain."""
    
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash", **kwargs):
        """Initialize the Gemini LLM wrapper.
        
        Args:
            api_key: Google Gemini API key
            model_name: Name of the Gemini model to use
            **kwargs: Additional arguments for LangChain LLM
        """
        super().__init__(**kwargs)
        # Store private attributes to avoid Pydantic field validation
        self._api_key = api_key
        self._model_name = model_name
        self._llm = GoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
        )

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs
    ) -> str:
        """Call the Gemini model with the given prompt.
        
        Args:
            prompt: The input prompt
            stop: Optional list of stop sequences
            run_manager: Optional run manager for callbacks
            **kwargs: Additional arguments
            
        Returns:
            Generated response from the model
        """
        try:
            # Use invoke method instead of deprecated __call__
            response = self._llm.invoke(prompt, stop=stop, **kwargs)
            return response
        except Exception as e:
            return f"Error generating response: {str(e)}"

    @property
    def _identifying_params(self) -> dict:
        """Return identifying parameters for the model."""
        return {"model_name": self._model_name}

    @property
    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return "gemini"
