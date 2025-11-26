"""
Custom LLM integration using OpenAI-compatible API.

This module provides a callable function to interact with a custom LLM
via an OpenAI-compatible API endpoint, similar to the Ollama integration.

Please, modify the settings in `fastapi_app/settings.py` to configure the custom LLM.
Make sure the custom LLM supports OpenAI-compatible API calls. 
Edit as needed for specific model parameters.
"""

import os
from typing import Optional, Callable
from openai import OpenAI

from fastapi_app.settings import settings as set

# Load settings from app settings
LLM_API_KEY = set.custom_llm_api_key
LLM_BASE_URL = set.custom_llm_base_url
LLM_MODEL = set.custom_llm_model


def _create_client(api_key: str = LLM_API_KEY, base_url: str = LLM_BASE_URL) -> OpenAI:
    # Construct an OpenAI client that talks to the custom LLM's OpenAI-compatible endpoint
    return OpenAI(base_url=base_url, api_key=api_key)

def get_llm_callable(model: Optional[str] = None, timeout: int = 30) -> Callable[[str], str]:
    """
    Return a tiny callable llm_fn(prompt) -> str that uses the custom LLM via OpenAI-compatible API.
    Usage:
        llm_fn = get_llm_callable()
        synthesize_answer(..., llm_fn=llm_fn)
    """
    client = _create_client()
    chosen_model = model or LLM_MODEL

    def llm_fn(prompt: str, max_tokens: int, temp: float = 0.0) -> str:
        resp = client.chat.completions.create(
            model=chosen_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temp,
            max_tokens=max_tokens,
            timeout=timeout,
        )

        return resp.choices[0].message.content
       
    return llm_fn
