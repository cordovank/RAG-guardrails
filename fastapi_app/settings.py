"""Application settings.

Pydantic moved ``BaseSettings`` into the ``pydantic-settings`` package starting
with pydantic v2.12. Prefer importing from ``pydantic_settings`` but fall back
to the older location so this project works with a range of environments.
"""

try:
    # New location (pydantic v2.12+)
    from pydantic_settings import BaseSettings
except Exception:
    # Fallback for older pydantic versions
    try:
        from pydantic import BaseSettings  # type: ignore
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "BaseSettings is required. Install 'pydantic-settings' or use a compatible pydantic version"
        ) from exc


class Settings(BaseSettings):
    max_retrieved_docs: int = 50
    max_passage_chars: int = 2000
    max_snippet_chars: int = 200
    max_response_length: int = 500
    enable_self_check: bool = True
    llm_choice: str = "ollama"  # Options: "ollama", "custom"

# OLLAMA settings
    default_llm: str = "ollama_llama3.2"
    ollama_api_key: str = "ollama"
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model_llama: str = "llama3.2"

# CUSTOM LLM settings
    custom_llm : str = ""   # Use "" to disable custom LLM
    custom_llm_api_key: str = ""
    custom_llm_model: str = ""
    custom_llm_base_url: str = ""


settings = Settings()