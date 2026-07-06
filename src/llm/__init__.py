"""LLM provider adapters: one interface (base.LLMProvider), one adapter per
backend, so src/agent.py's ReAct loop never talks to llama_cpp or an HTTP API
directly. Add a new provider by writing one *_provider.py implementing
LLMProvider and adding its name to src.models.registry.KNOWN_PROVIDERS."""

from src.llm.base import LLMProvider
from src.llm.llama_cpp_provider import LlamaCppProvider
from src.llm.openai_compatible_provider import OpenAICompatibleProvider

# Maps a model card's "provider" field to the adapter class that loads it.
PROVIDERS = {
    "llama_cpp": LlamaCppProvider,
    "openai_compatible": OpenAICompatibleProvider,
}


def build_provider(model_card: dict) -> LLMProvider:
    provider_name = model_card["provider"]
    if provider_name not in PROVIDERS:
        raise ValueError(f"No provider adapter registered for '{provider_name}'.")
    return PROVIDERS[provider_name](model_card)
