"""The interface src/agent.py's ReAct loop drives, regardless of what's
actually generating text underneath it. Both methods are shaped after
llama.cpp's own create_chat_completion, since that's the lowest common
denominator every adapter has to normalize to:

  complete(messages=[...], **sampling_params) -> {"choices": [{"message":
      {"content": "..."}}]}
  stream_completion(messages=[...], **sampling_params) -> iterator of
      {"choices": [{"delta": {"content": "..."}}]} chunks

Every concrete provider (llama_cpp_provider.py, openai_compatible_provider.py,
...) implements exactly these two methods plus load()/unload() lifecycle
hooks. src/agent.py never imports llama_cpp or an HTTP client directly — it
only ever calls through this interface.
"""

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    def __init__(self, model_card: dict):
        self.model_card = model_card

    @abstractmethod
    def load(self) -> None:
        """Prepares the provider to serve completions: loads local weights,
        or just validates config/credentials for an API provider."""

    @abstractmethod
    def unload(self) -> None:
        """Releases any resources load() acquired (local model memory, open
        connections). Safe to call even if load() was never called."""

    @abstractmethod
    def complete(self, **kwargs) -> dict:
        """Non-streaming completion, in llama.cpp's create_chat_completion
        response shape: {"choices": [{"message": {"content": str}}]}."""

    @abstractmethod
    def stream_completion(self, messages: list, **sampling_params):
        """Yields chunks shaped like {"choices": [{"delta": {"content": str}}]}
        as they're generated."""

    @property
    def supports_vision(self) -> bool:
        return False

    @property
    def vision_disabled_reason(self):
        return None
