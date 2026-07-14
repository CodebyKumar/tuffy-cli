"""LlamaCppProvider.stream_completion: the one place allowed to translate a
bare RuntimeError from llama-cpp-python's C bindings into OutOfMemoryError.
Doesn't load a real model - swaps in a stub for self.llm to control exactly
what create_chat_completion does."""

import threading

import pytest

from src.engine.errors import OutOfMemoryError
from src.llm.llama_cpp_provider import LlamaCppProvider


class _StubLlm:
    def __init__(self, behavior):
        self._behavior = behavior

    def create_chat_completion(self, **kwargs):
        return self._behavior()


def _make_provider(behavior) -> LlamaCppProvider:
    provider = LlamaCppProvider.__new__(LlamaCppProvider)
    provider.sampling_params = {}
    provider.llm = _StubLlm(behavior)
    # __new__ bypasses LLMProvider.__init__, which is where the inference
    # lock normally gets created - set it directly, same as any other
    # attribute this bypass pattern has to fill in by hand.
    provider.inference_lock = threading.Lock()
    return provider


class TestStreamCompletionErrorTranslation:
    def test_decode_runtime_error_becomes_out_of_memory_error(self):
        def behavior():
            raise RuntimeError("llama_decode returned -3")
        provider = _make_provider(behavior)
        with pytest.raises(OutOfMemoryError):
            list(provider.stream_completion([{"role": "user", "content": "hi"}]))

    def test_successful_stream_passes_through_unmodified(self):
        def behavior():
            return iter([{"choices": [{"delta": {"content": "hi"}}]}])
        provider = _make_provider(behavior)
        chunks = list(provider.stream_completion([{"role": "user", "content": "hi"}]))
        assert chunks == [{"choices": [{"delta": {"content": "hi"}}]}]

    def test_non_runtime_error_is_not_translated(self):
        def behavior():
            raise ValueError("something else entirely")
        provider = _make_provider(behavior)
        with pytest.raises(ValueError):
            list(provider.stream_completion([{"role": "user", "content": "hi"}]))
