"""Local gguf models via llama.cpp — the provider Tuffy originally shipped
with, now behind the LLMProvider interface. All the load-time quirks (Metal
warmup segfault workaround, native log silencing, vision/mmproj wiring) that
used to live directly in src/agent.py's LocalAgent now live here."""

import ctypes
import os

import llama_cpp
from llama_cpp import Llama
from llama_cpp.llama_chat_format import MTMDChatHandler

from src.llm.base import LLMProvider

_VISION_CAPABILITIES = {"vision", "omni"}

# llama.cpp's mtmd (vision) library logs through its own native callbacks —
# separate from the main llama log — and by default dumps the entire rendered
# prompt (add_text: ...) and clip loader spam straight into the chat output.
# Install a no-op callback on every native logger. The callback object must
# stay referenced at module level or ctypes garbage-collects it and the next
# native log call crashes.
_NULL_LOG_CALLBACK = llama_cpp.llama_log_callback(lambda level, text, user_data: None)


def _silence_native_logs():
    llama_cpp.llama_log_set(_NULL_LOG_CALLBACK, ctypes.c_void_p(0))
    try:
        import llama_cpp.mtmd_cpp as mtmd_cpp
        mtmd_cpp.mtmd_log_set(_NULL_LOG_CALLBACK, ctypes.c_void_p(0))
        mtmd_cpp.mtmd_helper_log_set(_NULL_LOG_CALLBACK, ctypes.c_void_p(0))
    except (ImportError, AttributeError):
        pass  # older llama-cpp-python without mtmd log hooks


class _NoWarmupMTMDChatHandler(MTMDChatHandler):
    """MTMDChatHandler with the mtmd dummy-image warmup disabled.

    llama.cpp's Metal backend segfaults while encoding the oversized
    (1472x1472) warmup image for Qwen3-VL projectors. Real Tuffy images are
    capped at 1024px by src/vision.py and encode fine on GPU, so skipping
    only the warmup keeps the whole vision path on GPU without the crash.
    """

    def _init_mtmd_context(self, llama_model):
        original_default = self._mtmd_cpp.mtmd_context_params_default

        def default_without_warmup():
            params = original_default()
            params.warmup = False
            return params

        self._mtmd_cpp.mtmd_context_params_default = default_without_warmup
        try:
            super()._init_mtmd_context(llama_model)
        finally:
            self._mtmd_cpp.mtmd_context_params_default = original_default


class LlamaCppProvider(LLMProvider):
    def __init__(self, model_card: dict):
        super().__init__(model_card)
        self.sampling_params = model_card["sampling_params"]
        self.llm = None
        self._supports_vision = False
        self._vision_disabled_reason = None

    def load(self) -> None:
        model_card = self.model_card
        if not model_card["load_params"].get("verbose"):
            _silence_native_logs()

        mmproj_path = model_card.get("mmproj_path")
        mmproj_available = bool(mmproj_path) and os.path.exists(mmproj_path)
        self._supports_vision = bool(_VISION_CAPABILITIES & set(model_card["capabilities"])) and mmproj_available

        load_params = dict(model_card["load_params"])
        if self._supports_vision:
            # MTMDChatHandler reads the projector type from the mmproj GGUF
            # itself, so any llama.cpp-supported vision model works without
            # per-model handler code.
            load_params["chat_handler"] = _NoWarmupMTMDChatHandler(
                clip_model_path=mmproj_path,
                verbose=bool(load_params.get("verbose")),
                use_gpu=True,
            )
        self._vision_disabled_reason = None
        if mmproj_path and not mmproj_available:
            self._vision_disabled_reason = (
                f"mmproj file not found at '{mmproj_path}'. Download it and place it "
                "there to enable vision for this model."
            )

        self.llm = Llama(model_path=model_card["path"], **load_params)

    def unload(self) -> None:
        """Frees the llama.cpp context and any vision (mtmd) context so their
        backing memory (weights, KV cache, clip buffers) is released before
        another model is loaded in its place."""
        if self.llm is not None:
            handler = self.llm.chat_handler
            if handler is not None and hasattr(handler, "close"):
                handler.close()
            self.llm.close()
        self.llm = None

    @property
    def supports_vision(self) -> bool:
        return self._supports_vision

    @property
    def vision_disabled_reason(self):
        return self._vision_disabled_reason

    def complete(self, **kwargs) -> dict:
        return self.llm.create_chat_completion(**kwargs)

    def stream_completion(self, messages: list, **sampling_params):
        params = sampling_params or self.sampling_params
        return self.llm.create_chat_completion(messages=messages, stream=True, **params)
