import os
import contextlib
import numpy as np
from pywhispercpp.model import Model

from src.voice.assets import WHISPER_MODELS_DIR, ensure_whisper_model

DEFAULT_MODEL = "small.en"


@contextlib.contextmanager
def suppress_stdout_stderr():
    """Suppresses all console outputs, including low-level C library writes."""
    try:
        null_fd = os.open(os.devnull, os.O_RDWR)
        save_stdout = os.dup(1)
        save_stderr = os.dup(2)
        os.dup2(null_fd, 1)
        os.dup2(null_fd, 2)
        try:
            yield
        finally:
            os.dup2(save_stdout, 1)
            os.dup2(save_stderr, 2)
            os.close(save_stdout)
            os.close(save_stderr)
            os.close(null_fd)
    except Exception:
        yield


class WhisperSTT:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        # Pre-flight check / download of model
        ensure_whisper_model(model_name)
        # Load whisper.cpp model silently
        with suppress_stdout_stderr():
            self._model = Model(model=model_name, models_dir=str(WHISPER_MODELS_DIR))

    def transcribe(self, pcm: np.ndarray) -> str:
        """pcm: float32 mono audio at 16kHz (whisper.cpp's native rate)."""
        if pcm is None or len(pcm) == 0:
            return ""
        segments = self._model.transcribe(pcm, print_progress=False)
        return " ".join(segment.text.strip() for segment in segments).strip()
