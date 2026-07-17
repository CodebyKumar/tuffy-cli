"""Piper Text-to-Speech wrapper using piper-tts.
"""

from typing import Iterator
from piper import PiperVoice

from src.voice.assets import ensure_piper_voice

DEFAULT_VOICE = "en_US-lessac-medium"


class PiperTTS:
    def __init__(self, voice_id: str = DEFAULT_VOICE):
        model_path, config_path = ensure_piper_voice(voice_id)
        self._voice = PiperVoice.load(model_path, config_path=config_path)
        self.sample_rate = self._voice.config.sample_rate
        self.sample_width = 2
        self.channels = 1

    def synthesize(self, text: str) -> Iterator[bytes]:
        """Synthesizes text into mono PCM16 bytes, yielding chunks live."""
        if not text.strip():
            return
        for chunk in self._voice.synthesize(text):
            yield chunk.audio_int16_bytes
