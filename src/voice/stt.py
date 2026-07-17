import os
import contextlib
import numpy as np
import sys
import tempfile
import subprocess

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


def save_pcm_to_wav(pcm: np.ndarray, wav_path: str):
    """Saves float32 mono PCM audio to a 16kHz mono 16-bit WAV file."""
    # Convert float32 [-1.0, 1.0] to int16
    int16_data = (pcm * 32767.0).clip(-32768, 32767).astype(np.int16)
    raw_bytes = int16_data.tobytes()
    
    num_channels = 1
    sample_rate = 16000
    bytes_per_sample = 2
    byte_rate = sample_rate * num_channels * bytes_per_sample
    block_align = num_channels * bytes_per_sample
    data_size = len(raw_bytes)
    
    header = bytearray(44)
    header[0:4] = b"RIFF"
    header[4:8] = (36 + data_size).to_bytes(4, "little")
    header[8:12] = b"WAVE"
    header[12:16] = b"fmt "
    header[16:20] = (16).to_bytes(4, "little")  # Subchunk1Size
    header[20:22] = (1).to_bytes(2, "little")   # AudioFormat (PCM)
    header[22:24] = (num_channels).to_bytes(2, "little")
    header[24:28] = (sample_rate).to_bytes(4, "little")
    header[28:32] = (byte_rate).to_bytes(4, "little")
    header[32:34] = (block_align).to_bytes(2, "little")
    header[34:36] = (bytes_per_sample * 8).to_bytes(2, "little")
    header[36:40] = b"data"
    header[40:44] = (data_size).to_bytes(4, "little")
    
    with open(wav_path, "wb") as f:
        f.write(header)
        f.write(raw_bytes)


class WhisperSTT:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        # Pre-flight check / download of model
        ensure_whisper_model(model_name)
        self._model_name = model_name
        self._models_dir = WHISPER_MODELS_DIR

    def transcribe(self, pcm: np.ndarray) -> str:
        """pcm: float32 mono audio at 16kHz (whisper.cpp's native rate)."""
        if pcm is None or len(pcm) == 0:
            return ""
            
        fd, temp_wav = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        
        try:
            save_pcm_to_wav(pcm, temp_wav)
            
            # Run the transcription in a helper subprocess
            python_exe = sys.executable
            script_dir = os.path.dirname(os.path.abspath(__file__))
            helper_script = os.path.join(script_dir, "transcribe_helper.py")
            
            cmd = [
                python_exe,
                helper_script,
                temp_wav,
                self._model_name,
                str(self._models_dir)
            ]
            
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except Exception as e:
            print(f"STT subprocess error: {e}", file=sys.stderr)
            return ""
        finally:
            try:
                os.remove(temp_wav)
            except OSError:
                pass
