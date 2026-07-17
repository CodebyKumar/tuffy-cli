"""Dual-driver audio utilities for capturing mic input and playing speaker output.

Compatible with macOS and Linux (including Jetson Orin Nano).
Probes for sounddevice/soundfile libraries, falling back to subprocess calls
to native ALSA utilities (arecord, aplay) on Linux or afplay on macOS.
"""

import os
import sys
import tempfile
import subprocess
import time

try:
    import sounddevice as sd
    import numpy as np
    import soundfile as sf
    HAS_AUDIO_LIBS = True
except ImportError:
    HAS_AUDIO_LIBS = False

_SAMPLE_RATE = 16000  # Whisper STT expected sample rate
C_DIM = "\033[2m"
C_RESET = "\033[0m"


class AudioInterface:
    def __init__(self):
        self.use_fallback = not HAS_AUDIO_LIBS
        if HAS_AUDIO_LIBS:
            try:
                # Test query of default device to confirm PortAudio is working
                sd.query_devices(kind='input')
            except Exception:
                print(f"{C_DIM}[audio] sounddevice initialization failed. Falling back to native system commands.{C_RESET}")
                self.use_fallback = True

    def record_audio(self) -> "np.ndarray":
        """Records mono audio at 16kHz from the microphone until Enter is pressed.

        Returns a float32 NumPy array with range [-1.0, 1.0].
        """
        if not self.use_fallback:
            try:
                return self._record_sounddevice()
            except Exception as e:
                print(f"{C_DIM}[audio] sounddevice record error ({e}). Trying fallback...{C_RESET}")
                self.use_fallback = True

        return self._record_fallback()

    def play_audio(self, pcm_chunks, sample_rate: int, stop_event=None) -> None:
        """Plays mono PCM16 audio chunks.

        If sounddevice is active, streams them in real-time.
        Otherwise, buffers and plays using system commands (aplay/afplay).
        """
        if not self.use_fallback:
            try:
                self._play_sounddevice(pcm_chunks, sample_rate, stop_event)
                return
            except Exception as e:
                print(f"{C_DIM}[audio] sounddevice play error ({e}). Trying fallback...{C_RESET}")
                self.use_fallback = True

        self._play_fallback(pcm_chunks, sample_rate, stop_event)

    def _record_sounddevice(self) -> "np.ndarray":
        chunks = []

        def callback(indata, frames, time_info, status):
            if status:
                print(f"\n[audio status] {status}", file=sys.stderr)
            chunks.append(indata.copy())

        # Open float32 mono stream
        stream = sd.InputStream(samplerate=_SAMPLE_RATE, channels=1, dtype='float32', callback=callback)
        with stream:
            # Wait for user input to stop
            input()

        if not chunks:
            return np.array([], dtype=np.float32)
        return np.concatenate(chunks, axis=0).flatten()

    def _record_fallback(self) -> "np.ndarray":
        import numpy as np
        
        fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        is_mac = sys.platform == "darwin"
        
        # Start recording subprocess
        if is_mac:
            # macOS has no default CLI recorder like arecord. Try 'rec' (from sox) if installed.
            print(f"{C_DIM}[audio] Fallback recording uses 'rec' (Sox). Press Enter to stop.{C_RESET}")
            cmd = ["rec", "-q", "-r", str(_SAMPLE_RATE), "-c", "1", "-b", "16", temp_path]
        else:
            # Linux/Jetson ALSA utility
            print(f"{C_DIM}[audio] Fallback recording uses ALSA 'arecord'. Press Enter to stop.{C_RESET}")
            cmd = ["arecord", "-q", "-f", "S16_LE", "-r", str(_SAMPLE_RATE), "-c", "1", temp_path]

        try:
            proc = subprocess.Popen(cmd)
        except FileNotFoundError:
            if is_mac:
                raise RuntimeError("No recording command available. Please install 'sox' or use sounddevice.")
            else:
                raise RuntimeError("ALSA 'arecord' command not found. Please install alsa-utils.")

        # Wait for Enter to stop recording
        try:
            input()
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                proc.kill()

        # Load file and convert to float32 np.ndarray
        if not os.path.exists(temp_path) or os.path.getsize(temp_path) < 44:
            # WAV header is 44 bytes
            try:
                os.remove(temp_path)
            except OSError:
                pass
            return np.array([], dtype=np.float32)

        try:
            if HAS_AUDIO_LIBS:
                data, sr = sf.read(temp_path, dtype='float32')
                return data
            else:
                # Basic WAV parser if soundfile library is missing
                with open(temp_path, "rb") as f:
                    f.seek(44)  # Skip standard WAV header
                    pcm_data = f.read()
                int16 = np.frombuffer(pcm_data, dtype="<i2")
                return int16.astype(np.float32) / 32768.0
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass

    def _play_sounddevice(self, pcm_chunks, sample_rate: int, stop_event=None) -> None:
        # Piper outputs 16-bit signed PCM mono
        stream = sd.RawOutputStream(samplerate=sample_rate, channels=1, dtype='int16')
        with stream:
            for chunk in pcm_chunks:
                if stop_event and stop_event.is_set():
                    break
                stream.write(chunk)

    def _play_fallback(self, pcm_chunks, sample_rate: int, stop_event=None) -> None:
        import numpy as np

        fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        # Buffer all chunks to write a single WAV file
        full_audio_list = []
        for chunk in pcm_chunks:
            if stop_event and stop_event.is_set():
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
                return
            full_audio_list.append(chunk)

        full_audio = b"".join(full_audio_list)
        if not full_audio:
            try:
                os.remove(temp_path)
            except OSError:
                pass
            return

        # Write simple WAV file manually (so we don't depend on soundfile/sf)
        num_channels = 1
        bytes_per_sample = 2
        byte_rate = sample_rate * num_channels * bytes_per_sample
        block_align = num_channels * bytes_per_sample
        data_size = len(full_audio)
        
        # Build 44-byte WAV header
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

        with open(temp_path, "wb") as f:
            f.write(header)
            f.write(full_audio)

        is_mac = sys.platform == "darwin"
        if is_mac:
            cmd = ["afplay", temp_path]
        else:
            # Linux/Jetson ALSA utility
            cmd = ["aplay", "-q", temp_path]

        try:
            proc = subprocess.Popen(cmd)
            while proc.poll() is None:
                if stop_event and stop_event.is_set():
                    proc.terminate()
                    proc.wait()
                    break
                time.sleep(0.05)
        except Exception as e:
            print(f"{C_DIM}[audio] Playback failed: {e}{C_RESET}", file=sys.stderr)
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass
