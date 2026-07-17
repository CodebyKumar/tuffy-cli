"""Model/asset download-and-cache locations for voice components.

Cached under ~/.tuffy/ to be shared across CLI and UI interfaces.
"""

from pathlib import Path

WHISPER_MODELS_DIR = Path.home() / ".tuffy" / "models" / "whisper"
PIPER_MODELS_DIR = Path.home() / ".tuffy" / "models" / "piper"

WHISPER_MODELS_DIR.mkdir(parents=True, exist_ok=True)
PIPER_MODELS_DIR.mkdir(parents=True, exist_ok=True)


def ensure_whisper_model(model_name: str) -> Path:
    """Verifies that the requested Whisper model exists locally, or downloads it.

    Uses pywhispercpp's built-in download utility.
    """
    from pywhispercpp.utils import download_model

    expected = WHISPER_MODELS_DIR / f"ggml-{model_name}.bin"
    if expected.exists():
        return expected
    
    # pywhispercpp returns the downloaded file path
    path = download_model(model_name, download_dir=str(WHISPER_MODELS_DIR))
    return Path(path)


def ensure_piper_voice(voice_id: str) -> tuple[Path, Path]:
    """Downloads a Piper voice (.onnx + .onnx.json) if not already cached."""
    from piper.download_voices import download_voice

    onnx_path = PIPER_MODELS_DIR / f"{voice_id}.onnx"
    json_path = PIPER_MODELS_DIR / f"{voice_id}.onnx.json"
    if not (onnx_path.exists() and json_path.exists()):
        download_voice(voice_id, PIPER_MODELS_DIR)
    return onnx_path, json_path
