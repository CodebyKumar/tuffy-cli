"""Vision pipeline: gets an image (from an existing file path, or a freshly
captured webcam frame) in front of a vision-capable model.

llama.cpp's multimodal support (via Llava15ChatHandler, wired in src/agent.py
for any model whose model card declares 'vision'/'omni' capability) expects
the image as an OpenAI-style {"type": "image_url", "image_url": {"url": ...}}
content block on a *user* message, not as plain tool-output text. So both
tools here don't return the image itself - they encode it to a base64 data
URI and return a sentinel-prefixed string (IMAGE_SENTINEL + real file path +
"\n" + data URI). src/agent.py's _execute_tool_call recognizes the sentinel,
tells the model the real path in the observation (so it never has to guess or
invent one), and attaches the image itself to the next user turn via
LocalAgent.attach_image — instead of feeding either as ordinary tool-output
text.

capture_image is a real tool (registered with the tool registry) so the
model itself can decide to take a photo. Sending an image by typing a path
in the chat is handled directly in main.py's input loop, ahead of the tool
registry, since that's a user action rather than a model tool call - it
calls encode_image_to_data_uri the same way.
"""

import base64
import mimetypes
import os
from datetime import datetime

import cv2
import numpy as np

from src.tools.registry import registry
from src.tools.editing import WORKSPACE_DIR, safe_workspace_path

IMAGES_SUBDIR = "images"  # camera captures live under agent_workspace/images/, not loose in the root

IMAGE_SENTINEL = "__TUFFY_IMAGE__:"

_CAMERA_INDEX = 0
_CAPTURE_WARMUP_FRAMES = 5  # first few frames off a webcam are often dark/unfocused

# Longest image side sent to the model. Qwen3-VL spends roughly one token per
# 32x32 pixel block, so a 1024px-max image costs ~1024 context tokens; a
# full-resolution photo would cost ~4000 and overflow the 4096-token context.
_MAX_IMAGE_DIM = 1024
_JPEG_QUALITY = 90


def encode_image_to_data_uri(path: str) -> str:
    """Reads an image file from disk and returns it as a base64 data: URI,
    downscaling anything larger than _MAX_IMAGE_DIM on its longest side so a
    single image can never overflow the model's context window."""
    if not os.path.isfile(path):
        raise ValueError(f"No such image file: {path}")

    # Read the bytes first: raises a clean PermissionError/OSError for
    # unreadable files (e.g. macOS privacy-protected folders) instead of
    # letting cv2.imread print a warning and silently return None.
    with open(path, "rb") as f:
        raw = f.read()

    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is not None:
        h, w = img.shape[:2]
        scale = _MAX_IMAGE_DIM / max(h, w)
        if scale < 1.0:
            img = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
        ok, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
        if ok:
            encoded = base64.b64encode(jpeg.tobytes()).decode("ascii")
            return f"data:image/jpeg;base64,{encoded}"

    # Fall back to the raw bytes for formats cv2 can't decode.
    mime_type, _ = mimetypes.guess_type(path)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


@registry.register(
    name="capture_image",
    description="Take a photo with the machine's camera right now. Call this IMMEDIATELY whenever the user asks you to take/click/snap a picture, look at them, see them, or check the camera — never reply 'I'm taking a picture' or 'just say the word' in text; the only way to actually take a photo is calling this tool. Not for files already on disk.",
    parameters={},
    required=[],
    group="system",
)
def capture_image(placeholder: str = "") -> str:
    try:
        cam = cv2.VideoCapture(_CAMERA_INDEX)
        if not cam.isOpened():
            return f"Failed to open camera at index {_CAMERA_INDEX}."

        try:
            frame = None
            for _ in range(_CAPTURE_WARMUP_FRAMES):
                ok, frame = cam.read()
                if not ok:
                    return "Failed to read a frame from the camera."
        finally:
            cam.release()

        os.makedirs(os.path.join(WORKSPACE_DIR, IMAGES_SUBDIR), exist_ok=True)
        # Timestamped, not a fixed name: a fixed name silently overwrites the
        # previous capture, which made a second "take another photo" request
        # look like it did nothing.
        filename = os.path.join(IMAGES_SUBDIR, f"capture_{datetime.now():%Y%m%d_%H%M%S}.jpg")
        file_path = safe_workspace_path(filename)
        cv2.imwrite(file_path, frame)

        data_uri = encode_image_to_data_uri(file_path)
        return f"{IMAGE_SENTINEL}{os.path.abspath(file_path)}\n{data_uri}"
    except Exception as e:
        return f"Camera capture failed: {str(e)}"


@registry.register(
    name="view_image",
    description="Load an image file from a filesystem path so you can look at its contents. Use this ONLY when the user typed a file path in their message — pass that exact path, never an invented or remembered one. If an image is already attached to the conversation, just look at it directly; no tool call needed.",
    parameters={"path": {"type": "string", "description": "The exact filesystem path the user gave in their message."}},
    required=["path"],
    group="system",
)
def view_image(path: str) -> str:
    try:
        real_path = os.path.expanduser(path.strip())
        data_uri = encode_image_to_data_uri(real_path)
        return f"{IMAGE_SENTINEL}{os.path.abspath(real_path)}\n{data_uri}"
    except Exception as e:
        return f"Failed to load image: {str(e)}"
