"""Local workspace file I/O: safe path resolution plus the save/read/list tools.
Everything related to reading and writing files in the agent's workspace lives here."""

import os

from src.tools.registry import registry

WORKSPACE_DIR = "./agent_workspace"


def safe_workspace_path(filename: str) -> str:
    """Resolves a user-given filename inside WORKSPACE_DIR, rejecting path traversal."""
    os.makedirs(WORKSPACE_DIR, exist_ok=True)
    candidate = os.path.normpath(os.path.join(WORKSPACE_DIR, filename))
    workspace_abs = os.path.abspath(WORKSPACE_DIR)
    if not os.path.abspath(candidate).startswith(workspace_abs + os.sep) and os.path.abspath(candidate) != workspace_abs:
        raise ValueError(f"Filename '{filename}' escapes the workspace directory.")
    return candidate


@registry.register(
    name="save_to_file",
    description="Write logs, summaries, notes, or structured text content to a named file in the local workspace.",
    parameters={
        "filename": {"type": "string", "description": "Name for the file, e.g. 'notes.txt' or 'summary.md'."},
        "file_content": {"type": "string", "description": "The text content to write to the file."}
    },
    required=["filename", "file_content"]
)
def save_to_file(filename: str, file_content: str) -> str:
    try:
        file_path = safe_workspace_path(filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(file_content)
        return f"Successfully wrote content to {file_path}"
    except Exception as e:
        return f"File operation failed: {str(e)}"


@registry.register(
    name="read_file",
    description="Read back the contents of a file previously saved in the local workspace.",
    parameters={"filename": {"type": "string", "description": "Name of the file to read, e.g. 'notes.txt'."}},
    required=["filename"]
)
def read_file(filename: str) -> str:
    try:
        file_path = safe_workspace_path(filename)
        if not os.path.isfile(file_path):
            return f"No such file in workspace: {filename}"
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"File operation failed: {str(e)}"


@registry.register(
    name="list_workspace_files",
    description="List all files currently saved in the local workspace directory.",
    parameters={},
    required=[]
)
def list_workspace_files(placeholder: str = "") -> str:
    try:
        os.makedirs(WORKSPACE_DIR, exist_ok=True)
        files = sorted(
            os.path.relpath(os.path.join(root, f), WORKSPACE_DIR)
            for root, _, filenames in os.walk(WORKSPACE_DIR)
            for f in filenames
        )
        if not files:
            return "Workspace is empty."
        return "\n".join(files)
    except Exception as e:
        return f"File operation failed: {str(e)}"
