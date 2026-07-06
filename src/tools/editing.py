"""Editing & file tools: safe path resolution plus save/read/list/edit for
files in the agent's workspace. Everything related to reading and writing
files in the agent's workspace lives here."""

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
    description="Write logs, summaries, notes, or structured text content to a named file in the local workspace. Overwrites the whole file — for a small change to an existing file, prefer edit_file.",
    parameters={
        "filename": {"type": "string", "description": "Name for the file, e.g. 'notes.txt' or 'summary.md'."},
        "file_content": {"type": "string", "description": "The text content to write to the file."}
    },
    required=["filename", "file_content"],
    group="editing",
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
    required=["filename"],
    group="editing",
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
    required=[],
    group="editing",
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


@registry.register(
    name="edit_file",
    description="Make a targeted change to an existing workspace file by replacing one exact occurrence of "
                "some existing text with new text, without rewriting the whole file. Use this instead of "
                "save_to_file when you only need to fix or add a small part of a file that already exists. "
                "Fails if old_text isn't found, or is found more than once (be specific enough to be unique).",
    parameters={
        "filename": {"type": "string", "description": "Name of the existing file to edit, e.g. 'app.py'."},
        "old_text": {"type": "string", "description": "The exact existing text to find and replace. Must appear exactly once in the file."},
        "new_text": {"type": "string", "description": "The text to replace old_text with."}
    },
    required=["filename", "old_text", "new_text"],
    group="editing",
)
def edit_file(filename: str, old_text: str, new_text: str) -> str:
    try:
        file_path = safe_workspace_path(filename)
        if not os.path.isfile(file_path):
            return f"No such file in workspace: {filename}"

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        occurrences = content.count(old_text)
        if occurrences == 0:
            return f"old_text not found in {filename} — no changes made."
        if occurrences > 1:
            return f"old_text appears {occurrences} times in {filename} — make it more specific so it matches exactly once."

        content = content.replace(old_text, new_text, 1)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully edited {file_path}."
    except Exception as e:
        return f"File operation failed: {str(e)}"
