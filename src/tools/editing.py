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
        # Small models reliably hallucinate a plausible-looking absolute
        # path ('/home/user/scratch/foo.py') despite the tool description
        # saying "bare filename" - observed in practice to burn multiple
        # retry hops before landing on the right form. Naming the exact
        # corrected value inline (not just describing the rule) gives the
        # model something to literally copy on retry instead of guessing
        # again.
        suggestion = os.path.basename(filename.rstrip("/\\")) or "file.txt"
        raise ValueError(
            f"Filename '{filename}' escapes the workspace directory — use a bare filename with "
            f"no directory, e.g. '{suggestion}'."
        )
    return candidate


@registry.register(
    name="save_to_file",
    description="Write text to a named file in the local workspace, creating or fully overwriting it. For a small change to a file that already exists, use edit_file instead so the rest of the file isn't lost.",
    parameters={
        "filename": {"type": "string", "description": "A bare filename only, e.g. 'notes.txt' or 'summary.md' — no leading slash, no directory (not '/home/...', not 'scratch/...'). The workspace is a flat folder you don't otherwise see or choose a path within."},
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
        # Echo back the bare filename the model gave, never the resolved
        # absolute path - observed in practice to contaminate the model's
        # NEXT tool call, which would copy the full echoed path back in
        # (e.g. run_python) and trip the same workspace-escape rejection
        # save_to_file itself just succeeded past.
        return f"Successfully wrote content to '{filename}'."
    except Exception as e:
        return f"File operation failed: {str(e)}"


@registry.register(
    name="read_file",
    description="Read the full contents of a file already saved in the local workspace.",
    parameters={"filename": {"type": "string", "description": "A bare filename only, e.g. 'notes.txt' — no leading slash, no directory. Use list_workspace_files first if unsure what's there."}},
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
    description="List every file currently saved in the local workspace.",
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
    description="Replace one exact occurrence of existing text in a workspace file, without rewriting the rest. Use this instead of save_to_file for a small fix to a file that already exists. Fails if old_text isn't found in the file, or is found more than once.",
    parameters={
        "filename": {"type": "string", "description": "A bare filename only, e.g. 'app.py' — no leading slash, no directory."},
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
        # Same reasoning as save_to_file above: echo the bare filename, not
        # the resolved absolute path, so it can't leak into the model's
        # next tool call.
        return f"Successfully edited '{filename}'."
    except Exception as e:
        return f"File operation failed: {str(e)}"
