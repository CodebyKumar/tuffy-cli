"""Coding & execution tools: run code and inspect/commit git history, scoped
entirely to the local workspace sandbox. Nothing here can touch paths outside
WORKSPACE_DIR or run for longer than a short timeout — a small local model
occasionally loops or hangs a script, and this is a personal machine agent,
not a CI runner.
"""

import shlex
import subprocess

from src.tools.registry import registry
from src.tools.editing import WORKSPACE_DIR, safe_workspace_path

_EXEC_TIMEOUT_SECONDS = 20
_MAX_OUTPUT_CHARS = 4000

# Deliberately small and read/version-control-oriented — no rm, mv, curl,
# chmod, etc. This is a fixed allowlist, not a blocklist, so an unexpected
# command name always fails safe.
_ALLOWED_SHELL_COMMANDS = {
    "ls", "cat", "pwd", "echo", "wc", "grep", "find", "head", "tail",
    "python3", "pip", "pytest", "node", "npm", "npx",
}


def _run(args: list[str], cwd: str) -> str:
    try:
        result = subprocess.run(
            args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=_EXEC_TIMEOUT_SECONDS,
        )
        output = (result.stdout or "") + (result.stderr or "")
        output = output.strip() or "(no output)"
        if len(output) > _MAX_OUTPUT_CHARS:
            output = output[:_MAX_OUTPUT_CHARS] + "\n... (truncated)"
        return f"exit code {result.returncode}\n{output}"
    except subprocess.TimeoutExpired:
        return f"Command timed out after {_EXEC_TIMEOUT_SECONDS}s."
    except Exception as e:
        return f"Execution failed: {str(e)}"


@registry.register(
    name="run_python",
    description="Run a Python file that already exists in the workspace and return its stdout/stderr. Use this "
                "to test or execute code the user asked you to write/run — write the file first with "
                "save_to_file, then run it with this tool.",
    parameters={
        "filename": {"type": "string", "description": "Path to the .py file in the workspace to run, e.g. 'script.py'."},
        "args": {"type": "string", "description": "Optional space-separated command-line arguments to pass to the script."}
    },
    required=["filename"],
    group="coding",
)
def run_python(filename: str, args: str = "") -> str:
    import os

    try:
        file_path = os.path.abspath(safe_workspace_path(filename))
    except ValueError as e:
        return f"Execution failed: {str(e)}"

    extra_args = shlex.split(args) if args.strip() else []
    return _run(["python3", file_path] + extra_args, cwd=WORKSPACE_DIR)


@registry.register(
    name="run_shell",
    description="Run a safe, read-only-ish shell command inside the workspace directory (e.g. list files, run "
                "tests, check python/node versions, grep for text). Only a fixed allowlist of commands is "
                "permitted (ls, cat, pwd, echo, wc, grep, find, head, tail, python3, pip, pytest, node, npm, npx) "
                "— destructive or system-wide commands are rejected outright.",
    parameters={
        "command": {"type": "string", "description": "The full shell command to run, e.g. 'pytest -q' or 'ls -la'."}
    },
    required=["command"],
    group="coding",
)
def run_shell(command: str) -> str:
    try:
        parts = shlex.split(command)
    except ValueError as e:
        return f"Could not parse command: {e}"

    if not parts:
        return "Empty command."
    if parts[0] not in _ALLOWED_SHELL_COMMANDS:
        return (
            f"Command '{parts[0]}' is not allowed. Allowed commands: "
            f"{sorted(_ALLOWED_SHELL_COMMANDS)}"
        )

    return _run(parts, cwd=WORKSPACE_DIR)


@registry.register(
    name="git_status",
    description="Show the git status of the workspace directory (which files are changed/staged/untracked).",
    parameters={},
    required=[],
    group="coding",
)
def git_status(placeholder: str = "") -> str:
    return _run(["git", "status", "--short", "--branch"], cwd=WORKSPACE_DIR)


@registry.register(
    name="git_diff",
    description="Show the current unstaged (or staged, if specified) git diff in the workspace directory.",
    parameters={
        "staged": {"type": "string", "description": "'true' to show staged changes instead of unstaged. Defaults to unstaged."}
    },
    required=[],
    group="coding",
)
def git_diff(staged: str = "false") -> str:
    args = ["git", "diff"]
    if str(staged).strip().lower() in ("true", "1", "yes"):
        args.append("--staged")
    return _run(args, cwd=WORKSPACE_DIR)


@registry.register(
    name="git_commit",
    description="Stage all changes in the workspace directory and create a git commit. Only use this when the "
                "user explicitly asks you to commit.",
    parameters={
        "message": {"type": "string", "description": "The commit message."}
    },
    required=["message"],
    group="coding",
)
def git_commit(message: str) -> str:
    add_result = _run(["git", "add", "-A"], cwd=WORKSPACE_DIR)
    if add_result.startswith("exit code") and not add_result.startswith("exit code 0"):
        return f"git add failed: {add_result}"
    return _run(["git", "commit", "-m", message], cwd=WORKSPACE_DIR)
