"""run_shell's allowlist rejection must raise ToolExecutionError (a failed
Observation the model is told to react to), not return a plain string that
tool_dispatch would treat as a successful result indistinguishable from real
command output. Also: run_python's output must never leak the resolved
absolute workspace path (e.g. inside a script's own traceback) back to the
model - a live session showed the model copying an echoed absolute path
from a traceback into its NEXT run_python call, which then got rejected."""

import os

import pytest

from src.engine.errors import ToolExecutionError
from src.tools import coding, editing
from src.tools.coding import run_python, run_shell


def test_disallowed_command_raises_tool_execution_error():
    with pytest.raises(ToolExecutionError) as exc_info:
        run_shell("git status")
    assert "git" in str(exc_info.value)
    assert "not allowed" in str(exc_info.value)


def test_allowed_command_still_returns_plain_string():
    result = run_shell("pwd")
    assert isinstance(result, str)
    assert result.startswith("exit code")


class TestRunPythonOutputSanitization:
    @pytest.fixture(autouse=True)
    def isolated_workspace(self, tmp_path, monkeypatch):
        workspace = str(tmp_path / "agent_workspace")
        monkeypatch.setattr(editing, "WORKSPACE_DIR", workspace)
        monkeypatch.setattr(coding, "WORKSPACE_DIR", workspace)

    def test_traceback_absolute_path_is_stripped_to_bare_filename(self):
        editing.save_to_file("broken.py", "def f(:\n    pass\n")  # syntax error
        result = run_python("broken.py")
        assert os.path.abspath(editing.WORKSPACE_DIR) not in result
        assert "broken.py" in result

    def test_successful_run_output_unaffected(self):
        editing.save_to_file("hello.py", "print('hi')")
        result = run_python("hello.py")
        assert "exit code 0" in result
        assert "hi" in result
