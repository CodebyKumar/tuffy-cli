"""save_to_file/edit_file must never echo the resolved absolute workspace
path back to the model - only the bare filename it was given. A live session
showed the model copying an echoed absolute path into its NEXT tool call,
which then got rejected by safe_workspace_path's escape check, burning
retry hops on a problem the tool's own success message caused."""

import os

import pytest

from src.tools import editing


@pytest.fixture
def isolated_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(editing, "WORKSPACE_DIR", str(tmp_path / "agent_workspace"))
    yield


class TestSaveToFile:
    def test_success_message_echoes_bare_filename_not_absolute_path(self, isolated_workspace):
        result = editing.save_to_file("triangle.py", "print('*')")
        assert "triangle.py" in result
        assert str(editing.WORKSPACE_DIR) not in result
        assert not os.path.isabs(result.split("'")[1]) if "'" in result else True

    def test_actually_writes_the_file(self, isolated_workspace):
        editing.save_to_file("notes.txt", "hello")
        with open(os.path.join(editing.WORKSPACE_DIR, "notes.txt")) as f:
            assert f.read() == "hello"


class TestEditFile:
    def test_success_message_echoes_bare_filename_not_absolute_path(self, isolated_workspace):
        editing.save_to_file("app.py", "old_line")
        result = editing.edit_file("app.py", "old_line", "new_line")
        assert "app.py" in result
        assert str(editing.WORKSPACE_DIR) not in result


class TestSafeWorkspacePath:
    def test_absolute_path_rejected_with_bare_filename_suggestion(self, isolated_workspace):
        with pytest.raises(ValueError) as exc_info:
            editing.safe_workspace_path("/home/user/scratch/triangle.py")
        message = str(exc_info.value)
        assert "escapes the workspace" in message
        assert "triangle.py" in message  # the corrected suggestion is named inline

    def test_bare_filename_accepted(self, isolated_workspace):
        path = editing.safe_workspace_path("script.py")
        assert path.endswith("script.py")

    def test_path_traversal_rejected(self, isolated_workspace):
        with pytest.raises(ValueError):
            editing.safe_workspace_path("../../etc/passwd")
