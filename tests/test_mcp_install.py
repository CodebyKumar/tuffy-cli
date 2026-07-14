"""Unit tests for src.tools.mcp_install: GitHub URL parsing and npm/Python
package-manifest resolution for `/mcp add`. HTTP calls are mocked — no live
network access in the test suite."""

import json
from unittest.mock import patch, MagicMock

import pytest

from src.tools import mcp_install
from src.tools.mcp_install import (
    MCPInstallError,
    _parse_github_url,
    resolve_server_config,
    append_server_config,
    remove_server_config,
    load_config_entries,
)


class TestParseGithubUrl:
    def test_https_url(self):
        assert _parse_github_url("https://github.com/owner/repo") == ("owner", "repo")

    def test_url_with_git_suffix(self):
        assert _parse_github_url("https://github.com/owner/repo.git") == ("owner", "repo")

    def test_url_with_trailing_slash(self):
        assert _parse_github_url("https://github.com/owner/repo/") == ("owner", "repo")

    def test_bare_host_path(self):
        assert _parse_github_url("github.com/owner/repo") == ("owner", "repo")

    def test_non_github_url_raises(self):
        with pytest.raises(MCPInstallError):
            _parse_github_url("https://gitlab.com/owner/repo")

    def test_garbage_input_raises(self):
        with pytest.raises(MCPInstallError):
            _parse_github_url("not a url at all")


def _mock_response(status_code=200, json_body=None, text_body=""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text_body if text_body else (json.dumps(json_body) if json_body is not None else "")
    resp.json = MagicMock(return_value=json_body or {})
    return resp


class TestResolveServerConfig:
    @patch("src.tools.mcp_install.requests.get")
    def test_resolves_npm_package_with_bin(self, mock_get):
        def side_effect(url, timeout):
            if "api.github.com" in url:
                return _mock_response(json_body={"default_branch": "main"})
            if "package.json" in url:
                return _mock_response(json_body={"name": "@scope/my-mcp-server", "bin": {"my-mcp-server": "cli.js"}})
            return _mock_response(status_code=404)

        mock_get.side_effect = side_effect
        config = resolve_server_config("https://github.com/owner/repo")
        assert config == {
            "command": "npx",
            "args": ["-y", "@scope/my-mcp-server"],
            "name": "repo",
        }

    @patch("src.tools.mcp_install.requests.get")
    def test_custom_name_overrides_repo_name(self, mock_get):
        def side_effect(url, timeout):
            if "api.github.com" in url:
                return _mock_response(json_body={"default_branch": "main"})
            if "package.json" in url:
                return _mock_response(json_body={"name": "pkg", "bin": {"pkg": "cli.js"}})
            return _mock_response(status_code=404)

        mock_get.side_effect = side_effect
        config = resolve_server_config("https://github.com/owner/repo", name="custom")
        assert config["name"] == "custom"

    @patch("src.tools.mcp_install.requests.get")
    def test_falls_back_to_python_package_when_no_npm_bin(self, mock_get):
        pyproject = """
[project]
name = "my-python-mcp"
version = "0.1.0"

[project.scripts]
my-python-mcp = "my_python_mcp:main"
"""

        def side_effect(url, timeout):
            if "api.github.com" in url:
                return _mock_response(json_body={"default_branch": "main"})
            if "package.json" in url:
                return _mock_response(status_code=404)
            if "pyproject.toml" in url:
                return _mock_response(text_body=pyproject)
            return _mock_response(status_code=404)

        mock_get.side_effect = side_effect
        config = resolve_server_config("https://github.com/owner/repo")
        assert config == {"command": "uvx", "args": ["my-python-mcp"], "name": "repo"}

    @patch("src.tools.mcp_install.requests.get")
    def test_no_resolvable_manifest_raises_with_actionable_message(self, mock_get):
        def side_effect(url, timeout):
            if "api.github.com" in url:
                return _mock_response(json_body={"default_branch": "main"})
            return _mock_response(status_code=404)

        mock_get.side_effect = side_effect
        with pytest.raises(MCPInstallError, match="Couldn't automatically resolve"):
            resolve_server_config("https://github.com/owner/repo")

    @patch("src.tools.mcp_install.requests.get")
    def test_npm_package_without_bin_is_not_treated_as_launchable(self, mock_get):
        def side_effect(url, timeout):
            if "api.github.com" in url:
                return _mock_response(json_body={"default_branch": "main"})
            if "package.json" in url:
                return _mock_response(json_body={"name": "just-a-library"})
            return _mock_response(status_code=404)

        mock_get.side_effect = side_effect
        with pytest.raises(MCPInstallError):
            resolve_server_config("https://github.com/owner/repo")


class TestAppendServerConfig:
    def test_appends_new_entry(self, tmp_path, monkeypatch):
        config_path = tmp_path / ".tuffy" / "mcp.json"
        monkeypatch.setattr(mcp_install, "MCP_CONFIG_PATH", str(config_path))

        append_server_config({"name": "svc", "command": "npx", "args": ["-y", "svc"]})
        entries = load_config_entries()
        assert entries == [{"name": "svc", "command": "npx", "args": ["-y", "svc"]}]

    def test_duplicate_name_raises_without_modifying_file(self, tmp_path, monkeypatch):
        config_path = tmp_path / ".tuffy" / "mcp.json"
        monkeypatch.setattr(mcp_install, "MCP_CONFIG_PATH", str(config_path))

        append_server_config({"name": "svc", "command": "npx", "args": ["-y", "svc"]})
        with pytest.raises(MCPInstallError, match="already"):
            append_server_config({"name": "svc", "command": "npx", "args": ["-y", "other"]})
        assert len(load_config_entries()) == 1


class TestRemoveServerConfig:
    def test_removes_matching_entry(self, tmp_path, monkeypatch):
        config_path = tmp_path / ".tuffy" / "mcp.json"
        monkeypatch.setattr(mcp_install, "MCP_CONFIG_PATH", str(config_path))

        append_server_config({"name": "svc1", "command": "npx", "args": ["-y", "svc1"]})
        append_server_config({"name": "svc2", "command": "npx", "args": ["-y", "svc2"]})

        assert remove_server_config("svc1") is True
        remaining = load_config_entries()
        assert len(remaining) == 1
        assert remaining[0]["name"] == "svc2"

    def test_removing_unknown_name_returns_false_and_leaves_file_untouched(self, tmp_path, monkeypatch):
        config_path = tmp_path / ".tuffy" / "mcp.json"
        monkeypatch.setattr(mcp_install, "MCP_CONFIG_PATH", str(config_path))

        append_server_config({"name": "svc1", "command": "npx", "args": ["-y", "svc1"]})
        assert remove_server_config("nope") is False
        assert len(load_config_entries()) == 1

    def test_removing_from_missing_file_returns_false(self, tmp_path, monkeypatch):
        config_path = tmp_path / ".tuffy" / "mcp.json"
        monkeypatch.setattr(mcp_install, "MCP_CONFIG_PATH", str(config_path))
        assert remove_server_config("svc1") is False
