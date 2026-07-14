"""Resolves a GitHub repo URL to an MCP server config (name/command/args) and
writes it into .tuffy/mcp.json, for the `/mcp add <github-url>` CLI command.

Covers the two shapes almost every real-world MCP server on GitHub actually
takes: an npm package (has a package.json with a `bin` entry, launched via
`npx -y <package-name>` straight from the npm registry, no clone needed) or a
Python package (has a pyproject.toml with a console-script entry point,
launched via `uvx <package-name>`). Deliberately does NOT clone the repo and
run its build/install scripts — arbitrary-repo code execution during
"just add a server" is a real risk this stays away from; if a repo doesn't
fit one of those two well-known, registry-published shapes, resolution fails
with a message telling the user to add it to .tuffy/mcp.json by hand instead
(see docs/configure-mcp.md), rather than guessing at a shell command.
"""

import json
import os
import re

import requests

MCP_CONFIG_PATH = "./.tuffy/mcp.json"
_FETCH_TIMEOUT = 10

_GITHUB_URL_RE = re.compile(
    r"github\.com[:/]+(?P<owner>[\w.-]+)/(?P<repo>[\w.-]+?)(?:\.git)?/?$"
)


class MCPInstallError(Exception):
    pass


def _parse_github_url(url: str) -> tuple[str, str]:
    match = _GITHUB_URL_RE.search(url.strip())
    if not match:
        raise MCPInstallError(
            f"'{url}' doesn't look like a GitHub repo URL (expected something like "
            "https://github.com/<owner>/<repo>)."
        )
    return match.group("owner"), match.group("repo")


def _fetch_raw_file(owner: str, repo: str, branch: str, path: str) -> str | None:
    url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
    try:
        resp = requests.get(url, timeout=_FETCH_TIMEOUT)
    except requests.RequestException:
        return None
    if resp.status_code != 200:
        return None
    return resp.text


def _fetch_default_branch(owner: str, repo: str) -> str:
    url = f"https://api.github.com/repos/{owner}/{repo}"
    try:
        resp = requests.get(url, timeout=_FETCH_TIMEOUT)
        if resp.status_code == 200:
            return resp.json().get("default_branch", "main")
    except requests.RequestException:
        pass
    return "main"


def _resolve_npm_package(owner: str, repo: str, branch: str) -> dict | None:
    raw = _fetch_raw_file(owner, repo, branch, "package.json")
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if "bin" not in data:
        # No bin entry means it's not launchable as a standalone CLI via
        # npx - most likely a library, not an MCP server itself.
        return None
    package_name = data.get("name")
    if not package_name:
        return None
    return {"command": "npx", "args": ["-y", package_name]}


def _resolve_python_package(owner: str, repo: str, branch: str) -> dict | None:
    raw = _fetch_raw_file(owner, repo, branch, "pyproject.toml")
    if not raw:
        return None
    name_match = re.search(r'(?m)^\s*name\s*=\s*"([^"]+)"', raw)
    has_scripts = re.search(r"(?m)^\[project\.scripts\]", raw)
    if not name_match or not has_scripts:
        # No console-script entry point means there's nothing for `uvx` to
        # launch as a standalone command.
        return None
    package_name = name_match.group(1)
    return {"command": "uvx", "args": [package_name]}


def resolve_server_config(github_url: str, name: str | None = None) -> dict:
    """Returns a {"name", "command", "args"} dict ready to write into
    mcp.json, or raises MCPInstallError with a message explaining why the
    repo couldn't be resolved automatically."""
    owner, repo = _parse_github_url(github_url)
    branch = _fetch_default_branch(owner, repo)

    resolved = _resolve_npm_package(owner, repo, branch) or _resolve_python_package(owner, repo, branch)
    if not resolved:
        raise MCPInstallError(
            f"Couldn't automatically resolve '{owner}/{repo}' to a launchable MCP server — "
            "it doesn't have an npm package.json with a 'bin' entry or a pyproject.toml with "
            "a [project.scripts] entry point on its default branch. Check the repo's own README "
            "for its exact run command and add it to .tuffy/mcp.json by hand instead — see "
            "docs/configure-mcp.md."
        )

    resolved["name"] = name or repo
    return resolved


def load_config_entries() -> list[dict]:
    if not os.path.isfile(MCP_CONFIG_PATH):
        return []
    with open(MCP_CONFIG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Mirrors src.tools.mcp_client._normalize_configs, kept independent
    # rather than imported to avoid a mcp_install -> mcp_client dependency
    # for what's otherwise a pure config-file concern.
    if isinstance(data, dict) and isinstance(data.get("mcpServers"), dict):
        return [{"name": name, **cfg} for name, cfg in data["mcpServers"].items() if isinstance(cfg, dict)]
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("servers", [])
    return []


def _write_entries(entries: list[dict]) -> None:
    mcp_servers = {e["name"]: {k: v for k, v in e.items() if k != "name"} for e in entries}
    with open(MCP_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump({"mcpServers": mcp_servers}, f, indent=2)
        f.write("\n")


def append_server_config(config: dict) -> None:
    """Adds one server entry to .tuffy/mcp.json, creating the file/directory
    if needed. Writes in the standard {"mcpServers": {"<name>": {...}}}
    shape (Claude Desktop/Code/Cursor's own format) so the file stays
    copy-pasteable into/out of other tools. Raises MCPInstallError if a
    server with the same name is already configured, rather than silently
    duplicating or overwriting it."""
    os.makedirs(os.path.dirname(MCP_CONFIG_PATH), exist_ok=True)
    entries = load_config_entries()
    name = config["name"]
    if any(e.get("name") == name for e in entries):
        raise MCPInstallError(
            f"A server named '{name}' is already in {MCP_CONFIG_PATH}. "
            "Remove it first or pass a different name."
        )
    entries.append(config)
    _write_entries(entries)


def remove_server_config(name: str) -> bool:
    """Removes one server entry from .tuffy/mcp.json by name. Returns False
    (file left untouched) if no server with that name was configured — the
    caller (`/mcp remove`) is responsible for reporting that as 'not
    found' rather than this silently no-op'ing without signal."""
    entries = load_config_entries()
    remaining = [e for e in entries if e.get("name") != name]
    if len(remaining) == len(entries):
        return False
    _write_entries(remaining)
    return True
