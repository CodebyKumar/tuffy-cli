# src/tools/

Native tool implementations the agent can call, organized by domain. Importing `src.tools`
registers every module below as a side effect (see [__init__.py](__init__.py)) — nothing else
needs to change to make a new tool visible to the model.

- [registry.py](registry.py) - `ToolRegistry`: the `@registry.register(...)` decorator, schema
  bookkeeping, and `GROUP_ORDER`/`GROUP_TITLES` (the canonical list of tool groups and their
  display headers used by `/tools` and the system prompt). No tools of its own.
- [editing.py](editing.py) - `save_to_file`/`read_file`/`list_workspace_files`/`edit_file`, plus
  `safe_workspace_path()` — the sandbox boundary every workspace-touching tool (including
  `coding.py`) resolves paths through. Group: `editing`.
- [coding.py](coding.py) - `run_python`, `run_shell` (fixed command allowlist, not a blocklist),
  `git_status`/`git_diff`/`git_commit` — all scoped to the workspace directory with a short
  timeout. Group: `coding`.
- [research.py](research.py) - `web_search`, `get_datetime`, `translate`. Group: `research`.
- [system.py](system.py) - `get_system_stats`, `top_processes`, `capture_image`, `view_image`.
  Group: `system`.
- [mcp_client.py](mcp_client.py) - bridges the async-only MCP Python SDK into this synchronous
  registry: reads `./mcp_servers.json` (gitignored — see
  [docs/configure-mcp.md](../../docs/configure-mcp.md) for the config shape) plus each loaded
  skill's `mcp.json`, connects over stdio, and registers each remote tool as `<server>_<tool>`
  under group `mcp:<server>`. A server that fails to connect is skipped with a warning rather
  than blocking startup.

(`src/memory.py`'s `remember` tool and `src/skills/__init__.py`'s `read_skill` tool register
themselves directly rather than living in this package — `main.py` already imports both modules
for other reasons.)

## Adding a new tool

See [docs/configure-tools.md](../../docs/configure-tools.md) for the full guide. In short: pick
the module matching its domain (or add a new one), write a plain function, decorate it:

```python
from src.tools.registry import registry

@registry.register(
    name="my_tool",
    description="What this does and when the model should call it.",
    parameters={"arg": {"type": "string", "description": "..."}},
    required=["arg"],
    group="research",  # or editing/coding/system/docs/memory/mcp:<server>
)
def my_tool(arg: str) -> str:
    ...
```

That's it — no registration list to update elsewhere. If it's a new module, add
`import src.tools.<module>  # noqa: F401` to `__init__.py`.

## Design notes

- **All tools are always visible to the model, every turn.** There's no runtime mode-switching
  or filtering; `group` is metadata only, used for `/tools` output and system-prompt section
  headers, not for hiding tools from the model.
- **The workspace sandbox is a real boundary, not decorative.** `safe_workspace_path()` in
  `editing.py` rejects path traversal; every file-touching tool (including `coding.py`'s
  `run_python`/`run_shell`) resolves through it, so nothing here can touch files outside
  `agent_workspace/` without the user explicitly widening that scope in code.
- **`run_shell` is allowlisted, not blocklisted.** An unrecognized command name fails safe by
  default rather than needing to be added to a "don't run this" list.
