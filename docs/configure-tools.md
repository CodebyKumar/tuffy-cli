# Configuring Tools

Native tools are plain Python functions registered into a shared `ToolRegistry`. See
[src/tools/README.md](../src/tools/README.md) for the module layout (`editing.py`, `coding.py`,
`research.py`, `system.py`) and group conventions.

## Add a tool to an existing module

Pick the module matching the tool's domain and add:

```python
from src.tools.registry import registry

@registry.register(
    name="my_tool",
    description="What this does and when the model should call it — be specific, this is the "
                 "only signal the model has for whether to use it.",
    parameters={
        "arg": {"type": "string", "description": "What this argument means."},
    },
    required=["arg"],
    group="research",  # editing / coding / research / system / docs / memory
)
def my_tool(arg: str) -> str:
    return f"did something with {arg}"
```

That's it — no separate registration list. The function becomes callable by the model on the
very next run, and shows up in `/tools` and the system prompt under its group's header
automatically.

## Add a whole new module

1. Create `src/tools/<domain>.py` with your `@registry.register(...)` functions.
2. Add `import src.tools.<domain>  # noqa: F401` to [src/tools/__init__.py](../src/tools/__init__.py).
3. If the group name is new, add it to `GROUP_ORDER`/`GROUP_TITLES` in
   [src/tools/registry.py](../src/tools/registry.py) for a proper display title (otherwise it
   falls back to a title-cased version of the group string).

## Design constraints to keep in mind

- **Every tool is visible to the model every turn.** There's no per-task filtering — `group` is
  organizational metadata only (used in `/tools` and system-prompt headers), not a runtime mode
  switch. Don't rely on a group being "inactive."
- **Stay inside the sandbox.** Anything touching files should resolve paths through
  `safe_workspace_path()` in [editing.py](../src/tools/editing.py) — it rejects path traversal.
  `run_shell`/`run_python` in `coding.py` are also workspace-scoped and time-boxed.
- **Return a string.** Tool outputs are fed back into the conversation as plain text; if you need
  structured data, serialize it yourself (JSON string, etc.) inside the tool.
- **Keep the description model-facing, not developer-facing.** The description is the only thing
  the model sees when deciding whether to call a tool — write it like a docstring aimed at an
  agent that has never read your source code.

## Alternative: don't write a native tool at all

If the capability already exists as an MCP server (GitHub, browser automation, a database
client, ...), connect it instead of writing a Python wrapper — see
[configure-mcp.md](configure-mcp.md). If it's guidance rather than a callable action, a
[skill](configure-skills.md) may be a better fit than a tool.
