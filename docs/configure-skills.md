# Configuring Skills

A skill is a droppable folder that teaches the agent how to approach a kind of task — pure
guidance, or guidance plus its own tools/MCP server — without touching core code. See
[src/skills/loader.py](../src/skills/loader.py) for the discovery mechanism.

## 1. Create the folder

```
skills/
  my-skill/
    SKILL.md    required
    tools.py    optional
    mcp.json    optional
```

## 2. Write SKILL.md

```markdown
---
name: my-skill
description: One line — what this is for and when the model should reach for it.
---

Guidance body: step-by-step instructions for how to approach this kind of task. This is only
fetched on demand (via the `read_skill` tool) when the model decides the description matches
what the user's asking — it is NOT inlined into every system prompt.
```

`name` and `description` are both required in the frontmatter — a `SKILL.md` missing a
description is skipped at startup with a printed warning. Keep the description to one line; it's
what's shown in `/skills` and in the system prompt for every other skill you have installed, so
a wordy description costs prompt budget on every turn regardless of whether this skill gets used.

## 3. (Optional) Add tools.py

Plain functions decorated with `@registry.register(...)`, exactly like
[src/tools/*.py](../src/tools/README.md) — same decorator, same registry, auto-imported at
startup:

```python
from src.tools.registry import registry

@registry.register(
    name="my_skill_helper",
    description="What this does.",
    parameters={"arg": {"type": "string", "description": "..."}},
    required=["arg"],
    group="docs",
)
def my_skill_helper(arg: str) -> str:
    ...
```

## 4. (Optional) Add mcp.json

A single MCP server config (same shape as an entry in `mcp_servers.json` — see
[configure-mcp.md](configure-mcp.md)) that this skill wants connected. Tuffy merges it into the
MCP client's server list at startup automatically.

## 5. Restart and verify

Run `/skills` to confirm it's listed with the right description. If it shipped tools, run
`/tools` and check they appear (group `docs` unless the tool specified otherwise). Ask something
that should trigger the skill and confirm the model calls `read_skill` before acting on it.

## Example

[skills/code-review/](../skills/code-review/) is a minimal working example: guidance-only, no
`tools.py` or `mcp.json`.
