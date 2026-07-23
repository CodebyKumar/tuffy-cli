# .tuffy/skills/

Droppable capability packs. Each subfolder here teaches Tuffy how to approach a kind of task —
no core code changes needed. See [src/skills/](../../src/skills/) for the loading mechanism this
folder is scanned by.

```
.tuffy/skills/
  <name>/
    SKILL.md    required — YAML frontmatter (name, description) + a markdown body of guidance
    tools.py    optional — plain @registry.register(...) functions, auto-imported at startup
    mcp.json    optional — one MCP server config this skill wants connected
```

## Adding a skill

1. Create `.tuffy/skills/<name>/SKILL.md` with frontmatter:
   ```yaml
   ---
   name: my-skill
   description: One line — what this is for and when the model should reach for it.
   ---

   Guidance body: when/how to approach this kind of task, step by step.
   ```
   `description` is required — a skill with no description is skipped with a warning at startup.
2. (Optional) Add `tools.py` with `@registry.register(...)` functions, exactly like
   `src/tools/*.py` — same decorator, same registry, zero new mechanism to learn.
3. (Optional) Add `mcp.json` — a single `{name, command, args, env}` server config (same shape
   as `.tuffy/mcp.json` at the repo root) this skill wants connected.
4. Restart Tuffy. Check `/skills` lists it and, if it has one, that its `read_skill` description
   shows up correctly.

## Example

[scratchpad/SKILL.md](scratchpad/SKILL.md) is a working example that also ships a `tools.py`
(`scratch_note`/`scratch_list`/`scratch_clear`), showing how a skill can extend the tool surface.
