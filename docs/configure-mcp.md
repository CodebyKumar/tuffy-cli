# Configuring MCP Servers

Tuffy can connect to any [Model Context Protocol](https://modelcontextprotocol.io) server over
stdio and use its tools exactly like a native one — same registry, same dispatch path, no code
changes required. See [src/tools/mcp_client.py](../src/tools/mcp_client.py) for the
implementation.

## Fastest path: `/mcp add <github-url>`

For a server published as an npm package (has a `package.json` with a `bin` entry) or a Python
package (has a `pyproject.toml` with a `[project.scripts]` entry point) — which covers most MCP
servers on GitHub — run inside Tuffy:

```
/mcp add https://github.com/modelcontextprotocol/servers-filesystem
/mcp add https://github.com/owner/repo my-custom-name
```

This fetches the repo's manifest from its default branch, infers the `npx -y <package>` or
`uvx <package>` launch command, appends it to `.tuffy/mcp.json`, and connects it immediately —
no restart needed, no hand-written JSON. See [src/tools/mcp_install.py](../src/tools/mcp_install.py)
for the resolution logic.

It deliberately never clones the repo or runs its build/install scripts — only the two
well-known, registry-published launch shapes above are supported. If a repo doesn't fit either
(a server that needs a local build, a non-standard entry point, ...), `/mcp add` fails with an
explanation and you fall back to the manual steps below.

## Removing a server: `/mcp remove <name>`

```
/mcp remove filesystem
```

Undoes everything `/mcp add` did, in one step: terminates the server's subprocess if it's
currently connected, unregisters its tools from the live registry immediately (the model stops
seeing them this turn, not just after a restart), and removes its entry from `.tuffy/mcp.json`.
Works even if only one of those is true — e.g. a server that's in the config but failed to
connect this session still gets its config entry removed. Reports "not found" if the name
matches neither a config entry nor a live connection.

## 1. Create the config file (manual path)

Create `.tuffy/mcp.json` (it's gitignored — safe to put secrets in it). Tuffy accepts the same
`"mcpServers"` shape used by Claude Desktop, Claude Code, Cursor, and virtually every MCP
server's own README "add to your config" snippet — copy one straight in, no translation needed:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allowed/directory"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_..."
      }
    }
  }
}
```

Each server's key is its name (prefixes every tool it registers, e.g. `filesystem_read_file`);
the value is:

| Field | Required | Meaning |
|---|---|---|
| `command` | yes | Executable to launch the server (`npx`, `python3`, a compiled binary, ...). |
| `args` | no | Argument list passed to `command`. |
| `env` | no | Extra environment variables merged into the subprocess's environment (for API tokens, etc). |
| `type` | no | Accepted and ignored if present (some configs include `"type": "stdio"`) — stdio is the only transport this client supports. |

`args` values are passed straight through to `command` with no path resolution by Tuffy itself —
a relative path (e.g. `"agent_workspace"` for `server-filesystem`) is resolved by the launched
process against whatever directory Tuffy itself was started from, which is always this repo's
root for both entry points (`main.py` and `tuffy-ui`'s backend). Never hardcode an absolute path
containing your own home directory or username here — it won't work on anyone else's machine (or
your own, after a home directory rename); use a path relative to this repo's root instead.

Two older shapes are also still accepted, for configs already written against them or written by
`/mcp add`: a bare list, or `{"servers": [...]}`, of `{name, command, args?, env?}` dicts (`name`
as a field inside each entry rather than the object key). See
[src/tools/mcp_client.py](../src/tools/mcp_client.py)'s `_normalize_configs` for the exact
resolution logic — a malformed entry is skipped with a printed warning, never a hard failure.

## 2. Start Tuffy

Tuffy connects to every configured server at startup, before the first system prompt is built.
A server that fails to connect (bad command, missing binary, crashed on init) is skipped with a
`[mcp] Failed to connect to server '<name>': ...` warning — it never blocks startup or takes
down the rest of the session. A server that connects successfully prints one line to Tuffy's own
terminal (`[mcp] Connected to '<name>' (N tool(s)).`) as part of the normal startup summary.

The server's *own* stderr output (startup banners, its own internal logging/warnings — not
Tuffy's) is not shown in the terminal at all; it's redirected to `logs/mcp_servers.log`
(gitignored) instead, since it isn't Tuffy's own output and would otherwise clutter every
startup. Check that file if a server connects but seems to be misbehaving and you need its own
logs to debug why.

## 3. Verify it worked

Run `/mcp` inside Tuffy to see every connected server and the tools it registered. Run `/tools`
to see them alongside native tools, under a `MCP: <server-name>` header.

## Where a skill's own MCP server fits in

A skill (see [configure-skills.md](configure-skills.md)) can ship its own `mcp.json` — one
server config, same shape as an entry above — inside its folder. Tuffy merges every loaded
skill's `mcp.json` into the same connection list at startup, so a skill can bring its own MCP
server without the user hand-editing `.tuffy/mcp.json`.

## Notes

- Every MCP server's tools are visible to the model on every turn, same as native tools — there
  is no per-turn filtering or "enable this server for this task" mode.
- Two servers can't collide: tool names are always registered as `<server-name>_<tool-name>`.
- Connections are stdio-only for now. A server config is essentially "how do I launch this
  process," so anything conforming to MCP's stdio transport works.
