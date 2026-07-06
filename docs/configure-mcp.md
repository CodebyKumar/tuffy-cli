# Configuring MCP Servers

Tuffy can connect to any [Model Context Protocol](https://modelcontextprotocol.io) server over
stdio and use its tools exactly like a native one — same registry, same dispatch path, no code
changes required. See [src/tools/mcp_client.py](../src/tools/mcp_client.py) for the
implementation.

## 1. Create the config file

Create `mcp_servers.json` in the repo root (it's gitignored — safe to put secrets in it). It's
either a bare JSON list of server configs, or `{"servers": [...]}`:

```json
[
  {
    "name": "filesystem",
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allowed/directory"]
  },
  {
    "name": "github",
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-github"],
    "env": {
      "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_..."
    }
  }
]
```

Each entry:

| Field | Required | Meaning |
|---|---|---|
| `name` | yes | Short identifier. Prefixes every tool this server registers, e.g. `filesystem_read_file`. |
| `command` | yes | Executable to launch the server (`npx`, `python3`, a compiled binary, ...). |
| `args` | no | Argument list passed to `command`. |
| `env` | no | Extra environment variables merged into the subprocess's environment (for API tokens, etc). |

This is the same shape used by Claude Desktop/Code's MCP config, so you can usually copy an
existing config over directly.

## 2. Start Tuffy

Tuffy connects to every configured server at startup, before the first system prompt is built.
A server that fails to connect (bad command, missing binary, crashed on init) is skipped with a
`[mcp] Failed to connect to server '<name>': ...` warning — it never blocks startup or takes
down the rest of the session.

## 3. Verify it worked

Run `/mcp` inside Tuffy to see every connected server and the tools it registered. Run `/tools`
to see them alongside native tools, under a `MCP: <server-name>` header.

## Where a skill's own MCP server fits in

A skill (see [configure-skills.md](configure-skills.md)) can ship its own `mcp.json` — one
server config, same shape as an entry above — inside its folder. Tuffy merges every loaded
skill's `mcp.json` into the same connection list at startup, so a skill can bring its own MCP
server without the user hand-editing `mcp_servers.json`.

## Notes

- Every MCP server's tools are visible to the model on every turn, same as native tools — there
  is no per-turn filtering or "enable this server for this task" mode.
- Two servers can't collide: tool names are always registered as `<server-name>_<tool-name>`.
- Connections are stdio-only for now. A server config is essentially "how do I launch this
  process," so anything conforming to MCP's stdio transport works.
