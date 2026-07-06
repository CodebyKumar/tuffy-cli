# Tuffy Documentation

Configuration guides for extending and running Tuffy. This folder is meant to double as the
source for a future docs site (e.g. GitHub Pages) — plain markdown, one topic per file, no
build step required to read it today.

- [configure-models.md](configure-models.md) - register a local or API-provider model
- [configure-mcp.md](configure-mcp.md) - connect external MCP servers
- [configure-skills.md](configure-skills.md) - author a skill (guidance + optional tools/MCP)
- [configure-tools.md](configure-tools.md) - write a new native tool
- [cli-reference.md](cli-reference.md) - every slash command, grouped by purpose

For the system design behind these (why tools are grouped, why the LLM provider is an
interface, why skills only inline a one-line description), see
[../ARCHITECTURE.md](../ARCHITECTURE.md). For a map of every source folder, see
[../src/README.md](../src/README.md).
