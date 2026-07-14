"""Unit tests for src.tools.mcp_client's config-shape normalization: every
config shape actually seen in the wild (the real Claude Desktop/Code
"mcpServers" object, Tuffy's own bare-list/"servers" shapes, and a stray
"type": "stdio" field some real-world configs include) must resolve to the
same internal {name, command, args?, env?} dict list."""

from src.tools.mcp_client import _normalize_configs


class TestNormalizeConfigs:
    def test_standard_mcp_servers_object_shape(self):
        """The shape actually used by Claude Desktop/Code, Cursor, and
        virtually every MCP server's own README - a name-keyed object, not
        a list with a 'name' field inside each entry."""
        data = {
            "mcpServers": {
                "filesystem": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]},
                "github": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-github"], "env": {"TOKEN": "x"}},
            }
        }
        result = _normalize_configs(data)
        by_name = {c["name"]: c for c in result}
        assert set(by_name) == {"filesystem", "github"}
        assert by_name["filesystem"]["command"] == "npx"
        assert by_name["github"]["env"] == {"TOKEN": "x"}

    def test_type_stdio_field_is_preserved_not_required(self):
        """Some real-world configs (including ones copied from Anthropic's
        own docs) include "type": "stdio" per server - accepted and simply
        carried through (stdio is the only transport supported), never
        required."""
        data = {"mcpServers": {"svc": {"type": "stdio", "command": "python", "args": ["main.py"]}}}
        result = _normalize_configs(data)
        assert result == [{"name": "svc", "type": "stdio", "command": "python", "args": ["main.py"]}]

    def test_bare_list_shape(self):
        data = [{"name": "svc", "command": "npx", "args": ["-y", "svc"]}]
        assert _normalize_configs(data) == data

    def test_servers_key_shape(self):
        data = {"servers": [{"name": "svc", "command": "npx"}]}
        assert _normalize_configs(data) == [{"name": "svc", "command": "npx"}]

    def test_empty_dict_yields_empty_list(self):
        assert _normalize_configs({}) == []

    def test_mcp_servers_with_non_dict_entry_is_skipped(self):
        data = {"mcpServers": {"svc": "not-a-dict"}}
        assert _normalize_configs(data) == []
