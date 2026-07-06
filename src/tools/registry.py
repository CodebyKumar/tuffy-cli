"""Tool registration infrastructure: the decorator each tool module uses to
expose Python functions to the model as callable tools, plus the schema
bookkeeping needed to build the system prompt and validate tool calls.

The system prompt shows the model one compact signature line per tool
(see tool_lines()) instead of a wall of few-shot examples — the ReAct
protocol examples live in src/prompts/templates.py and are generic, so the
model can't parrot tool-specific example values as if they were real input.
"""

import json


# Display order for known groups in tool_lines()/cmd_tools(); anything not
# listed here (e.g. a future "mcp:<server-name>" group) is appended after,
# in first-seen order.
GROUP_ORDER = ["editing", "coding", "docs", "research", "system", "memory"]
GROUP_TITLES = {
    "editing": "Editing & Files",
    "coding": "Coding & Execution",
    "docs": "Documentation",
    "research": "Research & Lookup",
    "system": "System",
    "memory": "Memory",
}


def group_title(group: str) -> str:
    """Display header for a group name, including the dynamic 'mcp:<server>'
    groups mcp_client.py registers tools under."""
    if group.startswith("mcp:"):
        return f"MCP: {group[len('mcp:'):]}"
    return GROUP_TITLES.get(group, group.title())


class ToolRegistry:
    def __init__(self):
        self.functions = {}
        self.schemas = []
        self.groups = {}  # tool name -> group

    def register(self, name: str, description: str, parameters: dict, required: list, group: str = "general"):
        """parameters: dict of {param_name: {"type": ..., "description": ...}}
        group: category used to organize /tools output and the system prompt
        (e.g. 'editing', 'coding', 'docs', 'research', 'system', 'memory')."""
        def decorator(func):
            self.functions[name] = func
            self.groups[name] = group
            schema = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": parameters,
                        "required": required
                    }
                }
            }
            self.schemas.append(schema)
            return func
        return decorator

    def required_args(self, name: str) -> list:
        for schema in self.schemas:
            if schema["function"]["name"] == name:
                return schema["function"]["parameters"]["required"]
        return []

    def _grouped_schemas(self) -> list[tuple[str, list[dict]]]:
        """Schemas bucketed by group, ordered per GROUP_ORDER then first-seen
        for any unlisted group."""
        buckets = {}
        for schema in self.schemas:
            group = self.groups.get(schema["function"]["name"], "general")
            buckets.setdefault(group, []).append(schema)

        ordered_groups = [g for g in GROUP_ORDER if g in buckets]
        ordered_groups += [g for g in buckets if g not in ordered_groups]
        return [(g, buckets[g]) for g in ordered_groups]

    @staticmethod
    def _tool_line(schema: dict) -> str:
        fn = schema["function"]
        params = fn["parameters"]["properties"]
        required = fn["parameters"]["required"]
        sig_parts = [
            f"{pname}{'' if pname in required else '?'}"
            for pname in params
        ]
        arg_docs = "; ".join(
            f"{pname}: {pdef.get('description', '')}" for pname, pdef in params.items()
        )
        line = f"- {fn['name']}({', '.join(sig_parts)}) — {fn['description']}"
        if arg_docs:
            line += f" [{arg_docs}]"
        return line

    def tool_lines(self) -> list[str]:
        """Prompt-ready lines, grouped under '== Title ==' headers so the
        model gets a domain signal alongside each tool's signature. Optional
        arguments are marked with '?'."""
        lines = []
        for group, schemas in self._grouped_schemas():
            title = group_title(group)
            lines.append(f"== {title} ==")
            lines.extend(self._tool_line(s) for s in schemas)
        return lines

    def tools_by_group(self) -> list[tuple[str, list[dict]]]:
        """Public grouped view for CLI display (e.g. /tools)."""
        return self._grouped_schemas()


registry = ToolRegistry()
