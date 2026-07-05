"""Tool registration infrastructure: the decorator each tool module uses to
expose Python functions to the model as callable tools, plus the schema
bookkeeping needed to build the system prompt and validate tool calls.

The system prompt shows the model one compact signature line per tool
(see tool_lines()) instead of a wall of few-shot examples — the ReAct
protocol examples live in src/prompts/templates.py and are generic, so the
model can't parrot tool-specific example values as if they were real input.
"""

import json


class ToolRegistry:
    def __init__(self):
        self.functions = {}
        self.schemas = []

    def register(self, name: str, description: str, parameters: dict, required: list):
        """parameters: dict of {param_name: {"type": ..., "description": ...}}"""
        def decorator(func):
            self.functions[name] = func
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

    def tool_lines(self) -> list[str]:
        """One compact, prompt-ready line per tool: signature plus what each
        argument means. Optional arguments are marked with '?'."""
        lines = []
        for schema in self.schemas:
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
            lines.append(line)
        return lines


registry = ToolRegistry()
