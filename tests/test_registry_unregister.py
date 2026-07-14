"""Unit test for ToolRegistry.unregister_group: the live-registry half of
`/mcp remove` — a disconnected server's tools must stop being visible to the
model in the same session, not just on the next restart."""

import pytest

from src.tools.registry import registry


@pytest.fixture
def clean_registry():
    saved_functions = dict(registry.functions)
    saved_schemas = list(registry.schemas)
    saved_groups = dict(registry.groups)
    yield registry
    registry.functions = saved_functions
    registry.schemas = saved_schemas
    registry.groups = saved_groups


def _register(name, group):
    registry.register(
        name=name,
        description="test tool",
        parameters={},
        required=[],
        group=group,
    )(lambda: "ok")


class TestUnregisterGroup:
    def test_removes_only_matching_group(self, clean_registry):
        _register("scheduler_list_tasks", "mcp:scheduler")
        _register("scheduler_add_task", "mcp:scheduler")
        _register("everything_echo", "mcp:everything")

        removed = registry.unregister_group("mcp:scheduler")

        assert set(removed) == {"scheduler_list_tasks", "scheduler_add_task"}
        assert "scheduler_list_tasks" not in registry.functions
        assert "scheduler_add_task" not in registry.functions
        assert "everything_echo" in registry.functions
        assert not any(s["function"]["name"].startswith("scheduler_") for s in registry.schemas)

    def test_unknown_group_returns_empty_list_and_changes_nothing(self, clean_registry):
        _register("everything_echo", "mcp:everything")
        before = dict(registry.functions)

        removed = registry.unregister_group("mcp:does-not-exist")

        assert removed == []
        assert registry.functions == before
