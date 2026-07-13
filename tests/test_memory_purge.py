"""Regression test for the /purge LLM-rewiring gap found in the pre-redesign
audit: clear_memory() used to reassign the module-level `mem` singleton to a
freshly opened store without re-calling attach_llm(), silently degrading
fact-extraction/summarization to 'no LLM' mode until the next model switch.
Fixed by having clear_memory() remember the last-attached complete_fn and
re-wire it onto the new store automatically."""

import os
import tempfile

import pytest


@pytest.fixture
def isolated_memory(monkeypatch, tmp_path):
    """src.memory opens its DB at import time from a fixed path; redirect it
    to a scratch directory for this test so we don't touch the real app's
    memory database, then reload the module fresh."""
    db_dir = str(tmp_path / "memory")
    monkeypatch.setenv("TUFFY_NO_AUTO_MEMORY", "")
    import importlib
    import src.memory as memory_module
    monkeypatch.setattr(memory_module, "DB_DIR", db_dir)
    monkeypatch.setattr(memory_module, "DB_PATH", os.path.join(db_dir, "tuffy.db"))
    os.makedirs(db_dir, exist_ok=True)
    memory_module.mem.close()
    import elastimem
    memory_module.mem = elastimem.open(memory_module.DB_PATH, context_tokens=4096)
    memory_module._last_complete_fn = None
    memory_module._last_model_card = None
    memory_module._last_static_prompt_tokens = None
    yield memory_module
    memory_module.mem.close()


def test_purge_rewires_llm_onto_fresh_store(isolated_memory):
    memory = isolated_memory
    calls = []

    def fake_complete_fn(**kwargs):
        calls.append(kwargs)
        return {"choices": [{"message": {"content": "ok"}}]}

    memory.attach_llm(fake_complete_fn)
    assert memory.mem.complete_fn is not None

    memory.clear_memory()

    assert memory.mem.complete_fn is not None, (
        "clear_memory() must re-attach the LLM onto the new store - "
        "otherwise fact extraction/summarization silently runs with no LLM "
        "until the next model switch"
    )


def test_purge_without_prior_attach_does_not_crash(isolated_memory):
    memory = isolated_memory
    # No attach_llm() call at all (e.g. TUFFY_NO_AUTO_MEMORY=1 at startup) -
    # clear_memory() must tolerate _last_complete_fn being None.
    memory.clear_memory()
    assert memory.mem is not None


def test_purge_reapplies_real_model_context_budget(isolated_memory):
    """Regression test: clear_memory() used to always reopen the fresh store
    at the hardcoded 4096-token default, silently re-budgeting a large-
    context model's memory sections down to a fraction of what they should
    be until the next /models switch happened to fix it. clear_memory() must
    re-derive the real budget from the last-known model card, the same way
    a model switch does."""
    memory = isolated_memory
    large_context_card = {"id": "big-model", "context_length": 131072}
    memory.reconfigure_for_model(large_context_card, static_prompt_tokens=500)
    assert memory.mem.config.context_tokens == 131072

    memory.clear_memory()

    assert memory.mem.config.context_tokens == 131072, (
        "clear_memory() must re-apply the real model's context length, not "
        "silently fall back to the 4096-token default"
    )


def test_purge_without_any_model_card_falls_back_to_default(isolated_memory):
    memory = isolated_memory
    # No reconfigure_for_model() call yet (shouldn't happen in practice -
    # main.py always calls it at startup - but must not crash).
    memory.clear_memory()
    assert memory.mem.config.context_tokens == 4096
