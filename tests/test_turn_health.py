"""TurnHealth: a passive rolling record of recent turn outcomes fed from real
Done/Failed events turn.py already sees. No model calls, no threads - pure
bookkeeping, so it's tested directly with no fixtures beyond the class
itself."""

from src.cli.session import TurnHealth


def test_no_turns_yet():
    health = TurnHealth()
    assert health.summary() == "no turns yet"
    assert health.consecutive_failures() == 0
    assert not health.should_nudge()


def test_all_successes_no_nudge():
    health = TurnHealth()
    for _ in range(5):
        health.record(None)
    assert health.consecutive_failures() == 0
    assert not health.should_nudge()
    assert "5/5" in health.summary()


def test_consecutive_failures_trigger_nudge_at_threshold():
    health = TurnHealth()
    health.record("oom")
    health.record("oom")
    assert not health.should_nudge()
    health.record("provider")
    assert health.should_nudge()
    assert health.consecutive_failures() == 3


def test_a_success_breaks_the_failure_streak():
    health = TurnHealth()
    health.record("oom")
    health.record("oom")
    health.record(None)
    assert health.consecutive_failures() == 0
    assert not health.should_nudge()


def test_reset_clears_history_across_a_model_switch():
    health = TurnHealth()
    health.record("oom")
    health.record("oom")
    health.record("oom")
    assert health.should_nudge()
    health.reset()
    assert health.consecutive_failures() == 0
    assert health.summary() == "no turns yet"


def test_window_is_bounded_not_unbounded_memory():
    health = TurnHealth()
    for _ in range(1000):
        health.record(None)
    assert len(health._outcomes) <= 10
