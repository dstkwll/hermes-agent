"""Copilot 401 refresh budget — a time-driven 401 may legitimately recur.

Regression witness for a real incident (2026-07-29): a long Copilot turn died
with ``HTTP 401: IDE token expired`` and *only a gateway restart* recovered it,
even though the 401 self-heal existed and had already fired once.

Root cause: the 401 recovery branch in ``agent/conversation_loop.py`` was
guarded by a one-shot **boolean** (``copilot_auth_retry_attempted``). Copilot's
exchanged IDE token has a ~30-minute TTL, so a single API attempt whose retry
loop straddles the expiry boundary can see 401 **twice** — the second one finds
the guard spent and aborts the turn as non-retryable.

The distinction that matters: a *model-availability* 400 must stay single-shot
(re-asking for a genuinely unavailable model loops forever), but a *clock-driven*
401 is expected to recur and refreshing again is both safe and correct. So the
401 guard becomes a small bounded **budget** rather than a boolean.

These tests assert the behavioral contract (how many refreshes a turn may spend,
and that the budget is per-attempt), not the specific integer.
"""

from __future__ import annotations

from dataclasses import fields

import pytest

from agent.turn_retry_state import TurnRetryState


def _budget(state: TurnRetryState) -> int:
    """How many 401 refreshes this state still permits, via the public helper."""
    n = 0
    while state.may_refresh_copilot_auth():
        state.record_copilot_auth_refresh()
        n += 1
        if n > 50:  # pragma: no cover - runaway guard
            pytest.fail("copilot 401 budget is unbounded; it must be capped")
    return n


class TestCopilotAuthRefreshBudget:
    def test_allows_more_than_one_refresh_per_attempt(self):
        """THE WITNESS: the old one-shot boolean allowed exactly 1.

        A token that expires mid-attempt yields a second 401; the turn must be
        able to re-mint rather than dying and forcing a gateway restart.
        """
        assert _budget(TurnRetryState()) >= 2

    def test_budget_is_bounded(self):
        """A permanently-bad credential must not spin forever."""
        assert _budget(TurnRetryState()) <= 5

    def test_first_refresh_is_permitted(self):
        assert TurnRetryState().may_refresh_copilot_auth() is True

    def test_budget_is_exhaustible(self):
        s = TurnRetryState()
        _budget(s)
        assert s.may_refresh_copilot_auth() is False

    def test_budget_is_per_state_instance(self):
        """A fresh attempt gets a fresh budget (state is rebuilt per API call)."""
        spent = TurnRetryState()
        _budget(spent)
        assert spent.may_refresh_copilot_auth() is False
        assert TurnRetryState().may_refresh_copilot_auth() is True

    def test_recording_does_not_leak_into_other_guards(self):
        s = TurnRetryState()
        s.record_copilot_auth_refresh()
        # The 400 stale-credential path is a SEPARATE, still-single-shot guard:
        # a model-availability 400 must not inherit the 401's larger budget.
        assert s.copilot_stale_cred_retry_attempted is False
        assert s.anthropic_auth_retry_attempted is False
        assert s.has_retried_429 is False


class TestStaleCredGuardStaysSingleShot:
    """The 400 path keeps one-shot semantics — retrying an unavailable model loops."""

    def test_stale_cred_guard_is_still_a_boolean_one_shot(self):
        s = TurnRetryState()
        assert s.copilot_stale_cred_retry_attempted is False
        s.copilot_stale_cred_retry_attempted = True
        assert s.copilot_stale_cred_retry_attempted is True


class TestStateContract:
    def test_all_guard_booleans_still_default_false(self):
        """The dataclass gained a counter; every *boolean* guard stays False."""
        s = TurnRetryState()
        for name, value in s:
            if isinstance(value, bool):
                assert value is False, f"{name} should default to False"

    def test_counter_field_defaults_to_zero(self):
        """Spend counters start at 0; ``max_*`` ceilings are config and start > 0."""
        s = TurnRetryState()
        ints = [(n, v) for n, v in s if isinstance(v, int) and not isinstance(v, bool)]
        counters = [(n, v) for n, v in ints if not n.startswith("max_")]
        ceilings = [(n, v) for n, v in ints if n.startswith("max_")]
        assert counters, "expected at least one spend counter for the 401 budget"
        for name, value in counters:
            assert value == 0, f"{name} should start at 0"
        for name, value in ceilings:
            assert value > 0, f"{name} is a ceiling and must be positive"

    def test_loop_control_vars_are_not_on_state(self):
        names = {f.name for f in fields(TurnRetryState)}
        for loop_local in ("retry_count", "max_retries", "max_compression_attempts"):
            assert loop_local not in names
