"""Unit tests for TurnRetryState (god-file Phase 1b).

The dataclass holds the inner-retry-loop's one-shot recovery guards + restart
signals. These tests pin its *semantics* — the behavioral guarantee for the loop
itself is the existing recovery-branch tests in tests/run_agent/ which exercise
these fields via `_retry.<flag>`.

Note: this file deliberately does NOT assert an exact field set. It used to, and
that made it a change-detector — adding a legitimate new guard turned the suite
red for no behavioral reason (it broke for the Copilot 400 stale-credential
guard, and again for the 401 refresh budget). The contract that actually matters
is "guards start un-fired and are independently mutable", which is what is
asserted below. The Copilot 401 budget has its own dedicated behavioral tests in
test_copilot_401_refresh_budget.py.

LOCAL PATCH MARKER: turn-retry-state-invariant-not-snapshot
This conversion is part of the local Copilot hardening patch set and is
reinstalled by ~/.hermes/local-patches/copilot_fix_guard.sh after an upstream
update reverts it. See tech/2026-07-05-durable-local-core-patches in the wiki.
"""

from __future__ import annotations

from dataclasses import fields

from agent.turn_retry_state import TurnRetryState


# Guards that must exist for the loop's recovery branches to be reachable at all.
# This is a floor (subset check), not a snapshot — new guards may be added freely.
REQUIRED_GUARDS = {
    "codex_auth_retry_attempted",
    "anthropic_auth_retry_attempted",
    "nous_auth_retry_attempted",
    "nous_paid_entitlement_refresh_attempted",
    # The Copilot 401 guard is a bounded BUDGET, not a one-shot boolean: the
    # exchanged IDE token expires on a ~30-minute clock, so a single long
    # attempt can straddle the boundary and legitimately 401 twice. The old
    # `copilot_auth_retry_attempted` boolean made the second 401 fatal (turn
    # aborted as non-retryable; only a gateway restart recovered). It is
    # replaced by the count/max pair below and is intentionally absent.
    "copilot_auth_refresh_count",
    "max_copilot_auth_refreshes",
    # Contrast: the 400 model-availability guard MUST stay single-shot —
    # re-asking for a model the integrator genuinely lacks would loop forever.
    "copilot_stale_cred_retry_attempted",
    "vertex_auth_retry_attempted",
    "thinking_sig_retry_attempted",
    "image_shrink_retry_attempted",
    "primary_recovery_attempted",
    "has_retried_429",
    "auth_failover_attempted",
    "restart_with_compressed_messages",
    "restart_with_length_continuation",
}




def test_required_guards_present():
    names = {f.name for f in fields(TurnRetryState)}
    missing = REQUIRED_GUARDS - names
    assert not missing, f"recovery branches would be unreachable: {missing}"




def test_guards_are_independently_mutable():
    s = TurnRetryState()
    s.codex_auth_retry_attempted = True
    s.restart_with_compressed_messages = True
    assert s.codex_auth_retry_attempted is True
    assert s.restart_with_compressed_messages is True
    # untouched guards stay False
    assert s.has_retried_429 is False
    assert s.anthropic_auth_retry_attempted is False


def test_copilot_provider_check_accepts_alias_spellings():
    """`/model` and profile configs can leave `github-copilot` / `github` as
    the provider spelling; the recovery gates must not silently skip them."""
    from agent.conversation_loop import _is_copilot_provider
    from run_agent import AIAgent

    class _Agent:
        # Reuse the real single-owner check unbound; only provider/_base_url
        # state is faked.
        _is_copilot_provider = AIAgent._is_copilot_provider
        _is_copilot_url = AIAgent._is_copilot_url

        def __init__(self, provider, base_url=""):
            self.provider = provider
            self._base_url_lower = base_url.lower()

    assert _is_copilot_provider(_Agent("copilot"))
    assert _is_copilot_provider(_Agent("github-copilot"))
    assert _is_copilot_provider(_Agent("GitHub-Copilot"))
    assert _is_copilot_provider(_Agent("github"))
    # URL fallback: unnormalized provider but a Copilot base URL.
    assert _is_copilot_provider(_Agent("custom", "https://api.githubcopilot.com"))
    assert not _is_copilot_provider(_Agent("openrouter", "https://openrouter.ai/api/v1"))

    class _NoMethod:
        provider = "github-copilot"

    # Fallback path when the agent object lacks the method entirely.
    assert _is_copilot_provider(_NoMethod())
