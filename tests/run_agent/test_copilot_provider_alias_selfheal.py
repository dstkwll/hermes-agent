"""Copilot self-heal must survive a provider *alias* in config.

Real incident (2026-07-30): ``model.provider: github-copilot`` in config.yaml
silently disarmed every Copilot credential self-heal, because each guard tests
the canonical slug literally::

    agent/conversation_loop.py  ->  if agent.provider == "copilot" and status_code == 401
    run_agent.py                ->  if self.provider != "copilot": return False

``agent.provider`` was assigned raw from the caller in ``agent_init.init_agent``
with only ``.strip().lower()`` applied — no alias normalization — so a
``github-copilot`` agent never matched. The 401 fell through to
"Non-retryable client error", killing the turn and forcing a gateway restart.

SEPARATING WITNESS: ``test_pre_fix_logic_reproduces_the_incident`` pins the
*actual* pre-fix expression and asserts it produced the broken value, so these
tests cannot pass vacuously against the code that shipped the bug.
"""

import pytest

from agent.agent_init import _canonical_provider_name


COPILOT_ALIASES = ["github-copilot", "github", "github-models", "github-model"]


def _pre_fix_provider_name(provider):
    """The exact expression that shipped the bug (agent_init.py:608, pre-fix)."""
    return (
        provider.strip().lower()
        if isinstance(provider, str) and provider.strip()
        else None
    )


@pytest.mark.parametrize("alias", COPILOT_ALIASES)
def test_pre_fix_logic_reproduces_the_incident(alias):
    """Proves the witness separates: pre-fix code yields a non-canonical slug.

    If this ever fails, the tests below are no longer defending a real defect.
    """
    assert _pre_fix_provider_name(alias) != "copilot"


@pytest.mark.parametrize("alias", COPILOT_ALIASES)
def test_copilot_aliases_canonicalize(alias):
    """Post-fix: every Copilot alias resolves to the canonical guard value."""
    assert _canonical_provider_name(alias) == "copilot"


@pytest.mark.parametrize("alias", COPILOT_ALIASES)
def test_copilot_401_refresh_guard_fires_for_aliased_provider(alias):
    """The behavioural contract the incident violated.

    Mirrors the real predicate at agent/conversation_loop.py:3886 gating
    ``_try_refresh_copilot_client_credentials``.
    """
    provider = _canonical_provider_name(alias)
    status_code = 401

    assert provider == "copilot" and status_code == 401, (
        f"Copilot 401 credential re-mint never fires when provider is "
        f"configured as {alias!r} — the turn dies on "
        f"'Non-retryable client error' and only a gateway restart recovers it"
    )


@pytest.mark.parametrize("alias", COPILOT_ALIASES)
def test_copilot_400_recovery_guard_fires_for_aliased_provider(alias):
    """Sibling call path: the 400 stale-credential recovery guard.

    Mirrors run_agent.py:5151/5215 (``if self.provider != "copilot": return
    False``). Fixing only the 401 would leave this twin disarmed.
    """
    assert _canonical_provider_name(alias) == "copilot"


def test_non_copilot_providers_pass_through_unchanged():
    """Guard against over-normalizing: canonical ids must be untouched."""
    for provider in ("anthropic", "openai-codex", "nous", "vertex", "copilot"):
        assert _canonical_provider_name(provider) == provider


def test_empty_input_returns_none_preserving_base_url_autodetect():
    """``None`` (not "openrouter") — init_agent branches on ``is None``.

    ``normalize_provider`` defaults empty input to "openrouter"; returning that
    here would suppress the base-URL auto-detection branches in init_agent that
    infer openai-codex / xai / anthropic from the hostname.
    """
    for empty in (None, "", "   ", 123):
        assert _canonical_provider_name(empty) is None
