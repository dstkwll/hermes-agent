"""Rehydrating a persisted /model override must re-resolve api_mode for the MODEL.

Real incident (2026-07-30): a Discord thread pinned to ``gpt-5.6-sol`` via
``/model`` failed every turn, across gateway restarts, with::

    HTTP 400: model "gpt-5.6-sol" is not accessible via the
              /chat/completions endpoint          (unsupported_api_for_model)

Copilot serves GPT-5.x ONLY on the Responses API. The resolver knows this —
``_copilot_runtime_api_mode`` derives ``codex_responses`` correctly *when it is
told which model it is resolving for*.

The defect is in the rehydration path. ``api_mode`` is deliberately NOT
persisted (see ``sanitize_model_override`` — only model/provider/base_url are
written to disk) and is re-resolved on first use after a restart. But
``_rehydrate_session_model_override`` (gateway/run.py) re-resolved it for the
**provider alone**::

    runtime = _resolve_runtime_agent_kwargs_for_provider(provider)
    override["api_mode"] = runtime.get("api_mode")

``resolve_runtime_provider`` then fell back to ``model_cfg["default"]`` — the
*global* default (a Claude model → ``chat_completions``) — and that stale mode
was written onto an override whose model is ``gpt-5.6-sol``. The persisted
model was sitting right there in the same function and was never passed.

Net effect: the override survived the restart but was rehydrated with an
api_mode belonging to a different model, permanently bricking any session
pinned to a Responses-only model. A gateway restart could not clear it —
restarting is what re-triggered the bad rehydration.

SEPARATING WITNESS: ``test_pre_fix_call_shape_reproduces_the_incident`` pins
the exact pre-fix call (provider only) and asserts it yields the WRONG mode,
so these tests cannot pass vacuously against the code that shipped the bug.
"""

from __future__ import annotations

import pytest

import hermes_cli.runtime_provider as rp


COPILOT_RESPONSES_ONLY_MODELS = [
    "gpt-5.6-sol",
    "gpt-5.6-sol-pro",
    "gpt-5.6-terra",
    "gpt-5.5",
]


@pytest.fixture
def fake_resolve(monkeypatch):
    """Stub the network-touching Copilot api_mode probe with the documented rule."""

    def _fake(model_id, *, api_key=None, catalog=None):
        m = (model_id or "").strip().lower()
        if m.startswith("gpt-5") and not m.startswith("gpt-5-mini"):
            return "codex_responses"
        return "chat_completions"

    import hermes_cli.models as models

    monkeypatch.setattr(models, "copilot_model_api_mode", _fake)


def _api_mode(model_cfg, *, target_model):
    return rp._copilot_runtime_api_mode(model_cfg, "tok", target_model=target_model)


# Global default is a Claude model — chat_completions. This is the stale value
# the buggy path leaked onto GPT-5.x overrides.
CLAUDE_DEFAULT_CFG = {"default": "claude-opus-5", "provider": "copilot"}


@pytest.mark.parametrize("model", COPILOT_RESPONSES_ONLY_MODELS)
def test_pre_fix_call_shape_reproduces_the_incident(fake_resolve, model):
    """Proves the witness separates: resolving WITHOUT the model yields the bug.

    This is the exact shape of the shipped rehydration call — provider only,
    no target_model — and it must produce the wrong mode for a Responses-only
    model. If this ever stops being true, the tests below no longer defend a
    real defect.
    """
    assert _api_mode(CLAUDE_DEFAULT_CFG, target_model=None) == "chat_completions"


@pytest.mark.parametrize("model", COPILOT_RESPONSES_ONLY_MODELS)
def test_rehydrated_override_resolves_api_mode_for_its_own_model(fake_resolve, model):
    """The contract the incident violated.

    A persisted override pinned to a Responses-only model must rehydrate with
    ``codex_responses``, regardless of what the global default model is.
    """
    assert _api_mode(CLAUDE_DEFAULT_CFG, target_model=model) == "codex_responses", (
        f"override pinned to {model!r} rehydrated with the global default's "
        f"api_mode; every turn in that session 400s with "
        f"unsupported_api_for_model and a restart cannot clear it"
    )


def test_non_responses_model_still_resolves_chat_completions(fake_resolve):
    """Guard against over-correcting: Claude/Gemini overrides stay on chat.

    The inverse leak is equally real — a GPT-5 *default* must not push
    codex_responses onto a Claude override.
    """
    gpt_default_cfg = {"default": "gpt-5.6-sol", "provider": "copilot"}
    assert _api_mode(gpt_default_cfg, target_model="claude-opus-5") == "chat_completions"


def test_resolve_runtime_provider_accepts_target_model():
    """The plumbing the fix depends on must exist with that exact keyword.

    ``_rehydrate_session_model_override`` reaches api_mode via
    ``_resolve_runtime_agent_kwargs_for_provider`` ->
    ``resolve_runtime_provider``; the fix threads the persisted model through
    both. Pin the parameter name so a refactor can't silently drop it and
    revert the behaviour to the provider-only shape.
    """
    import inspect

    sig = inspect.signature(rp.resolve_runtime_provider)
    assert "target_model" in sig.parameters

    from gateway.run import _resolve_runtime_agent_kwargs_for_provider

    gw_sig = inspect.signature(_resolve_runtime_agent_kwargs_for_provider)
    assert "target_model" in gw_sig.parameters, (
        "the gateway helper must accept target_model, or the rehydration path "
        "has no way to tell the resolver which model it is resolving for"
    )
