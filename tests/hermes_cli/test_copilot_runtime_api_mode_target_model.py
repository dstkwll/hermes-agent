"""Regression tests for ``_copilot_runtime_api_mode`` honouring target_model.

Bug: when a user runs an explicit one-off model override against Copilot
(``hermes -z … -m gpt-5.6-sol --provider copilot``) while their config pins
``model.api_mode: chat_completions`` (needed for the default Claude model),
the pinned api_mode leaked onto the GPT-5.x override. Copilot serves GPT-5+
ONLY on the Responses API and returns HTTP 400 for ``/chat/completions``:

    HTTP 400: model "gpt-5.6-sol" is not accessible via the
              /chat/completions endpoint

The resolver derived api_mode from the stale ``model_cfg["default"]`` (the
Claude model → chat_completions) instead of the model the caller actually
switched TO. These tests lock in the invariant: an explicit ``target_model``
wins over the persisted default AND over a config api_mode pin, so GPT-5.x
overrides route to ``codex_responses`` while non-GPT-5 models stay on
``chat_completions``.
"""

from __future__ import annotations

import hermes_cli.runtime_provider as rp


def _patch_copilot_api_mode(monkeypatch):
    """Stub the network-touching copilot_model_api_mode with the documented
    Copilot rule: GPT-5+ (except gpt-5-mini) → Responses API, else chat."""

    def _fake(model_id, *, api_key=None, catalog=None):
        m = (model_id or "").strip().lower()
        if m.startswith("gpt-5") and not m.startswith("gpt-5-mini"):
            return "codex_responses"
        return "chat_completions"

    # copilot_model_api_mode is imported lazily inside the function under test.
    import hermes_cli.models as models

    monkeypatch.setattr(models, "copilot_model_api_mode", _fake)


def test_target_model_overrides_stale_config_default(monkeypatch):
    """A GPT-5.x override must route to Responses even when the config default
    is a Claude model pinned to chat_completions."""
    _patch_copilot_api_mode(monkeypatch)
    model_cfg = {
        "default": "claude-opus-4.8",
        "provider": "copilot",
        "api_mode": "chat_completions",
    }
    assert (
        rp._copilot_runtime_api_mode(model_cfg, "tok", target_model="gpt-5.6-sol")
        == "codex_responses"
    )


def test_target_model_non_gpt5_stays_chat_completions(monkeypatch):
    """A non-GPT-5 override keeps chat_completions (Claude/Gemini/gpt-4o)."""
    _patch_copilot_api_mode(monkeypatch)
    model_cfg = {"default": "gpt-5.6-sol", "provider": "copilot"}
    assert (
        rp._copilot_runtime_api_mode(model_cfg, "tok", target_model="claude-sonnet-4.6")
        == "chat_completions"
    )


def test_no_target_model_honours_config_pin(monkeypatch):
    """Without a target override, an explicit config api_mode pin is honoured
    for the configured default (unchanged legacy behaviour)."""
    _patch_copilot_api_mode(monkeypatch)
    model_cfg = {
        "default": "claude-opus-4.8",
        "provider": "copilot",
        "api_mode": "chat_completions",
    }
    assert rp._copilot_runtime_api_mode(model_cfg, "tok") == "chat_completions"


def test_no_target_model_derives_from_default(monkeypatch):
    """Without a pin or target, api_mode derives from the config default."""
    _patch_copilot_api_mode(monkeypatch)
    model_cfg = {"default": "gpt-5.6-luna", "provider": "copilot"}
    assert rp._copilot_runtime_api_mode(model_cfg, "tok") == "codex_responses"


def test_gpt5_mini_override_stays_chat_completions(monkeypatch):
    """gpt-5-mini is the documented Copilot exception — Chat Completions."""
    _patch_copilot_api_mode(monkeypatch)
    model_cfg = {"default": "claude-opus-4.8", "provider": "copilot"}
    assert (
        rp._copilot_runtime_api_mode(model_cfg, "tok", target_model="gpt-5-mini")
        == "chat_completions"
    )
