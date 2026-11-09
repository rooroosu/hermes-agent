"""rooroosu local patch: delegate_task per-task specialist profile (SOUL only)."""
from unittest import mock

import pytest

from tools import delegate_tool


def test_delegated_profile_loads_soul_and_rejects_unknown(tmp_path):
    (tmp_path / "SOUL.md").write_text("I am Nami.", encoding="utf-8")
    with mock.patch("hermes_cli.profiles.profile_exists", side_effect=lambda n: n == "nami"), \
         mock.patch("hermes_cli.profiles.get_profile_dir", return_value=tmp_path):
        name, soul = delegate_tool._resolve_delegated_profile("nami")
        assert (name, soul) == ("nami", "I am Nami.")
        prefix = delegate_tool._profile_context_prefix(name, soul)
        assert "'nami'" in prefix and "I am Nami." in prefix
        with pytest.raises(ValueError):
            delegate_tool._resolve_delegated_profile("ghost")
    assert delegate_tool._resolve_delegated_profile(None) == (None, None)
    assert delegate_tool._profile_context_prefix(None, None) == ""
    assert "profile" in delegate_tool.DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"]["items"]["properties"]


def test_delegate_task_profile_reaches_child_context_and_is_not_refused(tmp_path, monkeypatch):
    """End to end through delegate_task on the default backend: `profile` must not be refused as an
    undeclared routing parameter, and the child's context must carry the profile SOUL."""
    import json
    import types

    (tmp_path / "SOUL.md").write_text("I am Nami.", encoding="utf-8")
    built = []

    def build(**kwargs):
        built.append(kwargs)
        return types.SimpleNamespace(tool_progress_callback=None, model=kwargs.get("model"))

    monkeypatch.setattr(delegate_tool, "_build_child_preserving_parent_tools", build)
    monkeypatch.setattr(delegate_tool, "_run_single_child", lambda task_index, goal, child=None, parent_agent=None, **kw: {
        "task_index": task_index, "status": "completed", "summary": "ok", "api_calls": 1, "duration_seconds": 0})
    parent = types.SimpleNamespace(session_id="root", model="m", provider="openrouter", tool_progress_callback=None,
                                   _delegate_spinner=None, request_overrides={}, _safe_print=lambda line: None)
    with mock.patch("hermes_cli.profiles.profile_exists", side_effect=lambda n: n == "nami"), \
         mock.patch("hermes_cli.profiles.get_profile_dir", return_value=tmp_path):
        out = json.loads(delegate_tool.delegate_task(
            tasks=[{"goal": "summarize the quick thing please", "context": "ctx", "profile": "nami"}],
            parent_agent=parent))
        assert [r["status"] for r in out["results"]] == ["completed"]
        assert "I am Nami." in built[0]["context"] and built[0]["context"].endswith("ctx")
        err = delegate_tool.delegate_task(tasks=[{"goal": "summarize the quick thing please", "profile": "ghost"}],
                                          parent_agent=parent)
        assert "Unknown delegated profile" in err
