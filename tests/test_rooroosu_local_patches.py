"""Checks for the two rooroosu local patches carried on top of upstream releases."""
from unittest import mock

import pytest

from hermes_cli import profiles
from tools import delegate_tool


def test_named_multiplex_host_does_not_serve_default(tmp_path):
    with mock.patch.object(profiles, "get_active_profile_name", return_value="luffy"), \
         mock.patch.object(profiles, "_get_default_hermes_home", return_value=tmp_path), \
         mock.patch.object(profiles, "_iter_named_profile_dirs", return_value=[]):
        assert profiles.profiles_to_serve(True) == []
        # Roster callers still see the default home.
        assert [n for n, _ in profiles.profiles_to_serve(True, include_parked=True)] == ["default"]
    with mock.patch.object(profiles, "get_active_profile_name", return_value="default"), \
         mock.patch.object(profiles, "_get_default_hermes_home", return_value=tmp_path), \
         mock.patch.object(profiles, "_iter_named_profile_dirs", return_value=[]):
        assert [n for n, _ in profiles.profiles_to_serve(True)] == ["default"]


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
