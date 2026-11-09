"""rooroosu local patch: a named multiplex host never serves the root home."""
from unittest import mock

from hermes_cli import profiles


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
