"""Tests for agent/file_safety.py read guards — env file blocking.

Run with:  python -m pytest tests/agent/test_file_safety.py -v
"""

import os
from unittest.mock import patch

import pytest

from agent.file_safety import (
    _BLOCKED_PROJECT_ENV_BASENAMES,
    get_read_block_error,
)


# ---------------------------------------------------------------------------
# Project-local .env file blocking (issue #20734)
# ---------------------------------------------------------------------------


class TestEnvFileReadBlocking:
    """Secret-bearing .env files must be blocked by get_read_block_error."""

    @pytest.mark.parametrize("basename", [
        ".env",
        ".env.local",
        ".env.development",
        ".env.production",
        ".env.test",
        ".env.staging",
        ".envrc",
    ])
    def test_blocked_env_basenames(self, basename):
        """All secret-bearing .env basenames are blocked regardless of directory."""
        path = f"/tmp/project/{basename}"
        error = get_read_block_error(path)
        assert error is not None, f"{basename} should be blocked"
        assert "Access denied" in error
        assert "secret-bearing" in error.lower() or "environment file" in error.lower()

    def test_blocked_env_in_subdirectory(self):
        """Nested .env files are also blocked."""
        error = get_read_block_error("/home/user/app/services/api/.env.production")
        assert error is not None

    @pytest.mark.parametrize("basename", [
        ".ENV",
        ".Env.Local",
        ".ENV.PRODUCTION",
        ".ENVRC",
    ])
    def test_blocked_env_basenames_case_insensitive(self, basename):
        """Secret-bearing .env basenames are blocked regardless of case."""
        error = get_read_block_error(f"/tmp/project/{basename}")
        assert error is not None, f"{basename} should be blocked"
        assert "Access denied" in error
        assert "environment file" in error.lower()

    def test_blocked_env_absolute_path(self):
        """Absolute paths to .env files are blocked."""
        error = get_read_block_error("/opt/myapp/.env")
        assert error is not None

    def test_allowed_env_example(self):
        """"The .env.example file is explicitly allowed — it's documentation, not a secret."""
        error = get_read_block_error("/tmp/project/.env.example")
        assert error is None

    def test_allowed_env_sample(self):
        """Other .env variants like .env.sample are allowed."""
        error = get_read_block_error("/tmp/project/.env.sample")
        assert error is None

    def test_allowed_non_env_files(self):
        """Regular files are not affected by the env guard."""
        for path in ["/tmp/project/config.yaml", "/tmp/project/main.py",
                     "/tmp/project/README.md", "/tmp/project/.gitignore"]:
            error = get_read_block_error(path)
            assert error is None, f"{path} should be allowed"

    def test_allowed_hermes_env(self):
        """Hermes' own .env inside HERMES_HOME is NOT blocked by this rule
        (it's handled by other mechanisms). Only project-local .env is blocked."""
        # Note: hermes internal .env is in ~/.hermes/.env which is NOT a project-local
        # path, but the basename check applies to ANY .env. This is intentional —
        # even ~/.hermes/.env should not be readable via read_file.
        error = get_read_block_error(os.path.expanduser("~/.hermes/.env"))
        assert error is not None

    def test_blocked_set_is_lowercase(self):
        """All entries in the blocked set are lowercase for case-insensitive matching."""
        for name in _BLOCKED_PROJECT_ENV_BASENAMES:
            assert name == name.lower(), f"{name} should be lowercase"


# ---------------------------------------------------------------------------
# Existing cache-file blocking (regression — must still work)
# ---------------------------------------------------------------------------


class TestCacheFileReadBlocking:
    """Internal Hermes cache files must remain blocked."""

    def test_hub_index_cache_blocked(self, tmp_path):
        """Hub index-cache reads are blocked."""
        hermes_home = tmp_path / ".hermes"
        cache = hermes_home / "skills" / ".hub" / "index-cache" / "data.json"
        cache.parent.mkdir(parents=True)
        cache.write_text("{}")

        with patch("agent.file_safety._hermes_home_path", return_value=hermes_home):
            error = get_read_block_error(str(cache))
            assert error is not None
            assert "internal Hermes cache" in error

    def test_hub_directory_blocked(self, tmp_path):
        """Hub directory reads are blocked."""
        hermes_home = tmp_path / ".hermes"
        hub = hermes_home / "skills" / ".hub" / "metadata.json"
        hub.parent.mkdir(parents=True)
        hub.write_text("{}")

        with patch("agent.file_safety._hermes_home_path", return_value=hermes_home):
            error = get_read_block_error(str(hub))
            assert error is not None


# ---------------------------------------------------------------------------
# Combined: env guard + cache guard don't interfere
# ---------------------------------------------------------------------------


class TestCombinedGuards:
    """Both guards should work independently without interference."""

    def test_env_guard_works_regardless_of_hermes_home(self, tmp_path):
        """The env basename guard does not depend on HERMES_HOME resolution."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()

        with patch("agent.file_safety._hermes_home_path", return_value=hermes_home):
            # Regular project .env should still be blocked
            error = get_read_block_error("/workspace/.env")
            assert error is not None

            # .env.example should still be allowed
            error = get_read_block_error("/workspace/.env.example")
            assert error is None

    def test_cache_guard_still_works_with_env_guard(self, tmp_path):
        """Cache file blocking still works when env guard is active."""
        hermes_home = tmp_path / ".hermes"
        cache = hermes_home / "skills" / ".hub" / "index-cache" / "x"
        cache.parent.mkdir(parents=True)
        cache.write_text("")

        with patch("agent.file_safety._hermes_home_path", return_value=hermes_home):
            error = get_read_block_error(str(cache))
            assert error is not None
            assert "internal Hermes cache" in error


# ---------------------------------------------------------------------------
# Write-denial reason classification (workspace confinement vs protected)
# ---------------------------------------------------------------------------


class TestWriteDenialReason:
    def test_outside_safe_root_classified(self, tmp_path, monkeypatch):
        from agent.file_safety import write_denial_error, write_denial_reason

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(workspace))

        assert write_denial_reason(str(tmp_path / "outside.txt")) == "outside_safe_root"
        assert write_denial_reason(str(workspace / "inside.txt")) is None
        error = write_denial_error(str(tmp_path / "outside.txt"))
        assert "workspace confinement" in error
        assert "credential" not in error.split("not a")[0]

    def test_protected_beats_confinement(self, tmp_path, monkeypatch):
        from agent.file_safety import write_denial_reason

        monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(tmp_path))
        assert write_denial_reason(os.path.expanduser("~/.ssh/id_rsa")) == "protected"

    def test_no_env_mutation_and_thread_safety(self, tmp_path, monkeypatch):
        """Regression: classification must never clear HERMES_WRITE_SAFE_ROOT
        for other threads (a concurrent is_write_denied could allow a
        forbidden write)."""
        import threading

        from agent.file_safety import is_write_denied, write_denial_reason

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(workspace))
        outside = str(tmp_path / "forbidden.txt")

        violations = []
        stop = threading.Event()

        def hammer_check():
            while not stop.is_set():
                if not is_write_denied(outside):
                    violations.append("allowed")
                    return

        checker = threading.Thread(target=hammer_check)
        checker.start()
        try:
            for _ in range(500):
                assert write_denial_reason(outside) == "outside_safe_root"
                assert os.environ["HERMES_WRITE_SAFE_ROOT"] == str(workspace)
        finally:
            stop.set()
            checker.join()
        assert not violations

    def test_multiple_safe_roots_pathsep_separated(self, tmp_path, monkeypatch):
        """0.18.2-specific: HERMES_WRITE_SAFE_ROOT supports os.pathsep-joined
        multiple roots (get_safe_write_roots() returns a set) — a path
        inside ANY listed root must be allowed, not just the first one."""
        from agent.file_safety import write_denial_reason

        root_a = tmp_path / "root_a"
        root_b = tmp_path / "root_b"
        root_a.mkdir()
        root_b.mkdir()
        monkeypatch.setenv(
            "HERMES_WRITE_SAFE_ROOT", os.pathsep.join([str(root_a), str(root_b)])
        )

        assert write_denial_reason(str(root_a / "a.txt")) is None
        assert write_denial_reason(str(root_b / "b.txt")) is None
        assert write_denial_reason(str(tmp_path / "outside.txt")) == "outside_safe_root"
