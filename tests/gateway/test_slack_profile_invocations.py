"""Slack single-app profile invocation behavior."""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, SendResult
from gateway.session import SessionSource, build_session_key
from hermes_cli import slack_cli
from plugins.platforms.slack import adapter as slack_adapter_module
from plugins.platforms.slack.adapter import SlackAdapter


def _config(store, *, profile="nami"):
    return PlatformConfig(
        enabled=True,
        token="test-token",
        extra={
            "profile_invocation_store": str(store),
            "profile_invocations": [
                {
                    "profile": profile,
                    "slash": profile,
                    "aliases": [profile, "나미"],
                    "display_name": "Nami",
                    "icon_emoji": "hermes_nami",
                }
            ],
        },
    )


def _adapter(store, *, profile="nami"):
    adapter = SlackAdapter(_config(store, profile=profile))
    adapter._app = MagicMock()
    client = AsyncMock()
    client.chat_postMessage = AsyncMock(return_value={"ok": True, "ts": "200.001"})
    adapter._get_client = MagicMock(return_value=client)
    adapter.stop_typing = AsyncMock()
    return adapter, client


def test_alias_parser_is_explicit_and_boundary_safe(tmp_path):
    adapter, _ = _adapter(tmp_path / "routes.json")
    with (
        patch("agent.secret_scope.is_multiplex_active", return_value=True),
        patch("hermes_cli.profiles.profile_exists", return_value=True),
    ):
        spec, text = adapter._match_profile_alias("나미: 재고를 확인해줘")
        assert spec["profile"] == "nami"
        assert text == "재고를 확인해줘"
        assert adapter._match_profile_alias("나미야 재고를 확인해줘")[0] is None


def test_missing_profile_is_rejected(tmp_path):
    adapter, _ = _adapter(tmp_path / "routes.json", profile="missing")
    with (
        patch("agent.secret_scope.is_multiplex_active", return_value=True),
        patch("hermes_cli.profiles.profile_exists", return_value=False),
    ):
        assert adapter._profile_invocation_specs() == {}


def test_profile_invocations_fail_closed_without_multiplex(tmp_path):
    adapter, _ = _adapter(tmp_path / "routes.json")
    with patch("agent.secret_scope.is_multiplex_active", return_value=False):
        assert adapter._profile_invocation_specs() == {}
        assert adapter._match_profile_alias("나미 재고 확인")[0] is None


@pytest.mark.asyncio
async def test_profile_slash_is_public_customized_and_stamps_source(tmp_path):
    store = tmp_path / "routes.json"
    adapter, client = _adapter(store)
    captured = []

    async def handle(event):
        captured.append(event)
        await adapter.send(
            event.source.chat_id,
            "done",
            metadata={"slack_team_id": event.source.scope_id},
        )

    adapter.handle_message = handle
    command = {
        "command": "/nami",
        "text": "calculate coverage",
        "user_id": "U1",
        "channel_id": "C1",
        "team_id": "T1",
        "response_url": "https://example.invalid/ephemeral",
    }
    with (
        patch("agent.secret_scope.is_multiplex_active", return_value=True),
        patch("hermes_cli.profiles.profile_exists", return_value=True),
    ):
        await adapter._handle_slash_command(command)

    assert captured[0].source.profile == "nami"
    assert captured[0].text == "calculate coverage"
    assert adapter._slash_command_contexts == {}
    kwargs = client.chat_postMessage.await_args.kwargs
    assert kwargs["username"] == "Nami"
    assert kwargs["icon_emoji"] == ":hermes_nami:"
    assert json.loads(store.read_text())["routes"]["T1\u001fC1\u001f200.001"]["profile"] == "nami"


@pytest.mark.asyncio
async def test_legacy_hermes_slash_routes_profile_without_extra_command_slot(tmp_path):
    adapter, client = _adapter(tmp_path / "routes.json")
    captured = []

    async def handle(event):
        captured.append(event)
        await adapter.send(
            event.source.chat_id,
            "done",
            metadata={"slack_team_id": event.source.scope_id},
        )

    adapter.handle_message = handle
    command = {
        "command": "/hermes",
        "text": "나미 calculate coverage",
        "user_id": "U1",
        "channel_id": "C1",
        "team_id": "T1",
        "response_url": "https://example.invalid/ephemeral",
    }
    with (
        patch("agent.secret_scope.is_multiplex_active", return_value=True),
        patch("hermes_cli.profiles.profile_exists", return_value=True),
    ):
        await adapter._handle_slash_command(command)

    assert captured[0].source.profile == "nami"
    assert captured[0].text == "calculate coverage"
    assert adapter._slash_command_contexts == {}
    kwargs = client.chat_postMessage.await_args.kwargs
    assert kwargs["username"] == "Nami"
    assert kwargs["icon_emoji"] == ":hermes_nami:"


@pytest.mark.asyncio
async def test_normal_send_and_cron_path_keep_app_identity(tmp_path):
    adapter, client = _adapter(tmp_path / "routes.json")
    await adapter.send("C1", "scheduled report", metadata={"slack_team_id": "T1"})
    kwargs = client.chat_postMessage.await_args.kwargs
    assert "username" not in kwargs
    assert "icon_emoji" not in kwargs


@pytest.mark.asyncio
async def test_queued_profile_switch_restores_event_persona(tmp_path):
    adapter, client = _adapter(tmp_path / "routes.json")
    adapter.config.typing_indicator = False
    adapter.set_message_handler(lambda _event: asyncio.sleep(0, result="done"))
    event = MessageEvent(
        text="review this",
        source=SessionSource(
            platform=Platform.SLACK,
            chat_id="C1",
            chat_type="group",
            user_id="U1",
            scope_id="T1",
            profile="chopper",
        ),
        metadata={
            "_slack_profile_persona": {
                "profile": "chopper",
                "display_name": "Chopper",
                "icon_emoji": ":hermes_chopper:",
            }
        },
    )

    stale = slack_adapter_module._profile_persona.set(
        {"profile": "nami", "display_name": "Nami", "icon_emoji": ":hermes_nami:"}
    )
    try:
        await adapter._process_message_background(event, build_session_key(event.source))
    finally:
        slack_adapter_module._profile_persona.reset(stale)

    kwargs = client.chat_postMessage.await_args.kwargs
    assert kwargs["username"] == "Chopper"
    assert kwargs["icon_emoji"] == ":hermes_chopper:"


@pytest.mark.asyncio
async def test_queued_ordinary_slash_clears_stale_persona(tmp_path):
    adapter, client = _adapter(tmp_path / "routes.json")
    adapter.config.typing_indicator = False
    adapter.set_message_handler(lambda _event: asyncio.sleep(0, result="status"))
    adapter._send_slash_ephemeral = AsyncMock(
        return_value=SendResult(success=True, message_id="ephemeral")
    )
    adapter._slash_command_contexts[("T1", "C1", "U2")] = {
        "response_url": "https://example.invalid/ephemeral",
        "user_id": "U2",
        "ts": time.monotonic(),
    }
    event = MessageEvent(
        text="/status",
        message_type=MessageType.COMMAND,
        source=SessionSource(
            platform=Platform.SLACK,
            chat_id="C1",
            chat_type="group",
            user_id="U2",
            scope_id="T1",
        ),
        metadata={
            "_slack_profile_persona": None,
            "_slack_slash_user_id": "U2",
        },
    )

    stale = slack_adapter_module._profile_persona.set(
        {"profile": "nami", "display_name": "Nami", "icon_emoji": ":hermes_nami:"}
    )
    try:
        await adapter._process_message_background(event, build_session_key(event.source))
    finally:
        slack_adapter_module._profile_persona.reset(stale)

    adapter._send_slash_ephemeral.assert_awaited_once()
    client.chat_postMessage.assert_not_awaited()


@pytest.mark.asyncio
async def test_profile_slash_is_blocked_before_dispatch_in_ignored_channel(tmp_path):
    config = _config(tmp_path / "routes.json")
    config.extra["ignored_channels"] = ["C_BLOCKED"]
    adapter = SlackAdapter(config)
    adapter.handle_message = AsyncMock()
    with (
        patch("agent.secret_scope.is_multiplex_active", return_value=True),
        patch("hermes_cli.profiles.profile_exists", return_value=True),
    ):
        await adapter._handle_slash_command(
            {
                "command": "/nami",
                "text": "run",
                "user_id": "U1",
                "channel_id": "C_BLOCKED",
                "team_id": "T1",
            }
        )
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_profile_slash_obeys_allowed_channels_before_dispatch(tmp_path):
    config = _config(tmp_path / "routes.json")
    config.extra["allowed_channels"] = ["C_ALLOWED"]
    adapter = SlackAdapter(config)
    adapter.handle_message = AsyncMock()
    with (
        patch("agent.secret_scope.is_multiplex_active", return_value=True),
        patch("hermes_cli.profiles.profile_exists", return_value=True),
    ):
        await adapter._handle_slash_command(
            {
                "command": "/nami",
                "text": "run",
                "user_id": "U1",
                "channel_id": "C_OTHER",
                "team_id": "T1",
            }
        )
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_profile_slash_obeys_user_authorization_before_dispatch(tmp_path):
    adapter, _ = _adapter(tmp_path / "routes.json")
    runner = MagicMock()
    runner._is_user_authorized.return_value = False
    adapter._message_handler = AsyncMock()
    adapter._message_handler.__self__ = runner
    adapter.handle_message = AsyncMock()
    with (
        patch("agent.secret_scope.is_multiplex_active", return_value=True),
        patch("hermes_cli.profiles.profile_exists", return_value=True),
    ):
        await adapter._handle_slash_command(
            {
                "command": "/nami",
                "text": "run",
                "user_id": "U_BAD",
                "channel_id": "C_ALLOWED",
                "team_id": "T1",
            }
        )
    adapter.handle_message.assert_not_awaited()
    runner._is_user_authorized.assert_called_once()


def test_thread_binding_survives_adapter_restart(tmp_path):
    store = tmp_path / "routes.json"
    adapter, _ = _adapter(store)
    spec = {
        "profile": "nami",
        "display_name": "Nami",
        "icon_emoji": ":hermes_nami:",
    }
    adapter._bind_profile_thread("T1", "C1", "100.001", spec)

    restarted, _ = _adapter(store)
    with (
        patch("agent.secret_scope.is_multiplex_active", return_value=True),
        patch("hermes_cli.profiles.profile_exists", return_value=True),
    ):
        restored = restarted._continued_profile_spec("T1", "C1", "100.001")
        assert restored["profile"] == "nami"
        assert restored["display_name"] == "Nami"
        assert restarted._continued_profile_spec("T2", "C1", "100.001") is None


def test_stale_thread_binding_cannot_bypass_current_config(tmp_path):
    store = tmp_path / "routes.json"
    adapter, _ = _adapter(store)
    adapter._bind_profile_thread(
        "T1",
        "C1",
        "100.001",
        {"profile": "nami", "display_name": "Nami", "icon_emoji": ":hermes_nami:"},
    )
    adapter.config.extra["profile_invocations"] = []
    assert adapter._continued_profile_spec("T1", "C1", "100.001") is None


def test_manifest_pins_profile_slashes_and_customize_scope(monkeypatch):
    monkeypatch.setattr(
        slack_cli,
        "_configured_profile_slashes",
        lambda: [
            {
                "command": "/nami",
                "description": "Run Nami",
                "usage_hint": "[request]",
                "should_escape": False,
                "url": "https://hermes-agent.local/slack/commands",
            }
        ],
    )
    manifest = slack_cli._build_full_manifest("Luffy", "Crew gateway")
    commands = [entry["command"] for entry in manifest["features"]["slash_commands"]]
    assert commands[:2] == ["/hermes", "/nami"]
    assert len(commands) <= 50
    assert "chat:write.customize" in manifest["oauth_config"]["scopes"]["bot"]


def test_configured_profile_slashes_reads_real_gateway_shape(monkeypatch):
    slack = MagicMock()
    slack.extra = {
        "profile_invocations": [
            {"profile": name, "display_name": name.title()}
            for name in ("luffy", "nami", "chopper", "franky")
        ]
    }
    gateway = MagicMock(multiplex_profiles=True, platforms={Platform.SLACK: slack})
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: gateway)
    assert [entry["command"] for entry in slack_cli._configured_profile_slashes()] == [
        "/luffy",
        "/nami",
        "/chopper",
        "/franky",
    ]


def test_manifest_config_failure_is_not_silent(monkeypatch):
    def fail():
        raise ValueError("broken config")

    monkeypatch.setattr("gateway.config.load_gateway_config", fail)
    with pytest.raises(RuntimeError, match="profile invocation configuration"):
        slack_cli._configured_profile_slashes()


def test_customize_scope_absent_without_profile_invocations(monkeypatch):
    monkeypatch.setattr(slack_cli, "_configured_profile_slashes", lambda: [])
    manifest = slack_cli._build_full_manifest("Luffy", "Gateway")
    assert "chat:write.customize" not in manifest["oauth_config"]["scopes"]["bot"]


def test_slashes_only_merge_also_pins_profile_commands(monkeypatch):
    profile_entry = {
        "command": "/nami",
        "description": "Run Nami",
        "usage_hint": "[request]",
        "should_escape": False,
        "url": "https://hermes-agent.local/slack/commands",
    }
    monkeypatch.setattr(slack_cli, "_configured_profile_slashes", lambda: [profile_entry])
    base = [
        {"command": "/hermes"},
        *({"command": f"/cmd{i}"} for i in range(1, 50)),
    ]
    merged = slack_cli._merge_profile_slashes(base)
    assert [entry["command"] for entry in merged[:2]] == ["/hermes", "/nami"]
    assert len(merged) == 50
