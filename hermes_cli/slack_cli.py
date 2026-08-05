"""``hermes slack ...`` CLI subcommands.

Today only ``hermes slack manifest`` is implemented — it generates the
Slack app manifest JSON for registering every gateway command as a native
Slack slash (``/btw``, ``/stop``, ``/model``, …) so users get the same
first-class slash UX Discord and Telegram already have.

Typical workflow::

    $ hermes slack manifest > slack-manifest.json
    # or:
    $ hermes slack manifest --write

Then paste the printed JSON into the Slack app config (Features → App
Manifest → Edit) and click Save. Slack diffs the manifest and prompts
for reinstall when scopes/commands change.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


SLACK_LONG_DESCRIPTION_MIN_CHARACTERS = 175
SLACK_LONG_DESCRIPTION_MAX_CHARACTERS = 4000


def _configured_profile_slashes() -> list[dict]:
    """Build manifest entries for opt-in Slack profile invocations."""
    try:
        from gateway.config import Platform, load_gateway_config

        config = load_gateway_config()
        if not config.multiplex_profiles:
            return []
        slack = config.platforms.get(Platform.SLACK)
        raw = (slack.extra if slack else {}).get("profile_invocations", [])
    except Exception as exc:
        raise RuntimeError(
            "could not load Slack profile invocation configuration"
        ) from exc
    if not isinstance(raw, list):
        return []
    entries: list[dict] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        profile = str(item.get("profile") or "").strip().lower()
        slash = str(item.get("slash") or profile).strip().lower().lstrip("/")
        if (
            not profile
            or not slash
            or slash in seen
            or len(slash) > 32
            or any(ch not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for ch in slash)
        ):
            continue
        seen.add(slash)
        display_name = str(item.get("display_name") or profile.title())
        entries.append(
            {
                "command": f"/{slash}",
                "description": f"Run the {display_name} Hermes profile"[:140],
                "usage_hint": "[request]",
                "should_escape": False,
                "url": "https://hermes-agent.local/slack/commands",
            }
        )
    return entries


def _merge_profile_slashes(
    slashes: list[dict], profile_slashes: list[dict] | None = None
) -> list[dict]:
    """Pin configured profile slashes while preserving Slack's 50-command cap."""
    if profile_slashes is None:
        profile_slashes = _configured_profile_slashes()
    if not profile_slashes:
        return slashes
    profile_names = {entry["command"] for entry in profile_slashes}
    remaining = [
        entry
        for entry in slashes
        if entry["command"] != "/hermes" and entry["command"] not in profile_names
    ]
    hermes_entry = next(
        (entry for entry in slashes if entry["command"] == "/hermes"),
        None,
    )
    merged = ([hermes_entry] if hermes_entry else []) + profile_slashes + remaining
    return merged[:50]


def _build_full_manifest(
    bot_name: str,
    bot_description: str,
    include_assistant: bool = True,
    messaging_experience: str | None = None,
    long_description: str | None = None,
) -> dict:
    """Build a full Slack manifest merging display info + our slash list.

    The slash-command list is always generated from ``COMMAND_REGISTRY`` so
    it stays in sync with the rest of Hermes. Other manifest sections
    (display info, OAuth scopes, socket mode) are set to sensible defaults
    for a Hermes deployment — users can tweak them in the Slack UI after
    pasting.

    By default, this keeps Hermes on Slack's older Assistant messaging
    experience (``assistant_view``) for backward compatibility. Pass
    ``messaging_experience="agent"`` (``--agent-view``) to emit Slack's Agent
    messaging experience (``agent_view`` + ``app_home_opened``). Pass
    ``include_assistant=False`` or ``messaging_experience="none"``
    (``--no-assistant``) to omit Slack AI messaging features and get a flat DM
    surface where ``/help``, ``/new``, etc. work inline.
    """
    from hermes_cli.commands import slack_app_manifest

    if messaging_experience is None:
        messaging_experience = "assistant" if include_assistant else "none"
    messaging_experience = str(messaging_experience).strip().lower()
    if messaging_experience not in {"assistant", "agent", "none"}:
        raise ValueError(
            "messaging_experience must be one of: assistant, agent, none"
        )

    partial = slack_app_manifest()
    # Slack caps an app at 50 slash commands. Keep /hermes first, then pin
    # configured profile invocations ahead of lower-priority native commands;
    # every displaced command remains reachable via /hermes.
    profile_slashes = _configured_profile_slashes()
    slashes = _merge_profile_slashes(
        partial["features"]["slash_commands"], profile_slashes
    )

    features = {
        "app_home": {
            "home_tab_enabled": False,
            "messages_tab_enabled": True,
            "messages_tab_read_only_enabled": False,
        },
        "bot_user": {
            "display_name": bot_name[:80],
            "always_online": True,
        },
        "slash_commands": slashes,
    }

    bot_scopes = [
        "app_mentions:read",
        "channels:history",
        "channels:read",
        "chat:write",
        "commands",
        "files:read",
        "files:write",
        "groups:history",
        "groups:read",
        "im:history",
        "im:read",
        "im:write",
        "mpim:history",
        "mpim:read",
        "reactions:read",
        "users:read",
    ]
    if profile_slashes:
        bot_scopes.append("chat:write.customize")

    bot_events = [
        "app_mention",
        "message.channels",
        "message.groups",
        "message.im",
        "message.mpim",
        "reaction_added",
        "reaction_removed",
    ]

    if messaging_experience == "assistant":
        features["assistant_view"] = {
            "assistant_description": "Chat with Hermes in threads and DMs.",
        }
        bot_scopes.append("assistant:write")
        bot_events.extend(
            [
                "assistant_thread_context_changed",
                "assistant_thread_started",
            ]
        )
    elif messaging_experience == "agent":
        features["agent_view"] = {
            "agent_description": "Chat with Hermes in Slack Messages.",
        }
        bot_scopes.append("assistant:write")
        # Slack includes current viewing context in Agent DM events only after
        # this subscription is enabled; the adapter consumes that context to
        # preserve the referred channel across the agent turn.
        bot_events.extend(["app_context_changed", "app_home_opened"])

    bot_scopes.sort()
    bot_events.sort()

    display_information = {
        "name": bot_name[:35],
        "description": (bot_description or "Your Hermes agent on Slack")[:140],
        "background_color": "#1a1a2e",
    }
    if long_description is not None:
        display_information["long_description"] = long_description

    return {
        "_metadata": {
            "major_version": 1,
            "minor_version": 1,
        },
        "display_information": display_information,
        "features": features,
        "oauth_config": {
            "scopes": {
                "bot": bot_scopes,
            },
        },
        "settings": {
            "event_subscriptions": {
                "bot_events": bot_events,
            },
            "interactivity": {
                "is_enabled": True,
            },
            "org_deploy_enabled": False,
            "socket_mode_enabled": True,
            "token_rotation_enabled": False,
        },
    }


def slack_manifest_command(args) -> int:
    """Print or write a Slack app manifest JSON.

    Flags (all parsed in ``hermes_cli/main.py``):
      --write [PATH]  Write to file instead of stdout (default path:
                      ``$HERMES_HOME/slack-manifest.json``)
      --name NAME     Override the bot display name (default: "Hermes")
      --description DESC  Override the bot description
      --long-description TEXT  Override the long app description (175-4,000 characters)
      --long-description-file PATH  Read the long app description from a UTF-8 file
      --slashes-only  Emit only the ``features.slash_commands`` array (for
                      merging into an existing manifest manually)
      --no-assistant  Omit Slack AI Assistant mode (assistant_view feature,
                      assistant:write scope, assistant_thread_* events) so
                      DMs render as a flat chat where bare slash commands
                      work inline instead of the Assistant thread pane.
      --agent-view    Use Slack's Agent messaging experience (agent_view,
                      app_home_opened + message.im) instead of the legacy
                      Assistant messaging experience.
    """
    name = getattr(args, "name", None) or "Hermes"
    description = getattr(args, "description", None) or "Your Hermes agent on Slack"
    long_description = getattr(args, "long_description", None)
    long_description_file = getattr(args, "long_description_file", None)
    if getattr(args, "slashes_only", False) and (
        long_description is not None or long_description_file is not None
    ):
        print(
            "hermes slack manifest: long description options cannot be used "
            "with --slashes-only",
            file=sys.stderr,
        )
        return 2
    if long_description_file is not None:
        source_arg = str(long_description_file)
        try:
            source = Path(source_arg).expanduser()
            with source.open("r", encoding="utf-8", newline="") as handle:
                long_description = handle.read()
        except (OSError, UnicodeError, RuntimeError) as exc:
            print(
                f"hermes slack manifest: cannot read long description from "
                f"{source_arg}: {exc}",
                file=sys.stderr,
            )
            return 2
    if (
        long_description is not None
        and len(long_description) < SLACK_LONG_DESCRIPTION_MIN_CHARACTERS
    ):
        print(
            "hermes slack manifest: long description must be at least "
            f"{SLACK_LONG_DESCRIPTION_MIN_CHARACTERS} characters "
            f"(got {len(long_description)})",
            file=sys.stderr,
        )
        return 2
    if (
        long_description is not None
        and len(long_description) > SLACK_LONG_DESCRIPTION_MAX_CHARACTERS
    ):
        print(
            "hermes slack manifest: long description must be at most "
            f"{SLACK_LONG_DESCRIPTION_MAX_CHARACTERS} characters "
            f"(got {len(long_description)})",
            file=sys.stderr,
        )
        return 2
    if getattr(args, "agent_view", False):
        messaging_experience = "agent"
    elif getattr(args, "no_assistant", False):
        messaging_experience = "none"
    else:
        messaging_experience = "assistant"

    if getattr(args, "slashes_only", False):
        from hermes_cli.commands import slack_app_manifest

        manifest = _merge_profile_slashes(
            slack_app_manifest()["features"]["slash_commands"]
        )
    else:
        manifest = _build_full_manifest(
            name,
            description,
            messaging_experience=messaging_experience,
            long_description=long_description,
        )

    payload = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"

    write_target = getattr(args, "write", None)
    if write_target is not None:
        if isinstance(write_target, bool) and write_target:
            # --write with no value → default location
            from hermes_constants import get_hermes_home

            target = Path(get_hermes_home()) / "slack-manifest.json"
        else:
            target = Path(write_target).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload, encoding="utf-8")
        print(f"Slack manifest written to: {target}", file=sys.stderr)
        print(
            "\nNext steps:\n"
            "  1. Open https://api.slack.com/apps and pick your Hermes app\n"
            "     (or create a new one: Create New App → From an app manifest).\n"
            f"  2. Features → App Manifest → paste the contents of\n"
            f"     {target}\n"
            "  3. Save; Slack will prompt to reinstall the app if scopes or\n"
            "     slash commands changed.\n"
            "  4. Make sure Socket Mode is enabled and you have a bot token\n"
            "     (xoxb-...) and app token (xapp-...) configured via\n"
            "     `hermes setup`.\n",
            file=sys.stderr,
        )
    else:
        sys.stdout.write(payload)
    return 0
