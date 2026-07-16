"""Conservative topic-shift detection for gateway preflight compression.

This module intentionally owns only policy and the bounded auxiliary request.
GatewayRunner remains responsible for history loading and the established safe
compression/persistence path.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable, Mapping, Optional, Sequence

from agent.auxiliary_client import async_call_llm, extract_content_or_reasoning

logger = logging.getLogger(__name__)

_RECENT_USER_TURN_LIMIT = 4
_RECENT_USER_TURN_MAX_CHARS = 800
_MIN_INCOMING_CHARS = 12
_AUXILIARY_TIMEOUT_SECONDS = 8.0
_CONTROL_PREFIXES = (
    "background execution contract",
    "completion guard",
    "context summary",
    "context compaction",
    "[context summary]",
    "[prior context",
    "[system note:",
)
_APPENDED_CONTROL_PREFIXES = (
    "background execution contract:",
    "[completion guard active]",
)
_GATEWAY_REPLY_WRAPPER_PREFIXES = (
    '[Replying to: "',
    '[Replying to your previous message: "',
)
_MAX_GATEWAY_REPLY_QUOTE_CHARS = 500
_GATEWAY_LEADING_SEPARATOR = re.compile(r"^\r?\n[ \t]*\r?\n")
_DISCORD_TRIGGER_MESSAGE_ID = re.compile(
    r"^\[Triggering message id: `[^`\r\n]+` — use as `message_id` for reply/react/pin via the discord tools\.\]\r?\n[ \t]*\r?\n"
)


class TopicShiftDecision(str, Enum):
    SAME = "SAME"
    SHIFT = "SHIFT"
    AMBIGUOUS = "AMBIGUOUS"


@dataclass(frozen=True)
class TopicShiftConfig:
    """Feature gate and candidate threshold for topic-shift compaction."""

    enabled: bool = False
    min_messages: int = 80
    min_tokens: int = 80_000

    @classmethod
    def from_compression_config(cls, compression: Any) -> "TopicShiftConfig":
        topic_shift = compression.get("topic_shift", {}) if isinstance(compression, Mapping) else {}
        if not isinstance(topic_shift, Mapping):
            topic_shift = {}

        def _positive_int(name: str, default: int) -> int:
            try:
                value = int(topic_shift.get(name, default))
            except (TypeError, ValueError):
                return default
            return value if value > 0 else default

        enabled = str(topic_shift.get("enabled", False)).strip().lower() in {"true", "1", "yes", "on"}
        return cls(
            enabled=enabled,
            min_messages=_positive_int("min_messages", 80),
            min_tokens=_positive_int("min_tokens", 80_000),
        )


def is_topic_shift_candidate(
    history: Sequence[Mapping[str, Any]],
    incoming_text: str,
    config: TopicShiftConfig,
    *,
    estimated_tokens: int,
) -> bool:
    """Return whether a safe, sufficiently large transcript merits a detector call."""
    if not config.enabled or not _is_meaningful_incoming_text(incoming_text):
        return False
    return len(history) >= config.min_messages or estimated_tokens >= config.min_tokens


def extract_recent_real_user_turns(
    history: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Return a bounded sample of real user text, excluding synthetic context."""
    recent: list[str] = []
    for message in reversed(history):
        if message.get("role") != "user" or message.get("_compressed_summary"):
            continue
        text = _clean_real_user_text(message.get("content"))
        if not text:
            continue
        recent.append(text[:_RECENT_USER_TURN_MAX_CHARS])
        if len(recent) >= _RECENT_USER_TURN_LIMIT:
            break
    recent.reverse()
    return recent


async def classify_topic_shift(
    history: Sequence[Mapping[str, Any]],
    incoming_text: str,
    config: TopicShiftConfig,
    *,
    estimated_tokens: int,
    main_runtime: Optional[dict[str, Any]] = None,
    auxiliary_call: Optional[Callable[..., Awaitable[Any]]] = None,
) -> TopicShiftDecision:
    """Classify a candidate session, failing open unless the reply is exact SHIFT."""
    cleaned_incoming_text = _clean_real_user_text(incoming_text)
    if not is_topic_shift_candidate(
        history, cleaned_incoming_text, config, estimated_tokens=estimated_tokens
    ):
        return TopicShiftDecision.AMBIGUOUS

    recent_turns = extract_recent_real_user_turns(history)
    if not recent_turns:
        return TopicShiftDecision.AMBIGUOUS

    call = auxiliary_call or async_call_llm
    messages = [
        {"role": "system", "content": _CLASSIFIER_INSTRUCTIONS},
        {
            "role": "user",
            "content": _render_classifier_input(recent_turns, cleaned_incoming_text),
        },
    ]
    try:
        response = await asyncio.wait_for(
            call(
                task="topic_shift",
                messages=messages,
                max_tokens=8,
                temperature=0,
                timeout=_AUXILIARY_TIMEOUT_SECONDS,
                main_runtime=main_runtime,
            ),
            timeout=_AUXILIARY_TIMEOUT_SECONDS,
        )
        verdict = (extract_content_or_reasoning(response) or "").strip()
    except Exception as exc:
        logger.debug("Gateway topic-shift detector failed open: %s", exc)
        return TopicShiftDecision.AMBIGUOUS

    try:
        return TopicShiftDecision(verdict)
    except ValueError:
        logger.debug("Gateway topic-shift detector returned malformed verdict: %r", verdict)
        return TopicShiftDecision.AMBIGUOUS


def _is_meaningful_incoming_text(text: Any) -> bool:
    cleaned = _clean_real_user_text(text)
    return len(cleaned) >= _MIN_INCOMING_CHARS and not cleaned.startswith("/")


def _clean_real_user_text(content: Any) -> str:
    if isinstance(content, list):
        content = " ".join(
            str(part.get("text", "")) if isinstance(part, Mapping) else str(part)
            for part in content
        )
    text = str(content or "").strip()
    if not text:
        return ""
    text = _strip_leading_gateway_reply_wrapper(text)
    text = _strip_leading_discord_trigger_message_id(text)
    lines = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(">"):
            continue
        if stripped.lower().startswith(_APPENDED_CONTROL_PREFIXES):
            break
        lines.append(line)
    text = "\n".join(lines).strip()
    if text.lower().startswith(_CONTROL_PREFIXES):
        return ""
    return " ".join(text.split())


def _strip_leading_gateway_reply_wrapper(text: str) -> str:
    """Remove a complete leading gateway reply wrapper and blank separator."""
    prefix = next((item for item in _GATEWAY_REPLY_WRAPPER_PREFIXES if text.startswith(item)), None)
    if prefix is None:
        return text

    quote_end_limit = len(prefix) + _MAX_GATEWAY_REPLY_QUOTE_CHARS
    wrapper_end = text.find('"]', len(prefix), quote_end_limit + 2)
    while wrapper_end >= 0:
        separator = _GATEWAY_LEADING_SEPARATOR.match(text[wrapper_end + 2 :])
        if separator is not None:
            return text[wrapper_end + 2 + separator.end() :]
        wrapper_end = text.find('"]', wrapper_end + 2, quote_end_limit + 2)
    return text


def _strip_leading_discord_trigger_message_id(text: str) -> str:
    """Remove only a complete leading Discord trigger pointer and separator."""
    pointer = _DISCORD_TRIGGER_MESSAGE_ID.match(text)
    if pointer is None:
        return text
    return text[pointer.end() :]


def _render_classifier_input(recent_turns: Sequence[str], incoming_text: str) -> str:
    previous = "\n".join(f"- {turn}" for turn in recent_turns)
    return f"RECENT REAL USER TURNS:\n{previous}\n\nINCOMING USER MESSAGE:\n{incoming_text.strip()}"


_CLASSIFIER_INSTRUCTIONS = """Classify whether the incoming user message needs the recent task context.
Return exactly one token: SAME, SHIFT, or AMBIGUOUS.
SAME: a follow-up, correction, status request, pronoun/reference to the recent task, or the same business objective.
SHIFT: an independent objective or domain that does not need the recent task context.
AMBIGUOUS: any uncertainty. Do not infer continuity from boilerplate or quoted text."""
