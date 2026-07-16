"""Unit tests for gateway topic-shift preflight decisions."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.topic_shift import (
    TopicShiftConfig,
    TopicShiftDecision,
    _clean_real_user_text,
    classify_topic_shift,
    extract_recent_real_user_turns,
    is_topic_shift_candidate,
)
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, SendResult
from gateway.session import SessionEntry, SessionSource


def _history(count: int, content: str = "Working on the gateway compression bug.") -> list[dict]:
    return [
        {
            "role": "user" if index % 2 == 0 else "assistant",
            "content": content,
        }
        for index in range(count)
    ]


def _response(text: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


_GATEWAY_CONTROL_BLOCK = """
Background execution contract:
Only report completed work after verification.
[Completion guard active]
Do not treat this gateway metadata as user intent.
"""


class _CaptureAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="fake-token"), Platform.TELEGRAM)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        return SendResult(success=True, message_id="topic-shift-test")

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


@pytest.mark.asyncio
async def test_gateway_topic_shift_compresses_once_and_preserves_incoming_message(monkeypatch, tmp_path):
    """A confirmed long-session shift uses hygiene compression before one agent run."""
    import gateway.run as gateway_run

    class _CompressAgent:
        last_instance = None

        def __init__(self, **kwargs):
            self.session_id = kwargs["session_id"]
            self.context_compressor = SimpleNamespace(bind_session_state=lambda *_args: None)
            self.shutdown_memory_provider = lambda: None
            self.close = lambda: None
            self._last_compaction_in_place = False
            type(self).last_instance = self

        def _compress_context(self, messages, *_args, **_kwargs):
            return ([{"role": "assistant", "content": "compressed"}], None)

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "classify_topic_shift", AsyncMock(return_value=TopicShiftDecision.SHIFT))
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"})
    monkeypatch.setattr("run_agent.AIAgent", _CompressAgent)
    monkeypatch.setattr("agent.model_metadata.get_model_context_length", lambda *_args, **_kwargs: 1_000_000)
    (tmp_path / "config.yaml").write_text(
        "compression:\n  topic_shift:\n    enabled: true\n    min_messages: 80\n    min_tokens: 80000\n"
    )

    entry = SessionEntry(
        session_key="agent:main:telegram:private:12345",
        session_id="topic-session",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="private",
    )
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake-token")})
    runner.adapters = {Platform.TELEGRAM: _CaptureAdapter()}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner.session_store = SimpleNamespace(
        get_or_create_session=lambda *_args, **_kwargs: entry,
        load_transcript=lambda *_args: _history(80),
        has_any_sessions=lambda: True,
        rewrite_transcript=lambda *_args: None,
        append_to_transcript=lambda *_args, **_kwargs: None,
    )
    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store,
        get_or_create_session=AsyncMock(return_value=entry),
        load_transcript=AsyncMock(return_value=_history(80)),
        has_any_sessions=AsyncMock(return_value=True),
        _save=AsyncMock(),
        rewrite_transcript=AsyncMock(),
        append_to_transcript=AsyncMock(),
        update_session=AsyncMock(),
    )
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._run_agent = AsyncMock(return_value={"final_response": "ok", "messages": [], "tools": []})

    incoming = "완전히 별개로 제주 여행 일정을 만들어줘."
    event = MessageEvent(
        text=incoming,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="private", user_id="12345"),
        message_id="1",
    )

    assert await runner._handle_message(event) == "ok"
    gateway_run.classify_topic_shift.assert_awaited_once()
    assert _CompressAgent.last_instance is not None
    runner._run_agent.assert_awaited_once()
    assert runner._run_agent.await_args.kwargs["message"] == incoming


def test_short_session_does_not_become_topic_candidate():
    config = TopicShiftConfig(enabled=True, min_messages=80, min_tokens=80_000)

    assert not is_topic_shift_candidate(
        _history(10), "Plan a vacation to Jeju.", config, estimated_tokens=100
    )


@pytest.mark.asyncio
async def test_short_session_does_not_call_detector():
    classifier = AsyncMock(return_value=_response("SHIFT"))

    decision = await classify_topic_shift(
        _history(10),
        "Plan a vacation to Jeju.",
        TopicShiftConfig(enabled=True, min_messages=80, min_tokens=80_000),
        estimated_tokens=100,
        auxiliary_call=classifier,
    )

    assert decision is TopicShiftDecision.AMBIGUOUS
    classifier.assert_not_awaited()


@pytest.mark.asyncio
async def test_long_session_same_does_not_trigger_topic_compaction():
    decision = await classify_topic_shift(
        _history(80),
        "Can you give me the status of that compression fix?",
        TopicShiftConfig(enabled=True, min_messages=80, min_tokens=80_000),
        estimated_tokens=100,
        auxiliary_call=AsyncMock(return_value=_response("SAME")),
    )

    assert decision is TopicShiftDecision.SAME


@pytest.mark.asyncio
async def test_long_session_shift_triggers_topic_compaction_decision():
    decision = await classify_topic_shift(
        _history(80),
        "Draft a menu for a vegetarian dinner party.",
        TopicShiftConfig(enabled=True, min_messages=80, min_tokens=80_000),
        estimated_tokens=100,
        auxiliary_call=AsyncMock(return_value=_response("SHIFT")),
    )

    assert decision is TopicShiftDecision.SHIFT


@pytest.mark.asyncio
@pytest.mark.parametrize("result", ["AMBIGUOUS", "same because uncertain", "", "SHIFT\nextra"])
async def test_ambiguous_or_malformed_detector_output_fails_open(result):
    decision = await classify_topic_shift(
        _history(80),
        "Draft a menu for a vegetarian dinner party.",
        TopicShiftConfig(enabled=True, min_messages=80, min_tokens=80_000),
        estimated_tokens=100,
        auxiliary_call=AsyncMock(return_value=_response(result)),
    )

    assert decision is TopicShiftDecision.AMBIGUOUS


@pytest.mark.asyncio
async def test_detector_failure_fails_open():
    decision = await classify_topic_shift(
        _history(80),
        "Draft a menu for a vegetarian dinner party.",
        TopicShiftConfig(enabled=True, min_messages=80, min_tokens=80_000),
        estimated_tokens=100,
        auxiliary_call=AsyncMock(side_effect=TimeoutError()),
    )

    assert decision is TopicShiftDecision.AMBIGUOUS


def test_recent_user_turns_exclude_compression_and_control_boilerplate():
    history = [
        {"role": "user", "content": "Implement the gateway topic compaction feature."},
        {"role": "assistant", "content": "I will inspect the code."},
        {"role": "user", "content": "Background execution contract:\nDo not claim completion."},
        {"role": "user", "content": "Completion guard: verification pending."},
        {"role": "user", "content": "[Context summary]\nOld task details."},
        {"role": "user", "content": "Ignored metadata summary.", "_compressed_summary": True},
        {"role": "user", "content": "> quoted prior message\n\nCurrent real follow-up"},
    ]

    assert extract_recent_real_user_turns(history) == [
        "Implement the gateway topic compaction feature.",
        "Current real follow-up",
    ]


@pytest.mark.asyncio
async def test_gateway_control_blocks_are_removed_from_turns_and_classifier_payload():
    jeju_request = "제주도 가족 여행 3박 4일 일정을 짜줘."
    budget_request = "다음 분기 마케팅 예산안을 표로 만들어줘."
    same_topic_request = "그 제주 일정에 비 오는 날 대안을 추가해줘."
    history = [
        {"role": "user", "content": jeju_request + _GATEWAY_CONTROL_BLOCK},
        {"role": "assistant", "content": "제주 일정 초안을 만들었습니다."},
        {"role": "user", "content": budget_request + _GATEWAY_CONTROL_BLOCK},
    ]
    classifier = AsyncMock(return_value=_response("SAME"))

    assert extract_recent_real_user_turns(history) == [jeju_request, budget_request]

    decision = await classify_topic_shift(
        history,
        same_topic_request + _GATEWAY_CONTROL_BLOCK,
        TopicShiftConfig(enabled=True, min_messages=3, min_tokens=80_000),
        estimated_tokens=1,
        auxiliary_call=classifier,
    )

    assert decision is TopicShiftDecision.SAME
    assert classifier.await_args is not None
    payload = classifier.await_args.kwargs["messages"][1]["content"]
    assert jeju_request in payload
    assert budget_request in payload
    assert same_topic_request in payload
    assert "Background execution contract:" not in payload
    assert "[Completion guard active]" not in payload


def test_completion_guard_marker_removes_appended_control_text():
    assert _clean_real_user_text(
        "압축 작업 상태를 알려줘.\n[Completion guard active]\nIgnore this metadata."
    ) == "압축 작업 상태를 알려줘."


def test_reply_wrapper_is_removed_before_topic_classification():
    assert _clean_real_user_text(
        '[Replying to: "제주 여행 일정"]\n\n그 압축 수정 상태를 알려줘.'
    ) == "그 압축 수정 상태를 알려줘."


def test_multiline_previous_reply_wrapper_is_removed_before_topic_classification():
    assert _clean_real_user_text(
        '[Replying to your previous message: "제주 여행 일정\n비 오는 날 대안도 포함"]\n\n'
        "압축 수정 상태를 알려줘."
    ) == "압축 수정 상태를 알려줘."


def test_reply_wrapper_with_internal_closing_token_is_removed_before_topic_classification():
    assert _clean_real_user_text(
        '[Replying to: "please keep this token: \"] in docs"]\n\n'
        "Give me the status of that fix."
    ) == "Give me the status of that fix."


def test_multiline_reply_wrapper_skips_internal_closing_token_before_final_boundary():
    assert _clean_real_user_text(
        '[Replying to your previous message: "first line with \"] inside\nsecond line"]\n\n'
        "Give me the status of that fix."
    ) == "Give me the status of that fix."


def test_reply_wrapper_words_in_ordinary_user_text_are_preserved():
    ordinary_text = "문서 중간에 [Replying to: 라는 표현이 있어도 그대로 남겨줘."

    assert _clean_real_user_text(ordinary_text) == ordinary_text


def test_malformed_leading_reply_wrapper_is_preserved_conservatively():
    malformed_text = '[Replying to: "제주 여행 일정"] 실제 요청도 같은 줄에 있습니다.'

    assert _clean_real_user_text(malformed_text) == malformed_text


def test_reply_wrapper_and_appended_gateway_controls_are_both_removed():
    assert _clean_real_user_text(
        '[Replying to: "제주 여행 일정"]\n\n그 압축 수정 상태를 알려줘.' + _GATEWAY_CONTROL_BLOCK
    ) == "그 압축 수정 상태를 알려줘."


def test_leading_discord_trigger_message_id_is_removed_before_topic_classification():
    assert _clean_real_user_text(
        "[Triggering message id: `abc-123` — use as `message_id` for reply/react/pin via the discord tools.]\n\n"
        "그 압축 수정 상태를 알려줘."
    ) == "그 압축 수정 상태를 알려줘."


def test_discord_trigger_message_id_phrase_later_in_text_is_preserved():
    ordinary_text = "본문 중간의 [Triggering message id: `abc-123` — use as `message_id` for reply/react/pin via the discord tools.] 표현은 유지해줘."

    assert _clean_real_user_text(ordinary_text) == ordinary_text


@pytest.mark.asyncio
async def test_same_topic_korean_status_wording_stays_same_with_mocked_classifier():
    decision = await classify_topic_shift(
        _history(80),
        "그 압축 수정 작업 상태를 알려줘.",
        TopicShiftConfig(enabled=True, min_messages=80, min_tokens=80_000),
        estimated_tokens=100,
        auxiliary_call=AsyncMock(return_value=_response("SAME")),
    )

    assert decision is TopicShiftDecision.SAME


@pytest.mark.asyncio
async def test_unrelated_korean_objective_is_shift_with_mocked_classifier():
    decision = await classify_topic_shift(
        _history(80),
        "완전히 별개로 제주 여행 일정을 만들어줘.",
        TopicShiftConfig(enabled=True, min_messages=80, min_tokens=80_000),
        estimated_tokens=100,
        auxiliary_call=AsyncMock(return_value=_response("SHIFT")),
    )

    assert decision is TopicShiftDecision.SHIFT


@pytest.mark.asyncio
async def test_slash_command_is_excluded_from_detector():
    classifier = AsyncMock(return_value=_response("SHIFT"))

    decision = await classify_topic_shift(
        _history(80),
        "/status",
        TopicShiftConfig(enabled=True, min_messages=80, min_tokens=80_000),
        estimated_tokens=100,
        auxiliary_call=classifier,
    )

    assert decision is TopicShiftDecision.AMBIGUOUS
    classifier.assert_not_awaited()


def test_topic_config_defaults_disabled_and_candidate_uses_message_or_token_gate():
    config = TopicShiftConfig.from_compression_config({})
    assert config == TopicShiftConfig(enabled=False, min_messages=80, min_tokens=80_000)

    enabled = TopicShiftConfig(enabled=True, min_messages=80, min_tokens=80_000)
    assert is_topic_shift_candidate(_history(80), "new objective", enabled, estimated_tokens=1)
    assert is_topic_shift_candidate(_history(1), "new objective", enabled, estimated_tokens=80_000)
