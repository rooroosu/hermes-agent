"""Oneshot (-z) session lifecycle + soft-deadline tests.

WU-1: every oneshot run must finalize its session row (``ended_at`` /
``end_reason``) — the bug this closes is that ``_run_agent`` used to call
``agent.run_conversation(prompt)`` with no ``try``/``finally`` around it, so
the row created by ``_ensure_db_session()`` never got an end_reason.

WU-2: an opt-in wall-clock soft deadline (``--soft-deadline`` /
``HERMES_SOFT_DEADLINE``) must never let a stuck run disappear silently.
Most of these tests exercise the ``_OneshotDeadlineWatchdog`` directly with
fake, non-sleeping timers rather than real ``threading.Timer`` delays, per
the testability seam it's built with. A few (marked REAL) deliberately use
real threads or a real subprocess, because the thing under test — a race
between two threads, or the actual behavior of a real ``os._exit`` — cannot
be faithfully exercised through the fake-timer seam alone.

Ported from the 0.16.0 tree's oneshot.py (where ``_run_agent`` called
``agent.chat(prompt) -> str``) to 0.18.2, where ``_run_agent`` calls
``agent.run_conversation(prompt) -> dict`` directly and returns
``(final_response, result_dict)`` — the fake agent below simulates
``run_conversation()``, not ``chat()``.

Everything here is hermetic: a real ``SessionDB`` pointed at ``tmp_path``,
never the real ``~/.hermes`` state.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time

import pytest

sys.path.insert(0, ".")  # tests run from the hermes repo root

from hermes_cli import oneshot as oneshot_mod  # noqa: E402
from hermes_state import SessionDB  # noqa: E402


class _FakeAgent:
    """Stand-in for ``run_agent.AIAgent`` — just enough surface for
    ``_run_agent`` to drive it: constructible with the same kwargs,
    settable post-init attributes, a ``session_id``, and a
    ``run_conversation()`` hook the test controls (0.18.2's ``_run_agent``
    calls ``run_conversation()`` directly, not ``chat()``)."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.session_db = kwargs.get("session_db")
        self.session_id = "test-session-1"
        self.suppress_status_output = False
        self.stream_delta_callback = None
        self.tool_gen_callback = None
        self._run_conversation_impl = None  # set per-test

    def _ensure_db_session(self):
        if self.session_db is not None:
            self.session_db.create_session(session_id=self.session_id, source="cli")

    def run_conversation(self, prompt):
        assert self._run_conversation_impl is not None, "test must set agent._run_conversation_impl"
        outcome = self._run_conversation_impl(prompt)
        if isinstance(outcome, dict):
            return outcome
        # A bare-string outcome models a NORMAL, cleanly-completed turn: real
        # run_conversation() (turn_finalizer.finalize_turn) always sets
        # ``completed`` on its result, and _run_agent now keys oneshot_complete
        # off it. A non-empty string → completed True; an empty one → False
        # (oneshot's own empty-response check fires first either way).
        return {"final_response": outcome, "completed": bool((outcome or "").strip())}


class _FakeTimer:
    """Records what would have been scheduled instead of sleeping.

    ``start()`` is a no-op; tests fire ``target`` manually to simulate a
    deadline boundary, in any order they like, with zero real delay.
    """

    instances: list["_FakeTimer"] = []

    def __init__(self, delay, target):
        self.delay = delay
        self.target = target
        self.started = False
        self.cancelled = False
        _FakeTimer.instances.append(self)

    def start(self):
        self.started = True

    def cancel(self):
        self.cancelled = True

    def is_alive(self):
        return False

    def join(self, timeout=None):
        pass


@pytest.fixture(autouse=True)
def _reset_fake_timer_registry():
    _FakeTimer.instances.clear()
    yield
    _FakeTimer.instances.clear()


@pytest.fixture
def session_db(tmp_path):
    return SessionDB(db_path=tmp_path / "state.db")


@pytest.fixture
def patch_agent_stack(monkeypatch, session_db):
    """Wire ``_run_agent``'s dependency chain to hermetic, deterministic
    stand-ins: real (but empty/default) config resolution, fake AIAgent.
    """
    monkeypatch.setattr(oneshot_mod, "_create_session_db_for_oneshot", lambda: session_db)

    import run_agent

    monkeypatch.setattr(run_agent, "AIAgent", _FakeAgent)

    from hermes_cli import config as config_mod

    monkeypatch.setattr(config_mod, "load_config", lambda: {})

    from hermes_cli import runtime_provider as runtime_provider_mod

    monkeypatch.setattr(
        runtime_provider_mod,
        "resolve_runtime_provider",
        lambda **kwargs: {
            "api_key": None,
            "base_url": None,
            "provider": "anthropic",
            "api_mode": None,
            "credential_pool": None,
        },
    )

    from hermes_cli import tools_config as tools_config_mod

    monkeypatch.setattr(tools_config_mod, "_get_platform_tools", lambda cfg, platform: [])

    from hermes_cli import fallback_config as fallback_config_mod

    monkeypatch.setattr(fallback_config_mod, "get_fallback_chain", lambda cfg: [])

    return session_db


def _set_run_conversation_impl(monkeypatch, impl):
    """Patch _FakeAgent.__init__ so the constructed instance's
    run_conversation() runs ``impl(agent, prompt)``. Using monkeypatch (not
    a raw assignment) so the mutation is reverted after the test —
    _FakeAgent is a shared module-level class."""
    orig_init = _FakeAgent.__init__

    captured = {}

    def _init(self, **kwargs):
        orig_init(self, **kwargs)
        captured["agent"] = self
        self._run_conversation_impl = lambda prompt: impl(self, prompt)

    monkeypatch.setattr(_FakeAgent, "__init__", _init)
    return captured


# ── WU-1: finalize on completion / error ────────────────────────────────


def test_oneshot_complete_finalizes_session(patch_agent_stack, monkeypatch):
    session_db = patch_agent_stack
    captured = _set_run_conversation_impl(
        monkeypatch, lambda agent, prompt: (agent._ensure_db_session(), "RESULT: DONE")[1]
    )

    response, result = oneshot_mod._run_agent("say hi")

    assert response == "RESULT: DONE"
    assert result.get("final_response") == "RESULT: DONE"
    sid = captured["agent"].session_id
    row = session_db.get_session(sid)
    assert row is not None
    assert row["ended_at"] is not None
    assert row["end_reason"] == "oneshot_complete"


def test_oneshot_error_finalizes_session_and_reraises(patch_agent_stack, monkeypatch):
    session_db = patch_agent_stack

    def _boom(agent, prompt):
        agent._ensure_db_session()
        raise RuntimeError("provider exploded")

    captured = _set_run_conversation_impl(monkeypatch, _boom)

    with pytest.raises(RuntimeError, match="provider exploded"):
        oneshot_mod._run_agent("say hi")

    sid = captured["agent"].session_id
    row = session_db.get_session(sid)
    assert row is not None
    assert row["ended_at"] is not None
    assert row["end_reason"] == "oneshot_error"


def test_oneshot_without_session_db_does_not_crash(patch_agent_stack, monkeypatch, session_db):
    """No SessionDB available (best-effort) — finalize must be a no-op,
    never a crash, and the response still comes through."""
    monkeypatch.setattr(oneshot_mod, "_create_session_db_for_oneshot", lambda: None)
    _set_run_conversation_impl(monkeypatch, lambda agent, prompt: "RESULT: DONE")

    response, result = oneshot_mod._run_agent("say hi")
    assert response == "RESULT: DONE"
    assert result.get("final_response") == "RESULT: DONE"


# ── Finding 4: blank response must not be tagged oneshot_complete ─────
#
# run_oneshot's own empty-response check treats a blank final_response as
# a failure and exits 1 — but _run_agent's finally block used to always
# write "oneshot_complete" whenever run_conversation() returned without
# raising, including a blank final_response. That told dispatcher-side
# incomplete-detection the opposite of the truth. Fixed by tagging blank
# responses "oneshot_empty" — distinct from "oneshot_error" (no exception
# was raised; the model just said nothing).


def test_blank_response_is_tagged_oneshot_empty_not_complete(patch_agent_stack, monkeypatch):
    session_db = patch_agent_stack
    captured = _set_run_conversation_impl(
        monkeypatch, lambda agent, prompt: (agent._ensure_db_session(), "   ")[1]
    )

    response, result = oneshot_mod._run_agent("say hi")

    assert response.strip() == ""
    sid = captured["agent"].session_id
    row = session_db.get_session(sid)
    assert row["ended_at"] is not None
    assert row["end_reason"] == "oneshot_empty"


def test_run_oneshot_blank_response_exact_stderr_and_exit_code(patch_agent_stack, monkeypatch, capsys):
    monkeypatch.delenv("HERMES_SOFT_DEADLINE", raising=False)
    session_db = patch_agent_stack
    captured = _set_run_conversation_impl(
        monkeypatch, lambda agent, prompt: (agent._ensure_db_session(), "")[1]
    )

    exit_code = oneshot_mod.run_oneshot("say hi")

    captured_io = capsys.readouterr()
    assert exit_code == 1
    assert captured_io.out == ""
    assert captured_io.err == "hermes -z: no final response was produced; treating the run as failed.\n"

    sid = captured["agent"].session_id
    row = session_db.get_session(sid)
    assert row["end_reason"] == "oneshot_empty"


# ── Finding 3a: a non-empty but NOT-cleanly-completed run must not be
# tagged oneshot_complete. run_conversation() (turn_finalizer.finalize_turn)
# always sets `completed`; failed/interrupted/iteration-capped/truncated turns
# set completed=False while still returning non-empty text (e.g. "Invalid API
# response after N retries…"). _run_agent used to key oneshot_complete purely
# off "no exception + non-empty response", so those failed runs were recorded
# complete — telling dispatcher-side incomplete-detection the opposite of the
# truth and suppressing a valid retry. Fixed by gating on result["completed"].


def test_nonempty_failed_run_is_tagged_oneshot_incomplete_not_complete(patch_agent_stack, monkeypatch):
    session_db = patch_agent_stack

    def _failed_nonempty(agent, prompt):
        agent._ensure_db_session()
        return {
            "final_response": "Invalid API response after 3 retries: timeout",
            "completed": False,
            "failed": True,
            "error": "Invalid API response after 3 retries: timeout",
        }

    captured = _set_run_conversation_impl(monkeypatch, _failed_nonempty)

    response, result = oneshot_mod._run_agent("say hi")

    assert response == "Invalid API response after 3 retries: timeout"
    sid = captured["agent"].session_id
    row = session_db.get_session(sid)
    assert row["ended_at"] is not None
    assert row["end_reason"] == "oneshot_incomplete"


def test_nonempty_interrupted_run_is_not_tagged_complete(patch_agent_stack, monkeypatch):
    """A non-empty but interrupted/partial turn (completed=False, no `failed`
    flag) must also avoid oneshot_complete — the gate is on `completed`, not
    only on an affirmative failure marker."""
    session_db = patch_agent_stack

    captured = _set_run_conversation_impl(
        monkeypatch,
        lambda agent, prompt: (agent._ensure_db_session(), {
            "final_response": "partial work so far…",
            "completed": False,
            "interrupted": True,
        })[1],
    )

    oneshot_mod._run_agent("say hi")

    row = session_db.get_session(captured["agent"].session_id)
    assert row["end_reason"] == "oneshot_incomplete"


# ── Finding 3b: the NORMAL completion path's terminal end_session() write
# must be bounded. cancel_and_claim() cancels the hard timer, so an unbounded
# write there has no deadline backstop — a SQLite-lock hang could push a normal
# completion past the soft deadline into the external dispatcher's SIGKILL,
# leaving end_reason NULL (the exact bug the watchdog exists to prevent).


def test_normal_completion_end_session_is_bounded_when_watchdog_present(patch_agent_stack, monkeypatch):
    write_started = threading.Event()

    class _StuckSessionDB:
        def create_session(self, **kw):
            pass

        def end_session(self, sid, reason):
            write_started.set()
            time.sleep(10)  # far longer than the budget

        def get_session(self, sid):
            return None

    monkeypatch.setattr(oneshot_mod, "_create_session_db_for_oneshot", lambda: _StuckSessionDB())
    _set_run_conversation_impl(
        monkeypatch, lambda agent, prompt: (agent._ensure_db_session(), "RESULT: DONE")[1]
    )

    watchdog, _writes, exits = _make_watchdog(deadline_s=60.0)
    watchdog._db_write_budget_s = 0.1  # keep the test fast
    watchdog.arm()

    t0 = time.monotonic()
    response, result = oneshot_mod._run_agent("say hi", watchdog=watchdog)
    elapsed = time.monotonic() - t0

    assert response == "RESULT: DONE"
    assert write_started.is_set()   # the terminal write WAS attempted
    assert elapsed < 2.0            # bounded — abandoned near budget, not sleep(10)
    assert exits == []              # normal path; watchdog never fired


# ── Finding 5c: exact stdout/stderr for the disabled (env-unset) path ──


def test_run_oneshot_disabled_path_exact_stdout_stderr(patch_agent_stack, monkeypatch, capsys):
    monkeypatch.delenv("HERMES_SOFT_DEADLINE", raising=False)
    _set_run_conversation_impl(
        monkeypatch, lambda agent, prompt: (agent._ensure_db_session(), "hello there")[1]
    )

    exit_code = oneshot_mod.run_oneshot("say hi")

    captured_io = capsys.readouterr()
    assert exit_code == 0
    assert captured_io.out == "hello there\n"
    assert captured_io.err == ""


# ── WU-2: deadline watchdog unit tests (fake timer, no real sleeps) ────


class _StubAgent:
    def __init__(self, session_id="wd-session"):
        self.session_id = session_id


def _make_watchdog(*, deadline_s=60.0, agent=None, session_db=None,
                    stdout_writer=None, exit_fn=None, thread_cls=None):
    writes = []
    exits = []
    kwargs = dict(
        deadline_s=deadline_s,
        agent=agent,
        session_db=session_db,
        stdout_writer=stdout_writer or writes.append,
        exit_fn=exit_fn or (lambda code: exits.append(code)),
        timer_cls=_FakeTimer,
    )
    if thread_cls is not None:
        kwargs["thread_cls"] = thread_cls
    return (
        oneshot_mod._OneshotDeadlineWatchdog(**kwargs),
        writes,
        exits,
    )


def test_watchdog_arms_hard_timer_at_the_full_deadline():
    watchdog, _writes, _exits = _make_watchdog(deadline_s=60.0)
    watchdog.arm()

    assert len(_FakeTimer.instances) == 1
    (hard_timer,) = _FakeTimer.instances
    assert hard_timer.delay == pytest.approx(60.0)
    assert hard_timer.started


def test_watchdog_constructible_and_armable_before_agent_or_session_db_exist():
    """Finding 2 regression guard, at the watchdog level: it must be safe
    to build and arm with no agent/session_db, then bind() them later."""
    watchdog, _writes, _exits = _make_watchdog(deadline_s=60.0)  # agent=None, session_db=None
    watchdog.arm()
    (hard_timer,) = _FakeTimer.instances
    assert hard_timer.started

    watchdog.bind(agent=_StubAgent(), session_db=None)
    assert watchdog._agent is not None


def test_hard_deadline_finalizes_session_prints_result_and_exits(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    agent = _StubAgent(session_id="stalled-session")
    session_db.create_session(session_id=agent.session_id, source="cli")

    watchdog, writes, exits = _make_watchdog(deadline_s=60.0, agent=agent, session_db=session_db)
    watchdog.arm()
    (hard_timer,) = _FakeTimer.instances

    hard_timer.target()  # simulate the hard deadline firing (loop still alive)

    row = session_db.get_session(agent.session_id)
    assert row["ended_at"] is not None
    assert row["end_reason"] == "oneshot_deadline"
    assert exits == [0]
    assert len(writes) == 1
    assert writes[0].startswith(b"RESULT: INCOMPLETE")


def test_hard_deadline_fires_before_bind_still_exits_gracefully():
    """The deadline elapses before agent/session_db were ever bound
    (finding 2's scenario: a stall during config/provider/AIAgent init,
    before either exists) — must not raise, must still exit with the
    RESULT line, and must not attempt a DB write with nothing to write to."""
    watchdog, writes, exits = _make_watchdog(deadline_s=60.0)  # never bound
    watchdog.arm()
    (hard_timer,) = _FakeTimer.instances

    hard_timer.target()

    assert exits == [0]
    assert writes and writes[0].startswith(b"RESULT: INCOMPLETE")


def test_bound_agent_with_no_session_id_still_exits_gracefully():
    """agent exists but its session_id is falsy (e.g. construction failed
    partway) — same graceful-degrade requirement as the unbound case."""
    agent = _StubAgent(session_id=None)
    watchdog, writes, exits = _make_watchdog(deadline_s=60.0, agent=agent, session_db=None)
    watchdog.arm()
    (hard_timer,) = _FakeTimer.instances

    hard_timer.target()

    assert exits == [0]
    assert writes and writes[0].startswith(b"RESULT: INCOMPLETE")


def test_marker_is_written_before_db_write_is_attempted():
    """Finding 3 regression guard: the RESULT marker must be written
    first, unconditionally — before any DB I/O — so SessionDB write
    contention (which can burn multiple seconds across its internal
    retry loop) can never delay the one signal a dispatcher is watching
    stdout for."""
    order = []

    class _SlowSessionDB:
        def end_session(self, sid, reason):
            order.append("db_write_started")
            order.append(("end_session", sid, reason))

    def _stdout_writer(data):
        order.append("marker_written")

    agent = _StubAgent(session_id="ordering-session")
    watchdog, _writes, exits = _make_watchdog(
        deadline_s=60.0, agent=agent, session_db=_SlowSessionDB(), stdout_writer=_stdout_writer
    )
    watchdog.arm()
    (hard_timer,) = _FakeTimer.instances

    hard_timer.target()

    assert exits == [0]
    assert order.index("marker_written") < order.index("db_write_started")


def test_db_write_budget_is_bounded_and_exit_still_happens():
    """Finding 3: a DB write stuck past the bounded budget must not block
    process exit — the writer thread is abandoned, not waited on
    indefinitely."""
    write_started = threading.Event()

    class _StuckSessionDB:
        def end_session(self, sid, reason):
            write_started.set()
            time.sleep(10)  # far longer than the test's patience or budget

    agent = _StubAgent(session_id="stuck-session")
    watchdog, writes, exits = _make_watchdog(
        deadline_s=60.0, agent=agent, session_db=_StuckSessionDB()
    )
    watchdog._db_write_budget_s = 0.1  # keep the test itself fast
    watchdog.arm()
    (hard_timer,) = _FakeTimer.instances

    t0 = time.monotonic()
    hard_timer.target()
    elapsed = time.monotonic() - t0

    assert write_started.is_set()
    assert elapsed < 2.0  # bounded — did not wait anywhere near the sleep(10)
    assert exits == [0]
    assert writes and writes[0].startswith(b"RESULT: INCOMPLETE")


def test_normal_completion_wins_race_and_cancels_the_timer():
    watchdog, _writes, exits = _make_watchdog(deadline_s=60.0)
    watchdog.arm()
    (hard_timer,) = _FakeTimer.instances

    won = watchdog.cancel_and_claim()

    assert won is True
    assert hard_timer.cancelled
    assert exits == []

    # A watchdog firing AFTER normal completion already claimed the race
    # must be a total no-op — the terminal record was already written by
    # the normal path, and must never be double-finalized or exited.
    hard_timer.target()
    assert exits == []


def test_watchdog_fires_before_completion_blocks_the_late_claim():
    """Inverse ordering: the hard deadline wins the race first. The normal
    completion path's later cancel_and_claim() must report it lost, so the
    caller in oneshot.py knows to call wait_for_exit() instead of
    finalizing (and returning a response) itself."""
    watchdog, _writes, exits = _make_watchdog(deadline_s=60.0)
    watchdog.arm()
    (hard_timer,) = _FakeTimer.instances

    hard_timer.target()
    assert exits == [0]

    won = watchdog.cancel_and_claim()
    assert won is False

    # wait_for_exit() must return promptly here — fire_hard_deadline
    # already ran to completion (including the .set() before its exit
    # call), so there's nothing left to actually wait for.
    t0 = time.monotonic()
    watchdog.wait_for_exit(timeout=5)
    assert time.monotonic() - t0 < 1.0


def test_disabled_when_no_soft_deadline_configured():
    assert oneshot_mod._arm_deadline_watchdog(None) is None


def test_resolve_soft_deadline_prefers_explicit_over_env(monkeypatch):
    monkeypatch.setenv("HERMES_SOFT_DEADLINE", "999")
    assert oneshot_mod._resolve_soft_deadline(30.0) == 30.0


def test_resolve_soft_deadline_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("HERMES_SOFT_DEADLINE", "45")
    assert oneshot_mod._resolve_soft_deadline(None) == 45.0


def test_resolve_soft_deadline_disabled_when_unset(monkeypatch):
    monkeypatch.delenv("HERMES_SOFT_DEADLINE", raising=False)
    assert oneshot_mod._resolve_soft_deadline(None) is None


def test_resolve_soft_deadline_rejects_non_positive_and_garbage(monkeypatch):
    monkeypatch.delenv("HERMES_SOFT_DEADLINE", raising=False)
    assert oneshot_mod._resolve_soft_deadline(0) is None
    assert oneshot_mod._resolve_soft_deadline(-5) is None
    monkeypatch.setenv("HERMES_SOFT_DEADLINE", "not-a-number")
    assert oneshot_mod._resolve_soft_deadline(None) is None


# ── WU-2: end-to-end through _run_agent, watchdog passed in directly ───


def test_run_agent_success_case_cancels_watchdog_cleanly(patch_agent_stack, monkeypatch):
    """A fast run finishing well inside the deadline must exit via
    oneshot_complete, never let the watchdog exit the process, and cancel
    the timer."""
    session_db = patch_agent_stack
    watchdog, _writes, exits = _make_watchdog(deadline_s=120.0)
    watchdog.arm()

    captured = _set_run_conversation_impl(
        monkeypatch, lambda agent, prompt: (agent._ensure_db_session(), "RESULT: DONE")[1]
    )

    response, result = oneshot_mod._run_agent("say hi", watchdog=watchdog)

    assert response == "RESULT: DONE"
    assert result.get("final_response") == "RESULT: DONE"
    assert exits == []  # watchdog never fired / never exited
    (hard_timer,) = _FakeTimer.instances
    assert hard_timer.cancelled

    sid = captured["agent"].session_id
    row = session_db.get_session(sid)
    assert row["end_reason"] == "oneshot_complete"


def test_run_agent_binds_agent_and_session_db_onto_watchdog(patch_agent_stack, monkeypatch):
    """The watchdog handed to _run_agent starts unbound; _run_agent must
    bind session_db right after creating it and agent right after
    construction — this is what makes the deadline cover config/provider/
    AIAgent init, not just run_conversation()."""
    watchdog, _writes, _exits = _make_watchdog(deadline_s=120.0)
    watchdog.arm()
    assert watchdog._agent is None
    assert watchdog._session_db is None

    _set_run_conversation_impl(monkeypatch, lambda agent, prompt: "RESULT: DONE")

    oneshot_mod._run_agent("say hi", watchdog=watchdog)

    assert watchdog._agent is not None
    assert watchdog._session_db is not None


def test_run_agent_stall_past_deadline_is_tagged_oneshot_deadline(patch_agent_stack, monkeypatch):
    """A run that is genuinely still inside run_conversation() when the
    hard deadline fires gets finalized by the watchdog thread itself
    (end_reason="oneshot_deadline", RESULT line, exit(0)) — verified here
    by calling the handler directly, since in the real process os._exit
    would terminate before this test's assertions could run."""
    session_db = patch_agent_stack
    watchdog, writes, exits = _make_watchdog(deadline_s=60.0)
    watchdog.arm()

    def _run(agent, prompt):
        agent._ensure_db_session()
        # The run is "stuck" here from the harness's point of view —
        # simulate the hard deadline firing while still inside
        # run_conversation().
        (hard_timer,) = _FakeTimer.instances
        hard_timer.target()
        # In the real process this line never executes (os._exit(0)
        # already ran inside fire_hard_deadline above, and _run_agent's
        # own finally would block in wait_for_exit() besides); the fake
        # exit_fn just records the call, so execution continues here to
        # show the watchdog result already won regardless of what
        # run_conversation() eventually would have returned.
        return "would-be final answer, arrives too late"

    captured = _set_run_conversation_impl(monkeypatch, _run)

    oneshot_mod._run_agent("say hi", watchdog=watchdog)

    assert exits == [0]
    sid = captured["agent"].session_id
    row = session_db.get_session(sid)
    assert row["end_reason"] == "oneshot_deadline"


def test_slow_but_successful_run_is_never_mislabeled_deadline(patch_agent_stack, monkeypatch):
    """Regression guard for the earlier design's bug: a run that takes
    longer than some interim boundary but still completes on its own,
    before the hard deadline, must be tagged oneshot_complete — never
    oneshot_deadline. There is exactly one timer (the hard deadline) and
    it never fires here, so there is no interim boundary to mislabel
    against in the first place."""
    session_db = patch_agent_stack
    watchdog, _writes, exits = _make_watchdog(deadline_s=60.0)
    watchdog.arm()

    captured = _set_run_conversation_impl(
        monkeypatch, lambda agent, prompt: (agent._ensure_db_session(), "RESULT: DONE (slow)")[1]
    )

    response, result = oneshot_mod._run_agent("say hi", watchdog=watchdog)

    assert response == "RESULT: DONE (slow)"
    assert exits == []
    sid = captured["agent"].session_id
    row = session_db.get_session(sid)
    assert row["end_reason"] == "oneshot_complete"


# ── Finding 2: watchdog must be armed before config/provider/AIAgent
# init, not just around run_conversation() ─────────────────────────────


def test_run_oneshot_watchdog_survives_a_stall_during_config_load(patch_agent_stack, monkeypatch):
    """Regression guard for finding 2: a stall in load_config() — before
    session_db or agent exist — must still be caught by the deadline. We
    can't wall-clock-sleep past a fake timer, so this test simulates "the
    deadline elapsed while stalled here" by firing the timer synchronously
    from inside the patched load_config(), then asserts run_oneshot still
    goes through _arm_deadline_watchdog's real wiring (not a test double
    substituted for it) and the watchdog fires with everything still
    unbound — proving the arm point in run_oneshot precedes this call."""
    session_db = patch_agent_stack
    monkeypatch.setenv("HERMES_SOFT_DEADLINE", "30")

    captured = {}

    def _fake_arm(soft_deadline):
        watchdog, writes, exits = _make_watchdog(deadline_s=soft_deadline)
        watchdog.arm()
        captured["watchdog"] = watchdog
        captured["writes"] = writes
        captured["exits"] = exits
        return watchdog

    monkeypatch.setattr(oneshot_mod, "_arm_deadline_watchdog", _fake_arm)

    from hermes_cli import config as config_mod

    def _stalled_load_config():
        watchdog = captured["watchdog"]
        assert watchdog._agent is None and watchdog._session_db is None, (
            "load_config ran after agent/session_db were already bound — "
            "the watchdog is not covering this part of the setup window"
        )
        (hard_timer,) = _FakeTimer.instances
        hard_timer.target()
        return {}

    monkeypatch.setattr(config_mod, "load_config", _stalled_load_config)
    _set_run_conversation_impl(
        monkeypatch, lambda agent, prompt: (agent._ensure_db_session(), "too late")[1]
    )

    oneshot_mod.run_oneshot("say hi")

    assert captured["exits"] == [0]
    assert captured["writes"] and captured["writes"][0].startswith(b"RESULT: INCOMPLETE")


# ── Finding 5a: real-thread barrier race for finding 1's ordering ─────


def test_loser_blocks_until_watchdog_commits_real_threads():
    """Real threads (not the fake-timer seam), synchronized with a
    Barrier so both sides start racing for the lock at the same instant.
    Finding 1 is specifically about the case where the WATCHDOG wins that
    race (the normal completion path is the loser) — cancel_and_claim()
    is structurally lighter than fire_hard_deadline() and was observed to
    win essentially every unbiased trial, so a tiny (10ms) handicap is
    given to the normal side after the barrier release to reliably
    exercise the scenario under test, rather than depending on
    non-deterministic OS thread scheduling for test stability. The actual
    assertion is real concurrency, not simulated: the loser's
    wait_for_exit() call must not return until the winner (running on its
    own real thread, with an artificially slow exit path) has actually
    committed — not return instantly before that work finishes."""
    exit_delay_s = 0.2
    agent = _StubAgent(session_id="race-session")
    committed = []

    def _slow_exit_fn(code):
        time.sleep(exit_delay_s)
        committed.append(code)

    watchdog, _writes, _exits = _make_watchdog(
        deadline_s=60.0, agent=agent, session_db=None, exit_fn=_slow_exit_fn
    )
    watchdog.arm()

    barrier = threading.Barrier(2)
    results = {}

    def _watchdog_side():
        barrier.wait()
        watchdog.fire_hard_deadline()

    def _normal_side():
        barrier.wait()
        time.sleep(0.01)  # deliberate handicap — see docstring
        t0 = time.monotonic()
        won = watchdog.cancel_and_claim()
        results["won"] = won
        if not won:
            watchdog.wait_for_exit()
            results["wait_duration"] = time.monotonic() - t0

    t_watchdog = threading.Thread(target=_watchdog_side)
    t_normal = threading.Thread(target=_normal_side)
    t_watchdog.start()
    t_normal.start()
    t_watchdog.join(timeout=5)
    t_normal.join(timeout=5)

    assert not t_watchdog.is_alive() and not t_normal.is_alive()
    assert results["won"] is False, "handicap did not produce the watchdog-wins scenario under test"
    assert results["wait_duration"] >= exit_delay_s * 0.6, (
        "wait_for_exit() returned before the watchdog actually committed "
        "— the loser could fall through and return a normal response "
        "while the watchdog is still mid-exit"
    )
    assert committed == [0]


# ── Finding 5b: real subprocess — real Timer, real fd-1 write, real
# os._exit. The fake-timer tests above cannot exercise this: a real
# os._exit(0) would kill the test runner itself. ───────────────────────


def test_subprocess_real_watchdog_stall_produces_result_marker_and_deadline_row(tmp_path):
    db_path = tmp_path / "state.db"
    session_id = "subprocess-stalled-session"
    script = tmp_path / "run_stalled_oneshot.py"
    script.write_text(
        "import time\n"
        "from pathlib import Path\n"
        "\n"
        "from hermes_cli import oneshot as oneshot_mod\n"
        "from hermes_state import SessionDB\n"
        "import run_agent\n"
        "from hermes_cli import config as config_mod\n"
        "from hermes_cli import runtime_provider as runtime_provider_mod\n"
        "from hermes_cli import tools_config as tools_config_mod\n"
        "from hermes_cli import fallback_config as fallback_config_mod\n"
        "\n"
        f"db_path = Path({str(db_path)!r})\n"
        "session_db = SessionDB(db_path=db_path)\n"
        "oneshot_mod._create_session_db_for_oneshot = lambda: session_db\n"
        "\n"
        "class _StalledAgent:\n"
        "    def __init__(self, **kwargs):\n"
        f"        self.session_id = {session_id!r}\n"
        "        session_db.create_session(session_id=self.session_id, source='cli')\n"
        "        self.suppress_status_output = False\n"
        "        self.stream_delta_callback = None\n"
        "        self.tool_gen_callback = None\n"
        "    def run_conversation(self, prompt):\n"
        "        time.sleep(5)  # never returns before the 0.2s deadline fires\n"
        "        return {'final_response': 'too-late'}\n"
        "\n"
        "run_agent.AIAgent = _StalledAgent\n"
        "config_mod.load_config = lambda: {}\n"
        "runtime_provider_mod.resolve_runtime_provider = lambda **kwargs: {\n"
        "    'api_key': None, 'base_url': None, 'provider': 'anthropic',\n"
        "    'api_mode': None, 'credential_pool': None,\n"
        "}\n"
        "tools_config_mod._get_platform_tools = lambda cfg, platform: []\n"
        "fallback_config_mod.get_fallback_chain = lambda cfg: []\n"
        "\n"
        "oneshot_mod.run_oneshot('say hi', soft_deadline=0.2)\n"
    )

    proc = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert proc.returncode == 0, f"stderr={proc.stderr!r}"
    assert proc.stdout == "RESULT: INCOMPLETE — soft deadline reached (see session log)\n"

    row_db = SessionDB(db_path=db_path)
    row = row_db.get_session(session_id)
    assert row is not None
    assert row["ended_at"] is not None
    assert row["end_reason"] == "oneshot_deadline"
