"""Oneshot (-z) mode: send a prompt, get the final content block, exit.

Bypasses cli.py entirely.  No banner, no spinner, no session_id line,
no stderr chatter.  Just the agent's final text to stdout.

Toolsets = explicit --toolsets when provided, otherwise whatever the user has
configured for "cli" in `hermes tools`.
Rules / memory / AGENTS.md / preloaded skills = same as a normal chat turn.
Approvals = auto-bypassed (HERMES_YOLO_MODE=1 is set for the call).
Working directory = the user's CWD (AGENTS.md etc. resolve from there as usual).

Model / provider selection mirrors `hermes chat`:
    - Both optional. If omitted, use the user's configured default.
    - If both given, pair them exactly as given.
    - If only --model given, auto-detect the provider that serves it.
    - If only --provider given, error out (ambiguous — caller must pick a model).

Env var fallbacks (used when the corresponding arg is not passed):
    - HERMES_INFERENCE_MODEL
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Callable, Optional

from hermes_cli.fallback_config import get_fallback_chain


def _normalize_toolsets(toolsets: object = None) -> list[str] | None:
    if not toolsets:
        return None

    raw_items = [toolsets] if isinstance(toolsets, str) else toolsets
    if not isinstance(raw_items, (list, tuple)):
        raw_items = [raw_items]

    normalized: list[str] = []
    for item in raw_items:
        if isinstance(item, str):
            normalized.extend(part.strip() for part in item.split(","))
        else:
            normalized.append(str(item).strip())

    return [item for item in normalized if item] or None


def _validate_explicit_toolsets(toolsets: object = None) -> tuple[list[str] | None, str | None]:
    normalized = _normalize_toolsets(toolsets)
    if normalized is None:
        return None, None

    try:
        from toolsets import validate_toolset
    except Exception as exc:
        return None, f"hermes -z: failed to validate --toolsets: {exc}\n"

    built_in = [name for name in normalized if validate_toolset(name)]
    unresolved = [name for name in normalized if name not in built_in]

    if unresolved:
        try:
            from hermes_cli.plugins import discover_plugins

            discover_plugins()
            plugin_valid = [name for name in unresolved if validate_toolset(name)]
        except Exception:
            plugin_valid = []

        if plugin_valid:
            built_in.extend(plugin_valid)
            unresolved = [name for name in unresolved if name not in plugin_valid]

    if any(name in {"all", "*"} for name in built_in):
        ignored = [name for name in normalized if name not in {"all", "*"}]
        if ignored:
            sys.stderr.write(
                "hermes -z: --toolsets all enables every toolset; "
                f"ignoring additional entries: {', '.join(ignored)}\n"
            )
        return None, None

    mcp_names: set[str] = set()
    mcp_disabled: set[str] = set()
    if unresolved:
        try:
            from hermes_cli.config import read_raw_config
            from hermes_cli.tools_config import _parse_enabled_flag

            cfg = read_raw_config()
            mcp_servers = cfg.get("mcp_servers") if isinstance(cfg.get("mcp_servers"), dict) else {}
            for name, server_cfg in mcp_servers.items():
                if not isinstance(server_cfg, dict):
                    continue
                if _parse_enabled_flag(server_cfg.get("enabled", True), default=True):
                    mcp_names.add(str(name))
                else:
                    mcp_disabled.add(str(name))
        except Exception:
            mcp_names = set()
            mcp_disabled = set()

    mcp_valid = [name for name in unresolved if name in mcp_names]
    disabled = [name for name in unresolved if name in mcp_disabled]
    unknown = [name for name in unresolved if name not in mcp_names and name not in mcp_disabled]
    valid = built_in + mcp_valid

    if unknown:
        sys.stderr.write(f"hermes -z: ignoring unknown --toolsets entries: {', '.join(unknown)}\n")
    if disabled:
        sys.stderr.write(
            "hermes -z: ignoring disabled MCP servers (set enabled: true in config.yaml to use): "
            f"{', '.join(disabled)}\n"
        )

    if not valid:
        return None, "hermes -z: --toolsets did not contain any valid toolsets.\n"

    return valid, None


def _write_usage_file(path: Optional[str], result: dict, failure: Optional[str] = None) -> None:
    """Best-effort JSON usage report for pipelines (``-z --usage-file``).

    Written even on failure so callers can always account for spend. Never
    raises — a broken usage write must not mask the run's own outcome.
    """
    if not path:
        return
    try:
        import json

        report = {
            "estimated_cost_usd": result.get("estimated_cost_usd"),
            "cost_status": result.get("cost_status"),
            "cost_source": result.get("cost_source"),
            "input_tokens": result.get("input_tokens"),
            "output_tokens": result.get("output_tokens"),
            "cache_read_tokens": result.get("cache_read_tokens"),
            "cache_write_tokens": result.get("cache_write_tokens"),
            "reasoning_tokens": result.get("reasoning_tokens"),
            "total_tokens": result.get("total_tokens"),
            "api_calls": result.get("api_calls"),
            "model": result.get("model"),
            "provider": result.get("provider"),
            "session_id": result.get("session_id"),
            "completed": result.get("completed"),
            "failed": bool(result.get("failed")) or failure is not None,
        }
        if failure is not None:
            report["failure"] = failure
        out = Path(path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except Exception:
        pass


def run_oneshot(
    prompt: str,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    toolsets: object = None,
    usage_file: Optional[str] = None,
    soft_deadline: Optional[float] = None,
) -> int:
    """Execute a single prompt and print only the final content block.

    Args:
        prompt: The user message to send.
        model: Optional model override. Falls back to HERMES_INFERENCE_MODEL
            env var, then config.yaml's model.default / model.model.
        provider: Optional provider override. Falls back to config.yaml's
            model.provider, then "auto".
        toolsets: Optional comma-separated string or iterable of toolsets.
        usage_file: Optional path; when set, a JSON usage report (estimated
            cost, token counts, model, api_calls) is written there after the
            run — even when the run fails — so pipelines can account for
            spend per invocation.
        soft_deadline: Optional wall-clock budget in seconds for the whole
            run. Opt-in only — falls back to the HERMES_SOFT_DEADLINE env
            var when not passed, and is a no-op when neither is set. See
            ``_arm_deadline_watchdog`` for the enforcement mechanism.

    Returns the exit code.  Caller should sys.exit() with the return.
    """
    # Armed FIRST, before anything else in this function — including the
    # validation checks and env-var setup just below, and (inside
    # _run_agent) config load, provider resolution, and AIAgent
    # construction. Any of those can stall exactly like a stuck tool call
    # or provider request can; arming only around agent.run_conversation()
    # would leave that whole setup window unprotected. watchdog is None (a
    # pure no-op everywhere below) when the feature isn't opted into.
    watchdog = _arm_deadline_watchdog(_resolve_soft_deadline(soft_deadline))
    try:
        return _run_oneshot_body(
            prompt,
            model=model,
            provider=provider,
            toolsets=toolsets,
            usage_file=usage_file,
            watchdog=watchdog,
        )
    finally:
        # No-op if _run_agent's own race already resolved this (the normal
        # case). Only matters for early-return paths below that never
        # reached _run_agent at all — otherwise this timer would be left
        # armed and could fire later in whatever process outlives this
        # call, silently overriding its exit with RESULT: INCOMPLETE.
        if watchdog is not None:
            watchdog.cancel_and_claim()


def _run_oneshot_body(
    prompt: str,
    *,
    model: Optional[str],
    provider: Optional[str],
    toolsets: object,
    usage_file: Optional[str],
    watchdog,
) -> int:
    """The original run_oneshot() body, factored out so run_oneshot() can
    guarantee the watchdog is cleaned up via a single outer finally
    regardless of which return path below is taken."""
    # Silence every stdlib logger for the duration.  AIAgent, tools, and
    # provider adapters all log to stderr through the root logger; file
    # handlers added by setup_logging() keep working (they're attached to
    # the root logger's handler list, not affected by level), but no
    # bytes reach the terminal.
    logging.disable(logging.CRITICAL)

    # --provider without --model is ambiguous: carrying the user's configured
    # model across to a different provider is usually wrong (that provider may
    # not host it), and silently picking the provider's catalog default hides
    # the mismatch.  Require the caller to be explicit.  Validate BEFORE the
    # stderr redirect so the message actually reaches the terminal.
    env_model_early = os.getenv("HERMES_INFERENCE_MODEL", "").strip()
    if provider and not ((model or "").strip() or env_model_early):
        sys.stderr.write(
            "hermes -z: --provider requires --model (or HERMES_INFERENCE_MODEL). "
            "Pass both explicitly, or neither to use your configured defaults.\n"
        )
        return 2

    explicit_toolsets, toolsets_error = _validate_explicit_toolsets(toolsets)
    if toolsets_error:
        sys.stderr.write(toolsets_error)
        return 2
    use_config_toolsets = _normalize_toolsets(toolsets) is None

    # Auto-approve any shell / tool approvals.  Non-interactive by
    # definition — a prompt would hang forever.
    os.environ["HERMES_YOLO_MODE"] = "1"
    os.environ["HERMES_ACCEPT_HOOKS"] = "1"

    # Redirect stderr AND stdout to devnull for the entire call tree.
    # We'll print the final response to the real stdout at the end.
    real_stdout = sys.stdout
    real_stderr = sys.stderr
    devnull = open(os.devnull, "w", encoding="utf-8")

    response: Optional[str] = None
    result: dict = {}
    failure: BaseException | None = None
    try:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            try:
                response, result = _run_agent(
                    prompt,
                    model=model,
                    provider=provider,
                    toolsets=explicit_toolsets,
                    use_config_toolsets=use_config_toolsets,
                    watchdog=watchdog,
                )
            except BaseException as exc:  # noqa: BLE001
                # Capture anything that escapes the agent (including OSError
                # from prompt_toolkit/Vt100 when stdout is a non-TTY pipe,
                # KeyboardInterrupt, SystemExit, etc.) so we can surface it on
                # the real stderr instead of crashing past the redirect with a
                # traceback that the caller never sees. A silent exit in a
                # cron / SSH / subprocess context is the worst failure mode.
                # See #30623.
                failure = exc
    finally:
        try:
            devnull.close()
        except Exception:
            pass

    if failure is not None:
        # Re-raise control-flow exceptions so the parent handles them as usual
        # (Ctrl-C / explicit sys.exit() inside the agent).
        if isinstance(failure, (KeyboardInterrupt, SystemExit)):
            _write_usage_file(usage_file, result, failure=repr(failure))
            raise failure
        _write_usage_file(usage_file, result, failure=str(failure))
        real_stderr.write(f"hermes -z: agent failed: {failure}\n")
        real_stderr.flush()
        return 1

    _write_usage_file(usage_file, result)

    if response:
        real_stdout.write(response)
        if not response.endswith("\n"):
            real_stdout.write("\n")
        real_stdout.flush()

    if (result.get("failed") or result.get("partial")) and not (response or "").strip():
        return 2

    if not (response or "").strip():
        real_stderr.write("hermes -z: no final response was produced; treating the run as failed.\n")
        real_stderr.flush()
        return 1

    return 0


def _create_session_db_for_oneshot():
    """Best-effort SessionDB for ``hermes -z`` / oneshot mode.

    Oneshot bypasses ``HermesCLI._init_agent()``, so it must wire the SQLite
    session store itself. Without this, the ``session_search``/recall tool is
    advertised but every call returns "Session database not available.".
    """
    try:
        from hermes_state import SessionDB

        return SessionDB()
    except Exception as exc:
        logging.debug("SQLite session store not available for oneshot mode: %s", exc)
        return None


def _resolve_soft_deadline(explicit: Optional[float]) -> Optional[float]:
    """Resolve the effective soft-deadline budget in seconds.

    Opt-in only: an explicit value (e.g. a future --soft-deadline flag)
    wins; otherwise HERMES_SOFT_DEADLINE; otherwise disabled (None).
    A non-numeric or non-positive value disables the feature rather than
    raising, so a malformed env var never turns into a hard failure.
    """
    candidate = explicit
    if candidate is None:
        raw = os.getenv("HERMES_SOFT_DEADLINE", "").strip()
        if not raw:
            return None
        candidate = raw

    try:
        value = float(candidate)
    except (TypeError, ValueError):
        logging.debug("oneshot: ignoring invalid soft deadline %r", candidate)
        return None

    return value if value > 0 else None


class _OneshotDeadlineWatchdog:
    """Pure-code wall-clock deadline enforcement around
    ``agent.run_conversation()``.

    One timer, armed at ``deadline`` (relative to the moment the run
    started): an unconditional backstop. If the run is still alive when it
    fires, this forces a terminal session record and exits the process —
    a guarantee that no blocking call can ever suppress the graceful
    record, independent of the model, the loop, or any cooperative check.

    A dedicated "finalize boundary" cooperative-interrupt timer was
    considered and deliberately dropped: the agent loop has no hook that
    lets an external caller inject a graceful finalize turn, so the only
    thing a cooperative interrupt could do here was cut the loop off early
    and mislabel a run that was going to complete successfully (just
    slowly) as ``oneshot_deadline`` — non-retryable at the dispatch layer,
    i.e. it would turn successful slow runs into permanent failures. The
    real finalize turn and a dynamic tool-call clamp both need a hook
    inside the agent loop itself; deliberately deferred to the dispatch
    layer (larger margins, incomplete-run detection) rather than faked
    here.

    Deliberately constructible with ``agent``/``session_db`` unbound: the
    caller arms this BEFORE building either (config load, provider
    resolution, and AIAgent construction can all stall too — a silent
    death there is exactly as bad as one inside ``run_conversation()``),
    then calls ``bind()`` once they exist. ``fire_hard_deadline`` degrades
    gracefully if the deadline fires before ``bind()`` ever ran: it still
    writes the RESULT marker and exits, it just has no session row to
    finalize.

    A single lock-protected flag decides exactly one winner between the
    normal completion path and the watchdog: whichever reaches the flag
    first "wins" and the other becomes a no-op. See ``cancel_and_claim``
    (normal path) and ``fire_hard_deadline`` (watchdog path). The losing
    side must call ``wait_for_exit()`` and never return a normal response
    afterward — see that method's docstring for why.
    """

    #: Cap on how long fire_hard_deadline will wait for end_session() to
    #: land before giving up on the DB write and exiting anyway.
    #: SessionDB's internal write-retry loop can burn up to ~1s of SQLite
    #: busy-timeout plus 20-150ms jitter PER attempt, times up to 15
    #: attempts under lock contention — multiple seconds that must never
    #: delay the one signal (the RESULT marker) a human or dispatcher is
    #: actually watching for.
    _DEFAULT_DB_WRITE_BUDGET_S = 2.0

    def __init__(
        self,
        *,
        deadline_s: float,
        agent=None,
        session_db=None,
        stdout_writer: Callable[[bytes], None] = lambda data: os.write(1, data),
        exit_fn: Callable[[int], None] = os._exit,
        timer_cls=threading.Timer,
        thread_cls=threading.Thread,
        db_write_budget_s: float = _DEFAULT_DB_WRITE_BUDGET_S,
    ) -> None:
        self._deadline_s = max(0.0, deadline_s)
        self._agent = agent
        self._session_db = session_db
        self._stdout_writer = stdout_writer
        self._exit_fn = exit_fn
        self._timer_cls = timer_cls
        self._thread_cls = thread_cls
        self._db_write_budget_s = db_write_budget_s

        self._lock = threading.Lock()
        self._finalized = False
        self._exit_committed = threading.Event()
        self._hard_timer = None

    def bind(self, *, agent=None, session_db=None) -> None:
        """Attach the real agent/session_db once they exist.

        Safe to call any time after ``arm()`` — including after the
        deadline has already fired, in which case it's simply a no-op
        (there's nothing left to finalize with the newly-bound refs; the
        watchdog already committed to exiting without them).
        """
        with self._lock:
            if agent is not None:
                self._agent = agent
            if session_db is not None:
                self._session_db = session_db

    def arm(self) -> None:
        self._hard_timer = self._timer_cls(self._deadline_s, self.fire_hard_deadline)
        self._hard_timer.daemon = True
        self._hard_timer.start()

    def fire_hard_deadline(self) -> None:
        """Unconditional, race-guarded terminal action.

        Exposed as a plain method (not just a Timer callback) so tests can
        invoke it synchronously without a real timer or sleep.
        """
        with self._lock:
            if self._finalized:
                return
            self._finalized = True
            agent = self._agent
            session_db = self._session_db

        # Marker first, unconditionally, before any DB I/O — see
        # _DEFAULT_DB_WRITE_BUDGET_S for why the DB write must never gate
        # this line.
        try:
            self._stdout_writer(b"RESULT: INCOMPLETE \xe2\x80\x94 soft deadline reached (see session log)\n")
        except Exception:
            logging.debug("oneshot watchdog: failed to write RESULT line", exc_info=True)

        sid = getattr(agent, "session_id", None) if agent is not None else None
        if session_db is not None and sid:
            self._attempt_bounded_end_session(session_db, sid)

        self._exit_fn(0)
        # In production (real os._exit) this line never runs — os._exit is
        # a direct kernel call that terminates the whole process, so
        # nothing after it ever executes, and nothing needs to: any waiter
        # blocked in wait_for_exit() is killed outright along with
        # everything else. Setting the event AFTER exit_fn (not before)
        # only matters for a faked exit_fn (tests): it means a waiter
        # can't unblock and proceed until exit_fn has actually returned —
        # including however long it took — rather than the instant the
        # marker/DB-write work above finished.
        self._exit_committed.set()

    def _attempt_bounded_end_session(self, session_db, sid, reason="oneshot_deadline") -> None:
        """Best-effort end_session(), capped so DB contention can never
        meaningfully delay process exit. A writer thread still stuck past
        the budget is simply abandoned — the process is exiting either way.

        Shared by the watchdog's own deadline path (reason="oneshot_deadline")
        and the NORMAL completion path via ``finalize_session_bounded``."""
        done = threading.Event()

        def _write():
            try:
                session_db.end_session(sid, reason)
            except Exception:
                logging.debug("oneshot watchdog: end_session failed", exc_info=True)
            finally:
                done.set()

        writer = self._thread_cls(target=_write, daemon=True)
        writer.start()
        done.wait(timeout=self._db_write_budget_s)

    def cancel_and_claim(self) -> bool:
        """Called by the normal completion path.

        Returns True if this call won the race (the hard deadline had not
        already fired), meaning the caller is responsible for finalizing
        the session itself. Returns False if the watchdog got there first
        — the caller MUST then call ``wait_for_exit()`` and never return a
        normal response.
        """
        with self._lock:
            if self._finalized:
                return False
            self._finalized = True

        if self._hard_timer is not None:
            self._hard_timer.cancel()
        return True

    def finalize_session_bounded(self, session_db, sid, reason) -> None:
        """Bounded terminal write for the NORMAL completion path.

        ``cancel_and_claim`` has already cancelled the hard timer, so a plain
        ``end_session`` here would run with NO deadline backstop: a SQLite-lock
        hang could push a normal completion past the soft deadline into the
        external dispatcher's SIGKILL, leaving ``end_reason`` NULL — the exact
        failure this watchdog exists to prevent. Cap the write with the same
        budget as the watchdog's own deadline write so a contended write is
        abandoned in favour of a graceful, marker-clean exit."""
        self._attempt_bounded_end_session(session_db, sid, reason)

    def wait_for_exit(self, timeout: Optional[float] = None) -> None:
        """Block until ``fire_hard_deadline`` (running on the timer
        thread) has committed to exiting.

        Called by the loser of ``cancel_and_claim()``'s race. Without
        this, a run that loses the race could still fall through to
        `return`ing its (possibly complete) response from ``_run_agent``,
        which ``run_oneshot`` would then print and exit through the
        ordinary path — racing the watchdog's own RESULT-marker-then-exit
        for real stdout and a real exit code. Blocking here means the
        watchdog's outcome is the only one that is ever observed; in
        production the process is force-exited out from under this wait
        very shortly after, so blocking with no timeout is safe (the
        watchdog's own bounded DB-write budget caps how long that takes).
        """
        self._exit_committed.wait(timeout=timeout)


def _arm_deadline_watchdog(soft_deadline: Optional[float]):
    """Build and arm a deadline watchdog, or return None when disabled.

    Deliberately built with no agent/session_db yet — call ``bind()`` on
    the result once those exist. Must be armed before config load,
    provider resolution, and AIAgent construction, not just around
    ``agent.run_conversation()`` — see the class docstring.
    """
    if soft_deadline is None:
        return None
    watchdog = _OneshotDeadlineWatchdog(deadline_s=soft_deadline)
    watchdog.arm()
    return watchdog


def _run_agent(
    prompt: str,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    toolsets: object = None,
    use_config_toolsets: bool = True,
    watchdog=None,
) -> tuple[str, dict]:
    """Build an AIAgent exactly like a normal CLI chat turn would, then
    run a single conversation.  Returns ``(final_response, run_result)``.

    ``watchdog`` (an already-armed ``_OneshotDeadlineWatchdog`` or None) is
    built and armed by the caller BEFORE any of the config/provider/
    AIAgent construction below — arming it only around
    ``agent.run_conversation()`` would leave that whole setup window
    (config load, provider resolution, SessionDB, AIAgent init)
    unprotected, exactly the kind of stall this exists to catch.
    ``watchdog.bind()`` attaches the real agent/session_db refs as they
    come into existence.
    """
    # Imports are local so they don't run when hermes is invoked for
    # other commands (keeps top-level CLI startup cheap).
    from hermes_cli.config import load_config
    from hermes_cli.models import detect_provider_for_model
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from hermes_cli.tools_config import _get_platform_tools
    from run_agent import AIAgent

    agent = None
    session_db = None
    session_end_reason = "oneshot_complete"
    try:
        cfg = load_config()

        # Resolve effective model: explicit arg → env var → config.
        model_cfg = cfg.get("model") or {}
        if isinstance(model_cfg, str):
            cfg_model = model_cfg
        else:
            cfg_model = model_cfg.get("default") or model_cfg.get("model") or ""

        env_model = os.getenv("HERMES_INFERENCE_MODEL", "").strip()
        effective_model = (model or "").strip() or env_model or cfg_model

        # Resolve effective provider: explicit arg → (auto-detect from model if
        # model was explicit) → env / config (handled inside resolve_runtime_provider).
        #
        # When --model is given without --provider, auto-detect the provider that
        # serves that model — same semantic as `/model <name>` in an interactive
        # session.  Without this, resolve_runtime_provider() would fall back to
        # the user's configured default provider, which may not host the model
        # the caller just asked for.
        effective_provider = (provider or "").strip() or None
        explicit_base_url_from_alias: Optional[str] = None
        if effective_provider is None and (model or env_model):
            # Only auto-detect when the model was explicitly requested via arg or
            # env var (not when it came from config — that's the "use my defaults"
            # path and the configured provider is already correct).
            explicit_model = (model or "").strip() or env_model
            if explicit_model:
                # First check DIRECT_ALIASES populated from config.yaml `model_aliases:`.
                # These map a user-defined alias to (model, provider, base_url) for
                # endpoints not in any catalog (local servers, custom proxies, etc.).
                try:
                    from hermes_cli import model_switch as _ms
                    _ms._ensure_direct_aliases()
                    direct = _ms.DIRECT_ALIASES.get(explicit_model.strip().lower())
                except Exception:
                    direct = None
                if direct is not None:
                    effective_model = direct.model
                    effective_provider = direct.provider
                    if direct.base_url:
                        explicit_base_url_from_alias = direct.base_url.rstrip("/")
                else:
                    cfg_provider = ""
                    if isinstance(model_cfg, dict):
                        cfg_provider = str(model_cfg.get("provider") or "").strip().lower()
                    current_provider = (
                        cfg_provider
                        or os.getenv("HERMES_INFERENCE_PROVIDER", "").strip().lower()
                        or "auto"
                    )
                    detected = detect_provider_for_model(explicit_model, current_provider)
                    if detected:
                        effective_provider, effective_model = detected

        runtime = resolve_runtime_provider(
            requested=effective_provider,
            target_model=effective_model or None,
            explicit_base_url=explicit_base_url_from_alias,
        )

        # Pull in explicit toolsets when provided; otherwise use whatever the user
        # has enabled for "cli". sorted() gives stable ordering for config-derived
        # sets; explicit values preserve user order.
        toolsets_list = _normalize_toolsets(toolsets)
        if toolsets_list is None and use_config_toolsets:
            toolsets_list = sorted(_get_platform_tools(cfg, "cli"))

        session_db = _create_session_db_for_oneshot()
        if watchdog is not None:
            watchdog.bind(session_db=session_db)
        # Read the effective fallback chain from profile config so oneshot workers
        # honour the same merge semantics as interactive CLI and gateway sessions.
        _fb = get_fallback_chain(cfg)

        agent = AIAgent(
            api_key=runtime.get("api_key"),
            base_url=runtime.get("base_url"),
            provider=runtime.get("provider"),
            api_mode=runtime.get("api_mode"),
            model=effective_model,
            enabled_toolsets=toolsets_list,
            quiet_mode=True,
            platform="cli",
            session_db=session_db,
            credential_pool=runtime.get("credential_pool"),
            fallback_model=_fb or None,
            # Interactive callbacks are intentionally NOT wired beyond this
            # one.  In oneshot mode there's no user sitting at a terminal:
            #   - clarify  → returns a synthetic "pick a default" instruction
            #                so the agent continues instead of stalling on
            #                the tool's built-in "not available" error
            #   - sudo password prompt → terminal_tool gates on
            #                HERMES_INTERACTIVE which we never set
            #   - shell-hook approval → auto-approved via HERMES_ACCEPT_HOOKS=1
            #                (set above); also falls back to deny on non-tty
            #   - dangerous-command approval → bypassed via HERMES_YOLO_MODE=1
            #   - skill secret capture → returns gracefully when no callback set
            clarify_callback=_oneshot_clarify_callback,
        )
        if watchdog is not None:
            watchdog.bind(agent=agent)

        # Belt-and-braces: make sure AIAgent doesn't invoke any streaming
        # display callbacks that would bypass our stdout capture.
        agent.suppress_status_output = True
        agent.stream_delta_callback = None
        agent.tool_gen_callback = None

        result = agent.run_conversation(prompt)
        response = result.get("final_response") or ""
        if not response.strip():
            # A blank response is not a clean completion: run_oneshot's own
            # empty-response check treats this as a failure. Recording
            # oneshot_complete here would tell dispatcher-side incomplete-
            # detection the opposite — that everything is fine — for a run
            # whose actual exit code says otherwise. Distinct from
            # oneshot_error (no exception was raised here; the model just
            # returned nothing).
            session_end_reason = "oneshot_empty"
        elif not result.get("completed", False):
            # Non-empty but the turn did NOT complete cleanly. run_conversation
            # (via turn_finalizer.finalize_turn) sets completed=False for
            # failed / interrupted / iteration-capped / output-truncated turns,
            # and such results still return non-empty text while carrying a
            # `failed`/`error` marker (e.g. "Invalid API response after N
            # retries…"). Recording oneshot_complete for these would tell
            # dispatcher-side incomplete-detection (qa_dispatch_store.
            # retry_decision) the run was fine and suppress a valid retry. No
            # exception was raised, so this is neither oneshot_error nor
            # oneshot_empty — mark it incomplete (retryable; NOT the
            # non-retryable oneshot_deadline contract).
            session_end_reason = "oneshot_incomplete"
        return (response, result)
    except BaseException:
        session_end_reason = "oneshot_error"
        raise
    finally:
        # cancel_and_claim() returning False means the hard watchdog fired
        # first and already wrote end_reason="oneshot_deadline" itself
        # (from its own thread, via os._exit) — this thread must not also
        # write, since the process is already on its way out.
        finalize_owned = watchdog.cancel_and_claim() if watchdog is not None else True
        if finalize_owned:
            sid = getattr(agent, "session_id", None) if agent is not None else None
            if session_db is not None and sid:
                if watchdog is not None:
                    # cancel_and_claim() cancelled the hard timer, so this
                    # terminal write no longer has a deadline backstop. Bound it
                    # (same budget as the watchdog's own write) so a SQLite-lock
                    # hang can't push a normal completion past the soft deadline
                    # into an external SIGKILL (→ end_reason NULL, the original
                    # bug). With no soft deadline (watchdog is None) there is no
                    # deadline to honour, so the plain write below is fine.
                    watchdog.finalize_session_bounded(session_db, sid, session_end_reason)
                else:
                    try:
                        session_db.end_session(sid, session_end_reason)
                    except Exception:
                        logging.debug("oneshot: end_session failed", exc_info=True)
        elif watchdog is not None:
            # We lost the race: the watchdog already committed to owning
            # the terminal outcome (RESULT marker written, end_session
            # attempted, about to os._exit). Block rather than returning a
            # normal response — falling through here would let run_oneshot
            # print/exit through the ordinary path, racing the watchdog's
            # own output for real stdout and a real exit code. See
            # wait_for_exit()'s docstring.
            watchdog.wait_for_exit()


def _oneshot_clarify_callback(question: str, choices=None) -> str:
    """Clarify is disabled in oneshot mode — tell the agent to pick a
    default and proceed instead of stalling or erroring."""
    if choices:
        return (
            f"[oneshot mode: no user available. Pick the best option from "
            f"{choices} using your own judgment and continue.]"
        )
    return (
        "[oneshot mode: no user available. Make the most reasonable "
        "assumption you can and continue.]"
    )
