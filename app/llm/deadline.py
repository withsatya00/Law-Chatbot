from collections.abc import Iterator
from typing import Any

"""One wall-clock deadline shared by every LLM call in a single request.

The incident this exists for (reproduced live against the running backend,
2026-09-02, session `probe-6e689577`):

    09:44:50  POST /chat -- a Mobile Theft Complaint draft, nine labelled
              fields in one message
    09:46:14  optional LLM field-extraction pass returns after ~84s (and
              every narrative value it produced was rejected as unfaithful,
              so the 84s bought nothing)
    09:47:50  the Streamlit client gives up at its 180s timeout and shows
              "Sorry, I couldn't reach the assistant just now"
    09:48:14  gemini_overall_timeout, timeout_seconds=120.0 -- the FIRST
              drafting call finally times out, 24s after the user was already
              told the request had failed

and `ResilientLLMProvider` would then have retried that same 120s call twice
more, plus each configured fallback, for a total server-side ceiling of
roughly 450s on a request nobody was still waiting for.

Every individual timeout in the stack was already bounded. What did not exist
was a bound on their SUM. `settings.llm_timeout` bounds one Gemini call;
`max_attempts` bounds the retries; neither knows about the other, and neither
knows how long the caller is willing to wait. This module supplies the missing
piece: a single monotonic expiry, set once per request, that every layer reads
and clamps itself to.

Design notes:

* A `ContextVar` rather than a parameter threaded through ten call sites.
  `LLMProvider.chat()` is implemented by seven providers and called from
  chat, drafting, extraction, summarisation and analysis -- widening that
  signature everywhere would be a far larger and riskier change than the bug
  warrants, and any call site that forgets to pass it silently loses the
  bound. A contextvar is inherited by `asyncio` tasks created inside the
  request, so background work started per-request inherits it correctly too.
* Absent deadline == unbounded, exactly as before. Tests, scripts, CLI tools
  and the background KB indexer never set one and are completely unaffected.
* `remaining()` can go negative; callers treat "<= 0" as "no time left".
"""

import asyncio
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import structlog

log = structlog.get_logger(__name__)

# Never start an LLM call with less than this much budget left. A call that
# cannot plausibly finish is worse than not starting: it burns the remaining
# budget AND still fails, where returning immediately leaves time to degrade
# gracefully (deterministic draft, safe structured answer) inside the response
# the caller is still waiting for.
MIN_VIABLE_CALL_SECONDS = 5.0


@dataclass(frozen=True)
class Deadline:
    """A monotonic expiry plus the label of what it belongs to."""

    expires_at: float
    label: str

    def remaining(self) -> float:
        return self.expires_at - time.monotonic()

    def expired(self) -> bool:
        return self.remaining() <= 0


_current: ContextVar[Deadline | None] = ContextVar("llm_deadline", default=None)


def current_deadline() -> Deadline | None:
    return _current.get()


def remaining_seconds() -> float | None:
    """Seconds left on the active deadline, or `None` when none is set.

    `None` means "unbounded" and must be handled distinctly from `0.0`, which
    means "bounded, and already out of time".
    """
    deadline = _current.get()
    return None if deadline is None else deadline.remaining()


def has_time_for(seconds: float = MIN_VIABLE_CALL_SECONDS) -> bool:
    """Whether a call needing roughly `seconds` is still worth starting."""
    left = remaining_seconds()
    return True if left is None else left >= seconds


def clamp_timeout(preferred: float) -> float:
    """`preferred`, reduced to what the deadline actually allows.

    Never returns more than `preferred` and never less than
    `MIN_VIABLE_CALL_SECONDS` -- a caller that has already decided to make the
    call gets a timeout short enough to respect the deadline but long enough
    to be a real attempt rather than an instant failure. Guard with
    `has_time_for()` first if the call should be skipped entirely.
    """
    left = remaining_seconds()
    if left is None:
        return preferred
    return max(MIN_VIABLE_CALL_SECONDS, min(preferred, left))


@contextmanager
def deadline(seconds: float, label: str = "request") -> Iterator["Deadline | None"]:
    """Bind a deadline for the duration of the block.

    A deadline already set by an outer block is never extended -- the tighter
    of the two wins, so an inner "drafting budget" can shorten the request
    budget but a stray long inner budget can never overrun the client's
    timeout.
    """
    existing = _current.get()
    candidate = time.monotonic() + seconds
    if existing is not None and existing.expires_at <= candidate:
        # The outer deadline is already at least as strict; keep it.
        yield existing
        return
    token = _current.set(Deadline(expires_at=candidate, label=label))
    try:
        yield _current.get()
    finally:
        _current.reset(token)


def log_exhausted(stage: str, **context: Any) -> None:
    """One consistent log line for "we ran out of budget at <stage>"."""
    log.warning(
        "llm_deadline_exhausted",
        stage=stage,
        deadline_label=getattr(_current.get(), "label", None),
        remaining_seconds=round(remaining_seconds() or 0.0, 2),
        **context,
    )


# How much longer than the ambient deadline's remaining time (or, absent a
# deadline, than the caller's own `fallback_seconds`) `call_with_hard_timeout`
# waits before forcibly cancelling a call. This is a BACKSTOP, not the primary
# bound -- `clamp_timeout()`/`has_time_for()` above are what a well-behaved
# provider consults, and are what normally end a call first.
#
# It exists because QA (session 2026-09-11/12, `docs/qa/QA_TEST_MATRIX_
# 20260911.md`) reproduced `GeminiProvider.chat()` itself logging
# `gemini_overall_timeout` with `elapsed_seconds` of 317s and 250s+ against a
# ~100s budget it had itself computed and handed to `asyncio.wait_for` --
# i.e. the cancellation of the in-flight call did not actually take effect at
# the time it was supposed to (consistent with a connection stuck at the
# transport layer, e.g. a dead socket that never sends a FIN/RST, where
# cancelling the awaiting coroutine has to wait for the underlying I/O to
# unblock before it can actually stop). A per-call timeout computed correctly
# is worthless as a user-facing guarantee if enforcing it can itself stall --
# this wraps that same call in a SECOND, independent `asyncio.wait_for` with
# its own grace window, so a caller always gets control back within a bounded
# time regardless of what the inner call is actually doing.
HARD_TIMEOUT_GRACE_SECONDS = 15.0


async def call_with_hard_timeout(
    coro: Any, *, fallback_seconds: float = 120.0, grace_seconds: float | None = None,
) -> Any:
    """Awaits `coro`, forcibly cancelling it if it outlives the ambient
    deadline's remaining time (or `fallback_seconds`, absent a deadline) plus
    `grace_seconds` (module default `HARD_TIMEOUT_GRACE_SECONDS`, read at call
    time rather than bound as the parameter default, so a test -- or an
    operator via `monkeypatch`/direct assignment -- can shrink it without
    needing to pass it through every call site).

    Raises `TimeoutError` on cancellation -- never a bare `asyncio.
    CancelledError` -- so callers can handle it exactly like any other
    bounded-timeout failure (see `_RETRYABLE_ERROR_KINDS`'s `"timeout"`).
    """
    if grace_seconds is None:
        grace_seconds = HARD_TIMEOUT_GRACE_SECONDS
    left = remaining_seconds()
    budget = fallback_seconds if left is None else max(MIN_VIABLE_CALL_SECONDS, left)
    return await asyncio.wait_for(coro, timeout=budget + grace_seconds)
