"""Regression coverage for `call_with_hard_timeout` (`app/llm/deadline.py`).

QA (session 2026-09-11/12, `docs/qa/QA_TEST_MATRIX_20260911.md`, BUG-013a /
"New latency finding") reproduced a draft-generation call stalling 250-317s
against a ~100s budget `GeminiProvider.chat()` had itself computed and handed
to `asyncio.wait_for` -- i.e. the existing per-call timeout mechanism did not
actually bound the call in practice. `call_with_hard_timeout` is the backstop
this suite pins: an independent, outer `asyncio.wait_for` that guarantees
control returns to the caller within a bounded time regardless of what the
wrapped call is actually doing.
"""

import asyncio

import pytest

from app.llm import deadline as deadline_module
from app.llm.deadline import call_with_hard_timeout, deadline


async def _hang_forever() -> str:
    await asyncio.Event().wait()
    return "never reached"


async def _quick_success() -> str:
    return "ok"


def test_a_call_that_never_returns_is_cancelled_within_the_fallback_plus_grace(monkeypatch) -> None:
    monkeypatch.setattr(deadline_module, "HARD_TIMEOUT_GRACE_SECONDS", 0.05)

    async def run() -> float:
        loop_start = asyncio.get_running_loop().time()
        with pytest.raises(TimeoutError):
            await call_with_hard_timeout(_hang_forever(), fallback_seconds=0.05)
        return asyncio.get_running_loop().time() - loop_start

    elapsed = asyncio.run(run())
    # Bounded well under what "hang forever" would otherwise mean -- the
    # exact number just has to be small and deterministic, not tied to the
    # production budget.
    assert elapsed < 2.0


def test_a_call_that_returns_quickly_is_unaffected(monkeypatch) -> None:
    monkeypatch.setattr(deadline_module, "HARD_TIMEOUT_GRACE_SECONDS", 0.05)
    result = asyncio.run(call_with_hard_timeout(_quick_success(), fallback_seconds=0.05))
    assert result == "ok"


def test_the_ambient_deadline_shrinks_the_effective_budget(monkeypatch) -> None:
    """When a `deadline()` is already active, its remaining time -- not
    `fallback_seconds` -- governs, exactly like every other consumer of
    `remaining_seconds()`/`clamp_timeout()` in this module."""
    monkeypatch.setattr(deadline_module, "HARD_TIMEOUT_GRACE_SECONDS", 0.05)
    # Never start a call with less than this much budget left (see
    # `MIN_VIABLE_CALL_SECONDS`'s own docstring) -- shrunk here too, purely so
    # this test proves the ambient deadline governs without an unrelated
    # 5-second floor slowing it down.
    monkeypatch.setattr(deadline_module, "MIN_VIABLE_CALL_SECONDS", 0.01)

    async def run() -> None:
        with deadline(0.05, label="test"), pytest.raises(TimeoutError):
            # A generous fallback that would never fire on its own --
            # proves the ambient deadline (0.05s), not this, is what cut
            # the call off.
            await call_with_hard_timeout(_hang_forever(), fallback_seconds=120.0)

    asyncio.run(run())
