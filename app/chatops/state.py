"""Workflow session state: the stack that makes pause/resume/switch work.

Held in the existing conversation-memory dict under one key, so it persists
exactly like every other piece of chat state and needs no new store.

A STACK rather than a single slot, because the requirement is that a user can
start a second task without losing the first: "pehle cyber complaint complete
karo" while a notarization is half-filled must park the notarization, run the
complaint, and then offer the notarization back with its facts intact.

Deterministic by construction. Nothing here consults a model: which workflow
is active, what has been collected, and what is parked are all read from this
structure, never re-derived from the transcript.
"""

from datetime import datetime, timedelta
from typing import Any

from app.core import clock

# One key in conversation memory holds the whole stack.
MEMORY_KEY = "chatops_stack"
# And one holds the idempotency ledger of side effects already performed.
EXECUTED_KEY = "chatops_executed"
_MAX_EXECUTED = 50
# Facts survive a pause; a cancelled workflow is dropped entirely.
_MAX_STACK_DEPTH = 5

# How long a half-finished workflow stays resumable. A user who wandered off
# mid-form and came back the next day should not have yesterday's half-typed
# address silently reused as if they had just said it -- the facts are stale
# evidence, and stale evidence in a legal document is exactly what the fact
# audit exists to prevent. Expired entries are dropped, not answered from.
STALE_AFTER = timedelta(minutes=45)


def _parse(stamp: Any) -> datetime | None:
    if not isinstance(stamp, str) or not stamp:
        return None
    try:
        parsed = datetime.fromisoformat(stamp)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=clock.LEGAL_TIMEZONE)


def touch(entry: dict[str, Any]) -> None:
    """Records that this workflow was active on this turn."""
    entry["updated_at"] = clock.now().isoformat()


def expire_stale(memory: dict[str, Any]) -> list[str]:
    """Drops workflows untouched for longer than `STALE_AFTER`.

    Returns the names dropped, so the caller can tell the user rather than
    letting a workflow vanish without explanation.
    """
    stack = _stack(memory)
    cutoff = clock.now() - STALE_AFTER
    fresh: list[dict[str, Any]] = []
    dropped: list[str] = []
    for entry in stack:
        stamp = _parse(entry.get("updated_at"))
        if stamp is not None and stamp < cutoff:
            dropped.append(str(entry.get("name", "")))
        else:
            fresh.append(entry)
    if dropped:
        memory[MEMORY_KEY] = fresh
    return dropped


def _stack(memory: dict[str, Any]) -> list[dict[str, Any]]:
    stack = memory.get(MEMORY_KEY)
    if not isinstance(stack, list):
        stack = []
        memory[MEMORY_KEY] = stack
    return stack


def active(memory: dict[str, Any]) -> dict[str, Any] | None:
    """The workflow currently in the foreground, or None."""
    stack = _stack(memory)
    return stack[-1] if stack else None


def active_name(memory: dict[str, Any]) -> str | None:
    entry = active(memory)
    return entry.get("name") if entry else None


def start(memory: dict[str, Any], name: str, *, facts: dict[str, Any] | None = None) -> dict[str, Any]:
    """Pushes a workflow onto the stack, parking whatever was in front.

    Re-entering a workflow already ON the stack RESUMES it (moving it to the
    front with its facts intact) rather than starting a duplicate -- asking
    for a notarization twice must not discard the first attempt's answers.
    """
    stack = _stack(memory)
    for index, entry in enumerate(stack):
        if entry.get("name") == name:
            resumed = stack.pop(index)
            resumed.update(facts=({**resumed.get("facts", {}), **(facts or {})}), status="collecting")
            touch(resumed)
            stack.append(resumed)
            return resumed
    entry = {
        "name": name,
        "facts": dict(facts or {}),
        "status": "collecting",
        "asked": [],
        "pending_confirmation": False,
        "updated_at": clock.now().isoformat(),
        # Side-effecting executions already performed, keyed by idempotency
        # key. Replaying a turn (a retry, a double-tapped send, a client
        # resend) must not run the same action twice.
        "executed": {},
    }
    stack.append(entry)
    # Bounded so a user who names many tasks cannot grow memory without limit.
    # The OLDEST parked workflow is dropped -- it is the one least likely to
    # still be wanted, and the user can simply ask for it again.
    del stack[:-_MAX_STACK_DEPTH]
    return entry


def update(memory: dict[str, Any], **values: Any) -> None:
    entry = active(memory)
    if entry is not None:
        entry.update(values)


def remember_facts(memory: dict[str, Any], facts: dict[str, Any]) -> None:
    entry = active(memory)
    if entry is None:
        return
    entry.setdefault("facts", {}).update({k: v for k, v in facts.items() if v not in (None, "")})


def mark_asked(memory: dict[str, Any], field_name: str) -> None:
    """Records that we have already asked for `field_name`.

    Used to rotate the question when a user does not answer the thing we
    asked -- repeating the identical question at someone who just ignored it
    is how a chat flow becomes a wall.
    """
    entry = active(memory)
    if entry is not None and field_name:
        asked = entry.setdefault("asked", [])
        if field_name not in asked:
            asked.append(field_name)


def finish(memory: dict[str, Any], name: str | None = None) -> None:
    """Pops a completed/cancelled workflow off the stack."""
    stack = _stack(memory)
    if not stack:
        return
    if name is None:
        stack.pop()
        return
    memory[MEMORY_KEY] = [entry for entry in stack if entry.get("name") != name]


def pause(memory: dict[str, Any]) -> dict[str, Any] | None:
    """Parks the foreground workflow, keeping its facts. Returns it."""
    entry = active(memory)
    if entry is None:
        return None
    entry["status"] = "paused"
    # Moved to the BACK of the stack rather than removed, so "continue" finds
    # it and an unrelated new task can take the foreground.
    stack = _stack(memory)
    stack.insert(0, stack.pop())
    return entry


def parked(memory: dict[str, Any]) -> list[dict[str, Any]]:
    """Workflows in progress but not in the foreground."""
    stack = _stack(memory)
    return list(stack[:-1]) if stack else []


def resume(memory: dict[str, Any], name: str | None = None) -> dict[str, Any] | None:
    """Brings a parked workflow back to the foreground, facts intact.

    With no `name`, the most recently parked one is chosen -- "continue"
    after a single interruption is by far the common case, and asking "which
    one?" when there is only one candidate is the kind of question this
    product is supposed to stop asking.
    """
    stack = _stack(memory)
    if not stack:
        return None
    if name is None:
        paused = [e for e in stack[:-1] if e.get("status") == "paused"]
        if not paused:
            current = stack[-1]
            if current.get("status") == "paused":
                current["status"] = "collecting"
                touch(current)
                return current
            return None
        name = str(paused[-1].get("name", ""))
    for index, entry in enumerate(stack):
        if entry.get("name") == name:
            revived = stack.pop(index)
            revived["status"] = "collecting"
            touch(revived)
            stack.append(revived)
            return revived
    return None


def forget_fact(memory: dict[str, Any], field_name: str) -> bool:
    """Clears one collected fact so it is asked for again. True if it existed."""
    entry = active(memory)
    if entry is None:
        return False
    facts = entry.setdefault("facts", {})
    if field_name in facts:
        facts.pop(field_name, None)
        asked = entry.setdefault("asked", [])
        if field_name in asked:
            asked.remove(field_name)
        touch(entry)
        return True
    return False


def already_executed(memory: dict[str, Any], key: str) -> bool:
    """Whether this exact side effect has already run in this conversation.

    The guard against a replayed turn doing the work twice: submitting the
    same notarization request, deleting the same draft, retrying the same
    job. Idempotency is enforced here, in deterministic state, rather than
    relying on every service to be idempotent on its own.

    Deliberately kept in MEMORY rather than on the workflow entry: the entry
    is popped off the stack the moment a workflow completes, so a record
    living there would disappear exactly when the replay it guards against
    becomes possible.
    """
    executed = memory.get(EXECUTED_KEY)
    return isinstance(executed, dict) and key in executed


def mark_executed(memory: dict[str, Any], key: str, summary: str = "") -> None:
    executed = memory.get(EXECUTED_KEY)
    if not isinstance(executed, dict):
        executed = {}
        memory[EXECUTED_KEY] = executed
    executed[key] = {"at": clock.now().isoformat(), "summary": summary}
    # Bounded: a long conversation must not grow this without limit. Oldest
    # first, which is also least likely to be replayed.
    for stale_key in list(executed)[:-_MAX_EXECUTED]:
        executed.pop(stale_key, None)


def clear(memory: dict[str, Any]) -> None:
    memory[MEMORY_KEY] = []
