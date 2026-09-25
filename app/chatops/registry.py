"""Workflow registry: name -> `ChatWorkflow`.

Adding a capability to chat is registering one class here. The orchestrator
never names a workflow directly, so nothing about routing has to change.
"""

import structlog

from app.chatops.base import ChatWorkflow
from app.language.typo_tolerance import normalize_for_routing

log = structlog.get_logger(__name__)

WORKFLOWS: dict[str, ChatWorkflow] = {}

# Minimum `matches_intent` score for a workflow to take a turn. Set high on
# purpose: a near-miss should fall through to the ordinary RAG answer, which
# is always a safe outcome, rather than dragging the user into a legal
# workflow they did not ask for.
MATCH_THRESHOLD = 0.5


def register_workflow(workflow_class: type[ChatWorkflow]) -> type[ChatWorkflow]:
    """Registers a workflow class under its `name`. Usable as a decorator."""
    name = getattr(workflow_class, "name", "")
    if not name:
        raise ValueError("A ChatWorkflow must define a non-empty `name`.")
    if name in WORKFLOWS:
        raise ValueError(f"Duplicate workflow name: {name}")
    WORKFLOWS[name] = workflow_class()
    return workflow_class


def get_workflow(name: str) -> ChatWorkflow | None:
    return WORKFLOWS.get(name)


def _match_texts(message: str) -> tuple[str, ...]:
    """The message as typed, plus its routing-normalized form when the bounded
    typo normalizer actually changed something.

    Workflow matchers are keyword matchers, so one dropped or swapped letter in
    a capability word used to lose the request entirely: "notry ke liye
    affidavit tyar kro" scored below threshold against every workflow and fell
    through to retrieval. Scoring the allowlisted normalization as well fixes
    that without loosening any matcher. The ORIGINAL text is what workflows are
    later run with -- this only affects which workflow is selected.
    """
    normalized = normalize_for_routing(message).normalized_for_routing
    return (message,) if normalized == message else (message, normalized)


def _score(workflow: ChatWorkflow, message: str, language: str) -> float:
    return max(workflow.matches_intent(text, language) for text in _match_texts(message))


def best_match(message: str, language: str) -> tuple[ChatWorkflow, float] | None:
    """The highest-scoring workflow for `message`, or None below threshold.

    Ties are broken by registration order, which is deterministic -- two
    workflows scoring identically means the vocabularies overlap and the
    fix is to tighten a pattern, not to add randomness here.
    """
    best: tuple[ChatWorkflow, float] | None = None
    for workflow in WORKFLOWS.values():
        try:
            confidence = _score(workflow, message, language)
        except Exception as exc:  # noqa: BLE001 - one bad pattern must not break routing
            log.warning("chatops_match_failed", workflow=workflow.name, error=str(exc))
            continue
        if confidence >= MATCH_THRESHOLD and (best is None or confidence > best[1]):
            best = (workflow, confidence)
    return best


def interrupts_draft(message: str, language: str) -> bool:
    """Whether `message` explicitly asks for a capability that outranks a draft.

    Read by `ChatService` before it decides whether to run the orchestrator at
    all. While a draft is open, an ordinary message is presumed to be a field
    answer and the orchestrator stays out of the way -- but "meri saved drafts
    dikhao", "is document ka summary do" and "affidavit ko notarization ke liye
    prepare karo" are none of them field answers, and answering them from
    retrieval (as an observed session did) loses the request entirely.

    Only workflows that declare `may_interrupt_draft` qualify, and only on a
    match above the normal threshold -- so this can never be triggered by a
    near-miss on a narrative field value.
    """
    match = best_match(message, language)
    return match is not None and match[0].may_interrupt_draft


def all_matches(message: str, language: str) -> list[tuple[ChatWorkflow, float]]:
    """Every workflow above threshold, best first.

    Used for multi-intent detection: "analyse karo, risky clauses fix karo
    aur PDF bana do" legitimately names three capabilities, and the
    orchestrator chains them rather than picking one and dropping the rest.
    """
    matches: list[tuple[ChatWorkflow, float]] = []
    for workflow in WORKFLOWS.values():
        try:
            confidence = _score(workflow, message, language)
        except Exception:
            # One workflow's matcher must not suppress the others, but a
            # silently dropped capability is exactly the kind of failure that
            # shows up as "the bot ignored half my request" with nothing to
            # diagnose from. Isolate it, then record it.
            log.exception("workflow_intent_match_failed", workflow=workflow.name)
            continue
        if confidence >= MATCH_THRESHOLD:
            matches.append((workflow, confidence))
    return sorted(matches, key=lambda pair: pair[1], reverse=True)
