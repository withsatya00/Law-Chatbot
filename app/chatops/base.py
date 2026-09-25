"""The `ChatWorkflow` contract every conversational capability implements.

A workflow is an ADAPTER over an existing service, not a place to put legal
logic. `execute()` should read as a thin call into `NotarizationService`,
`Phase2WorkflowService`, `LegalDraftEngine` and friends. If a workflow starts
containing rules about what the law requires, that rule belongs in the service
and the workflow should call it.

The eight methods below are the whole interface:

    matches_intent()            -- does this message start/continue me?
    extract_facts()             -- pull field values out of what was said
    required_fields()           -- what must be known before executing
    next_question()             -- the single most useful thing to ask next
    validate()                  -- sanity-check collected facts
    summarize_for_confirmation()-- what the user is about to authorise
    execute()                   -- call the real service
    render_chat_response()      -- turn the result into a chat turn

`next_question` returns ONE question. That is a product rule, not a style
preference: the reported failure mode of the old flow was dumping eight
labelled fields at a user who had asked to be asked one thing at a time.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Literal

from app.core.security import Role

# What the orchestrator tells the caller about where a workflow stands.
WorkflowStatus = Literal[
    "collecting",           # still gathering facts
    "awaiting_confirmation",  # summary shown, waiting for yes/no
    "awaiting_upload",      # needs a document before it can continue
    "awaiting_reauth",      # needs a fresh credential before acting
    "completed",            # executed successfully
    "cancelled",            # user abandoned it
    "paused",               # user parked it; resumable
    "failed",               # execution failed
    "forbidden",            # role check refused it
]


@dataclass
class WorkflowContext:
    """Everything a workflow may look at. Assembled per turn by the orchestrator.

    `memory` is the live conversation-memory dict, so a workflow reads facts
    the rest of the chat already established (an uploaded document id, the
    active draft, the language) instead of asking again -- which is the
    "never ask twice" requirement, enforced by construction rather than by
    each workflow remembering to check.
    """

    session_id: str
    message: str
    language: str
    memory: dict[str, Any]
    # Facts collected across turns FOR THIS WORKFLOW, keyed by field name.
    facts: dict[str, Any] = field(default_factory=dict)
    authenticated_user_id: str | None = None
    # JWT claims, for role checks. Never trusted from the message itself.
    claims: dict[str, Any] = field(default_factory=dict)
    # Documents uploaded in this conversation, most recent first.
    document_ids: list[str] = field(default_factory=list)

    def role(self) -> str:
        return str(self.claims.get("role") or "")

    def remember(self, **values: Any) -> None:
        self.facts.update({key: value for key, value in values.items() if value not in (None, "")})


@dataclass
class WorkflowTurn:
    """One workflow's contribution to a chat reply.

    Maps onto the additive `ChatResponse` fields. Everything except `message`
    is optional, so a workflow that just wants to say something says only
    that.
    """

    message: str
    status: WorkflowStatus = "collecting"
    # The single field being asked for right now, if any.
    missing_field: str | None = None
    # -- structured progress, filled in by the orchestrator ---------------
    # A workflow may set these itself when it knows better (a workflow whose
    # steps are not one-per-field), but it never has to: the orchestrator
    # derives them from `required_fields` when they are left at their
    # defaults, so every workflow reports progress without writing any code
    # to do so.
    workflow_name: str = ""
    current_step: str = ""
    completed_steps: int = 0
    total_steps: int = 0
    progress_percentage: int = 0
    collected_facts: dict[str, Any] = field(default_factory=dict)
    requires_confirmation: bool = False
    # Natural-language actions the user may take next, for UI affordances.
    # Advisory only -- the orchestrator never refuses an action because it
    # is absent from this list.
    allowed_actions: list[str] = field(default_factory=list)
    # A produced file, as a SECURE reference. Never a filesystem path --
    # see `artifact_url` in `app/chatops/artifacts.py`.
    artifact: dict[str, Any] | None = None
    citations: list[dict[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    upload_required: bool = False
    # For an unavoidable third-party step (an external e-sign ceremony).
    secure_action_url: str | None = None
    # True when the workflow is finished with the turn and should be popped
    # off the session's workflow stack.
    finished: bool = False


class ChatWorkflow(ABC):
    """Base class for every conversational capability."""

    #: Stable identifier, used as the workflow-stack key and in telemetry.
    name: str
    #: Human-readable, shown when listing what the user has in progress.
    title: str
    #: Minimum role. `None` means any authenticated or anonymous user.
    required_role: Role | None = None
    #: Whether `execute()` does something irreversible or outward-facing and
    #: therefore must pass through an explicit confirmation turn first.
    requires_confirmation: bool = False
    #: Whether `execute()` additionally needs a fresh credential.
    requires_reauthentication: bool = False
    #: Whether an EXPLICIT request for this capability outranks a draft that
    #: is currently being filled in.
    #:
    #: Default `False`, because a message arriving mid-draft is usually a
    #: field answer and stealing it would lose the user's work. It is set on
    #: the capabilities a user demonstrably asks for BY NAME while a draft is
    #: open -- their saved drafts, a document, a case, a notarization step.
    #: In an observed session every one of those fell through to retrieval
    #: and was answered with "no verified document is available", because an
    #: open draft suppressed the orchestrator entirely.
    may_interrupt_draft: bool = False

    @abstractmethod
    def matches_intent(self, message: str, language: str) -> float:
        """Confidence in [0, 1] that `message` asks for this workflow.

        Pattern-based. A workflow returning 0.0 is not considered at all;
        the highest scorer above `MATCH_THRESHOLD` wins.
        """

    @abstractmethod
    async def extract_facts(self, context: WorkflowContext) -> dict[str, Any]:
        """Field values discoverable from this message and existing context."""

    @abstractmethod
    def required_fields(self, context: WorkflowContext) -> list[str]:
        """Field names that must be non-empty before `execute()` may run."""

    @abstractmethod
    def next_question(self, context: WorkflowContext, missing: list[str]) -> str:
        """The ONE question to ask next, in `context.language`.

        Receives every missing field so the workflow can choose which matters
        most -- but must return a single question about a single field.
        """

    async def validate(self, context: WorkflowContext) -> list[str]:
        """Human-readable problems with the collected facts.

        Default: nothing to check. Override to catch contradictions the user
        must resolve before an irreversible step.
        """
        return []

    def needs_confirmation(self, context: WorkflowContext) -> bool:
        """Whether THIS run needs an explicit yes before it acts.

        Defaults to the class-level flag. Overridden by workflows whose
        actions differ in kind: listing a case's hearings and deleting the
        case are the same workflow, and only one of them may proceed on a
        bare request.
        """
        return self.requires_confirmation

    def summarize_for_confirmation(self, context: WorkflowContext) -> str:
        """What the user is about to authorise, in `context.language`.

        Only consulted when `requires_confirmation` is set.
        """
        lines = [f"- {key}: {value}" for key, value in context.facts.items() if value]
        return "\n".join(lines)

    @abstractmethod
    async def execute(self, context: WorkflowContext) -> WorkflowTurn:
        """Call the real service and return the result as a chat turn."""

    def render_chat_response(self, turn: WorkflowTurn, context: WorkflowContext) -> WorkflowTurn:
        """Last chance to shape the turn. Default: unchanged."""
        return turn

    def idempotency_key(self, context: WorkflowContext) -> str:
        """A stable key identifying THIS execution of THIS workflow.

        Used to refuse a second run of the same side effect when a turn is
        replayed -- a retried request, a double-tapped send, a client that
        resends on a flaky connection. Default: the workflow name plus a
        digest of the collected facts, which is exactly "the same action with
        the same inputs". A workflow whose identity is better expressed by
        one field (a draft id, a case id) should override this.
        """
        material = repr(sorted((str(key), str(value)) for key, value in context.facts.items()))
        return f"{self.name}:{sha256(material.encode('utf-8')).hexdigest()[:16]}"

    def field_label(self, field_name: str) -> str:
        """Human-readable name for a field, for progress display.

        Never shown as an internal identifier -- `draft_id` reads as "draft",
        not as something the user is expected to know or type.
        """
        return field_name.removesuffix("_id").replace("_", " ").strip() or field_name

    # -- helpers available to every workflow -----------------------------

    def missing_fields(self, context: WorkflowContext) -> list[str]:
        """Required fields still unknown.

        Central, so "never ask for something we already have" is one
        implementation rather than a rule each workflow must remember.
        """
        return [
            name
            for name in self.required_fields(context)
            if not str(context.facts.get(name, "") or "").strip()
        ]
